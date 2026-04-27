"""
Active Learning Feedback Loop.

Integrates experimental SPR/BLI binding measurements back into the
ImmunoForge scoring pipeline, enabling iterative improvement of
ranking weights and guided selection of the next design round.

Workflow:
    1. Submit experimental K_D data from SPR/BLI for designed binders
    2. Compare predicted vs measured K_D → update Bayesian scoring weights
    3. Select next design candidates via acquisition function (UCB / EI)
    4. Export updated configuration for next pipeline run
"""

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ExperimentalMeasurement:
    """A single SPR/BLI measurement for a designed binder."""
    binder_id: str
    sequence: str
    target: str
    measured_kd_nM: float
    measurement_method: str = "SPR"   # SPR | BLI | ELISA
    measured_kon: float | None = None
    measured_koff: float | None = None
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PredictionRecord:
    """Stored prediction for comparison with experiment."""
    binder_id: str
    predicted_kd_nM: float
    composite_score: float
    ranking_weights: dict = field(default_factory=dict)


@dataclass
class FeedbackRound:
    """A complete feedback iteration."""
    round_id: int
    measurements: list[ExperimentalMeasurement] = field(default_factory=list)
    predictions: list[PredictionRecord] = field(default_factory=list)
    updated_weights: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# Weight update engine
# ═══════════════════════════════════════════════════════════════════

DEFAULT_WEIGHTS = {
    "affinity": 0.40,
    "structure_quality": 0.25,
    "sequence_qc": 0.15,
    "developability": 0.10,
    "immunogenicity": 0.10,
}


def compute_prediction_error(
    measurements: list[ExperimentalMeasurement],
    predictions: list[PredictionRecord],
) -> dict:
    """Compare predicted vs measured K_D values."""
    pred_map = {p.binder_id: p for p in predictions}

    pairs = []
    for m in measurements:
        p = pred_map.get(m.binder_id)
        if p is None:
            continue
        if m.measured_kd_nM > 0 and p.predicted_kd_nM > 0:
            log_error = math.log10(p.predicted_kd_nM) - math.log10(m.measured_kd_nM)
            pairs.append({
                "binder_id": m.binder_id,
                "measured_kd_nM": m.measured_kd_nM,
                "predicted_kd_nM": p.predicted_kd_nM,
                "log10_error": log_error,
                "abs_log10_error": abs(log_error),
            })

    if not pairs:
        return {"n_pairs": 0, "error": "No matching prediction-measurement pairs"}

    abs_errors = [p["abs_log10_error"] for p in pairs]
    signed_errors = [p["log10_error"] for p in pairs]

    return {
        "n_pairs": len(pairs),
        "mean_abs_log10_error": float(np.mean(abs_errors)),
        "mean_signed_error": float(np.mean(signed_errors)),
        "median_abs_error": float(np.median(abs_errors)),
        "fraction_within_1_log": sum(1 for e in abs_errors if e <= 1.0) / len(abs_errors),
        "pairs": pairs,
    }


def update_weights(
    current_weights: dict[str, float],
    measurements: list[ExperimentalMeasurement],
    predictions: list[PredictionRecord],
    learning_rate: float = 0.1,
) -> dict[str, float]:
    """Update scoring weights based on experimental feedback.

    Uses a simplified gradient-free Bayesian update:
    - If predictions systematically overestimate affinity, increase
      weight on structure/developability constraints.
    - If predictions are noisy but unbiased, keep current weights.
    """
    error_stats = compute_prediction_error(measurements, predictions)
    if error_stats.get("n_pairs", 0) == 0:
        return current_weights.copy()

    signed_error = error_stats["mean_signed_error"]
    abs_error = error_stats["mean_abs_log10_error"]

    new_weights = current_weights.copy()

    if signed_error > 0.5:
        # Over-predicting affinity → increase structure + developability weight
        new_weights["structure_quality"] += learning_rate * 0.05
        new_weights["developability"] += learning_rate * 0.03
        new_weights["affinity"] -= learning_rate * 0.08
    elif signed_error < -0.5:
        # Under-predicting → increase affinity weight
        new_weights["affinity"] += learning_rate * 0.05
        new_weights["structure_quality"] -= learning_rate * 0.03
        new_weights["developability"] -= learning_rate * 0.02

    if abs_error > 1.5:
        # Very noisy → dampen all changes, increase QC emphasis
        new_weights["sequence_qc"] += learning_rate * 0.03

    # Normalize to sum = 1
    total = sum(new_weights.values())
    if total > 0:
        new_weights = {k: round(v / total, 4) for k, v in new_weights.items()}

    # Clamp: no weight below 0.02 or above 0.60
    for k in new_weights:
        new_weights[k] = max(0.02, min(0.60, new_weights[k]))

    # Re-normalize after clamping
    total = sum(new_weights.values())
    new_weights = {k: round(v / total, 4) for k, v in new_weights.items()}

    return new_weights


# ═══════════════════════════════════════════════════════════════════
# Acquisition functions for next-round selection
# ═══════════════════════════════════════════════════════════════════

def upper_confidence_bound(
    candidates: list[dict],
    kappa: float = 2.0,
) -> list[dict]:
    """Select next candidates using UCB acquisition function.

    Args:
        candidates: List of dicts with keys:
            - binder_id, sequence, predicted_kd_nM, prediction_uncertainty
        kappa: Exploration-exploitation tradeoff (higher = more exploration).

    Returns:
        Candidates sorted by UCB score (best first).
    """
    scored = []
    for c in candidates:
        pred_kd = c.get("predicted_kd_nM", 1000.0)
        uncertainty = c.get("prediction_uncertainty", 1.0)

        # Lower predicted K_D is better → negate for maximization
        # Higher uncertainty adds exploration bonus
        ucb = -math.log10(max(pred_kd, 0.001)) + kappa * uncertainty
        scored.append({**c, "ucb_score": round(ucb, 4)})

    return sorted(scored, key=lambda x: x["ucb_score"], reverse=True)


def expected_improvement(
    candidates: list[dict],
    best_measured_kd_nM: float,
    xi: float = 0.01,
) -> list[dict]:
    """Select next candidates using Expected Improvement.

    Args:
        candidates: List with predicted_kd_nM and prediction_uncertainty.
        best_measured_kd_nM: Best K_D observed experimentally so far.
        xi: Exploration parameter.
    """
    best_log = math.log10(max(best_measured_kd_nM, 0.001))
    scored = []

    for c in candidates:
        pred_kd = c.get("predicted_kd_nM", 1000.0)
        sigma = c.get("prediction_uncertainty", 1.0)

        pred_log = math.log10(max(pred_kd, 0.001))
        # Lower log K_D is better
        improvement = best_log - pred_log - xi

        if sigma > 0:
            z = improvement / sigma
            # Approximate standard normal CDF
            ei = improvement * _norm_cdf(z) + sigma * _norm_pdf(z)
        else:
            ei = max(improvement, 0.0)

        scored.append({**c, "ei_score": round(ei, 4)})

    return sorted(scored, key=lambda x: x["ei_score"], reverse=True)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


# ═══════════════════════════════════════════════════════════════════
# Feedback loop orchestrator
# ═══════════════════════════════════════════════════════════════════

def run_feedback_round(
    round_id: int,
    measurements: list[ExperimentalMeasurement],
    predictions: list[PredictionRecord],
    current_weights: dict[str, float] | None = None,
    learning_rate: float = 0.1,
) -> FeedbackRound:
    """Execute one feedback iteration.

    1. Compare predictions to measurements
    2. Update scoring weights
    3. Return updated round with metrics
    """
    if current_weights is None:
        current_weights = DEFAULT_WEIGHTS.copy()

    error_stats = compute_prediction_error(measurements, predictions)
    updated = update_weights(current_weights, measurements, predictions, learning_rate)

    return FeedbackRound(
        round_id=round_id,
        measurements=measurements,
        predictions=predictions,
        updated_weights=updated,
        metrics={
            "prediction_error": error_stats,
            "weight_changes": {
                k: round(updated.get(k, 0) - current_weights.get(k, 0), 4)
                for k in set(list(updated.keys()) + list(current_weights.keys()))
            },
        },
    )


def save_feedback_state(
    feedback_round: FeedbackRound,
    output_dir: str,
) -> Path:
    """Save feedback round state to JSON for reproducibility."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"feedback_round_{feedback_round.round_id}.json"

    data = {
        "round_id": feedback_round.round_id,
        "updated_weights": feedback_round.updated_weights,
        "metrics": feedback_round.metrics,
        "n_measurements": len(feedback_round.measurements),
        "n_predictions": len(feedback_round.predictions),
        "measurements": [
            {
                "binder_id": m.binder_id,
                "target": m.target,
                "measured_kd_nM": m.measured_kd_nM,
                "method": m.measurement_method,
            }
            for m in feedback_round.measurements
        ],
    }

    path.write_text(json.dumps(data, indent=2))
    logger.info("Saved feedback state to %s", path)
    return path


def load_latest_weights(feedback_dir: str) -> dict[str, float]:
    """Load scoring weights from most recent feedback round."""
    fb_dir = Path(feedback_dir)
    if not fb_dir.exists():
        return DEFAULT_WEIGHTS.copy()

    rounds = sorted(fb_dir.glob("feedback_round_*.json"))
    if not rounds:
        return DEFAULT_WEIGHTS.copy()

    data = json.loads(rounds[-1].read_text())
    return data.get("updated_weights", DEFAULT_WEIGHTS.copy())

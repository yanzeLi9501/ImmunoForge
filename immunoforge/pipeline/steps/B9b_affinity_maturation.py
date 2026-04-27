"""
B9b — Conditional Affinity Maturation for Weak Binders.

Identifies candidates with K_D > threshold (default 100 nM) from B6 ranking
and runs iterative affinity maturation using ProteinMPNN T=0.3 resampling
+ ESMFold structure scoring + affinity re-evaluation.

Flow:
  1. Read B6 ranked candidates + B9a AF3 results (if available)
  2. Filter candidates with K_D > kd_threshold
  3. For each weak binder:
     a. ProteinMPNN T=0.3 sequence resampling (conservative)
     b. ESMFold structure validation + pLDDT scoring
     c. Affinity re-prediction (3-method consensus)
     d. Select best improved variant
  4. Output improved candidates with before/after comparison

Input:  B6_candidate_ranking.json (+ optional B9a_af3_validation.json)
Output: B9b_affinity_maturation.json with maturation trajectories
"""

import json
import logging
from pathlib import Path

from immunoforge.core.maturation import run_maturation, MaturationResult
from immunoforge.core.utils import save_json

logger = logging.getLogger(__name__)


def _result_to_dict(r: MaturationResult, original_kd: float) -> dict:
    """Convert MaturationResult to a serializable summary dict."""
    return {
        "parent_id": r.best_candidate.parent_id or r.best_candidate.id,
        "best_id": r.best_candidate.id,
        "best_sequence": r.best_candidate.sequence,
        "original_kd_nM": round(original_kd, 1),
        "initial_kd_nM": r.initial_kd_nM,
        "final_kd_nM": r.final_kd_nM,
        "improvement_fold": r.improvement_fold,
        "n_generations": r.n_generations,
        "n_evaluated": r.n_candidates_evaluated,
        "trajectory": r.trajectory,
        "mutations": [
            {
                "position": m.get("position"),
                "original": m.get("original"),
                "mutated": m.get("mutated"),
                "notation": m.get("notation"),
            }
            for m in r.best_candidate.mutations
        ],
    }


def main(config: dict) -> dict:
    """Execute B9b: Conditional Affinity Maturation.

    Reads B6 ranking, filters weak binders (K_D > threshold),
    and runs in-silico affinity maturation on them.
    """
    logger.info("  B9b: Conditional Affinity Maturation")

    output_dir = Path(config.get("paths", {}).get("output_dir", "outputs"))

    # Configuration
    mat_cfg = config.get("affinity_maturation", {})
    kd_threshold = mat_cfg.get("kd_threshold_nM", 100.0)
    target_kd = mat_cfg.get("target_kd_nM", 50.0)
    max_generations = mat_cfg.get("max_generations", 5)
    candidates_per_gen = mat_cfg.get("candidates_per_gen", 20)
    top_k = mat_cfg.get("top_k_per_gen", 5)
    temperatures = mat_cfg.get("temperatures", [0.1, 0.3])  # Focus on T=0.3
    seed = mat_cfg.get("seed", 42)

    # Load B6 ranking
    b6_path = output_dir / "B6_candidate_ranking.json"
    if not b6_path.exists():
        for subdir in sorted(output_dir.iterdir(), reverse=True):
            candidate = subdir / "B6_candidate_ranking.json"
            if candidate.exists():
                b6_path = candidate
                break

    if not b6_path.exists():
        logger.warning("  B6 ranking not found, skipping maturation")
        return {"status": "skipped", "reason": "B6_ranking_not_found"}

    with open(b6_path) as f:
        b6 = json.load(f)

    ranked = b6.get("ranked", [])
    if not ranked:
        return {"status": "skipped", "reason": "no_ranked_candidates"}

    # Optionally load B9a AF3 results for enhanced scoring
    b9a_path = output_dir / "B9a_af3_validation.json"
    af3_scores = {}
    if b9a_path.exists():
        with open(b9a_path) as f:
            b9a = json.load(f)
        for r in b9a.get("results", []):
            af3_scores[r["binder_id"]] = {
                "iptm": r.get("iptm", 0),
                "quality": r.get("quality", "UNKNOWN"),
            }
        logger.info("  Loaded %d AF3 validation scores", len(af3_scores))

    # Filter candidates needing maturation (K_D > threshold)
    weak_binders = []
    strong_binders = []

    for cand in ranked:
        kd = cand.get("kd_nM", 1e6)
        if kd > kd_threshold:
            weak_binders.append(cand)
        else:
            strong_binders.append(cand)

    logger.info("  %d/%d candidates have K_D > %.0f nM (need maturation)",
                len(weak_binders), len(ranked), kd_threshold)
    logger.info("  %d candidates already below threshold", len(strong_binders))

    if not weak_binders:
        logger.info("  All candidates below K_D threshold — no maturation needed")
        return {
            "status": "completed",
            "n_matured": 0,
            "n_below_threshold": len(strong_binders),
            "message": "All candidates already below K_D threshold",
        }

    # Run maturation on weak binders
    maturation_results = []

    for cand in weak_binders:
        cand_id = cand["id"]
        sequence = cand["sequence"]
        kd = cand.get("kd_nM", 1e6)
        bsa = cand.get("bsa", 1200.0)
        sc = cand.get("sc", 0.65)

        logger.info("  Maturing %s: K_D=%.1f nM (target: %.1f nM)", cand_id, kd, target_kd)

        result = run_maturation(
            parent_sequence=sequence,
            parent_id=cand_id,
            bsa=bsa,
            sc=sc,
            target_kd_nM=target_kd,
            max_generations=max_generations,
            candidates_per_gen=candidates_per_gen,
            top_k=top_k,
            temperatures=temperatures,
            seed=seed,
            force=True,  # Run even without structure backend
        )

        summary = _result_to_dict(result, original_kd=kd)

        # Attach AF3 score if available
        if cand_id in af3_scores:
            summary["af3_iptm"] = af3_scores[cand_id]["iptm"]
            summary["af3_quality"] = af3_scores[cand_id]["quality"]

        maturation_results.append(summary)

        logger.info("    %s: %.1f → %.1f nM (%.1fx improvement, %d gens)",
                     cand_id, kd, result.final_kd_nM,
                     result.improvement_fold, result.n_generations)

    # Summary stats
    n_improved = sum(1 for r in maturation_results if r["improvement_fold"] > 1.5)
    n_target_reached = sum(1 for r in maturation_results if r["final_kd_nM"] <= target_kd)
    mean_improvement = sum(r["improvement_fold"] for r in maturation_results) / max(1, len(maturation_results))

    logger.info("  Maturation summary:")
    logger.info("    %d/%d improved >1.5x", n_improved, len(maturation_results))
    logger.info("    %d/%d reached target K_D (%.0f nM)", n_target_reached, len(maturation_results), target_kd)
    logger.info("    Mean improvement: %.1fx", mean_improvement)

    output = {
        "status": "completed",
        "kd_threshold_nM": kd_threshold,
        "target_kd_nM": target_kd,
        "n_total_candidates": len(ranked),
        "n_below_threshold": len(strong_binders),
        "n_matured": len(weak_binders),
        "n_improved_gt_1_5x": n_improved,
        "n_target_reached": n_target_reached,
        "mean_improvement_fold": round(mean_improvement, 2),
        "temperatures": temperatures,
        "max_generations": max_generations,
        "maturation_results": maturation_results,
    }

    save_json(output, output_dir / "B9b_affinity_maturation.json")
    logger.info("  B9b complete: %d candidates matured", len(maturation_results))

    return output

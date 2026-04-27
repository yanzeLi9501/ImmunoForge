"""
Candidate Ranking — Multi-criteria weighted scoring system.

Based on: pLDDT, ΔΔG, K_D, MPNN score, BSA, shape complementarity,
with penalties for aggregation risk and extreme pI.

v5: Adaptive weight redistribution when structural scores (ESMFold
pLDDT + interface PAE) are available, and K_D confidence weighting.
"""

import logging
import math
from dataclasses import dataclass

from immunoforge.core.utils import compute_mw, compute_pi, hydrophobic_fraction

logger = logging.getLogger(__name__)


@dataclass
class RankingWeights:
    """Adaptive ranking weight configuration.

    When ESMFold structural scores are available, ``structural_score``
    and ``iptm_proxy`` carry positive weights. Otherwise their budget
    is automatically redistributed to ``kd`` and ``mpnn_score``.
    """
    # Structure-derived (populated when ESMFold is available)
    structural_score: float = 0.25
    iptm_proxy: float = 0.15

    # Sequence / design quality
    mpnn_score: float = 0.15
    plddt: float = 0.10
    ddg: float = 0.10

    # Affinity prediction
    kd: float = 0.15
    kd_confidence: float = 0.05

    # Physicochemical
    bsa: float = 0.05

    # Penalties
    high_aggregation_penalty: float = -0.10
    extreme_pI_penalty: float = -0.05
    immunogenicity_penalty: float = -0.10


def compute_physicochemical(sequence: str) -> dict:
    """Compute basic physicochemical properties of a binder sequence."""
    pi = compute_pi(sequence)
    hf = hydrophobic_fraction(sequence)
    agg_risk = "low" if hf < 0.35 else ("moderate" if hf < 0.45 else "high")

    return {
        "length": len(sequence),
        "mw_da": round(compute_mw(sequence), 1),
        "pI": pi,
        "hydrophobic_fraction": hf,
        "aggregation_risk": agg_risk,
    }


def compute_composite_score(
    plddt: float,
    ddg: float,
    kd_nM: float,
    mpnn_score: float,
    bsa: float,
    sc: float,
    physico: dict,
    weights: dict | None = None,
    structural_score: float | None = None,
    iptm_proxy: float | None = None,
    kd_confidence: str | None = None,
) -> float:
    """Adaptive weighted composite score (higher = better).

    When *structural_score* (from ESMFold) is provided, the
    ``structural_score`` and ``iptm_proxy`` weight slots are active.
    Otherwise their budget is redistributed to ``kd`` and ``mpnn_score``
    so that the total weight always sums to ~1.0.

    Similarly, *kd_confidence* ("high"/"moderate"/"low") modulates the
    affinity weight: low-confidence K_D predictions get down-weighted
    and the freed budget goes to structural / MPNN components.

    Default weights from Methods §1.8 (v5 adaptive scheme).
    """
    if weights is None:
        weights = {
            "plddt": 0.10,
            "ddg": 0.10,
            "kd": 0.15,
            "mpnn_score": 0.15,
            "bsa": 0.05,
            "shape_complementarity": 0.10,
            "baseline": 0.10,
            "structural_score": 0.15,
            "iptm_proxy": 0.10,
        }

    w = dict(weights)  # mutable copy

    # ── Dynamic weight redistribution ──
    has_structure = structural_score is not None
    if not has_structure:
        # Shift structural budget → affinity + MPNN
        spare = w.pop("structural_score", 0) + w.pop("iptm_proxy", 0)
        w["kd"] = w.get("kd", 0.15) + spare * 0.6
        w["mpnn_score"] = w.get("mpnn_score", 0.15) + spare * 0.4

    # K_D confidence modulation
    if kd_confidence == "low":
        kd_penalty = w.get("kd", 0.15) * 0.4
        w["kd"] = w.get("kd", 0.15) - kd_penalty
        if has_structure:
            w["structural_score"] = w.get("structural_score", 0.15) + kd_penalty
        else:
            w["mpnn_score"] = w.get("mpnn_score", 0.15) + kd_penalty

    # ── Normalize components to [0, 1] ──
    s_plddt = min(plddt / 90.0, 1.0)
    s_ddg = min(abs(ddg) / 15.0, 1.0)
    s_kd = min(1.0, 1.0 / (1.0 + math.log10(max(kd_nM, 0.1)) / 6.0)) if kd_nM > 0 else 0
    s_mpnn = min(mpnn_score / 1.5, 1.0) if mpnn_score > 0 else 0
    s_bsa = min(bsa / 1200.0, 1.0)
    s_sc = min(sc / 0.7, 1.0)

    score = (
        w.get("plddt", 0.10) * s_plddt
        + w.get("ddg", 0.10) * s_ddg
        + w.get("kd", 0.15) * s_kd
        + w.get("mpnn_score", 0.15) * s_mpnn
        + w.get("bsa", 0.05) * s_bsa
        + w.get("shape_complementarity", 0.10) * s_sc
        + w.get("baseline", 0.10) * 1.0
    )

    # Structural components (when available)
    if has_structure:
        score += w.get("structural_score", 0.15) * min(structural_score, 1.0)
        score += w.get("iptm_proxy", 0.10) * min(iptm_proxy or 0.5, 1.0)

    # Penalties
    if physico.get("aggregation_risk") == "high":
        score -= 0.10
    if physico.get("pI", 7.0) < 4.5 or physico.get("pI", 7.0) > 10.5:
        score -= 0.05
    if physico.get("immunogenicity_risk") == "high":
        score -= 0.10

    return round(max(score, 0.0), 4)


def rank_candidates(
    candidates: list[dict],
    top_n: int = 20,
    weights: dict | None = None,
) -> list[dict]:
    """Score and rank a list of candidate dicts.

    Each candidate dict should have:
      id, sequence, target, plddt, ddg, kd_nM, mpnn_score, bsa, sc
    """
    scored = []
    for c in candidates:
        physico = compute_physicochemical(c["sequence"])
        score = compute_composite_score(
            plddt=c.get("plddt", 70),
            ddg=c.get("ddg", -10),
            kd_nM=c.get("kd_nM", 1000),
            mpnn_score=c.get("mpnn_score", 1.0),
            bsa=c.get("bsa", 1000),
            sc=c.get("sc", 0.6),
            physico=physico,
            weights=weights,
        )
        scored.append({
            **c,
            "composite_score": score,
            **physico,
        })

    scored.sort(key=lambda x: x["composite_score"], reverse=True)

    for rank, item in enumerate(scored, 1):
        item["rank"] = rank

    return scored[:top_n]

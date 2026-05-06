"""
Candidate Ranking — Multi-criteria weighted scoring system.

Based on: pLDDT, ΔΔG, K_D, MPNN score, BSA, shape complementarity,
with penalties for aggregation risk and extreme pI.

v6 (optimised): Weights calibrated against 18-entry benchmark per
Supplementary Method 10.  Three separate Boltz-2 structural
components (complex_iptm 20%, interface_ptm 12%, interface_pae 8%).
When Boltz-2 data are absent the 40% structural budget is redistributed
55% → K_D and 45% → MPNN score.
"""

import logging
import math
from dataclasses import dataclass

from immunoforge.core.utils import compute_mw, compute_pi, hydrophobic_fraction

logger = logging.getLogger(__name__)


@dataclass
class RankingWeights:
    """Adaptive ranking weight configuration (Supplementary Method 10 v6).

    With Boltz-2 structural data: complex_iptm + interface_ptm +
    interface_pae sum to 40% of the total weight budget.  Without
    structural data this 40% is redistributed 55% → kd, 45% → mpnn.
    Weights sum to 1.0 before penalties.
    """
    # Boltz-2 structural (populate when available)
    complex_iptm: float = 0.20
    interface_ptm: float = 0.12
    interface_pae: float = 0.08

    # Sequence / design quality
    plddt: float = 0.08
    ddg: float = 0.08
    mpnn_score: float = 0.12

    # Affinity prediction
    kd: float = 0.12

    # Physicochemical
    bsa: float = 0.05
    shape_complementarity: float = 0.08

    # Baseline offset
    baseline: float = 0.07

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
    complex_iptm: float | None = None,
    interface_ptm: float | None = None,
    interface_pae: float | None = None,
    kd_confidence: str | None = None,
) -> float:
    """Adaptive weighted composite score (higher = better).

    Implements Supplementary Method 10 v6 optimised weights.
    When Boltz-2 structural data are provided (*complex_iptm*,
    *interface_ptm*, *interface_pae*), the 40% structural budget is
    active.  Without structural data this budget is redistributed:
    55% to K_D and 45% to MPNN score.

    *interface_pae* is normalised as ``1 - pae/30`` so that lower PAE
    contributes a higher score (clamped to [0, 1]).

    *kd_confidence* ("high"/"moderate"/"low") triggers 40% K_D weight
    reallocation to structural or MPNN scores when confidence is low.
    """
    if weights is None:
        weights = {
            "plddt": 0.08,
            "ddg": 0.08,
            "kd": 0.12,
            "mpnn_score": 0.12,
            "bsa": 0.05,
            "shape_complementarity": 0.08,
            "baseline": 0.07,
            "complex_iptm": 0.20,
            "interface_ptm": 0.12,
            "interface_pae": 0.08,
        }

    w = dict(weights)  # mutable copy

    # ── Dynamic weight redistribution ──
    has_structure = complex_iptm is not None
    _STRUCTURAL_KEYS = ("complex_iptm", "interface_ptm", "interface_pae")
    if not has_structure:
        # Shift 40% structural budget → 55% kd, 45% mpnn
        spare = sum(w.pop(k, 0) for k in _STRUCTURAL_KEYS)
        w["kd"] = w.get("kd", 0.12) + spare * 0.55
        w["mpnn_score"] = w.get("mpnn_score", 0.12) + spare * 0.45

    # K_D confidence modulation (spread > 3.0 log units)
    if kd_confidence == "low":
        kd_penalty = w.get("kd", 0.12) * 0.4
        w["kd"] = w.get("kd", 0.12) - kd_penalty
        if has_structure:
            w["complex_iptm"] = w.get("complex_iptm", 0.20) + kd_penalty
        else:
            w["mpnn_score"] = w.get("mpnn_score", 0.12) + kd_penalty

    # ── Normalize components to [0, 1] ──
    s_plddt = min(plddt / 90.0, 1.0)
    s_ddg = min(abs(ddg) / 15.0, 1.0)
    s_kd = min(1.0, 1.0 / (1.0 + math.log10(max(kd_nM, 0.1)) / 6.0)) if kd_nM > 0 else 0
    s_mpnn = min(mpnn_score / 1.5, 1.0) if mpnn_score > 0 else 0
    s_bsa = min(bsa / 1200.0, 1.0)
    s_sc = min(sc / 0.7, 1.0)

    score = (
        w.get("plddt", 0.08) * s_plddt
        + w.get("ddg", 0.08) * s_ddg
        + w.get("kd", 0.12) * s_kd
        + w.get("mpnn_score", 0.12) * s_mpnn
        + w.get("bsa", 0.05) * s_bsa
        + w.get("shape_complementarity", 0.08) * s_sc
        + w.get("baseline", 0.07) * 1.0
    )

    # Structural components (when Boltz-2 data available)
    if has_structure:
        s_iptm = min(float(complex_iptm), 1.0)
        s_ptm = min(float(interface_ptm) if interface_ptm is not None else s_iptm * 0.9, 1.0)
        # PAE: lower PAE → higher score; normalise to 30 Å ceiling
        raw_pae = float(interface_pae) if interface_pae is not None else 10.0
        s_pae = max(0.0, 1.0 - raw_pae / 30.0)
        score += w.get("complex_iptm", 0.20) * s_iptm
        score += w.get("interface_ptm", 0.12) * s_ptm
        score += w.get("interface_pae", 0.08) * s_pae

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
            complex_iptm=c.get("complex_iptm"),
            interface_ptm=c.get("interface_ptm"),
            interface_pae=c.get("interface_pae"),
            kd_confidence=c.get("kd_confidence"),
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

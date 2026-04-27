"""
Developability Scoring Module — Comprehensive engineering property assessment.

Extends the basic QC module with additional manufacturability and
stability predictions for therapeutic protein development.

References:
    - Sormanni P et al. J Mol Biol 427:478 (2015) — CamSol solubility
    - Fernandez-Escamilla AM et al. Nat Biotechnol 22:1302 (2004) — TANGO aggregation
    - Lauer TM et al. J Pharm Sci 101:102 (2012) — Spatial aggregation propensity
"""

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from immunoforge.core.utils import compute_mw, compute_pi, hydrophobic_fraction

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Amino acid property scales
# ═══════════════════════════════════════════════════════════════════

# CamSol intrinsic solubility scale (Sormanni et al., 2015)
# Positive = solubility-promoting, Negative = aggregation-promoting
CAMSOL_SCALE = {
    "A":  0.02, "R":  1.31, "N":  0.62, "D":  0.98,
    "C": -0.50, "E":  0.96, "Q":  0.53, "G":  0.14,
    "H":  0.37, "I": -0.73, "L": -0.69, "K":  1.36,
    "M": -0.47, "F": -0.82, "P":  0.21, "S":  0.24,
    "T":  0.13, "W": -0.51, "Y": -0.32, "V": -0.55,
}

# Thermal stability contribution (simplified, from database statistics)
# Higher = more stabilizing
STABILITY_SCALE = {
    "A":  0.40, "R":  0.05, "N": -0.30, "D": -0.10,
    "C":  0.20, "E":  0.10, "Q": -0.20, "G": -0.50,
    "H": -0.10, "I":  0.80, "L":  0.70, "K":  0.10,
    "M":  0.30, "F":  0.60, "P": -0.40, "S": -0.10,
    "T":  0.10, "W":  0.50, "Y":  0.30, "V":  0.70,
}

# Disorder propensity (IUPred2-like simplified scale)
DISORDER_SCALE = {
    "A":  0.00, "R":  0.20, "N":  0.25, "D":  0.30,
    "C": -0.30, "E":  0.30, "Q":  0.20, "G":  0.10,
    "H": -0.10, "I": -0.40, "L": -0.30, "K":  0.25,
    "M": -0.20, "F": -0.35, "P":  0.35, "S":  0.15,
    "T":  0.05, "W": -0.40, "Y": -0.25, "V": -0.30,
}


# ═══════════════════════════════════════════════════════════════════
# Solubility prediction
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SolubilityResult:
    camsol_score: float        # per-residue average CamSol score
    solubility_class: str      # "high" | "moderate" | "low"
    insoluble_patches: list[dict] = field(default_factory=list)


def predict_solubility(sequence: str, window: int = 7) -> SolubilityResult:
    """CamSol-like intrinsic solubility prediction.

    Computes per-residue solubility scores and identifies insoluble patches.
    """
    residue_scores = [CAMSOL_SCALE.get(aa, 0.0) for aa in sequence]

    # Sliding window smoothing
    if len(residue_scores) >= window:
        smoothed = np.convolve(
            residue_scores, np.ones(window) / window, mode="valid"
        ).tolist()
    else:
        smoothed = residue_scores

    avg_score = np.mean(residue_scores)

    # Find insoluble patches (consecutive low-scoring regions)
    patches = []
    threshold = -0.3
    i = 0
    while i < len(smoothed):
        if smoothed[i] < threshold:
            start = i
            while i < len(smoothed) and smoothed[i] < threshold:
                i += 1
            if i - start >= 5:
                patches.append({
                    "start": start,
                    "end": i,
                    "length": i - start,
                    "mean_score": round(float(np.mean(smoothed[start:i])), 3),
                })
        else:
            i += 1

    if avg_score > 0.3:
        sol_class = "high"
    elif avg_score > -0.1:
        sol_class = "moderate"
    else:
        sol_class = "low"

    return SolubilityResult(
        camsol_score=round(float(avg_score), 3),
        solubility_class=sol_class,
        insoluble_patches=patches,
    )


# ═══════════════════════════════════════════════════════════════════
# Thermal stability prediction
# ═══════════════════════════════════════════════════════════════════

@dataclass
class StabilityResult:
    estimated_tm_celsius: float
    stability_score: float       # normalized 0-1
    stability_class: str         # "high" | "moderate" | "low"
    secondary_structure_content: dict = field(default_factory=dict)


def predict_thermal_stability(sequence: str) -> StabilityResult:
    """Estimate thermal stability (Tm) from sequence composition.

    Uses secondary structure propensity and amino acid stability scales
    to approximate folding stability. Helix-rich sequences tend to be
    more thermally stable than disordered sequences.
    """
    # Stability score from amino acid composition
    stability_scores = [STABILITY_SCALE.get(aa, 0.0) for aa in sequence]
    avg_stability = float(np.mean(stability_scores))

    # Secondary structure estimation (simplified)
    helix_promoters = sum(1 for aa in sequence if aa in "AELKM")
    sheet_promoters = sum(1 for aa in sequence if aa in "VIY")
    disorder_tendency = sum(DISORDER_SCALE.get(aa, 0.0) for aa in sequence) / max(len(sequence), 1)

    helix_frac = helix_promoters / max(len(sequence), 1)
    sheet_frac = sheet_promoters / max(len(sequence), 1)
    coil_frac = max(0, 1.0 - helix_frac - sheet_frac)

    # Tm estimation: baseline 50°C, adjusted by composition
    tm_base = 50.0
    tm_helix_bonus = helix_frac * 25.0    # High helix → more stable
    tm_stability_bonus = avg_stability * 10.0
    tm_disorder_penalty = max(0, disorder_tendency) * 15.0
    tm_length_factor = min(len(sequence) / 100.0, 1.0) * 5.0  # Larger proteins slightly more stable

    estimated_tm = tm_base + tm_helix_bonus + tm_stability_bonus - tm_disorder_penalty + tm_length_factor
    estimated_tm = max(30.0, min(90.0, estimated_tm))

    stability_normalized = (estimated_tm - 30.0) / 60.0  # 0-1 scale

    if estimated_tm >= 65.0:
        stability_class = "high"
    elif estimated_tm >= 50.0:
        stability_class = "moderate"
    else:
        stability_class = "low"

    return StabilityResult(
        estimated_tm_celsius=round(estimated_tm, 1),
        stability_score=round(stability_normalized, 3),
        stability_class=stability_class,
        secondary_structure_content={
            "helix_fraction": round(helix_frac, 3),
            "sheet_fraction": round(sheet_frac, 3),
            "coil_fraction": round(coil_frac, 3),
            "disorder_tendency": round(disorder_tendency, 3),
        },
    )


# ═══════════════════════════════════════════════════════════════════
# Charge distribution analysis
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ChargeResult:
    net_charge_pH7: float
    charge_distribution: dict
    charge_symmetry: float       # 0-1, higher = more symmetric
    nonspecific_binding_risk: str  # "low" | "moderate" | "high"


def analyze_charge_distribution(sequence: str) -> ChargeResult:
    """Analyze charge distribution and predict non-specific binding risk.

    Highly asymmetric charge distributions or extreme net charge
    increase risk of non-specific binding to cell surfaces.
    """
    pos_aa = sum(1 for aa in sequence if aa in "KRH")
    neg_aa = sum(1 for aa in sequence if aa in "DE")
    net = pos_aa - neg_aa

    # Sliding window charge map (window = 10)
    window = min(10, len(sequence))
    local_charges = []
    for i in range(len(sequence) - window + 1):
        window_seq = sequence[i: i + window]
        local = sum(1 for aa in window_seq if aa in "KRH") - sum(
            1 for aa in window_seq if aa in "DE"
        )
        local_charges.append(local)

    if local_charges:
        max_local = max(local_charges)
        min_local = min(local_charges)
        charge_range = max_local - min_local
    else:
        max_local = min_local = charge_range = 0

    # Symmetry: ratio of positive to negative charges
    total_charged = pos_aa + neg_aa
    if total_charged > 0:
        symmetry = 1.0 - abs(pos_aa - neg_aa) / total_charged
    else:
        symmetry = 1.0

    # Non-specific binding risk
    charge_density = total_charged / max(len(sequence), 1)
    if abs(net) > 10 or charge_range > 8:
        nsb_risk = "high"
    elif abs(net) > 5 or charge_range > 5:
        nsb_risk = "moderate"
    else:
        nsb_risk = "low"

    return ChargeResult(
        net_charge_pH7=net,
        charge_distribution={
            "positive_residues": pos_aa,
            "negative_residues": neg_aa,
            "total_charged": total_charged,
            "charge_density": round(charge_density, 3),
            "max_local_charge": max_local,
            "min_local_charge": min_local,
        },
        charge_symmetry=round(symmetry, 3),
        nonspecific_binding_risk=nsb_risk,
    )


# ═══════════════════════════════════════════════════════════════════
# Disulfide bond feasibility
# ═══════════════════════════════════════════════════════════════════

def check_disulfide_feasibility(sequence: str) -> dict:
    """Check cysteine pairing feasibility for disulfide bonds.

    Even cysteine count is required, and pairs should be spaced
    appropriately for structural disulfide formation.
    """
    cys_positions = [i for i, aa in enumerate(sequence) if aa == "C"]
    n_cys = len(cys_positions)

    if n_cys == 0:
        return {
            "n_cysteines": 0,
            "can_form_disulfide": False,
            "is_even": True,
            "potential_pairs": [],
            "note": "No cysteines present",
        }

    is_even = n_cys % 2 == 0

    # Check potential pairs: sequence separation > 10 residues preferred
    pairs = []
    for i in range(len(cys_positions)):
        for j in range(i + 1, len(cys_positions)):
            sep = cys_positions[j] - cys_positions[i]
            feasible = sep >= 10
            pairs.append({
                "cys1": cys_positions[i],
                "cys2": cys_positions[j],
                "separation": sep,
                "feasible": feasible,
            })

    return {
        "n_cysteines": n_cys,
        "can_form_disulfide": is_even and any(p["feasible"] for p in pairs),
        "is_even": is_even,
        "potential_pairs": pairs,
    }


# ═══════════════════════════════════════════════════════════════════
# Comprehensive developability report
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DevelopabilityReport:
    """Complete developability assessment for a binder candidate."""
    sequence: str
    overall_score: float          # 0-1, higher = more developable
    developability_class: str     # "excellent" | "good" | "acceptable" | "poor"
    solubility: SolubilityResult
    thermal_stability: StabilityResult
    charge: ChargeResult
    disulfide: dict
    physicochemical: dict
    flags: list[str] = field(default_factory=list)


def run_developability_assessment(sequence: str) -> DevelopabilityReport:
    """Run comprehensive developability assessment.

    Combines solubility, thermal stability, charge distribution,
    disulfide bond feasibility, and physicochemical properties
    into a single developability score.
    """
    sol = predict_solubility(sequence)
    stab = predict_thermal_stability(sequence)
    charge = analyze_charge_distribution(sequence)
    disulfide = check_disulfide_feasibility(sequence)
    pi = compute_pi(sequence)
    mw = compute_mw(sequence)
    hf = hydrophobic_fraction(sequence)

    physico = {
        "length": len(sequence),
        "mw_da": round(mw, 1),
        "pI": pi,
        "hydrophobic_fraction": hf,
    }

    # Composite score
    flags = []
    score = 0.0

    # Solubility (25%)
    sol_score = {"high": 1.0, "moderate": 0.6, "low": 0.2}.get(sol.solubility_class, 0.5)
    score += 0.25 * sol_score
    if sol.solubility_class == "low":
        flags.append("low_solubility")

    # Thermal stability (25%)
    score += 0.25 * stab.stability_score
    if stab.stability_class == "low":
        flags.append("low_thermal_stability")

    # Charge symmetry (15%)
    score += 0.15 * charge.charge_symmetry
    if charge.nonspecific_binding_risk == "high":
        flags.append("high_nonspecific_binding_risk")

    # pI in acceptable range (10%)
    if 5.0 <= pi <= 9.0:
        score += 0.10
    elif 4.5 <= pi <= 10.5:
        score += 0.05
    else:
        flags.append("extreme_pI")

    # Low aggregation (15%)
    agg_score = 1.0 if hf < 0.30 else (0.7 if hf < 0.40 else 0.3)
    score += 0.15 * agg_score
    if hf > 0.45:
        flags.append("high_aggregation_propensity")

    # Size reasonable for expression (10%)
    if 50 <= len(sequence) <= 300:
        score += 0.10
    elif 30 <= len(sequence) <= 500:
        score += 0.05
    else:
        flags.append("unusual_size")

    score = round(max(0.0, min(1.0, score)), 3)

    if score >= 0.80:
        dev_class = "excellent"
    elif score >= 0.60:
        dev_class = "good"
    elif score >= 0.40:
        dev_class = "acceptable"
    else:
        dev_class = "poor"

    return DevelopabilityReport(
        sequence=sequence,
        overall_score=score,
        developability_class=dev_class,
        solubility=sol,
        thermal_stability=stab,
        charge=charge,
        disulfide=disulfide,
        physicochemical=physico,
        flags=flags,
    )


def batch_developability(
    sequences: list[tuple[str, str]],
) -> list[dict]:
    """Run developability assessment on a batch of (id, sequence) tuples."""
    results = []
    for seq_id, seq in sequences:
        report = run_developability_assessment(seq)
        results.append({
            "id": seq_id,
            "overall_score": report.overall_score,
            "developability_class": report.developability_class,
            "solubility_class": report.solubility.solubility_class,
            "camsol_score": report.solubility.camsol_score,
            "estimated_tm": report.thermal_stability.estimated_tm_celsius,
            "stability_class": report.thermal_stability.stability_class,
            "net_charge": report.charge.net_charge_pH7,
            "nsb_risk": report.charge.nonspecific_binding_risk,
            "n_cysteines": report.disulfide["n_cysteines"],
            "mw_da": report.physicochemical["mw_da"],
            "pI": report.physicochemical["pI"],
            "flags": report.flags,
        })
    return results

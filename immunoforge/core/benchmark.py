"""
Benchmark Validation Module — Retrodiction benchmark against known proteins.

Provides a curated gold-standard validation dataset of known binder–target
pairs with experimentally measured K_D values, plus evaluation metrics
for assessing ImmunoForge pipeline prediction accuracy.

References:
    - Cao L et al. Nature 605:551 (2022)
    - Watson JL et al. Nature 620:1089 (2023)
    - Kastritis PL & Bonvin AMJJ. J R Soc Interface 10:20120835 (2013)
    - Cao L et al. Science 370:426 (2020)  — LCB1/LCB3 miniproteins
    - Silva DA et al. Nature 565:186 (2019) — Neo-2/15 (de novo IL-2 mimetic)
"""

import logging
import math
import hashlib
from dataclasses import dataclass, field

import numpy as np

from immunoforge.core.affinity import run_affinity_analysis

logger = logging.getLogger(__name__)


def _stable_seed_offset(text: str) -> int:
    """Return a deterministic 31-bit offset for reproducible benchmark runs."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31)


# ═══════════════════════════════════════════════════════════════════
# Gold-standard validation dataset
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkEntry:
    """A known binder-target pair for validation."""
    target_name: str
    binder_name: str
    target_pdb: str
    binder_type: str          # "antibody" | "de_novo" | "natural_ligand" | "engineered"
    experimental_kd_nM: float
    binder_sequence: str | None = None
    target_sequence: str | None = None
    bsa_experimental: float | None = None
    sc_experimental: float | None = None
    cell_type: str = ""
    species: str = "human"
    source: str = ""          # Literature reference
    pdb_complex: str = ""     # Complex structure PDB ID


# T cell targets
BENCHMARK_ENTRIES: list[BenchmarkEntry] = [
    # ── T Cell ──
    BenchmarkEntry(
        target_name="CD3E", binder_name="OKT3",
        target_pdb="6JXR", binder_type="antibody",
        experimental_kd_nM=1.0,
        binder_sequence="QVQLQQSGAELARPGASVKLSCKASGYTFTSYWMHWVKQRPGQGLEWIGEINPTNGHTNYNEKFKSKATLTVDKSSSTAYMQLSSLTSEDSAVYYCARRGYYYGSRDAMDY",
        bsa_experimental=1650.0, sc_experimental=0.71,
        cell_type="T cell", source="OKT3 clinical mAb",
    ),
    BenchmarkEntry(
        target_name="PD-1", binder_name="Nivolumab_VH",
        target_pdb="5WT9", binder_type="antibody",
        experimental_kd_nM=2.6,
        binder_sequence="QVQLVESGGGVVQPGRSLRLDCKASGITFSNSGMHWVRQAPGKGLEWVAVIWYDGSKRYYADSVKGRFTISRDNSKNTLFLQMNSLRAEDTAVYYCATNDDYWGQGTLVTVSS",
        bsa_experimental=1820.0, sc_experimental=0.68,
        cell_type="T cell", source="Nivolumab (Opdivo)",
    ),
    BenchmarkEntry(
        target_name="PD-1", binder_name="Pembrolizumab_VH",
        target_pdb="4ZQK", binder_type="antibody",
        experimental_kd_nM=0.027,
        binder_sequence="QVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTNFNEKFKNRVTLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQGTTVTVSS",
        bsa_experimental=2100.0, sc_experimental=0.72,
        cell_type="T cell", source="Pembrolizumab (Keytruda)",
    ),
    BenchmarkEntry(
        target_name="CTLA-4", binder_name="Ipilimumab_VH",
        target_pdb="3OSK", binder_type="antibody",
        experimental_kd_nM=6.2,
        binder_sequence="QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYTMHWVRQAPGKGLEWVTFISYDGNNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAIYYCARTGWLGPFDYWGQGTLVTVSS",
        bsa_experimental=1550.0, sc_experimental=0.65,
        cell_type="T cell", source="Ipilimumab (Yervoy)",
    ),

    # ── DC ──
    BenchmarkEntry(
        target_name="CLEC9A", binder_name="10B12_VH",
        target_pdb="3VPP", binder_type="antibody",
        experimental_kd_nM=0.4,
        binder_sequence="EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYPDSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDRLSITIRPRYYGLDVWGQGTTVTVSS",
        bsa_experimental=1750.0, sc_experimental=0.70,
        cell_type="DC", source="Anti-CLEC9A 10B12 mAb",
    ),

    # ── NK cell ──
    BenchmarkEntry(
        target_name="NKG2D", binder_name="MICA_ectodomain",
        target_pdb="1HQ8", binder_type="natural_ligand",
        experimental_kd_nM=1000.0,
        binder_sequence="EPHSLRYNLTVLSWDGSVQSGFLAEGHLDGQPFLRCDRQKCRAKPQGQWAEDVLGNKTWDRETRDLTGNGKDLRMTLAHIKDQKEGLHSLQEIRVCEIHEDSSTRGSRHFYYDGELFLSQNLETKEWTMPQSSRAQTLAMNVRNFLKEDAMKTKTHYHAMHADCLQELRRYLKSGVVLRRTVPPMVNVTRSEASEGNITVTCRASGFYPWNITLSWRQDGVSLSHDTQQWGDVLPDGNGTYQTWVATRICQGEEQRFTCYMEHSGNHSTHPVPSGKVLVLQSHW",
        bsa_experimental=1900.0, sc_experimental=0.62,
        cell_type="NK cell", source="NKG2D-MICA crystal structure",
    ),
    BenchmarkEntry(
        target_name="NKG2D", binder_name="ULBP2",
        target_pdb="4PBL", binder_type="natural_ligand",
        experimental_kd_nM=500.0,
        cell_type="NK cell", source="NKG2D-ULBP2 published K_D",
    ),

    # ── B cell ──
    BenchmarkEntry(
        target_name="CD19", binder_name="FMC63_scFv",
        target_pdb="6AL5", binder_type="antibody",
        experimental_kd_nM=5.0,
        binder_sequence="DIQMTQTTSSLSASLGDRVTISCRASQDISKYLNWYQQKPDGTVKLLIYHTSRLHSGVPSRFSGSGSGTDYSLTISNLEQEDIATYFCQQGNTLPYTFGGGTKLEIT",
        bsa_experimental=1500.0, sc_experimental=0.67,
        cell_type="B cell", source="FMC63 CAR-T scFv",
    ),
    BenchmarkEntry(
        target_name="BCMA", binder_name="Belantamab_VH",
        target_pdb="1OQD", binder_type="antibody",
        experimental_kd_nM=0.4,
        cell_type="B cell", source="Belantamab mafodotin",
    ),

    # ── Macrophage ──
    BenchmarkEntry(
        target_name="CD47", binder_name="SIRPa_D1",
        target_pdb="2JJS", binder_type="natural_ligand",
        experimental_kd_nM=1000.0,
        binder_sequence="EEELQVIQPDKSVSVAAGESAILHCTVTSLIPVGPIQWFRGAGPARELIYNQKEGHFPRVTTVSESTKRENMDFSISISNITPADAGTYYCVKFRKGSPDTEFKSGAGTELSVRAKPS",
        bsa_experimental=1350.0, sc_experimental=0.61,
        cell_type="macrophage", source="SIRPa-CD47 crystal",
    ),
    BenchmarkEntry(
        target_name="CD47", binder_name="Magrolimab_VH",
        target_pdb="2JJS", binder_type="antibody",
        experimental_kd_nM=1.0,
        cell_type="macrophage", source="Magrolimab (Hu5F9-G4)",
    ),
    BenchmarkEntry(
        target_name="SIRPa", binder_name="CD47v_engineered",
        target_pdb="4CMF", binder_type="engineered",
        experimental_kd_nM=0.011,
        cell_type="macrophage", source="High-affinity CD47 variant",
    ),

    # ── De novo design gold standards ──
    BenchmarkEntry(
        target_name="TrkA", binder_name="RFdiffusion_binder",
        target_pdb="7SI2", binder_type="de_novo",
        experimental_kd_nM=15.0,
        source="Cao et al., Nature 2022",
    ),
    BenchmarkEntry(
        target_name="IL-7Ra", binder_name="RFdiffusion_binder",
        target_pdb="7SI3", binder_type="de_novo",
        experimental_kd_nM=50.0,
        source="Cao et al., Nature 2022",
    ),
    BenchmarkEntry(
        target_name="PD-L1", binder_name="RFdiffusion_binder",
        target_pdb="8DS6", binder_type="de_novo",
        experimental_kd_nM=2.0,
        source="Watson et al., Nature 2023",
    ),
    BenchmarkEntry(
        target_name="Influenza_HA", binder_name="HA_20_RFdiffusion",
        target_pdb="8SK7", binder_type="de_novo",
        experimental_kd_nM=5.0,
        binder_sequence="MEKEKELKEYAEKIKKEIGDIESVEVKDGKILVKAKKITDKTVDAIMKLTVKAARLGFKVEVELV",
        bsa_experimental=900.0, sc_experimental=0.65,
        source="Watson et al., Nature 620:1089 (2023)",
        pdb_complex="8SK7",
    ),

    # ── Published de novo miniprotein binders (PDB-confirmed sequences) ──
    BenchmarkEntry(
        target_name="SARS-CoV-2_RBD", binder_name="LCB1",
        target_pdb="7JZL", binder_type="de_novo",
        experimental_kd_nM=0.2,
        binder_sequence="DKEWILQKIYEIMRLLDELGHAEASMRVSDLIYEFMKKGDERLLEEAERLLEEVER",
        bsa_experimental=1050.0, sc_experimental=0.72,
        source="Cao et al., Science 370:426 (2020)",
        pdb_complex="7JZL",
    ),
    BenchmarkEntry(
        target_name="SARS-CoV-2_RBD", binder_name="LCB3",
        target_pdb="7JZM", binder_type="de_novo",
        experimental_kd_nM=48.0,
        binder_sequence="LNDELHMLMTDLVYEALHFAKDEEIKKRVFQLFELADKAYKNNDRQKLEKVVEELKELLERLL",
        bsa_experimental=980.0, sc_experimental=0.68,
        source="Cao et al., Science 370:426 (2020)",
        pdb_complex="7JZM",
    ),
    BenchmarkEntry(
        target_name="IL-2Rbg", binder_name="Neo-2/15",
        target_pdb="6DG5", binder_type="de_novo",
        experimental_kd_nM=4.5,
        binder_sequence="HMPKKKIQLHAEHALYDALMLNIVKTNSPPAEEKLEDYAFNFELILEEIARLFESGDQKDEAEKAKRMKEWMKRIKTTASEDEQEEMANAIITILQSWIFS",
        bsa_experimental=1400.0, sc_experimental=0.71,
        cell_type="T cell", source="Silva et al., Nature 565:186 (2019)",
        pdb_complex="6DG5",
    ),
]


# ═══════════════════════════════════════════════════════════════════
# Benchmark evaluation metrics
# ═══════════════════════════════════════════════════════════════════

def get_benchmark_entries(
    cell_type: str | None = None,
    binder_type: str | None = None,
    with_sequence: bool = False,
) -> list[BenchmarkEntry]:
    """Filter benchmark entries by criteria."""
    entries = BENCHMARK_ENTRIES
    if cell_type:
        entries = [e for e in entries if e.cell_type.lower() == cell_type.lower()]
    if binder_type:
        entries = [e for e in entries if e.binder_type == binder_type]
    if with_sequence:
        entries = [e for e in entries if e.binder_sequence is not None]
    return entries


def run_benchmark(
    entries: list[BenchmarkEntry] | None = None,
    seed: int = 42,
) -> dict:
    """Run retrodiction benchmark on gold-standard entries.

    For entries with binder sequences, runs the affinity prediction
    pipeline and compares predicted K_D to experimental values.

    Returns:
        Comprehensive benchmark results with metrics.
    """
    if entries is None:
        entries = get_benchmark_entries(with_sequence=True)

    predictions = []

    for entry in entries:
        if entry.binder_sequence is None:
            continue

        bsa = entry.bsa_experimental or (len(entry.binder_sequence) * 14)
        sc = entry.sc_experimental or 0.65

        # Map benchmark binder_type to affinity calibration class
        _BT_TO_CAL = {"de_novo": "denovo", "engineered": "denovo"}
        type_override = _BT_TO_CAL.get(entry.binder_type)

        result = run_affinity_analysis(
            entry.binder_sequence, bsa, sc,
            seed=seed + _stable_seed_offset(entry.binder_name),
            binder_type_override=type_override,
        )
        consensus = result.get("consensus", {})
        predicted_kd = consensus.get("consensus_kd_nM", None)

        if predicted_kd is not None and predicted_kd > 0:
            log_error = math.log10(predicted_kd) - math.log10(entry.experimental_kd_nM)
        else:
            log_error = None

        predictions.append({
            "target": entry.target_name,
            "binder": entry.binder_name,
            "binder_type": entry.binder_type,
            "cell_type": entry.cell_type,
            "experimental_kd_nM": entry.experimental_kd_nM,
            "predicted_kd_nM": round(predicted_kd, 2) if predicted_kd else None,
            "log10_error": round(log_error, 3) if log_error is not None else None,
            "confidence": consensus.get("confidence", "N/A"),
            "prodigy_kd": result.get("prodigy", {}).get("kd_nM"),
            "rosetta_kd": result.get("rosetta", {}).get("kd_nM"),
            "bsa_reg_kd": result.get("bsa_reg", {}).get("kd_nM"),
        })

    # Compute summary metrics
    metrics = _compute_metrics(predictions)

    return {
        "n_entries": len(predictions),
        "metrics": metrics,
        "predictions": predictions,
    }


def run_benchmark_by_class(seed: int = 42) -> dict:
    """Run benchmark and compute per-class metrics (antibody vs de_novo vs natural_ligand).

    Returns:
        Dict with overall and per-class metrics + predictions.
    """
    all_entries = get_benchmark_entries(with_sequence=True)
    overall = run_benchmark(all_entries, seed=seed)

    class_results = {}
    for cls in ("antibody", "de_novo", "natural_ligand"):
        cls_entries = [e for e in all_entries if e.binder_type == cls]
        if cls_entries:
            cls_result = run_benchmark(cls_entries, seed=seed)
            class_results[cls] = cls_result

    return {
        "overall": overall,
        "by_class": class_results,
    }


def _compute_metrics(predictions: list[dict]) -> dict:
    """Compute benchmark evaluation metrics."""
    valid = [p for p in predictions if p["log10_error"] is not None]
    if not valid:
        return {"error": "No valid predictions to evaluate"}

    log_errors = [abs(p["log10_error"]) for p in valid]
    signed_errors = [p["log10_error"] for p in valid]
    exp_kds = [math.log10(p["experimental_kd_nM"]) for p in valid]
    pred_kds = [math.log10(p["predicted_kd_nM"]) for p in valid if p["predicted_kd_nM"]]

    # Mean Absolute Log Error
    male = float(np.mean(log_errors))

    # Mean Signed Error (systematic bias)
    mse = float(np.mean(signed_errors))

    # Spearman rank correlation
    spearman_rho = _spearman_correlation(exp_kds, pred_kds[:len(exp_kds)])

    # Fraction within 1 order of magnitude
    within_1_log = sum(1 for e in log_errors if e <= 1.0) / max(len(log_errors), 1)

    # Fraction within 1.5 orders of magnitude
    within_1_5_log = sum(1 for e in log_errors if e <= 1.5) / max(len(log_errors), 1)

    return {
        "mean_absolute_log10_error": round(male, 3),
        "mean_signed_log10_error": round(mse, 3),
        "spearman_rho": round(spearman_rho, 3) if spearman_rho is not None else None,
        "fraction_within_1_log": round(within_1_log, 3),
        "fraction_within_1.5_log": round(within_1_5_log, 3),
        "n_evaluated": len(valid),
        "systematic_bias": (
            "over-predicting (weaker)" if mse > 0.5
            else "under-predicting (stronger)" if mse < -0.5
            else "acceptable"
        ),
    }


def _spearman_correlation(x: list[float], y: list[float]) -> float | None:
    """Compute Spearman rank correlation coefficient."""
    n = min(len(x), len(y))
    if n < 3:
        return None

    x = x[:n]
    y = y[:n]

    # Rank
    def _rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        for r, idx in enumerate(order):
            ranks[idx] = float(r + 1)
        return ranks

    rx = _rank(x)
    ry = _rank(y)

    d_sq = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    rho = 1.0 - 6.0 * d_sq / (n * (n ** 2 - 1))
    return max(-1.0, min(1.0, rho))


# ═══════════════════════════════════════════════════════════════════
# Leave-one-out cross-validation and bootstrap CIs
# ═══════════════════════════════════════════════════════════════════

def run_loo_cv(
    entries: list[BenchmarkEntry] | None = None,
    seed: int = 42,
) -> dict:
    """Leave-one-out cross-validation for the benchmark.

    For each left-out entry, the remaining n-1 entries are used to
    evaluate whether the scoring system generalises.  Because the
    consensus weights and calibration offset are fixed hyperparameters
    (not re-fitted per fold), LOO-CV here measures prediction
    stability rather than true out-of-sample calibration.

    Returns dict with LOO metrics and per-fold predictions.
    """
    if entries is None:
        entries = get_benchmark_entries(with_sequence=True)

    loo_predictions = []
    for i, held_out in enumerate(entries):
        if held_out.binder_sequence is None:
            continue
        bsa = held_out.bsa_experimental or (len(held_out.binder_sequence) * 14)
        sc = held_out.sc_experimental or 0.65
        result = run_affinity_analysis(
            held_out.binder_sequence, bsa, sc,
            seed=seed + _stable_seed_offset(held_out.binder_name),
        )
        consensus = result.get("consensus", {})
        pred_kd = consensus.get("consensus_kd_nM")
        if pred_kd and pred_kd > 0:
            log_err = math.log10(pred_kd) - math.log10(held_out.experimental_kd_nM)
        else:
            log_err = None
        loo_predictions.append({
            "fold": i,
            "binder": held_out.binder_name,
            "target": held_out.target_name,
            "experimental_kd_nM": held_out.experimental_kd_nM,
            "predicted_kd_nM": round(pred_kd, 4) if pred_kd else None,
            "log10_error": round(log_err, 3) if log_err is not None else None,
        })

    valid = [p for p in loo_predictions if p["log10_error"] is not None]
    if not valid:
        return {"error": "No valid LOO predictions"}

    abs_errors = [abs(p["log10_error"]) for p in valid]
    exp_kds = [math.log10(p["experimental_kd_nM"]) for p in valid]
    pred_kds = [math.log10(p["predicted_kd_nM"]) for p in valid if p["predicted_kd_nM"]]

    return {
        "loo_male": round(float(np.mean(abs_errors)), 3),
        "loo_spearman_rho": round(
            _spearman_correlation(exp_kds, pred_kds[:len(exp_kds)]), 3
        ) if _spearman_correlation(exp_kds, pred_kds[:len(exp_kds)]) is not None else None,
        "loo_within_1_log": round(
            sum(1 for e in abs_errors if e <= 1.0) / len(abs_errors), 3
        ),
        "n_folds": len(valid),
        "per_fold": loo_predictions,
    }


def bootstrap_ci(
    entries: list[BenchmarkEntry] | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict:
    """Bootstrap confidence intervals for MALE and Spearman ρ.

    Resamples (with replacement) from the benchmark predictions and
    computes percentile-based CIs.

    Returns dict with point estimates and (lower, upper) CIs.
    """
    if entries is None:
        entries = get_benchmark_entries(with_sequence=True)

    # First get full predictions
    full_result = run_benchmark(entries, seed=seed)
    predictions = full_result["predictions"]
    valid = [p for p in predictions if p["log10_error"] is not None]

    if len(valid) < 3:
        return {"error": "Too few entries for bootstrap"}

    rng = np.random.RandomState(seed)
    alpha = (1 - confidence) / 2

    boot_males = []
    boot_rhos = []

    for _ in range(n_bootstrap):
        indices = rng.choice(len(valid), size=len(valid), replace=True)
        sample = [valid[i] for i in indices]

        abs_errors = [abs(p["log10_error"]) for p in sample]
        boot_males.append(float(np.mean(abs_errors)))

        exp_kds = [math.log10(p["experimental_kd_nM"]) for p in sample]
        pred_kds = [
            math.log10(p["predicted_kd_nM"]) for p in sample
            if p["predicted_kd_nM"] and p["predicted_kd_nM"] > 0
        ]
        rho = _spearman_correlation(exp_kds, pred_kds[:len(exp_kds)])
        if rho is not None:
            boot_rhos.append(rho)

    male_ci = (
        round(float(np.percentile(boot_males, alpha * 100)), 3),
        round(float(np.percentile(boot_males, (1 - alpha) * 100)), 3),
    )
    rho_ci = (
        round(float(np.percentile(boot_rhos, alpha * 100)), 3),
        round(float(np.percentile(boot_rhos, (1 - alpha) * 100)), 3),
    ) if boot_rhos else (None, None)

    return {
        "male_point": full_result["metrics"]["mean_absolute_log10_error"],
        "male_95ci": male_ci,
        "rho_point": full_result["metrics"]["spearman_rho"],
        "rho_95ci": rho_ci,
        "n_bootstrap": n_bootstrap,
        "n_entries": len(valid),
    }


# ═══════════════════════════════════════════════════════════════════
# PDB structure download helpers
# ═══════════════════════════════════════════════════════════════════

def get_pdb_ids_for_benchmark() -> list[str]:
    """Return list of PDB IDs needed for benchmark experiments."""
    pdbs = set()
    for entry in BENCHMARK_ENTRIES:
        if entry.target_pdb:
            pdbs.add(entry.target_pdb)
        if entry.pdb_complex:
            pdbs.add(entry.pdb_complex)
    return sorted(pdbs)


def download_benchmark_structures(output_dir: str) -> dict:
    """Download all PDB structures needed for benchmark validation."""
    from immunoforge.pipeline.steps.B1_target_prep import download_pdb
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {"downloaded": [], "failed": []}
    for pdb_id in get_pdb_ids_for_benchmark():
        path = download_pdb(pdb_id, output_dir)
        if path and path.exists():
            results["downloaded"].append(pdb_id)
        else:
            results["failed"].append(pdb_id)

    return results


def generate_benchmark_report(benchmark_result: dict) -> str:
    """Generate a human-readable benchmark report as Markdown."""
    metrics = benchmark_result.get("metrics", {})
    predictions = benchmark_result.get("predictions", [])

    lines = [
        "# ImmunoForge Benchmark Validation Report",
        "",
        "## Summary Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Entries evaluated | {metrics.get('n_evaluated', 0)} |",
        f"| Mean |Δlog₁₀K_D| | {metrics.get('mean_absolute_log10_error', 'N/A')} |",
        f"| Mean signed error | {metrics.get('mean_signed_log10_error', 'N/A')} |",
        f"| Spearman ρ | {metrics.get('spearman_rho', 'N/A')} |",
        f"| Fraction within 1 log | {metrics.get('fraction_within_1_log', 'N/A')} |",
        f"| Fraction within 1.5 log | {metrics.get('fraction_within_1.5_log', 'N/A')} |",
        f"| Systematic bias | {metrics.get('systematic_bias', 'N/A')} |",
        "",
        "## Per-Entry Results",
        "",
        "| Target | Binder | Exp K_D (nM) | Pred K_D (nM) | |Δlog₁₀| | Confidence |",
        "|--------|--------|-------------|--------------|---------|------------|",
    ]

    for p in predictions:
        exp = f"{p['experimental_kd_nM']:.2f}"
        pred = f"{p['predicted_kd_nM']:.2f}" if p["predicted_kd_nM"] else "N/A"
        err = f"{abs(p['log10_error']):.2f}" if p["log10_error"] is not None else "N/A"
        conf = p.get("confidence", "N/A")
        lines.append(
            f"| {p['target']} | {p['binder']} | {exp} | {pred} | {err} | {conf} |"
        )

    lines.extend(["", "---", f"*Generated by ImmunoForge benchmark module*"])
    return "\n".join(lines)

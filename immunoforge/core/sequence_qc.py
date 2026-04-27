"""
Sequence Quality Control — Multi-layer filtering for designed binder sequences.

Checks:
  - Protease cleavage site motifs (furin, thrombin, enterokinase, MMP, cathepsin B)
  - Toxicity motifs (poly-cationic, NLS)
  - Aggregation-prone regions (APR)
  - Cysteine parity (odd Cys → unpaired → aggregation)
  - Isoelectric point extremes
"""

import logging
import re

from immunoforge.core.utils import compute_pi

logger = logging.getLogger(__name__)

# ── Protease cleavage motifs ──
PROTEASE_MOTIFS = {
    "furin": r"R.[KR]R",
    "thrombin": r"LVPR.S",
    "enterokinase": r"DDDDK",
    "mmp_PxxP": r"P..P",
    "mmp_PLG": r"PLG",
    "cathepsin_b_GFLG": r"GFLG",
}

DIBASIC_MOTIFS = ["KK", "RR", "KR", "RK"]

# ── Toxicity motifs ──
TOXICITY_MOTIFS = {
    "poly_cationic": r"[RK]{6,}",
    "nls_signal": r"[KR]{4}.[KR]{3}",
}

# ── Aggregation-prone region ──
APR_PATTERN = r"[VILMFYW]{5,}"


def _count_motif(sequence: str, pattern: str) -> int:
    return len(re.findall(pattern, sequence))


def check_protease_sites(sequence: str) -> tuple[dict, list[str]]:
    """Screen for protease cleavage sites.

    Returns (details_dict, list_of_failure_reasons).
    """
    details = {}
    failures = []
    length = len(sequence)

    for name, pattern in PROTEASE_MOTIFS.items():
        count = _count_motif(sequence, pattern)
        details[name] = count
        # Furin, thrombin, enterokinase, GFLG: any hit = reject
        if name in ("furin", "thrombin", "enterokinase", "cathepsin_b_GFLG") and count > 0:
            failures.append(name)

    # Cathepsin B dibasic density check
    dibasic_count = sum(sequence.count(m) for m in DIBASIC_MOTIFS)
    dibasic_density = dibasic_count / (length / 100) if length > 0 else 0
    details["CatB_dibasic_count"] = dibasic_count
    details["CatB_dibasic_density"] = round(dibasic_density, 1)
    if dibasic_density > 5.0:
        failures.append("CatB_dibasic_high")

    return details, failures


def check_toxicity(sequence: str) -> list[str]:
    """Screen for known toxicity motifs. Returns list of hit names."""
    hits = []
    for name, pattern in TOXICITY_MOTIFS.items():
        if re.search(pattern, sequence):
            hits.append(name)
    return hits


def check_aggregation(sequence: str) -> dict:
    """Check for aggregation-prone regions."""
    matches = re.findall(APR_PATTERN, sequence)
    return {
        "apr_count": len(matches),
        "apr_regions": matches[:5],
        "pass": len(matches) == 0,
    }


def check_cysteine_parity(sequence: str) -> dict:
    """Check cysteine count is even (can form disulfide pairs)."""
    n_cys = sequence.count("C")
    return {
        "n_cysteines": n_cys,
        "is_even": n_cys % 2 == 0,
        "pass": n_cys % 2 == 0,
    }


def check_isoelectric_point(sequence: str, min_pi=4.5, max_pi=10.5) -> dict:
    """Check pI within acceptable range."""
    pi = compute_pi(sequence)
    return {
        "pI": pi,
        "pass": min_pi <= pi <= max_pi,
    }


def run_full_qc(sequence: str, config: dict | None = None) -> dict:
    """Run complete QC pipeline on a single sequence.

    Returns dict with pass/fail status and detailed results.
    """
    if config is None:
        config = {}

    qc_cfg = config.get("sequence_qc", {})
    min_pi = qc_cfg.get("isoelectric_point", {}).get("min_pi", 4.5)
    max_pi = qc_cfg.get("isoelectric_point", {}).get("max_pi", 10.5)

    failures = []

    # Protease sites
    protease_details, protease_fails = check_protease_sites(sequence)
    failures.extend(protease_fails)

    # Toxicity
    toxicity_hits = check_toxicity(sequence)
    if toxicity_hits:
        failures.append("toxic")

    # Aggregation
    agg = check_aggregation(sequence)
    if not agg["pass"]:
        failures.append("aggregation_prone")

    # Cysteine parity
    cys = check_cysteine_parity(sequence)
    if not cys["pass"]:
        failures.append("odd_Cys")

    # Isoelectric point
    pi_result = check_isoelectric_point(sequence, min_pi, max_pi)
    if not pi_result["pass"]:
        failures.append("extreme_pI")

    overall_pass = len(failures) == 0

    return {
        "pass": overall_pass,
        "failures": failures,
        "protease": protease_details,
        "toxicity": toxicity_hits,
        "aggregation": agg,
        "cysteine": cys,
        "isoelectric_point": pi_result,
        "sequence_length": len(sequence),
    }


def batch_qc(sequences: list[tuple[str, str]], config: dict | None = None) -> dict:
    """Run QC on a batch of (id, sequence) tuples.

    Returns summary dict with passed/failed lists.
    """
    passed = []
    failed = []
    failure_reasons = {}

    for seq_id, seq in sequences:
        result = run_full_qc(seq, config)
        result["id"] = seq_id
        if result["pass"]:
            passed.append(result)
        else:
            failed.append(result)
            for reason in result["failures"]:
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    return {
        "total": len(sequences),
        "n_passed": len(passed),
        "n_failed": len(failed),
        "pass_rate": round(len(passed) / max(len(sequences), 1), 3),
        "failure_reasons": failure_reasons,
        "passed": passed,
        "failed": failed,
    }

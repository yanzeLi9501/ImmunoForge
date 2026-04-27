"""
B4c: AF2-Multimer Orthogonal Filter — ColabFold-based complex quality scoring.

Uses ColabFold (AlphaFold2-multimer v3) to predict binder–target complexes
and re-rank candidates by ipTM. This provides an orthogonal structural
quality signal independent of single-chain ESMFold pLDDT.

When ColabFold is not available, falls back to single-chain ESMFold scoring
with a pTM-based proxy (no ipTM available without multimer model).

Flow:
  1. Load candidates from B6 ranking (or B4 structure validation)
  2. For each candidate: run ColabFold AF2-multimer on (binder, target)
  3. Parse ipTM, pTM, pLDDT from output
  4. Re-rank candidates; flag LOW quality (ipTM < 0.4) for rejection
  5. Output: B4c_af2_multimer_filter.json

Input:  B6_candidate_ranking.json + target sequences from config
Output: B4c_af2_multimer_filter.json with ipTM scores and re-ranking

This is the leverage point for pipeline accuracy: AF2-multimer ipTM
directly measures predicted interface quality, filtering out binders
that fold well alone but fail to form a correct complex.
"""

import json
import logging
from pathlib import Path

from immunoforge.core.structure_validation import (
    run_af2_multimer,
    validate_structure,
    available_backends,
)
from immunoforge.core.utils import save_json

logger = logging.getLogger(__name__)


def main(config: dict) -> dict:
    """Execute B4c: AF2-Multimer Orthogonal Filter.

    Requires ColabFold installed on the system. When unavailable, produces
    a degraded result using single-chain ESMFold only.
    """
    logger.info("  B4c: AF2-Multimer Orthogonal Filter")

    output_dir = Path(config.get("paths", {}).get("output_dir", "outputs"))

    # Load B6 ranked candidates (preferred) or B4 validated candidates
    candidates = _load_candidates(output_dir)
    if not candidates:
        logger.warning("  No candidates found for AF2-multimer filtering")
        return {"status": "skipped", "reason": "no_candidates_found"}

    # Get target sequences from config
    target_seqs = _get_target_sequences(config)
    if not target_seqs:
        logger.warning("  No target sequences in config, skipping AF2 filter")
        return {"status": "skipped", "reason": "no_target_sequences"}

    # Configuration
    af2_cfg = config.get("af2_filter", {})
    max_candidates = af2_cfg.get("max_candidates", 10)
    iptm_reject_threshold = af2_cfg.get("iptm_reject_threshold", 0.4)
    iptm_high_threshold = af2_cfg.get("iptm_high_threshold", 0.7)

    # Check backend availability
    backends = available_backends()
    use_af2 = "af2_multimer" in backends

    if use_af2:
        logger.info("  ColabFold AF2-multimer backend available — running complex prediction")
    else:
        logger.info("  ColabFold not available — using ESMFold single-chain proxy")

    # Process candidates
    results = []
    for i, cand in enumerate(candidates[:max_candidates]):
        cand_id = cand.get("id", f"cand_{i}")
        sequence = cand.get("sequence", "")
        if not sequence:
            continue

        logger.info(f"  [{i+1}/{min(len(candidates), max_candidates)}] {cand_id}")

        # Pick the first available target sequence for binary complex
        target_name, target_seq = next(iter(target_seqs.items()))

        if use_af2:
            result = _run_af2_filter(
                cand_id, sequence, target_seq, target_name, output_dir
            )
        else:
            result = _run_esmfold_proxy(cand_id, sequence)

        result["original_rank"] = i + 1
        result["original_kd_nM"] = cand.get("kd_nM")
        results.append(result)

    # Re-rank by ipTM (or pTM proxy)
    score_key = "iptm" if use_af2 else "ptm_proxy"
    results.sort(key=lambda r: r.get(score_key, 0), reverse=True)
    for i, r in enumerate(results):
        r["af2_rank"] = i + 1

    # Classify quality
    n_high = 0
    n_medium = 0
    n_low = 0
    for r in results:
        score = r.get(score_key, 0)
        if score >= iptm_high_threshold:
            r["af2_quality"] = "HIGH"
            n_high += 1
        elif score >= iptm_reject_threshold:
            r["af2_quality"] = "MEDIUM"
            n_medium += 1
        else:
            r["af2_quality"] = "LOW"
            n_low += 1

    output = {
        "status": "completed",
        "backend": "af2_multimer" if use_af2 else "esmfold_proxy",
        "n_candidates_scored": len(results),
        "n_high": n_high,
        "n_medium": n_medium,
        "n_low": n_low,
        "iptm_reject_threshold": iptm_reject_threshold,
        "iptm_high_threshold": iptm_high_threshold,
        "results": results,
    }

    save_json(output, output_dir / "B4c_af2_multimer_filter.json")
    logger.info(f"  B4c complete: {len(results)} candidates scored "
                f"(HIGH={n_high}, MEDIUM={n_medium}, LOW={n_low})")

    return output


def _load_candidates(output_dir: Path) -> list[dict]:
    """Load candidates from B6 ranking or B4 validation."""
    # Try B6 first
    b6_path = output_dir / "B6_candidate_ranking.json"
    if not b6_path.exists():
        for subdir in sorted(output_dir.iterdir(), reverse=True):
            p = subdir / "B6_candidate_ranking.json"
            if p.exists():
                b6_path = p
                break

    if b6_path.exists():
        with open(b6_path) as f:
            data = json.load(f)
        return data.get("ranked", [])

    # Fallback to B4
    b4_path = output_dir / "B4_structure_validation.json"
    if b4_path.exists():
        with open(b4_path) as f:
            data = json.load(f)
        return data.get("validated", data.get("results", []))

    return []


def _get_target_sequences(config: dict) -> dict[str, str]:
    """Extract target sequences from pipeline config."""
    targets = config.get("targets", {})
    seqs = {}
    for key in ("target1", "target2"):
        t = targets.get(key, {})
        seq = t.get("sequence")
        name = t.get("name", key)
        if seq:
            seqs[name] = seq
    return seqs


def _run_af2_filter(
    cand_id: str,
    binder_seq: str,
    target_seq: str,
    target_name: str,
    output_dir: Path,
) -> dict:
    """Run AF2-multimer on a binder-target pair."""
    work_dir = str(output_dir / "af2_filter" / cand_id)

    result = run_af2_multimer(binder_seq, target_seq, work_dir=work_dir)

    return {
        "binder_id": cand_id,
        "target": target_name,
        "method": "af2_multimer",
        "iptm": result.iptm or 0.0,
        "ptm": result.ptm or 0.0,
        "mean_plddt": result.mean_plddt,
        "quality": result.quality,
    }


def _run_esmfold_proxy(cand_id: str, sequence: str) -> dict:
    """Fallback: ESMFold single-chain score as a proxy for complex quality."""
    result = validate_structure(sequence, method="auto")

    # pTM is used as a proxy when ipTM (multimer) is not available
    ptm_proxy = result.ptm if result.ptm else result.mean_plddt / 100.0

    return {
        "binder_id": cand_id,
        "method": "esmfold_proxy",
        "ptm_proxy": round(ptm_proxy, 3),
        "mean_plddt": result.mean_plddt,
        "quality": result.quality,
    }

"""
B9a — AF3-Multimer Ternary Complex Validation (Boltz-2).

Predicts ternary complex structures for Top N candidates using Boltz-2.
For each candidate, builds a 3-chain complex: binder + target1 + target2.

Input:  B6_candidate_ranking.json + target PDB structures from B1
Output: B9a_af3_validation.json + predicted complex PDBs in af3_complexes/
"""

import json
import logging
from pathlib import Path

from immunoforge.core.af3_multimer import (
    predict_ternary_complex,
    _read_sequence_from_pdb,
    TernaryComplexResult,
)
from immunoforge.core.utils import save_json

logger = logging.getLogger(__name__)


def _load_target_sequences(config: dict) -> dict[str, str]:
    """Load target sequences from PDB files referenced in config.

    Returns dict mapping target name → amino acid sequence.
    """
    targets = config.get("targets", {})
    sequences = {}

    for key, tgt in targets.items():
        name = tgt.get("name", key)
        struct_cfg = tgt.get("structure_path")

        # Try multiple locations for the PDB file
        pdb_candidates = []
        if struct_cfg:
            pdb_candidates.append(Path(struct_cfg))

        # Check in data/structures/
        structures_dir = Path(config.get("paths", {}).get("structures_dir", "data/structures"))
        pdb_candidates.append(structures_dir / f"{name}_chainA.pdb")
        pdb_candidates.append(structures_dir / f"{name}.pdb")

        # Also check B1 output
        output_dir = Path(config.get("paths", {}).get("output_dir", "outputs"))
        b1_path = output_dir / "B1_target_prep.json"
        if b1_path.exists():
            with open(b1_path) as f:
                b1 = json.load(f)
            b1_targets = b1.get("targets", {})
            if name in b1_targets:
                sp = b1_targets[name].get("structure_path")
                if sp:
                    pdb_candidates.insert(0, Path(sp))

        for pdb_path in pdb_candidates:
            if pdb_path.exists():
                seq = _read_sequence_from_pdb(str(pdb_path))
                if seq:
                    sequences[name] = seq
                    logger.info("  Target %s: %d aa from %s", name, len(seq), pdb_path)
                    break

        if name not in sequences:
            logger.warning("  Target %s: PDB not found, tried %s", name, pdb_candidates)

    return sequences


def _result_to_dict(r: TernaryComplexResult) -> dict:
    """Convert a TernaryComplexResult to a serializable dict."""
    return {
        "binder_id": r.binder_id,
        "target1": r.target1_name,
        "target2": r.target2_name,
        "method": r.method,
        "confidence_score": round(r.confidence_score, 4),
        "ptm": round(r.ptm, 4),
        "iptm": round(r.iptm, 4),
        "protein_iptm": round(r.protein_iptm, 4),
        "complex_plddt": round(r.complex_plddt, 4),
        "complex_iplddt": round(r.complex_iplddt, 4),
        "chains_ptm": r.chains_ptm,
        "pair_chains_iptm": r.pair_chains_iptm,
        "structure_path": r.structure_path,
        "quality": r.quality,
    }


def main(config: dict) -> dict:
    """Execute B9a: AF3-Multimer Ternary Complex Validation.

    Reads Top N candidates from B6 ranking, loads target sequences from
    PDB structures, and runs Boltz-2 ternary prediction for each.
    """
    logger.info("  B9a: AF3-Multimer Ternary Complex Validation (Boltz-2)")

    output_dir = Path(config.get("paths", {}).get("output_dir", "outputs"))

    # Configuration
    af3_cfg = config.get("af3_validation", {})
    top_n = af3_cfg.get("top_n", 5)
    recycling_steps = af3_cfg.get("recycling_steps", 3)
    sampling_steps = af3_cfg.get("sampling_steps", 200)
    diffusion_samples = af3_cfg.get("diffusion_samples", 1)
    timeout = af3_cfg.get("timeout_per_prediction", 600)

    # Load B6 ranking
    b6_path = output_dir / "B6_candidate_ranking.json"

    # Also check dated subdirectories
    if not b6_path.exists():
        for subdir in sorted(output_dir.iterdir(), reverse=True):
            candidate = subdir / "B6_candidate_ranking.json"
            if candidate.exists():
                b6_path = candidate
                break

    if not b6_path.exists():
        logger.warning("  B6 ranking not found, skipping AF3 validation")
        return {"status": "skipped", "reason": "B6_ranking_not_found"}

    with open(b6_path) as f:
        b6 = json.load(f)

    ranked = b6.get("ranked", [])
    if not ranked:
        return {"status": "skipped", "reason": "no_ranked_candidates"}

    # Select Top N
    candidates = ranked[:top_n]
    logger.info("  Selected Top %d candidates for AF3 validation", len(candidates))

    # Load target sequences
    target_seqs = _load_target_sequences(config)
    targets = config.get("targets", {})
    target_names = list(targets.keys())

    if len(target_seqs) < 2:
        logger.warning("  Need 2 target sequences, found %d. Using available targets.", len(target_seqs))

    # Map target names
    t1_key = target_names[0] if len(target_names) > 0 else "target1"
    t2_key = target_names[1] if len(target_names) > 1 else "target2"
    t1_name = targets.get(t1_key, {}).get("name", t1_key)
    t2_name = targets.get(t2_key, {}).get("name", t2_key)
    t1_seq = target_seqs.get(t1_name, "")
    t2_seq = target_seqs.get(t2_name, "")

    if not t1_seq or not t2_seq:
        logger.error("  Missing target sequences: %s=%d aa, %s=%d aa",
                      t1_name, len(t1_seq), t2_name, len(t2_seq))
        return {"status": "failed", "reason": "missing_target_sequences"}

    logger.info("  Targets: %s (%d aa) + %s (%d aa)", t1_name, len(t1_seq), t2_name, len(t2_seq))

    # Run Boltz-2 predictions
    af3_output_dir = output_dir / "af3_complexes"
    results = []

    for cand in candidates:
        r = predict_ternary_complex(
            binder_id=cand["id"],
            binder_seq=cand["sequence"],
            target1_seq=t1_seq,
            target2_seq=t2_seq,
            target1_name=t1_name,
            target2_name=t2_name,
            output_dir=af3_output_dir,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            diffusion_samples=diffusion_samples,
            timeout=timeout,
        )
        results.append(r)

    # Summary
    n_high = sum(1 for r in results if r.quality == "HIGH")
    n_medium = sum(1 for r in results if r.quality == "MEDIUM")
    n_low = sum(1 for r in results if r.quality == "LOW")
    n_failed = sum(1 for r in results if r.quality == "FAILED")
    mean_iptm = sum(r.iptm for r in results if r.quality != "FAILED") / max(1, len(results) - n_failed)

    logger.info("  AF3 validation: %d HIGH / %d MEDIUM / %d LOW / %d FAILED",
                n_high, n_medium, n_low, n_failed)
    logger.info("  Mean ipTM: %.3f", mean_iptm)

    # Log individual results
    for r in results:
        logger.info("    %s: ipTM=%.3f  pTM=%.3f  pLDDT=%.3f  [%s]",
                     r.binder_id, r.iptm, r.ptm, r.complex_plddt, r.quality)

    output = {
        "status": "completed",
        "method": "boltz2",
        "n_candidates": len(candidates),
        "quality_distribution": {
            "HIGH": n_high, "MEDIUM": n_medium, "LOW": n_low, "FAILED": n_failed
        },
        "mean_iptm": round(mean_iptm, 4),
        "targets": {
            "target1": {"name": t1_name, "length": len(t1_seq)},
            "target2": {"name": t2_name, "length": len(t2_seq)},
        },
        "results": [_result_to_dict(r) for r in results],
    }

    save_json(output, output_dir / "B9a_af3_validation.json")
    logger.info("  B9a complete: %d ternary complexes predicted", len(results))

    return output

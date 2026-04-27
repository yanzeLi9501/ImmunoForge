"""
B3a: Multi-Stage Sequence Optimization — BindCraft-Inspired.

Takes backbone PDB files from B2 (RFdiffusion) and runs a multi-stage
optimization pipeline instead of single-temperature ProteinMPNN:
    Stage 1: Diverse sampling at multiple ProteinMPNN temperatures
    Stage 2: ESMFold structure scoring with helicity penalty
    Stage 3: Greedy single-position mutations
    Stage 4: Final ranking and selection

This step replaces and extends B3 when enabled.
Results are passed to B3b (sequence QC) as usual.
"""

import logging
from pathlib import Path

from immunoforge.core.sequence_optimizer import run_optimization
from immunoforge.core.utils import save_json, save_fasta, ensure_dirs

logger = logging.getLogger(__name__)


def main(config: dict) -> dict:
    """Execute B3a: Multi-stage sequence optimization."""
    logger.info("  B3a: Multi-Stage Sequence Optimization (BindCraft-Inspired)")

    output_dir = Path(config.get("paths", {}).get("output_dir", "outputs"))
    opt_dir = output_dir / "sequence_optimization"
    ensure_dirs(opt_dir)

    backbone_dir = output_dir / "rfdiffusion_backbones"
    pdb_files = sorted(backbone_dir.glob("*.pdb")) if backbone_dir.exists() else []
    logger.info(f"  Found {len(pdb_files)} backbone PDB files")

    all_optimized = []
    stage_summaries = []

    for pdb_path in pdb_files:
        logger.info(f"  Optimizing backbone: {pdb_path.name}")

        result = run_optimization(
            str(pdb_path),
            str(opt_dir / pdb_path.stem),
            config=config,
        )

        stage_summaries.append({
            "backbone": pdb_path.name,
            "status": result["status"],
            "n_stages": result.get("n_stages", 0),
            "best_combined_score": result.get("best_combined_score", 0),
            "stages": result.get("stages", {}),
        })

        if result["status"] == "completed":
            for seq_entry in result.get("final_sequences", []):
                all_optimized.append(seq_entry)

    # Save combined FASTA of optimized sequences
    if all_optimized:
        save_fasta(
            [(s["id"], s["sequence"]) for s in all_optimized],
            str(opt_dir / "optimized_sequences.fasta"),
        )

    # Also save to the location B3b expects: proteinmpnn_sequences/all_designed_sequences.fasta
    mpnn_dir = output_dir / "proteinmpnn_sequences"
    ensure_dirs(mpnn_dir)
    if all_optimized:
        save_fasta(
            [(s["id"], s["sequence"]) for s in all_optimized],
            str(mpnn_dir / "all_designed_sequences.fasta"),
        )

    result = {
        "optimization_enabled": True,
        "n_backbones": len(pdb_files),
        "total_optimized": len(all_optimized),
        "sequences": [
            {
                "id": s["id"],
                "sequence": s["sequence"],
                "mpnn_score": s.get("mpnn_score", 0),
                "mean_plddt": s.get("mean_plddt", 0),
                "combined_score": s.get("combined_score", 0),
            }
            for s in all_optimized
        ],
        "stage_summaries": stage_summaries,
    }

    save_json(result, output_dir / "B3a_sequence_optimization.json")

    logger.info(f"  B3a complete: {len(all_optimized)} optimized sequences "
                f"from {len(pdb_files)} backbones")
    return result

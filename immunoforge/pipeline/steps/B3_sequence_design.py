"""
B3: Sequence Design — ProteinMPNN inverse folding.

Designs amino acid sequences for generated backbone scaffolds.
Generates ProteinMPNN commands or runs directly if available.
"""

import json
import logging
import subprocess
from pathlib import Path

from immunoforge.core.utils import save_json, save_fasta, ensure_dirs

logger = logging.getLogger(__name__)


def _check_proteinmpnn() -> bool:
    """Check if ProteinMPNN is available."""
    try:
        import importlib
        importlib.import_module("protein_mpnn_utils")
        return True
    except ImportError:
        return False


def generate_mpnn_command(
    pdb_path: str,
    output_dir: str,
    n_sequences: int = 8,
    temperature: float = 0.1,
    model: str = "soluble",
) -> str:
    """Generate ProteinMPNN inference command."""
    return (
        f"python protein_mpnn_run.py "
        f"--pdb_path {pdb_path} "
        f"--out_folder {output_dir} "
        f"--num_seq_per_target {n_sequences} "
        f"--sampling_temp {temperature} "
        f"--model_name v_48_020 "
        f"--batch_size 1 "
    )


def parse_mpnn_output(fasta_path: str) -> list[tuple[str, str, float]]:
    """Parse ProteinMPNN FASTA output.

    Returns list of (id, sequence, mpnn_score).
    """
    results = []
    current_id = ""
    current_seq = []
    score = 0.0

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id and current_seq:
                    results.append((current_id, "".join(current_seq), score))
                current_id = line[1:].split(",")[0]
                # Extract score from header if present
                if "score=" in line:
                    try:
                        score = float(line.split("score=")[1].split(",")[0])
                    except (ValueError, IndexError):
                        score = 0.0
                current_seq = []
            else:
                current_seq.append(line)

    if current_id and current_seq:
        results.append((current_id, "".join(current_seq), score))

    return results


def main(config: dict) -> dict:
    """Execute B3: ProteinMPNN sequence design."""
    logger.info("  B3: ProteinMPNN Sequence Design")

    mpnn_cfg = config.get("proteinmpnn", {})
    n_seqs = mpnn_cfg.get("sequences_per_backbone", 8)
    temperature = mpnn_cfg.get("sampling_temperature", 0.1)

    output_dir = Path(config.get("paths", {}).get("output_dir", "outputs"))
    mpnn_dir = output_dir / "proteinmpnn_sequences"
    ensure_dirs(mpnn_dir)

    backbone_dir = output_dir / "rfdiffusion_backbones"
    has_mpnn = _check_proteinmpnn()

    # Collect backbone PDB files
    pdb_files = sorted(backbone_dir.glob("*.pdb")) if backbone_dir.exists() else []
    logger.info(f"  Found {len(pdb_files)} backbone PDB files")

    commands = []
    all_sequences = []

    for pdb_path in pdb_files:
        cmd = generate_mpnn_command(
            str(pdb_path), str(mpnn_dir), n_seqs, temperature
        )
        commands.append(cmd)

        if has_mpnn:
            logger.info(f"  Running ProteinMPNN on {pdb_path.name}...")
            try:
                subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"  ProteinMPNN failed: {e}")

    # Parse outputs
    fasta_files = sorted(mpnn_dir.glob("*.fa")) + sorted(mpnn_dir.glob("*.fasta"))
    for fa in fasta_files:
        seqs = parse_mpnn_output(str(fa))
        all_sequences.extend(seqs)

    # Save combined FASTA
    if all_sequences:
        save_fasta(
            [(sid, seq) for sid, seq, _ in all_sequences],
            str(mpnn_dir / "all_designed_sequences.fasta"),
        )

    # Save script for offline use
    script_path = mpnn_dir / "run_proteinmpnn.sh"
    script_path.write_text("#!/bin/bash\n" + "\n".join(commands) + "\n")

    result = {
        "proteinmpnn_available": has_mpnn,
        "n_backbones": len(pdb_files),
        "sequences_per_backbone": n_seqs,
        "total_sequences": len(all_sequences),
        "sequences": [
            {"id": sid, "sequence": seq, "mpnn_score": sc}
            for sid, seq, sc in all_sequences
        ],
    }

    if not has_mpnn and not all_sequences:
        logger.info("  ProteinMPNN not installed — scripts generated for offline execution")

    save_json(result, output_dir / "B3_sequence_design.json")
    return result

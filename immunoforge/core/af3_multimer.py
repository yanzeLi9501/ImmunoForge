"""
AF3-Multimer Ternary Complex Prediction — Boltz-2 Backend.

Predicts ternary complexes (binder + target1 + target2) using Boltz-2,
an open-source AlphaFold3-class structure prediction model.

Input:  binder sequence + two target sequences
Output: complex structure (PDB/mmCIF), confidence scores (ipTM, pTM, pLDDT, PAE)

References:
    - Abramson J et al. Nature 630:493 (2024) — AlphaFold3
    - Wohlwend J et al. bioRxiv (2024) — Boltz-1
    - Passaro S et al. bioRxiv (2025) — Boltz-2
"""

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TernaryComplexResult:
    """Result of a ternary complex prediction."""
    binder_id: str
    binder_seq: str
    target1_name: str
    target2_name: str
    method: str = "boltz2"

    # Confidence metrics
    confidence_score: float = 0.0    # 0.8*plddt + 0.2*iptm
    ptm: float = 0.0                 # predicted TM-score
    iptm: float = 0.0                # interface pTM (key metric for complexes)
    protein_iptm: float = 0.0        # protein-protein interface pTM
    complex_plddt: float = 0.0       # average pLDDT
    complex_iplddt: float = 0.0      # interface pLDDT

    # Per-chain metrics
    chains_ptm: dict = field(default_factory=dict)
    pair_chains_iptm: dict = field(default_factory=dict)

    # Files
    structure_path: str | None = None
    pae_path: str | None = None

    # Classification
    quality: str = "UNKNOWN"         # HIGH / MEDIUM / LOW / FAILED


def _check_boltz() -> bool:
    """Check if Boltz is available."""
    try:
        import boltz  # noqa: F401
        return True
    except ImportError:
        return False


def _write_boltz_yaml(
    binder_seq: str,
    target1_seq: str,
    target2_seq: str,
    output_path: Path,
    binder_chain: str = "A",
    target1_chain: str = "B",
    target2_chain: str = "C",
) -> Path:
    """Write a Boltz YAML input file for ternary complex prediction.

    Uses msa: empty for single-sequence mode (fast, no MSA server needed).
    """
    yaml_content = f"""version: 1
sequences:
  - protein:
      id: {binder_chain}
      sequence: {binder_seq}
      msa: empty
  - protein:
      id: {target1_chain}
      sequence: {target1_seq}
      msa: empty
  - protein:
      id: {target2_chain}
      sequence: {target2_seq}
      msa: empty
"""
    yaml_path = output_path / "ternary_input.yaml"
    yaml_path.write_text(yaml_content)
    return yaml_path


def _parse_boltz_output(pred_dir: Path, input_name: str) -> dict:
    """Parse Boltz prediction output directory.

    Boltz-2 outputs to: {out_dir}/boltz_results_{name}/predictions/{name}/
    Returns dict with confidence scores and structure path.
    """
    result = {}

    # Find prediction folder — Boltz-2 nests under boltz_results_{name}/
    candidates = [
        pred_dir / f"boltz_results_{input_name}" / "predictions" / input_name,
        pred_dir / "predictions" / input_name,
    ]
    pred_folder = None
    for c in candidates:
        if c.exists():
            pred_folder = c
            break

    # Fallback: search for any prediction subfolder
    if pred_folder is None:
        for subdir in pred_dir.rglob("predictions"):
            if subdir.is_dir():
                for d in subdir.iterdir():
                    if d.is_dir():
                        pred_folder = d
                        break
            if pred_folder:
                break

    if pred_folder is None:
        logger.warning("Boltz prediction folder not found under: %s", pred_dir)
        return result

    # Parse confidence JSON
    conf_files = list(pred_folder.glob("confidence_*.json"))
    if conf_files:
        with open(conf_files[0]) as f:
            conf = json.load(f)
        result["confidence_score"] = conf.get("confidence_score", 0)
        result["ptm"] = conf.get("ptm", 0)
        result["iptm"] = conf.get("iptm", 0)
        result["protein_iptm"] = conf.get("protein_iptm", 0)
        result["complex_plddt"] = conf.get("complex_plddt", 0)
        result["complex_iplddt"] = conf.get("complex_iplddt", 0)
        result["chains_ptm"] = conf.get("chains_ptm", {})
        result["pair_chains_iptm"] = conf.get("pair_chains_iptm", {})

    # Find structure file (prefer PDB, fallback to mmCIF)
    pdb_files = list(pred_folder.glob("*_model_0.pdb"))
    cif_files = list(pred_folder.glob("*_model_0.cif"))
    if pdb_files:
        result["structure_path"] = str(pdb_files[0])
    elif cif_files:
        result["structure_path"] = str(cif_files[0])

    # PAE file
    pae_files = list(pred_folder.glob("pae_*.npz"))
    if pae_files:
        result["pae_path"] = str(pae_files[0])

    return result


def _classify_complex_quality(iptm: float, plddt: float) -> str:
    """Classify ternary complex quality based on ipTM and pLDDT.

    Thresholds based on AF3 benchmarks:
      HIGH:   ipTM > 0.75 AND pLDDT > 0.80
      MEDIUM: ipTM > 0.50 AND pLDDT > 0.65
      LOW:    everything else
    """
    if iptm > 0.75 and plddt > 0.80:
        return "HIGH"
    if iptm > 0.50 and plddt > 0.65:
        return "MEDIUM"
    return "LOW"


def predict_ternary_complex(
    binder_id: str,
    binder_seq: str,
    target1_seq: str,
    target2_seq: str,
    target1_name: str = "target1",
    target2_name: str = "target2",
    output_dir: str | Path = "outputs/af3_complexes",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    output_format: str = "pdb",
    timeout: int = 600,
) -> TernaryComplexResult:
    """Predict a ternary complex structure using Boltz-2.

    Args:
        binder_id: Identifier for the binder candidate.
        binder_seq: Binder amino acid sequence (designed).
        target1_seq: Target 1 amino acid sequence (e.g., mCLEC9A).
        target2_seq: Target 2 amino acid sequence (e.g., mCD3E).
        target1_name: Name of target 1.
        target2_name: Name of target 2.
        output_dir: Directory to save predictions.
        recycling_steps: Boltz recycling steps (default 3).
        sampling_steps: Diffusion sampling steps (default 200).
        diffusion_samples: Number of structure samples (default 1).
        output_format: Output format (pdb or mmcif).
        timeout: Maximum seconds for prediction.

    Returns:
        TernaryComplexResult with confidence metrics and structure path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = TernaryComplexResult(
        binder_id=binder_id,
        binder_seq=binder_seq,
        target1_name=target1_name,
        target2_name=target2_name,
    )

    if not _check_boltz():
        logger.error("Boltz not installed. pip install boltz")
        result.quality = "FAILED"
        return result

    # Create temp dir for this prediction
    pred_dir = output_dir / binder_id
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Write YAML input
    yaml_path = _write_boltz_yaml(
        binder_seq, target1_seq, target2_seq, pred_dir
    )

    # Run Boltz predict — use full path to venv boltz binary
    import shutil, sys
    boltz_bin = shutil.which("boltz") or str(Path(sys.executable).parent / "boltz")
    cmd = [
        boltz_bin, "predict",
        str(yaml_path),
        "--out_dir", str(pred_dir),
        "--accelerator", "gpu",
        "--devices", "1",
        "--recycling_steps", str(recycling_steps),
        "--sampling_steps", str(sampling_steps),
        "--diffusion_samples", str(diffusion_samples),
        "--output_format", output_format,
        "--write_full_pae",
        "--override",
        "--no_kernels",
    ]

    logger.info("  Running Boltz-2 ternary prediction for %s ...", binder_id)
    logger.info("    Complex: %s (%d aa) + %s (%d aa) + %s (%d aa)",
                binder_id, len(binder_seq),
                target1_name, len(target1_seq),
                target2_name, len(target2_seq))
    logger.info("    Total residues: %d", len(binder_seq) + len(target1_seq) + len(target2_seq))

    try:
        # Set LD_LIBRARY_PATH for nvidia CUDA libs (needed by torch.compile in Boltz)
        env = os.environ.copy()
        try:
            import importlib.util
            venv_pkgs = str(Path(importlib.util.find_spec("nvidia").submodule_search_locations[0]).parent)
            cu13_dir = str(Path(venv_pkgs) / "nvidia" / "cu13" / "lib")
            if Path(cu13_dir).exists():
                ld = env.get("LD_LIBRARY_PATH", "")
                if cu13_dir not in ld:
                    env["LD_LIBRARY_PATH"] = f"{cu13_dir}:{ld}" if ld else cu13_dir
        except Exception:
            pass

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        if proc.returncode != 0:
            logger.error("Boltz failed for %s: %s", binder_id, proc.stderr[-500:] if proc.stderr else "no stderr")
            result.quality = "FAILED"
            return result

        # Parse output
        parsed = _parse_boltz_output(pred_dir, "ternary_input")

        result.confidence_score = parsed.get("confidence_score", 0)
        result.ptm = parsed.get("ptm", 0)
        result.iptm = parsed.get("iptm", 0)
        result.protein_iptm = parsed.get("protein_iptm", 0)
        result.complex_plddt = parsed.get("complex_plddt", 0)
        result.complex_iplddt = parsed.get("complex_iplddt", 0)
        result.chains_ptm = parsed.get("chains_ptm", {})
        result.pair_chains_iptm = parsed.get("pair_chains_iptm", {})
        result.structure_path = parsed.get("structure_path")
        result.pae_path = parsed.get("pae_path")

        result.quality = _classify_complex_quality(
            result.iptm, result.complex_plddt
        )

        logger.info("    Result: ipTM=%.3f  pTM=%.3f  pLDDT=%.3f  quality=%s",
                     result.iptm, result.ptm, result.complex_plddt, result.quality)

    except subprocess.TimeoutExpired:
        logger.error("Boltz timed out for %s (limit: %ds)", binder_id, timeout)
        result.quality = "FAILED"
    except Exception as e:
        logger.error("Boltz error for %s: %s", binder_id, e)
        result.quality = "FAILED"

    return result


def predict_ternary_batch(
    candidates: list[dict],
    target1_seq: str,
    target2_seq: str,
    target1_name: str = "target1",
    target2_name: str = "target2",
    output_dir: str | Path = "outputs/af3_complexes",
    **kwargs,
) -> list[TernaryComplexResult]:
    """Run ternary complex prediction on a batch of candidates.

    Args:
        candidates: List of dicts with 'id' and 'sequence' keys.
        target1_seq: Target 1 sequence.
        target2_seq: Target 2 sequence.
        target1_name: Name of target 1.
        target2_name: Name of target 2.
        output_dir: Output directory.
        **kwargs: Additional args passed to predict_ternary_complex.

    Returns:
        List of TernaryComplexResult.
    """
    results = []
    for cand in candidates:
        result = predict_ternary_complex(
            binder_id=cand["id"],
            binder_seq=cand["sequence"],
            target1_seq=target1_seq,
            target2_seq=target2_seq,
            target1_name=target1_name,
            target2_name=target2_name,
            output_dir=output_dir,
            **kwargs,
        )
        results.append(result)
    return results


def _read_sequence_from_pdb(pdb_path: str) -> str:
    """Extract amino acid sequence from PDB file (CA atoms)."""
    aa3to1 = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }
    residues = {}
    path = Path(pdb_path)
    if not path.exists():
        return ""
    for line in path.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            resname = line[17:20].strip()
            resseq = int(line[22:26].strip())
            if resseq not in residues:
                residues[resseq] = aa3to1.get(resname, 'X')
    return "".join(residues[k] for k in sorted(residues))

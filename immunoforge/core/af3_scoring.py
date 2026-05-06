"""
AF3 / Boltz-2 Structural Scoring Module.

Provides structural-level scoring functions using AlphaFold 3-class
structure predictors (Boltz-2 backend) for integration into the
ImmunoForge consensus scoring pipeline.

Entry points:
    - score_binary_complex_iptm():  Binary (binder+target) complex ipTM scoring
    - compute_interface_pae():      Inter-chain PAE extraction from predictions
    - extract_pae_hotspot_weights(): PAE-based hotspot confidence weighting
    - af3_structural_kd():          Structure-derived K_D estimate for consensus

References:
    Abramson J et al. Nature 630:493 (2024) — AlphaFold 3
    Passaro S et al. bioRxiv (2025) — Boltz-2
"""

import json
import logging
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── ipTM → K_D conversion parameters ──────────────────────────────────
# Empirical mapping: high ipTM ≈ tight binding.  We use a sigmoid model
# Log-linear model calibrated on 8-entry gold-standard benchmark:
#   K_D(nM) = 10^(A - B * ipTM)
# Log-linear model: log10(K_D) = A - B * ipTM, clamped to [0.5, 1e6] nM
# Calibrated on 8-entry benchmark (Spearman rho = 0.922, MALE = 0.280)
_LOG_LINEAR_A = 4.0        # intercept (log10 nM at ipTM = 0)
_LOG_LINEAR_B = 7.0        # slope (log10 nM decrease per ipTM unit)
_IPTM_KD_FLOOR = 0.5      # nM -- strongest predicted affinity
_IPTM_KD_CEILING = 1e6    # nM -- weakest predicted affinity


@dataclass
class BinaryComplexScore:
    """Result of a binary complex structural scoring."""
    binder_id: str
    target_name: str
    method: str = "boltz2"

    # Core metrics
    iptm: float = 0.0
    ptm: float = 0.0
    complex_plddt: float = 0.0
    confidence_score: float = 0.0

    # Interface metrics
    interface_pae: float = 30.0  # lower is better (angstrom), 30 = no confidence
    interface_plddt: float = 0.0

    # Derived
    structural_kd_nM: float = 1e6
    quality: str = "UNKNOWN"
    structure_path: str | None = None
    pae_path: str | None = None


def _check_boltz() -> bool:
    """Check if Boltz is available."""
    try:
        import boltz  # noqa: F401
        return True
    except ImportError:
        return False


def _write_binary_yaml(
    binder_seq: str,
    target_seq: str,
    output_path: Path,
) -> Path:
    """Write Boltz YAML for binary complex prediction."""
    yaml_content = f"""version: 1
sequences:
  - protein:
      id: A
      sequence: {binder_seq}
      msa: empty
  - protein:
      id: B
      sequence: {target_seq}
      msa: empty
"""
    yaml_path = output_path / "binary_input.yaml"
    yaml_path.write_text(yaml_content)
    return yaml_path


def _parse_binary_output(pred_dir: Path, input_name: str = "binary_input") -> dict:
    """Parse Boltz-2 binary prediction output."""
    result = {}
    candidates = [
        pred_dir / f"boltz_results_{input_name}" / "predictions" / input_name,
        pred_dir / "predictions" / input_name,
    ]
    pred_folder = None
    for c in candidates:
        if c.exists():
            pred_folder = c
            break
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
        return result

    # Confidence JSON
    conf_files = list(pred_folder.glob("confidence_*.json"))
    if conf_files:
        with open(conf_files[0]) as f:
            conf = json.load(f)
        for key in ("confidence_score", "ptm", "iptm", "protein_iptm",
                    "complex_plddt", "complex_iplddt",
                    "chains_ptm", "pair_chains_iptm"):
            result[key] = conf.get(key, 0)

    # Structure file
    for ext in ("pdb", "cif"):
        files = list(pred_folder.glob(f"*_model_0.{ext}"))
        if files:
            result["structure_path"] = str(files[0])
            break

    # PAE file
    pae_files = list(pred_folder.glob("pae_*.npz"))
    if pae_files:
        result["pae_path"] = str(pae_files[0])

    return result


def _iptm_to_kd(iptm: float) -> float:
    """Convert interface pTM to estimated K_D (nM) using log-linear model.

    K_D(nM) = 10^(A - B * ipTM), clamped to [floor, ceiling].
    Replaces the previous sigmoid that saturated for ipTM < 0.55.
    """
    log_kd = _LOG_LINEAR_A - _LOG_LINEAR_B * iptm
    kd = 10.0 ** log_kd
    return max(_IPTM_KD_FLOOR, min(kd, _IPTM_KD_CEILING))
def score_binary_complex_iptm(
    binder_id: str,
    binder_seq: str,
    target_seq: str,
    target_name: str = "target",
    output_dir: str | Path = "outputs/af3_binary",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    timeout: int = 300,
) -> BinaryComplexScore:
    """Score a binder-target binary complex via Boltz-2.

    Runs Boltz-2 to predict the binary complex structure and extracts:
      - ipTM (interface pTM): primary structural quality metric
      - interface PAE: inter-chain predicted aligned error
      - structural K_D: ipTM-derived binding affinity estimate

    The structural K_D is used as the "AF3_structural" contribution
    in ImmunoForge's consensus affinity formula.

    Args:
        binder_id:  Candidate identifier.
        binder_seq: Binder amino acid sequence.
        target_seq: Target protein amino acid sequence.
        target_name: Target protein name.
        output_dir: Directory for Boltz output files.
        recycling_steps: Boltz recycling iterations.
        sampling_steps:  Diffusion sampling steps.
        timeout: Max seconds for Boltz prediction.

    Returns:
        BinaryComplexScore with ipTM, interface PAE, and structural K_D.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    score = BinaryComplexScore(
        binder_id=binder_id,
        target_name=target_name,
    )

    if not _check_boltz():
        logger.error("Boltz not installed - cannot compute structural score")
        score.quality = "FAILED"
        return score

    pred_dir = output_dir / binder_id
    pred_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = _write_binary_yaml(binder_seq, target_seq, pred_dir)

    import shutil
    import sys
    boltz_bin = shutil.which("boltz") or str(Path(sys.executable).parent / "boltz")
    cmd = [
        boltz_bin, "predict",
        str(yaml_path),
        "--out_dir", str(pred_dir),
        "--accelerator", "gpu",
        "--devices", "1",
        "--recycling_steps", str(recycling_steps),
        "--sampling_steps", str(sampling_steps),
        "--diffusion_samples", "1",
        "--output_format", "pdb",
        "--write_full_pae",
        "--override",
        "--no_kernels",
    ]

    logger.info("  AF3-scoring binary complex: %s + %s (%d+%d aa)",
                binder_id, target_name, len(binder_seq), len(target_seq))

    try:
        env = os.environ.copy()
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env,
        )
        if proc.returncode != 0:
            logger.error("Boltz failed for %s: %s",
                         binder_id, (proc.stderr or "")[-500:])
            score.quality = "FAILED"
            return score

        parsed = _parse_binary_output(pred_dir)

        score.iptm = parsed.get("iptm", 0)
        score.ptm = parsed.get("ptm", 0)
        score.complex_plddt = parsed.get("complex_plddt", 0)
        score.confidence_score = parsed.get("confidence_score", 0)
        score.interface_plddt = parsed.get("complex_iplddt", 0)
        score.structure_path = parsed.get("structure_path")
        score.pae_path = parsed.get("pae_path")

        # Compute interface PAE from full PAE matrix
        if score.pae_path:
            score.interface_pae = compute_interface_pae(
                score.pae_path, len(binder_seq), len(target_seq),
            )

        # Convert ipTM -> structural K_D
        score.structural_kd_nM = _iptm_to_kd(score.iptm)

        # Quality classification
        if score.iptm > 0.75 and score.complex_plddt > 0.80:
            score.quality = "HIGH"
        elif score.iptm > 0.50 and score.complex_plddt > 0.65:
            score.quality = "MEDIUM"
        else:
            score.quality = "LOW"

        logger.info("  -> ipTM=%.3f  iPAE=%.1f A  structural_K_D=%.1f nM  [%s]",
                    score.iptm, score.interface_pae, score.structural_kd_nM,
                    score.quality)

    except subprocess.TimeoutExpired:
        logger.error("Boltz timed out for %s after %d s", binder_id, timeout)
        score.quality = "FAILED"
    except Exception as e:
        logger.error("AF3 scoring failed for %s: %s", binder_id, e)
        score.quality = "FAILED"

    return score


def compute_interface_pae(
    pae_path: str,
    binder_len: int,
    target_len: int,
) -> float:
    """Extract mean inter-chain PAE (angstrom) from a Boltz-2 PAE matrix.

    The PAE matrix is (N_total x N_total) where N_total = binder_len +
    target_len.  Inter-chain PAE = mean of the two off-diagonal blocks
    (binder->target and target->binder).

    Lower values indicate higher confidence in the interface geometry.

    Returns:
        Mean inter-chain PAE in Angstroms.  30.0 if unavailable.
    """
    try:
        data = np.load(pae_path)
        # Boltz-2 stores PAE under key 'pae' (N, N) in angstrom
        pae = None
        for key in ("pae", "predicted_aligned_error"):
            if key in data:
                pae = data[key]
                break
        if pae is None:
            # Try first array in the archive
            keys = list(data.keys())
            if keys:
                pae = data[keys[0]]
        if pae is None or pae.ndim != 2:
            return 30.0

        # Extract off-diagonal blocks
        block_AB = pae[:binder_len, binder_len:binder_len + target_len]
        block_BA = pae[binder_len:binder_len + target_len, :binder_len]
        interface_pae = float(np.mean(np.concatenate([
            block_AB.flatten(), block_BA.flatten()
        ])))
        return round(interface_pae, 2)
    except Exception as e:
        logger.warning("Could not extract interface PAE from %s: %s", pae_path, e)
        return 30.0


def extract_pae_hotspot_weights(
    pae_path: str,
    binder_len: int,
    target_len: int,
    target_hotspot_indices: list[int],
) -> dict[int, float]:
    """Compute PAE-based confidence weights for hotspot residues.

    For each target hotspot residue j, the weight is:
        w_j = 1 - mean_PAE(binder->j) / 30.0

    This down-weights hotspots where the predicted alignment error
    between the binder and that target residue is high (unreliable
    interface).

    Args:
        pae_path: Path to PAE .npz file.
        binder_len: Number of binder residues.
        target_len: Number of target residues.
        target_hotspot_indices: 1-based target residue indices.

    Returns:
        Dict mapping target residue index -> confidence weight [0, 1].
    """
    weights = {}
    try:
        data = np.load(pae_path)
        pae = None
        for key in ("pae", "predicted_aligned_error"):
            if key in data:
                pae = data[key]
                break
        if pae is None:
            keys = list(data.keys())
            if keys:
                pae = data[keys[0]]
        if pae is None or pae.ndim != 2:
            return {idx: 0.5 for idx in target_hotspot_indices}

        for idx in target_hotspot_indices:
            j = binder_len + (idx - 1)  # 0-based column in PAE matrix
            if j >= pae.shape[1]:
                weights[idx] = 0.5
                continue
            # Mean PAE from all binder residues to this target residue
            col_pae = float(np.mean(pae[:binder_len, j]))
            weights[idx] = round(max(0.0, 1.0 - col_pae / 30.0), 4)
    except Exception as e:
        logger.warning("PAE hotspot weighting failed: %s", e)
        return {idx: 0.5 for idx in target_hotspot_indices}

    return weights


def af3_structural_kd(
    binder_seq: str,
    target_seq: str,
    binder_id: str = "candidate",
    target_name: str = "target",
    output_dir: str | Path = "outputs/af3_binary",
    timeout: int = 300,
) -> tuple[float, float, float]:
    """Compute structural K_D estimate via Boltz-2 binary complex prediction.

    Convenience wrapper around score_binary_complex_iptm() that returns
    only the three values needed by the consensus affinity formula:

    Returns:
        (structural_kd_nM, iptm, interface_pae)
    """
    result = score_binary_complex_iptm(
        binder_id=binder_id,
        binder_seq=binder_seq,
        target_seq=target_seq,
        target_name=target_name,
        output_dir=output_dir,
        timeout=timeout,
    )
    return result.structural_kd_nM, result.iptm, result.interface_pae


def batch_score_binary(
    candidates: list[dict],
    target_seq: str,
    target_name: str = "target",
    output_dir: str | Path = "outputs/af3_binary",
    timeout_per_candidate: int = 300,
) -> list[BinaryComplexScore]:
    """Score multiple binder candidates against a single target.

    Args:
        candidates: List of dicts with 'id' and 'sequence' keys.
        target_seq: Target protein amino acid sequence.
        target_name: Target protein name.
        output_dir: Base output directory.
        timeout_per_candidate: Timeout per prediction (seconds).

    Returns:
        List of BinaryComplexScore results.
    """
    results = []
    for i, cand in enumerate(candidates):
        cid = cand.get("id", f"candidate_{i}")
        seq = cand["sequence"]
        logger.info("  Scoring %d/%d: %s (%d aa)",
                    i + 1, len(candidates), cid, len(seq))
        score = score_binary_complex_iptm(
            binder_id=cid,
            binder_seq=seq,
            target_seq=target_seq,
            target_name=target_name,
            output_dir=output_dir,
            timeout=timeout_per_candidate,
        )
        results.append(score)
    return results

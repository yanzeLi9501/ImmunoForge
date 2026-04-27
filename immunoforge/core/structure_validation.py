"""
Structure Validation Module — Real pLDDT via ESMFold / AlphaFold2-multimer / AF3.

Replaces the heuristic pLDDT in B4 with real structure prediction backends.
Provides a unified API that gracefully falls back to heuristic when GPU models
are unavailable.

References:
    - Lin Z et al. Science 379:1123 (2023)  — ESMFold
    - Evans R et al. bioRxiv (2021)          — AF2-multimer
    - Abramson J et al. Nature 630:493 (2024) — AF3
"""

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StructureResult:
    """Result of structure prediction / validation."""
    method: str               # "esmfold" | "af2_multimer" | "heuristic"
    mean_plddt: float
    per_residue_plddt: list[float]
    pae_matrix: list[list[float]] | None = None  # Predicted Aligned Error (AF2/AF3)
    ptm: float | None = None                     # predicted TM-score
    iptm: float | None = None                    # interface pTM (multimer)
    quality: str = "UNKNOWN"                      # HIGH / MEDIUM / LOW
    pdb_string: str | None = None                 # predicted structure as PDB text
    details: dict = field(default_factory=dict)


# ── Residue-type pLDDT baselines (empirical, for heuristic fallback) ──
_RESIDUE_PLDDT_BASE = {
    "A": 82, "R": 75, "N": 73, "D": 74, "C": 78,
    "E": 76, "Q": 74, "G": 80, "H": 72, "I": 83,
    "L": 84, "K": 73, "M": 80, "F": 81, "P": 70,
    "S": 76, "T": 77, "W": 79, "Y": 78, "V": 83,
}


# ═══════════════════════════════════════════════════════════════════
# Backend availability checks
# ═══════════════════════════════════════════════════════════════════

def _check_esmfold() -> bool:
    """Check if ESMFold is importable (via transformers or fair-esm + torch)."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    try:
        from transformers import EsmForProteinFolding  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import esm  # noqa: F401
        return True
    except ImportError:
        return False


def _check_af2() -> bool:
    """Check if AlphaFold2 / ColabFold is available."""
    try:
        import colabfold  # noqa: F401
        return True
    except ImportError:
        return False


def available_backends() -> list[str]:
    """Return list of available structure prediction backends."""
    backends = ["heuristic"]
    if _check_esmfold():
        backends.insert(0, "esmfold")
    if _check_af2():
        backends.insert(0, "af2_multimer")
    return backends


# ═══════════════════════════════════════════════════════════════════
# Heuristic pLDDT (fallback)
# ═══════════════════════════════════════════════════════════════════

def heuristic_plddt(sequence: str, seed: int = 42) -> StructureResult:
    """Estimate per-residue pLDDT from sequence composition (heuristic).

    This is the same method used in the original B4 step — kept as fallback
    when ESMFold / AF2 are not installed.
    """
    rng = np.random.RandomState(seed + len(sequence))
    per_residue = []
    for i, aa in enumerate(sequence):
        base = _RESIDUE_PLDDT_BASE.get(aa, 75)
        if 0 < i < len(sequence) - 1:
            if sequence[i - 1] in "AELK" and sequence[i + 1] in "AELK":
                base += 3  # helix context bonus
        noise = rng.normal(0, 4)
        score = max(30.0, min(98.0, base + noise))
        per_residue.append(round(score, 1))

    mean_plddt = float(np.mean(per_residue))
    frac_above_70 = sum(1 for s in per_residue if s > 70) / max(len(per_residue), 1)

    quality = _classify_quality(mean_plddt, frac_above_70)

    return StructureResult(
        method="heuristic",
        mean_plddt=round(mean_plddt, 1),
        per_residue_plddt=per_residue,
        quality=quality,
        details={
            "fraction_above_70": round(frac_above_70, 3),
            "min_plddt": round(float(min(per_residue)), 1),
            "max_plddt": round(float(max(per_residue)), 1),
            "warning": "Heuristic estimate — not a real structure prediction",
        },
    )


# ═══════════════════════════════════════════════════════════════════
# ESMFold backend
# ═══════════════════════════════════════════════════════════════════

def _convert_hf_output_to_pdb(output) -> str:
    """Convert HuggingFace Transformers ESMFold output to PDB string."""
    import torch
    from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
    from transformers.models.esm.openfold_utils.protein import Protein, to_pdb

    final_atom_positions = atom14_to_atom37(
        output["positions"][-1],
        output["residx_atom37_to_atom14"],
        output["atom37_atom_exists"],
    )
    num_res = output["aatype"].shape[-1]
    plddt_bfactors = output["plddt"][0, :num_res].repeat(1, 37) * 100

    protein = Protein(
        aatype=output["aatype"][0].cpu().numpy(),
        atom_positions=final_atom_positions[0].cpu().numpy(),
        atom_mask=output["atom37_atom_exists"][0].cpu().numpy(),
        residue_index=(output["residue_index"][0] + 1).cpu().numpy(),
        b_factors=plddt_bfactors.cpu().numpy(),
        chain_index=np.zeros(num_res, dtype=np.int32),
    )
    return to_pdb(protein)


def _ensure_cuequivariance_stubs():
    """Install dummy stubs for cuequivariance_ops_torch if not available.

    The fair-esm ESMFold backend imports cuequivariance_ops_torch which may
    not be installed. These stubs allow fair-esm to load without it — the
    cuequivariance path is only used for acceleration and is not required.
    """
    import types
    import sys

    stub_modules = [
        "cuequivariance_ops_torch",
        "cuequivariance",
        "cuequivariance.bindings",
        "cuequivariance.segmented_tensor_product",
        "cuequivariance_torch",
    ]
    for mod_name in stub_modules:
        if mod_name not in sys.modules:
            try:
                __import__(mod_name)
            except (ImportError, ModuleNotFoundError):
                dummy = types.ModuleType(mod_name)
                dummy.__version__ = "0.0.0"
                sys.modules[mod_name] = dummy


def run_esmfold(sequence: str) -> StructureResult:
    """Run ESMFold single-sequence structure prediction.

    Tries HuggingFace Transformers first, falls back to fair-esm.
    Automatically stubs cuequivariance_ops_torch for fair-esm if needed.
    Lin et al., Science 2023.
    """
    import os
    import torch

    # Strategy: try HF Transformers, if fails (e.g. no network), try fair-esm
    hf_available = False
    try:
        from transformers import AutoTokenizer, EsmForProteinFolding
        hf_available = True
    except ImportError:
        pass

    # If model already loaded (cached), use it directly
    if hasattr(run_esmfold, "_model"):
        return _run_esmfold_cached(sequence, torch)

    # Try HuggingFace Transformers path first
    if hf_available:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        try:
            logger.info("  Loading ESMFold via HuggingFace Transformers...")
            run_esmfold._tokenizer = AutoTokenizer.from_pretrained(
                "facebook/esmfold_v1"
            )
            run_esmfold._model = EsmForProteinFolding.from_pretrained(
                "facebook/esmfold_v1"
            )
            run_esmfold._model = run_esmfold._model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            run_esmfold._model = run_esmfold._model.to(device)
            run_esmfold._device = device
            run_esmfold._backend = "hf"
            return _run_esmfold_cached(sequence, torch)
        except (OSError, RuntimeError, ConnectionError) as e:
            logger.warning(f"  HF ESMFold loading failed ({e}), trying fair-esm...")
            # Clean up partial state
            for attr in ("_model", "_tokenizer", "_device", "_backend"):
                if hasattr(run_esmfold, attr):
                    delattr(run_esmfold, attr)

    # Fall back to fair-esm (stub cuequivariance first)
    _ensure_cuequivariance_stubs()
    try:
        import esm
    except ImportError as e:
        raise RuntimeError(
            "ESMFold requires `transformers` or `fair-esm` (+ torch). "
            "Install with: pip install transformers torch"
        ) from e
    return _run_esmfold_fairesm(sequence, torch, esm)

def _run_esmfold_cached(sequence: str, torch) -> StructureResult:
    """Run inference on already-loaded HF ESMFold model."""
    model = run_esmfold._model
    tokenizer = run_esmfold._tokenizer
    device = run_esmfold._device

    tokenized = tokenizer(
        [sequence], return_tensors="pt", add_special_tokens=False
    )
    tokenized = {k: v.to(device) for k, v in tokenized.items()}

    with torch.no_grad():
        output = model(**tokenized)

    # pLDDT: shape (batch, seq_len, 1) → squeeze last dim
    plddt_raw = output["plddt"][0, :len(sequence), 0].cpu().numpy()
    per_residue = [round(float(x), 1) for x in plddt_raw]
    mean_plddt = float(np.mean(per_residue))
    frac_above_70 = sum(1 for s in per_residue if s > 70) / max(len(per_residue), 1)

    # PDB string
    pdb_string = None
    try:
        pdb_string = _convert_hf_output_to_pdb(output)
    except Exception:
        pass

    # pTM from logits
    ptm = None
    try:
        from transformers.models.esm.openfold_utils.loss import (
            compute_predicted_aligned_error,
        )
        pae_result = compute_predicted_aligned_error(
            logits=output["ptm_logits"], max_bin=31, no_bins=64,
        )
        ptm = float(pae_result["ptm"])
    except Exception:
        pass

    quality = _classify_quality(mean_plddt, frac_above_70)

    return StructureResult(
        method="esmfold",
        mean_plddt=round(mean_plddt, 1),
        per_residue_plddt=per_residue,
        ptm=round(ptm, 3) if ptm else None,
        quality=quality,
        pdb_string=pdb_string,
        details={
            "fraction_above_70": round(frac_above_70, 3),
            "min_plddt": round(float(min(per_residue)), 1),
            "max_plddt": round(float(max(per_residue)), 1),
            "device": str(device),
            "backend": "huggingface_transformers",
        },
    )


def _load_esmfold_with_key_remap(torch, esm):
    """Load ESMFold checkpoint with key remapping for fair-esm 2.0.0.

    fair-esm 2.0.0 changed IPA linear layers to nested `Linear` modules,
    but the cached esmfold_3B_v1.pt checkpoint uses flat parameter names.
    We remap old→new keys to bridge the gap.
    """
    from pathlib import Path as _Path
    from esm.esmfold.v1.esmfold import ESMFold

    # Locate cached checkpoint
    cache_dir = _Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    ckpt_path = cache_dir / "esmfold_3B_v1.pt"

    if ckpt_path.exists():
        logger.info(f"  Loading ESMFold from cached checkpoint: {ckpt_path}")
        model_data = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    else:
        # Let fair-esm download (may fail without network)
        logger.info("  ESMFold checkpoint not cached, using esm.pretrained...")
        return esm.pretrained.esmfold_v1()

    cfg = model_data["cfg"]["model"]
    model_state = model_data["model"]
    model = ESMFold(esmfold_config=cfg)

    expected_keys = set(model.state_dict().keys())
    found_keys = set(model_state.keys())
    missing = expected_keys - found_keys

    # Remap: model expects 'X.linear.weight' but checkpoint has 'X.weight'
    if missing:
        remap_count = 0
        new_state = dict(model_state)
        for mkey in list(missing):
            # Try removing the extra '.linear' nesting
            parts = mkey.rsplit(".linear.", 1)
            if len(parts) == 2:
                old_key = parts[0] + "." + parts[1]
                if old_key in found_keys:
                    new_state[mkey] = new_state.pop(old_key)
                    remap_count += 1
        if remap_count:
            logger.info(f"  Remapped {remap_count} checkpoint keys for fair-esm 2.0.0")
        model_state = new_state

    model.load_state_dict(model_state, strict=False)
    return model


def _run_esmfold_fairesm(sequence: str, torch, esm) -> StructureResult:
    """Fair-ESM fallback for ESMFold inference."""
    if not hasattr(_run_esmfold_fairesm, "_model"):
        logger.info("  Loading ESMFold via fair-esm...")
        _run_esmfold_fairesm._model = _load_esmfold_with_key_remap(torch, esm)
        _run_esmfold_fairesm._model = _run_esmfold_fairesm._model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _run_esmfold_fairesm._model = _run_esmfold_fairesm._model.to(device)
        _run_esmfold_fairesm._device = device
    model = _run_esmfold_fairesm._model
    device = _run_esmfold_fairesm._device

    with torch.no_grad():
        output = model.infer(sequence)

    plddt_raw = output["plddt"][0, :len(sequence)]  # (seq_len,) or (seq_len, 37)
    if plddt_raw.dim() == 2:
        # Per-atom pLDDT in atom37 format — use Cα (index 1)
        plddt_raw = plddt_raw[:, 1]
    plddt = plddt_raw.cpu().numpy()
    per_residue = [round(float(x), 1) for x in plddt]
    mean_plddt = float(np.mean(per_residue))
    frac_above_70 = sum(1 for s in per_residue if s > 70) / max(len(per_residue), 1)

    pdb_string = None
    try:
        pdb_string = model.output_to_pdb(output)[0]
    except Exception:
        pass

    ptm = float(output["ptm"]) if "ptm" in output else None
    quality = _classify_quality(mean_plddt, frac_above_70)

    return StructureResult(
        method="esmfold",
        mean_plddt=round(mean_plddt, 1),
        per_residue_plddt=per_residue,
        ptm=round(ptm, 3) if ptm else None,
        quality=quality,
        pdb_string=pdb_string,
        details={
            "fraction_above_70": round(frac_above_70, 3),
            "min_plddt": round(float(min(per_residue)), 1),
            "max_plddt": round(float(max(per_residue)), 1),
            "device": str(device),
            "backend": "fair_esm",
        },
    )


# ═══════════════════════════════════════════════════════════════════
# AF2-multimer stub (command-line wrapper)
# ═══════════════════════════════════════════════════════════════════

def run_af2_multimer(
    binder_seq: str,
    target_seq: str,
    work_dir: str = "af2_work",
) -> StructureResult:
    """Run AlphaFold2-multimer for complex structure prediction.

    This generates the ColabFold / AF2 command for external execution,
    then parses the output if available.
    """
    import json as _json
    import subprocess

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    # Write input FASTA
    fasta_path = work / "complex.fasta"
    fasta_path.write_text(
        f">binder\n{binder_seq}\n>target\n{target_seq}\n"
    )

    # Try ColabFold — resolve binary from venv
    import sys
    venv_bin = Path(sys.executable).parent
    colabfold_bin = venv_bin / "colabfold_batch"
    if not colabfold_bin.exists():
        import shutil
        colabfold_bin = Path(shutil.which("colabfold_batch") or "colabfold_batch")
    cmd = (
        f"{colabfold_bin} {fasta_path} {work} "
        f"--model-type alphafold2_multimer_v3 --num-recycle 3"
    )
    logger.info(f"  Running AF2-multimer: {cmd}")

    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"  AF2-multimer execution failed: {e}")
        return StructureResult(
            method="af2_multimer",
            mean_plddt=0.0,
            per_residue_plddt=[],
            quality="FAILED",
            details={"error": str(e), "command": cmd},
        )

    # Parse output
    result_jsons = sorted(work.glob("*_scores_rank_001*.json"))
    if not result_jsons:
        return StructureResult(
            method="af2_multimer",
            mean_plddt=0.0,
            per_residue_plddt=[],
            quality="FAILED",
            details={"error": "No output JSON found", "command": cmd},
        )

    with open(result_jsons[0]) as f:
        scores = _json.load(f)

    plddt = scores.get("plddt", [])
    mean_plddt = float(np.mean(plddt)) if plddt else 0.0
    frac_above_70 = sum(1 for s in plddt if s > 70) / max(len(plddt), 1)
    quality = _classify_quality(mean_plddt, frac_above_70)

    return StructureResult(
        method="af2_multimer",
        mean_plddt=round(mean_plddt, 1),
        per_residue_plddt=[round(float(x), 1) for x in plddt],
        ptm=round(float(scores.get("ptm", 0)), 3),
        iptm=round(float(scores.get("iptm", 0)), 3),
        pae_matrix=scores.get("pae"),
        quality=quality,
        details={
            "fraction_above_70": round(frac_above_70, 3),
            "model": "alphafold2_multimer_v3",
            "recycles": 3,
        },
    )


# ═══════════════════════════════════════════════════════════════════
# Unified validation entry point
# ═══════════════════════════════════════════════════════════════════

def validate_structure(
    sequence: str,
    method: str = "auto",
    target_seq: str | None = None,
    seed: int = 42,
) -> StructureResult:
    """Validate a binder structure with the best available backend.

    Args:
        sequence: Binder amino acid sequence.
        method: "auto", "esmfold", "af2_multimer", or "heuristic".
        target_seq: Target sequence (needed for af2_multimer).
        seed: Random seed for heuristic fallback.

    Returns:
        StructureResult with pLDDT, quality, and optional PAE/pTM.
    """
    if method == "auto":
        backends = available_backends()
        if target_seq and "af2_multimer" in backends:
            method = "af2_multimer"
        elif "esmfold" in backends:
            method = "esmfold"
        else:
            method = "heuristic"

    if method == "esmfold":
        try:
            return run_esmfold(sequence)
        except Exception as e:
            logger.warning(f"  ESMFold failed, falling back to heuristic: {e}")
            return heuristic_plddt(sequence, seed)
    elif method == "af2_multimer":
        if target_seq is None:
            logger.warning("  AF2-multimer requires target_seq, using heuristic")
            return heuristic_plddt(sequence, seed)
        return run_af2_multimer(sequence, target_seq)
    else:
        return heuristic_plddt(sequence, seed)


def batch_validate(
    sequences: list[tuple[str, str]],
    method: str = "auto",
    seed: int = 42,
) -> list[dict]:
    """Validate a batch of sequences.

    Args:
        sequences: List of (id, sequence) tuples.
        method: Prediction backend.
        seed: Base random seed.

    Returns:
        List of dicts with id, sequence, and validation results.
    """
    results = []
    for i, (seq_id, seq) in enumerate(sequences):
        result = validate_structure(seq, method=method, seed=seed + i)
        results.append({
            "id": seq_id,
            "sequence": seq,
            "method": result.method,
            "mean_plddt": result.mean_plddt,
            "quality": result.quality,
            "ptm": result.ptm,
            "iptm": result.iptm,
            "details": result.details,
        })
    return results


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _classify_quality(mean_plddt: float, frac_above_70: float) -> str:
    """Classify structure quality from pLDDT statistics."""
    if mean_plddt > 85 and frac_above_70 > 0.80:
        return "HIGH"
    elif mean_plddt > 70 and frac_above_70 > 0.60:
        return "MEDIUM"
    return "LOW"

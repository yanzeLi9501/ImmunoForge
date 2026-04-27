"""
B4: Structure Validation — Dual Evaluation with ESMFold + ESM-2 Perplexity.

Uses ESMFold for primary structure prediction (pLDDT, pTM) and ESM-2
pseudo-perplexity as an orthogonal designability metric.

Dual evaluation provides two independent quality signals:
    1. ESMFold pLDDT: confidence in the predicted structure
    2. ESM-2 perplexity: how "natural" the sequence looks to the LM
       (lower = more natural, higher = less designable)
"""

import json
import logging
from pathlib import Path

import numpy as np

from immunoforge.core.structure_validation import validate_structure
from immunoforge.core.utils import save_json, ensure_dirs

logger = logging.getLogger(__name__)


def _compute_esm2_perplexity(sequence: str) -> float | None:
    """Compute ESM-2 pseudo-perplexity for a sequence.

    Pseudo-perplexity = exp(mean negative log-likelihood under masked LM).
    Lower perplexity indicates a more natural/designable sequence.
    """
    try:
        import torch
        import esm
    except ImportError:
        return None

    if not hasattr(_compute_esm2_perplexity, "_model"):
        try:
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.eval().to(device)
            _compute_esm2_perplexity._model = model
            _compute_esm2_perplexity._alphabet = alphabet
            _compute_esm2_perplexity._batch_converter = alphabet.get_batch_converter()
            _compute_esm2_perplexity._device = device
        except Exception:
            return None

    model = _compute_esm2_perplexity._model
    alphabet = _compute_esm2_perplexity._alphabet
    batch_converter = _compute_esm2_perplexity._batch_converter
    device = _compute_esm2_perplexity._device

    data = [("seq", sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    # Compute log-probabilities under the masked LM
    import torch
    log_probs = []
    with torch.no_grad():
        for i in range(1, len(sequence) + 1):
            masked = batch_tokens.clone()
            masked[0, i] = alphabet.mask_idx
            logits = model(masked)["logits"]
            log_p = torch.log_softmax(logits[0, i], dim=-1)
            true_token = batch_tokens[0, i]
            log_probs.append(log_p[true_token].item())

    mean_nll = -np.mean(log_probs)
    perplexity = float(np.exp(mean_nll))
    return round(perplexity, 2)


def main(config: dict) -> dict:
    """Execute B4: Structure Validation step."""
    logger.info("  B4: Structure Validation (ESMFold + ESM-2 Dual Evaluation)")

    output_dir = Path(config.get("paths", {}).get("output_dir", "outputs"))
    pdb_dir = output_dir / "predicted_structures"
    ensure_dirs(pdb_dir)

    qc_path = output_dir / "B3b_sequence_qc.json"
    # Also check B3a output
    opt_path = output_dir / "B3a_sequence_optimization.json"

    if not qc_path.exists():
        logger.warning("  B3b QC output not found")
        return {"status": "skipped"}

    with open(qc_path) as f:
        qc = json.load(f)

    passed = qc.get("passed", [])
    if not passed:
        logger.warning("  No passed sequences for structure validation")
        return {"status": "skipped", "reason": "no_passed_sequences"}

    val_cfg = config.get("structure_validation", {})
    pass_levels = val_cfg.get("pass_levels", ["HIGH", "MEDIUM"])
    dual_eval = val_cfg.get("dual_evaluation", True)

    validated = []
    quality_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

    for i, entry in enumerate(passed):
        seq = entry.get("sequence", entry.get("id", ""))
        if len(seq) < 10:
            continue

        seq_id = entry.get("id", f"seq_{i}")

        # Primary: ESMFold structure prediction
        result = validate_structure(seq, method=val_cfg.get("method", "auto"))

        quality_counts[result.quality] = quality_counts.get(result.quality, 0) + 1

        entry_result = {
            "id": seq_id,
            "sequence": seq,
            "method": result.method,
            "mean_plddt": result.mean_plddt,
            "ptm": result.ptm,
            "quality": result.quality,
            "details": result.details,
        }

        # Save PDB if available
        if result.pdb_string:
            pdb_path = pdb_dir / f"{seq_id}.pdb"
            pdb_path.write_text(result.pdb_string)
            entry_result["pdb_path"] = str(pdb_path)

        # Orthogonal: ESM-2 pseudo-perplexity
        if dual_eval:
            ppl = _compute_esm2_perplexity(seq)
            if ppl is not None:
                entry_result["esm2_perplexity"] = ppl
                # Low perplexity (<10) = natural sequence; high (>20) = unusual
                entry_result["esm2_designability"] = (
                    "good" if ppl < 10.0
                    else "moderate" if ppl < 20.0
                    else "poor"
                )

        if result.quality in pass_levels:
            validated.append(entry_result)

    result = {
        "total_input": len(passed),
        "quality_counts": quality_counts,
        "n_validated": len(validated),
        "method": val_cfg.get("method", "auto"),
        "dual_evaluation": dual_eval,
        "validated": validated,
    }

    logger.info(
        f"  Validation: {quality_counts} → {len(validated)} passed"
    )

    save_json(result, output_dir / "B4_structure_validation.json")
    return result

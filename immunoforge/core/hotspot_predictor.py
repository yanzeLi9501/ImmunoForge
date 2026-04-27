"""
AF2BIND-Style Automatic Hotspot Prediction.

Predicts protein-protein interaction hotspot residues using ESM-2 attention maps
as a proxy for contact propensity, eliminating the need for manual hotspot
specification in the pipeline configuration.

Approach:
    1. Run ESM-2 forward pass to extract per-layer attention matrices.
    2. Average attention across upper layers (layers 30-33 for ESM-2 3B)
       to capture long-range contact patterns.
    3. Compute per-residue attention entropy — low-entropy positions
       (focused attention) indicate structurally constrained residues.
    4. Combine with surface accessibility heuristic (hydrophilicity score)
       to rank candidate hotspot positions.
    5. Return top-K residues within the specified residue range.

References:
    - Tubiana J et al. bioRxiv (2022) — AF2BIND
    - Rao R et al. bioRxiv (2020) — ESM attention contacts
    - Lin Z et al. Science (2023) — ESM-2 / ESMFold
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Kyte-Doolittle hydrophilicity (inverted hydrophobicity) — surface proxy
_HYDROPHILICITY = {
    "A": -1.8, "R": 4.5, "N": 3.5, "D": 3.5, "C": -2.5,
    "E": 3.5, "Q": 3.5, "G": -0.4, "H": 3.2, "I": -4.5,
    "L": -3.8, "K": 3.9, "M": -1.9, "F": -2.8, "P": -1.6,
    "S": 0.8, "T": 0.7, "W": -0.9, "Y": 1.3, "V": -4.2,
}

# Charged / polar residues preferred at PPI interfaces
_INTERFACE_PROPENSITY = {
    "W": 1.5, "Y": 1.3, "R": 1.2, "H": 1.1, "F": 1.0,
    "M": 0.9, "D": 0.9, "N": 0.9, "E": 0.8, "Q": 0.8,
    "K": 0.8, "T": 0.7, "S": 0.7, "L": 0.7, "I": 0.6,
    "V": 0.6, "A": 0.5, "C": 0.5, "P": 0.4, "G": 0.3,
}


@dataclass
class HotspotResult:
    """Result of automatic hotspot prediction."""
    method: str                           # "esm2_attention" | "heuristic"
    residue_indices: list[int]            # 1-based residue indices
    scores: list[float]                   # per-residue hotspot score
    top_k_indices: list[int]              # selected top-K hotspot residues
    details: dict = field(default_factory=dict)


def _check_esm2() -> bool:
    """Check if ESM-2 model is importable."""
    try:
        import torch  # noqa: F401
        import esm    # noqa: F401
        return True
    except ImportError:
        return False


def predict_hotspots_esm2(
    sequence: str,
    residue_start: int = 1,
    residue_end: int | None = None,
    top_k: int = 10,
    attention_layers: tuple[int, ...] = (30, 31, 32, 33),
) -> HotspotResult:
    """Predict interface hotspots using ESM-2 attention maps.

    Uses the observation that attention heads in deep transformer layers
    capture long-range contacts (Rao et al., 2020). Residues with high
    focused attention in the upper layers correlate with structurally
    important / interface positions.

    Args:
        sequence: Target protein amino acid sequence.
        residue_start: First residue index to consider (1-based).
        residue_end: Last residue index to consider (1-based, inclusive).
        top_k: Number of top hotspot residues to return.
        attention_layers: Which ESM-2 layers to extract attention from.

    Returns:
        HotspotResult with ranked hotspot positions.
    """
    import torch
    import esm

    if residue_end is None:
        residue_end = len(sequence)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ESM-2 model (cached singleton)
    if not hasattr(predict_hotspots_esm2, "_model"):
        logger.info("  Loading ESM-2 model for hotspot prediction...")
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model = model.eval().to(device)
        predict_hotspots_esm2._model = model
        predict_hotspots_esm2._alphabet = alphabet
        predict_hotspots_esm2._batch_converter = alphabet.get_batch_converter()

    model = predict_hotspots_esm2._model
    batch_converter = predict_hotspots_esm2._batch_converter

    # Prepare input
    data = [("target", sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    # Forward pass with attention output
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], need_head_weights=True)

    # Extract attention: shape (layers, heads, seq_len+2, seq_len+2)
    # +2 for BOS/EOS tokens
    attentions = results["attentions"]  # (1, n_layers, n_heads, L+2, L+2)
    attentions = attentions[0]  # (n_layers, n_heads, L+2, L+2)

    # Select upper layers and average across heads
    n_layers = attentions.shape[0]
    # Clamp layer indices to valid range
    valid_layers = [l for l in attention_layers if l < n_layers]
    if not valid_layers:
        valid_layers = list(range(max(0, n_layers - 4), n_layers))

    # Average attention across selected layers and heads
    # Remove BOS (index 0) and EOS (last index)
    attn_selected = attentions[valid_layers].mean(dim=(0, 1))  # (L+2, L+2)
    attn_map = attn_selected[1:-1, 1:-1]  # (L, L) — residue-to-residue

    attn_map = attn_map.cpu().numpy()
    seq_len = len(sequence)

    # Compute per-residue metrics
    scores = np.zeros(seq_len)
    for i in range(seq_len):
        # 1. Column attention sum (how much other residues attend to this one)
        col_attn = np.sum(attn_map[:, i])

        # 2. Attention entropy (low entropy = focused = structurally important)
        row = attn_map[i, :]
        row = row / (row.sum() + 1e-12)
        entropy = -np.sum(row * np.log(row + 1e-12))
        max_entropy = np.log(seq_len)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 1.0

        # 3. Interface propensity from amino acid type
        aa = sequence[i]
        propensity = _INTERFACE_PROPENSITY.get(aa, 0.5)

        # 4. Surface accessibility proxy (hydrophilicity)
        hydro = _HYDROPHILICITY.get(aa, 0.0)
        surface_score = max(0, (hydro + 4.5) / 9.0)  # normalize to [0, 1]

        # Combined score: high column attention + low entropy + interface propensity
        scores[i] = (
            0.40 * col_attn +
            0.25 * (1.0 - normalized_entropy) +
            0.20 * propensity +
            0.15 * surface_score
        )

    # Restrict to specified residue range (convert to 0-based)
    range_mask = np.zeros(seq_len, dtype=bool)
    for i in range(max(0, residue_start - 1), min(seq_len, residue_end)):
        range_mask[i] = True

    masked_scores = scores.copy()
    masked_scores[~range_mask] = -np.inf

    # Select top-K
    top_indices = np.argsort(masked_scores)[::-1][:top_k]
    top_indices = sorted(top_indices)  # sort by position

    # Convert to 1-based
    top_k_residues = [int(i) + 1 for i in top_indices if masked_scores[i] > -np.inf]

    return HotspotResult(
        method="esm2_attention",
        residue_indices=list(range(1, seq_len + 1)),
        scores=[round(float(s), 4) for s in scores],
        top_k_indices=top_k_residues,
        details={
            "attention_layers": valid_layers,
            "device": str(device),
            "residue_range": f"{residue_start}-{residue_end}",
            "n_candidates": int(range_mask.sum()),
        },
    )


def predict_hotspots_heuristic(
    sequence: str,
    residue_start: int = 1,
    residue_end: int | None = None,
    top_k: int = 10,
) -> HotspotResult:
    """Predict hotspots using sequence-only heuristics (fallback).

    Combines interface propensity, hydrophilicity, and local sequence
    context to rank residue positions.
    """
    if residue_end is None:
        residue_end = len(sequence)

    seq_len = len(sequence)
    scores = np.zeros(seq_len)

    for i, aa in enumerate(sequence):
        # Interface propensity
        propensity = _INTERFACE_PROPENSITY.get(aa, 0.5)

        # Surface accessibility proxy
        hydro = _HYDROPHILICITY.get(aa, 0.0)
        surface = max(0, (hydro + 4.5) / 9.0)

        # Local context: residues flanked by polar/charged residues
        # are more likely surface-exposed
        context_bonus = 0.0
        if i > 0 and _HYDROPHILICITY.get(sequence[i - 1], 0) > 0:
            context_bonus += 0.1
        if i < seq_len - 1 and _HYDROPHILICITY.get(sequence[i + 1], 0) > 0:
            context_bonus += 0.1

        # Loop propensity: Pro, Gly, Asn, Asp common in loops (accessible)
        loop_bonus = 0.15 if aa in "PGND" else 0.0

        scores[i] = (
            0.40 * propensity +
            0.25 * surface +
            0.20 * context_bonus +
            0.15 * loop_bonus
        )

    # Apply residue range filter
    range_mask = np.zeros(seq_len, dtype=bool)
    for i in range(max(0, residue_start - 1), min(seq_len, residue_end)):
        range_mask[i] = True

    masked_scores = scores.copy()
    masked_scores[~range_mask] = -np.inf

    top_indices = np.argsort(masked_scores)[::-1][:top_k]
    top_indices = sorted(top_indices)

    top_k_residues = [int(i) + 1 for i in top_indices if masked_scores[i] > -np.inf]

    return HotspotResult(
        method="heuristic",
        residue_indices=list(range(1, seq_len + 1)),
        scores=[round(float(s), 4) for s in scores],
        top_k_indices=top_k_residues,
        details={
            "residue_range": f"{residue_start}-{residue_end}",
            "n_candidates": int(range_mask.sum()),
            "warning": "Heuristic-only — ESM-2 attention provides better accuracy",
        },
    )


def predict_hotspots(
    sequence: str,
    residue_start: int = 1,
    residue_end: int | None = None,
    top_k: int = 10,
    method: str = "auto",
) -> HotspotResult:
    """Predict interface hotspot residues.

    Args:
        sequence: Target protein amino acid sequence.
        residue_start: First residue index (1-based).
        residue_end: Last residue index (1-based, inclusive).
        top_k: Number of hotspot residues to predict.
        method: "auto", "esm2_attention", or "heuristic".

    Returns:
        HotspotResult with predicted hotspot positions.
    """
    if method == "auto":
        method = "esm2_attention" if _check_esm2() else "heuristic"

    if method == "esm2_attention":
        try:
            return predict_hotspots_esm2(
                sequence, residue_start, residue_end, top_k
            )
        except Exception as e:
            logger.warning(f"  ESM-2 hotspot prediction failed: {e}")
            return predict_hotspots_heuristic(
                sequence, residue_start, residue_end, top_k
            )
    else:
        return predict_hotspots_heuristic(
            sequence, residue_start, residue_end, top_k
        )

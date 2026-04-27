"""
Structure Diversity & Helicity Loss Module.

Implements BindCraft-inspired negative helicity loss and structural diversity
scoring to prevent over-representation of alpha-helical designs.

BindCraft uses a helicity penalty during optimization to encourage diverse
secondary structure content in designed binders. This module provides:
    1. Sequence-based helicity estimation (Chou-Fasman propensities)
    2. Helicity penalty computation
    3. Structural diversity scoring between candidate sequences

References:
    - Jendrusch et al. bioRxiv (2024) — BindCraft negative helicity loss
    - Chou PY, Fasman GD. Biochemistry (1974) — helix propensity
"""

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Chou-Fasman helix propensity parameters (normalized to [0, 1])
# Higher value = stronger helix propensity
_HELIX_PROPENSITY = {
    "A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70,
    "E": 1.51, "Q": 1.11, "G": 0.57, "H": 1.00, "I": 1.08,
    "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13, "P": 0.57,
    "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06,
}
_MAX_HELIX = max(_HELIX_PROPENSITY.values())

# Sheet propensity (used for diversity calculation)
_SHEET_PROPENSITY = {
    "A": 0.83, "R": 0.93, "N": 0.89, "D": 0.54, "C": 1.19,
    "E": 0.37, "Q": 1.10, "G": 0.75, "H": 0.87, "I": 1.60,
    "L": 1.30, "K": 0.74, "M": 1.05, "F": 1.38, "P": 0.55,
    "S": 0.75, "T": 1.19, "W": 1.37, "Y": 1.47, "V": 1.70,
}


def compute_helicity(sequence: str) -> float:
    """Compute normalized helicity score from sequence (0.0 to 1.0).

    Uses Chou-Fasman helix propensity averaged over the sequence.
    Score of 1.0 means maximum helix propensity (all Ala/Glu/Met).
    """
    if not sequence:
        return 0.0

    total = sum(_HELIX_PROPENSITY.get(aa, 0.8) for aa in sequence.upper())
    mean_propensity = total / len(sequence)

    # Normalize to [0, 1]
    return min(1.0, mean_propensity / _MAX_HELIX)


def compute_helicity_penalty(
    sequence: str,
    target_helicity: float = 0.50,
    penalty_scale: float = 1.0,
) -> float:
    """Compute helicity penalty for a sequence.

    BindCraft-inspired: penalize sequences that are excessively helical.
    The penalty is zero when helicity is at the target level, and increases
    quadratically above the target.

    Args:
        sequence: Amino acid sequence.
        target_helicity: Desired helicity level (default 0.50).
        penalty_scale: Scaling factor for the penalty.

    Returns:
        Penalty value (0.0 = no penalty, higher = more helical).
    """
    helicity = compute_helicity(sequence)

    if helicity <= target_helicity:
        return 0.0

    # Quadratic penalty above target
    excess = helicity - target_helicity
    return penalty_scale * excess * excess


def compute_ss_composition(sequence: str) -> dict:
    """Compute secondary structure composition from sequence propensities.

    Returns dict with helix_fraction, sheet_fraction, coil_fraction estimates.
    """
    if not sequence:
        return {"helix": 0.0, "sheet": 0.0, "coil": 0.0}

    helix_scores = [_HELIX_PROPENSITY.get(aa, 0.8) for aa in sequence.upper()]
    sheet_scores = [_SHEET_PROPENSITY.get(aa, 0.8) for aa in sequence.upper()]

    n = len(sequence)
    n_helix = sum(1 for h, s in zip(helix_scores, sheet_scores)
                  if h > 1.0 and h > s)
    n_sheet = sum(1 for h, s in zip(helix_scores, sheet_scores)
                  if s > 1.0 and s >= h)
    n_coil = n - n_helix - n_sheet

    return {
        "helix": round(n_helix / n, 3),
        "sheet": round(n_sheet / n, 3),
        "coil": round(n_coil / n, 3),
    }


def sequence_diversity(seq1: str, seq2: str) -> float:
    """Compute sequence diversity between two sequences (0-1).

    Returns fraction of positions that differ (higher = more diverse).
    """
    if not seq1 or not seq2:
        return 1.0

    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))

    mismatches = sum(1 for a, b in zip(seq1[:min_len], seq2[:min_len]) if a != b)
    mismatches += max_len - min_len  # length differences count as mismatches

    return mismatches / max_len


def population_diversity(sequences: Sequence[str]) -> float:
    """Compute average pairwise diversity of a sequence population.

    Returns mean diversity (0 = all identical, 1 = all completely different).
    """
    if len(sequences) < 2:
        return 0.0

    n = len(sequences)
    total_div = 0.0
    n_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            total_div += sequence_diversity(sequences[i], sequences[j])
            n_pairs += 1

    return round(total_div / n_pairs, 4) if n_pairs > 0 else 0.0

"""
VESM (Variant Effect from Shared Models) — Mutation Effect Prediction.

Wraps VESM_650M from ntranoslab/vesm for single-point variant
effect scoring during affinity maturation (B6/B9b).  VESM provides
log-likelihood ratios (LLR) for every possible amino acid substitution
at every position in a protein sequence, enabling rapid pre-screening
of mutation candidates before expensive structure prediction.

VESM loads the base ESM2 architecture (facebook/esm2_t33_650M_UR50D)
then overlays co-distilled weights from ntranoslab/vesm/VESM_650M.pth.

Integration point:
    maturation.py → vesm_prescreen_mutations() filters candidate
    mutations, keeping only those with ΔScore > threshold (neutral
    or beneficial).  This reduces the number of structure evaluations
    per maturation generation by ~60-70%.

Reference:
    Dinh T, Jang SK, Zaitlen N, Ntranos V.  VESM: Compressing the
    collective knowledge of ESM into a single protein language model.
    arXiv (2024).
"""

import logging
import os
from functools import lru_cache
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Model configuration ──────────────────────────────────────────

VESM_BASE_MODEL = "facebook/esm2_t33_650M_UR50D"
VESM_REPO_ID = "ntranoslab/vesm"
VESM_WEIGHT_FILE = "VESM_650M.pth"
_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
_AA_SET = set(_AA_ORDER)

# Cache the loaded model/tokenizer globally to avoid repeated loading
_model_cache: dict = {}


def _check_vesm_available() -> bool:
    """Check if VESM dependencies (transformers, torch) are importable."""
    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer, EsmForMaskedLM  # noqa: F401
        from huggingface_hub import hf_hub_download  # noqa: F401
        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def _vesm_available() -> bool:
    return _check_vesm_available()


def load_vesm_model(device: str = "auto"):
    """Load VESM model and tokenizer, caching for reuse.

    Procedure:
        1. Load base ESM2-650M from HuggingFace
        2. Download VESM_650M.pth weights from ntranoslab/vesm
        3. Overlay VESM weights onto base model (strict=False)

    Args:
        device: "auto" (GPU if available), "cpu", or "cuda:N".

    Returns:
        (model, tokenizer, device_str)
    """
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["tokenizer"], _model_cache["device"]

    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM
    from huggingface_hub import hf_hub_download

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading VESM model (base={VESM_BASE_MODEL}) on {device}...")

    # 1. Load base ESM2 model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(VESM_BASE_MODEL)
    model = EsmForMaskedLM.from_pretrained(VESM_BASE_MODEL)

    # 2. Download VESM co-distilled weights
    logger.info(f"Downloading VESM weights ({VESM_REPO_ID}/{VESM_WEIGHT_FILE})...")
    weight_path = hf_hub_download(
        repo_id=VESM_REPO_ID, filename=VESM_WEIGHT_FILE
    )

    # 3. Overlay VESM weights
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    logger.info("VESM weights loaded successfully")

    model = model.eval().to(device)

    _model_cache["model"] = model
    _model_cache["tokenizer"] = tokenizer
    _model_cache["device"] = device

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"VESM model ready ({n_params:.0f}M params, device={device})")
    return model, tokenizer, device


def compute_mutation_scores(
    sequence: str,
    positions: Optional[list[int]] = None,
    device: str = "auto",
) -> dict[int, dict[str, float]]:
    """Compute VESM log-likelihood ratios for mutations at given positions.

    For each position, returns the LLR for every possible substitution:
        LLR(pos, mut) = log P(mut | context) - log P(wt | context)

    Positive LLR → mutation is *more* likely than wildtype (potentially beneficial).
    Negative LLR → mutation is *less* likely (potentially deleterious).

    Args:
        sequence: Parent amino acid sequence.
        positions: List of 0-indexed positions to score.
            If None, scores all positions.
        device: Device for inference.

    Returns:
        Dict mapping position → {aa: LLR_score} for all 20 amino acids.
    """
    import torch

    model, tokenizer, device = load_vesm_model(device)

    if positions is None:
        positions = list(range(len(sequence)))

    # Tokenize the full sequence
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"].to(device)

    # Get wildtype logits
    with torch.no_grad():
        outputs = model(**{k: v.to(device) for k, v in inputs.items()})
        logits = outputs.logits[0]  # (seq_len, vocab_size)

    # Convert logits to log-probabilities
    log_probs = torch.log_softmax(logits, dim=-1)

    # Map amino acids to token IDs
    aa_token_ids = {}
    for aa in _AA_ORDER:
        token_id = tokenizer.convert_tokens_to_ids(aa)
        if token_id != tokenizer.unk_token_id:
            aa_token_ids[aa] = token_id

    scores = {}
    for pos in positions:
        if pos < 0 or pos >= len(sequence):
            continue

        wt_aa = sequence[pos]
        if wt_aa not in _AA_SET:
            continue

        # Token index = position + 1 (due to [CLS] / <cls> token)
        tok_idx = pos + 1

        wt_token_id = aa_token_ids.get(wt_aa)
        if wt_token_id is None:
            continue

        wt_logprob = log_probs[tok_idx, wt_token_id].item()

        pos_scores = {}
        for aa, tid in aa_token_ids.items():
            llr = log_probs[tok_idx, tid].item() - wt_logprob
            pos_scores[aa] = round(llr, 4)

        scores[pos] = pos_scores

    return scores


def vesm_prescreen_mutations(
    sequence: str,
    candidate_mutations: list[dict],
    threshold: float = -7.0,
    device: str = "auto",
) -> list[dict]:
    """Pre-screen candidate mutations using VESM, keeping beneficial/neutral ones.

    Mutations with VESM LLR < threshold are filtered out (likely deleterious).
    This reduces the number of expensive structure evaluations needed during
    affinity maturation.

    Args:
        sequence: Parent amino acid sequence.
        candidate_mutations: List of mutation dicts, each with keys:
            - "position": int (0-indexed)
            - "mutated": str (target amino acid)
            - (optional) "original", "notation"
        threshold: LLR cutoff. Mutations below this are rejected.
            Default -7.0 (permissive; reject only clearly deleterious).
        device: Device for VESM inference.

    Returns:
        Filtered list of mutations that pass the VESM screen,
        each annotated with "vesm_score" key.
    """
    if not candidate_mutations:
        return []

    if not _vesm_available():
        logger.warning("VESM not available, returning all mutations unfiltered")
        for m in candidate_mutations:
            m["vesm_score"] = None
        return candidate_mutations

    # Collect unique positions to score
    positions = sorted({m["position"] for m in candidate_mutations})

    try:
        scores = compute_mutation_scores(sequence, positions, device=device)
    except Exception as e:
        logger.warning(f"VESM scoring failed: {e}, returning all mutations unfiltered")
        for m in candidate_mutations:
            m["vesm_score"] = None
        return candidate_mutations

    # Filter mutations
    passed = []
    rejected = 0
    for m in candidate_mutations:
        pos = m["position"]
        mut_aa = m["mutated"]
        pos_scores = scores.get(pos, {})
        llr = pos_scores.get(mut_aa)

        if llr is not None:
            m["vesm_score"] = llr
            if llr >= threshold:
                passed.append(m)
            else:
                rejected += 1
        else:
            # If we can't score it, keep it (conservative)
            m["vesm_score"] = None
            passed.append(m)

    logger.info(
        f"VESM pre-screen: {len(passed)} passed, {rejected} rejected "
        f"(threshold={threshold})"
    )
    return passed


def rank_mutations_by_vesm(
    sequence: str,
    positions: Optional[list[int]] = None,
    top_k: int = 20,
    device: str = "auto",
) -> list[dict]:
    """Rank all possible mutations by VESM score (best first).

    Useful for identifying the most promising mutation sites for
    directed evolution / maturation campaigns.

    Args:
        sequence: Parent amino acid sequence.
        positions: Positions to consider (None = all).
        top_k: Number of top mutations to return.
        device: Device for inference.

    Returns:
        List of dicts with keys: position, original, mutated, notation, vesm_score.
    """
    if not _vesm_available():
        logger.warning("VESM not available")
        return []

    scores = compute_mutation_scores(sequence, positions, device=device)

    all_mutations = []
    for pos, pos_scores in scores.items():
        wt_aa = sequence[pos]
        for aa, llr in pos_scores.items():
            if aa != wt_aa:
                all_mutations.append({
                    "position": pos,
                    "original": wt_aa,
                    "mutated": aa,
                    "notation": f"{wt_aa}{pos + 1}{aa}",
                    "vesm_score": llr,
                })

    all_mutations.sort(key=lambda m: m["vesm_score"], reverse=True)
    return all_mutations[:top_k]

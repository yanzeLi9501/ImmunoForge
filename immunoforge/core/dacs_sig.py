"""
DACS-Sig — real-data-optimised consensus K_D scorer (v6) with an optional
ESM-2 sequence-reliability gate (v7).

Motivation
----------
The publication DACS model (v5) fits two free class offsets (delta_ppi,
delta_denovo) on top of a 4-method weighted geometric mean that *includes* an
ipTM->K_D structural term.  On the canonical fixed held-out split
(Ipili / SIRPa / LCB3) this over-fits: held-out MALE = 1.72 — worse than the
pre-DACS multi-method baseline B0 (held-out MALE = 0.98).

DACS-Sig (v6) directly fixes this highest-priority weakness:

  * adaptive per-class / per-strength weights over the 3 sequence/energy
    experts only (BSA, PRODIGY, Rosetta) — the ipTM->K_D term is dropped as a
    direct K_D expert (leakage-controlled ipTM-K_D rho < 0.30);
  * BSA winsorisation to TRAIN 5/95 log-percentile bounds;
  * a single, strongly-shrunk GLOBAL log-K_D offset (tau = 3.0) instead of free
    per-class offsets, so holding out a scarce class cannot destabilise a
    class-specific term;
  * a 50/50 ensemble with the raw pre-DACS multi-method baseline B0 for
    variance reduction.

All hyper-parameters (ensemble weight 0.5, tau 3.0) were selected by
TRAIN-only leave-one-out CV; the fixed held-out split and the 96
class-balanced splits are reported as frozen out-of-sample outcomes.

Held-out (Ipili / SIRPa / LCB3) MALE:
    B0 0.98  ·  DACS v5 1.72  ·  DACS-Sig v6 0.37  ·  DACS-Sig v7 (gate) 0.30.

ESM-2 gate (v7, optional)
-------------------------
A per-entry sequence-quality signal g in [0, 1] derived ONLY from the binder
sequence via ESM-2 masked pseudo-perplexity (PPL).  It never sees the target
and is never a K_D term; it only modulates how strongly each entry is blended
toward the assumption-light baseline B0:

    ens = ens_lo + (ens_hi - ens_lo) * sigmoid((PPL - ppl0) / s)
    pred = (1 - ens) * (adaptive_consensus + global_offset) + ens * B0

High PPL (out-of-distribution sequence) -> lean on B0; low PPL (natural-looking
sequence) -> trust the calibrated consensus.  The manuscript reports the gate
as an honest negative ablation (it lowers held-out/LOO point estimates but
removes the 96-split significance), so it is OFF by default and offered here as
an opt-in filter.

Frozen calibration constants live in ``core/data/dacs_sig_benchmark.json``
(fit on the 15-entry training split); they are loaded once at import.
"""

from __future__ import annotations

import json
import logging
import math
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_EPS = 1e-3

_DATA_PATH = Path(__file__).resolve().parent / "data" / "dacs_sig_benchmark.json"


# ─────────────────────────────────────────────────────────────────────────────
# Frozen calibration constants (loaded from the shipped benchmark artifact).
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _calibration() -> dict:
    with open(_DATA_PATH, encoding="utf-8") as fh:
        art = json.load(fh)
    return art["constants"]


@lru_cache(maxsize=1)
def benchmark() -> dict:
    """Return the frozen 18-entry DACS-Sig benchmark artifact (inputs only)."""
    with open(_DATA_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _safe_log(x: float) -> float:
    return math.log10(max(float(x), _EPS))


def normalise_class(binder_type: str | None) -> str:
    """Map a production binder_type to a DACS-Sig calibration class.

    Returns one of ``'ab'`` (antibody), ``'denovo'`` (de novo miniprotein) or
    ``'nat'`` (natural protein / everything else).
    """
    if binder_type in ("ab", "nat", "denovo"):
        return binder_type
    if binder_type in ("VH", "VL", "scFv"):
        return "ab"
    if binder_type == "denovo":  # pragma: no cover - covered by first branch
        return "denovo"
    return "nat"


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive per-class / per-strength expert weights (3 sequence/energy experts).
# Strength is estimated from the median of the three experts (log10 nM).
# NO ipTM->K_D term: ipTM is a reliability gate only, never a direct K_D expert.
# ─────────────────────────────────────────────────────────────────────────────
def adaptive_weights(log_p: float, log_r: float, log_b: float,
                     binder_class: str) -> dict:
    """Binder-type/strength-dependent weights for BSA / PRODIGY / Rosetta."""
    log_median = float(sorted([log_p, log_r, log_b])[1])
    if binder_class == "nat":
        if log_median < 1.5:
            return {"bsa": 1.0, "prodigy": 2.0, "rosetta": 1.5}
        if log_median < 3.0:
            return {"bsa": 1.0, "prodigy": 2.5, "rosetta": 1.0}
        return {"bsa": 0.5, "prodigy": 2.0, "rosetta": 1.5}
    if binder_class == "denovo":
        if log_median < 1.5:
            return {"bsa": 1.0, "prodigy": 1.5, "rosetta": 1.5}
        return {"bsa": 0.7, "prodigy": 1.8, "rosetta": 1.5}
    # antibody
    if log_median < 1.5:        # strong < 30 nM
        return {"bsa": 2.0, "prodigy": 1.5, "rosetta": 1.0}
    if log_median < 3.0:        # medium 30-1000 nM
        return {"bsa": 3.0, "prodigy": 1.0, "rosetta": 0.5}
    return {"bsa": 1.0, "prodigy": 2.0, "rosetta": 1.0}  # weak > 1000 nM


def adaptive_log_consensus(bsa_kd: float, prodigy_kd: float, rosetta_kd: float,
                           binder_class: str) -> float:
    """Winsorised, adaptively-weighted log10 consensus of the 3 experts."""
    cal = _calibration()
    log_p = _safe_log(prodigy_kd)
    log_r = _safe_log(rosetta_kd)
    log_b = _safe_log(bsa_kd)
    log_b = min(max(log_b, cal["winsor_floor_log10"]), cal["winsor_ceil_log10"])
    w = adaptive_weights(log_p, log_r, log_b, binder_class)
    wsum = w["bsa"] + w["prodigy"] + w["rosetta"]
    return (w["bsa"] * log_b + w["prodigy"] * log_p + w["rosetta"] * log_r) / wsum


def gate_weight(esm2_ppl: float) -> float:
    """ESM-2 sequence-reliability ensemble weight (v7) in [ens_lo, ens_hi]."""
    g = _calibration()["gate"]
    sig = 1.0 / (1.0 + math.exp(-(float(esm2_ppl) - g["ppl0"]) / g["s"]))
    return g["ens_lo"] + (g["ens_hi"] - g["ens_lo"]) * sig


def dacs_sig_log_kd(bsa_kd: float, prodigy_kd: float, rosetta_kd: float,
                    binder_class: str, b0_kd: float,
                    esm2_gate: bool = False,
                    esm2_ppl: float | None = None) -> dict:
    """DACS-Sig (v6) consensus log10 K_D, optionally ESM-2-gated (v7).

    Args:
        bsa_kd, prodigy_kd, rosetta_kd: per-method K_D predictions (nM).
        binder_class: ``'ab'`` | ``'nat'`` | ``'denovo'`` (see
            :func:`normalise_class`).
        b0_kd: raw pre-DACS multi-method baseline K_D (nM) — the ensemble
            anchor (B0).
        esm2_gate: when True, use the per-entry ESM-2 reliability gate (v7)
            instead of the fixed 0.5 ensemble weight (v6).
        esm2_ppl: ESM-2 masked pseudo-perplexity of the binder sequence.
            Required when ``esm2_gate`` is True; if None the gate is disabled
            and the v6 fixed ensemble weight is used (with a warning).

    Returns:
        dict with ``log_kd``, ``kd_nM``, ``ensemble_weight``, ``mode``,
        ``version`` (``"v6"`` or ``"v7"``) and the intermediate
        ``calibrated_log_kd`` / ``b0_log_kd``.
    """
    cal = _calibration()
    calibrated = adaptive_log_consensus(bsa_kd, prodigy_kd, rosetta_kd,
                                        binder_class) + cal["global_offset_log10"]
    b0_log = _safe_log(b0_kd)

    mode = "dacs_sig_v6"
    version = "v6"
    if esm2_gate:
        if esm2_ppl is None:
            logger.warning(
                "esm2_gate=True but no esm2_ppl provided; falling back to the "
                "v6 fixed ensemble weight. Install the [gpu] extra and supply "
                "an ESM-2 pseudo-perplexity to enable the gate."
            )
            ens = cal["ensemble_b0"]
        else:
            ens = gate_weight(esm2_ppl)
            mode = "dacs_sig_v7_gate"
            version = "v7"
    else:
        ens = cal["ensemble_b0"]

    log_pred = (1.0 - ens) * calibrated + ens * b0_log
    return {
        "log_kd": log_pred,
        "kd_nM": 10.0 ** log_pred,
        "ensemble_weight": ens,
        "calibrated_log_kd": calibrated,
        "b0_log_kd": b0_log,
        "mode": mode,
        "version": version,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Optional ESM-2 masked pseudo-perplexity (heavy; needs torch + fair-esm).
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_esm2():  # pragma: no cover - requires the optional [gpu] extra
    import torch  # noqa: F401
    import esm

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    return model, alphabet


def esm2_pseudo_perplexity(sequence: str) -> float | None:
    """ESM-2 masked pseudo-perplexity of *sequence* (None if ESM-2 absent).

    This is the v7 gate feature.  It is intentionally optional: the package
    ships and runs without ``torch``/``fair-esm``.  When those are unavailable
    this returns ``None`` and the gate is disabled by the caller.
    """
    try:  # pragma: no cover - exercised only when the optional extra is present
        import torch

        model, alphabet = _load_esm2()
        bc = alphabet.get_batch_converter()
        _, _, toks = bc([("q", sequence)])
        with torch.no_grad():
            nlls = []
            for pos in range(1, toks.size(1) - 1):
                masked = toks.clone()
                masked[0, pos] = alphabet.mask_idx
                logits = model(masked)["logits"]
                lp = torch.log_softmax(logits[0, pos], dim=-1)
                nlls.append(-lp[toks[0, pos]].item())
        return float(math.exp(sum(nlls) / len(nlls)))
    except Exception as exc:  # noqa: BLE001
        logger.info("ESM-2 pseudo-perplexity unavailable (%s); gate disabled.", exc)
        return None

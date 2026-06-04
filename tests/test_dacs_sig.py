"""Reproducibility + unit tests for the DACS-Sig (v6) and ESM-2 gate (v7) scorer.

These load the frozen 18-entry benchmark artifact shipped with the package and
assert that the integrated scorer reproduces the published held-out metrics:

    B0 0.98  ·  DACS v5 1.72  ·  DACS-Sig v6 0.37  ·  DACS-Sig v7 (gate) 0.30.
"""

import math

import pytest

from immunoforge.core import dacs_sig


def _log(x):
    return math.log10(max(float(x), 1e-3))


@pytest.fixture(scope="module")
def bench():
    return dacs_sig.benchmark()


def _holdout(bench):
    return [e for e in bench["entries"] if e["split"] == "holdout"]


def _male(pairs):
    return sum(abs(p - e) for p, e in pairs) / len(pairs)


def test_constants_loaded():
    c = dacs_sig._calibration()
    assert c["global_offset_log10"] == pytest.approx(-2.296084, abs=1e-6)
    assert c["winsor_floor_log10"] == pytest.approx(-1.878558, abs=1e-6)
    assert c["winsor_ceil_log10"] == pytest.approx(4.782821, abs=1e-6)
    assert c["ensemble_b0"] == 0.5
    assert c["tau"] == 3.0


def test_v6_reproduces_holdout_male(bench):
    pairs = []
    for e in _holdout(bench):
        r = dacs_sig.dacs_sig_log_kd(
            e["bsa_kd_nM"], e["prodigy_kd_nM"], e["rosetta_kd_nM"],
            e["binder_class"], e["v4_b0_kd_nM"],
        )
        pairs.append((r["log_kd"], _log(e["exp_kd_nM"])))
    assert _male(pairs) == pytest.approx(0.3651, abs=1e-3)


def test_v7_gate_reproduces_holdout_male(bench):
    pairs = []
    for e in _holdout(bench):
        r = dacs_sig.dacs_sig_log_kd(
            e["bsa_kd_nM"], e["prodigy_kd_nM"], e["rosetta_kd_nM"],
            e["binder_class"], e["v4_b0_kd_nM"],
            esm2_gate=True, esm2_ppl=e["esm2_ppl"],
        )
        assert r["mode"] == "dacs_sig_v7_gate"
        pairs.append((r["log_kd"], _log(e["exp_kd_nM"])))
    assert _male(pairs) == pytest.approx(0.3020, abs=1e-3)


def test_v6_beats_b0_and_v5_on_holdout(bench):
    """DACS-Sig held-out MALE must beat both the raw baseline and DACS v5."""
    ref = bench["reference_metrics_holdout"]
    assert ref["DACS_Sig_v6"]["MALE"] < ref["B0"]["MALE"]
    assert ref["DACS_Sig_v6"]["MALE"] < ref["DACS_v5"]["MALE"]


def test_gate_disabled_without_ppl(bench):
    """esm2_gate=True with no PPL must fall back to the v6 fixed weight."""
    e = _holdout(bench)[0]
    gated = dacs_sig.dacs_sig_log_kd(
        e["bsa_kd_nM"], e["prodigy_kd_nM"], e["rosetta_kd_nM"],
        e["binder_class"], e["v4_b0_kd_nM"], esm2_gate=True, esm2_ppl=None,
    )
    plain = dacs_sig.dacs_sig_log_kd(
        e["bsa_kd_nM"], e["prodigy_kd_nM"], e["rosetta_kd_nM"],
        e["binder_class"], e["v4_b0_kd_nM"],
    )
    assert gated["mode"] == "dacs_sig_v6"
    assert gated["ensemble_weight"] == plain["ensemble_weight"]
    assert gated["log_kd"] == pytest.approx(plain["log_kd"])


def test_gate_weight_monotonic():
    """Higher ESM-2 PPL (more out-of-distribution) -> more weight on B0."""
    lo = dacs_sig.gate_weight(1.0)
    hi = dacs_sig.gate_weight(16.0)
    assert lo < hi
    assert 0.25 <= lo <= 0.5
    assert 0.25 <= hi <= 0.5


def test_normalise_class():
    assert dacs_sig.normalise_class("VH") == "ab"
    assert dacs_sig.normalise_class("scFv") == "ab"
    assert dacs_sig.normalise_class("denovo") == "denovo"
    assert dacs_sig.normalise_class("natural_protein") == "nat"
    assert dacs_sig.normalise_class(None) == "nat"


def test_esm2_pseudo_perplexity_optional():
    """Must return None (not raise) when torch/fair-esm are unavailable."""
    val = dacs_sig.esm2_pseudo_perplexity("MSQAKKDPLDPATAQLASARGT")
    assert val is None or val > 0


def test_dacs_sig_log_kd_reports_version(bench):
    e = _holdout(bench)[0]
    v6 = dacs_sig.dacs_sig_log_kd(
        e["bsa_kd_nM"], e["prodigy_kd_nM"], e["rosetta_kd_nM"],
        e["binder_class"], e["v4_b0_kd_nM"],
    )
    v7 = dacs_sig.dacs_sig_log_kd(
        e["bsa_kd_nM"], e["prodigy_kd_nM"], e["rosetta_kd_nM"],
        e["binder_class"], e["v4_b0_kd_nM"],
        esm2_gate=True, esm2_ppl=e["esm2_ppl"],
    )
    assert v6["version"] == "v6"
    assert v7["version"] == "v7"


# ═══════════════════════════════════════════════════════════════════
# Integration: DACS-Sig (v6/v7) is the publication default in affinity.py
# ═══════════════════════════════════════════════════════════════════

import warnings  # noqa: E402

from immunoforge.core import affinity  # noqa: E402


def _real_three():
    """Three AffinityResults whose method names map to BSA/PRODIGY/Rosetta."""
    return [
        affinity.AffinityResult("BSA_regression", -10.0, 100.0, {}),
        affinity.AffinityResult("PRODIGY-binding", -10.0, 200.0, {}),
        affinity.AffinityResult("Rosetta_REF2015", -10.0, 150.0, {}),
    ]


def test_consensus_default_is_dacs_sig_v6():
    """Default consensus_kd now uses the publication DACS-Sig v6 (no gate)."""
    res = affinity.consensus_kd(_real_three(), binder_type="natural_protein")
    assert res["dacs_version"] == "v6"
    assert res["dacs_mode"] == "dacs_sig_v6"


def test_consensus_v7_alias_enables_gate_then_falls_back():
    """dacs_mode='v7' requests the gate; without a PPL it degrades to v6."""
    res = affinity.consensus_kd(_real_three(), binder_type="natural_protein",
                                dacs_mode="v7")
    # No esm2_ppl provided and ESM-2 not installed -> graceful v6 fallback.
    assert res["dacs_version"] == "v6"


def test_consensus_v7_with_ppl_is_v7():
    res = affinity.consensus_kd(_real_three(), binder_type="natural_protein",
                                dacs_mode="v7", esm2_ppl=4.0)
    assert res["dacs_version"] == "v7"
    assert res["dacs_mode"] == "dacs_sig_v7_gate"


def test_v5_emits_deprecation_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        affinity.consensus_kd(_real_three(), binder_type="natural_protein",
                              dacs_mode="v5")
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_run_affinity_analysis_defaults_to_v6():
    out = affinity.run_affinity_analysis("MSQAKKDPLDPATAQLASARGTGGSGGSRGT", 1200)
    assert out["consensus"]["dacs_version"] == "v6"


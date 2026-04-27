"""Tests for extension modules: structure_validation, immunogenicity,
maturation, linker_design, developability, benchmark, multi_codon_opt,
active_learning, visualization."""

import math
import pytest

# ═══════════════════════════════════════════════════════════════════
# Structure validation
# ═══════════════════════════════════════════════════════════════════

from immunoforge.core.structure_validation import (
    validate_structure,
    batch_validate,
    available_backends,
    heuristic_plddt,
    StructureResult,
)

TEST_SEQ = "MAELIKSYGVSGATNFSLLKQAGDVEENPGP"
LONG_SEQ = "A" * 200


def test_heuristic_plddt():
    result = heuristic_plddt(TEST_SEQ)
    assert isinstance(result, StructureResult)
    assert result.method == "heuristic"
    assert 0 < result.mean_plddt < 100
    assert len(result.per_residue_plddt) == len(TEST_SEQ)


def test_validate_structure_auto():
    result = validate_structure(TEST_SEQ, method="auto")
    assert isinstance(result, StructureResult)
    assert result.quality in ("HIGH", "MEDIUM", "LOW")


def test_validate_structure_heuristic():
    result = validate_structure(TEST_SEQ, method="heuristic")
    assert result.method == "heuristic"


def test_batch_validate():
    results = batch_validate([("s1", TEST_SEQ), ("s2", LONG_SEQ)])
    assert len(results) == 2
    for r in results:
        assert isinstance(r, dict)
        assert "mean_plddt" in r


def test_available_backends():
    backends = available_backends()
    assert "heuristic" in backends
    assert isinstance(backends, list)


# ═══════════════════════════════════════════════════════════════════
# Immunogenicity
# ═══════════════════════════════════════════════════════════════════

from immunoforge.core.immunogenicity import (
    predict_immunogenicity,
    batch_immunogenicity,
    ImmunogenicityResult,
)


def test_predict_immunogenicity():
    result = predict_immunogenicity(TEST_SEQ, species="human")
    assert isinstance(result, ImmunogenicityResult)
    assert 0 <= result.immunogenicity_score <= 1
    assert result.risk_level in ("low", "moderate", "high")


def test_predict_immunogenicity_mouse():
    result = predict_immunogenicity(TEST_SEQ, species="mouse")
    assert isinstance(result, ImmunogenicityResult)


def test_batch_immunogenicity():
    results = batch_immunogenicity([("s1", TEST_SEQ), ("s2", LONG_SEQ)])
    assert len(results) == 2


def test_immunogenicity_epitope_density():
    result = predict_immunogenicity(LONG_SEQ, species="human")
    assert result.epitope_density >= 0


# ═══════════════════════════════════════════════════════════════════
# Maturation
# ═══════════════════════════════════════════════════════════════════

from immunoforge.core.maturation import (
    run_maturation,
    batch_maturation,
    point_mutate,
    MaturationResult,
)

MAT_SEQ = "MAELIKSYGVSGATNFSLLK"


def test_point_mutate():
    import random
    rng = random.Random(42)
    mutant, mutations = point_mutate(MAT_SEQ, n_mutations=1, rng=rng)
    assert len(mutant) == len(MAT_SEQ)
    diffs = sum(1 for a, b in zip(MAT_SEQ, mutant) if a != b)
    assert diffs >= 1


def test_run_maturation():
    result = run_maturation(MAT_SEQ, target_kd_nM=1.0, max_generations=2, seed=42)
    assert isinstance(result, MaturationResult)
    assert result.initial_kd_nM > 0
    assert result.final_kd_nM > 0
    assert result.improvement_fold > 0
    assert result.best_candidate is not None


def test_batch_maturation():
    candidates = [
        {"id": "c1", "sequence": MAT_SEQ, "bsa": 1200, "sc": 0.65},
        {"id": "c2", "sequence": "DIQMTQTTSSLSASLGDR", "bsa": 1100, "sc": 0.60},
    ]
    results = batch_maturation(candidates, max_generations=2)
    assert len(results) == 2


# ═══════════════════════════════════════════════════════════════════
# Linker design
# ═══════════════════════════════════════════════════════════════════

from immunoforge.core.linker_design import (
    design_linker,
    design_bispecific,
    design_all_linker_variants,
    LinkerDesign,
    BispecificConstruct,
)

BINDER1 = "MAELIKSYGVS"
BINDER2 = "DIQMTQTTSS"


def test_design_linker_flexible():
    result = design_linker("flexible", target_length=20)
    assert isinstance(result, LinkerDesign)
    assert result.linker_type == "flexible"
    assert len(result.sequence) >= 10


def test_design_linker_rigid():
    result = design_linker("rigid")
    assert result.linker_type == "rigid"


def test_design_bispecific():
    result = design_bispecific("B1", BINDER1, "B2", BINDER2)
    assert isinstance(result, BispecificConstruct)
    assert BINDER1 in result.fusion_seq
    assert BINDER2 in result.fusion_seq
    assert result.total_length > len(BINDER1) + len(BINDER2)
    assert result.estimated_mw_da > 0


def test_design_all_variants():
    results = design_all_linker_variants("B1", BINDER1, "B2", BINDER2)
    assert len(results) >= 3
    for r in results:
        assert isinstance(r, BispecificConstruct)


# ═══════════════════════════════════════════════════════════════════
# Developability
# ═══════════════════════════════════════════════════════════════════

from immunoforge.core.developability import (
    run_developability_assessment,
    batch_developability,
    predict_solubility,
    predict_thermal_stability,
    analyze_charge_distribution,
    DevelopabilityReport,
)


def test_predict_solubility():
    result = predict_solubility(TEST_SEQ)
    assert hasattr(result, "camsol_score")
    assert isinstance(result.camsol_score, float)


def test_predict_thermal_stability():
    result = predict_thermal_stability(TEST_SEQ)
    assert hasattr(result, "estimated_tm_celsius")
    assert result.estimated_tm_celsius > 0


def test_analyze_charge():
    result = analyze_charge_distribution(TEST_SEQ)
    assert hasattr(result, "net_charge_pH7")


def test_developability_assessment():
    report = run_developability_assessment(TEST_SEQ)
    assert isinstance(report, DevelopabilityReport)
    assert 0 <= report.overall_score <= 1
    assert report.developability_class in ("excellent", "good", "acceptable", "poor")


def test_batch_developability():
    results = batch_developability([("s1", TEST_SEQ), ("s2", LONG_SEQ)])
    assert len(results) == 2


# ═══════════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════════

from immunoforge.core.benchmark import (
    BENCHMARK_ENTRIES,
    get_benchmark_entries,
    run_benchmark,
    generate_benchmark_report,
    get_pdb_ids_for_benchmark,
    run_loo_cv,
    bootstrap_ci,
)


def test_benchmark_entries_exist():
    assert len(BENCHMARK_ENTRIES) >= 10


def test_get_benchmark_entries_filter():
    t_cells = get_benchmark_entries(cell_type="T cell")
    assert len(t_cells) >= 3
    for e in t_cells:
        assert e.cell_type == "T cell"


def test_get_benchmark_with_sequence():
    with_seq = get_benchmark_entries(with_sequence=True)
    assert len(with_seq) >= 5
    for e in with_seq:
        assert e.binder_sequence is not None


def test_run_benchmark():
    result = run_benchmark(seed=42)
    assert "n_entries" in result
    assert "metrics" in result
    assert "predictions" in result
    assert result["n_entries"] > 0


def test_benchmark_metrics():
    result = run_benchmark(seed=42)
    m = result["metrics"]
    assert "mean_absolute_log10_error" in m
    assert "spearman_rho" in m
    assert "fraction_within_1_log" in m


def test_generate_benchmark_report():
    result = run_benchmark(seed=42)
    report = generate_benchmark_report(result)
    assert "ImmunoForge Benchmark" in report
    assert "Spearman" in report


def test_pdb_ids():
    pdbs = get_pdb_ids_for_benchmark()
    assert len(pdbs) >= 5
    assert all(len(p) == 4 for p in pdbs)


def test_run_loo_cv():
    result = run_loo_cv(seed=42)
    assert "loo_male" in result
    assert "loo_spearman_rho" in result
    assert "n_folds" in result
    assert result["n_folds"] >= 5
    assert result["loo_male"] < 2.5


def test_bootstrap_ci():
    result = bootstrap_ci(n_bootstrap=100, seed=42)
    assert "male_point" in result
    assert "male_95ci" in result
    assert "rho_95ci" in result
    lo, hi = result["male_95ci"]
    assert lo <= result["male_point"] <= hi


# ═══════════════════════════════════════════════════════════════════
# Multi-expression codon optimization
# ═══════════════════════════════════════════════════════════════════

from immunoforge.core.multi_codon_opt import (
    optimize_for_aav,
    optimize_for_mrna,
    optimize_for_ecoli,
    optimize_for_pichia,
    optimize_multi_system,
    generate_comparison_table,
    deplete_cpg,
    count_cpg,
    MultiSystemResult,
)

SHORT_PROTEIN = "MAELIK"


def test_optimize_for_aav():
    result = optimize_for_aav(SHORT_PROTEIN)
    assert isinstance(result, MultiSystemResult)
    assert result.system == "aav"
    assert result.cds_length_bp == len(SHORT_PROTEIN) * 3


def test_optimize_for_mrna():
    result = optimize_for_mrna(SHORT_PROTEIN)
    assert result.system == "mRNA"


def test_optimize_for_ecoli():
    result = optimize_for_ecoli(SHORT_PROTEIN)
    assert result.system == "ecoli"
    assert result.host == "ecoli"


def test_optimize_for_pichia():
    result = optimize_for_pichia(SHORT_PROTEIN)
    assert result.system == "pichia"


def test_optimize_multi_system():
    results = optimize_multi_system(SHORT_PROTEIN)
    assert len(results) >= 4
    for name, r in results.items():
        assert isinstance(r, MultiSystemResult)
        assert r.cds_length_bp > 0


def test_deplete_cpg():
    original = "ATGCGATCGATCGATCG"
    depleted = deplete_cpg(original)
    assert count_cpg(depleted) <= count_cpg(original)


def test_comparison_table():
    results = optimize_multi_system(SHORT_PROTEIN)
    table = generate_comparison_table(results)
    assert "System" in table
    assert "GC%" in table


# ═══════════════════════════════════════════════════════════════════
# Active learning
# ═══════════════════════════════════════════════════════════════════

from immunoforge.core.active_learning import (
    ExperimentalMeasurement,
    PredictionRecord,
    compute_prediction_error,
    update_weights,
    upper_confidence_bound,
    expected_improvement,
    run_feedback_round,
    DEFAULT_WEIGHTS,
)


def test_compute_prediction_error():
    measurements = [
        ExperimentalMeasurement("B1", "AAA", "CD3E", 1.0),
        ExperimentalMeasurement("B2", "BBB", "PD-1", 10.0),
    ]
    predictions = [
        PredictionRecord("B1", 2.0, 0.8),
        PredictionRecord("B2", 5.0, 0.6),
    ]
    result = compute_prediction_error(measurements, predictions)
    assert result["n_pairs"] == 2
    assert "mean_abs_log10_error" in result


def test_update_weights():
    measurements = [
        ExperimentalMeasurement("B1", "AAA", "T", 1.0),
    ]
    predictions = [
        PredictionRecord("B1", 100.0, 0.8),  # Over-predict
    ]
    updated = update_weights(DEFAULT_WEIGHTS, measurements, predictions)
    assert sum(updated.values()) == pytest.approx(1.0, abs=0.01)
    assert all(v >= 0.02 for v in updated.values())


def test_ucb_acquisition():
    candidates = [
        {"binder_id": "C1", "predicted_kd_nM": 1.0, "prediction_uncertainty": 0.5},
        {"binder_id": "C2", "predicted_kd_nM": 100.0, "prediction_uncertainty": 2.0},
    ]
    ranked = upper_confidence_bound(candidates)
    assert len(ranked) == 2
    assert "ucb_score" in ranked[0]


def test_expected_improvement():
    candidates = [
        {"binder_id": "C1", "predicted_kd_nM": 0.5, "prediction_uncertainty": 0.3},
        {"binder_id": "C2", "predicted_kd_nM": 50.0, "prediction_uncertainty": 1.0},
    ]
    ranked = expected_improvement(candidates, best_measured_kd_nM=1.0)
    assert len(ranked) == 2
    assert "ei_score" in ranked[0]


def test_feedback_round():
    measurements = [ExperimentalMeasurement("B1", "AAA", "T", 5.0)]
    predictions = [PredictionRecord("B1", 10.0, 0.7)]
    fb = run_feedback_round(1, measurements, predictions)
    assert fb.round_id == 1
    assert "prediction_error" in fb.metrics


# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════

from immunoforge.core.visualization import (
    kd_vs_score_scatter,
    kd_distribution_histogram,
    calibration_plot,
    developability_radar,
    generate_pipeline_report,
    generate_markdown_report,
)


def test_kd_scatter():
    candidates = [
        {"binder_id": "C1", "predicted_kd_nM": 1.0, "composite_score": 0.9, "quality": "HIGH"},
        {"binder_id": "C2", "predicted_kd_nM": 100.0, "composite_score": 0.3, "quality": "LOW"},
    ]
    html = kd_vs_score_scatter(candidates)
    assert "C1" in html or "table" in html.lower()


def test_kd_histogram():
    html = kd_distribution_histogram([1.0, 5.0, 10.0, 50.0, 100.0])
    assert isinstance(html, str)
    assert len(html) > 10


def test_calibration_plot():
    html = calibration_plot([1.0, 10.0, 100.0], [2.0, 8.0, 150.0])
    assert isinstance(html, str)


def test_radar_chart():
    scores = {"solubility": 0.8, "stability": 0.6, "pI": 0.7, "aggregation": 0.9}
    html = developability_radar(scores)
    assert isinstance(html, str)


def test_pipeline_report():
    html = generate_pipeline_report({
        "candidates": [
            {"binder_id": "Test", "predicted_kd_nM": 5.0, "composite_score": 0.7, "quality": "HIGH"},
        ],
        "config": {"target": "CD3E"},
    })
    assert "ImmunoForge" in html
    assert "Test" in html


def test_markdown_report():
    md = generate_markdown_report({
        "candidates": [
            {"binder_id": "Test", "predicted_kd_nM": 5.0, "composite_score": 0.7},
        ],
        "config": {"target": "CD3E", "species": "human"},
    })
    assert "# ImmunoForge" in md


# ═══════════════════════════════════════════════════════════════════
# Pipeline runner extensions
# ═══════════════════════════════════════════════════════════════════

from immunoforge.pipeline.runner import STEP_REGISTRY, ALL_STEPS, EXTENDED_STEPS


def test_extended_steps_registered():
    for step in ["B4b", "B5b", "B5c", "B6b", "B7b", "B9"]:
        assert step in STEP_REGISTRY, f"Missing step: {step}"


def test_extended_steps_list():
    assert len(EXTENDED_STEPS) > len(ALL_STEPS)
    for s in ALL_STEPS:
        assert s in EXTENDED_STEPS

"""Smoke test for Round 6 modules: AF3-multimer + conditional maturation."""
import sys
import json
from pathlib import Path

print("=" * 60)
print("Round 6 Smoke Test: AF3 + Affinity Maturation")
print("=" * 60)

# 1. Import all new modules
print("\n[1] Testing imports...")
from immunoforge.core.af3_multimer import (
    predict_ternary_complex,
    predict_ternary_batch,
    _write_boltz_yaml,
    _parse_boltz_output,
    _classify_complex_quality,
    _read_sequence_from_pdb,
    TernaryComplexResult,
    _check_boltz,
)
print("  af3_multimer: OK")

from immunoforge.pipeline.steps.B9a_af3_validation import main as b9a_main
print("  B9a_af3_validation: OK")

from immunoforge.pipeline.steps.B9b_affinity_maturation import main as b9b_main
print("  B9b_affinity_maturation: OK")

from immunoforge.pipeline.runner import (
    STEP_REGISTRY, FULL_PIPELINE, ALL_STEPS_WITH_OPT, EXTENDED_STEPS
)
print("  runner (FULL_PIPELINE): OK")

# 2. Boltz availability
print("\n[2] Boltz availability...")
boltz_ok = _check_boltz()
print(f"  Boltz available: {boltz_ok}")
if boltz_ok:
    import boltz
    print(f"  Boltz version: {boltz.__version__}")

# 3. Test Boltz YAML generation
print("\n[3] Boltz YAML generation...")
import tempfile
tmpdir = Path(tempfile.mkdtemp())
binder_seq = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSDYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDRLSITIRPRYYGLDVWGQGTLVTVSS"
target1_seq = "MKFLVNVALVFMVVYISYIYADKDTEVQLLESGGGLVQPGGSLRLSCAAS"
target2_seq = "QDGNEEMGGITQTPYKVSISGTTVILTCPQ"
yaml_path = _write_boltz_yaml(binder_seq, target1_seq, target2_seq, tmpdir)
print(f"  YAML written to: {yaml_path}")
content = yaml_path.read_text()
assert "version: 1" in content
assert binder_seq in content
assert "msa: empty" in content
print(f"  YAML content valid (3 protein chains)")

# 4. Test quality classification
print("\n[4] Complex quality classification...")
assert _classify_complex_quality(0.85, 0.90) == "HIGH"
assert _classify_complex_quality(0.60, 0.70) == "MEDIUM"
assert _classify_complex_quality(0.30, 0.50) == "LOW"
print("  HIGH/MEDIUM/LOW classification: OK")

# 5. Test TernaryComplexResult dataclass
print("\n[5] TernaryComplexResult dataclass...")
r = TernaryComplexResult(
    binder_id="test_1",
    binder_seq="ACDE",
    target1_name="T1",
    target2_name="T2",
    iptm=0.72,
    ptm=0.80,
    complex_plddt=0.85,
    quality="MEDIUM",
)
assert r.binder_id == "test_1"
assert r.iptm == 0.72
print(f"  Dataclass: {r.binder_id} ipTM={r.iptm} quality={r.quality}")

# 6. Test runner registry
print("\n[6] Runner step registry...")
assert "B9a" in STEP_REGISTRY, "B9a not in registry"
assert "B9b" in STEP_REGISTRY, "B9b not in registry"
assert "B9a" in FULL_PIPELINE, "B9a not in FULL_PIPELINE"
assert "B9b" in FULL_PIPELINE, "B9b not in FULL_PIPELINE"
assert "B9a" in EXTENDED_STEPS, "B9a not in EXTENDED_STEPS"
print(f"  B9a: {STEP_REGISTRY['B9a'][1]}")
print(f"  B9b: {STEP_REGISTRY['B9b'][1]}")
print(f"  FULL_PIPELINE: {FULL_PIPELINE}")

# 7. Test B9b maturation with minimal config
print("\n[7] B9b conditional maturation (dry run)...")
import tempfile, os
tmpout = Path(tempfile.mkdtemp()) / "outputs"
tmpout.mkdir()

# Create a fake B6 ranking JSON
fake_b6 = {
    "total_input": 2,
    "top_n": 2,
    "ranked": [
        {
            "id": "cand_1",
            "sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSDYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDRLSITIRPRYYGLDVWGQGTLVTVSS",
            "kd_nM": 2698.3,
            "bsa": 1167.2,
            "sc": 0.611,
        },
        {
            "id": "cand_2",
            "sequence": "SLAALLRAAAEAAAAAAAAARAAAAAAATAAADAALAAATKAAAEAAARAAAAEAERAAEEARREAERREEEARR",
            "kd_nM": 50.0,
            "bsa": 1226.1,
            "sc": 0.557,
        },
    ],
}
(tmpout / "B6_candidate_ranking.json").write_text(json.dumps(fake_b6))

test_config = {
    "paths": {"output_dir": str(tmpout)},
    "affinity_maturation": {
        "kd_threshold_nM": 100.0,
        "target_kd_nM": 50.0,
        "max_generations": 2,
        "candidates_per_gen": 8,
        "top_k_per_gen": 3,
        "temperatures": [0.1, 0.3],
        "seed": 42,
    },
}

result = b9b_main(test_config)
print(f"  Status: {result['status']}")
print(f"  Matured: {result.get('n_matured', 0)} candidates")
print(f"  Below threshold: {result.get('n_below_threshold', 0)}")
if result.get("maturation_results"):
    for mr in result["maturation_results"]:
        print(f"    {mr['parent_id']}: {mr['initial_kd_nM']:.1f} → {mr['final_kd_nM']:.1f} nM "
              f"({mr['improvement_fold']:.1f}x, {mr['n_generations']} gens)")

# Cleanup
import shutil
shutil.rmtree(tmpdir, ignore_errors=True)
shutil.rmtree(tmpout.parent, ignore_errors=True)

print("\n" + "=" * 60)
print("All Round 6 smoke tests PASSED")
print("=" * 60)

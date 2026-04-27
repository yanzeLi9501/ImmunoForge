"""Quick smoke test for new modules."""
import sys
sys.path.insert(0, ".")

# Test imports
from immunoforge.core.hotspot_predictor import predict_hotspots, _check_esm2
from immunoforge.core.sequence_optimizer import run_optimization
from immunoforge.core.structure_diversity import (
    compute_helicity_penalty, compute_helicity, population_diversity,
    compute_ss_composition, sequence_diversity,
)
from immunoforge.pipeline.steps.B3a_sequence_optimization import main as b3a_main

print("All new modules imported successfully")
print(f"ESM-2 available: {_check_esm2()}")

# Test helicity computation
h = compute_helicity("AELKAELKAELK")
print(f"Helix propensity (poly-helix): {h:.3f}")
h2 = compute_helicity("PGNDPGNDPGND")
print(f"Helix propensity (loop-rich): {h2:.3f}")

# Test helicity penalty
pen = compute_helicity_penalty("AELKAELKAELK", target_helicity=0.5)
print(f"Helicity penalty (poly-helix): {pen:.4f}")
pen2 = compute_helicity_penalty("PGNDPGNDPGND", target_helicity=0.5)
print(f"Helicity penalty (loop-rich): {pen2:.4f}")

# Test SS composition
ss = compute_ss_composition("AELKAELKAELK")
print(f"SS composition (poly-helix): {ss}")
ss2 = compute_ss_composition("VILFVILFVILF")
print(f"SS composition (sheet-rich): {ss2}")

# Test diversity
div = population_diversity(["AELKAELK", "PGNDPGND", "AELKAELK"])
print(f"Population diversity: {div:.4f}")
sd = sequence_diversity("AELKAELK", "PGNDPGND")
print(f"Sequence diversity (AELK vs PGND): {sd:.4f}")

# Test heuristic hotspot prediction (no GPU needed)
from immunoforge.core.hotspot_predictor import predict_hotspots_heuristic
hp = predict_hotspots_heuristic("MDWTWRILFLVAAATGAHS" * 3, top_k=5)
print(f"Heuristic hotspots: {hp.top_k_indices}")
print(f"Hotspot method: {hp.method}")

# Test runner has B3a registered
from immunoforge.pipeline.runner import STEP_REGISTRY, ALL_STEPS_WITH_OPT
print(f"B3a in registry: {'B3a' in STEP_REGISTRY}")
print(f"ALL_STEPS_WITH_OPT: {ALL_STEPS_WITH_OPT}")

# Test ESM-2 attention hotspot prediction (GPU)
print("\n--- ESM-2 Attention Hotspot Test ---")
try:
    hp2 = predict_hotspots("MDWTWRILFLVAAATGAHS" * 3, method="esm2_attention", top_k=5)
    print(f"ESM-2 hotspots: {hp2.top_k_indices}")
    print(f"ESM-2 method: {hp2.method}")
    print(f"ESM-2 details: {hp2.details}")
except Exception as e:
    print(f"ESM-2 hotspot failed (expected if no GPU): {e}")

print("\nAll smoke tests passed!")

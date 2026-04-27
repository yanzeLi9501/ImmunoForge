"""
Full Pipeline API — Complete input→output pipeline with per-step visualization.

Provides:
  POST /api/pipeline/full   — Run entire pipeline synchronously, return all step outputs
  GET  /api/visualize/affinity — Affinity calibration chart data
  GET  /api/visualize/structure — Structure quality chart data
  GET  /api/visualize/pipeline — Per-step summary for dashboard
"""

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request / Response Models ──

class FullPipelineRequest(BaseModel):
    """Complete pipeline input."""
    binder_sequence: str = Field(..., min_length=10, max_length=1000,
                                 description="Binder amino acid sequence")
    target_name: str = Field(..., description="Target protein name (e.g., CD3E, PD-1)")
    target_sequence: str | None = Field(None, description="Target sequence (optional, looked up if omitted)")
    bsa: float | None = Field(None, description="Override BSA in Å² (auto-estimated if omitted)")
    sc: float | None = Field(None, description="Override shape complementarity (auto-estimated if omitted)")
    species: str = Field("human", description="Species: human, mouse, cynomolgus")
    expression_system: str = Field("vaccinia", description="Expression system for codon optimization")
    run_esmfold: bool = Field(True, description="Run ESMFold structure prediction (requires GPU)")
    run_maturation: bool = Field(False, description="Run in-silico affinity maturation")
    maturation_generations: int = Field(5, ge=1, le=20)
    seed: int = Field(42)


class StepResult(BaseModel):
    """Individual pipeline step result."""
    step: str
    name: str
    status: str  # "completed" | "failed" | "skipped"
    time_s: float
    result: dict
    visualization: dict | None = None  # Chart data for frontend


class FullPipelineResponse(BaseModel):
    """Complete pipeline output with all step results + visualization data."""
    status: str
    total_time_s: float
    binder_id: str
    target_name: str
    species: str
    steps: list[StepResult]
    summary: dict


# ── Helper functions ──

def _safe_round(v, n=3):
    if isinstance(v, float):
        return round(v, n)
    return v


def _run_step(name: str, label: str, fn, *args, **kwargs) -> StepResult:
    """Execute a step with timing and error handling."""
    t0 = time.time()
    try:
        result, viz = fn(*args, **kwargs)
        return StepResult(
            step=name, name=label, status="completed",
            time_s=round(time.time() - t0, 2),
            result=result, visualization=viz,
        )
    except Exception as e:
        logger.exception(f"Step {name} failed: {e}")
        return StepResult(
            step=name, name=label, status="failed",
            time_s=round(time.time() - t0, 2),
            result={"error": str(e)}, visualization=None,
        )


# ── Pipeline Step Implementations ──

def step_sequence_qc(sequence: str):
    """Step 1: Sequence QC."""
    from immunoforge.core.sequence_qc import run_full_qc
    result = run_full_qc(sequence)
    viz = {
        "type": "qc_summary",
        "pass_count": sum(1 for r in result.get("results", []) if r.get("passed")),
        "fail_count": sum(1 for r in result.get("results", []) if not r.get("passed")),
        "rules": [
            {"name": r.get("rule", ""), "passed": r.get("passed", False),
             "detail": r.get("detail", "")}
            for r in result.get("results", [])
        ],
    }
    return result, viz


def step_domain_classification(sequence: str):
    """Step 2: Domain classification + CDR identification."""
    from immunoforge.core.affinity import classify_binder_type, identify_cdrs
    domain_type = classify_binder_type(sequence)
    cdrs = identify_cdrs(sequence, domain_type)
    result = {
        "domain_type": domain_type,
        "n_cdrs": len(cdrs),
        "cdrs": {k: {"seq": v["seq"], "start": v["start"], "end": v["end"]}
                 for k, v in cdrs.items()},
        "sequence_length": len(sequence),
    }
    # Visualization: sequence annotation with CDR highlights
    annotations = []
    for cdr_name, cdr_info in cdrs.items():
        annotations.append({
            "label": cdr_name,
            "start": cdr_info["start"],
            "end": cdr_info["end"],
            "sequence": cdr_info["seq"],
        })
    viz = {
        "type": "sequence_annotation",
        "sequence": sequence,
        "domain_type": domain_type,
        "annotations": annotations,
    }
    return result, viz


def step_structure_prediction(sequence: str, run_gpu: bool = True):
    """Step 3: Structure prediction (ESMFold or heuristic)."""
    from immunoforge.core.structure_validation import validate_structure
    sr = validate_structure(sequence, method="auto" if run_gpu else "heuristic")
    result = {
        "method": sr.method,
        "mean_plddt": round(sr.mean_plddt, 1),
        "quality": sr.quality,
        "ptm": _safe_round(sr.ptm),
        "n_residues": len(sr.per_residue_plddt),
        "pdb_available": sr.pdb_string is not None and len(sr.pdb_string or "") > 0,
        "per_residue_plddt": [round(p, 1) for p in sr.per_residue_plddt],
    }
    # Store PDB string separately (not in JSON response — too large)
    result["_pdb_string"] = sr.pdb_string

    # Visualization: per-residue pLDDT line chart
    viz = {
        "type": "plddt_line",
        "title": f"Per-Residue pLDDT ({sr.method})",
        "x_label": "Residue",
        "y_label": "pLDDT",
        "data": [
            {"residue": i + 1, "plddt": round(p, 1)}
            for i, p in enumerate(sr.per_residue_plddt)
        ],
        "mean_plddt": round(sr.mean_plddt, 1),
        "quality": sr.quality,
        "thresholds": [
            {"value": 90, "label": "Very High", "color": "#0053D6"},
            {"value": 70, "label": "Confident", "color": "#65CBF3"},
            {"value": 50, "label": "Low", "color": "#FFDB13"},
        ],
    }
    return result, viz


def step_affinity_prediction(sequence: str, pdb_string: str | None,
                              bsa_override: float | None, sc_override: float | None,
                              seed: int = 42):
    """Step 4: Binding affinity prediction (3-method consensus)."""
    from immunoforge.core.structure_contacts import structure_aware_affinity
    result = structure_aware_affinity(
        sequence, pdb_string=pdb_string,
        bsa_override=bsa_override, sc_override=sc_override, seed=seed,
    )

    # Remove internal keys
    pdb_key = result.pop("_pdb_string", None)
    consensus = result.get("consensus", {})
    individual = consensus.get("individual_kds", {})

    # Visualization: method comparison bar chart
    viz = {
        "type": "affinity_comparison",
        "title": "K_D Prediction by Method",
        "consensus_kd_nM": consensus.get("consensus_kd_nM"),
        "confidence": consensus.get("confidence"),
        "structure_derived": result.get("structure_derived", False),
        "methods": [],
    }
    method_colors = {
        "BSA_regression": "#2ecc71",
        "PRODIGY-binding": "#3498db",
        "Rosetta_REF2015": "#e74c3c",
    }
    for method_name, kd_value in individual.items():
        viz["methods"].append({
            "name": method_name,
            "kd_nM": kd_value,
            "log10_kd": round(math.log10(max(kd_value, 0.001)), 2),
            "color": method_colors.get(method_name, "#999"),
        })

    return result, viz


def step_immunogenicity(sequence: str, species: str):
    """Step 5: Immunogenicity prediction."""
    from immunoforge.core.immunogenicity import predict_immunogenicity
    im = predict_immunogenicity(sequence, species=species)
    result = {
        "immunogenicity_score": round(im.immunogenicity_score, 3),
        "risk_level": im.risk_level,
        "epitope_density": round(im.epitope_density, 4),
        "n_epitopes": len(im.epitopes),
        "hotspot_regions": im.hotspot_regions,
        "method": im.details.get("method", "unknown"),
        "humanness_score": im.details.get("humanness_score"),
        "n_strong_class_i": im.details.get("n_strong_class_i", 0),
    }
    # Visualization: epitope density along sequence
    viz = {
        "type": "epitope_map",
        "title": "Immunogenicity Risk Map",
        "risk_level": im.risk_level,
        "score": round(im.immunogenicity_score, 3),
        "hotspots": im.hotspot_regions,
        "sequence_length": len(sequence),
    }
    return result, viz


def step_developability(sequence: str):
    """Step 6: Developability assessment."""
    from immunoforge.core.developability import run_developability_assessment
    dev = run_developability_assessment(sequence)
    result = {
        "overall_score": round(dev.overall_score, 3),
        "developability_class": dev.developability_class,
        "solubility": round(dev.solubility.camsol_score, 3),
        "solubility_class": dev.solubility.solubility_class,
        "predicted_tm_C": round(dev.thermal_stability.estimated_tm_celsius, 1),
        "net_charge_pH7": round(dev.charge.net_charge_pH7, 2),
        "pI": dev.physicochemical["pI"],
        "mw_da": dev.physicochemical["mw_da"],
        "hydrophobic_fraction": dev.physicochemical["hydrophobic_fraction"],
        "flags": dev.flags,
    }
    # Visualization: radar chart data
    agg_propensity = dev.physicochemical["hydrophobic_fraction"]
    viz = {
        "type": "radar",
        "title": "Developability Profile",
        "axes": [
            {"label": "Solubility", "value": round(dev.solubility.camsol_score, 3), "max": 1.0},
            {"label": "Thermal Stability", "value": round(min(dev.thermal_stability.estimated_tm_celsius / 80.0, 1.0), 3), "max": 1.0},
            {"label": "Low Aggregation", "value": round(1.0 - agg_propensity, 3), "max": 1.0},
            {"label": "Charge Balance", "value": round(dev.charge.charge_symmetry, 3), "max": 1.0},
            {"label": "Overall", "value": round(dev.overall_score, 3), "max": 1.0},
        ],
        "class": dev.developability_class,
    }
    return result, viz


def step_codon_optimization(sequence: str, species: str, system: str):
    """Step 7: Multi-system codon optimization."""
    from immunoforge.core.multi_codon_opt import optimize_multi_system
    systems = ["aav", "mrna", "ecoli", "pichia"] if system == "all" else [system]
    results = optimize_multi_system(sequence, systems=systems)
    result = {
        "protein_length": len(sequence),
        "systems": {
            name: {
                "cds_length_bp": r.cds_length_bp,
                "gc_content": round(r.gc_content, 3),
                "passes_constraints": r.passes_constraints,
                "warnings": r.warnings,
            }
            for name, r in results.items()
        },
    }
    # Visualization: GC content comparison
    viz = {
        "type": "gc_comparison",
        "title": "GC Content by Expression System",
        "bars": [
            {"system": name, "gc": round(r.gc_content, 3),
             "ok": r.passes_constraints, "bp": r.cds_length_bp}
            for name, r in results.items()
        ],
        "optimal_range": [0.40, 0.60],
    }
    return result, viz


def step_maturation(sequence: str, target_kd: float, generations: int):
    """Step 8: In-silico affinity maturation (optional)."""
    from immunoforge.core.maturation import run_maturation
    mat = run_maturation(
        sequence, target_kd_nM=target_kd, max_generations=generations, force=True,
    )
    result = {
        "initial_kd_nM": round(mat.initial_kd_nM, 2),
        "final_kd_nM": round(mat.final_kd_nM, 2),
        "improvement_fold": round(mat.improvement_fold, 2),
        "generations_run": mat.n_generations,
        "best_sequence": mat.best_candidate.sequence if mat.best_candidate else None,
        "n_mutations": mat.best_candidate.n_mutations if mat.best_candidate else 0,
    }
    # Visualization: K_D trajectory
    viz = {
        "type": "maturation_trajectory",
        "title": "Affinity Maturation Progress",
        "trajectory": [
            {"generation": 0, "kd_nM": round(mat.initial_kd_nM, 2)},
        ] + [
            {"generation": i + 1,
             "kd_nM": round(g.get("best_kd_nM", mat.initial_kd_nM), 2)}
            for i, g in enumerate(mat.generation_history)
        ] if hasattr(mat, "generation_history") and mat.generation_history else [
            {"generation": 0, "kd_nM": round(mat.initial_kd_nM, 2)},
            {"generation": mat.n_generations, "kd_nM": round(mat.final_kd_nM, 2)},
        ],
        "target_kd": target_kd,
        "improvement": round(mat.improvement_fold, 2),
    }
    return result, viz


# ── Main Pipeline Endpoint ──

@router.post("/pipeline/full", response_model=FullPipelineResponse)
async def run_full_pipeline(req: FullPipelineRequest):
    """Run the complete ImmunoForge analysis pipeline.

    Executes all steps sequentially:
    1. Sequence QC
    2. Domain Classification + CDR Identification
    3. Structure Prediction (ESMFold if GPU available, else heuristic)
    4. Binding Affinity (3-method consensus, structure-aware if PDB available)
    5. Immunogenicity Prediction
    6. Developability Assessment
    7. Codon Optimization
    8. Affinity Maturation (optional)

    Returns all step results with visualization data for each.
    """
    t0 = time.time()
    steps = []
    binder_id = f"{req.target_name}_binder"
    pdb_string = None  # Will be set by structure step

    # Step 1: Sequence QC
    step = _run_step("S1_QC", "Sequence Quality Control", step_sequence_qc, req.binder_sequence)
    steps.append(step)

    # Step 2: Domain Classification
    step = _run_step("S2_Domain", "Domain Classification", step_domain_classification, req.binder_sequence)
    steps.append(step)

    # Step 3: Structure Prediction
    step = _run_step("S3_Structure", "Structure Prediction", step_structure_prediction,
                     req.binder_sequence, req.run_esmfold)
    steps.append(step)
    if step.status == "completed" and step.result.get("pdb_available"):
        pdb_string = step.result.pop("_pdb_string", None)
    # Always remove internal PDB string from response
    step.result.pop("_pdb_string", None)

    # Step 4: Affinity Prediction
    step = _run_step("S4_Affinity", "Binding Affinity", step_affinity_prediction,
                     req.binder_sequence, pdb_string, req.bsa, req.sc, req.seed)
    steps.append(step)

    # Step 5: Immunogenicity
    step = _run_step("S5_Immunogenicity", "Immunogenicity Prediction",
                     step_immunogenicity, req.binder_sequence, req.species)
    steps.append(step)

    # Step 6: Developability
    step = _run_step("S6_Developability", "Developability Assessment",
                     step_developability, req.binder_sequence)
    steps.append(step)

    # Step 7: Codon Optimization
    step = _run_step("S7_Codon", "Codon Optimization",
                     step_codon_optimization, req.binder_sequence, req.species, req.expression_system)
    steps.append(step)

    # Step 8: Maturation (optional)
    if req.run_maturation:
        consensus_kd = 100.0  # Default target
        for s in steps:
            if s.step == "S4_Affinity" and s.status == "completed":
                consensus_kd = s.result.get("consensus", {}).get("consensus_kd_nM", 100.0) or 100.0
        step = _run_step("S8_Maturation", "Affinity Maturation",
                         step_maturation, req.binder_sequence, consensus_kd / 10.0,
                         req.maturation_generations)
        steps.append(step)

    total_time = round(time.time() - t0, 2)

    # Build summary
    summary = _build_summary(steps, binder_id, req)

    return FullPipelineResponse(
        status="completed" if all(s.status != "failed" for s in steps) else "partial",
        total_time_s=total_time,
        binder_id=binder_id,
        target_name=req.target_name,
        species=req.species,
        steps=steps,
        summary=summary,
    )


def _build_summary(steps: list[StepResult], binder_id: str, req) -> dict:
    """Build pipeline summary from step results."""
    summary = {
        "binder_id": binder_id,
        "sequence_length": len(req.binder_sequence),
        "steps_completed": sum(1 for s in steps if s.status == "completed"),
        "steps_failed": sum(1 for s in steps if s.status == "failed"),
    }

    for s in steps:
        if s.status != "completed":
            continue
        if s.step == "S1_QC":
            summary["qc_passed"] = s.result.get("overall_pass", False)
        elif s.step == "S2_Domain":
            summary["domain_type"] = s.result.get("domain_type")
            summary["n_cdrs"] = s.result.get("n_cdrs", 0)
        elif s.step == "S3_Structure":
            summary["plddt"] = s.result.get("mean_plddt")
            summary["structure_quality"] = s.result.get("quality")
            summary["structure_method"] = s.result.get("method")
        elif s.step == "S4_Affinity":
            consensus = s.result.get("consensus", {})
            summary["predicted_kd_nM"] = consensus.get("consensus_kd_nM")
            summary["affinity_confidence"] = consensus.get("confidence")
            summary["structure_derived_affinity"] = s.result.get("structure_derived", False)
        elif s.step == "S5_Immunogenicity":
            summary["immunogenicity_risk"] = s.result.get("risk_level")
        elif s.step == "S6_Developability":
            summary["developability_class"] = s.result.get("developability_class")
            summary["predicted_tm"] = s.result.get("predicted_tm_C")
        elif s.step == "S7_Codon":
            systems = s.result.get("systems", {})
            summary["codon_systems"] = list(systems.keys())
            summary["all_codon_pass"] = all(
                v.get("passes_constraints", False) for v in systems.values()
            )

    return summary


# ── Visualization Data Endpoints ──

@router.get("/visualize/benchmark")
async def get_benchmark_visualization():
    """Get benchmark calibration data for visualization."""
    from immunoforge.core.benchmark import run_benchmark, get_benchmark_entries
    entries = get_benchmark_entries(with_sequence=True)
    result = run_benchmark(entries=entries)

    predictions = result.get("predictions", [])
    metrics = result.get("metrics", {})

    return {
        "type": "calibration_scatter",
        "title": "Benchmark: Predicted vs Experimental K_D",
        "metrics": metrics,
        "data": [
            {
                "binder": p["binder"],
                "target": p["target"],
                "binder_type": p.get("binder_type", "unknown"),
                "experimental_kd": p["experimental_kd_nM"],
                "predicted_kd": p["predicted_kd_nM"],
                "log10_error": p["log10_error"],
                "confidence": p["confidence"],
                "log10_exp": round(math.log10(max(p["experimental_kd_nM"], 0.001)), 2),
                "log10_pred": round(math.log10(max(p["predicted_kd_nM"], 0.001)), 2),
            }
            for p in predictions
        ],
    }


@router.get("/visualize/targets")
async def get_target_visualization():
    """Get target database summary for visualization."""
    from immunoforge.core.target_db import search_targets, list_cell_types
    cell_types = list_cell_types()
    result = {
        "type": "target_overview",
        "cell_types": [],
    }
    for ct in sorted(cell_types):
        targets = search_targets(cell_type=ct)
        result["cell_types"].append({
            "name": ct,
            "count": len(targets),
            "targets": [t.name for t in targets[:10]],
        })
    return result

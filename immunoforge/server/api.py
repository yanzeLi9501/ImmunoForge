"""
REST API routes for ImmunoForge server.

Endpoints:
  POST /api/pipeline/run       — Start a pipeline run
  GET  /api/pipeline/status    — Get pipeline run status
  GET  /api/pipeline/results   — Get latest results
  GET  /api/targets            — Search target database
  GET  /api/targets/{name}     — Get specific target
  POST /api/qc                 — Run sequence QC
  POST /api/affinity           — Run affinity analysis
  POST /api/codon-optimize     — Run codon optimization
  GET  /api/config             — Get current configuration
"""

import json
import logging
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from immunoforge.core.target_db import (
    search_targets,
    get_target_by_name,
    list_cell_types,
    list_species,
    get_benchmark_targets,
)
from immunoforge.core.sequence_qc import run_full_qc, batch_qc
from immunoforge.core.affinity import run_affinity_analysis
from immunoforge.core.codon_opt import full_codon_optimization
from immunoforge.core.utils import load_config
from immunoforge.pipeline.runner import run_pipeline, ALL_STEPS, EXTENDED_STEPS

logger = logging.getLogger(__name__)
router = APIRouter()

# Pipeline execution state
_pipeline_state = {"status": "idle", "result": None}
_pipeline_lock = threading.Lock()


# ── Request models ──
class PipelineRequest(BaseModel):
    steps: list[str] | None = None
    species: str = "mouse"
    config_overrides: dict | None = None


class SequenceQCRequest(BaseModel):
    sequences: list[dict]  # [{"id": "...", "sequence": "..."}]


class AffinityRequest(BaseModel):
    sequence: str
    bsa: float = 1200.0
    sc: float = 0.65


class CodonOptRequest(BaseModel):
    sequence: str
    species: str = "mouse"
    system: str = "vaccinia"
    signal_peptide: str = "il2_leader"


class TargetSearchRequest(BaseModel):
    cell_type: str | None = None
    species: str | None = None
    name: str | None = None
    min_benchmark: int = 0


# ── Pipeline endpoints ──
@router.post("/pipeline/run")
async def start_pipeline(req: PipelineRequest):
    """Start a pipeline run in background."""
    with _pipeline_lock:
        if _pipeline_state["status"] == "running":
            raise HTTPException(400, "Pipeline already running")

    config = load_config()

    # Apply overrides
    if req.species:
        config.setdefault("species", {})["default"] = req.species
    if req.config_overrides:
        config.update(req.config_overrides)

    steps = req.steps or ALL_STEPS

    def _run():
        with _pipeline_lock:
            _pipeline_state["status"] = "running"
            _pipeline_state["result"] = None
        try:
            result = run_pipeline(config, steps=steps)
            with _pipeline_lock:
                _pipeline_state["status"] = "completed"
                _pipeline_state["result"] = result
        except Exception as e:
            with _pipeline_lock:
                _pipeline_state["status"] = "failed"
                _pipeline_state["result"] = {"error": str(e)}

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {"message": "Pipeline started", "steps": steps, "species": req.species}


@router.get("/pipeline/status")
async def pipeline_status():
    """Get current pipeline status."""
    with _pipeline_lock:
        return {
            "status": _pipeline_state["status"],
            "has_result": _pipeline_state["result"] is not None,
        }


@router.get("/pipeline/results")
async def pipeline_results():
    """Get latest pipeline results."""
    # Try from state first
    with _pipeline_lock:
        if _pipeline_state["result"]:
            return _pipeline_state["result"]

    # Try from file
    config = load_config()
    output_dir = Path(config.get("paths", {}).get("output_dir", "outputs"))
    summary_path = output_dir / "pipeline_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)

    return {"status": "no_results"}


@router.get("/pipeline/step/{step_key}")
async def get_step_result(step_key: str):
    """Get result of a specific pipeline step."""
    config = load_config()
    output_dir = Path(config.get("paths", {}).get("output_dir", "outputs"))

    filename_map = {
        "B1": "B1_target_prep.json",
        "B3b": "B3b_sequence_qc.json",
        "B4": "B4_structure_validation.json",
        "B5": "B5_binding_affinity.json",
        "B6": "B6_candidate_ranking.json",
        "B7": "B7_codon_optimization.json",
        "B8": "pipeline_summary.json",
    }

    filename = filename_map.get(step_key)
    if not filename:
        raise HTTPException(404, f"Unknown step: {step_key}")

    path = output_dir / filename
    if not path.exists():
        raise HTTPException(404, f"No results for step {step_key}")

    with open(path) as f:
        return json.load(f)


# ── Target database endpoints ──
@router.get("/targets")
async def query_targets(
    cell_type: str | None = None,
    species: str | None = None,
    name: str | None = None,
    min_benchmark: int = 0,
):
    """Search the immune target database."""
    results = search_targets(cell_type, species, name, min_benchmark)
    return {
        "count": len(results),
        "targets": [
            {
                "name": t.name,
                "gene": t.gene,
                "cell_type": t.cell_type,
                "species": t.species,
                "uniprot": t.uniprot,
                "pdb_ids": t.pdb_ids,
                "benchmark_value": t.benchmark_value,
                "description": t.description,
                "known_binders": t.known_binders,
            }
            for t in results
        ],
    }


@router.get("/targets/cell-types")
async def get_cell_types():
    return {"cell_types": list_cell_types()}


@router.get("/targets/species")
async def get_species():
    return {"species": list_species()}


@router.get("/targets/benchmarks")
async def get_benchmarks():
    results = get_benchmark_targets()
    return {
        "count": len(results),
        "targets": [{"name": t.name, "gene": t.gene, "cell_type": t.cell_type} for t in results],
    }


@router.get("/targets/{name}")
async def get_target(name: str):
    t = get_target_by_name(name)
    if not t:
        raise HTTPException(404, f"Target not found: {name}")
    return {
        "name": t.name,
        "gene": t.gene,
        "cell_type": t.cell_type,
        "species": t.species,
        "uniprot": t.uniprot,
        "pdb_ids": t.pdb_ids,
        "alphafold_id": t.alphafold_id,
        "residue_range": t.residue_range,
        "hotspot_residues": t.hotspot_residues,
        "benchmark_value": t.benchmark_value,
        "known_binders": t.known_binders,
        "description": t.description,
    }


# ── Analysis endpoints ──
@router.post("/qc")
async def run_qc(req: SequenceQCRequest):
    """Run sequence QC on provided sequences."""
    sequences = [(s.get("id", f"seq_{i}"), s["sequence"]) for i, s in enumerate(req.sequences)]
    result = batch_qc(sequences)
    return result


@router.post("/affinity")
async def run_affinity(req: AffinityRequest):
    """Run binding affinity analysis on a single sequence."""
    result = run_affinity_analysis(req.sequence, req.bsa, req.sc)
    return result


@router.post("/codon-optimize")
async def run_codon_opt(req: CodonOptRequest):
    """Run codon optimization."""
    result = full_codon_optimization(
        req.sequence, species=req.species, system=req.system,
        signal_peptide=req.signal_peptide,
    )
    return result


# ── Configuration ──
@router.get("/config")
async def get_config():
    """Get current pipeline configuration (non-sensitive fields)."""
    config = load_config()
    return {
        "project": config.get("project"),
        "species": config.get("species"),
        "rfdiffusion": config.get("rfdiffusion"),
        "proteinmpnn": config.get("proteinmpnn"),
        "ranking": config.get("ranking"),
    }


# ── Extension endpoints ──

class ImmunogenicityRequest(BaseModel):
    sequence: str
    species: str = "human"


class DevelopabilityRequest(BaseModel):
    sequence: str


class MaturationRequest(BaseModel):
    sequence: str
    target_kd_nM: float = 1.0
    max_generations: int = 5
    force: bool = True


class StructureValidationRequest(BaseModel):
    sequence: str
    method: str = "auto"


class MultiCodonRequest(BaseModel):
    sequence: str
    systems: list[str] | None = None


class BenchmarkRequest(BaseModel):
    cell_type: str | None = None
    binder_type: str | None = None


class BispecificRequest(BaseModel):
    binder1_id: str
    binder1_sequence: str
    binder2_id: str
    binder2_sequence: str


class PeptideEngagerRequest(BaseModel):
    cd3_target: str = "mCD3E"
    clec9a_target: str = "mCLEC9A"
    linker_type: str = "flexible"
    seed: int = 42


@router.post("/structure-validate")
async def run_structure_validation(req: StructureValidationRequest):
    """Run deep structure validation."""
    from immunoforge.core.structure_validation import validate_structure
    result = validate_structure(req.sequence, method=req.method)
    return {
        "method": result.method,
        "mean_plddt": round(result.mean_plddt, 1),
        "quality": result.quality,
        "ptm": result.ptm,
    }


@router.post("/immunogenicity")
async def run_immunogenicity(req: ImmunogenicityRequest):
    """Run immunogenicity prediction with PSSM-based MHC binding."""
    from immunoforge.core.immunogenicity import predict_immunogenicity
    result = predict_immunogenicity(req.sequence, species=req.species)
    return {
        "immunogenicity_score": round(result.immunogenicity_score, 3),
        "risk_level": result.risk_level,
        "epitope_density": round(result.epitope_density, 4),
        "n_epitopes": len(result.epitopes),
        "hotspot_regions": result.hotspot_regions,
        "method": result.details.get("method", "unknown"),
        "humanness_score": result.details.get("humanness_score"),
        "n_strong_class_i": result.details.get("n_strong_class_i", 0),
    }


@router.post("/maturation")
async def run_maturation_api(req: MaturationRequest):
    """Run in-silico affinity maturation.

    Requires a structure prediction backend (ESMFold/AF2) for meaningful
    results. Set force=true to override (results will carry a warning).
    """
    from immunoforge.core.maturation import run_maturation, MaturationStructureWarning, _has_structure_backend
    has_structure = _has_structure_backend()
    result = run_maturation(
        req.sequence, target_kd_nM=req.target_kd_nM,
        max_generations=req.max_generations,
        force=req.force,
    )
    response = {
        "initial_kd_nM": round(result.initial_kd_nM, 2),
        "final_kd_nM": round(result.final_kd_nM, 2),
        "improvement_fold": round(result.improvement_fold, 2),
        "generations_run": result.n_generations,
        "best_sequence": result.best_candidate.sequence if result.best_candidate else None,
        "requires_structure": MaturationStructureWarning.requires_structure,
        "has_structure_backend": has_structure,
    }
    if not has_structure:
        response["warning"] = MaturationStructureWarning.message
    return response


@router.post("/developability")
async def run_developability(req: DevelopabilityRequest):
    """Run developability assessment."""
    from immunoforge.core.developability import run_developability_assessment
    result = run_developability_assessment(req.sequence)
    return {
        "overall_score": round(result.overall_score, 3),
        "developability_class": result.developability_class,
        "solubility": round(result.solubility.camsol_score, 3),
        "predicted_tm_C": round(result.thermal_stability.estimated_tm_celsius, 1),
        "flags": result.flags,
    }


@router.post("/multi-codon")
async def run_multi_codon(req: MultiCodonRequest):
    """Run multi-expression system codon optimization."""
    from immunoforge.core.multi_codon_opt import optimize_multi_system
    results = optimize_multi_system(req.sequence, systems=req.systems)
    return {
        "systems": {
            name: {
                "cds_length_bp": r.cds_length_bp,
                "gc_content": r.gc_content,
                "cpg_count": r.cpg_count,
                "cpg_oe_ratio": r.cpg_oe_ratio,
                "passes_constraints": r.passes_constraints,
                "warnings": r.warnings,
            }
            for name, r in results.items()
        },
    }


@router.post("/benchmark")
async def run_benchmark_api(req: BenchmarkRequest):
    """Run benchmark validation."""
    from immunoforge.core.benchmark import run_benchmark, get_benchmark_entries
    entries = get_benchmark_entries(
        cell_type=req.cell_type, binder_type=req.binder_type, with_sequence=True,
    )
    result = run_benchmark(entries=entries)
    return result


@router.post("/bispecific")
async def design_bispecific(req: BispecificRequest):
    """Design bispecific construct with linker."""
    from immunoforge.core.linker_design import design_bispecific
    result = design_bispecific(
        req.binder1_id, req.binder1_sequence,
        req.binder2_id, req.binder2_sequence,
    )
    return {
        "binder1_id": result.binder1_id,
        "binder2_id": result.binder2_id,
        "linker_type": result.linker.linker_type,
        "linker_sequence": result.linker.sequence,
        "linker_length": result.linker.length_aa,
        "full_construct": result.fusion_seq,
        "total_length_aa": result.total_length,
        "estimated_mw_Da": result.estimated_mw_da,
    }


@router.post("/peptide-engager")
async def design_peptide_engager(req: PeptideEngagerRequest):
    """Design de-novo peptide bispecific engager and compare with scFv.

    Runs the full peptide engager workflow:
    1. Score RFdiffusion+ProteinMPNN-designed peptide binders for both targets.
    2. Select top candidate per target and assemble bispecific with linker.
    3. Return head-to-head comparison against scFv bispecific (OKT3 VH × 10B12 VH).
    """
    from immunoforge.core.peptide_engager import run_peptide_engager_design
    return run_peptide_engager_design(
        cd3_target=req.cd3_target,
        clec9a_target=req.clec9a_target,
        linker_type=req.linker_type,
        seed=req.seed,
    )


@router.get("/pipeline/extended-steps")
async def get_extended_steps():
    """List all available pipeline steps including extensions."""
    from immunoforge.pipeline.runner import STEP_REGISTRY
    return {
        "core_steps": ALL_STEPS,
        "extended_steps": EXTENDED_STEPS,
        "step_details": {
            k: {"description": v[1]} for k, v in STEP_REGISTRY.items()
        },
    }

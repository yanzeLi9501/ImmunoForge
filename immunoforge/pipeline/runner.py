"""
Pipeline Runner — Orchestrates the full ImmunoForge pipeline.

Steps:
  B1   Target Preparation (+ auto-hotspot prediction)
  B2   RFdiffusion Backbone Generation
  B3   ProteinMPNN Sequence Design
  B3a  Multi-Stage Sequence Optimization (BindCraft-inspired, optional)
  B3b  Sequence Quality Control
  B4   Structure Validation (ESMFold + ESM-2 dual evaluation)
  B5   Binding Affinity Prediction
  B6   Candidate Ranking
  B7   Codon Optimization
  B8   Deliverables & Report
  B9a  AF3-Multimer Ternary Complex Validation (Boltz-2)
  B9b  Conditional Affinity Maturation (K_D > threshold → MPNN T=0.3)
"""

import importlib
import json
import logging
import time
import traceback
from pathlib import Path

from immunoforge.core.utils import ensure_dirs, save_json

logger = logging.getLogger(__name__)

# Step registry: step_key → (module_path, description)
STEP_REGISTRY = {
    "B1":  ("immunoforge.pipeline.steps.B1_target_prep",       "靶点准备与表位定义"),
    "B2":  ("immunoforge.pipeline.steps.B2_rfdiffusion",       "RFdiffusion骨架生成"),
    "B3":  ("immunoforge.pipeline.steps.B3_sequence_design",   "ProteinMPNN序列设计"),
    "B3a": ("immunoforge.pipeline.steps.B3a_sequence_optimization", "多阶段序列优化 (BindCraft)"),
    "B3b": ("immunoforge.pipeline.steps.B3b_sequence_qc",      "序列质量控制"),
    "B4":  ("immunoforge.pipeline.steps.B4_structure_validation", "结构验证 (ESMFold+ESM-2双重评估)"),
    "B5":  ("immunoforge.pipeline.steps.B5_binding_affinity",  "结合亲和力预测"),
    "B6":  ("immunoforge.pipeline.steps.B6_candidate_ranking", "候选排序"),
    "B7":  ("immunoforge.pipeline.steps.B7_codon_optimization", "密码子优化"),
    "B8":  ("immunoforge.pipeline.steps.B8_deliverables",      "交付物与报告"),
    # Extension steps
    "B4b": ("immunoforge.pipeline.steps.B4b_structure_deep",   "深度结构验证 (ESMFold/AF2)"),
    "B5b": ("immunoforge.pipeline.steps.B5b_immunogenicity",   "免疫原性预测"),
    "B5c": ("immunoforge.pipeline.steps.B5c_maturation",       "亲和力成熟"),
    "B6b": ("immunoforge.pipeline.steps.B6b_developability",   "可开发性评估"),
    "B7b": ("immunoforge.pipeline.steps.B7b_multi_codon",      "多表达系统密码子优化"),
    "B9":  ("immunoforge.pipeline.steps.B9_benchmark",         "基准验证"),
    "B9a": ("immunoforge.pipeline.steps.B9a_af3_validation",   "AF3三元复合物验证 (Boltz-2)"),
    "B9b": ("immunoforge.pipeline.steps.B9b_affinity_maturation", "条件性亲和力成熟 (K_D>阈值)"),
    "B4c": ("immunoforge.pipeline.steps.B4c_af2_multimer_filter", "AF2-Multimer正交过滤 (ColabFold)"),
}

ALL_STEPS = ["B1", "B2", "B3", "B3b", "B4", "B5", "B6", "B7", "B8"]
ALL_STEPS_WITH_OPT = ["B1", "B2", "B3", "B3a", "B3b", "B4", "B5", "B6", "B7", "B8"]
FULL_PIPELINE = ALL_STEPS_WITH_OPT + ["B9a", "B9b"]
EXTENDED_STEPS = ALL_STEPS_WITH_OPT + ["B4c", "B4b", "B5b", "B5c", "B6b", "B7b", "B9", "B9a", "B9b"]


def run_step(step_key: str, config: dict) -> dict:
    """Run a single pipeline step."""
    if step_key not in STEP_REGISTRY:
        return {"step": step_key, "status": "unknown_step"}

    module_path, desc = STEP_REGISTRY[step_key]
    logger.info(f"\n{'='*70}")
    logger.info(f"  {step_key}: {desc}")
    logger.info(f"{'='*70}")

    t0 = time.time()
    try:
        mod = importlib.import_module(module_path)
        result = mod.main(config)
        elapsed = time.time() - t0
        logger.info(f"  {step_key} completed in {elapsed:.1f}s")
        return {
            "step": step_key,
            "description": desc,
            "status": "completed",
            "time_s": round(elapsed, 1),
            "result": result,
        }
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"  {step_key} FAILED: {e}")
        logger.debug(traceback.format_exc())
        return {
            "step": step_key,
            "description": desc,
            "status": "failed",
            "time_s": round(elapsed, 1),
            "error": str(e),
        }


def run_pipeline(
    config: dict,
    steps: list[str] | None = None,
    stop_on_failure: bool = False,
) -> dict:
    """Run the full pipeline or selected steps.

    Args:
        config: Pipeline configuration dict.
        steps: List of step keys to run (default: all).
        stop_on_failure: Stop pipeline if a step fails.

    Returns:
        Pipeline summary dict with all step results.
    """
    if steps is None:
        # Use optimization pipeline if enabled in config
        opt_enabled = config.get("sequence_optimization", {}).get("enabled", False)
        af3_enabled = config.get("af3_validation", {}).get("enabled", False)
        mat_enabled = config.get("affinity_maturation", {}).get("enabled", False)
        if af3_enabled or mat_enabled:
            steps = FULL_PIPELINE
        elif opt_enabled:
            steps = ALL_STEPS_WITH_OPT
        else:
            steps = ALL_STEPS

    # Ensure output directories
    output_dir = Path(config.get("paths", {}).get("output_dir", "outputs"))
    logs_dir = Path(config.get("paths", {}).get("logs_dir", "outputs/logs"))
    ensure_dirs(output_dir, logs_dir)

    species = config.get("species", {}).get("default", "mouse")
    logger.info(f"{'='*70}")
    logger.info(f"  IMMUNOFORGE — De Novo Immune Cell Binder Design")
    logger.info(f"  Species: {species}")
    logger.info(f"  Steps: {steps}")
    logger.info(f"{'='*70}")

    results = []
    for step_key in steps:
        r = run_step(step_key, config)
        results.append(r)

        if r["status"] == "failed" and stop_on_failure:
            logger.error(f"  Pipeline stopped at {step_key}")
            break

    completed = sum(1 for r in results if r["status"] == "completed")
    failed = sum(1 for r in results if r["status"] == "failed")
    total_time = sum(r.get("time_s", 0) for r in results)

    summary = {
        "pipeline": "ImmunoForge",
        "species": species,
        "steps_requested": steps,
        "completed": completed,
        "failed": failed,
        "total_time_s": round(total_time, 1),
        "step_results": [
            {k: v for k, v in r.items() if k != "result"}
            for r in results
        ],
    }

    save_json(summary, output_dir / "pipeline_run_summary.json")

    logger.info(f"\n{'='*70}")
    logger.info(
        f"  PIPELINE COMPLETE: {completed}/{len(results)} steps "
        f"| {failed} failed | {total_time:.1f}s"
    )
    logger.info(f"{'='*70}")

    return summary

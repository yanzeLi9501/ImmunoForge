"""
B8: Deliverables — Final summary, synthesis order, and report generation.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from immunoforge.core.utils import save_json, ensure_dirs

logger = logging.getLogger(__name__)


def generate_synthesis_order(ranked: list[dict], output_path: str) -> None:
    """Generate CSV synthesis order for gene synthesis vendors."""
    lines = [
        "Order_ID,Gene_Name,Target,Length_aa,MW_Da,pI,Expression_System,Priority"
    ]
    for i, c in enumerate(ranked, 1):
        lines.append(
            f"IF-{i:03d},"
            f"{c.get('id', 'unknown')},"
            f"{c.get('target', 'N/A')},"
            f"{c.get('length', len(c.get('sequence', '')))},"
            f"{c.get('mw_da', 'N/A')},"
            f"{c.get('pI', 'N/A')},"
            f"Vaccinia_TK,"
            f"{'HIGH' if i <= 3 else 'MEDIUM' if i <= 5 else 'LOW'}"
        )
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def main(config: dict) -> dict:
    """Execute B8: Generate deliverables and final report."""
    logger.info("  B8: Deliverables & Final Report")

    output_dir = Path(config.get("paths", {}).get("output_dir", "outputs"))

    # Collect all pipeline results
    steps = {}
    for step_file in [
        "B1_target_prep.json",
        "B3b_sequence_qc.json",
        "B4_structure_validation.json",
        "B5_binding_affinity.json",
        "B6_candidate_ranking.json",
        "B7_codon_optimization.json",
    ]:
        path = output_dir / step_file
        if path.exists():
            with open(path) as f:
                steps[step_file.replace(".json", "")] = json.load(f)

    # Generate synthesis order
    b6 = steps.get("B6_candidate_ranking", {})
    ranked = b6.get("ranked", [])
    synthesis_top = ranked[:config.get("ranking", {}).get("synthesis_top_n", 5)]

    if synthesis_top:
        ensure_dirs(output_dir / "synthesis_ready")
        generate_synthesis_order(
            synthesis_top, str(output_dir / "synthesis_ready" / "synthesis_order.csv")
        )

    # Build pipeline summary
    species = config.get("species", {}).get("default", "mouse")
    qc = steps.get("B3b_sequence_qc", {})
    b7 = steps.get("B7_codon_optimization", {})

    summary = {
        "project": "ImmunoForge",
        "version": config.get("project", {}).get("version", "0.1.0"),
        "species": species,
        "generated_at": datetime.now().isoformat(),
        "pipeline_funnel": {
            "rfdiffusion_backbones": config.get("rfdiffusion", {}).get("n_designs_per_target", 50) * 2,
            "proteinmpnn_sequences": qc.get("total", 0),
            "qc_passed": qc.get("n_passed", 0),
            "qc_pass_rate": qc.get("pass_rate", 0),
            "structure_validated": steps.get("B4_structure_validation", {}).get("n_validated", 0),
            "affinity_scored": steps.get("B5_binding_affinity", {}).get("n_analyzed", 0),
            "top_ranked": len(ranked),
            "synthesis_ready": len(synthesis_top),
        },
        "top_candidates": [
            {
                "rank": c.get("rank"),
                "id": c.get("id"),
                "target": c.get("target"),
                "score": c.get("composite_score"),
                "kd_nM": c.get("kd_nM"),
                "plddt": c.get("plddt"),
                "length": c.get("length"),
                "mw_da": c.get("mw_da"),
                "pI": c.get("pI"),
            }
            for c in synthesis_top
        ],
        "codon_optimization": {
            "species": species,
            "expression_system": b7.get("expression_system", "vaccinia"),
            "n_optimized": b7.get("n_optimized", 0),
        },
    }

    save_json(summary, output_dir / "pipeline_summary.json")
    logger.info(f"  Pipeline summary saved to {output_dir / 'pipeline_summary.json'}")
    logger.info(f"  B8 complete: {len(synthesis_top)} synthesis-ready candidates")

    return summary

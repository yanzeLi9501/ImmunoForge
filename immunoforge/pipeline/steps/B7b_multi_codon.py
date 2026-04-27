"""B7b — Multi-Expression System Codon Optimization."""

import logging
from immunoforge.core.multi_codon_opt import optimize_multi_system, generate_comparison_table

logger = logging.getLogger(__name__)


def main(config: dict) -> dict:
    """Run codon optimization for multiple expression systems."""
    sequences = config.get("_sequences", [])
    systems = config.get("multi_codon", {}).get("systems", ["aav", "mRNA", "ecoli", "pichia"])

    if not sequences:
        return {"status": "skipped", "reason": "no sequences provided"}

    all_results = []
    for seq in sequences:
        results = optimize_multi_system(seq, systems=systems)
        all_results.append({
            "protein_length": len(seq),
            "systems": {
                name: {
                    "cds_length_bp": r.cds_length_bp,
                    "gc_content": r.gc_content,
                    "cpg_count": r.cpg_count,
                    "passes_constraints": r.passes_constraints,
                    "warnings": r.warnings,
                }
                for name, r in results.items()
            },
        })

    logger.info("Multi-codon opt: %d sequences × %d systems", len(sequences), len(systems))

    return {
        "status": "completed",
        "n_sequences": len(sequences),
        "systems": systems,
        "results": all_results,
    }

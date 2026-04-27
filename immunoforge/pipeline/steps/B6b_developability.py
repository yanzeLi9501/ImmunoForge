"""B6b — Developability Assessment."""

import logging
from immunoforge.core.developability import run_developability_assessment, batch_developability

logger = logging.getLogger(__name__)


def main(config: dict) -> dict:
    """Run developability scoring on candidate sequences."""
    sequences = config.get("_sequences", [])

    if not sequences:
        return {"status": "skipped", "reason": "no sequences provided"}

    results = batch_developability(sequences)
    class_counts = {}
    for r in results:
        cls = r.get("developability_class", "unknown")
        class_counts[cls] = class_counts.get(cls, 0) + 1

    logger.info("Developability: %s", class_counts)

    return {
        "status": "completed",
        "n_sequences": len(sequences),
        "class_distribution": class_counts,
        "results": [
            {
                "overall_score": r.get("overall_score"),
                "developability_class": r.get("developability_class"),
                "camsol_score": r.get("camsol_score"),
                "estimated_tm": r.get("estimated_tm"),
                "flags": r.get("flags", []),
            }
            for r in results
        ],
    }

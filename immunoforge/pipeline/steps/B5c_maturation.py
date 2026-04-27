"""B5c — In-Silico Affinity Maturation."""

import logging
from immunoforge.core.maturation import run_maturation, batch_maturation

logger = logging.getLogger(__name__)


def main(config: dict) -> dict:
    """Run in-silico affinity maturation on top candidates."""
    sequences = config.get("_sequences", [])
    mat_cfg = config.get("maturation", {})
    target_kd = mat_cfg.get("target_kd_nM", 1.0)
    max_gen = mat_cfg.get("max_generations", 5)

    if not sequences:
        return {"status": "skipped", "reason": "no sequences provided"}

    results = batch_maturation(
        sequences, target_kd_nM=target_kd, max_generations=max_gen,
    )

    improved = sum(1 for r in results if r.get("improvement_fold", 0) > 1.5)
    logger.info("Maturation: %d/%d improved >1.5x", improved, len(results))

    return {
        "status": "completed",
        "n_sequences": len(sequences),
        "n_improved": improved,
        "results": results,
    }

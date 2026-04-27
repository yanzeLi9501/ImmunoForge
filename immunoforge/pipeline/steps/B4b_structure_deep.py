"""B4b — Deep Structure Validation (ESMFold / AlphaFold2-multimer)."""

import logging
from immunoforge.core.structure_validation import validate_structure, batch_validate

logger = logging.getLogger(__name__)


def main(config: dict) -> dict:
    """Run deep structure validation on designed sequences."""
    sequences = config.get("_sequences", [])
    method = config.get("structure_validation", {}).get("method", "auto")

    if not sequences:
        return {"status": "skipped", "reason": "no sequences provided"}

    results = batch_validate(sequences, method=method)
    high = sum(1 for r in results if r.quality == "HIGH")
    med = sum(1 for r in results if r.quality == "MEDIUM")
    low = sum(1 for r in results if r.quality == "LOW")

    logger.info("Structure validation: %d HIGH / %d MEDIUM / %d LOW", high, med, low)

    return {
        "status": "completed",
        "n_sequences": len(sequences),
        "quality_distribution": {"HIGH": high, "MEDIUM": med, "LOW": low},
        "results": [
            {
                "method": r.method,
                "mean_plddt": round(r.mean_plddt, 1),
                "quality": r.quality,
            }
            for r in results
        ],
    }

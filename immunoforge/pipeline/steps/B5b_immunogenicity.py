"""B5b — Immunogenicity Prediction (T-cell epitope analysis)."""

import logging
from immunoforge.core.immunogenicity import predict_immunogenicity, batch_immunogenicity

logger = logging.getLogger(__name__)


def main(config: dict) -> dict:
    """Run immunogenicity prediction on designed sequences."""
    sequences = config.get("_sequences", [])
    species = config.get("species", {}).get("default", "human")

    if not sequences:
        return {"status": "skipped", "reason": "no sequences provided"}

    results = batch_immunogenicity(sequences, species=species)
    risk_counts = {"low": 0, "moderate": 0, "high": 0}
    for r in results:
        risk_counts[r.risk_level] = risk_counts.get(r.risk_level, 0) + 1

    logger.info("Immunogenicity: %s", risk_counts)

    return {
        "status": "completed",
        "n_sequences": len(sequences),
        "risk_distribution": risk_counts,
        "results": [
            {
                "immunogenicity_score": round(r.immunogenicity_score, 3),
                "risk_level": r.risk_level,
                "n_epitopes": len(r.epitope_hits),
                "epitope_density": round(r.epitope_density, 4),
            }
            for r in results
        ],
    }

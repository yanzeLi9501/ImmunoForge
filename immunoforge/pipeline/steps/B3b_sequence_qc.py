"""
B3b: Sequence Quality Control — Post-design filtering.

Wraps core/sequence_qc module into the pipeline step interface.
"""

import json
import logging
from pathlib import Path

from immunoforge.core.sequence_qc import batch_qc
from immunoforge.core.utils import save_json

logger = logging.getLogger(__name__)


def main(config: dict) -> dict:
    """Execute B3b: Sequence QC step."""
    logger.info("  B3b: Sequence Quality Control")

    output_dir = Path(config.get("paths", {}).get("output_dir", "outputs"))
    b3_path = output_dir / "B3_sequence_design.json"

    if not b3_path.exists():
        logger.warning("  B3 output not found, skipping QC")
        return {"status": "skipped", "reason": "no_b3_output"}

    with open(b3_path) as f:
        b3 = json.load(f)

    sequences = [
        (s["id"], s["sequence"])
        for s in b3.get("sequences", [])
    ]

    if not sequences:
        logger.warning("  No sequences to QC")
        return {"status": "skipped", "reason": "no_sequences"}

    logger.info(f"  Running QC on {len(sequences)} sequences...")
    result = batch_qc(sequences, config)

    logger.info(
        f"  QC complete: {result['n_passed']}/{result['total']} passed "
        f"(rate: {result['pass_rate']:.1%})"
    )
    if result["failure_reasons"]:
        logger.info(f"  Failure reasons: {result['failure_reasons']}")

    save_json(result, output_dir / "B3b_sequence_qc.json")
    return result

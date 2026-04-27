"""B9 — Benchmark Validation (retrodiction against known binders)."""

import logging
from immunoforge.core.benchmark import (
    run_benchmark,
    get_benchmark_entries,
    generate_benchmark_report,
)

logger = logging.getLogger(__name__)


def main(config: dict) -> dict:
    """Run benchmark validation against gold-standard entries."""
    bm_cfg = config.get("benchmark", {})
    cell_type = bm_cfg.get("cell_type")
    binder_type = bm_cfg.get("binder_type")
    seed = bm_cfg.get("seed", 42)

    entries = get_benchmark_entries(
        cell_type=cell_type,
        binder_type=binder_type,
        with_sequence=True,
    )

    if not entries:
        return {"status": "skipped", "reason": "no benchmark entries with sequences"}

    result = run_benchmark(entries=entries, seed=seed)

    report_md = generate_benchmark_report(result)
    logger.info(
        "Benchmark: %d entries, Spearman ρ = %s",
        result.get("n_entries", 0),
        result.get("metrics", {}).get("spearman_rho", "N/A"),
    )

    return {
        "status": "completed",
        "n_entries": result["n_entries"],
        "metrics": result["metrics"],
        "report_markdown": report_md,
    }

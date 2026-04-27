"""
B6: Candidate Ranking — Multi-criteria scoring & selection.

Integrates all upstream data to produce a ranked candidate list.
"""

import json
import logging
from pathlib import Path

from immunoforge.core.ranking import rank_candidates
from immunoforge.core.utils import save_json, save_fasta

logger = logging.getLogger(__name__)


def main(config: dict) -> dict:
    """Execute B6: Candidate Ranking."""
    logger.info("  B6: Candidate Ranking")

    output_dir = Path(config.get("paths", {}).get("output_dir", "outputs"))
    b5_path = output_dir / "B5_binding_affinity.json"

    if not b5_path.exists():
        logger.warning("  B5 output not found")
        return {"status": "skipped"}

    with open(b5_path) as f:
        b5 = json.load(f)

    scored = b5.get("scored", [])
    if not scored:
        return {"status": "skipped", "reason": "no_scored_candidates"}

    # Prepare candidate dicts for ranking
    candidates = []
    for s in scored:
        affinity = s.get("affinity", {})
        consensus = affinity.get("consensus", {})
        plddt_data = s.get("plddt", {})

        candidates.append({
            "id": s["id"],
            "sequence": s["sequence"],
            "target": s.get("target", "unknown"),
            "plddt": plddt_data.get("mean_plddt", 75) if isinstance(plddt_data, dict) else 75,
            "ddg": affinity.get("rosetta", {}).get("dg", -10),
            "kd_nM": consensus.get("consensus_kd_nM", 1000),
            "mpnn_score": s.get("mpnn_score", 1.0),
            "bsa": s.get("bsa", 1000),
            "sc": s.get("sc", 0.6),
        })

    ranking_cfg = config.get("ranking", {})
    top_n = ranking_cfg.get("top_n", 20)
    weights = ranking_cfg.get("weights")

    ranked = rank_candidates(candidates, top_n=top_n, weights=weights)

    # Save Top N FASTA
    fasta_records = [(r["id"], r["sequence"]) for r in ranked]
    save_fasta(fasta_records, str(output_dir / "top_candidates.fasta"))

    result = {
        "total_input": len(candidates),
        "top_n": len(ranked),
        "ranked": ranked,
    }

    save_json(result, output_dir / "B6_candidate_ranking.json")
    logger.info(f"  B6 complete: Top {len(ranked)} candidates selected")

    # Log top 5
    for r in ranked[:5]:
        logger.info(
            f"    #{r['rank']} {r['id']}: score={r['composite_score']:.4f} "
            f"K_D={r.get('kd_nM', 'N/A')} nM"
        )

    return result

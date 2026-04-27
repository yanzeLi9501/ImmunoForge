"""
B7: Codon Optimization & Expression Cassette Design.

Generates synthesis-ready DNA for top candidates.
"""

import json
import logging
from pathlib import Path

from immunoforge.core.codon_opt import full_codon_optimization
from immunoforge.core.utils import save_json, save_fasta, ensure_dirs

logger = logging.getLogger(__name__)


def main(config: dict) -> dict:
    """Execute B7: Codon optimization & cassette design."""
    logger.info("  B7: Codon Optimization & Expression Cassette")

    output_dir = Path(config.get("paths", {}).get("output_dir", "outputs"))
    b6_path = output_dir / "B6_candidate_ranking.json"

    if not b6_path.exists():
        logger.warning("  B6 output not found")
        return {"status": "skipped"}

    with open(b6_path) as f:
        b6 = json.load(f)

    ranked = b6.get("ranked", [])
    if not ranked:
        return {"status": "skipped", "reason": "no_ranked_candidates"}

    # Config
    codon_cfg = config.get("codon_optimization", {})
    species = config.get("species", {}).get("default", "mouse")
    system = "vaccinia"  # default
    signal_peptide = "il2_leader"
    gc_range = codon_cfg.get("gc_content_target", [0.48, 0.52])
    target_gc = sum(gc_range) / 2

    synthesis_n = config.get("ranking", {}).get("synthesis_top_n", 5)
    top_synthesis = ranked[:synthesis_n]

    # Output directories
    synthesis_dir = output_dir / "synthesis_ready"
    top_dir = output_dir / "Top_synthesis"
    ensure_dirs(synthesis_dir, top_dir)

    results = []
    cds_records = []

    for i, candidate in enumerate(top_synthesis, 1):
        seq = candidate["sequence"]
        sid = candidate["id"]

        opt = full_codon_optimization(
            protein_seq=seq,
            species=species,
            system=system,
            signal_peptide=signal_peptide,
            target_gc=target_gc,
        )

        # Save CDS FASTA
        cds_name = f"Top{i}_{sid}_CDS"
        cds_records.append((cds_name, opt["cds_dna"]))
        save_fasta(
            [(cds_name, opt["cds_dna"])],
            str(top_dir / f"Top{i}_{sid}_CDS.fasta"),
        )

        results.append({
            "rank": i,
            "id": sid,
            "protein_length": len(seq),
            "cds_length_bp": len(opt["cds_dna"]),
            "gc_content": opt["gc_content_cds"],
            "has_t5nt": opt["has_t5nt"],
            "restriction_sites": opt["restriction_sites_found"],
            "cassette_length_bp": opt["cassette"]["cassette_length_bp"],
            "expression_system": system,
            "species": species,
        })

        logger.info(
            f"    Top{i} {sid}: CDS={len(opt['cds_dna'])}bp "
            f"GC={opt['gc_content_cds']:.1%} T5NT={'YES' if opt['has_t5nt'] else 'no'}"
        )

    # Save combined outputs
    save_fasta(cds_records, str(synthesis_dir / "all_cds.fasta"))

    # Save protein FASTA for top synthesis
    save_fasta(
        [(f"Top{i}_{c['id']}", c["sequence"]) for i, c in enumerate(top_synthesis, 1)],
        str(top_dir / "top_protein.fasta"),
    )

    result = {
        "species": species,
        "expression_system": system,
        "signal_peptide": signal_peptide,
        "n_optimized": len(results),
        "candidates": results,
    }

    save_json(result, output_dir / "B7_codon_optimization.json")
    logger.info(f"  B7 complete: {len(results)} candidates optimized")
    return result

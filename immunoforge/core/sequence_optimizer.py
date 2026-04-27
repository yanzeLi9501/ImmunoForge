"""
Multi-Stage Sequence Optimization — BindCraft-Inspired.

Replaces single-temperature ProteinMPNN sampling with a multi-stage
optimization pipeline that progressively refines sequences using
structural feedback from ESMFold.

Stages:
    1. Diverse sampling: ProteinMPNN at multiple temperatures (T=1.0, 0.5, 0.2, 0.1)
       to generate a broad initial population.
    2. Structure scoring: ESMFold pLDDT + helicity penalty scoring of all candidates.
    3. Greedy mutation: Single-position mutations on top candidates, re-scored
       by ESMFold to find local optima.
    4. Interface optimization: Select final candidates maximizing interface
       quality metrics (pLDDT at interface, low helicity penalty).

References:
    - Jendrusch et al. bioRxiv (2024) — BindCraft
    - Dauparas et al. Science 2022 — ProteinMPNN
    - Lin Z et al. Science 2023 — ESMFold
"""

import logging
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class OptimizationResult:
    """Result of multi-stage sequence optimization."""
    stage: str
    sequences: list[dict]          # [{id, sequence, score, plddt, ...}]
    n_evaluated: int
    best_score: float
    details: dict = field(default_factory=dict)


def _run_mpnn_at_temperature(
    pdb_path: str,
    output_dir: str,
    temperature: float,
    n_sequences: int = 4,
    model_weights_dir: str = "",
) -> list[tuple[str, str, float]]:
    """Run ProteinMPNN at a specific temperature and return parsed sequences."""
    python = sys.executable
    out_dir = Path(output_dir) / f"mpnn_T{temperature:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = (
        f"{python} -m protein_mpnn_run "
        f"--pdb_path {pdb_path} "
        f"--out_folder {out_dir} "
        f"--num_seq_per_target {n_sequences} "
        f"--sampling_temp {temperature} "
        f"--model_name v_48_020 "
        f"--batch_size 1 "
    )
    if model_weights_dir:
        cmd += f"--path_to_model_weights {model_weights_dir} "

    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.warning(f"  ProteinMPNN T={temperature} failed: {e}")
        return []

    # Parse FASTA outputs
    seqs_dir = out_dir / "seqs"
    if not seqs_dir.exists():
        return []

    results = []
    for fa in sorted(seqs_dir.glob("*.fa")):
        results.extend(_parse_mpnn_fasta(str(fa), temperature))

    return results


def _parse_mpnn_fasta(
    fasta_path: str,
    temperature: float,
) -> list[tuple[str, str, float]]:
    """Parse ProteinMPNN FASTA output. Returns (id, binder_seq, mpnn_score)."""
    backbone_name = Path(fasta_path).stem
    results = []
    current_id = ""
    current_seq = []
    score = 0.0
    entry_idx = 0

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id and current_seq:
                    full_seq = "".join(current_seq)
                    if entry_idx > 1:
                        binder = full_seq.split("/")[-1] if "/" in full_seq else full_seq
                        binder = binder.upper()
                        design_id = f"{backbone_name}_T{temperature:.2f}_seq{entry_idx - 1}"
                        results.append((design_id, binder, score))
                entry_idx += 1
                current_id = line[1:].split(",")[0]
                if "score=" in line:
                    try:
                        score = float(line.split("score=")[1].split(",")[0])
                    except (ValueError, IndexError):
                        score = 0.0
                current_seq = []
            else:
                current_seq.append(line)

    if current_id and current_seq:
        full_seq = "".join(current_seq)
        if entry_idx > 1:
            binder = full_seq.split("/")[-1] if "/" in full_seq else full_seq
            binder = binder.upper()
            design_id = f"{backbone_name}_T{temperature:.2f}_seq{entry_idx - 1}"
            results.append((design_id, binder, score))

    return results


def _score_with_esmfold(
    sequence: str,
    helicity_weight: float = 0.1,
) -> dict:
    """Score a sequence with ESMFold pLDDT + helicity penalty.

    Returns dict with plddt, helicity_penalty, combined_score.
    """
    from immunoforge.core.structure_validation import validate_structure
    from immunoforge.core.structure_diversity import compute_helicity_penalty

    result = validate_structure(sequence, method="esmfold")

    helicity_pen = compute_helicity_penalty(sequence)

    # Combined score: higher pLDDT is better, lower helicity penalty is better
    combined = result.mean_plddt - helicity_weight * helicity_pen * 100.0

    return {
        "mean_plddt": result.mean_plddt,
        "ptm": result.ptm,
        "quality": result.quality,
        "helicity_penalty": round(helicity_pen, 4),
        "combined_score": round(combined, 2),
        "pdb_string": result.pdb_string,
    }


def stage1_diverse_sampling(
    pdb_path: str,
    output_dir: str,
    temperatures: tuple[float, ...] = (1.0, 0.5, 0.2, 0.1),
    seqs_per_temp: int = 4,
    model_weights_dir: str = "",
) -> OptimizationResult:
    """Stage 1: Generate diverse initial pool via multi-temperature ProteinMPNN.

    Runs ProteinMPNN at multiple temperatures to explore sequence space broadly.
    Higher temperatures produce more diverse (but lower quality) sequences.
    """
    logger.info("  Stage 1: Multi-temperature ProteinMPNN sampling")

    all_sequences = []
    for temp in temperatures:
        logger.info(f"    T={temp}: sampling {seqs_per_temp} sequences...")
        seqs = _run_mpnn_at_temperature(
            pdb_path, output_dir, temp, seqs_per_temp,
            model_weights_dir=model_weights_dir,
        )
        all_sequences.extend(seqs)
        logger.info(f"    T={temp}: got {len(seqs)} sequences")

    # Deduplicate by sequence
    seen = set()
    unique = []
    for sid, seq, score in all_sequences:
        if seq not in seen:
            seen.add(seq)
            unique.append({"id": sid, "sequence": seq, "mpnn_score": score})

    logger.info(f"  Stage 1 complete: {len(unique)} unique sequences "
                f"from {len(all_sequences)} total")

    best_score = min((s["mpnn_score"] for s in unique), default=0.0)

    return OptimizationResult(
        stage="stage1_diverse_sampling",
        sequences=unique,
        n_evaluated=len(all_sequences),
        best_score=best_score,
        details={
            "temperatures": list(temperatures),
            "seqs_per_temp": seqs_per_temp,
            "total_raw": len(all_sequences),
            "unique": len(unique),
        },
    )


def stage2_structure_scoring(
    candidates: list[dict],
    top_k: int = 8,
    helicity_weight: float = 0.1,
) -> OptimizationResult:
    """Stage 2: Score all candidates with ESMFold.

    Evaluates each candidate sequence with ESMFold to get real pLDDT scores,
    applies helicity penalty, and selects top-K by combined score.
    """
    logger.info(f"  Stage 2: ESMFold scoring of {len(candidates)} candidates")

    scored = []
    for i, cand in enumerate(candidates):
        seq = cand["sequence"]
        try:
            scores = _score_with_esmfold(seq, helicity_weight)
            entry = {**cand, **scores}
            scored.append(entry)
            logger.info(
                f"    [{i+1}/{len(candidates)}] {cand['id']}: "
                f"pLDDT={scores['mean_plddt']:.1f}, "
                f"quality={scores['quality']}, "
                f"helicity_pen={scores['helicity_penalty']:.3f}, "
                f"combined={scores['combined_score']:.1f}"
            )
        except Exception as e:
            logger.warning(f"    [{i+1}/{len(candidates)}] {cand['id']} failed: {e}")
            scored.append({**cand, "combined_score": -999, "mean_plddt": 0})

    # Sort by combined score (higher is better)
    scored.sort(key=lambda x: x.get("combined_score", -999), reverse=True)

    # Select top-K
    top_candidates = scored[:top_k]
    best_score = top_candidates[0].get("combined_score", 0) if top_candidates else 0

    logger.info(f"  Stage 2 complete: top-{top_k} selected, "
                f"best combined={best_score:.1f}")

    return OptimizationResult(
        stage="stage2_structure_scoring",
        sequences=top_candidates,
        n_evaluated=len(candidates),
        best_score=best_score,
        details={
            "helicity_weight": helicity_weight,
            "all_scored": len(scored),
            "top_k": top_k,
        },
    )


def _estimate_interface_positions(sequence: str) -> list[int]:
    """Estimate likely interface positions from sequence composition.

    Interface residues tend to be enriched in aromatic (W/Y/F), charged (R/K/D/E),
    and large hydrophobic residues. Returns a sorted list of position indices
    ranked by interface propensity (top 40% of residues).
    """
    _INTERFACE_PROPENSITY = {
        "W": 1.8, "Y": 1.5, "F": 1.3, "R": 1.2, "K": 0.9,
        "H": 1.0, "D": 0.8, "E": 0.8, "L": 0.7, "I": 0.7,
        "V": 0.6, "M": 0.6, "Q": 0.5, "N": 0.5, "T": 0.4,
        "S": 0.3, "A": 0.2, "G": 0.1, "P": 0.1, "C": 0.3,
    }
    scored = [(i, _INTERFACE_PROPENSITY.get(aa, 0.3))
              for i, aa in enumerate(sequence)]
    scored.sort(key=lambda x: x[1], reverse=True)
    n_iface = max(3, int(len(sequence) * 0.4))
    return sorted(pos for pos, _ in scored[:n_iface])


def stage3_greedy_mutation(
    candidates: list[dict],
    n_mutations_per_candidate: int = 10,
    helicity_weight: float = 0.1,
    interface_bias: float = 0.7,
) -> OptimizationResult:
    """Stage 3: Interface-biased greedy single-position mutations on top candidates.

    For each candidate, mutations are preferentially sampled at predicted
    interface positions (70% probability by default) and use interface-preferred
    amino acids (W/Y/R/F/H/K/E/D) with higher probability, mirroring the
    composition bias observed at natural protein-protein interfaces.

    Args:
        candidates: Top candidates from Stage 2.
        n_mutations_per_candidate: Number of mutation trials per candidate
            (default increased from 5 to 10 for broader search).
        helicity_weight: Penalty for excessive alpha-helix.
        interface_bias: Probability of mutating at a predicted interface position.
    """
    logger.info(f"  Stage 3: Interface-biased greedy mutation on "
                f"{len(candidates)} candidates ({n_mutations_per_candidate} rounds)")

    rng = np.random.RandomState(42)
    all_improved = []
    n_total_eval = 0

    # Interface-preferred amino acids for substitution
    IFACE_PREFERRED = "YWRFHKED"
    IFACE_BIAS_FRAC = 0.6  # probability of picking an interface-preferred AA

    for cand in candidates:
        seq = cand["sequence"]
        best_seq = seq
        best_score = cand.get("combined_score", -999)
        best_entry = dict(cand)

        iface_positions = _estimate_interface_positions(seq)

        for mut_round in range(n_mutations_per_candidate):
            # Interface-biased position selection
            if iface_positions and rng.random() < interface_bias:
                pos = iface_positions[rng.randint(0, len(iface_positions))]
            else:
                pos = rng.randint(0, len(seq))

            # Interface-biased amino acid selection
            if rng.random() < IFACE_BIAS_FRAC:
                new_aa = IFACE_PREFERRED[rng.randint(0, len(IFACE_PREFERRED))]
            else:
                new_aa = AMINO_ACIDS[rng.randint(0, len(AMINO_ACIDS))]
            if new_aa == best_seq[pos]:
                continue

            mutant = best_seq[:pos] + new_aa + best_seq[pos + 1:]
            n_total_eval += 1

            try:
                scores = _score_with_esmfold(mutant, helicity_weight)
                if scores["combined_score"] > best_score:
                    old_score = best_score
                    best_score = scores["combined_score"]
                    best_seq = mutant
                    best_entry = {
                        **cand,
                        "sequence": mutant,
                        "id": f"{cand['id']}_mut{mut_round}",
                        **scores,
                    }
                    logger.info(
                        f"    {cand['id']} mut {seq[pos]}{pos+1}{new_aa}: "
                        f"{old_score:.1f} → {best_score:.1f} (improved)"
                    )
            except Exception:
                continue

        all_improved.append(best_entry)

    all_improved.sort(key=lambda x: x.get("combined_score", -999), reverse=True)
    best = all_improved[0].get("combined_score", 0) if all_improved else 0

    logger.info(f"  Stage 3 complete: {n_total_eval} mutations evaluated, "
                f"best={best:.1f}")

    return OptimizationResult(
        stage="stage3_greedy_mutation",
        sequences=all_improved,
        n_evaluated=n_total_eval,
        best_score=best,
        details={
            "mutations_per_candidate": n_mutations_per_candidate,
            "total_evaluations": n_total_eval,
        },
    )


def stage4_final_ranking(
    candidates: list[dict],
    top_n: int = 4,
) -> OptimizationResult:
    """Stage 4: Final ranking and selection.

    Applies final filtering and selects the best candidates for downstream
    analysis (B3b sequence QC, B4 structure validation, B5 affinity).
    """
    logger.info(f"  Stage 4: Final ranking of {len(candidates)} candidates")

    # Filter: require quality >= MEDIUM and pLDDT > 70
    qualified = [
        c for c in candidates
        if c.get("quality", "LOW") in ("HIGH", "MEDIUM")
        and c.get("mean_plddt", 0) > 70
    ]

    if not qualified:
        # Fallback: just take all candidates sorted by combined score
        qualified = sorted(
            candidates,
            key=lambda x: x.get("combined_score", -999),
            reverse=True,
        )

    final = qualified[:top_n]

    # Clean up PDB strings to save memory in results JSON
    for entry in final:
        if "pdb_string" in entry:
            del entry["pdb_string"]

    best = final[0].get("combined_score", 0) if final else 0

    logger.info(f"  Stage 4 complete: {len(final)} candidates selected")
    for i, c in enumerate(final):
        logger.info(
            f"    #{i+1}: {c.get('id', '?')} "
            f"pLDDT={c.get('mean_plddt', 0):.1f} "
            f"combined={c.get('combined_score', 0):.1f}"
        )

    return OptimizationResult(
        stage="stage4_final_ranking",
        sequences=final,
        n_evaluated=len(candidates),
        best_score=best,
        details={"top_n": top_n, "qualified": len(qualified)},
    )


def run_optimization(
    pdb_path: str,
    output_dir: str,
    config: dict | None = None,
) -> dict:
    """Run the full multi-stage optimization pipeline.

    Args:
        pdb_path: Path to backbone PDB file from RFdiffusion.
        output_dir: Directory for intermediate outputs.
        config: Optional config overrides.

    Returns:
        Dict with all stage results and final optimized sequences.
    """
    if config is None:
        config = {}

    opt_cfg = config.get("sequence_optimization", {})
    temperatures = tuple(opt_cfg.get("temperatures", [1.0, 0.5, 0.2, 0.1]))
    seqs_per_temp = opt_cfg.get("seqs_per_temperature", 4)
    helicity_weight = opt_cfg.get("helicity_weight", 0.1)
    stage2_top_k = opt_cfg.get("stage2_top_k", 8)
    stage3_mutations = opt_cfg.get("stage3_mutations_per_candidate", 5)
    stage4_top_n = opt_cfg.get("stage4_top_n", 4)
    model_weights_dir = config.get("proteinmpnn", {}).get("model_weights_dir", "")

    logger.info(f"  Multi-stage optimization: {pdb_path}")

    # Stage 1: Diverse ProteinMPNN sampling
    s1 = stage1_diverse_sampling(
        pdb_path, output_dir, temperatures, seqs_per_temp,
        model_weights_dir=model_weights_dir,
    )

    if not s1.sequences:
        logger.warning("  No sequences from Stage 1 — aborting optimization")
        return {
            "status": "failed",
            "error": "No sequences generated in Stage 1",
            "stages": {"stage1": s1.__dict__},
        }

    # Stage 2: ESMFold scoring
    s2 = stage2_structure_scoring(
        s1.sequences, top_k=stage2_top_k, helicity_weight=helicity_weight
    )

    # Stage 3: Greedy mutations
    s3 = stage3_greedy_mutation(
        s2.sequences,
        n_mutations_per_candidate=stage3_mutations,
        helicity_weight=helicity_weight,
    )

    # Stage 4: Final selection
    s4 = stage4_final_ranking(s3.sequences, top_n=stage4_top_n)

    return {
        "status": "completed",
        "n_stages": 4,
        "final_sequences": s4.sequences,
        "best_combined_score": s4.best_score,
        "stages": {
            "stage1": {
                "n_unique": len(s1.sequences),
                "n_total": s1.n_evaluated,
            },
            "stage2": {
                "n_scored": s2.n_evaluated,
                "top_k": len(s2.sequences),
                "best_plddt": max((s.get("mean_plddt", 0) for s in s2.sequences), default=0),
            },
            "stage3": {
                "n_mutations_evaluated": s3.n_evaluated,
                "best_after_mutation": s3.best_score,
            },
            "stage4": {
                "n_final": len(s4.sequences),
                "best_combined": s4.best_score,
            },
        },
    }

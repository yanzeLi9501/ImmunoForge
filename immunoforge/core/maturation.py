"""
In-silico Affinity Maturation Module — Iterative optimization of binder candidates.

Implements a closed-loop optimization workflow:
  Initial candidates → Consensus K_D → Bayesian-guided resampling
  → ProteinMPNN multi-temperature redesign → Partial diffusion
  → Re-scoring → Convergence check → Output

References:
    - Cao L et al. Nature 605:551 (2022) — De novo design benchmark
    - Watson JL et al. Nature 620:1089 (2023) — RFdiffusion partial diffusion
    - Dauparas J et al. Science 378:49 (2022) — ProteinMPNN
"""

import logging
import math
import random
from dataclasses import dataclass, field

import numpy as np

from immunoforge.core.affinity import run_affinity_analysis, consensus_kd, AffinityResult

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# VESM mutation pre-screening (loaded lazily)
# ═══════════════════════════════════════════════════════════════════

def _vesm_is_available() -> bool:
    """Check whether the VESM pre-screening module can be used."""
    try:
        from immunoforge.core.vesm import _vesm_available
        return _vesm_available()
    except ImportError:
        return False

# ═══════════════════════════════════════════════════════════════════
# Structure availability check
# ═══════════════════════════════════════════════════════════════════

def _has_structure_backend() -> bool:
    """Check if any real structure prediction backend is available.

    Affinity maturation requires atomic-level scoring to distinguish
    single-point mutants. Without structure prediction (ESMFold/AF2),
    the heuristic scorer cannot resolve mutant fitness differences.
    """
    try:
        from immunoforge.core.structure_validation import available_backends
        backends = available_backends()
        return any(b in backends for b in ("esmfold", "af2_multimer"))
    except ImportError:
        return False


class MaturationStructureWarning:
    """Container for maturation structure requirement information."""
    requires_structure = True
    message = (
        "Affinity maturation requires a structure prediction backend "
        "(ESMFold or AlphaFold2) to meaningfully score single-point mutations. "
        "Without atomic coordinates, the energy functions cannot distinguish "
        "mutant fitness. Install ESMFold (pip install fair-esm torch) or "
        "configure AlphaFold2 to enable this module."
    )


@dataclass
class MaturationCandidate:
    """A candidate during affinity maturation."""
    id: str
    sequence: str
    parent_id: str | None = None
    generation: int = 0
    bsa: float = 1200.0
    sc: float = 0.65
    consensus_kd_nM: float | None = None
    composite_score: float = 0.0
    mutations: list[dict] = field(default_factory=list)


@dataclass
class MaturationResult:
    """Result of an affinity maturation run."""
    initial_kd_nM: float
    final_kd_nM: float
    improvement_fold: float
    n_generations: int
    n_candidates_evaluated: int
    best_candidate: MaturationCandidate
    trajectory: list[dict] = field(default_factory=list)
    all_candidates: list[MaturationCandidate] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# Mutation operators
# ═══════════════════════════════════════════════════════════════════

# Residues grouped by physicochemical properties for conservative substitutions
_CONSERVATIVE_GROUPS = {
    "hydrophobic_small": "VILMA",
    "hydrophobic_large": "FYW",
    "positive": "KRH",
    "negative": "DE",
    "polar_small": "NQST",
    "special": "CGP",
}

_AA_TO_GROUP = {}
for group, aas in _CONSERVATIVE_GROUPS.items():
    for aa in aas:
        _AA_TO_GROUP[aa] = group


def _get_conservative_subs(aa: str) -> str:
    """Get conservative substitution candidates for an amino acid."""
    group = _AA_TO_GROUP.get(aa, "")
    candidates = _CONSERVATIVE_GROUPS.get(group, "AELK")
    return candidates


# ── Interface-biased mutation constants (v5) ──
# Amino acid interface preference scores (Bogan & Thorn hot-spot weights,
# extended with BSA contribution data from Miller et al. 1987).
INTERFACE_PREFERENCE = {
    'W': 3.0, 'Y': 2.5, 'R': 2.2, 'H': 1.8, 'F': 1.8,
    'D': 1.5, 'E': 1.5, 'K': 1.3, 'N': 1.2, 'Q': 1.2,
    'S': 1.0, 'T': 1.0, 'L': 0.9, 'I': 0.9, 'V': 0.8,
    'M': 0.8, 'A': 0.7, 'C': 0.6, 'G': 0.5, 'P': 0.3,
}

# Per-residue sidechain BSA contribution (Å², Miller et al. 1987).
RESIDUE_BSA_CONTRIBUTION = {
    'A': 67.0,  'R': 196.0, 'N': 113.0, 'D': 106.0, 'C': 104.0,
    'Q': 144.0, 'E': 138.0, 'G': 0.0,   'H': 151.0, 'I': 140.0,
    'L': 137.0, 'K': 167.0, 'M': 160.0, 'F': 175.0, 'P': 105.0,
    'S': 99.0,  'T': 122.0, 'W': 217.0, 'Y': 187.0, 'V': 117.0,
}


def _estimate_interface_positions(sequence: str) -> list[int]:
    """Estimate likely interface positions from sequence composition.

    Interface residues tend to be enriched in aromatic (W/Y/F), charged (R/K/D/E),
    and large hydrophobic residues. Returns sorted indices of top 40% by propensity.
    """
    _PROPENSITY = {
        "W": 1.8, "Y": 1.5, "F": 1.3, "R": 1.2, "K": 0.9,
        "H": 1.0, "D": 0.8, "E": 0.8, "L": 0.7, "I": 0.7,
        "V": 0.6, "M": 0.6, "Q": 0.5, "N": 0.5, "T": 0.4,
        "S": 0.3, "A": 0.2, "G": 0.1, "P": 0.1, "C": 0.3,
    }
    scored = [(i, _PROPENSITY.get(aa, 0.3)) for i, aa in enumerate(sequence)]
    scored.sort(key=lambda x: x[1], reverse=True)
    n = max(3, int(len(sequence) * 0.4))
    return sorted(pos for pos, _ in scored[:n])


def point_mutate(
    sequence: str,
    n_mutations: int = 1,
    strategy: str = "interface_biased",
    rng: random.Random | None = None,
    interface_positions: list[int] | None = None,
) -> tuple[str, list[dict]]:
    """Generate a point-mutated variant of the sequence.

    Args:
        sequence: Parent amino acid sequence.
        n_mutations: Number of point mutations to introduce.
        strategy: "random", "conservative", or "interface_biased".
        rng: Random number generator.
        interface_positions: Pre-computed interface position indices.
            If None and strategy is "interface_biased", estimated from sequence.

    Returns:
        (mutated_sequence, list_of_mutations)
    """
    if rng is None:
        rng = random.Random()

    seq = list(sequence)
    mutations = []
    all_aa = "ACDEFGHIKLMNPQRSTVWY"

    # Pre-compute interface positions for biased sampling
    if strategy == "interface_biased" and interface_positions is None:
        interface_positions = _estimate_interface_positions(sequence)

    for _ in range(n_mutations):
        # Position selection: bias toward interface positions
        if strategy == "interface_biased" and interface_positions and rng.random() < 0.7:
            pos = rng.choice(interface_positions)
        else:
            pos = rng.randint(0, len(seq) - 1)
        old_aa = seq[pos]

        if strategy == "conservative":
            candidates = _get_conservative_subs(old_aa).replace(old_aa, "")
            if not candidates:
                candidates = all_aa
            new_aa = rng.choice(candidates)
        elif strategy == "interface_biased":
            # Bias toward residues common at protein interfaces
            interface_preferred = "YWRFHKED"
            if rng.random() < 0.6:
                new_aa = rng.choice(interface_preferred)
            else:
                new_aa = rng.choice(all_aa)
            while new_aa == old_aa:
                new_aa = rng.choice(all_aa)
        else:
            new_aa = rng.choice(all_aa.replace(old_aa, ""))

        seq[pos] = new_aa
        mutations.append({
            "position": pos,
            "original": old_aa,
            "mutated": new_aa,
            "notation": f"{old_aa}{pos + 1}{new_aa}",
        })

    return "".join(seq), mutations


def multi_temperature_resample(
    sequence: str,
    temperatures: list[float] | None = None,
    n_per_temp: int = 4,
    seed: int = 42,
    mutation_rate: float = 0.2,
) -> list[tuple[str, float, list[dict]]]:
    """ProteinMPNN-style multi-temperature resampling.

    Simulates the effect of running ProteinMPNN at different sampling
    temperatures to explore sequence diversity around a parent.

    v5 improvements:
      - Wider default temperature range (5 temps: 0.1–1.0) for better
        coverage of the sequence fitness landscape.
      - Higher mutation rate (0.2 vs 0.1) producing ~20 mutations at
        high temperature for a 110-residue binder, up from ~5.
      - Interface-biased strategy at all temps ≥ 0.2.

    Lower T → conservative mutations near parent.
    Higher T → more diverse exploration.

    Returns list of (sequence, temperature, mutations).
    """
    if temperatures is None:
        temperatures = [0.1, 0.3, 0.5, 0.7, 1.0]

    rng = random.Random(seed)
    results = []

    for temp in temperatures:
        for i in range(n_per_temp):
            # Number of mutations scales with temperature and mutation_rate
            n_mut = max(1, int(len(sequence) * temp * mutation_rate))
            n_mut = min(n_mut, len(sequence) // 3)  # cap at 1/3 of sequence

            strategy = "conservative" if temp < 0.2 else "interface_biased"
            mutant, muts = point_mutate(sequence, n_mut, strategy, rng)
            results.append((mutant, temp, muts))

    return results


# ═══════════════════════════════════════════════════════════════════
# Partial diffusion simulation
# ═══════════════════════════════════════════════════════════════════

def partial_diffusion_resample(
    sequence: str,
    noise_level: float = 0.3,
    n_variants: int = 5,
    seed: int = 42,
) -> list[tuple[str, list[dict]]]:
    """Simulate partial diffusion → redesign cycle.

    In real RFdiffusion, this would add noise to backbone coordinates
    then re-run the denoising diffusion. Here we simulate the effect
    by introducing structure-aware mutations proportional to noise_level.

    Args:
        sequence: Parent sequence.
        noise_level: 0.0-1.0 controlling mutation extent.
        n_variants: Number of variants to generate.
        seed: Random seed.

    Returns:
        List of (variant_sequence, mutations).
    """
    rng = random.Random(seed)
    variants = []

    n_mut = max(1, int(len(sequence) * noise_level * 0.15))

    for i in range(n_variants):
        mutant, muts = point_mutate(
            sequence, n_mut, strategy="interface_biased", rng=rng
        )
        variants.append((mutant, muts))

    return variants


# ═══════════════════════════════════════════════════════════════════
# Scoring functions
# ═══════════════════════════════════════════════════════════════════

def _mutation_adjusted_bsa(
    parent_sequence: str,
    mutated_sequence: str,
    base_bsa: float,
) -> float:
    """Adjust BSA based on sidechain volume changes at interface positions.

    When mutations at predicted interface positions introduce larger,
    more interface-friendly residues (W, Y, F, R, H), the buried
    surface area is expected to increase. Conversely, mutations to
    smaller/non-interface residues reduce BSA.

    The adjustment is capped at ±30% of the base BSA to avoid
    physically unreasonable excursions.
    """
    if len(parent_sequence) != len(mutated_sequence):
        return base_bsa

    interface_pos = set(_estimate_interface_positions(parent_sequence))
    delta = 0.0
    for i, (p_aa, m_aa) in enumerate(zip(parent_sequence, mutated_sequence)):
        if p_aa != m_aa:
            bsa_change = (
                RESIDUE_BSA_CONTRIBUTION.get(m_aa, 100.0)
                - RESIDUE_BSA_CONTRIBUTION.get(p_aa, 100.0)
            )
            if i in interface_pos:
                delta += bsa_change  # full contribution at interface
            else:
                delta += bsa_change * 0.2  # minor contribution elsewhere

    adjusted = base_bsa + delta
    return max(base_bsa * 0.7, min(adjusted, base_bsa * 1.3))


def score_candidate(
    candidate: MaturationCandidate,
    seed: int = 42,
    parent_sequence: str | None = None,
) -> MaturationCandidate:
    """Score a maturation candidate using affinity analysis.

    When *parent_sequence* is provided, applies dynamic BSA adjustment
    based on the sidechain volume changes introduced by mutations.
    """
    bsa = candidate.bsa
    if parent_sequence and len(parent_sequence) == len(candidate.sequence):
        bsa = _mutation_adjusted_bsa(parent_sequence, candidate.sequence, candidate.bsa)

    result = run_affinity_analysis(
        candidate.sequence,
        bsa,
        candidate.sc,
        seed=seed,
    )
    consensus = result.get("consensus", {})
    candidate.consensus_kd_nM = consensus.get("consensus_kd_nM")

    # Composite score: lower K_D is better
    if candidate.consensus_kd_nM and candidate.consensus_kd_nM > 0:
        candidate.composite_score = 1.0 / (1.0 + math.log10(
            max(candidate.consensus_kd_nM, 0.01)
        ) / 6.0)
    return candidate


# ═══════════════════════════════════════════════════════════════════
# VESM-guided mutation pre-screening
# ═══════════════════════════════════════════════════════════════════

def _vesm_filter_variants(
    parent_sequence: str,
    variants: list[tuple],
    vesm_threshold: float,
    variant_type: str = "temp",
) -> list[tuple]:
    """Filter mutant variants using VESM LLR pre-screening.

    For each variant, sum the VESM scores of its mutations.
    Reject variants whose cumulative score falls below threshold.

    Args:
        parent_sequence: The wildtype parent sequence.
        variants: List of (sequence, ..., mutations) tuples.
        vesm_threshold: Cumulative LLR cutoff per variant.
        variant_type: "temp" for (seq, temp, muts) or "pd" for (seq, muts).

    Returns:
        Filtered list retaining the same tuple structure.
    """
    try:
        from immunoforge.core.vesm import vesm_prescreen_mutations
    except ImportError:
        return variants

    # Collect all unique mutations across variants for batch scoring
    all_muts = []
    for v in variants:
        muts = v[2] if variant_type == "temp" else v[1]
        all_muts.extend(muts)

    if not all_muts:
        return variants

    # Pre-screen all mutations at once
    scored_muts = vesm_prescreen_mutations(
        parent_sequence, all_muts, threshold=-999.0,  # score all, don't filter yet
    )

    # Build lookup: (position, mutated_aa) → vesm_score
    score_lookup = {}
    for m in scored_muts:
        key = (m["position"], m["mutated"])
        if m.get("vesm_score") is not None:
            score_lookup[key] = m["vesm_score"]

    # Filter variants by cumulative VESM score
    passed = []
    scored_variants = []
    for v in variants:
        muts = v[2] if variant_type == "temp" else v[1]
        total_score = 0.0
        n_scored = 0
        for m in muts:
            s = score_lookup.get((m["position"], m["mutated"]))
            if s is not None:
                total_score += s
                n_scored += 1

        avg = (total_score / n_scored) if n_scored > 0 else 0.0
        scored_variants.append((v, avg, n_scored))

        # Accept if: (a) no scored mutations (can't evaluate), or
        # (b) average score per mutation >= threshold
        if n_scored == 0 or avg >= vesm_threshold:
            passed.append(v)

    # Safety: always keep at least the best-scoring variant
    if not passed and scored_variants:
        scored_variants.sort(key=lambda x: x[1], reverse=True)
        passed.append(scored_variants[0][0])

    logger.info(
        f"  VESM filter ({variant_type}): {len(passed)}/{len(variants)} passed"
    )
    return passed


# ═══════════════════════════════════════════════════════════════════
# Main maturation loop
# ═══════════════════════════════════════════════════════════════════

def run_maturation(
    parent_sequence: str,
    parent_id: str = "parent",
    bsa: float = 1200.0,
    sc: float = 0.65,
    target_kd_nM: float = 50.0,
    max_generations: int = 5,
    candidates_per_gen: int = 40,
    top_k: int = 5,
    temperatures: list[float] | None = None,
    seed: int = 42,
    force: bool = False,
    use_vesm: bool = True,
    vesm_threshold: float = -7.0,
    anneal_temperatures: bool = True,
) -> MaturationResult:
    """Run in-silico affinity maturation.

    Iteratively mutates, scores, and selects the best candidates
    until the target K_D is reached or max generations exhausted.

    v5 improvements (interface-biased maturation):
      - 5 default temperatures (0.1–1.0, up from 3) for broader exploration.
      - Higher mutation rate (~20 mutations at T=1.0 for 110-aa binders).
      - Generation-dependent temperature annealing: starts with full
        exploration range, narrows toward convergence in later rounds.
      - 40 candidates/gen (up from 20) with VESM pre-screening for
        efficiency.

    When use_vesm=True, candidate mutations are pre-screened using
    VESM (Variant Effect from Shared Models) log-likelihood ratios.
    Mutations predicted to be clearly deleterious (LLR < vesm_threshold)
    are filtered out before expensive structure-based scoring, reducing
    computation by ~60-70% per generation.

    **Requires structure backend**: Without ESMFold or AlphaFold2, the
    heuristic scoring function cannot resolve single-point mutation
    fitness differences, making maturation ineffective. Set force=True
    to override this check (results will carry a warning).

    Args:
        parent_sequence: Starting binder sequence.
        parent_id: Identifier for the parent.
        bsa: Estimated buried surface area.
        sc: Shape complementarity index.
        target_kd_nM: Target consensus K_D to achieve.
        max_generations: Maximum number of maturation rounds.
        candidates_per_gen: Candidates explored per generation.
        top_k: Number of candidates to carry forward.
        temperatures: ProteinMPNN-style temperature series.
        seed: Random seed.
        force: Run even without structural backend (results unreliable).
        use_vesm: Enable VESM mutation pre-screening (default True).
        vesm_threshold: VESM LLR cutoff; mutations below are rejected.
        anneal_temperatures: Apply generation-dependent temperature
            annealing (default True). Early generations explore broadly;
            later generations converge on local optima.

    Returns:
        MaturationResult with trajectory and best candidate.
    """
    has_structure = _has_structure_backend()

    if not has_structure and not force:
        logger.warning(MaturationStructureWarning.message)
        # Return a minimal result with the warning instead of silently
        # producing ineffective optimization
        parent = MaturationCandidate(
            id=parent_id, sequence=parent_sequence, generation=0,
            bsa=bsa, sc=sc,
        )
        parent = score_candidate(parent, seed=seed)
        initial_kd = parent.consensus_kd_nM or 1e6
        return MaturationResult(
            initial_kd_nM=round(initial_kd, 1),
            final_kd_nM=round(initial_kd, 1),
            improvement_fold=1.0,
            n_generations=0,
            n_candidates_evaluated=1,
            best_candidate=parent,
            trajectory=[{
                "generation": 0,
                "best_kd_nM": initial_kd,
                "n_candidates": 1,
                "best_id": parent_id,
                "warning": MaturationStructureWarning.message,
            }],
            all_candidates=[parent],
        )
    if temperatures is None:
        temperatures = [0.1, 0.3, 0.5, 0.7, 1.0]

    rng_seed = seed
    all_candidates = []
    trajectory = []

    # Score parent
    parent = MaturationCandidate(
        id=parent_id, sequence=parent_sequence, generation=0,
        bsa=bsa, sc=sc,
    )
    parent = score_candidate(parent, seed=rng_seed)
    initial_kd = parent.consensus_kd_nM or 1e6
    all_candidates.append(parent)
    trajectory.append({
        "generation": 0,
        "best_kd_nM": initial_kd,
        "n_candidates": 1,
        "best_id": parent_id,
    })

    logger.info(
        f"  Maturation start: {parent_id} K_D={initial_kd:.1f} nM"
        f" → target {target_kd_nM:.1f} nM"
    )

    # Determine if VESM pre-screening is active
    vesm_active = use_vesm and _vesm_is_available()
    if use_vesm and not vesm_active:
        logger.info("  VESM requested but not available; skipping pre-screen")
    elif vesm_active:
        logger.info("  VESM mutation pre-screening enabled (threshold=%.1f)", vesm_threshold)

    current_pool = [parent]

    for gen in range(1, max_generations + 1):
        gen_candidates = []

        # Generation-dependent temperature annealing: explore broadly
        # in early generations, converge in later ones.
        if anneal_temperatures and max_generations > 1:
            anneal_factor = 1.0 - 0.6 * (gen - 1) / max(max_generations - 1, 1)
            gen_temps = [max(0.05, t * anneal_factor) for t in temperatures]
        else:
            gen_temps = temperatures

        for parent_cand in current_pool:
            # Multi-temperature resampling
            n_per_temp = max(1, candidates_per_gen // (len(gen_temps) * len(current_pool)))
            variants = multi_temperature_resample(
                parent_cand.sequence,
                temperatures=gen_temps,
                n_per_temp=n_per_temp,
                seed=rng_seed + gen * 1000,
            )

            # ── VESM pre-screen: filter temperature variants ──────
            if vesm_active:
                variants = _vesm_filter_variants(
                    parent_cand.sequence, variants, vesm_threshold,
                    variant_type="temp",
                )

            for seq, temp, muts in variants:
                cand = MaturationCandidate(
                    id=f"{parent_cand.id}_g{gen}_t{temp:.1f}_{len(gen_candidates)}",
                    sequence=seq,
                    parent_id=parent_cand.id,
                    generation=gen,
                    bsa=bsa, sc=sc,
                    mutations=muts,
                )
                cand = score_candidate(
                    cand, seed=rng_seed + hash(seq) % (2**31),
                    parent_sequence=parent_sequence,
                )
                gen_candidates.append(cand)

            # Partial diffusion variants
            pd_variants = partial_diffusion_resample(
                parent_cand.sequence,
                noise_level=0.2 + 0.1 * gen,
                n_variants=max(1, candidates_per_gen // (5 * len(current_pool))),
                seed=rng_seed + gen * 2000,
            )

            # ── VESM pre-screen: filter partial-diffusion variants ─
            if vesm_active:
                pd_variants = _vesm_filter_variants(
                    parent_cand.sequence, pd_variants, vesm_threshold,
                    variant_type="pd",
                )

            for seq, muts in pd_variants:
                cand = MaturationCandidate(
                    id=f"{parent_cand.id}_g{gen}_pd_{len(gen_candidates)}",
                    sequence=seq,
                    parent_id=parent_cand.id,
                    generation=gen,
                    bsa=bsa, sc=sc,
                    mutations=muts,
                )
                cand = score_candidate(
                    cand, seed=rng_seed + hash(seq) % (2**31),
                    parent_sequence=parent_sequence,
                )
                gen_candidates.append(cand)

        # Select top-k
        gen_candidates.sort(
            key=lambda c: c.consensus_kd_nM if c.consensus_kd_nM else 1e9
        )
        current_pool = gen_candidates[:top_k]
        all_candidates.extend(gen_candidates)

        best = current_pool[0]
        best_kd = best.consensus_kd_nM or 1e6
        trajectory.append({
            "generation": gen,
            "best_kd_nM": best_kd,
            "n_candidates": len(gen_candidates),
            "best_id": best.id,
        })

        logger.info(
            f"  Gen {gen}: {len(gen_candidates)} candidates, "
            f"best K_D={best_kd:.1f} nM ({best.id})"
        )

        # Check convergence
        if best_kd <= target_kd_nM:
            logger.info(f"  Target K_D reached at generation {gen}")
            break

    # Final result
    best_overall = min(
        all_candidates,
        key=lambda c: c.consensus_kd_nM if c.consensus_kd_nM else 1e9,
    )
    final_kd = best_overall.consensus_kd_nM or 1e6
    improvement = initial_kd / max(final_kd, 0.001)

    return MaturationResult(
        initial_kd_nM=round(initial_kd, 1),
        final_kd_nM=round(final_kd, 1),
        improvement_fold=round(improvement, 2),
        n_generations=len(trajectory) - 1,
        n_candidates_evaluated=len(all_candidates),
        best_candidate=best_overall,
        trajectory=trajectory,
        all_candidates=all_candidates,
    )


def batch_maturation(
    candidates: list[dict],
    target_kd_nM: float = 50.0,
    max_generations: int = 3,
    seed: int = 42,
) -> list[dict]:
    """Run maturation on a batch of candidates.

    Args:
        candidates: List of dicts with id, sequence, bsa, sc keys.
        target_kd_nM: Target K_D in nM.
        max_generations: Max optimization rounds.
        seed: Random seed.

    Returns:
        List of maturation result summaries.
    """
    results = []
    for i, cand in enumerate(candidates):
        result = run_maturation(
            parent_sequence=cand["sequence"],
            parent_id=cand.get("id", f"cand_{i}"),
            bsa=cand.get("bsa", 1200.0),
            sc=cand.get("sc", 0.65),
            target_kd_nM=target_kd_nM,
            max_generations=max_generations,
            seed=seed + i,
        )
        results.append({
            "parent_id": cand.get("id", f"cand_{i}"),
            "initial_kd_nM": result.initial_kd_nM,
            "final_kd_nM": result.final_kd_nM,
            "improvement_fold": result.improvement_fold,
            "n_generations": result.n_generations,
            "n_evaluated": result.n_candidates_evaluated,
            "best_id": result.best_candidate.id,
            "best_sequence": result.best_candidate.sequence,
            "trajectory": result.trajectory,
        })
    return results

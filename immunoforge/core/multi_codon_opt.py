"""
Multi-Expression System Codon Optimization Extension.

Extends the base codon optimization module with support for additional
expression systems (AAV, lentivirus, mRNA, CHO, HEK293, E. coli, yeast)
and system-specific constraints.

Features:
    - System-specific codon bias: CHO/HEK293 (mammalian), E. coli, Pichia
    - CpG depletion for mRNA and AAV
    - ITR size constraint check for AAV (≤4.7 kb)
    - Poly-A signal avoidance for mRNA cassettes
    - Rare codon elimination for E. coli
    - Multi-system simultaneous optimization with comparison report
"""

import logging
import re
from dataclasses import dataclass, field

from immunoforge.core.codon_opt import (
    SPECIES_FREQ_TABLES,
    RESTRICTION_SITES,
    VACCINIA_T5NT,
    gc_content,
    optimize_codons,
    remove_t5nt,
    check_restriction_sites,
    build_expression_cassette,
)
from immunoforge.core.utils import CODON_TABLE

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Additional species / host codon usage tables
# ═══════════════════════════════════════════════════════════════════

# E. coli K-12 (Kazusa)
ECOLI_CODON_FREQ = {
    "TTT": 22.0, "TTC": 16.2, "TTA": 13.7, "TTG": 13.4,
    "CTT": 11.0, "CTC": 11.0, "CTA": 3.9, "CTG": 52.6,
    "ATT": 30.1, "ATC": 24.8, "ATA": 4.4, "ATG": 27.6,
    "GTT": 18.1, "GTC": 15.2, "GTA": 10.9, "GTG": 26.3,
    "TCT": 8.6, "TCC": 8.6, "TCA": 7.2, "TCG": 8.9,
    "CCT": 7.0, "CCC": 5.5, "CCA": 8.4, "CCG": 23.1,
    "ACT": 9.0, "ACC": 23.0, "ACA": 7.1, "ACG": 14.5,
    "GCT": 15.4, "GCC": 25.5, "GCA": 20.1, "GCG": 33.7,
    "TAT": 16.2, "TAC": 12.1, "CAT": 12.8, "CAC": 9.7,
    "CAA": 15.2, "CAG": 28.8, "AAT": 17.7, "AAC": 21.7,
    "AAA": 33.6, "AAG": 10.4, "GAT": 32.1, "GAC": 19.1,
    "GAA": 39.4, "GAG": 18.0, "TGT": 5.2, "TGC": 6.5,
    "TGG": 15.2, "CGT": 20.9, "CGC": 21.8, "CGA": 3.6,
    "CGG": 5.6, "AGT": 8.8, "AGC": 16.0, "AGA": 2.1,
    "AGG": 1.2, "GGT": 24.5, "GGC": 29.4, "GGA": 8.0,
    "GGG": 11.1,
}

# CHO / Hamster (approx. mammalian, close to mouse)
CHO_CODON_FREQ = {
    "TTT": 17.0, "TTC": 21.5, "TTA": 6.0, "TTG": 12.5,
    "CTT": 12.0, "CTC": 19.5, "CTA": 7.5, "CTG": 39.5,
    "ATT": 15.5, "ATC": 22.0, "ATA": 7.0, "ATG": 22.0,
    "GTT": 10.5, "GTC": 15.0, "GTA": 6.8, "GTG": 29.0,
    "TCT": 16.0, "TCC": 19.0, "TCA": 11.5, "TCG": 4.5,
    "CCT": 18.0, "CCC": 19.5, "CCA": 17.0, "CCG": 6.5,
    "ACT": 13.0, "ACC": 20.0, "ACA": 15.0, "ACG": 6.0,
    "GCT": 19.5, "GCC": 28.5, "GCA": 15.0, "GCG": 7.0,
    "TAT": 12.0, "TAC": 16.0, "CAT": 10.5, "CAC": 15.0,
    "CAA": 11.5, "CAG": 34.5, "AAT": 15.5, "AAC": 20.5,
    "AAA": 22.0, "AAG": 34.0, "GAT": 21.0, "GAC": 26.5,
    "GAA": 27.0, "GAG": 41.0, "TGT": 10.0, "TGC": 12.5,
    "TGG": 13.0, "CGT": 4.5, "CGC": 11.0, "CGA": 6.5,
    "CGG": 11.5, "AGT": 12.0, "AGC": 20.0, "AGA": 11.5,
    "AGG": 11.5, "GGT": 11.0, "GGC": 23.5, "GGA": 16.5,
    "GGG": 16.5,
}

# Pichia pastoris (Komagataella)
PICHIA_CODON_FREQ = {
    "TTT": 25.9, "TTC": 18.4, "TTA": 16.0, "TTG": 27.2,
    "CTT": 12.9, "CTC": 5.4, "CTA": 9.0, "CTG": 10.8,
    "ATT": 29.2, "ATC": 17.2, "ATA": 11.5, "ATG": 21.0,
    "GTT": 22.1, "GTC": 11.8, "GTA": 11.8, "GTG": 11.0,
    "TCT": 23.7, "TCC": 14.2, "TCA": 16.5, "TCG": 8.0,
    "CCT": 15.6, "CCC": 6.8, "CCA": 18.5, "CCG": 5.3,
    "ACT": 20.3, "ACC": 12.7, "ACA": 16.3, "ACG": 7.8,
    "GCT": 21.2, "GCC": 12.1, "GCA": 16.0, "GCG": 6.4,
    "TAT": 18.8, "TAC": 14.8, "CAT": 13.6, "CAC": 8.2,
    "CAA": 27.5, "CAG": 12.3, "AAT": 24.8, "AAC": 21.0,
    "AAA": 30.8, "AAG": 30.8, "GAT": 36.4, "GAC": 20.2,
    "GAA": 45.3, "GAG": 19.1, "TGT": 8.0, "TGC": 4.8,
    "TGG": 10.4, "CGT": 6.4, "CGC": 2.6, "CGA": 3.5,
    "CGG": 3.1, "AGT": 14.2, "AGC": 9.4, "AGA": 21.3,
    "AGG": 9.2, "GGT": 23.8, "GGC": 9.8, "GGA": 10.8,
    "GGG": 6.0,
}

# HEK293 — essentially human, with minor adaptation
HEK293_CODON_FREQ = SPECIES_FREQ_TABLES.get("human", {}).copy()

# Register additional host tables
EXPRESSION_HOST_FREQ = {
    "ecoli": ECOLI_CODON_FREQ,
    "cho": CHO_CODON_FREQ,
    "pichia": PICHIA_CODON_FREQ,
    "hek293": HEK293_CODON_FREQ,
    # Also accessible via species names:
    "mouse": SPECIES_FREQ_TABLES.get("mouse", {}),
    "human": SPECIES_FREQ_TABLES.get("human", {}),
    "cynomolgus": SPECIES_FREQ_TABLES.get("cynomolgus", {}),
}


# ═══════════════════════════════════════════════════════════════════
# System-specific constraints
# ═══════════════════════════════════════════════════════════════════

# AAV payload limit including ITRs (~145 bp each)
AAV_MAX_PAYLOAD_BP = 4700

# mRNA-unfriendly motifs
MRNA_POLY_A_SIGNALS = ["AATAAA", "ATTAAA"]
MRNA_SPLICE_DONORS = ["GTAAGT", "GTRAGT"]

# E. coli rare codons to avoid
ECOLI_RARE_CODONS = {"AGA", "AGG", "CGA", "ATA", "CTA", "CCC"}

# CpG dinucleotide pattern
CPG_PATTERN = re.compile(r"CG", re.IGNORECASE)


# ═══════════════════════════════════════════════════════════════════
# CpG depletion
# ═══════════════════════════════════════════════════════════════════

def count_cpg(dna: str) -> int:
    """Count CpG dinucleotides."""
    return len(CPG_PATTERN.findall(dna))


def cpg_observed_expected(dna: str) -> float:
    """Compute CpG observed/expected ratio."""
    n = len(dna)
    if n < 2:
        return 0.0
    c_count = dna.upper().count("C")
    g_count = dna.upper().count("G")
    cpg_count = count_cpg(dna)
    expected = (c_count * g_count) / max(n, 1)
    return cpg_count / max(expected, 0.001)


def deplete_cpg(dna: str, host: str = "human") -> str:
    """Reduce CpG content by synonymous codon replacement.

    Iteratively replaces codons containing CpG dinucleotides with
    synonymous alternatives that lack CpG.
    """
    freq = EXPRESSION_HOST_FREQ.get(host, EXPRESSION_HOST_FREQ.get("human", {}))
    # Build reverse map: aa -> [(codon, freq), ...]
    aa_codons: dict[str, list[tuple[str, float]]] = {}
    for codon, aa in CODON_TABLE.items():
        if aa == "*":
            continue
        aa_codons.setdefault(aa, []).append((codon, freq.get(codon, 1.0)))

    result = list(dna)
    for i in range(0, len(dna) - 2, 3):
        codon = dna[i:i+3]
        # Check if this codon or its junction creates CpG
        has_cpg = "CG" in codon.upper()
        # Check cross-codon CpG: previous codon ends with C, current starts with G
        if i > 0 and result[i-1].upper() == "C" and codon[0].upper() == "G":
            has_cpg = True
        # Check next junction
        if i + 3 < len(dna) and codon[2].upper() == "C" and dna[i+3].upper() == "G":
            has_cpg = True

        if not has_cpg:
            continue

        aa = CODON_TABLE.get(codon.upper())
        if not aa or aa == "*":
            continue

        # Find best CpG-free alternative
        alternatives = []
        for alt_codon, alt_freq in aa_codons.get(aa, []):
            if alt_codon.upper() == codon.upper():
                continue
            if "CG" in alt_codon.upper():
                continue
            alternatives.append((alt_codon, alt_freq))

        if alternatives:
            # Pick highest-frequency CpG-free synonym
            best = max(alternatives, key=lambda x: x[1])
            result[i:i+3] = list(best[0])

    return "".join(result)


# ═══════════════════════════════════════════════════════════════════
# System-specific optimization pipelines
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MultiSystemResult:
    """Result of multi-system codon optimization."""
    system: str
    host: str
    cds_dna: str
    cds_length_bp: int
    gc_content: float
    cpg_count: int
    cpg_oe_ratio: float
    restriction_sites: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    cassette: dict | None = None
    passes_constraints: bool = True


def optimize_for_aav(
    protein_seq: str,
    host: str = "human",
    target_gc: float = 0.50,
    signal_peptide: str = "il2_leader",
    seed: int = 42,
) -> MultiSystemResult:
    """AAV-optimized codon sequence with ITR size check + CpG depletion."""
    species = "human" if host in ("human", "hek293", "cho") else host
    cds = optimize_codons(protein_seq, species=species, target_gc=target_gc, seed=seed)
    cds = deplete_cpg(cds, host=host)

    cassette = build_expression_cassette(cds, signal_peptide=signal_peptide, system="aav", species=species)
    warnings = []

    if cassette["cassette_length_bp"] > AAV_MAX_PAYLOAD_BP:
        warnings.append(
            f"Cassette {cassette['cassette_length_bp']} bp exceeds AAV payload limit ({AAV_MAX_PAYLOAD_BP} bp)"
        )

    re_sites = check_restriction_sites(cds)
    passes = cassette["cassette_length_bp"] <= AAV_MAX_PAYLOAD_BP

    return MultiSystemResult(
        system="aav", host=host, cds_dna=cds,
        cds_length_bp=len(cds), gc_content=round(gc_content(cds), 3),
        cpg_count=count_cpg(cds), cpg_oe_ratio=round(cpg_observed_expected(cds), 3),
        restriction_sites=re_sites, warnings=warnings,
        cassette=cassette, passes_constraints=passes,
    )


def optimize_for_mrna(
    protein_seq: str,
    host: str = "human",
    target_gc: float = 0.55,
    seed: int = 42,
) -> MultiSystemResult:
    """mRNA-optimized: high GC for stability, CpG depletion, polyA signal avoidance."""
    species = "human" if host in ("human", "hek293") else host
    cds = optimize_codons(protein_seq, species=species, target_gc=target_gc, seed=seed)
    cds = deplete_cpg(cds, host=host)

    warnings = []
    for signal in MRNA_POLY_A_SIGNALS:
        if signal in cds.upper():
            warnings.append(f"Internal poly-A signal {signal} found in CDS")

    # Build mRNA-style cassette
    five_utr = "GCCACCATG"  # Kozak
    three_utr = "TGATAATAG"  # double stop
    cassette_dna = f"{five_utr}{cds}{three_utr}"
    cassette = {
        "system": "mRNA",
        "cds_length_bp": len(cds),
        "cassette_length_bp": len(cassette_dna),
        "gc_content": round(gc_content(cassette_dna), 3),
        "cassette_dna": cassette_dna,
    }

    return MultiSystemResult(
        system="mRNA", host=host, cds_dna=cds,
        cds_length_bp=len(cds), gc_content=round(gc_content(cds), 3),
        cpg_count=count_cpg(cds), cpg_oe_ratio=round(cpg_observed_expected(cds), 3),
        restriction_sites=check_restriction_sites(cds),
        warnings=warnings, cassette=cassette,
        passes_constraints=len(warnings) == 0,
    )


def optimize_for_ecoli(
    protein_seq: str,
    target_gc: float = 0.50,
    seed: int = 42,
) -> MultiSystemResult:
    """E. coli-optimized: avoid rare codons, optimize CAI."""
    import random as _random
    freq = ECOLI_CODON_FREQ

    aa_codons: dict[str, list[tuple[str, float]]] = {}
    for codon, aa in CODON_TABLE.items():
        if aa == "*":
            continue
        if codon in ECOLI_RARE_CODONS:
            continue  # Skip rare
        aa_codons.setdefault(aa, []).append((codon, freq.get(codon, 1.0)))

    rng = _random.Random(seed)
    dna = []
    for aa in protein_seq:
        codons = aa_codons.get(aa)
        if not codons:
            # Fallback: allow rare if no alternatives
            for c, a in CODON_TABLE.items():
                if a == aa:
                    codons = [(c, freq.get(c, 1.0))]
                    break
        if not codons:
            raise ValueError(f"Unknown amino acid: {aa}")
        weights = [f for _, f in codons]
        chosen = rng.choices([c for c, _ in codons], weights=weights, k=1)[0]
        dna.append(chosen)

    cds = "".join(dna)
    re_sites = check_restriction_sites(cds)

    cassette_dna = f"ATGAAATACCTGCTGCCGACCGCT{cds}TAATAA"  # pelB + double stop
    cassette = {
        "system": "E. coli",
        "cds_length_bp": len(cds),
        "cassette_length_bp": len(cassette_dna),
        "gc_content": round(gc_content(cassette_dna), 3),
        "cassette_dna": cassette_dna,
    }

    return MultiSystemResult(
        system="ecoli", host="ecoli", cds_dna=cds,
        cds_length_bp=len(cds), gc_content=round(gc_content(cds), 3),
        cpg_count=count_cpg(cds), cpg_oe_ratio=round(cpg_observed_expected(cds), 3),
        restriction_sites=re_sites, warnings=[], cassette=cassette,
    )


def optimize_for_pichia(
    protein_seq: str,
    target_gc: float = 0.45,
    seed: int = 42,
) -> MultiSystemResult:
    """Pichia pastoris-optimized codon sequence."""
    # Register pichia freq temporarily for optimize_codons
    original = SPECIES_FREQ_TABLES.get("pichia")
    SPECIES_FREQ_TABLES["pichia"] = PICHIA_CODON_FREQ
    try:
        cds = optimize_codons(protein_seq, species="pichia", target_gc=target_gc, seed=seed)
    finally:
        if original is None:
            SPECIES_FREQ_TABLES.pop("pichia", None)
        else:
            SPECIES_FREQ_TABLES["pichia"] = original

    cassette_dna = f"GCCACCATG{cds}TAA"
    cassette = {
        "system": "Pichia pastoris",
        "cds_length_bp": len(cds),
        "cassette_length_bp": len(cassette_dna),
        "gc_content": round(gc_content(cassette_dna), 3),
        "cassette_dna": cassette_dna,
    }

    return MultiSystemResult(
        system="pichia", host="pichia", cds_dna=cds,
        cds_length_bp=len(cds), gc_content=round(gc_content(cds), 3),
        cpg_count=count_cpg(cds), cpg_oe_ratio=round(cpg_observed_expected(cds), 3),
        restriction_sites=check_restriction_sites(cds), warnings=[],
        cassette=cassette,
    )


# ═══════════════════════════════════════════════════════════════════
# Multi-system comparison
# ═══════════════════════════════════════════════════════════════════

SYSTEM_OPTIMIZERS = {
    "aav": optimize_for_aav,
    "mrna": optimize_for_mrna,
    "ecoli": optimize_for_ecoli,
    "pichia": optimize_for_pichia,
}


def optimize_multi_system(
    protein_seq: str,
    systems: list[str] | None = None,
    seed: int = 42,
) -> dict[str, MultiSystemResult]:
    """Run codon optimization for multiple expression systems simultaneously."""
    if systems is None:
        systems = list(SYSTEM_OPTIMIZERS.keys())

    results = {}
    for sys_name in systems:
        optimizer = SYSTEM_OPTIMIZERS.get(sys_name.lower())
        if optimizer is None:
            logger.warning("Unknown expression system: %s", sys_name)
            continue
        try:
            results[sys_name] = optimizer(protein_seq, seed=seed)
        except Exception as e:
            logger.error("Optimization failed for %s: %s", sys_name, e)
            results[sys_name] = MultiSystemResult(
                system=sys_name, host="unknown", cds_dna="",
                cds_length_bp=0, gc_content=0.0, cpg_count=0, cpg_oe_ratio=0.0,
                warnings=[f"Optimization failed: {e}"], passes_constraints=False,
            )

    return results


def generate_comparison_table(results: dict[str, MultiSystemResult]) -> str:
    """Generate Markdown comparison table for multi-system results."""
    lines = [
        "| System | Host | CDS (bp) | GC% | CpG | CpG O/E | RE sites | Passes |",
        "|--------|------|----------|-----|-----|---------|----------|--------|",
    ]
    for name, r in results.items():
        re_count = sum(len(v) for v in r.restriction_sites.values())
        lines.append(
            f"| {r.system} | {r.host} | {r.cds_length_bp} | "
            f"{r.gc_content:.1%} | {r.cpg_count} | {r.cpg_oe_ratio:.2f} | "
            f"{re_count} | {'✓' if r.passes_constraints else '✗'} |"
        )
    return "\n".join(lines)

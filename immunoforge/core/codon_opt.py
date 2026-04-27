"""
Codon Optimization Engine.

Supports multi-species codon usage tables (Mouse / Human / Cynomolgus),
GC-balanced optimization, Vaccinia T5NT avoidance, restriction site
screening, and expression cassette assembly.
"""

import logging
import random
import re
from pathlib import Path

from immunoforge.core.utils import CODON_TABLE

logger = logging.getLogger(__name__)


# ── Species-specific codon usage frequencies (Kazusa) ──
MOUSE_CODON_FREQ = {
    "TTT": 17.2, "TTC": 21.8, "TTA": 6.3, "TTG": 12.6,
    "CTT": 12.3, "CTC": 19.7, "CTA": 7.8, "CTG": 39.2,
    "ATT": 15.7, "ATC": 22.5, "ATA": 7.3, "ATG": 22.3,
    "GTT": 10.9, "GTC": 15.4, "GTA": 7.0, "GTG": 28.9,
    "TCT": 16.0, "TCC": 19.0, "TCA": 11.5, "TCG": 4.6,
    "CCT": 18.2, "CCC": 19.0, "CCA": 17.4, "CCG": 6.5,
    "ACT": 13.4, "ACC": 19.8, "ACA": 15.4, "ACG": 6.0,
    "GCT": 19.4, "GCC": 28.6, "GCA": 15.1, "GCG": 7.1,
    "TAT": 12.0, "TAC": 16.0, "CAT": 10.4, "CAC": 15.2,
    "CAA": 11.6, "CAG": 34.3, "AAT": 15.8, "AAC": 20.6,
    "AAA": 22.0, "AAG": 34.3, "GAT": 21.2, "GAC": 26.4,
    "GAA": 27.2, "GAG": 40.7, "TGT": 10.0, "TGC": 12.5,
    "TGG": 13.1, "CGT": 4.6, "CGC": 10.8, "CGA": 6.4,
    "CGG": 11.6, "AGT": 11.9, "AGC": 19.9, "AGA": 11.4,
    "AGG": 11.4, "GGT": 10.9, "GGC": 23.4, "GGA": 16.2,
    "GGG": 16.4,
}

HUMAN_CODON_FREQ = {
    "TTT": 17.6, "TTC": 20.3, "TTA": 7.7, "TTG": 12.9,
    "CTT": 13.2, "CTC": 19.6, "CTA": 7.2, "CTG": 39.6,
    "ATT": 16.0, "ATC": 20.8, "ATA": 7.5, "ATG": 22.0,
    "GTT": 11.0, "GTC": 14.5, "GTA": 7.1, "GTG": 28.1,
    "TCT": 15.2, "TCC": 17.7, "TCA": 12.2, "TCG": 4.4,
    "CCT": 17.5, "CCC": 19.8, "CCA": 16.9, "CCG": 6.9,
    "ACT": 13.1, "ACC": 18.9, "ACA": 15.1, "ACG": 6.1,
    "GCT": 18.4, "GCC": 27.7, "GCA": 15.8, "GCG": 7.4,
    "TAT": 12.2, "TAC": 15.3, "CAT": 10.9, "CAC": 15.1,
    "CAA": 12.3, "CAG": 34.2, "AAT": 17.0, "AAC": 19.1,
    "AAA": 24.4, "AAG": 31.9, "GAT": 21.8, "GAC": 25.1,
    "GAA": 29.0, "GAG": 39.6, "TGT": 10.6, "TGC": 12.6,
    "TGG": 13.2, "CGT": 4.5, "CGC": 10.4, "CGA": 6.2,
    "CGG": 11.4, "AGT": 12.1, "AGC": 19.5, "AGA": 12.2,
    "AGG": 12.0, "GGT": 10.8, "GGC": 22.2, "GGA": 16.5,
    "GGG": 16.5,
}

CYNOMOLGUS_CODON_FREQ = HUMAN_CODON_FREQ.copy()  # Very similar to human

SPECIES_FREQ_TABLES = {
    "mouse": MOUSE_CODON_FREQ,
    "human": HUMAN_CODON_FREQ,
    "cynomolgus": CYNOMOLGUS_CODON_FREQ,
}

# ── Signal peptides ──
SIGNAL_PEPTIDES = {
    "il2_leader": {
        "name": "IL-2 leader",
        "sequence": "MYRMQLLSCIALSLALVTNS",
    },
    "igk_leader": {
        "name": "IgG kappa leader",
        "sequence": "METDTLLLWVLLLWVPGSTGD",
    },
}

# ── Restriction enzyme recognition sites ──
RESTRICTION_SITES = {
    "BamHI": "GGATCC",
    "EcoRI": "GAATTC",
    "HindIII": "AAGCTT",
    "NotI": "GCGGCCGC",
    "XhoI": "CTCGAG",
    "NdeI": "CATATG",
    "NcoI": "CCATGG",
    "BglII": "AGATCT",
    "SalI": "GTCGAC",
}

# ── Vaccinia early termination signal ──
VACCINIA_T5NT = r"TTTTT.T"


def _build_aa_codons(species: str = "mouse") -> dict[str, list[tuple[str, float]]]:
    """Group codons by amino acid with their usage frequencies."""
    freq = SPECIES_FREQ_TABLES.get(species, MOUSE_CODON_FREQ)
    aa_codons: dict[str, list[tuple[str, float]]] = {}
    for codon, aa in CODON_TABLE.items():
        if aa == "*":
            continue
        aa_codons.setdefault(aa, []).append((codon, freq.get(codon, 1.0)))
    return aa_codons


def gc_content(seq: str) -> float:
    """Calculate GC content of a DNA sequence."""
    if not seq:
        return 0.0
    return sum(1 for c in seq.upper() if c in "GC") / len(seq)


def optimize_codons(
    protein_seq: str,
    species: str = "mouse",
    target_gc: float = 0.50,
    seed: int = 42,
) -> str:
    """GC-balanced codon optimization using species-specific usage tables."""
    aa_codons = _build_aa_codons(species)
    rng = random.Random(seed)
    dna = []
    gc_sum = 0
    length = 0

    for aa in protein_seq:
        codons = aa_codons.get(aa)
        if not codons:
            raise ValueError(f"Unknown amino acid: {aa}")
        if len(codons) == 1:
            dna.append(codons[0][0])
            gc_sum += sum(1 for c in codons[0][0] if c in "GC")
            length += 3
            continue

        current_gc = gc_sum / max(length, 1)
        # Bias toward codons that bring GC closer to target
        weights = []
        for codon, freq in codons:
            codon_gc = sum(1 for c in codon if c in "GC") / 3
            gc_bias = 1.0 + 2.0 * (target_gc - current_gc) * (codon_gc - 0.5)
            weights.append(max(freq * gc_bias, 0.1))

        total = sum(weights)
        probs = [w / total for w in weights]
        chosen = rng.choices([c[0] for c in codons], weights=probs, k=1)[0]
        dna.append(chosen)
        gc_sum += sum(1 for c in chosen if c in "GC")
        length += 3

    return "".join(dna)


def remove_t5nt(dna: str, species: str = "mouse") -> str:
    """Remove Vaccinia early termination signals (TTTTTNT) by synonymous substitution."""
    aa_codons = _build_aa_codons(species)
    result = list(dna)

    for match in re.finditer(VACCINIA_T5NT, dna):
        start = match.start()
        # Find codon boundary
        codon_start = (start // 3) * 3
        for pos in range(codon_start, min(codon_start + 9, len(dna) - 2), 3):
            codon = dna[pos : pos + 3]
            aa = CODON_TABLE.get(codon)
            if aa and aa != "*":
                alternatives = [c for c, _ in aa_codons.get(aa, []) if c != codon]
                if alternatives:
                    new_codon = alternatives[0]
                    result[pos : pos + 3] = list(new_codon)
                    break

    return "".join(result)


def check_restriction_sites(dna: str, sites_to_avoid: list[str] | None = None) -> dict:
    """Check for restriction enzyme sites in DNA sequence."""
    if sites_to_avoid is None:
        sites_to_avoid = list(RESTRICTION_SITES.keys())

    found = {}
    for name in sites_to_avoid:
        site = RESTRICTION_SITES.get(name, "")
        if site and site in dna:
            positions = [m.start() for m in re.finditer(re.escape(site), dna)]
            found[name] = positions

    return found


def build_expression_cassette(
    cds_dna: str,
    signal_peptide: str = "il2_leader",
    system: str = "vaccinia",
    species: str = "mouse",
) -> dict:
    """Build complete expression cassette for the given system."""

    # Signal peptide DNA
    sp_info = SIGNAL_PEPTIDES.get(signal_peptide, SIGNAL_PEPTIDES["il2_leader"])
    sp_dna = optimize_codons(sp_info["sequence"], species=species)

    # Promoter/Kozak/polyA
    if system == "vaccinia":
        promoter = "AAAAATTGAAATTTTATTTTTTTTTTTTGGAATATAAATA"  # pSE/L synthetic
        kozak = "GCCACCATG"
        poly_a = "AATAAA" + "T" * 30
        tk_arm_5 = "TGATGACACAAACCCCGCCCAGCGTCTTGTCATTGGCGAATTCGAACACGCAG"
        tk_arm_3 = "ATCATGTCAGATCCTGACGATCGCCCTAGCAGCTTGGCC"
        cassette = f"{tk_arm_5}{promoter}{kozak}{sp_dna}{cds_dna}TAA{poly_a}{tk_arm_3}"
    elif system == "aav":
        promoter = "CMV_PROMOTER_PLACEHOLDER"
        kozak = "GCCACCATG"
        cassette = f"{promoter}{kozak}{sp_dna}{cds_dna}TAA"
    elif system == "lentivirus":
        promoter = "EF1A_PROMOTER_PLACEHOLDER"
        kozak = "GCCACCATG"
        cassette = f"{promoter}{kozak}{sp_dna}{cds_dna}TAA"
    else:
        cassette = f"GCCACCATG{sp_dna}{cds_dna}TAA"

    return {
        "system": system,
        "signal_peptide": sp_info["name"],
        "cds_length_bp": len(cds_dna),
        "cassette_length_bp": len(cassette),
        "gc_content": round(gc_content(cassette), 3),
        "cassette_dna": cassette,
    }


def full_codon_optimization(
    protein_seq: str,
    species: str = "mouse",
    system: str = "vaccinia",
    signal_peptide: str = "il2_leader",
    target_gc: float = 0.50,
) -> dict:
    """Complete codon optimization pipeline: optimize → T5NT removal → RE check → cassette."""

    # Step 1: Optimize codons
    cds = optimize_codons(protein_seq, species=species, target_gc=target_gc)

    # Step 2: Remove Vaccinia T5NT signals
    if system == "vaccinia":
        cds = remove_t5nt(cds, species=species)

    # Step 3: Check restriction sites
    re_sites = check_restriction_sites(cds)

    # Step 4: Build expression cassette
    cassette = build_expression_cassette(
        cds, signal_peptide=signal_peptide, system=system, species=species
    )

    return {
        "protein_sequence": protein_seq,
        "cds_dna": cds,
        "gc_content_cds": round(gc_content(cds), 3),
        "restriction_sites_found": re_sites,
        "has_t5nt": bool(re.search(VACCINIA_T5NT, cds)),
        "species": species,
        "expression_system": system,
        "cassette": cassette,
    }

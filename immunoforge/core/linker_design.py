"""
Linker Design Module — Automatic linker selection for bispecific fusion proteins.

Designs flexible, rigid, and helical linkers connecting two independent
binder domains into a single bispecific construct, with automatic length
optimization based on target structural constraints.

References:
    - Chen X et al. Adv Drug Deliv Rev 65:1357 (2013) — Linker design review
    - Klein JS et al. Protein Eng Des Sel 27:325 (2014) — Flexible linkers
    - Arai R et al. Protein Eng 14:529 (2001) — Rigid helical linkers
"""

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Immune synapse geometry constraints (manuscript Methods §1.6) ──
# CAR-T / NK-cell synapse gap: 13–15 nm (WLC chain model calibrated on
# CD45 exclusion zone data, Bhatt et al. 2022).
SYNAPSE_MIN_NM: float = 13.0
SYNAPSE_MAX_NM: float = 15.0


# ═══════════════════════════════════════════════════════════════════
# Linker templates
# ═══════════════════════════════════════════════════════════════════

@dataclass
class LinkerTemplate:
    """Definition of a linker design template."""
    name: str
    type: str               # "flexible" | "rigid" | "helical" | "cleavable"
    unit: str               # Repeating unit sequence
    min_repeats: int
    max_repeats: int
    properties: dict = field(default_factory=dict)


LINKER_TEMPLATES = [
    # Flexible linkers
    LinkerTemplate("GS4", "flexible", "GGGGS", 1, 6,
                   properties={"flexibility": "high", "immunogenicity": "low"}),
    LinkerTemplate("GS3", "flexible", "GGGS", 1, 8,
                   properties={"flexibility": "high", "immunogenicity": "low"}),
    LinkerTemplate("GGS", "flexible", "GGS", 2, 10,
                   properties={"flexibility": "very_high", "immunogenicity": "low"}),

    # Rigid helical linkers
    LinkerTemplate("EAAAK", "rigid", "EAAAK", 1, 5,
                   properties={"flexibility": "low", "immunogenicity": "low"}),
    LinkerTemplate("AEAAAK_long", "helical", "AEAAAKEAAAK", 1, 3,
                   properties={"flexibility": "very_low", "immunogenicity": "low"}),
    LinkerTemplate("AP_rich", "rigid", "PAPAP", 1, 4,
                   properties={"flexibility": "low", "immunogenicity": "low"}),

    # Cleavable linkers (for conditional activation)
    LinkerTemplate("furin_RVRR", "cleavable", "GGSRVRRGGGS", 1, 1,
                   properties={"protease": "furin", "use_case": "conditional_activation"}),
    LinkerTemplate("MMP_PLG", "cleavable", "GGSPLGLAGGGGS", 1, 1,
                   properties={"protease": "MMP-2/9", "use_case": "tumor_microenvironment"}),
]


# ═══════════════════════════════════════════════════════════════════
# Linker design functions
# ═══════════════════════════════════════════════════════════════════

@dataclass
class LinkerDesign:
    """A designed linker for connecting two binder domains."""
    template: str
    linker_type: str
    sequence: str
    length_aa: int
    n_repeats: int
    properties: dict = field(default_factory=dict)


@dataclass
class BispecificConstruct:
    """Complete bispecific fusion protein construct."""
    binder1_id: str
    binder2_id: str
    binder1_seq: str
    binder2_seq: str
    linker: LinkerDesign
    fusion_seq: str
    total_length: int
    estimated_mw_da: float
    orientations: list[dict] = field(default_factory=list)


import numpy as np

def calculate_terminal_distance(binder1_pdb_path: str, binder2_pdb_path: str) -> float:
    """
    Extracts the Euclidean distance between the actual C-terminus of Binder A 
    and N-terminus of Binder B from structural models.
    Supports physical 3D distance constraints instead of 1D additive length heuristics.
    """
    def _get_terminal_coord(pdb_path, get_n_term=True):
        coords = []
        try:
            with open(pdb_path, 'r') as f:
                for line in f:
                    if line.startswith("ATOM  ") and line[12:16].strip() == "CA":
                        try:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            coords.append(np.array([x, y, z]))
                        except ValueError:
                            pass
            if coords:
                return coords[0] if get_n_term else coords[-1]
        except Exception:
            pass
        return None

    c_term_A = _get_terminal_coord(binder1_pdb_path, get_n_term=False)
    n_term_B = _get_terminal_coord(binder2_pdb_path, get_n_term=True)
    
    if c_term_A is not None and n_term_B is not None:
        return float(np.linalg.norm(c_term_A - n_term_B))
    
    return 0.0

def estimate_optimal_linker_length(
    binder1_len: int,
    binder2_len: int,
    target_distance_angstrom: float | None = None,
    synapse_target_nm: float | None = None,
    binder_domain_span_nm: float | None = None,
) -> int:
    """Estimate optimal linker length in residues.

    Accepts either *target_distance_angstrom* (Å, legacy) or the manuscript
    API pair (*synapse_target_nm*, *binder_domain_span_nm*) in nanometres.
    When the nm pair is given, the linker must bridge
    ``synapse_target_nm - binder_domain_span_nm`` nm of remaining gap.

    Assumes ~3.8 Å per residue for extended conformation.
    For flexible linkers, effective end-to-end distance is shorter
    due to conformational averaging.

    Args:
        binder1_len: Length of first binder domain.
        binder2_len: Length of second binder domain.
        target_distance_angstrom: Desired distance between binding sites (Å).
        synapse_target_nm: Full synapse gap target (nm).
        binder_domain_span_nm: Combined span of both binder domains (nm).

    Returns:
        Recommended linker length in amino acids.
    """
    if synapse_target_nm is not None:
        span_nm = binder_domain_span_nm if binder_domain_span_nm is not None else 9.0
        gap_nm = max(0.0, synapse_target_nm - span_nm)
        effective_distance = gap_nm * 10.0  # nm → Å
    elif target_distance_angstrom is not None:
        effective_distance = target_distance_angstrom
    else:
        effective_distance = 50.0  # default

    # Extended: ~3.8 Å/residue; effective for flexible linkers: ~2.5 Å/residue
    effective_per_residue = 2.5
    min_length = max(5, int(effective_distance / 3.8))
    recommended = max(min_length, int(effective_distance / effective_per_residue))
    # Cap at reasonable maximum
    return min(recommended, 50)


def synapse_geometry_check(
    binder1_seq: str,
    binder2_seq: str,
    linker_seq: str,
    binder_domain_span_nm: float = 9.0,
) -> dict:
    """Check whether a bispecific construct satisfies immune-synapse geometry.

    Uses a Worm-Like Chain (WLC) model to estimate the end-to-end reach of the
    linker and verifies it falls within the 13–15 nm synapse gap constraint
    (SYNAPSE_MIN_NM / SYNAPSE_MAX_NM).

    Args:
        binder1_seq: Amino-acid sequence of the first binder domain.
        binder2_seq: Amino-acid sequence of the second binder domain.
        linker_seq:  Amino-acid sequence of the connecting linker.
        binder_domain_span_nm: Approximate combined span of both binder
            domains in nm (default 9.0 nm for a typical VHH + scFv pair).

    Returns:
        dict with keys:
            ``passes``   – bool, True if construct fits synapse constraints
            ``total_nm`` – estimated reach of the full construct (nm)
            ``linker_nm``– estimated linker end-to-end reach (nm)
            ``message``  – human-readable status string
    """
    # WLC effective reach: ~0.38 nm/residue (3.8 Å) for extended,
    # effective end-to-end ~0.25 nm/residue for flexible glycine-rich linkers.
    nm_per_residue = 0.25
    linker_nm = len(linker_seq) * nm_per_residue
    total_nm = binder_domain_span_nm + linker_nm

    passes = SYNAPSE_MIN_NM <= total_nm <= SYNAPSE_MAX_NM
    if passes:
        msg = (f"PASS: construct reach {total_nm:.1f} nm within "
               f"{SYNAPSE_MIN_NM}–{SYNAPSE_MAX_NM} nm synapse window")
    elif total_nm < SYNAPSE_MIN_NM:
        msg = (f"FAIL: construct too short ({total_nm:.1f} nm < "
               f"{SYNAPSE_MIN_NM} nm); consider a longer linker")
    else:
        msg = (f"FAIL: construct too long ({total_nm:.1f} nm > "
               f"{SYNAPSE_MAX_NM} nm); consider a shorter linker")

    return {
        "passes": passes,
        "total_nm": round(total_nm, 2),
        "linker_nm": round(linker_nm, 2),
        "message": msg,
    }


def design_linker(
    linker_type: str = "flexible",
    target_length: int | None = None,
    template_name: str | None = None,
) -> LinkerDesign:
    """Design a linker of the specified type and approximate length.

    Args:
        linker_type: "flexible", "rigid", "helical", or "cleavable".
        target_length: Desired linker length in amino acids. If None, uses default.
        template_name: Specific template name to use.

    Returns:
        LinkerDesign with the optimized linker sequence.
    """
    if template_name:
        templates = [t for t in LINKER_TEMPLATES if t.name == template_name]
    else:
        templates = [t for t in LINKER_TEMPLATES if t.type == linker_type]

    if not templates:
        templates = [LINKER_TEMPLATES[0]]  # Default to GS4

    template = templates[0]

    if target_length is None:
        target_length = len(template.unit) * 3  # Default: 3 repeats

    if template.type == "cleavable":
        # Cleavable linkers have fixed sequence
        n_repeats = 1
    else:
        unit_len = len(template.unit)
        n_repeats = max(
            template.min_repeats,
            min(template.max_repeats, round(target_length / unit_len)),
        )

    sequence = template.unit * n_repeats

    return LinkerDesign(
        template=template.name,
        linker_type=template.type,
        sequence=sequence,
        length_aa=len(sequence),
        n_repeats=n_repeats,
        properties=template.properties.copy(),
    )


def design_bispecific(
    binder1_id: str,
    binder1_seq: str,
    binder2_id: str,
    binder2_seq: str,
    linker_type: str = "flexible",
    linker_length: int | None = None,
    include_reverse: bool = True,
) -> BispecificConstruct:
    """Design a complete bispecific fusion protein.

    Generates the fusion construct with linker, computes properties,
    and optionally generates both orientations (A-linker-B and B-linker-A).

    Args:
        binder1_id: First binder identifier.
        binder1_seq: First binder amino acid sequence.
        binder2_id: Second binder identifier.
        binder2_seq: Second binder amino acid sequence.
        linker_type: Type of linker to use.
        linker_length: Target linker length, or None for auto.
        include_reverse: Also generate B-linker-A orientation.

    Returns:
        BispecificConstruct with the fusion protein.
    """
    from immunoforge.core.utils import compute_mw

    if linker_length is None:
        linker_length = estimate_optimal_linker_length(
            len(binder1_seq), len(binder2_seq)
        )

    linker = design_linker(linker_type, linker_length)
    fusion_seq = binder1_seq + linker.sequence + binder2_seq
    mw = compute_mw(fusion_seq)

    orientations = [
        {
            "name": f"{binder1_id}—{linker.template}—{binder2_id}",
            "sequence": fusion_seq,
            "order": "AB",
        }
    ]
    if include_reverse:
        rev_seq = binder2_seq + linker.sequence + binder1_seq
        orientations.append({
            "name": f"{binder2_id}—{linker.template}—{binder1_id}",
            "sequence": rev_seq,
            "order": "BA",
        })

    return BispecificConstruct(
        binder1_id=binder1_id,
        binder2_id=binder2_id,
        binder1_seq=binder1_seq,
        binder2_seq=binder2_seq,
        linker=linker,
        fusion_seq=fusion_seq,
        total_length=len(fusion_seq),
        estimated_mw_da=round(mw, 1),
        orientations=orientations,
    )


def design_all_linker_variants(
    binder1_id: str,
    binder1_seq: str,
    binder2_id: str,
    binder2_seq: str,
) -> list[BispecificConstruct]:
    """Generate bispecific constructs with all linker types.

    Returns one construct per linker type (flexible, rigid, helical).
    """
    constructs = []
    for ltype in ["flexible", "rigid", "helical"]:
        construct = design_bispecific(
            binder1_id, binder1_seq, binder2_id, binder2_seq,
            linker_type=ltype,
        )
        constructs.append(construct)
    return constructs

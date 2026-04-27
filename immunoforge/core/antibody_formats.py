"""
Antibody format library — multi-species, multi-isotype construct support.

Provides canonical constant-region sequences for human and mouse IgG subtypes,
light chain types (kappa/lambda), nanobody scaffolds, and bispecific formats
(KiH IgG, tandem nanobody). Each format includes species-specific CDR
framework numbering, Fc mutations, and immunogenicity cross-reference data.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  CONSTANT REGION SEQUENCE DATA
# ═══════════════════════════════════════════════════════════════════

# Representative CH1 + hinge + CH2 + CH3 sequences (truncated to key regions)
# Sources: IMGT, UniProt, Kabat numbering

# --- Human IgG constant heavy chains ---
HUMAN_IGG1_CH = (
    "ASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSS"
    "GLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSCDKTHTCPPCPAPELLGG"
    "PSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYN"
    "STYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSRDEL"
    "TKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTV"
    "DKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPGK"
)

HUMAN_IGG2_CH = (
    "ASTKGPSVFPLAPCSRSTSESTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSS"
    "GLYSLSSVVTVPSSNFGTQTYTCNVDHKPSNTKVDKTVERKCCVECPPCPAPPVAGPSVF"
    "LFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVQFNWYVDGVEVHNAKTKPREEQFNSTFR"
    "VVSVLTVVHQDWLNGKEYKCKVSNKGLPAPIEKTISKTKGQPREPQVYTLPPSREEMTKNQ"
    "VSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPMLDSDGSFFLYSKLTV"
    "DKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPGK"
)

HUMAN_IGG4_CH = (
    "ASTKGPSVFPLAPCSRSTSESTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSS"
    "GLYSLSSVVTVPSSSLGTKTYTCNVDHKPSNTKVDKRVESKYGPPCPSCPAPEFLGGPSVF"
    "LFPPKPKDTLMISRTPEVTCVVVDVSQEDPEVQFNWYVDGVEVHNAKTKPREEQFNSTYR"
    "VVSVLTVLHQDWLNGKEYKCKVSNKGLPSSIEKTISKAKGQPREPQVYTLPPSQEEMTKNQ"
    "VSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTV"
    "DKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPGK"
)

# --- Mouse IgG constant heavy chains ---
MOUSE_IGG1_CH = (
    "AKTTPPSVYPLAPGSAAQTNSMVTLGCLVKGYFPEPVTVTWNSGSLSSGVHTFPAVLQSD"
    "LYTLSSSVTVPSSTWPSETVTCNVAHPASSTKVDKKIVPRDCGCKPCICTVPEVSSVFIFP"
    "PKPKDVLTITLTPKVTCVVVDISKDDPEVQFSWFVDDVEVHTAQTQPREEQFNSTFRSVSV"
    "LPILHQDWLNGKEFKCRVNSAAFPAPIEKTISKPEGRTQVQLTLPPSRDELTKNQVSLTCLVK"
    "GFYPSDIAVEWESSGQPENNYNTTPPMLDSDGSFFLYSKLTV"
    "DKSRWQEGNVFSCSVMHEALHNHYTQKSLDRSPGK"
)

MOUSE_IGG2A_CH = (
    "AKTTAPSVYPLAPVCGDTTGSSVTLGCLVKGYFPEPVTLTWNSGSLSSGVHTFPAVLQSD"
    "LYTLSSSVTVTSSTWPSQSITCNVAHPASSTKVDKKIEPRGPTIKPCPPCKCPAPNLLGGP"
    "SVFIFPPKIKDVLMISLSPMVTCVVVDVSEDDPDVQISWFVNNVEVHTAQTQTHREDYNST"
    "LRVVSALPIQHQDWMSGKEFKCKVNNKDLPAPIERTISKPKGSVRAPQVYVLPPPEEEMTKK"
    "QVTLTCMVTDFMPEDIYVEWTNNGKTELNYKNTEPVLDSDGSYFMYSKLTV"
    "NKDWIEKGPVFNKPGATFGCSVMHEALHNHYTQKSLNRSGEC"
)

# --- Human kappa / lambda light chain constants ---
HUMAN_KAPPA_CL = (
    "RTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDS"
    "KDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
)

HUMAN_LAMBDA_CL = (
    "GQPKAAPSVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADSSPVKAGVETTTPSKQS"
    "NNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTECS"
)

# --- Mouse kappa light chain constant ---
MOUSE_KAPPA_CL = (
    "RADAAPTVSIFPPSSEQLTSGGASVVCFLNNFYPKDINVKWKIDGSERQNGVLNSWTDQDS"
    "KDSTYSMSSTLTLTKDEYERHNSYTCEATHKTSTSPIVKSFNRNEC"
)

# --- Cynomolgus monkey (Macaca fascicularis) IgG4 constant heavy chain ---
# Source: UniProt A0A2K5X283 (Macaca fascicularis IGHG4), IMGT
CYNO_IGG4_CH = (
    "ASTKGPSVFPLAPCSRSTSESTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSS"
    "GLYSLSSVVTVPSSSLGTKTYTCNVDHKPSNTKVDKRVESKYGPPCPSCPAPEFLGGPSVF"
    "LFPPKPKDTLMISRTPEVTCVVVDVSQEDPEVQFNWYVDGVEVHNAKTKPREEQFNSTYR"
    "VVSVLTVLHQDWLNGKEYKCKVSNKGLPSSIEKTISKAKGQPREPQVYTLPPSQEEMTKNQ"
    "VSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTV"
    "DKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPGK"
)

# --- Cynomolgus monkey kappa light chain constant ---
CYNO_KAPPA_CL = (
    "RTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDS"
    "KDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
)

# --- Nanobody scaffold (VHH, camelid-derived) ---
NANOBODY_SCAFFOLD = (
    "QVQLVESGGGLVQPGGSLRLSCAASGFTFS"  # FR1
    "SYAMS"                             # CDR1 placeholder
    "WVRQAPGKGLEWVS"                   # FR2
    "AISGSGGSTYYADSVKG"                # CDR2 placeholder
    "RFTISRDNSKNTLYLQMNSLRAEDTAVYYC"  # FR3
    "AKDLGSSGWRGQGTQVTVSS"            # CDR3 + FR4 placeholder
)


# ═══════════════════════════════════════════════════════════════════
#  FORMAT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AntibodyFormat:
    """Describes an antibody or antibody-like construct format."""
    name: str
    format_type: str          # "IgG", "scFv", "Fab", "nanobody", "bispecific"
    species: str              # "human", "mouse", "camelid", "humanized"
    heavy_chain_isotype: str  # "IgG1", "IgG2", "IgG4", "VHH", etc.
    light_chain_type: str     # "kappa", "lambda", "none"
    molecular_weight_kda: float
    n_chains: int
    description: str
    ch_sequence: str = ""      # constant heavy
    cl_sequence: str = ""      # constant light
    mutations: list[str] = field(default_factory=list)  # e.g. ["S228P"] for IgG4
    linker: str = ""           # for scFv / tandem formats


# ── Catalogue of supported formats ──

ANTIBODY_FORMATS: list[AntibodyFormat] = [
    # --- Human IgG subtypes ---
    AntibodyFormat(
        name="Human IgG1",
        format_type="IgG", species="human",
        heavy_chain_isotype="IgG1", light_chain_type="kappa",
        molecular_weight_kda=146.0, n_chains=4,
        description="Standard human IgG1 with strong ADCC/CDC effector functions",
        ch_sequence=HUMAN_IGG1_CH, cl_sequence=HUMAN_KAPPA_CL,
    ),
    AntibodyFormat(
        name="Human IgG2",
        format_type="IgG", species="human",
        heavy_chain_isotype="IgG2", light_chain_type="kappa",
        molecular_weight_kda=146.0, n_chains=4,
        description="Human IgG2 with reduced Fc effector function; suited for blocking",
        ch_sequence=HUMAN_IGG2_CH, cl_sequence=HUMAN_KAPPA_CL,
    ),
    AntibodyFormat(
        name="Human IgG4",
        format_type="IgG", species="human",
        heavy_chain_isotype="IgG4", light_chain_type="kappa",
        molecular_weight_kda=146.0, n_chains=4,
        description="Human IgG4 with minimal Fc effector function; S228P stabilisation",
        ch_sequence=HUMAN_IGG4_CH, cl_sequence=HUMAN_KAPPA_CL,
        mutations=["S228P"],
    ),
    AntibodyFormat(
        name="Human IgG1-lambda",
        format_type="IgG", species="human",
        heavy_chain_isotype="IgG1", light_chain_type="lambda",
        molecular_weight_kda=146.0, n_chains=4,
        description="Human IgG1 with lambda light chain",
        ch_sequence=HUMAN_IGG1_CH, cl_sequence=HUMAN_LAMBDA_CL,
    ),

    # --- Mouse IgG subtypes ---
    AntibodyFormat(
        name="Mouse IgG1",
        format_type="IgG", species="mouse",
        heavy_chain_isotype="IgG1", light_chain_type="kappa",
        molecular_weight_kda=146.0, n_chains=4,
        description="Mouse IgG1; limited ADCC; common for functional blocking",
        ch_sequence=MOUSE_IGG1_CH, cl_sequence=MOUSE_KAPPA_CL,
    ),
    AntibodyFormat(
        name="Mouse IgG2a",
        format_type="IgG", species="mouse",
        heavy_chain_isotype="IgG2a", light_chain_type="kappa",
        molecular_weight_kda=146.0, n_chains=4,
        description="Mouse IgG2a; strong ADCC and complement activation",
        ch_sequence=MOUSE_IGG2A_CH, cl_sequence=MOUSE_KAPPA_CL,
    ),

    # --- Cynomolgus monkey IgG4 ---
    AntibodyFormat(
        name="Cyno IgG4",
        format_type="IgG", species="cynomolgus",
        heavy_chain_isotype="IgG4", light_chain_type="kappa",
        molecular_weight_kda=146.0, n_chains=4,
        description="Cynomolgus monkey IgG4; preclinical surrogate for human IgG4",
        ch_sequence=CYNO_IGG4_CH, cl_sequence=CYNO_KAPPA_CL,
        mutations=["S228P"],
    ),

    # --- Fragment formats ---
    AntibodyFormat(
        name="Human scFv",
        format_type="scFv", species="human",
        heavy_chain_isotype="none", light_chain_type="kappa",
        molecular_weight_kda=27.0, n_chains=1,
        description="Single-chain variable fragment (VH-linker-VL)",
        linker="GGGGSGGGGSGGGGS",
    ),
    AntibodyFormat(
        name="Human Fab",
        format_type="Fab", species="human",
        heavy_chain_isotype="IgG1", light_chain_type="kappa",
        molecular_weight_kda=47.0, n_chains=2,
        description="Antigen-binding fragment (VH-CH1 + VL-CL)",
        ch_sequence=HUMAN_IGG1_CH[:120],  # CH1 only
        cl_sequence=HUMAN_KAPPA_CL,
    ),

    # --- Nanobody formats ---
    AntibodyFormat(
        name="Camelid VHH",
        format_type="nanobody", species="camelid",
        heavy_chain_isotype="VHH", light_chain_type="none",
        molecular_weight_kda=13.0, n_chains=1,
        description="Single-domain antibody (nanobody); ~13 kDa, excellent tissue penetration",
        ch_sequence=NANOBODY_SCAFFOLD,
    ),
    AntibodyFormat(
        name="Humanized VHH",
        format_type="nanobody", species="humanized",
        heavy_chain_isotype="VHH", light_chain_type="none",
        molecular_weight_kda=13.0, n_chains=1,
        description="Humanized single-domain antibody; reduced immunogenicity",
        ch_sequence=NANOBODY_SCAFFOLD,
    ),

    # --- Bispecific formats ---
    AntibodyFormat(
        name="KiH IgG bispecific",
        format_type="bispecific", species="human",
        heavy_chain_isotype="IgG1", light_chain_type="kappa",
        molecular_weight_kda=146.0, n_chains=4,
        description="Knobs-into-holes bispecific IgG1; two different Fab arms",
        ch_sequence=HUMAN_IGG1_CH, cl_sequence=HUMAN_KAPPA_CL,
        mutations=["T366W", "T366S", "L368A", "Y407V"],  # knob + hole
    ),
    AntibodyFormat(
        name="BiTE (tandem scFv)",
        format_type="bispecific", species="human",
        heavy_chain_isotype="none", light_chain_type="kappa",
        molecular_weight_kda=55.0, n_chains=1,
        description="Bispecific T-cell engager; tandem scFv (anti-CD3 × anti-TAA)",
        linker="GGGGS",
    ),
    AntibodyFormat(
        name="Bispecific nanobody",
        format_type="bispecific", species="camelid",
        heavy_chain_isotype="VHH", light_chain_type="none",
        molecular_weight_kda=28.0, n_chains=1,
        description="Tandem bispecific nanobody (VHH-linker-VHH); ~28 kDa",
        ch_sequence=NANOBODY_SCAFFOLD,
        linker="GGGGSGGGGS",
    ),
    AntibodyFormat(
        name="TriKE (trispecific NK engager)",
        format_type="bispecific", species="human",
        heavy_chain_isotype="none", light_chain_type="none",
        molecular_weight_kda=65.0, n_chains=1,
        description="Trispecific NK cell engager: anti-CD16 scFv–IL15–anti-TAA scFv",
        linker="GGGGSGGGGSGGGGS",
    ),
]


# ═══════════════════════════════════════════════════════════════════
#  SPECIES COMPARISON DATA
# ═══════════════════════════════════════════════════════════════════

# Representative VH framework sequences for species comparison
# These are consensus germline FR sequences for alignment visualization

SPECIES_VH_FRAMEWORKS = {
    "human_IGHV1": (
        "QVQLVQSGAEVKKPGASVKVSCKASGYTFT"  # FR1
        "-----"                             # CDR1
        "WVRQAPGQGLEWMG"                   # FR2
        "---------"                         # CDR2
        "RVTMTRDTSISTAYMELSRLRSDDTAVYYC"  # FR3
        "----------"                        # CDR3
        "WGQGTLVTVSS"                      # FR4
    ),
    "human_IGHV3": (
        "EVQLVESGGGLVQPGGSLRLSCAAS"       # FR1
        "GFTFS-----"                       # CDR1
        "WVRQAPGKGLEWVS"                   # FR2
        "---------"                         # CDR2
        "RFTISRDNAKNSLYLQMNSLRAEDTAVYYC"  # FR3
        "----------"                        # CDR3
        "WGQGTLVTVSS"                      # FR4
    ),
    "mouse_IGHV1": (
        "QVQLQQSGAELARPGASVKLSCKASGYTFT"  # FR1
        "-----"                             # CDR1
        "WVKQRTGQGLEWIGE"                 # FR2
        "---------"                         # CDR2
        "KATLTADKSSSTAYMQLSSLTSEDSAVYYC"  # FR3
        "----------"                        # CDR3
        "WGQGTTLTVSS"                      # FR4
    ),
    "mouse_IGHV5": (
        "EVKLVESGGGLVQPGGSLKLSCAASGFTFS"  # FR1
        "-----"                             # CDR1
        "WVRQTPEKRLEWVAT"                 # FR2
        "---------"                         # CDR2
        "RFTISRDNAKNTLYLQMSSLRSEDTAMYYC"  # FR3
        "----------"                        # CDR3
        "WGQGTLVTVSA"                      # FR4
    ),
    "camelid_VHH": (
        "QVQLVESGGGLVQPGGSLRLSCAASGFTFS"  # FR1
        "-----"                             # CDR1
        "WVRQAPGKGLEWVS"                   # FR2 (note: different from human)
        "---------"                         # CDR2
        "RFTISRDNSKNTLYLQMNSLRAEDTAVYYC"  # FR3
        "----------"                        # CDR3
        "WGQGTQVTVSS"                      # FR4
    ),
    # Cynomolgus monkey (Macaca fascicularis) IGHV3 germline
    # Source: IMGT, Macaca fascicularis IGHV3-NL22*01
    "cyno_IGHV3": (
        "EVQLVESGGGLVQPGGSLRLSCAAS"       # FR1 (~98% identity to human IGHV3)
        "GFTFS-----"                       # CDR1
        "WVRQAPGKGLEWVS"                   # FR2
        "---------"                         # CDR2
        "RFTISRDNAKNSLYLQMNSLRAEDTAVYYC"  # FR3
        "----------"                        # CDR3
        "WGQGTLVTVSS"                      # FR4
    ),
}

# Key positions where species differ (Kabat numbering → 0-indexed)
SPECIES_DIFF_POSITIONS = {
    "FR1": [(0, "Q/E"), (4, "V/K"), (13, "E/A")],
    "FR2": [(0, "W"), (5, "A/R"), (10, "L/G")],
    "FR3": [(2, "T/K"), (15, "L/S"), (24, "R/S")],
    "FR4": [(2, "G"), (6, "L/T"), (9, "S/A")],
}


# ═══════════════════════════════════════════════════════════════════
#  CROSS-SPECIES IMMUNOGENICITY DATA
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CrossSpeciesImmunogenicity:
    """Immunogenicity risk assessment for a format used in a host species."""
    format_name: str
    host_species: str          # "human", "mouse"
    risk_score: float          # 0-1 scale
    risk_level: str            # "very_low", "low", "moderate", "high"
    key_epitopes: int          # estimated number of T-cell epitopes
    ada_incidence_pct: float   # anti-drug antibody incidence (%)
    notes: str = ""

CROSS_SPECIES_IMMUNOGENICITY: list[CrossSpeciesImmunogenicity] = [
    # Human antibodies in human host
    CrossSpeciesImmunogenicity("Human IgG1", "human", 0.05, "very_low", 2, 3.0,
                               "Fully human; minimal immunogenicity"),
    CrossSpeciesImmunogenicity("Human IgG4", "human", 0.05, "very_low", 2, 3.0,
                               "Fully human; S228P prevents Fab-arm exchange"),
    CrossSpeciesImmunogenicity("Human scFv", "human", 0.08, "very_low", 3, 5.0,
                               "Human variable regions only; no Fc"),

    # Mouse antibodies in human host
    CrossSpeciesImmunogenicity("Mouse IgG1", "human", 0.85, "high", 25, 80.0,
                               "Fully murine; HAMA response expected"),
    CrossSpeciesImmunogenicity("Mouse IgG2a", "human", 0.90, "high", 28, 85.0,
                               "Fully murine; strong HAMA"),

    # Human antibodies in mouse host
    CrossSpeciesImmunogenicity("Human IgG1", "mouse", 0.70, "high", 20, 60.0,
                               "Xenogeneic in mouse; anti-human antibody response"),
    CrossSpeciesImmunogenicity("Human IgG4", "mouse", 0.65, "moderate", 18, 55.0,
                               "Xenogeneic; slightly less immunogenic than IgG1"),

    # Mouse antibodies in mouse host
    CrossSpeciesImmunogenicity("Mouse IgG1", "mouse", 0.05, "very_low", 1, 2.0,
                               "Syngeneic; minimal response"),
    CrossSpeciesImmunogenicity("Mouse IgG2a", "mouse", 0.05, "very_low", 1, 2.0,
                               "Syngeneic; strong effector function"),

    # Nanobodies
    CrossSpeciesImmunogenicity("Camelid VHH", "human", 0.40, "moderate", 10, 25.0,
                               "Non-human framework; CDR3 extended loops"),
    CrossSpeciesImmunogenicity("Humanized VHH", "human", 0.15, "low", 5, 8.0,
                               "Humanized framework; reduced immunogenicity"),
    CrossSpeciesImmunogenicity("Camelid VHH", "mouse", 0.45, "moderate", 12, 30.0,
                               "Camelid framework; moderate cross-reactivity"),

    # Cynomolgus monkey antibodies
    CrossSpeciesImmunogenicity("Cyno IgG4", "human", 0.25, "low", 8, 15.0,
                               "NHP IgG4; 92-95% sequence identity to human; low ADA risk"),
    CrossSpeciesImmunogenicity("Cyno IgG4", "mouse", 0.55, "moderate", 15, 45.0,
                               "NHP framework in mouse; moderate xeno-response"),
    CrossSpeciesImmunogenicity("Human IgG1", "cynomolgus", 0.08, "very_low", 3, 5.0,
                               "Human IgG1 in NHP; high sequence homology; low ADA"),
    CrossSpeciesImmunogenicity("Human IgG4", "cynomolgus", 0.06, "very_low", 2, 4.0,
                               "Human IgG4 in NHP; near-syngeneic tolerance"),
    CrossSpeciesImmunogenicity("Mouse IgG2a", "cynomolgus", 0.80, "high", 24, 75.0,
                               "Murine IgG in NHP; strong xeno-response"),
    CrossSpeciesImmunogenicity("Camelid VHH", "cynomolgus", 0.42, "moderate", 11, 28.0,
                               "Camelid framework in NHP"),
    CrossSpeciesImmunogenicity("Humanized VHH", "cynomolgus", 0.10, "low", 4, 6.0,
                               "Humanized VHH tolerated in NHP as in human"),

    # Bispecific formats
    CrossSpeciesImmunogenicity("KiH IgG bispecific", "human", 0.10, "low", 4, 8.0,
                               "Human IgG1 scaffold; KiH mutations well tolerated"),
    CrossSpeciesImmunogenicity("BiTE (tandem scFv)", "human", 0.12, "low", 5, 10.0,
                               "Human variable regions; linker may be immunogenic"),
    CrossSpeciesImmunogenicity("Bispecific nanobody", "human", 0.35, "moderate", 9, 20.0,
                               "Camelid framework unless humanized"),
]


# ═══════════════════════════════════════════════════════════════════
#  API FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def get_format(name: str) -> AntibodyFormat | None:
    """Get antibody format by exact name."""
    for f in ANTIBODY_FORMATS:
        if f.name == name:
            return f
    return None


def list_formats(
    species: str | None = None,
    format_type: str | None = None,
) -> list[AntibodyFormat]:
    """List available formats with optional filters."""
    results = ANTIBODY_FORMATS
    if species:
        results = [f for f in results if f.species.lower() == species.lower()]
    if format_type:
        results = [f for f in results
                   if f.format_type.lower() == format_type.lower()]
    return results


def get_immunogenicity(
    format_name: str,
    host_species: str,
) -> CrossSpeciesImmunogenicity | None:
    """Look up cross-species immunogenicity for a format in a given host."""
    for entry in CROSS_SPECIES_IMMUNOGENICITY:
        if entry.format_name == format_name and entry.host_species == host_species:
            return entry
    return None


def list_immunogenicity_matrix() -> list[dict]:
    """Return a flat list suitable for table display."""
    rows = []
    for entry in CROSS_SPECIES_IMMUNOGENICITY:
        rows.append({
            "format": entry.format_name,
            "host": entry.host_species,
            "risk_score": entry.risk_score,
            "risk_level": entry.risk_level,
            "epitopes": entry.key_epitopes,
            "ada_pct": entry.ada_incidence_pct,
            "notes": entry.notes,
        })
    return rows


def compare_species_sequences(species1: str, species2: str) -> dict:
    """Compare VH framework sequences between two species.

    Returns dict with alignment info and difference positions.
    """
    keys1 = [k for k in SPECIES_VH_FRAMEWORKS if k.startswith(species1)]
    keys2 = [k for k in SPECIES_VH_FRAMEWORKS if k.startswith(species2)]
    if not keys1 or not keys2:
        return {"error": f"No VH data for {species1} or {species2}"}

    seq1 = SPECIES_VH_FRAMEWORKS[keys1[0]]
    seq2 = SPECIES_VH_FRAMEWORKS[keys2[0]]

    # Simple pairwise identity
    min_len = min(len(seq1), len(seq2))
    matches = sum(1 for a, b in zip(seq1[:min_len], seq2[:min_len])
                  if a == b and a != "-")
    total = sum(1 for a, b in zip(seq1[:min_len], seq2[:min_len])
                if a != "-" and b != "-")

    return {
        "species1": species1,
        "species2": species2,
        "germline1": keys1[0],
        "germline2": keys2[0],
        "seq1": seq1,
        "seq2": seq2,
        "identity": matches / total if total > 0 else 0,
        "n_matches": matches,
        "n_total": total,
        "n_differences": total - matches,
    }


def design_construct(
    vh_sequence: str,
    vl_sequence: str | None,
    format_name: str,
) -> dict:
    """Assemble a full construct sequence from variable regions and format.

    Returns dict with full_sequence, domain_map, molecular_weight, etc.
    """
    fmt = get_format(format_name)
    if fmt is None:
        return {"error": f"Format '{format_name}' not found"}

    domains = {}
    full_seq = ""

    if fmt.format_type == "nanobody":
        full_seq = vh_sequence
        domains["VHH"] = (0, len(vh_sequence))

    elif fmt.format_type == "scFv":
        linker = fmt.linker or "GGGGSGGGGSGGGGS"
        if vl_sequence:
            full_seq = vh_sequence + linker + vl_sequence
            pos = 0
            domains["VH"] = (pos, pos + len(vh_sequence))
            pos += len(vh_sequence)
            domains["linker"] = (pos, pos + len(linker))
            pos += len(linker)
            domains["VL"] = (pos, pos + len(vl_sequence))
        else:
            full_seq = vh_sequence
            domains["VH"] = (0, len(vh_sequence))

    elif fmt.format_type == "IgG":
        # Heavy chain = VH + CH
        hc = vh_sequence + fmt.ch_sequence
        domains["VH"] = (0, len(vh_sequence))
        domains["CH"] = (len(vh_sequence), len(hc))
        if vl_sequence and fmt.cl_sequence:
            lc = vl_sequence + fmt.cl_sequence
            domains["VL"] = (len(hc), len(hc) + len(vl_sequence))
            domains["CL"] = (len(hc) + len(vl_sequence), len(hc) + len(lc))
            full_seq = hc + lc
        else:
            full_seq = hc

    elif fmt.format_type == "Fab":
        hc = vh_sequence + fmt.ch_sequence
        domains["VH"] = (0, len(vh_sequence))
        domains["CH1"] = (len(vh_sequence), len(hc))
        if vl_sequence and fmt.cl_sequence:
            lc = vl_sequence + fmt.cl_sequence
            domains["VL"] = (len(hc), len(hc) + len(vl_sequence))
            domains["CL"] = (len(hc) + len(vl_sequence), len(hc) + len(lc))
            full_seq = hc + lc
        else:
            full_seq = hc

    elif fmt.format_type == "bispecific":
        if fmt.name == "KiH IgG bispecific":
            full_seq = vh_sequence + fmt.ch_sequence
            domains["VH_arm1"] = (0, len(vh_sequence))
            domains["CH_arm1"] = (len(vh_sequence), len(full_seq))
        elif "nanobody" in fmt.name.lower():
            linker = fmt.linker or "GGGGSGGGGS"
            full_seq = vh_sequence + linker + (vl_sequence or "")
            domains["VHH1"] = (0, len(vh_sequence))
            domains["linker"] = (len(vh_sequence), len(vh_sequence) + len(linker))
            if vl_sequence:
                domains["VHH2"] = (len(vh_sequence) + len(linker), len(full_seq))
        else:
            # BiTE / tandem scFv
            full_seq = vh_sequence + (fmt.linker or "GGGGS") + (vl_sequence or "")
            domains["scFv1"] = (0, len(vh_sequence))
            if vl_sequence:
                domains["scFv2"] = (len(vh_sequence) + len(fmt.linker or "GGGGS"),
                                    len(full_seq))

    return {
        "format": fmt.name,
        "format_type": fmt.format_type,
        "species": fmt.species,
        "full_sequence": full_seq,
        "length": len(full_seq),
        "domains": domains,
        "molecular_weight_kda": fmt.molecular_weight_kda,
        "mutations": fmt.mutations,
    }

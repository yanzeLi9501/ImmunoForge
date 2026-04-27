"""
Built-in immune cell surface protein target database.

Provides curated target information for T cell, DC, NK cell, B cell,
and macrophage surface molecules across mouse/human/cynomolgus.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TargetEntry:
    """A single immune cell surface protein target."""
    name: str
    gene: str
    cell_type: str           # e.g. "T cell", "DC", "NK cell"
    species: str             # mouse | human | cynomolgus
    uniprot: str
    pdb_ids: list[str] = field(default_factory=list)
    alphafold_id: str | None = None
    known_binders: list[dict] = field(default_factory=list)
    default_chain: str = "A"
    residue_range: str | None = None
    hotspot_residues: list[int] = field(default_factory=list)
    description: str = ""
    benchmark_value: int = 0  # 1-3 stars


# ═══════════════════════════════════════════════════════════════════════
# TARGET DATABASE
# ═══════════════════════════════════════════════════════════════════════

TARGET_DB: list[TargetEntry] = [
    # ──────────── T Cell Targets ────────────
    TargetEntry(
        name="CD3E", gene="CD3E", cell_type="T cell", species="human",
        uniprot="P07766", pdb_ids=["6JXR", "1XIW"],
        known_binders=[
            {"name": "OKT3", "kd_nM": 1.0, "type": "antibody"},
            {"name": "blinatumomab", "kd_nM": 2.0, "type": "BiTE"},
        ],
        description="CD3 epsilon chain, TCR co-receptor",
        benchmark_value=3,
    ),
    TargetEntry(
        name="mCD3E", gene="Cd3e", cell_type="T cell", species="mouse",
        uniprot="P22646", alphafold_id="AF-P22646-F1",
        residue_range="23-104",
        hotspot_residues=[30, 32, 36, 41, 52, 72, 77, 78, 79, 86],
        description="Mouse CD3 epsilon chain",
        benchmark_value=3,
    ),
    TargetEntry(
        name="PD-1", gene="PDCD1", cell_type="T cell", species="human",
        uniprot="Q15116", pdb_ids=["5WT9", "4ZQK"],
        known_binders=[
            {"name": "Nivolumab", "kd_nM": 2.6, "type": "antibody"},
            {"name": "Pembrolizumab", "kd_nM": 0.027, "type": "antibody"},
        ],
        description="Programmed cell death protein 1",
        benchmark_value=3,
    ),
    TargetEntry(
        name="CTLA-4", gene="CTLA4", cell_type="T cell", species="human",
        uniprot="P16410", pdb_ids=["3OSK", "1I8L"],
        known_binders=[
            {"name": "Ipilimumab", "kd_nM": 6.2, "type": "antibody"},
            {"name": "CD80", "kd_nM": 400.0, "type": "natural_ligand"},
        ],
        benchmark_value=3,
    ),
    TargetEntry(
        name="LAG-3", gene="LAG3", cell_type="T cell", species="human",
        uniprot="P18627", pdb_ids=["7EWQ"],
        known_binders=[
            {"name": "Relatlimab", "kd_nM": 4.5, "type": "antibody"},
        ],
        benchmark_value=2,
    ),
    TargetEntry(
        name="TIM-3", gene="HAVCR2", cell_type="T cell", species="human",
        uniprot="Q8TDQ0", pdb_ids=["6TDB"],
        known_binders=[
            {"name": "Sabatolimab", "kd_nM": 0.3, "type": "antibody"},
            {"name": "Galectin-9", "kd_nM": 90.0, "type": "natural_ligand"},
        ],
        benchmark_value=2,
    ),
    TargetEntry(
        name="CD28", gene="CD28", cell_type="T cell", species="human",
        uniprot="P10747", pdb_ids=["3QUM"],
        known_binders=[
            {"name": "CD80", "kd_nM": 4000.0, "type": "natural_ligand"},
            {"name": "CD86", "kd_nM": 20000.0, "type": "natural_ligand"},
        ],
        benchmark_value=2,
    ),

    # ──────────── Dendritic Cell Targets ────────────
    TargetEntry(
        name="CLEC9A", gene="CLEC9A", cell_type="DC", species="human",
        uniprot="Q6UXN8", pdb_ids=["3VPP", "6GM9"],
        known_binders=[
            {"name": "10B12", "kd_nM": 0.4, "type": "antibody"},
            {"name": "7H11", "kd_nM": 1.2, "type": "antibody"},
        ],
        description="C-type lectin DNGR-1, cDC1 marker",
        benchmark_value=3,
    ),
    TargetEntry(
        name="mCLEC9A", gene="Clec9a", cell_type="DC", species="mouse",
        uniprot="Q8BRU4", alphafold_id="AF-Q8BRU4-F1",
        residue_range="116-238",
        hotspot_residues=[120, 145, 147, 169, 187, 199, 221, 223, 224, 229],
        description="Mouse CLEC9A / DNGR-1",
        benchmark_value=3,
    ),
    TargetEntry(
        name="XCR1", gene="XCR1", cell_type="DC", species="human",
        uniprot="P46094", alphafold_id="AF-P46094-F1",
        known_binders=[
            {"name": "XCL1", "kd_nM": 30.0, "type": "chemokine"},
        ],
        description="XC chemokine receptor 1, cDC1 marker",
        benchmark_value=3,
    ),
    TargetEntry(
        name="CLEC4C", gene="CLEC4C", cell_type="DC", species="human",
        uniprot="Q8WTT0", pdb_ids=["4KN0"],
        known_binders=[
            {"name": "PF4", "kd_nM": 200.0, "type": "protein"},
        ],
        description="BDCA-2, pDC specific C-type lectin",
        benchmark_value=2,
    ),

    # ──────────── NK Cell Targets ────────────
    TargetEntry(
        name="NKG2D", gene="KLRK1", cell_type="NK cell", species="human",
        uniprot="P26718", pdb_ids=["1HQ8", "4PBL"],
        known_binders=[
            {"name": "MICA", "kd_nM": 1000.0, "type": "natural_ligand"},
            {"name": "ULBP2", "kd_nM": 500.0, "type": "natural_ligand"},
        ],
        description="NKG2D activating receptor",
        benchmark_value=3,
    ),
    TargetEntry(
        name="NKp46", gene="NCR1", cell_type="NK cell", species="human",
        uniprot="O76036", pdb_ids=["1P6F"],
        known_binders=[
            {"name": "B7-H6", "kd_nM": 250.0, "type": "natural_ligand"},
        ],
        benchmark_value=2,
    ),
    TargetEntry(
        name="CD16a", gene="FCGR3A", cell_type="NK cell", species="human",
        uniprot="P08637", pdb_ids=["3SGJ"],
        known_binders=[
            {"name": "IgG1_Fc", "kd_nM": 1000.0, "type": "natural_ligand"},
        ],
        benchmark_value=2,
    ),

    # ──────────── B Cell Targets ────────────
    TargetEntry(
        name="CD19", gene="CD19", cell_type="B cell", species="human",
        uniprot="P15391", pdb_ids=["6AL5"],
        known_binders=[
            {"name": "Blinatumomab", "kd_nM": 4.5, "type": "BiTE"},
            {"name": "Tafasitamab", "kd_nM": 0.3, "type": "antibody"},
        ],
        benchmark_value=3,
    ),
    TargetEntry(
        name="BCMA", gene="TNFRSF17", cell_type="B cell", species="human",
        uniprot="Q02223", pdb_ids=["1OQD"],
        known_binders=[
            {"name": "APRIL", "kd_nM": 1.0, "type": "natural_ligand"},
            {"name": "Belantamab", "kd_nM": 0.4, "type": "antibody"},
        ],
        benchmark_value=3,
    ),

    # ──────────── Macrophage / Myeloid Targets ────────────
    TargetEntry(
        name="CD47", gene="CD47", cell_type="macrophage", species="human",
        uniprot="Q08722", pdb_ids=["2JJS"],
        known_binders=[
            {"name": "SIRPa", "kd_nM": 1000.0, "type": "natural_ligand"},
            {"name": "Magrolimab", "kd_nM": 1.0, "type": "antibody"},
        ],
        description="Don't-eat-me signal",
        benchmark_value=3,
    ),
    TargetEntry(
        name="SIRPa", gene="SIRPA", cell_type="macrophage", species="human",
        uniprot="P78324", pdb_ids=["4CMF"],
        known_binders=[
            {"name": "CD47", "kd_nM": 1000.0, "type": "natural_ligand"},
            {"name": "CD47v_eng", "kd_nM": 0.011, "type": "engineered"},
        ],
        description="Signal regulatory protein alpha",
        benchmark_value=3,
    ),
    TargetEntry(
        name="CSF1R", gene="CSF1R", cell_type="macrophage", species="human",
        uniprot="P07333", pdb_ids=["3LCD"],
        known_binders=[
            {"name": "CSF1", "kd_nM": 0.4, "type": "natural_ligand"},
            {"name": "IL-34", "kd_nM": 0.3, "type": "natural_ligand"},
        ],
        benchmark_value=2,
    ),
    TargetEntry(
        name="TREM2", gene="TREM2", cell_type="macrophage", species="human",
        uniprot="Q9NZC2", pdb_ids=["5ELI"],
        known_binders=[
            {"name": "AL-002c", "kd_nM": 5.0, "type": "antibody"},
        ],
        description="TAM marker, tumor-associated macrophages",
        benchmark_value=2,
    ),

    # ──────────── Tumour Cell Surface Markers ────────────
    TargetEntry(
        name="HER2", gene="ERBB2", cell_type="tumour", species="human",
        uniprot="P04626", pdb_ids=["1N8Z", "6OGE", "1S78"],
        known_binders=[
            {"name": "Trastuzumab", "kd_nM": 5.0, "type": "antibody"},
            {"name": "Pertuzumab", "kd_nM": 1.0, "type": "antibody"},
        ],
        description="Breast/gastric cancer; 4-domain receptor tyrosine kinase",
        benchmark_value=3,
    ),
    TargetEntry(
        name="EpCAM", gene="EPCAM", cell_type="tumour", species="human",
        uniprot="P16422", pdb_ids=["4MZV"],
        known_binders=[
            {"name": "Catumaxomab", "kd_nM": 5.0, "type": "antibody"},
            {"name": "Solitomab", "kd_nM": 1.7, "type": "BiTE"},
        ],
        description="Epithelial cell adhesion molecule; carcinoma marker",
        benchmark_value=3,
    ),
    TargetEntry(
        name="EGFR", gene="EGFR", cell_type="tumour", species="human",
        uniprot="P00533", pdb_ids=["1NQL", "1YY9"],
        known_binders=[
            {"name": "Cetuximab", "kd_nM": 0.39, "type": "antibody"},
            {"name": "Necitumumab", "kd_nM": 2.0, "type": "antibody"},
        ],
        description="Lung/colorectal cancer; ErbB1 receptor tyrosine kinase",
        benchmark_value=3,
    ),
    TargetEntry(
        name="MSLN", gene="MSLN", cell_type="tumour", species="human",
        uniprot="Q13421", pdb_ids=["6LRJ"],
        known_binders=[
            {"name": "Anetumab", "kd_nM": 0.7, "type": "antibody"},
        ],
        description="Mesothelin; mesothelioma, ovarian, pancreatic cancer",
        benchmark_value=2,
    ),
    TargetEntry(
        name="GPC3", gene="GPC3", cell_type="tumour", species="human",
        uniprot="P51654", pdb_ids=["4YIR"],
        known_binders=[
            {"name": "Codrituzumab", "kd_nM": 1.0, "type": "antibody"},
        ],
        description="Glypican-3; hepatocellular carcinoma marker",
        benchmark_value=2,
    ),
    TargetEntry(
        name="CLDN18.2", gene="CLDN18", cell_type="tumour", species="human",
        uniprot="P56856", pdb_ids=["7B2W"],
        known_binders=[
            {"name": "Zolbetuximab", "kd_nM": 0.7, "type": "antibody"},
        ],
        description="Claudin 18.2; gastric cancer tight-junction protein",
        benchmark_value=2,
    ),
    TargetEntry(
        name="DLL3", gene="DLL3", cell_type="tumour", species="human",
        uniprot="Q9NYJ7", alphafold_id="AF-Q9NYJ7-F1",
        known_binders=[
            {"name": "Rovalpituzumab", "kd_nM": 0.5, "type": "antibody"},
            {"name": "Tarlatamab", "kd_nM": 2.0, "type": "BiTE"},
        ],
        description="Delta-like ligand 3; small-cell lung cancer, neuroendocrine",
        benchmark_value=2,
    ),
    TargetEntry(
        name="B7-H3", gene="CD276", cell_type="tumour", species="human",
        uniprot="Q5ZPR3", pdb_ids=["4I0K"],
        known_binders=[
            {"name": "Enoblituzumab", "kd_nM": 1.0, "type": "antibody"},
        ],
        description="Immune checkpoint; multiple solid tumours",
        benchmark_value=2,
    ),
    TargetEntry(
        name="CEA", gene="CEACAM5", cell_type="tumour", species="human",
        uniprot="P06731", pdb_ids=["2QSQ"],
        known_binders=[
            {"name": "Labetuzumab", "kd_nM": 2.0, "type": "antibody"},
            {"name": "Cibisatamab", "kd_nM": 3.0, "type": "bispecific"},
        ],
        description="Carcinoembryonic antigen; colorectal, lung, pancreatic",
        benchmark_value=2,
    ),
    TargetEntry(
        name="PSMA", gene="FOLH1", cell_type="tumour", species="human",
        uniprot="Q04609", pdb_ids=["2C6C", "5O5T"],
        known_binders=[
            {"name": "J591", "kd_nM": 1.0, "type": "antibody"},
        ],
        description="Prostate-specific membrane antigen; prostate cancer",
        benchmark_value=2,
    ),
]


def search_targets(
    cell_type: str | None = None,
    species: str | None = None,
    name: str | None = None,
    min_benchmark: int = 0,
) -> list[TargetEntry]:
    """Search the target database with optional filters."""
    results = TARGET_DB
    if cell_type:
        results = [t for t in results if t.cell_type.lower() == cell_type.lower()]
    if species:
        results = [t for t in results if t.species.lower() == species.lower()]
    if name:
        name_l = name.lower()
        results = [t for t in results if name_l in t.name.lower() or name_l in t.gene.lower()]
    if min_benchmark > 0:
        results = [t for t in results if t.benchmark_value >= min_benchmark]
    return results


def get_target_by_name(name: str) -> TargetEntry | None:
    """Get a specific target by exact name."""
    for t in TARGET_DB:
        if t.name == name:
            return t
    return None


def list_cell_types() -> list[str]:
    """Return available cell types."""
    return sorted(set(t.cell_type for t in TARGET_DB))


def list_species() -> list[str]:
    """Return available species."""
    return sorted(set(t.species for t in TARGET_DB))


def get_benchmark_targets() -> list[TargetEntry]:
    """Get all targets suitable for benchmark validation (3-star)."""
    return [t for t in TARGET_DB if t.benchmark_value >= 3]

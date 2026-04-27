"""
Structure-aware affinity analysis.

When a real PDB structure (from ESMFold/AF2) is available, extract
physical contacts and buried surface area to improve affinity predictions.
Falls back to sequence-based heuristics when no structure is provided.
"""

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StructureContacts:
    """Contacts and surface metrics extracted from a PDB structure."""
    bsa_A2: float
    sc: float
    n_contacts: int
    contacts: list[dict]
    n_hbonds: int
    n_salt_bridges: int
    n_hydrophobic: int
    n_aromatic: int
    interface_residues: list[int]


def extract_contacts_from_pdb(pdb_string: str, chain_break: int | None = None) -> StructureContacts:
    """Extract inter-chain contacts from a PDB string.

    For single-chain structures (monomer predictions), uses spatial
    proximity to identify the pseudo-interface (residues <8Å from
    chain center split point).

    Args:
        pdb_string: PDB format text from ESMFold/AF2.
        chain_break: Residue index where chain B starts (for complexes).
            If None, splits at midpoint for monomer structures.

    Returns:
        StructureContacts with physical metrics.
    """
    atoms = _parse_pdb_atoms(pdb_string)
    if not atoms:
        return _fallback_contacts()

    # Determine chain split
    residue_ids = sorted(set(a["resid"] for a in atoms))
    if chain_break is None:
        chain_break = residue_ids[len(residue_ids) // 2]

    chain_a = [a for a in atoms if a["resid"] < chain_break]
    chain_b = [a for a in atoms if a["resid"] >= chain_break]

    if not chain_a or not chain_b:
        return _fallback_contacts()

    # Find inter-chain contacts within 5Å (CA-CA for speed, CB for accuracy)
    contacts = []
    n_hbonds = 0
    n_salt_bridges = 0
    n_hydrophobic = 0
    n_aromatic = 0
    interface_res_a = set()
    interface_res_b = set()

    # Build spatial index for chain B CB/CA atoms
    cb_b = {}
    for a in chain_b:
        if a["atom"] in ("CB", "CA"):
            key = a["resid"]
            if key not in cb_b or a["atom"] == "CB":
                cb_b[key] = a

    for a_atom in chain_a:
        if a_atom["atom"] not in ("CB", "CA"):
            continue
        if a_atom["resid"] in interface_res_a and a_atom["atom"] == "CA":
            continue  # Already found a CB contact for this residue

        for b_resid, b_atom in cb_b.items():
            dist = _distance(a_atom, b_atom)
            if dist < 8.0:
                contact_type = _classify_contact(a_atom["resn"], b_atom["resn"])
                contacts.append({
                    "res_a": a_atom["resid"],
                    "resn_a": a_atom["resn"],
                    "res_b": b_atom["resid"],
                    "resn_b": b_atom["resn"],
                    "distance": round(dist, 2),
                    "type": contact_type,
                })
                interface_res_a.add(a_atom["resid"])
                interface_res_b.add(b_atom["resid"])

                if contact_type == "hbond":
                    n_hbonds += 1
                elif contact_type == "salt_bridge":
                    n_salt_bridges += 1
                elif contact_type == "hydrophobic":
                    n_hydrophobic += 1
                elif contact_type == "aromatic":
                    n_aromatic += 1

    # Estimate BSA from number of interface residues (~110 Å² per interface residue)
    n_interface = len(interface_res_a) + len(interface_res_b)
    bsa = n_interface * 110.0

    # Shape complementarity from contact distance distribution
    if contacts:
        mean_dist = sum(c["distance"] for c in contacts) / len(contacts)
        close_frac = sum(1 for c in contacts if c["distance"] < 5.0) / len(contacts)
        sc = min(0.9, 0.45 + close_frac * 0.35 + max(0, 6.0 - mean_dist) * 0.05)
    else:
        sc = 0.50

    return StructureContacts(
        bsa_A2=round(bsa, 1),
        sc=round(sc, 3),
        n_contacts=len(contacts),
        contacts=contacts[:50],  # Limit for serialization
        n_hbonds=n_hbonds,
        n_salt_bridges=n_salt_bridges,
        n_hydrophobic=n_hydrophobic,
        n_aromatic=n_aromatic,
        interface_residues=sorted(interface_res_a | interface_res_b),
    )


def _parse_pdb_atoms(pdb_string: str) -> list[dict]:
    """Parse ATOM records from PDB string."""
    atoms = []
    for line in pdb_string.split("\n"):
        if not line.startswith("ATOM"):
            continue
        try:
            atom_name = line[12:16].strip()
            resn = line[17:20].strip()
            resid = int(line[22:26].strip())
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            atoms.append({
                "atom": atom_name, "resn": resn, "resid": resid,
                "x": x, "y": y, "z": z,
            })
        except (ValueError, IndexError):
            continue
    return atoms


def _distance(a: dict, b: dict) -> float:
    """Euclidean distance between two atom dicts."""
    return math.sqrt(
        (a["x"] - b["x"]) ** 2 +
        (a["y"] - b["y"]) ** 2 +
        (a["z"] - b["z"]) ** 2
    )


_POLAR = set("STCNQHYW")
_CHARGED_POS = set("RK")
_CHARGED_NEG = set("DE")
_HYDROPHOBIC = set("VILMFAW")
_AROMATIC = set("FYW")

_AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _classify_contact(resn_a: str, resn_b: str) -> str:
    """Classify a contact pair by residue types."""
    a1 = _AA_3TO1.get(resn_a, "X")
    b1 = _AA_3TO1.get(resn_b, "X")

    if (a1 in _CHARGED_POS and b1 in _CHARGED_NEG) or \
       (a1 in _CHARGED_NEG and b1 in _CHARGED_POS):
        return "salt_bridge"
    if a1 in _AROMATIC and b1 in _AROMATIC:
        return "aromatic"
    if a1 in _HYDROPHOBIC and b1 in _HYDROPHOBIC:
        return "hydrophobic"
    if a1 in _POLAR or b1 in _POLAR:
        return "hbond"
    return "other"


def _fallback_contacts() -> StructureContacts:
    """Return empty contacts when PDB parsing fails."""
    return StructureContacts(
        bsa_A2=0.0, sc=0.0, n_contacts=0, contacts=[],
        n_hbonds=0, n_salt_bridges=0, n_hydrophobic=0, n_aromatic=0,
        interface_residues=[],
    )


def structure_aware_affinity(
    binder_seq: str,
    pdb_string: str | None = None,
    bsa_override: float | None = None,
    sc_override: float | None = None,
    seed: int = 42,
    is_complex: bool = False,
) -> dict:
    """Run affinity analysis using structure-derived features when available.

    If a PDB structure from a complex prediction (AF2-multimer) is provided,
    extracts real contacts, BSA, and shape complementarity.
    For monomer predictions (ESMFold), uses pLDDT as a quality signal
    but falls back to sequence heuristics for BSA/sc (splitting a monomer
    at the midpoint does not represent a real binding interface).

    Args:
        binder_seq: Amino acid sequence.
        pdb_string: PDB text from structure prediction.
        bsa_override: Manual BSA override in Å².
        sc_override: Manual shape complementarity override.
        seed: Random seed for stochastic estimates.
        is_complex: Whether the PDB is from a complex prediction (AF2-multimer).
            If False (monomer from ESMFold), structure contacts are not used for
            BSA/sc estimation.

    Returns:
        Full affinity analysis dict including structure_contacts if available.
    """
    from immunoforge.core.affinity import run_affinity_analysis

    structure_info = None

    if pdb_string and len(pdb_string) > 100 and is_complex:
        # Only extract contacts from real complex structures
        try:
            structure_info = extract_contacts_from_pdb(pdb_string)
            if structure_info.bsa_A2 > 0:
                bsa = bsa_override or structure_info.bsa_A2
                sc = sc_override or structure_info.sc
            else:
                bsa = bsa_override or _estimate_bsa_from_seq(binder_seq, seed)
                sc = sc_override or _estimate_sc_from_seq(binder_seq, seed)
        except Exception as e:
            logger.warning(f"PDB contact extraction failed: {e}")
            bsa = bsa_override or _estimate_bsa_from_seq(binder_seq, seed)
            sc = sc_override or _estimate_sc_from_seq(binder_seq, seed)
    else:
        # Monomer or no PDB — use sequence heuristics
        bsa = bsa_override or _estimate_bsa_from_seq(binder_seq, seed)
        sc = sc_override or _estimate_sc_from_seq(binder_seq, seed)

    result = run_affinity_analysis(binder_seq, bsa, sc, seed=seed)

    # Augment with structure info
    result["structure_derived"] = is_complex and structure_info is not None
    result["pdb_available"] = pdb_string is not None and len(pdb_string or "") > 100
    if structure_info:
        result["structure_contacts"] = {
            "bsa_A2": structure_info.bsa_A2,
            "sc": structure_info.sc,
            "n_contacts": structure_info.n_contacts,
            "n_hbonds": structure_info.n_hbonds,
            "n_salt_bridges": structure_info.n_salt_bridges,
            "n_hydrophobic": structure_info.n_hydrophobic,
            "n_aromatic": structure_info.n_aromatic,
            "n_interface_residues": len(structure_info.interface_residues),
        }

    return result


def _estimate_bsa_from_seq(sequence: str, seed: int = 42) -> float:
    """Sequence-based BSA estimate (same as B5 step)."""
    import numpy as np
    rng = np.random.RandomState(seed + len(sequence))
    base_bsa = len(sequence) * 14
    noise = rng.normal(0, 100)
    return max(600, min(2000, base_bsa + noise))


def _estimate_sc_from_seq(sequence: str, seed: int = 42) -> float:
    """Sequence-based SC estimate (same as B5 step)."""
    import numpy as np
    rng = np.random.RandomState(seed + len(sequence) * 7)
    aromatic = sum(1 for aa in sequence if aa in "FYW")
    base = 0.55 + 0.02 * aromatic / max(len(sequence), 1) * 10
    noise = rng.normal(0, 0.05)
    return max(0.3, min(0.9, base + noise))

"""Shared utilities for ImmunoForge."""

import json
import logging
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = BASE_DIR / "config"


def load_config(config_path=None):
    """Load YAML configuration file."""
    if config_path is None:
        config_path = CONFIG_DIR / "default_config.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_json(data, path):
    """Save dict to JSON with numpy support."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)


def load_json(path):
    """Load JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_fasta(records, path):
    """Save FASTA file. records: list of (name, sequence)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for name, seq in records:
            f.write(f">{name}\n")
            for i in range(0, len(seq), 70):
                f.write(seq[i : i + 70] + "\n")


def read_fasta(path) -> list[dict]:
    """Read FASTA file. Returns list of {id, sequence}."""
    records = []
    current_id = None
    current_seq = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    records.append({"id": current_id, "sequence": "".join(current_seq)})
                current_id = line[1:].split()[0]
                current_seq = []
            elif line:
                current_seq.append(line)
    if current_id is not None:
        records.append({"id": current_id, "sequence": "".join(current_seq)})
    return records


def ensure_dirs(*dirs):
    """Create directories if they don't exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


# ── Amino acid molecular weights (monoisotopic) ──
AA_MW = {
    "A": 71.04,  "R": 156.10, "N": 114.04, "D": 115.03,
    "C": 103.01, "E": 129.04, "Q": 128.06, "G": 57.02,
    "H": 137.06, "I": 113.08, "L": 113.08, "K": 128.09,
    "M": 131.04, "F": 147.07, "P": 97.05,  "S": 87.03,
    "T": 101.05, "W": 186.08, "Y": 163.06, "V": 99.07,
}

# ── Standard genetic code ──
CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def compute_mw(sequence: str) -> float:
    """Compute molecular weight in Daltons."""
    water = 18.015
    return sum(AA_MW.get(aa, 110.0) for aa in sequence) + water


def compute_pi(sequence: str) -> float:
    """Estimate isoelectric point by bisection."""
    pka_pos = {"K": 10.5, "R": 12.4, "H": 6.0}
    pka_neg = {"D": 3.9, "E": 4.1, "C": 8.3, "Y": 10.1}

    def charge_at_ph(ph):
        c = 1.0 / (1 + 10 ** (ph - 9.69))  # N-term
        c -= 1.0 / (1 + 10 ** (2.34 - ph))  # C-term
        for aa in sequence:
            if aa in pka_pos:
                c += 1.0 / (1 + 10 ** (ph - pka_pos[aa]))
            if aa in pka_neg:
                c -= 1.0 / (1 + 10 ** (pka_neg[aa] - ph))
        return c

    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if charge_at_ph(mid) > 0:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 2)


def hydrophobic_fraction(sequence: str) -> float:
    """Fraction of hydrophobic residues."""
    n = sum(1 for aa in sequence if aa in "VILMFYW")
    return round(n / max(len(sequence), 1), 3)

"""
B1: Target Preparation — PDB/AlphaFold download, chain extraction,
contig & hotspot definition for RFdiffusion.

Supports both manual hotspot specification and automatic AF2BIND-style
hotspot prediction using ESM-2 attention maps.
"""

import logging
from pathlib import Path

import requests

from immunoforge.core.utils import save_json, ensure_dirs

logger = logging.getLogger(__name__)


def _read_sequence_from_pdb(pdb_path: str) -> str:
    """Extract amino acid sequence from PDB ATOM records (Cα only)."""
    three_to_one = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    residues = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                resname = line[17:20].strip()
                resseq = int(line[22:26].strip())
                aa = three_to_one.get(resname, "X")
                residues[resseq] = aa
    return "".join(residues[k] for k in sorted(residues))


def download_pdb(pdb_id: str, out_dir: str) -> Path | None:
    """Download PDB file from RCSB."""
    out_path = Path(out_dir) / f"{pdb_id}.pdb"
    if out_path.exists():
        logger.info(f"  {pdb_id}.pdb exists, skipping")
        return out_path
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        logger.info(f"  Downloaded {pdb_id}.pdb ({len(r.content)} bytes)")
        return out_path
    except Exception as e:
        logger.error(f"  Failed to download {pdb_id}: {e}")
        return None


def download_alphafold(af_id: str, out_dir: str) -> Path | None:
    """Download AlphaFold predicted structure."""
    out_path = Path(out_dir) / f"{af_id}.pdb"
    if out_path.exists():
        logger.info(f"  {af_id}.pdb exists, skipping")
        return out_path
    url = f"https://alphafold.ebi.ac.uk/files/{af_id}-model_v4.pdb"
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        logger.info(f"  Downloaded AlphaFold {af_id} ({len(r.content)} bytes)")
        return out_path
    except Exception as e:
        logger.error(f"  AlphaFold download failed for {af_id}: {e}")
        return None


def extract_chain(pdb_path: str, chain_id: str, out_path: str) -> int:
    """Extract a single chain from PDB file. Returns atom count."""
    lines = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")) and len(line) > 21 and line[21] == chain_id:
                lines.append(line)
    lines.append("END\n")
    Path(out_path).write_text("".join(lines))
    n_atoms = sum(1 for l in lines if l.startswith("ATOM"))
    logger.info(f"  Extracted chain {chain_id}: {n_atoms} atoms")
    return n_atoms


def generate_contig(chain: str, residue_range: str, binder_length: str) -> str:
    """Generate RFdiffusion contig string.

    Format: ChainResStart-ResEnd/0 binderLenMin-binderLenMax
    """
    return f"{chain}{residue_range}/0 {binder_length}"


def generate_hotspot_args(chain: str, residues: list[int]) -> str:
    """Generate RFdiffusion hotspot argument string.

    Format: [A120,A145,A147,...]
    """
    return "[" + ",".join(f"{chain}{r}" for r in residues) + "]"


def main(config: dict) -> dict:
    """Execute B1: Target Preparation step."""
    logger.info("  B1: Target Preparation & Epitope Definition")

    structures_dir = Path(config.get("paths", {}).get("structures_dir", "data/structures"))
    ensure_dirs(structures_dir)

    results = {"targets": {}}
    binder_length = config.get("rfdiffusion", {}).get("binder_length", "70-100")
    auto_hotspot_cfg = config.get("hotspot_prediction", {})
    auto_hotspot = auto_hotspot_cfg.get("enabled", False)
    hotspot_method = auto_hotspot_cfg.get("method", "auto")
    hotspot_top_k = auto_hotspot_cfg.get("top_k", 10)

    for key in ("target1", "target2"):
        tgt = config.get("targets", {}).get(key, {})
        if not tgt:
            continue

        name = tgt.get("name", key)
        logger.info(f"  Processing target: {name}")

        # Download structure
        source = tgt.get("source", "pdb")
        pdb_path = None
        if source == "alphafold" and tgt.get("af_id"):
            pdb_path = download_alphafold(tgt["af_id"], str(structures_dir))
        elif tgt.get("pdb_id"):
            pdb_path = download_pdb(tgt["pdb_id"], str(structures_dir))

        # Extract chain
        chain = tgt.get("chain", "A")
        chain_path = structures_dir / f"{name}_chain{chain}.pdb"
        if pdb_path and pdb_path.exists():
            extract_chain(str(pdb_path), chain, str(chain_path))

        # Generate RFdiffusion inputs
        residue_range = tgt.get("residue_range", "1-100")
        hotspots = tgt.get("hotspot_residues", [])

        # Auto-hotspot prediction when no manual hotspots or auto mode enabled
        hotspot_prediction_info = None
        if auto_hotspot and chain_path.exists():
            logger.info(f"  Running automatic hotspot prediction for {name}...")
            try:
                from immunoforge.core.hotspot_predictor import predict_hotspots
                seq = _read_sequence_from_pdb(str(chain_path))
                if seq:
                    parts = residue_range.split("-")
                    res_start = int(parts[0]) if len(parts) == 2 else 1
                    res_end = int(parts[1]) if len(parts) == 2 else len(seq)
                    hp_result = predict_hotspots(
                        seq,
                        residue_start=res_start,
                        residue_end=res_end,
                        top_k=hotspot_top_k,
                        method=hotspot_method,
                    )
                    auto_hotspots = hp_result.top_k_indices
                    hotspot_prediction_info = {
                        "method": hp_result.method,
                        "predicted_hotspots": auto_hotspots,
                        "scores": {
                            str(i): s for i, s in zip(
                                hp_result.residue_indices, hp_result.scores
                            ) if i in auto_hotspots
                        },
                    }
                    if not hotspots:
                        # Use predicted hotspots when no manual ones
                        hotspots = auto_hotspots
                        logger.info(
                            f"  Auto-hotspot ({hp_result.method}): "
                            f"{hotspots}"
                        )
                    else:
                        logger.info(
                            f"  Auto-hotspot prediction: {auto_hotspots} "
                            f"(manual hotspots retained: {hotspots})"
                        )
            except Exception as e:
                logger.warning(f"  Auto-hotspot prediction failed: {e}")

        contig = generate_contig(chain, residue_range, binder_length)
        hotspot_str = generate_hotspot_args(chain, hotspots) if hotspots else ""

        target_result = {
            "structure_path": str(chain_path),
            "chain": chain,
            "residue_range": residue_range,
            "contig": contig,
            "hotspot_args": hotspot_str,
            "source": source,
        }
        if hotspot_prediction_info:
            target_result["hotspot_prediction"] = hotspot_prediction_info

        results["targets"][name] = target_result

    # Save results
    output_dir = Path(config.get("paths", {}).get("output_dir", "outputs"))
    ensure_dirs(output_dir)
    save_json(results, output_dir / "B1_target_prep.json")
    logger.info(f"  B1 complete: {len(results['targets'])} targets prepared")
    return results

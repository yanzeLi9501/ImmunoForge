"""
B2: RFdiffusion Backbone Generation.

Generates protein backbone scaffolds using RFdiffusion.
Supports both single-target and dual-target designs.

When GPU/RFdiffusion is not available, generates preparation scripts
for offline execution.
"""

import json
import logging
import subprocess
import shutil
from pathlib import Path

from immunoforge.core.utils import save_json, ensure_dirs

logger = logging.getLogger(__name__)


def _check_rfdiffusion() -> bool:
    """Check if RFdiffusion is available."""
    try:
        import importlib
        importlib.import_module("rfdiffusion")
        return True
    except ImportError:
        return False


def generate_rfdiffusion_script(
    target_name: str,
    structure_path: str,
    contig: str,
    hotspot_args: str,
    n_designs: int,
    output_dir: str,
    diffusion_steps: int = 50,
) -> str:
    """Generate RFdiffusion inference command."""
    cmd = (
        f"python -m rfdiffusion.inference.model_runners.InferenceRunner "
        f"inference.output_prefix={output_dir}/rfd_{target_name} "
        f"inference.input_pdb={structure_path} "
        f"contigmap.contigs=['{contig}'] "
        f"inference.num_designs={n_designs} "
        f"diffuser.T={diffusion_steps} "
    )
    if hotspot_args:
        cmd += f"ppi.hotspot_res={hotspot_args} "

    return cmd


def generate_dual_target_script(
    target1_path: str,
    target2_path: str,
    contig1: str,
    contig2: str,
    n_designs: int,
    output_dir: str,
    strategy: str = "shared_helix",
) -> str:
    """Generate dual-target RFdiffusion command."""
    cmd = (
        f"# Dual-target design strategy: {strategy}\n"
        f"python -m rfdiffusion.inference.model_runners.InferenceRunner "
        f"inference.output_prefix={output_dir}/dual_{strategy} "
        f"inference.input_pdb={target1_path},{target2_path} "
        f"contigmap.contigs=['{contig1}','{contig2}'] "
        f"inference.num_designs={n_designs} "
    )
    return cmd


def main(config: dict) -> dict:
    """Execute B2: RFdiffusion backbone generation."""
    logger.info("  B2: RFdiffusion Backbone Generation")

    rfd_cfg = config.get("rfdiffusion", {})
    n_designs = rfd_cfg.get("n_designs_per_target", 50)
    diff_steps = rfd_cfg.get("diffusion_steps", 50)
    dual_designs = rfd_cfg.get("dual_target_designs", 15)
    dual_strategies = rfd_cfg.get("dual_strategies", 3)

    output_dir = Path(config.get("paths", {}).get("output_dir", "outputs"))
    backbone_dir = output_dir / "rfdiffusion_backbones"
    ensure_dirs(backbone_dir)

    # Load B1 results
    b1_path = output_dir / "B1_target_prep.json"
    if b1_path.exists():
        with open(b1_path) as f:
            b1 = json.load(f)
        targets = b1.get("targets", {})
    else:
        targets = {}

    has_rfd = _check_rfdiffusion()
    scripts = []
    result = {
        "rfdiffusion_available": has_rfd,
        "n_designs_per_target": n_designs,
        "diffusion_steps": diff_steps,
        "commands": [],
        "backbones_generated": 0,
    }

    # Single-target designs
    for name, tgt in targets.items():
        cmd = generate_rfdiffusion_script(
            target_name=name,
            structure_path=tgt["structure_path"],
            contig=tgt["contig"],
            hotspot_args=tgt.get("hotspot_args", ""),
            n_designs=n_designs,
            output_dir=str(backbone_dir),
            diffusion_steps=diff_steps,
        )
        scripts.append(cmd)
        result["commands"].append({"target": name, "type": "single", "command": cmd})

        if has_rfd:
            logger.info(f"  Running RFdiffusion for {name}...")
            try:
                subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                result["backbones_generated"] += n_designs
            except subprocess.CalledProcessError as e:
                logger.error(f"  RFdiffusion failed for {name}: {e}")

    # Dual-target designs
    target_names = list(targets.keys())
    if len(target_names) >= 2:
        t1, t2 = target_names[0], target_names[1]
        strategies = ["shared_helix", "tandem_domain", "beta_arch"][:dual_strategies]
        designs_per = dual_designs // max(len(strategies), 1)

        for strategy in strategies:
            cmd = generate_dual_target_script(
                targets[t1]["structure_path"],
                targets[t2]["structure_path"],
                targets[t1]["contig"],
                targets[t2]["contig"],
                n_designs=designs_per,
                output_dir=str(backbone_dir),
                strategy=strategy,
            )
            scripts.append(cmd)
            result["commands"].append({
                "targets": [t1, t2],
                "type": "dual",
                "strategy": strategy,
                "command": cmd,
            })

    # Save inference script
    script_path = backbone_dir / "run_rfdiffusion.sh"
    script_path.write_text("#!/bin/bash\n" + "\n\n".join(scripts) + "\n")
    logger.info(f"  RFdiffusion script saved: {script_path}")

    if not has_rfd:
        logger.info("  RFdiffusion not installed — scripts generated for offline execution")
        logger.info(f"  Run: bash {script_path}")

    save_json(result, output_dir / "B2_rfdiffusion.json")
    return result

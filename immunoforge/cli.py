"""ImmunoForge CLI — command-line entry point."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="immunoforge",
        description="ImmunoForge: AI-driven immunological protein design platform",
    )
    sub = parser.add_subparsers(dest="command")

    # ── run ──
    run_p = sub.add_parser("run", help="Run the design pipeline")
    run_p.add_argument(
        "-c", "--config", default=None,
        help="Path to YAML config (default: config/default_config.yaml)",
    )
    run_p.add_argument(
        "-s", "--steps", nargs="*", default=None,
        help="Pipeline steps to run (e.g. B1 B3 B5). Omit for all.",
    )
    run_p.add_argument(
        "--species", default=None,
        help="Override species (mouse / human / cynomolgus)",
    )
    run_p.add_argument(
        "-o", "--output", default=None,
        help="Override output directory",
    )

    # ── server ──
    srv_p = sub.add_parser("server", help="Start the local web server")
    srv_p.add_argument("--host", default="0.0.0.0", help="Bind host (default 0.0.0.0)")
    srv_p.add_argument("--port", type=int, default=8000, help="Bind port (default 8000)")
    srv_p.add_argument("--reload", action="store_true", help="Auto-reload on changes")

    # ── qc ──
    qc_p = sub.add_parser("qc", help="Run sequence QC on a FASTA file")
    qc_p.add_argument("fasta", help="Input FASTA file")
    qc_p.add_argument("-o", "--output", default=None, help="Output JSON path")

    # ── targets ──
    tgt_p = sub.add_parser("targets", help="Browse the target database")
    tgt_p.add_argument("--cell-type", default=None, help="Filter by cell type")
    tgt_p.add_argument("--species", default=None, help="Filter by species")
    tgt_p.add_argument("--benchmark", action="store_true", help="Show benchmark targets only")

    # ── codon ──
    cod_p = sub.add_parser("codon", help="Optimize codons for a protein sequence")
    cod_p.add_argument("sequence", help="Amino acid sequence (or @file.fasta)")
    cod_p.add_argument("--species", default="mouse", help="Target species")
    cod_p.add_argument("--system", default="vaccinia", help="Expression system")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "server":
        _cmd_server(args)
    elif args.command == "qc":
        _cmd_qc(args)
    elif args.command == "targets":
        _cmd_targets(args)
    elif args.command == "codon":
        _cmd_codon(args)


# ────────────────────────────────────────────
# Subcommand implementations
# ────────────────────────────────────────────

def _cmd_run(args):
    from immunoforge.core.utils import load_config
    from immunoforge.pipeline.runner import run_pipeline

    config_path = args.config or str(
        Path(__file__).resolve().parent.parent / "config" / "default_config.yaml"
    )
    config = load_config(config_path)

    if args.species:
        species_cfg = config.get("species")
        if isinstance(species_cfg, dict):
            species_cfg["default"] = args.species
        else:
            config["species"] = {"default": args.species}
    if args.output:
        paths_cfg = config.setdefault("paths", {})
        paths_cfg["output_dir"] = args.output
        paths_cfg["logs_dir"] = str(Path(args.output) / "logs")

    steps = args.steps
    species_cfg = config.get("species", "mouse")
    species_name = species_cfg.get("default", "mouse") if isinstance(species_cfg, dict) else species_cfg
    print(f"[ImmunoForge] Running pipeline — species={species_name}")
    summary = run_pipeline(config, steps=steps)

    print("\n=== Pipeline Summary ===")
    for info in summary.get("step_results", []):
        step_key = info.get("step", "unknown")
        status = info.get("status", "unknown")
        marker = "✓" if status == "completed" else "✗"
        print(f"  {marker} {step_key}: {status}")


def _cmd_server(args):
    from immunoforge.server.app import run_server
    print(f"[ImmunoForge] Starting server at http://{args.host}:{args.port}")
    run_server(host=args.host, port=args.port, reload=args.reload)


def _cmd_qc(args):
    import json
    from immunoforge.core.utils import read_fasta
    from immunoforge.core.sequence_qc import batch_qc

    records = read_fasta(args.fasta)
    sequences = [(r["id"], r["sequence"]) for r in records]
    results = batch_qc(sequences)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"QC results written to {args.output}")
    else:
        print(f"Total: {results['total']}  Passed: {results['n_passed']}  Failed: {results['n_failed']}")
        for r in results.get("failed", []):
            print(f"  FAIL {r['id']}: {', '.join(r.get('failures', []))}")


def _cmd_targets(args):
    from immunoforge.core.target_db import (
        search_targets, get_benchmark_targets,
    )

    if args.benchmark:
        targets = get_benchmark_targets()
    else:
        targets = search_targets(
            cell_type=args.cell_type,
            species=args.species,
        )

    print(f"{'Name':<20} {'Species':<15} {'Cell':<12} {'Known Binders':<28} {'Benchmark'}")
    print("-" * 91)
    for t in targets:
        kd = ", ".join(f"{b['name']}({b['kd_nM']}nM)" for b in t.known_binders) if t.known_binders else "—"
        bm = f"{t.benchmark_value}★" if t.benchmark_value else "—"
        print(f"{t.name:<20} {t.species:<15} {t.cell_type:<12} {kd:<28} {bm}")


def _cmd_codon(args):
    from immunoforge.core.codon_opt import full_codon_optimization

    seq = args.sequence
    if seq.startswith("@"):
        from immunoforge.core.utils import read_fasta
        records = read_fasta(seq[1:])
        if not records:
            print("No sequences in file.")
            return
        seq = records[0]["sequence"]

    result = full_codon_optimization(
        protein_seq=seq,
        species=args.species,
        system=args.system,
    )

    print(f"Species:     {result['species']}")
    print(f"System:      {result['expression_system']}")
    print(f"GC content:  {result['gc_content_cds'] * 100:.1f}%")
    print(f"T5NT free:   {'Yes' if not result.get('has_t5nt') else 'No'}")
    print(f"CDS length:  {len(result.get('cds_dna', ''))} bp")
    print(f"\nCDS DNA:\n{result.get('cds_dna', '')}")


if __name__ == "__main__":
    main()

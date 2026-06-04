# ImmunoForge — De Novo Immune Cell Targeting Protein Design Platform

> *"Forging de novo binders for immune cell biology"*

ImmunoForge is an automated pipeline for de novo protein binder design targeting
immune cell surface molecules, with built-in support for bispecific bridging
molecule design, codon optimization, and command-line execution.
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/f0a3828e-efb7-43d2-9721-896b19f98ca9" />

## Features

- **De novo binder design** via RFdiffusion + ProteinMPNN
- **Multi-species support**: Mouse / Human / Cynomolgus macaque / Camelid (VHH)
- **Built-in immune target database**: 24 receptor entries (27 binder–receptor pairs) spanning T cell, NK cell, DC, macrophage, B cell
- **Sequence QC**: protease site, toxicity, APR, Cys parity, pI filtering
- **Domain-Aware Consensus Scoring (DACS)**: multi-method consensus K_D (BSA / PRODIGY / Rosetta / Boltz-2 ipTM), with a real-data-optimised **DACS-Sig** model (held-out MALE 1.72 → **0.37**) and an optional ESM-2 sequence-reliability filter
- **Bispecific geometry filter**: synapse-distance constraint (13–15 nm)
- **Codon optimization**: Vaccinia / Lentivirus / mRNA-LNP / AAV / CHO / HEK293T cassette modes

## Quick Start

```bash
# Install the package in editable mode
pip install -e .

# 1) Browse built-in targets
immunoforge targets --cell-type "T cell"

# 2) Run QC on a FASTA file
printf ">toy\nMSQAKKDPLDPATAQLASARGT\n" > toy.fasta
immunoforge qc toy.fasta

# 3) Build a codon-optimized cassette
immunoforge codon MSQAKKDPLDPATAQLASARGT --species human --system mrna
```

`immunoforge run` is intended for configured research environments. The full
pipeline can call external tools such as RFdiffusion, ProteinMPNN, ESMFold /
AF2, and Boltz-2 depending on the selected steps and config flags. For a first
local smoke test, start with `targets`, `qc`, and `codon`.

## Installation Modes

```bash
# Base install
pip install -e .

# Development tools (pytest, ruff)
pip install -e ".[dev]"

# Optional GPU extras
pip install -e ".[gpu]"
```

The console script exposed by the package is:

- `immunoforge` — main CLI

To inspect the available commands at any time, use `immunoforge --help`
(standard), `immunoforge -h`, or the short alias `immunoforge --h`. The same
help flags also work on subcommands such as `immunoforge run --h`.

For a detailed walkthrough of CLI arguments and every active configuration
section in `config/default_config.yaml`, see
[`notebooks/parameter_reference.ipynb`](notebooks/parameter_reference.ipynb).

## CLI Reference

### `immunoforge run`

Run the pipeline with a YAML config.

| Argument | Meaning | Notes |
|----------|---------|-------|
| `-c`, `--config PATH` | Config file to load | Defaults to `config/default_config.yaml` |
| `-s`, `--steps B1 B3 ...` | Explicit step list | Use actual CLI step IDs listed below |
| `--species NAME` | Override `species.default` | Useful for quick one-off runs |
| `-o`, `--output DIR` | Override `paths.output_dir` | Writes pipeline summary and step outputs there |

Example:

```bash
immunoforge run --config config/default_config.yaml --steps B1 --species mouse --output outputs_smoke
```

### `immunoforge targets`

Browse the built-in immune target database.

| Argument | Meaning |
|----------|---------|
| `--cell-type TEXT` | Filter by cell type such as `T cell`, `NK cell`, `DC`, `B cell`, `macrophage`, `tumour` |
| `--species TEXT` | Filter by species |
| `--benchmark` | Show benchmark-tagged targets only |

Examples:

```bash
immunoforge targets --cell-type "T cell"
immunoforge targets --species human --benchmark
```

### `immunoforge qc`

Run sequence QC on a FASTA file.

| Argument | Meaning |
|----------|---------|
| `FASTA` | Input FASTA path |
| `-o`, `--output PATH` | Optional JSON output path |

Example:

```bash
immunoforge qc toy.fasta --output qc_results.json
```

### `immunoforge codon`

Optimize a protein sequence and build an expression cassette.

| Argument | Meaning | Accepted values |
|----------|---------|-----------------|
| `sequence` | Amino-acid sequence or `@file.fasta` | FASTA mode reads the first record |
| `--species` | Codon usage table | `mouse`, `human`, `cynomolgus` |
| `--system` | Expression cassette mode | `vaccinia`, `aav`, `lentivirus`, `mrna`, `cho`, `hek293t` |

Examples:

```bash
immunoforge codon MSQAKKDPLDPATAQLASARGT --species human --system vaccinia
immunoforge codon @toy.fasta --species human --system cho
```

`vaccinia` adds T5NT cleanup and a longer cassette scaffold; `aav` and
`lentivirus` use promoter placeholders; `mrna`, `cho`, and `hek293t` use the
generic Kozak + signal peptide + CDS + stop-codon builder in
`immunoforge.core.codon_opt`.

## Project Structure

```
immunoforge/
├── core/               # Core computation modules
│   ├── target_db.py    # Immune cell target database (34 entries)
│   ├── sequence_qc.py  # Sequence quality control
│   ├── affinity.py     # DACS 4-method binding affinity prediction
│   ├── ranking.py      # Candidate scoring & ranking (v6 weights)
│   ├── linker_design.py# Bispecific linker design + synapse geometry filter
│   ├── codon_opt.py    # Codon optimization engine (6 CLI cassette modes)
│   └── utils.py        # Shared utilities
├── pipeline/           # Pipeline step orchestration
│   ├── steps/          # B1–B9 step modules
│   └── runner.py       # Pipeline orchestrator
├── notebooks/          # Detailed parameter reference notebook
├── cli.py              # CLI entry point
└── __init__.py
config/                 # YAML configuration files
tests/                  # Unit tests
```

## Pipeline Step IDs Used by `immunoforge run`

The CLI step IDs come from `immunoforge.pipeline.runner.STEP_REGISTRY`.

| Step ID | Module purpose | Runtime notes |
|---------|----------------|---------------|
| `B1` | Target preparation, structure download, chain extraction, hotspot setup | Needs PDB / AlphaFold downloads unless structures already exist |
| `B2` | RFdiffusion backbone generation | External RFdiffusion install required |
| `B3` | ProteinMPNN sequence design | External ProteinMPNN install required |
| `B3a` | Multi-stage sequence optimization | Optional optimization pass |
| `B3b` | Sequence QC | Consumes `B3_sequence_design.json` |
| `B4` | Structure validation | May require ESMFold / AF2-related tooling depending on config |
| `B5` | Binding affinity prediction | Wraps DACS-related affinity analysis |
| `B6` | Candidate ranking | Consumes `B5_binding_affinity.json` |
| `B7` | Codon optimization and cassette generation | Consumes `B6_candidate_ranking.json` |
| `B8` | Deliverables and report export | Consumes ranked candidates |
| `B4b`, `B4c`, `B5b`, `B5c`, `B6b`, `B7b`, `B9`, `B9a`, `B9b` | Optional / extension steps | Intended for advanced or GPU-backed workflows |

When `--steps` is omitted, `immunoforge run` decides the step list from config:

- `sequence_optimization.enabled: true` adds `B3a`
- `af3_validation.enabled: true` adds `B9a`
- `affinity_maturation.enabled: true` adds `B9b`

## Configuration Guide

The default configuration lives in `config/default_config.yaml`. The most
important top-level sections are:

| Section | What it controls | Common fields to edit |
|---------|------------------|-----------------------|
| `project` | Metadata only | `name`, `version`, `description` |
| `species` | Default organism for runs | `default` |
| `targets` | Input targets for B1 | `name`, `source`, `pdb_id`, `af_id`, `chain`, `residue_range`, `hotspot_residues` |
| `rfdiffusion` | Backbone generation search space | `n_designs_per_target`, `binder_length`, `diffusion_steps`, `hotspot_guidance` |
| `proteinmpnn` | Sequence sampling | `sampling_temperature`, `sequences_per_backbone`, `design_chain_only` |
| `sequence_qc` | Filtering thresholds | protease motifs, toxicity checks, APR rule, cysteine parity, pI bounds |
| `structure_validation` | Post-design structure screen | `method`, `dual_evaluation`, `high_threshold`, `medium_threshold` |
| `hotspot_prediction` | Auto-hotspot behavior in B1 | `enabled`, `method`, `top_k` |
| `sequence_optimization` | Optional B3a search intensification | `temperatures`, `seqs_per_temperature`, `stage*_...` fields |
| `affinity` | Affinity-analysis settings | `methods`, `consensus`, `confidence_threshold_log10`, `dacs_mode`, `esm2_gate` |
| `ranking` | Candidate selection controls | `top_n`, `synthesis_top_n`, optional `weights` override |
| `codon_optimization` | DNA design settings | codon tables, GC target, site-avoid rules, signal peptides |
| `paths` | Output locations | `output_dir`, `data_dir`, `structures_dir`, `logs_dir` |
| `af3_validation` | Optional Boltz-2 / AF3 validation | `enabled`, `top_n`, `recycling_steps`, `sampling_steps` |
| `af2_filter` | Optional AF2-multimer filter | `enabled`, `max_candidates`, `iptm_*_threshold` |
| `affinity_maturation` | Optional post-ranking maturation | `enabled`, `kd_threshold_nM`, `target_kd_nM`, `max_generations` |
| `denovo_profile` | Alternative settings for de novo binder campaigns | nested overrides for RFdiffusion, MPNN, ranking, maturation |

## Parameter Usage Notes

### DACS calibration parameters

These are hard-coded in `immunoforge.core.affinity` and affect the
class-dependent log-K_D offsets used by consensus scoring.

| Parameter | Value | Used in |
|-----------|-------|---------|
| `delta_ppi` | −2.4887 | PPI / antibody-like calibration branch |
| `delta_denovo` | −2.2122 | De novo miniprotein calibration branch |
| `model_hash` | `5a076f7aa039` | Canonical calibration artifact identifier |
| `N_train` | 15 | Training set size for the published fit |
| `N_holdout` | 3 | Held-out benchmark size |

Published summary metrics: train MALE = 0.96, LOO MALE = 1.16, LOO ρ = 0.25,
holdout MALE = 1.72.

### DACS-Sig (real-data optimization)

`DACS-Sig` is the real-data-optimised consensus model. It directly fixes the
highest-priority weakness of the publication DACS (v5): on the fixed held-out
split (Ipilimumab / SIRPα / LCB3), v5 over-fit two free class offsets on top of
a structural ipTM→K_D term and scored *worse* than the raw pre-DACS baseline.
DACS-Sig instead uses adaptive per-class/per-strength weights over the three
sequence/energy experts only (BSA / PRODIGY / Rosetta — the ipTM term is dropped
as a direct K_D expert), BSA winsorization, a single strongly-shrunk **global**
log-K_D offset, and a 0.5 ensemble with the baseline B0 for variance reduction.
All hyper-parameters were selected by TRAIN-only leave-one-out CV; the held-out
split is a frozen out-of-sample outcome.

Held-out (Ipili / SIRPα / LCB3) comparison:

| Metric | B0 baseline | DACS v5 (publication) | **DACS-Sig (v6)** |
|--------|-------------|-----------------------|-------------------|
| Held-out MALE | 0.9811 | 1.7210 | **0.3651** |
| ≤ 1.5-log entries | 3/3 | 1/3 | **3/3** |
| Held-out ρ | 0.50 | 1.00 | **1.00** |
| LOO-CV MALE (N=18) | 1.766 | — | **1.372** |

**Optional ESM-2 gate (v7).** With `esm2_gate: true` (or `--esm2-gate`), a
per-entry ensemble weight is modulated by the binder's ESM-2 masked
pseudo-perplexity — a sequence-only reliability signal that never sees the
target and is never a K_D term. The manuscript reports the gate as an honest
negative ablation (it improves point estimates but removes 96-split
significance), so it is **off by default** and offered as an opt-in filter. It
requires the optional `[gpu]` extra (`torch` + `fair-esm`); when ESM-2 is
unavailable the gate disables itself automatically and falls back to v6.

Select the model from config (`affinity.dacs_mode: dacs_sig` is the recommended
default) or per-run from the CLI:

```bash
# Real-data-optimised DACS-Sig (recommended; also the config default)
immunoforge run --dacs-mode dacs_sig

# Reproduce the publication DACS (v5)
immunoforge run --dacs-mode v5

# DACS-Sig + optional ESM-2 reliability gate (needs the [gpu] extra)
immunoforge run --dacs-mode dacs_sig --esm2-gate
```

Frozen benchmark inputs and calibration constants ship with the package at
`immunoforge/core/data/dacs_sig_benchmark.json` (inputs only — no private data),
so the held-out numbers above are reproduced by `tests/test_dacs_sig.py` on a
clean install.

### Boltz-2 ipTM -> K_D conversion

Used by the 4-method DACS path when structural ipTM is available.

```python
log10(K_D / nM) = 5.0 - 6.0 * iptm
# iptm=0.5 -> 100 nM
# iptm=0.7 -> 6.31 nM
# iptm=0.9 -> 0.50 nM (floor-clamped)
# upper clamp: 1e6 nM
```

### Ranking parameters

There are two places to look:

1. `immunoforge.core.ranking.RankingWeights` contains the canonical reference
   weights for the adaptive ranking logic.
2. `config/default_config.yaml -> ranking.weights` is an explicit pipeline
   override. If that block is present, B6 passes it directly to
   `rank_candidates()`.

If you want the pipeline to follow a custom ranking scheme, edit the
`ranking.weights` block. If you want to rely on the code defaults, remove that
override block or provide the exact keys expected by `compute_composite_score()`
such as `plddt`, `ddg`, `kd`, `mpnn_score`, `bsa`, `shape_complementarity`,
`baseline`, `structural_score`, `iptm_proxy`, and `interface_pae`.

### Sequence QC parameters

`sequence_qc` controls the hard filters applied by `immunoforge qc` and B3b:

- protease motifs: furin, thrombin, enterokinase, MMP, cathepsin B
- toxicity motifs: poly-cationic segments and NLS-like motifs
- APR rule: rejects long hydrophobic aggregation-prone stretches
- cysteine parity: rejects odd cysteine counts
- pI window: defaults to `4.5 <= pI <= 10.5`

### Codon and cassette parameters

`codon_optimization` defines:

- host species codon tables
- GC target range (`gc_content_target`)
- restriction sites to avoid
- signal peptides (`il2_leader`, `igk_leader`)
- named expression-system metadata

At the CLI level, `--system` determines which cassette builder branch is used.
This is the main user-facing switch for DNA output behavior.

### Output and run controls

- `--species` on `immunoforge run` overrides `species.default`
- `--output` on `immunoforge run` overrides `paths.output_dir`
- `--steps` limits the run to specific step IDs

## Reproducibility Reference

- Immune target database: 24 receptor entries, 27 documented binder-receptor pairs
- DACS modes: publication v5 (4-method, 2-class) and real-data-optimised DACS-Sig v6 (held-out MALE 0.37) + optional ESM-2 gate (v7)
- Bispecific synapse geometry window: **13-15 nm**
- Codon cassette modes exposed by CLI: `vaccinia`, `aav`, `lentivirus`, `mrna`, `cho`, `hek293t`

## Requirements

- Python 3.10+
- (Optional) NVIDIA GPU with CUDA ≥ 12.0 for RFdiffusion / ProteinMPNN / Boltz-2 inference

## License

ImmunoForge is distributed under a **custom MIT-style Non-Commercial license**.
Use, modification, and redistribution are allowed for research, academic,
personal, and internal evaluation purposes, but **commercial use is prohibited**
without prior written permission. See [`LICENSE`](LICENSE) for the full terms.

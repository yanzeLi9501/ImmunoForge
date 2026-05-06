# ImmunoForge — De Novo Immune Cell Targeting Protein Design Platform

> *"Forging de novo binders for immune cell biology"*

ImmunoForge is an automated pipeline for de novo protein binder design targeting
immune cell surface molecules, with built-in support for bispecific bridging
molecule design, codon optimization, and local web-based deployment.

## Features

- **De novo binder design** via RFdiffusion + ProteinMPNN
- **Multi-species support**: Mouse / Human / Cynomolgus macaque / Camelid (VHH)
- **Built-in immune target database**: 24 receptor entries (27 binder–receptor pairs) spanning T cell, NK cell, DC, macrophage, B cell
- **Sequence QC**: protease site, toxicity, APR, Cys parity, pI filtering
- **Domain-Aware Consensus Scoring (DACS)**: 4-method consensus K_D (BSA / PRODIGY / Rosetta / Boltz-2 ipTM), 2-class calibration
- **Bispecific geometry filter**: synapse-distance constraint (13–15 nm)
- **Codon optimization**: Vaccinia / mRNA-LNP / AAV / CHO / HEK293T expression cassettes
- **Web UI**: local FastAPI server with interactive dashboard

## Quick Start

```bash
# Install
pip install -e .

# Run CLI pipeline
immunoforge run --config config/default_config.yaml --species mouse

# Start web server
immunoforge-server
# Open http://localhost:8000
```

## Project Structure

```
immunoforge/
├── core/               # Core computation modules
│   ├── target_db.py    # Immune cell target database (34 entries)
│   ├── sequence_qc.py  # Sequence quality control
│   ├── affinity.py     # DACS 4-method binding affinity prediction
│   ├── ranking.py      # Candidate scoring & ranking (v6 weights)
│   ├── linker_design.py# Bispecific linker design + synapse geometry filter
│   ├── codon_opt.py    # Codon optimization engine (5 delivery systems)
│   └── utils.py        # Shared utilities
├── pipeline/           # Pipeline step orchestration
│   ├── steps/          # B1–B9 step modules
│   └── runner.py       # Pipeline orchestrator
├── server/             # FastAPI web server
│   ├── app.py          # Server entry point
│   ├── api.py          # REST API routes
│   ├── templates/      # Jinja2 HTML templates
│   └── static/         # CSS/JS assets
├── cli.py              # CLI entry point
└── __init__.py
config/                 # YAML configuration files
tests/                  # Unit tests
```

## Pipeline Steps Reference

The pipeline is organized into sequential steps B1–B9. Steps marked **[production]** are
fully exercised in the manuscript. Steps marked **[experimental]** are implemented but were
not used for primary results; steps marked **[internal use only]** are helper sub-steps.

| Step ID | Name | Status | Notes |
|---------|------|--------|-------|
| B1 | Target selection & database query | **production** | `immunoforge targets` CLI |
| B2 | Sequence QC | **production** | `immunoforge qc` CLI |
| B3a | RFdiffusion backbone generation | **production** | Requires GPU + RFdiffusion install |
| B3b | ProteinMPNN sequence design | **production** | Requires ProteinMPNN install |
| B4a | AlphaFold2/AF-Multimer structure prediction | **production** | Requires ColabFold/AF2 install |
| B4b | Boltz-2 structure prediction | **experimental** | Optional structural scorer; ipTM used by DACS 4-method; not run for all candidates in the manuscript benchmark |
| B5a | PRODIGY binding affinity | **production** | Part of DACS 3-method baseline |
| B5b | Rosetta REF2015 ΔΔG | **experimental** | Integrated but computationally expensive; used on the benchmark set only |
| B5c | BSA regression scorer | **experimental** | Empirical linear model; used on benchmark set; accuracy lower than PRODIGY/Rosetta on denovo class |
| B6a | DACS 3-method consensus K_D | **production** | Default scoring path when ipTM unavailable |
| B6b | DACS 4-method consensus K_D (+ Boltz-2) | **experimental** | Used only when B4b ipTM is available; this is the publication-mode DACS |
| B7a | Candidate ranking (v6 weights) | **production** | composite_iptm=0.20, pTM=0.12, iPAE=0.08, KD=0.12, MPNN=0.12 |
| B7b | Active-learning recalibration interface | **internal use only** | Operational; exercised only against the frozen 18-entry benchmark as a reproducibility check in this study |
| B8 | Bispecific assembly & synapse geometry filter | **production** | 13–15 nm WLC linker constraint |
| B9 | Expression cassette generation | **production** | 5 delivery systems |
| B9a | Full pipeline integration test (Round 6) | **internal use only** | End-to-end smoke test with Boltz-2; results reported in Supplementary Table 4 Panel f |

> **Reproducibility note**: B4b/B5b/B5c/B6b/B7b/B9a were run on the GPU server
> (`gpu-server`, RTX 4090 D) and are not invoked by default in `immunoforge run`.
> Set `config.steps` in `default_config.yaml` to enable specific steps.

## Key Parameters (Reproducibility Reference)

### DACS Calibration (2-class, N=18)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `delta_ppi` | −1.5659 | Log-shift for PPI/ab class (optimized on N=15 train set) |
| `delta_denovo` | −3.4742 | Log-shift for de novo class |
| `N_train` | 15 | Training entries |
| `N_holdout` | 3 | Held-out entries |
| `model_hash` | `9d7291409dd9` | SHA-12 of calibration parameters JSON |

Train MALE = 0.89; LOO MALE = 1.02; LOO ρ = 0.39; Holdout MALE = 1.07

### Boltz-2 ipTM → K_D Conversion

```python
# Fixed-anchor formula (A, B not fitted):
log10(K_D / nM) = A - B * iptm   # A=5.0, B=6.0
# Anchors: iptm=0.5 → 100 nM;  iptm=0.7 → 6.31 nM;  iptm=0.9 → 0.50 nM
# Ceiling: 1e6 nM (1 mM)
```

### Candidate Ranking Weights (v6)

| Feature | Weight |
|---------|--------|
| complex_iptm | 0.20 |
| interface_ptm | 0.12 |
| interface_pae | 0.08 |
| kd | 0.12 |
| mpnn_score | 0.12 |
| bsa | 0.10 |
| prodigy_kd | 0.10 |
| rosetta_ddg | 0.08 |
| qc_pass | 0.08 |

### Bispecific Geometry Filter

Synapse distance window: **13–15 nm** (WLC model, κT = 4.11 pN·nm at 25 °C)

## Requirements

- Python 3.10+
- (Optional) NVIDIA GPU with CUDA ≥ 12.0 for RFdiffusion / ProteinMPNN / Boltz-2 inference

## License

MIT

# ImmunoForge — De Novo Immune Cell Targeting Protein Design Platform

> *"Forging de novo binders for immune cell biology"*

ImmunoForge is an automated pipeline for de novo protein binder design targeting
immune cell surface molecules, with built-in support for bispecific bridging
molecule design, codon optimization, and local web-based deployment.

## Features

- **De novo binder design** via RFdiffusion + ProteinMPNN
- **Multi-species support**: Mouse / Human / Cynomolgus monkey
- **Built-in immune target database**: T cell, NK cell, DC, macrophage, B cell surface proteins
- **Sequence QC**: protease site, toxicity, APR, Cys parity, pI filtering
- **Binding affinity prediction**: PRODIGY / Rosetta / BSA consensus K_D
- **Codon optimization**: Vaccinia / AAV / Lentivirus expression systems
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
│   ├── target_db.py    # Immune cell target database
│   ├── sequence_qc.py  # Sequence quality control
│   ├── affinity.py     # Binding affinity prediction
│   ├── ranking.py      # Candidate scoring & ranking
│   ├── codon_opt.py    # Codon optimization engine
│   └── utils.py        # Shared utilities
├── pipeline/           # Pipeline step orchestration
│   ├── steps/          # B1-B8 step modules
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

## Requirements

- Python 3.10+
- (Optional) NVIDIA GPU with CUDA for RFdiffusion/ProteinMPNN inference

## License

MIT

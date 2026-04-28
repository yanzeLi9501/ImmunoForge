# ImmunoForge — De Novo Immune Cell Targeting Protein Design Platform

ImmunoForge is a command-line workflow for de novo protein binder design against immune-cell surface targets. The implemented user-facing interface is the CLI in `immunoforge/cli.py`, with batch execution controlled by `config/default_config.yaml` and file outputs written under `paths.output_dir`.

This README documents the CLI and the YAML-driven file workflow exactly as implemented in code. It does not describe the web UI workflow.
<img width="1379" height="864" alt="image" src="https://github.com/user-attachments/assets/b0ae4bc4-ee24-462e-a53b-4db52e7ead5e" />


## Features

- De novo binder design via RFdiffusion + ProteinMPNN
- Built-in target database for immune and tumour surface proteins
- Sequence QC, structure validation, affinity scoring, ranking, and codon optimization
- Optional AF3 ternary validation and conditional affinity maturation
- JSON, FASTA, PDB, shell-script, and CSV outputs for batch execution

## Requirements

- Python 3.10+
- Optional GPU / local toolchain support for RFdiffusion, ProteinMPNN, ESMFold, AF2-multimer, and Boltz-2 dependent steps

## Installation

```bash
pip install -e .
```

## Quick Start

1. Edit `config/default_config.yaml`.
2. Run the pipeline from the repository root.
3. Inspect the generated files under `outputs/` or the directory passed with `--output`.

```bash
# Full run with the shipped default config
immunoforge run --config config/default_config.yaml

# Restrict execution to explicit steps
immunoforge run --config config/default_config.yaml --steps B1 B2 B3

# Override species.default for this invocation
immunoforge run --config config/default_config.yaml --species human

# Redirect paths.output_dir and paths.logs_dir for this invocation
immunoforge run --config config/default_config.yaml --output outputs_human
```

When `--steps` is omitted, `immunoforge.pipeline.runner.run_pipeline` selects steps as follows:

- If `af3_validation.enabled` or `affinity_maturation.enabled` is `true`: `B1 B2 B3 B3a B3b B4 B5 B6 B7 B8 B9a B9b`
- Else if `sequence_optimization.enabled` is `true`: `B1 B2 B3 B3a B3b B4 B5 B6 B7 B8`
- Else: `B1 B2 B3 B3b B4 B5 B6 B7 B8`

The step keys registered in code are: `B1`, `B2`, `B3`, `B3a`, `B3b`, `B4`, `B4b`, `B4c`, `B5`, `B5b`, `B5c`, `B6`, `B6b`, `B7`, `B7b`, `B8`, `B9`, `B9a`, `B9b`.

## CLI Reference

### `immunoforge run`

| Argument | Type | Default | Implemented behavior |
| --- | --- | --- | --- |
| `-c`, `--config` | path | packaged `config/default_config.yaml` | Loads YAML with `immunoforge.core.utils.load_config`. |
| `-s`, `--steps` | list of step keys | `None` | Runs only the provided step keys. If omitted, runner auto-selects steps using the logic above. |
| `--species` | string | `None` | Overrides `species.default` in the loaded config for the current run. The code does not validate the value; the shipped config enumerates `mouse`, `human`, and `cynomolgus`. |
| `-o`, `--output` | path | `None` | Overrides `paths.output_dir` and also rewrites `paths.logs_dir` to `<output>/logs`. |

### `immunoforge qc`

| Argument | Type | Default | Implemented behavior |
| --- | --- | --- | --- |
| `fasta` | path | required | Reads a FASTA file with `read_fasta`. |
| `-o`, `--output` | path | `None` | If provided, writes JSON to that path. Otherwise prints a text summary to stdout. |

### `immunoforge targets`

| Argument | Type | Default | Implemented behavior |
| --- | --- | --- | --- |
| `--cell-type` | string | `None` | Case-insensitive exact match against `TargetEntry.cell_type`. Current database values include `T cell`, `DC`, `NK cell`, `B cell`, `macrophage`, and `tumour`. |
| `--species` | string | `None` | Case-insensitive exact match against `TargetEntry.species`. Current database values include `mouse`, `human`, and `cynomolgus`. |
| `--benchmark` | flag | `False` | Returns `get_benchmark_targets()`, which filters the built-in database to entries with `benchmark_value >= 3`. |

### `immunoforge codon`

| Argument | Type | Default | Implemented behavior |
| --- | --- | --- | --- |
| `sequence` | string | required | Amino-acid sequence literal, or `@file.fasta`. When `@file.fasta` is used, only the first FASTA record is consumed. |
| `--species` | string | `mouse` | Selects the codon-frequency table. Implemented tables are `mouse`, `human`, and `cynomolgus`; unknown values fall back to the mouse table. |
| `--system` | string | `vaccinia` | Implemented cassette branches are `vaccinia`, `aav`, and `lentivirus`. Any other value falls back to a generic `GCCACCATG + signal peptide + CDS + TAA` cassette. |

## Input and Output Formats

### `run` input format

`immunoforge run` consumes a YAML file. The shipped template is `config/default_config.yaml`. Its top-level sections are:

- `project`
- `species`
- `targets`
- `rfdiffusion`
- `proteinmpnn`
- `sequence_qc`
- `structure_validation`
- `hotspot_prediction`
- `sequence_optimization`
- `affinity`
- `ranking`
- `codon_optimization`
- `paths`
- `af3_validation`
- `af2_filter`
- `affinity_maturation`
- `server`
- `denovo_profile`

The `targets` section is a mapping keyed by `target1`, `target2`, and any additional target keys you choose to add. Each target entry uses this field layout:

```yaml
targets:
	target1:
		name: "mCLEC9A"
		uniprot: "Q8BRU4"
		source: "alphafold"
		pdb_id: null
		af_id: "AF-Q8BRU4-F1"
		chain: "A"
		residue_range: "116-238"
		hotspot_residues: [120, 145, 147]
		organism: "Mus musculus"
		cell_type: "cDC1"
```

`B1_target_prep.py` uses `source`, `pdb_id`, `af_id`, `chain`, `residue_range`, and `hotspot_residues` directly. `name`, `uniprot`, `organism`, and `cell_type` remain available as metadata in the config.

### `run` output layout

The pipeline writes step outputs under `paths.output_dir`.

```text
<output_dir>/
	B1_target_prep.json
	B2_rfdiffusion.json
	B3_sequence_design.json
	B3a_sequence_optimization.json
	B3b_sequence_qc.json
	B4_structure_validation.json
	B4c_af2_multimer_filter.json         # only if B4c is executed
	B5_binding_affinity.json
	B6_candidate_ranking.json
	B7_codon_optimization.json
	B9a_af3_validation.json              # only if B9a is executed
	B9b_affinity_maturation.json         # only if B9b is executed
	pipeline_run_summary.json
	pipeline_summary.json                # written by B8 when executed
	top_candidates.fasta
	rfdiffusion_backbones/
	proteinmpnn_sequences/
	sequence_optimization/
	predicted_structures/
	af3_complexes/
	synthesis_ready/
	Top_synthesis/
	logs/
```

### Step output files and top-level keys

| File | Produced by | Top-level keys |
| --- | --- | --- |
| `B1_target_prep.json` | `B1_target_prep.py` | `targets` |
| `B2_rfdiffusion.json` | `B2_rfdiffusion.py` | `rfdiffusion_available`, `n_designs_per_target`, `diffusion_steps`, `commands`, `backbones_generated` |
| `B3_sequence_design.json` | `B3_sequence_design.py` | `proteinmpnn_available`, `n_backbones`, `sequences_per_backbone`, `total_sequences`, `sequences` |
| `B3a_sequence_optimization.json` | `B3a_sequence_optimization.py` | `optimization_enabled`, `n_backbones`, `total_optimized`, `sequences`, `stage_summaries` |
| `B3b_sequence_qc.json` | `B3b_sequence_qc.py` | `total`, `n_passed`, `n_failed`, `pass_rate`, `failure_reasons`, `passed`, `failed` |
| `B4_structure_validation.json` | `B4_structure_validation.py` | `total_input`, `quality_counts`, `n_validated`, `method`, `dual_evaluation`, `validated` |
| `B4c_af2_multimer_filter.json` | `B4c_af2_multimer_filter.py` | `status`, `backend`, `n_candidates_scored`, `n_high`, `n_medium`, `n_low`, `iptm_reject_threshold`, `iptm_high_threshold`, `results` |
| `B5_binding_affinity.json` | `B5_binding_affinity.py` | `n_analyzed`, `methods`, `scored` |
| `B6_candidate_ranking.json` | `B6_candidate_ranking.py` | `total_input`, `top_n`, `ranked` |
| `B7_codon_optimization.json` | `B7_codon_optimization.py` | `species`, `expression_system`, `signal_peptide`, `n_optimized`, `candidates` |
| `B9a_af3_validation.json` | `B9a_af3_validation.py` | `status`, `method`, `n_candidates`, `quality_distribution`, `mean_iptm`, `targets`, `results` |
| `B9b_affinity_maturation.json` | `B9b_affinity_maturation.py` | `status`, `kd_threshold_nM`, `target_kd_nM`, `n_total_candidates`, `n_below_threshold`, `n_matured`, `n_improved_gt_1_5x`, `n_target_reached`, `mean_improvement_fold`, `temperatures`, `max_generations`, `maturation_results` |
| `pipeline_run_summary.json` | `pipeline.runner.run_pipeline` | `pipeline`, `species`, `steps_requested`, `completed`, `failed`, `total_time_s`, `step_results` |
| `pipeline_summary.json` | `B8_deliverables.py` | `project`, `version`, `species`, `generated_at`, `pipeline_funnel`, `top_candidates`, `codon_optimization` |

### `qc` input format

`immunoforge qc` uses `read_fasta`, which expects standard FASTA text:

```text
>sequence_id optional_description
MKTIIALSYIFCLVFADYKDDD...
>sequence_id_2
GQDPYHNNKQ...
```

Only the first whitespace-delimited token after `>` becomes the stored `id`.

### `qc` output format

With `-o`, the command writes JSON shaped like this:

```json
{
	"total": 2,
	"n_passed": 1,
	"n_failed": 1,
	"pass_rate": 0.5,
	"failure_reasons": {
		"odd_Cys": 1
	},
	"passed": [
		{
			"pass": true,
			"failures": [],
			"protease": {
				"furin": 0,
				"thrombin": 0,
				"enterokinase": 0,
				"mmp_PxxP": 0,
				"mmp_PLG": 0,
				"cathepsin_b_GFLG": 0,
				"CatB_dibasic_count": 0,
				"CatB_dibasic_density": 0.0
			},
			"toxicity": [],
			"aggregation": {
				"apr_count": 0,
				"apr_regions": [],
				"pass": true
			},
			"cysteine": {
				"n_cysteines": 0,
				"is_even": true,
				"pass": true
			},
			"isoelectric_point": {
				"pI": 7.0,
				"pass": true
			},
			"sequence_length": 80,
			"id": "seq1"
		}
	],
	"failed": []
}
```

Without `-o`, stdout is a text summary:

```text
Total: <total>  Passed: <n_passed>  Failed: <n_failed>
	FAIL <id>: <reason1>, <reason2>
```

### `targets` output format

`immunoforge targets` prints a fixed-width table with these columns:

| Column | Source |
| --- | --- |
| `Name` | `TargetEntry.name` |
| `Species` | `TargetEntry.species` |
| `Cell` | `TargetEntry.cell_type` |
| `Known Binders` | Comma-separated `name(kd_nM nM)` strings from `TargetEntry.known_binders` |
| `Benchmark` | `benchmark_value` rendered as `<value>★`, or `—` if zero |

### `codon` input and output format

The input sequence can be either:

- A raw amino-acid string on the command line
- A FASTA path prefixed with `@`, for example `@example.fasta`

When a FASTA file is used, only the first record is read.

The stdout format is:

```text
Species:     <species>
System:      <expression_system>
GC content:  <gc_percent>%
T5NT free:   Yes|No
CDS length:  <bp> bp

CDS DNA:
<dna_sequence>
```

## Full YAML Parameter Reference

The tables below list every field shipped in `config/default_config.yaml`. The last column states whether the current CLI pipeline reads the field directly.

### `project`

| Key | Default | Current code use |
| --- | --- | --- |
| `name` | `ImmunoForge` | Metadata only in the shipped config; `B8_deliverables.py` hardcodes `project: "ImmunoForge"`. |
| `version` | `0.1.0` | Read by `B8_deliverables.py` for `pipeline_summary.json`. |
| `description` | `De novo immune cell targeting protein design platform` | Metadata only in the shipped config. |

### `species`

| Key | Default | Current code use |
| --- | --- | --- |
| `default` | `mouse` | Read by the runner and downstream steps such as `B7` and `B8`; overridden by `run --species`. |
| `available` | `[mouse, human, cynomolgus]` | Declared in the shipped config; not validated by the CLI. |

### `targets.target1` / `targets.target2`

| Key | Default pattern | Current code use |
| --- | --- | --- |
| `name` | string | Used by `B1`, `B9a`, and output file naming. |
| `uniprot` | string | Metadata only in the current pipeline path. |
| `source` | `alphafold` or `pdb` | Used by `B1` to choose `download_alphafold` versus `download_pdb`. |
| `pdb_id` | string or `null` | Used by `B1` when `source != alphafold` and a PDB identifier is present. |
| `af_id` | string or `null` | Used by `B1` when `source == alphafold`. |
| `chain` | single-letter string | Used by `B1.extract_chain`. |
| `residue_range` | `"start-end"` string | Used by `B1` to build the RFdiffusion contig string. |
| `hotspot_residues` | list of integers | Used by `B1` to build `hotspot_args`; can be replaced by predicted hotspots when auto-hotspot is enabled and the list is empty. |
| `organism` | string | Metadata only in the current pipeline path. |
| `cell_type` | string | Metadata only in the current pipeline path. |

### `rfdiffusion`

| Key | Default | Current code use |
| --- | --- | --- |
| `model` | `RFdiffusion_v3` | Present in the config; `B2_rfdiffusion.py` does not read it directly. |
| `model_dir` | `/home/data/liyanze/tools/RFdiffusion/models` | Present in the config; `B2_rfdiffusion.py` does not read it directly. |
| `n_designs_per_target` | `50` | Read by `B2_rfdiffusion.py` and by `B8_deliverables.py`. |
| `binder_length` | `70-100` | Read by `B1_target_prep.py` to build RFdiffusion contigs. |
| `diffusion_steps` | `50` | Read by `B2_rfdiffusion.py`. |
| `noise_scale_ca` | `1.0` | Present in the config; `B2_rfdiffusion.py` does not read it directly. |
| `noise_scale_frame` | `1.0` | Present in the config; `B2_rfdiffusion.py` does not read it directly. |
| `hotspot_guidance` | `true` | Present in the config; `B2_rfdiffusion.py` does not read it directly. |
| `dual_target_designs` | `15` | Read by `B2_rfdiffusion.py`. |
| `dual_strategies` | `3` | Read by `B2_rfdiffusion.py`. |

### `proteinmpnn`

| Key | Default | Current code use |
| --- | --- | --- |
| `model` | `soluble` | Present in the config; `B3_sequence_design.py` hardcodes `--model_name v_48_020` and does not read this field. |
| `sampling_temperature` | `0.1` | Read by `B3_sequence_design.py`. |
| `sequences_per_backbone` | `8` | Read by `B3_sequence_design.py`. |
| `design_chain_only` | `true` | Present in the config; `B3_sequence_design.py` does not read it directly. |

### `sequence_qc`

| Key | Default | Current code use |
| --- | --- | --- |
| `protease_checks.furin.pattern` | `R.[KR]R` | Present in the config; the QC code uses hardcoded protease patterns instead of reading this field. |
| `protease_checks.furin.action` | `reject` | Present in the config; not read directly. |
| `protease_checks.thrombin.pattern` | `LVPRGS` | Present in the config; not read directly. |
| `protease_checks.thrombin.action` | `reject` | Present in the config; not read directly. |
| `protease_checks.enterokinase.pattern` | `DDDDK` | Present in the config; not read directly. |
| `protease_checks.enterokinase.action` | `reject` | Present in the config; not read directly. |
| `protease_checks.mmp_motifs.patterns` | `[PxxP, PLG]` | Present in the config; not read directly. |
| `protease_checks.mmp_motifs.max_density_pct` | `5.0` | Present in the config; not read directly. |
| `protease_checks.cathepsin_b_gflg.pattern` | `GFLG` | Present in the config; not read directly. |
| `protease_checks.cathepsin_b_gflg.action` | `reject` | Present in the config; not read directly. |
| `protease_checks.cathepsin_b_dibasic.patterns` | `[KK, RR, KR, RK]` | Present in the config; not read directly. |
| `protease_checks.cathepsin_b_dibasic.max_density_pct` | `5.0` | Present in the config; not read directly. |
| `toxicity.poly_cationic.pattern` | `[RK]{6,}` | Present in the config; the QC code uses hardcoded toxicity patterns instead. |
| `toxicity.poly_cationic.action` | `reject` | Present in the config; not read directly. |
| `toxicity.nls_signal.pattern` | `[KR]{4}.[KR]{3}` | Present in the config; not read directly. |
| `toxicity.nls_signal.action` | `reject` | Present in the config; not read directly. |
| `aggregation.apr_pattern` | `[VILMFYW]{5,}` | Present in the config; the QC code uses a hardcoded APR pattern instead. |
| `aggregation.action` | `reject` | Present in the config; not read directly. |
| `cysteine.require_even` | `true` | Present in the config; the QC code enforces even-Cys parity unconditionally. |
| `isoelectric_point.min_pi` | `4.5` | Read by `run_full_qc`. |
| `isoelectric_point.max_pi` | `10.5` | Read by `run_full_qc`. |

### `structure_validation`

| Key | Default | Current code use |
| --- | --- | --- |
| `method` | `auto` | Read by `B4_structure_validation.py`. |
| `dual_evaluation` | `true` | Read by `B4_structure_validation.py`. |
| `high_threshold.mean_plddt` | `85` | Present in the config; `B4_structure_validation.py` does not read threshold values directly. |
| `high_threshold.fraction_above_70` | `0.80` | Present in the config; not read directly by `B4_structure_validation.py`. |
| `medium_threshold.mean_plddt` | `70` | Present in the config; not read directly by `B4_structure_validation.py`. |
| `medium_threshold.fraction_above_70` | `0.60` | Present in the config; not read directly by `B4_structure_validation.py`. |
| `pass_levels` | `[HIGH, MEDIUM]` | Read by `B4_structure_validation.py`. |

### `hotspot_prediction`

| Key | Default | Current code use |
| --- | --- | --- |
| `enabled` | `true` | Read by `B1_target_prep.py`. |
| `method` | `auto` | Read by `B1_target_prep.py` and forwarded to `predict_hotspots`. |
| `top_k` | `10` | Read by `B1_target_prep.py`. |

### `sequence_optimization`

| Key | Default | Current code use |
| --- | --- | --- |
| `enabled` | `true` | Read by the runner for default step selection. |
| `temperatures` | `[1.0, 0.5, 0.2, 0.1]` | Read by `core.sequence_optimizer.run_optimization`. |
| `seqs_per_temperature` | `4` | Read by `core.sequence_optimizer.run_optimization`. |
| `helicity_weight` | `0.1` | Read by `core.sequence_optimizer.run_optimization`. |
| `stage2_top_k` | `8` | Read by `core.sequence_optimizer.run_optimization`. |
| `stage3_mutations_per_candidate` | `10` | Read by `core.sequence_optimizer.run_optimization`. |
| `stage3_interface_bias` | `0.7` | Present in the config; `run_optimization` does not pass it through, so Stage 3 currently uses the hardcoded default inside `stage3_greedy_mutation`. |
| `stage4_top_n` | `4` | Read by `core.sequence_optimizer.run_optimization`. |

### `affinity`

| Key | Default | Current code use |
| --- | --- | --- |
| `methods` | `[prodigy_ics, rosetta_ref2015, bsa_regression]` | Present in the config; `B5_binding_affinity.py` uses hardcoded methods from `core.affinity` and does not read this list directly. |
| `consensus` | `geometric_mean` | Present in the config; not read directly by `B5_binding_affinity.py`. |
| `hotspot_analysis` | `true` | Present in the config; not read directly by `B5_binding_affinity.py`. |
| `confidence_threshold_log10` | `1.0` | Present in the config; not read directly by `B5_binding_affinity.py`. |

### `ranking`

| Key | Default | Current code use |
| --- | --- | --- |
| `weights.plddt` | `0.20` | Read by `B6_candidate_ranking.py` and forwarded to `rank_candidates`. |
| `weights.ddg` | `0.20` | Read by `B6_candidate_ranking.py` and forwarded to `rank_candidates`. |
| `weights.kd` | `0.15` | Read by `B6_candidate_ranking.py` and forwarded to `rank_candidates`. |
| `weights.mpnn_score` | `0.15` | Read by `B6_candidate_ranking.py` and forwarded to `rank_candidates`. |
| `weights.bsa` | `0.10` | Read by `B6_candidate_ranking.py` and forwarded to `rank_candidates`. |
| `weights.shape_complementarity` | `0.10` | Read by `B6_candidate_ranking.py` and forwarded to `rank_candidates`. |
| `weights.baseline` | `0.10` | Read by `B6_candidate_ranking.py` and forwarded to `rank_candidates`. |
| `penalties.high_aggregation` | `-0.10` | Present in the config; penalties are hardcoded inside `core.ranking.compute_composite_score`. |
| `penalties.extreme_pi` | `-0.05` | Present in the config; not read directly by `core.ranking.compute_composite_score`. |
| `top_n` | `20` | Read by `B6_candidate_ranking.py`. |
| `synthesis_top_n` | `5` | Read by `B7_codon_optimization.py` and `B8_deliverables.py`. |

### `codon_optimization`

| Key | Default | Current code use |
| --- | --- | --- |
| `species_tables.mouse` | `kazusa_mus_musculus` | Present in the config; codon tables are hardcoded in `core.codon_opt`. |
| `species_tables.human` | `kazusa_homo_sapiens` | Present in the config; not read directly. |
| `species_tables.cynomolgus` | `kazusa_macaca_fascicularis` | Present in the config; not read directly. |
| `gc_content_target` | `[0.48, 0.52]` | Read by `B7_codon_optimization.py`, which uses the midpoint as `target_gc`. |
| `avoid_patterns.vaccinia_t5nt` | `TTTTTNT` | Present in the config; `core.codon_opt` uses a hardcoded regex for the same motif instead of reading this field. |
| `restriction_sites_avoid` | enzyme list | Present in the config; `core.codon_opt.check_restriction_sites` uses its own default enzyme list and `B7` does not pass this field in. |
| `signal_peptides.il2_leader.name` | `IL-2 leader` | Present in the config; `core.codon_opt` uses hardcoded signal-peptide definitions. |
| `signal_peptides.il2_leader.sequence` | `MYRMQLLSCIALSLALVTNS` | Present in the config; not read directly. |
| `signal_peptides.igk_leader.name` | `IgG kappa leader` | Present in the config; not read directly. |
| `signal_peptides.igk_leader.sequence` | `METDTLLLWVLLLWVPGSTGD` | Present in the config; not read directly. |
| `expression_systems.vaccinia.promoter` | `pSE/L` | Present in the config; `core.codon_opt.build_expression_cassette` uses hardcoded cassette strings instead of reading these values. |
| `expression_systems.vaccinia.kozak` | `GCCACCATG` | Present in the config; not read directly. |
| `expression_systems.vaccinia.insertion_locus` | `TK` | Present in the config; not read directly. |
| `expression_systems.vaccinia.cassette_format` | `TK-pSE/L-Kozak-SP-CDS-polyA-TK` | Present in the config; not read directly. |
| `expression_systems.aav.promoter` | `CMV` | Present in the config; not read directly. |
| `expression_systems.aav.kozak` | `GCCACCATG` | Present in the config; not read directly. |
| `expression_systems.lentivirus.promoter` | `EF1a` | Present in the config; not read directly. |
| `expression_systems.lentivirus.kozak` | `GCCACCATG` | Present in the config; not read directly. |

### `paths`

| Key | Default | Current code use |
| --- | --- | --- |
| `output_dir` | `outputs` | Read by the runner and by step modules. Overridden by `run --output`. |
| `data_dir` | `data` | Present in the config; not read directly by the steps covered in this README. |
| `structures_dir` | `data/structures` | Read by `B1_target_prep.py` and `B9a_af3_validation.py`. |
| `logs_dir` | `outputs/logs` | Read by the runner. Overridden to `<output>/logs` when `run --output` is used. |

### `af3_validation`

| Key | Default | Current code use |
| --- | --- | --- |
| `enabled` | `true` | Read by the runner for default step selection. |
| `top_n` | `5` | Read by `B9a_af3_validation.py`. |
| `recycling_steps` | `3` | Read by `B9a_af3_validation.py`. |
| `sampling_steps` | `200` | Read by `B9a_af3_validation.py`. |
| `diffusion_samples` | `1` | Read by `B9a_af3_validation.py`. |
| `timeout_per_prediction` | `600` | Read by `B9a_af3_validation.py`. |

### `af2_filter`

| Key | Default | Current code use |
| --- | --- | --- |
| `enabled` | `true` | Present in the config; the default runner path does not auto-insert `B4c`, so this field only matters if your workflow chooses `B4c` explicitly. |
| `max_candidates` | `10` | Read by `B4c_af2_multimer_filter.py`. |
| `iptm_reject_threshold` | `0.4` | Read by `B4c_af2_multimer_filter.py`. |
| `iptm_high_threshold` | `0.7` | Read by `B4c_af2_multimer_filter.py`. |

### `affinity_maturation`

| Key | Default | Current code use |
| --- | --- | --- |
| `enabled` | `true` | Read by the runner for default step selection. |
| `kd_threshold_nM` | `100.0` | Read by `B9b_affinity_maturation.py`. |
| `target_kd_nM` | `50.0` | Read by `B9b_affinity_maturation.py`. |
| `max_generations` | `8` | Read by `B9b_affinity_maturation.py`. |
| `candidates_per_gen` | `30` | Read by `B9b_affinity_maturation.py`. |
| `top_k_per_gen` | `5` | Read by `B9b_affinity_maturation.py`. |
| `temperatures` | `[0.1, 0.3]` | Read by `B9b_affinity_maturation.py`. |
| `seed` | `42` | Read by `B9b_affinity_maturation.py`. |

### `server`

The shipped default config also includes a `server` block with `host`, `port`, `workers`, and `reload`. This README intentionally omits that workflow and focuses on the CLI/file interface above.

### `denovo_profile`

The shipped default config includes a `denovo_profile` block containing nested `rfdiffusion`, `proteinmpnn`, `sequence_optimization`, `affinity_maturation`, and `ranking` overrides. The current CLI runner and step modules covered above do not read `denovo_profile` directly.

## Project Structure

```text
immunoforge/
	core/                  Core computation modules
	pipeline/              Runner and step implementations
	cli.py                 CLI entry point
config/
	default_config.yaml    Shipped YAML template
tests/
```

## License

MIT

# Project Structure

This file shows where each important file lives and what it is used for.

```text
ceng534/
|-- classifier.py
|-- paraphrase_detection.py
|-- sonnet_generation.py
|-- _PROJECT_STRUCTURE.md
|-- _RUN_COMMANDS.md
|-- data/
|-- predictions/
|-- results/
|-- scripts/
|-- src/
`-- others/
```

## Root Files

```text
ceng534/
|-- classifier.py
|-- paraphrase_detection.py
|-- sonnet_generation.py
|-- _PROJECT_STRUCTURE.md
`-- _RUN_COMMANDS.md
```

- `classifier.py`: Main script for sentiment classification on SST and CFIMDB.
- `paraphrase_detection.py`: Main script for Quora paraphrase detection.
- `sonnet_generation.py`: Main script for GPT-2 sonnet generation.
- `_PROJECT_STRUCTURE.md`: This file; explains the project layout.
- `_RUN_COMMANDS.md`: Copy-ready commands for full, last-layer, LoRA, GPU, and CPU runs.

## Source Code

```text
src/
|-- __init__.py
|-- config.py
|-- datasets.py
|-- distribution_metrics.py
|-- evaluation.py
|-- log_experiments.py
|-- lora_attention.py
|-- optimizer.py
|-- utils.py
|-- models/
|   |-- __init__.py
|   |-- base_gpt.py
|   `-- gpt2.py
`-- modules/
    |-- __init__.py
    |-- attention.py
    `-- gpt2_layer.py
```

- `src/config.py`: GPT-2 configuration classes.
- `src/datasets.py`: Dataset loaders and batching logic.
- `src/distribution_metrics.py`: Estimates weight distribution names such as `normal-like`, `uniform-like`, or `zero-like`.
- `src/evaluation.py`: Evaluation functions for paraphrase detection and sonnet generation.
- `src/log_experiments.py`: Writes experiment metrics to CSV files in `results/`.
- `src/lora_attention.py`: Prints aggregated Attention, LoRA-A, and LoRA-B weight statistics.
- `src/optimizer.py`: Custom AdamW optimizer.
- `src/utils.py`: Shared helper functions.
- `src/models/base_gpt.py`: Base pretrained model helper class.
- `src/models/gpt2.py`: Main GPT-2 model implementation.
- `src/modules/attention.py`: Causal self-attention and LoRA adapter logic.
- `src/modules/gpt2_layer.py`: One GPT-2 transformer block.

## Data

```text
data/
|-- ids-cfimdb-dev.csv
|-- ids-cfimdb-test-student.csv
|-- ids-cfimdb-train.csv
|-- ids-sst-dev.csv
|-- ids-sst-test-student.csv
|-- ids-sst-train.csv
|-- quora-dev.csv
|-- quora-test-student.csv
|-- quora-train.csv
|-- sonnets.txt
|-- sonnets_held_out.txt
|-- sonnets_held_out_dev.txt
`-- TRUE_sonnets_held_out_dev.txt
```

- `ids-sst-*`: SST sentiment classification data.
- `ids-cfimdb-*`: CFIMDB sentiment classification data.
- `quora-*`: Quora paraphrase detection data.
- `sonnets*`: Sonnet training and held-out generation data.
- `TRUE_sonnets_held_out_dev.txt`: Gold sonnets for chrF/BLEU evaluation.

## Outputs

```text
predictions/
|-- generated_sonnets.txt
|-- para-dev-output.csv
|-- para-test-output.csv
|-- last-linear-layer-*-out.csv
`-- full-model-*-out.csv

results/
|-- sonnet_generation.csv
|-- generate_figures.py
|-- figures/
|   `-- generated .png figures
|-- summaries/
|   `-- generated summary .csv files
|-- sonnet_generation_checkpoints/
|   `-- ...
`-- other result text/csv files
```

- `predictions/`: Submission-style output files.
- `results/`: Logs, metric files, and saved checkpoints.
- `results/generate_figures.py`: Reads result CSV files and generates task-specific figures.
- `results/figures/`: Generated `.png` figures.
- `results/summaries/`: Generated summary CSV files by task.
- `results/sonnet_generation_checkpoints/`: Sonnet `.pt` checkpoints grouped by run settings.

## Helper Scripts

```text
scripts/
|-- optimizer_test.py
|-- optimizer_test.npy
|-- prepare_submit.py
|-- sanity_check.py
`-- test_lora_init.py
```

- `scripts/optimizer_test.py`: Tests `src/optimizer.py`.
- `scripts/optimizer_test.npy`: Reference tensor for optimizer testing.
- `scripts/prepare_submit.py`: Creates the submission zip.
- `scripts/sanity_check.py`: Checks local GPT-2 output against Hugging Face GPT-2.
- `scripts/test_lora_init.py`: Demonstrates LoRA initialization methods.

## Other Files

```text
others/
|-- env.yml
|-- LICENSE
|-- README.md
`-- setup.sh
```

- `others/env.yml`: Conda environment file.
- `others/LICENSE`: License file.
- `others/README.md`: Original project README.
- `others/setup.sh`: Setup helper.

## Recommended Start

1. Open `_RUN_COMMANDS.md`.
2. Copy the command for the task you want.
3. Run it from the project root.
4. Check outputs in `predictions/` and `results/`.

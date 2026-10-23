# Green Code or Costly Code? Evaluating the Energy Costs of Iterative LLM-Based Code Optimizations

The following is the pipeline used in our paper.

## Repository layout

```
EffiLearner/
  src/                        Python source
    vllm_generation.py        generate initial completions via vLLM
    vllm_effilearner.py       self-optimization loop
    validate_completions.py   correctness validation
    profile_canonical.py      profile canonical (reference) solutions
    vllm_report.py            compute and write the CSV report
    code_efficiency_calculator.py  core profiling engine (mprof + kernprof)
    run_effibench_energy.py   measure code-execution energy via metrion (optional)
    ...                       analysis and plotting utilities
  scripts/
    run.sh                    Command 1 — generate + optimize
    profile_and_report.sh     Command 2 — validate + profile + report
    run_all_experiments.sh    full parameter sweep (paper replication)
    run_experiment.sh         single model × tag experiment
    experiment_lib.sh         sweep configuration (model registry, run tags)
    _lib.sh                   shared path helpers sourced by run.sh / profile_and_report.sh
    run_code.sh               low-level subprocess runner used by the profiler
    smoke_test.sh             quick end-to-end pipeline check (10 problems, 5 epochs)
  datasets/
    dataset.json              EffiBench benchmark (included)
```

## Requirements

- Python 3.12
- [vLLM](https://github.com/vllm-project/vllm) installed and serving your model
- Linux, `screen`, `curl` (used by the run scripts — install `screen` via `sudo apt-get install -y screen` if not present)
- A GPU with enough VRAM for the model you want to evaluate
- *(optional)* [Metrion](https://github.com/ds-lab/Metrion) for energy measurement,
  installed as a sibling directory (`../metrion/`) with passwordless `sudo` configured
  for the metrion binary

## Installation

Download and extract the archive from Zenodo, then:

```bash
cd EffiLearner
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

vLLM is not listed in `requirements.txt` because it requires hardware-specific
installation. Install it separately following the
[vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation.html):

```bash
pip install vllm
```

## Dataset

The EffiBench benchmark is already included at `datasets/dataset.json`.
HumanEval and MBPP are downloaded automatically from Hugging Face the first time
they are requested.

## Quickstart

### Step 0 — start your vLLM server

In a separate terminal, start vLLM with the model you want to evaluate:

```bash
vllm serve Qwen/Qwen3.5-4B --port 8000 \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

Keep this terminal open for the duration of the run.

`--default-chat-template-kwargs '{"enable_thinking":false}'` is needed for
Qwen3-family models, whose chat template emits `<think>...</think>` reasoning
by default — the scripts here parse completions as plain code and don't strip
thinking tags. Drop this flag for models that don't support it.

### Step 1 — generate completions and self-optimize

```bash
bash scripts/run.sh \
  --model-name  qwen3.5-4b \
  --checkpoint  Qwen/Qwen3.5-4B
```

Results are written to `results/qwen3.5-4b/`.

Common options:

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `EffiBench` | `EffiBench`, `HumanEval`, or `MBPP` |
| `--epochs` | `5` | number of self-optimization iterations |
| `--concurrency` | `32` | parallel requests to vLLM |
| `--temperature` | `0.0` | sampling temperature |
| `--port` | `8000` | vLLM server port |

Run `bash scripts/run.sh --help` for the full list.

### Step 2 — validate, profile, and report

```bash
bash scripts/profile_and_report.sh \
  --model-name qwen3.5-4b
```

Writes validated results to `results/qwen3.5-4b-validated/` and a CSV report
to `reports/qwen3.5-4b/`.

Common options:

| Flag | Default | Description |
|---|---|---|
| `--n-runs` | `5` | profiling sweeps averaged per entry |
| `--workers` | `32` | parallel profiling workers |

Run `bash scripts/profile_and_report.sh --help` for the full list.

### Smoke test (sanity-check your setup)

`scripts/smoke_test.sh` runs the whole pipeline — generation, self-optimization,
validation, canonical profiling, report, and (if metrion is set up) energy
measurement — on a 10-problem slice in one command. Use it to confirm a fresh
setup works end-to-end before kicking off a full run.

```bash
# Step 0 (same as above): start vLLM serving the checkpoint you want to test
vllm serve Qwen/Qwen3.5-4B --port 8000 \
  --default-chat-template-kwargs '{"enable_thinking":false}'

# Step 1: run the smoke test, pointing it at the same checkpoint
cd scripts
CHECKPOINT=Qwen/Qwen3.5-4B ./smoke_test.sh
```

Outputs land under `results/smoke-test/`, `results/smoke-test-validated/`,
and `reports/smoke-test/`. `CHECKPOINT` must match whatever identifier you
passed to `vllm serve`. If metrion isn't set up yet, the energy step is
skipped with a warning instead of failing the run.

Other variables you can override via the environment:

| Variable | Default | Description |
|---|---|---|
| `CHECKPOINT` | `$MODEL_ROOT/Qwen/Qwen3.5-4B` | model identifier passed to vLLM (must match `vllm serve`) |
| `MODEL_ROOT` | `../models` (sibling of `EffiLearner/`) | base dir for the default `CHECKPOINT` |
| `METRION_ROOT` | `../metrion` (sibling of `EffiLearner/`), falling back to `../metrion-internal` | metrion checkout used for the energy step |

## With Metrion (energy measurement)

[Metrion](https://github.com/ds-lab/metrion-internal) measures hardware energy consumption.
Clone it as a sibling of EffiLearner:

```bash
# from the parent directory that contains EffiLearner/
git clone https://github.com/ds-lab/metrion-internal.git metrion
cd metrion
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Metrion uses eBPF via the `bcc` library, which is a system package not installable
via pip. Install it before activating the venv:

```bash
sudo apt-get install -y python3-bcc
```

Then symlink it into the EffiLearner venv (adjust the site-packages path if your
Python version differs):

```bash
ln -sf /usr/lib/python3/dist-packages/bcc \
  EffiLearner/.venv/lib/python3.12/site-packages/bcc
```

Configure passwordless `sudo` for metrion (see metrion's README for instructions).

Then add `--with-metrion` to either command:

```bash
# Measure inference energy during generation + optimization:
bash scripts/run.sh \
  --model-name  qwen3.5-4b \
  --checkpoint  Qwen/Qwen3.5-4B \
  --with-metrion

# Measure code-execution energy during profiling:
bash scripts/profile_and_report.sh \
  --model-name qwen3.5-4b \
  --with-metrion
```

If metrion lives somewhere other than `../metrion/`, pass `--metrion-root PATH`
or set the `METRION_ROOT` environment variable.

## Replicating the full paper sweep

The paper evaluates multiple models across four parameter configurations
(default, temperature 0.5, 65 k context, 1 k max tokens).

1. Edit `scripts/experiment_lib.sh` to set the correct local paths for each
   model checkpoint under `checkpoint_of()`.

2. Start the sweep inside a persistent screen session so it survives disconnects:

```bash
screen -dmS orchestrator bash -lc \
  'bash /path/to/EffiLearner/scripts/run_all_experiments.sh'
```

   With metrion energy measurement:

```bash
screen -dmS orchestrator bash -lc \
  'bash /path/to/EffiLearner/scripts/run_all_experiments.sh --with-metrion'
```

3. After all experiments finish, validate and report everything in batch:

```bash
bash scripts/validate_all.sh
bash scripts/report_all.sh
```

Sweep logs are written to `logs/sweep.log`. Per-experiment logs are at
`logs/<model><suffix>.{vllm,infer}.log`.


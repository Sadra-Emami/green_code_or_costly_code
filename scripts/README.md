# EffiBench sweep orchestration

Runs 11 models × 3 parameter configs (`t0.5`, `ctx65536`, `mt1024`) end-to-end
using four `screen` sessions on the host. The host stays up; the laptop can
sleep freely.

## Files

- `experiment_lib.sh` — shared paths, model registry, screen helpers, metrion
  config generator. Sourced by the others.
- `run_experiment.sh MODEL_KEY RUN_TAG` — runs one experiment.
- `run_all_experiments.sh` — loops all 11 × 3 serially (top-level script).

## How to kick off the whole sweep

```bash
screen -dmS orchestrator bash -lc \
  '/path/to/EffiLearner/scripts/run_all_experiments.sh'
```

Attach with `screen -r orchestrator`, detach with `Ctrl+A d`. Laptop sleep
does not affect the orchestrator because it lives on the host.

## What happens per experiment

For each `(model_key, run_tag)`:

1. Start `vllm_<tag>` screen → `python3 -m vllm.entrypoints.openai.api_server …`
2. Poll `http://localhost:8000/v1/models` until ready (max 30 min).
3. Start `metrion_<tag>` screen → `sudo metrion monitor -c <tmp.yaml>`,
   writing to `metrion/dbs/<model><suffix>.db` (no timestamp — the
   temp config sets `debug: false`).
4. Start `infer_<tag>` screen → `vllm_generation.py` → `vllm_effilearner.py`
   (5 epochs) → `profile_canonical.py` → `vllm_report.py`. Writes
   `scripts/.flags/<name>.done` on exit with the return code.
5. Orchestrator waits on the done flag, sends Ctrl+C to metrion (so DuckDB
   flushes cleanly), then quits the vllm and infer screens.

## Output layout

```
EffiLearner/
  results/<model><suffix>/     # generation + effilearner + report outputs
  logs/<model><suffix>.vllm.log
  logs/<model><suffix>.metrion.log
  logs/<model><suffix>.infer.log
  logs/sweep.log
metrion/
  dbs/<model><suffix>.db       # one energy db per experiment
```

Suffixes: `_t0.5`, `_ctx65536`, `_mt1024`.

## Resume after a crash

`run_experiment.sh` skips any experiment whose final epoch output
(`results/<name>/EffiBench_vllm_<name>_5.json`) already exists. Rerun
`run_all_experiments.sh` and it picks up where it stopped.

## Subset runs

Filter via env vars (space-separated):

```bash
SWEEP_MODELS="qwen3.5-4b qwen3.5-9b" SWEEP_TAGS="t0.5" \
  ./run_all_experiments.sh
```

Or run a single experiment directly:

```bash
./run_experiment.sh qwen3.5-27b ctx65536
```

## Prerequisites

- Passwordless sudo for `metrion` (the metrion screen runs `sudo -n`). Test
  with `sudo -n $(which metrion) --help`; if it prompts for a password, add
  a sudoers rule.
- `screen`, `curl`, `awk`, `tee` installed on the host.
- vLLM, the EffiLearner venv at `EffiLearner/.venv`, and
  metrion installed so `$(which metrion)` resolves under root.

## Troubleshooting

- Metrion screen dies immediately → check `logs/<name>.metrion.log` and
  `sudo -n` behavior.
- vLLM never becomes ready → check `logs/<name>.vllm.log` (OOM, bad model
  path, port in use).
- Inference hangs → attach the infer screen: `screen -r infer_<tag>`.
- DB missing after a run → metrion may have been killed before flushing;
  this script sends Ctrl+C and waits up to 2 min before forcing quit.

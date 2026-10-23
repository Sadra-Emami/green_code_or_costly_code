import asyncio
import json
import copy
import argparse
import os
import time
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from openai import AsyncOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from code_efficiency_calculator import calculate_code_execution_efficiency

PROFILE_COOLDOWN_S = 15


def param_suffix(max_tokens: int, temperature: float, context_size: int | None) -> str:
    """Build a '_t0.2_mt4096_ctx16384' style suffix for non-default params."""
    parts = []
    if temperature != 0.0:
        parts.append(f"t{temperature:g}")
    if max_tokens != 2048:
        parts.append(f"mt{max_tokens}")
    if context_size is not None:
        parts.append(f"ctx{context_size}")
    return ("_" + "_".join(parts)) if parts else ""


def build_prompt(data_entry: dict) -> str:
    overhead = data_entry.get("overhead", "The code execution failed.")
    completion = data_entry["completion"]

    if data_entry["dataset"] == "EffiBench":
        test_case = data_entry["small_test_cases"]
        task_description = data_entry["markdown_description"]
    elif data_entry["dataset"] == "HumanEval":
        test_case = data_entry["open_test_cases"]
        task_description = data_entry["prompt"]
    elif data_entry["dataset"] == "MBPP":
        test_case = "\n".join(data_entry["test_list"])
        task_description = data_entry["prompt"]
    else:
        return ""

    return f"""Optimize the efficiency of the following Python code based on the task, test case, and overhead analysis provided. Ensure the optimized code can pass the given test case.

Task Description:
{task_description}

Test Case:
{test_case}

Original Code:
```python
{completion}
```

Overhead Analysis:
{overhead}

Optimization Rules:
- Encapsulate the optimized code within a Python code block (i.e., ```python\\n[Your Code Here]\\n```).
- Do not include the test case within the code block.
- Focus solely on code optimization; test cases are already provided.
- Ensure the provided test case passes with your optimized solution.
"""


async def fetch_completion(
    data_entry: dict,
    model: str,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    max_tokens: int,
    temperature: float,
) -> dict:
    prompt = build_prompt(data_entry)
    if not prompt:
        data_entry["tmp_completion"] = ""
        return data_entry

    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a code developer expert."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            data_entry["tmp_completion"] = resp.choices[0].message.content or ""
        except Exception as e:
            print(repr(e))
            data_entry["tmp_completion"] = ""
    return data_entry


async def run_epoch(
    dataset: list,
    model: str,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    max_tokens: int,
    temperature: float,
) -> list:
    tasks = [
        fetch_completion(copy.deepcopy(entry), model, client, semaphore,
                         max_tokens, temperature)
        for entry in dataset
    ]
    return list(await async_tqdm.gather(*tasks))


def _profile_one_entry(entry, tmp_path, stage):
    """Profile one entry's current `completion` once. Returns dict with metrics."""
    overhead, mu, et, mp, executable = calculate_code_execution_efficiency(
        entry, evaluation_code=True, stage=stage, path=tmp_path, n_runs=1
    )
    return {
        "executable":      executable,
        "overhead":        overhead,
        "memory_usage":    mu,
        "execution_time":  et,
        "max_memory_peak": mp,
    }


def profile_pass(items, tmp_path, stage, workers):
    n = len(items)
    out = [None] * n
    valid = [(i, e) for i, e in items if e.get("completion")]

    def _wrap(item):
        i, entry = item
        return i, _profile_one_entry(entry, tmp_path, stage)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {executor.submit(_wrap, item): item[0] for item in valid}
        for fut in tqdm(as_completed(future_to_idx), total=len(valid)):
            i, res = fut.result()
            out[i] = res
    for i in range(n):
        if out[i] is None:
            out[i] = {"executable": False, "overhead": "", "memory_usage": 0,
                      "execution_time": 0, "max_memory_peak": 0}
    return out


def average_passes(passes):
    n = len(passes[0])
    merged = []
    for i in range(n):
        per_pass = [p[i] for p in passes]
        ok = [r for r in per_pass if r["executable"]]
        if ok:
            merged.append({
                "executable":      True,
                "overhead":        next((r["overhead"] for r in ok if r["overhead"]), ""),
                "memory_usage":    sum(r["memory_usage"]    for r in ok) / len(ok),
                "execution_time":  sum(r["execution_time"]  for r in ok) / len(ok),
                "max_memory_peak": sum(r["max_memory_peak"] for r in ok) / len(ok),
                "n_runs_used":     len(ok),
            })
        else:
            merged.append({
                "executable":      False,
                "overhead":        "",
                "memory_usage":    0,
                "execution_time":  0,
                "max_memory_peak": 0,
                "n_runs_used":     0,
            })
    return merged


def profile_n_times(items, tmp_path, stage, workers, n_runs):
    """Run N independent sweeps over items (fresh ThreadPoolExecutor each time) and
    average the metrics per entry."""
    import subprocess as _sp
    passes = []
    for r in range(1, n_runs + 1):
        print(f"  pass {r}/{n_runs}")
        passes.append(profile_pass(items, tmp_path, stage, workers))
        # Kill any orphaned mprof processes left by timed-out runs before next pass
        _sp.run(['pkill', '-9', '-f', 'mprof run'], capture_output=True)
    return average_passes(passes)


async def main_async(args):
    client = AsyncOpenAI(
        base_url=f"http://localhost:{args.port}/v1",
        api_key="token-abc123",
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    suffix = param_suffix(args.max_tokens, args.temperature, args.context_size)
    model_name = args.model_name + suffix
    results_dir = f"../results/{model_name}"
    os.makedirs(results_dir, exist_ok=True)
    input_file = (
        args.input_file
        if args.input_file
        else os.path.join(results_dir, f"{args.dataset}_vllm_{model_name}.json")
    )
    with open(input_file, "r") as f:
        dataset = json.load(f)

    for entry in dataset:
        entry["dataset"] = args.dataset

    tmp_path = f"tmp/{model_name}"
    os.makedirs(tmp_path, exist_ok=True)

    overhead_dict = {
        "overhead": [],
        "memory_usage": [],
        "execution_time": [],
        "max_memory_peak": [],
        "correct": [],
    }

    print(f"\n=== Baseline profiling ({args.profile_runs} runs averaged) ===")
    items = [(i, dataset[i]) for i in range(len(dataset))]
    merged = profile_n_times(items, tmp_path, "base", args.workers, args.profile_runs)
    for i, m in enumerate(merged):
        if m["executable"]:
            dataset[i].update({
                "overhead":        m["overhead"],
                "memory_usage":    m["memory_usage"],
                "execution_time":  m["execution_time"],
                "max_memory_peak": m["max_memory_peak"],
                "executable":      True,
            })

    executable = [e for e in dataset if e.get("executable")]
    if executable:
        avg_mu = sum(e["memory_usage"] for e in executable) / len(executable)
        avg_et = sum(e["execution_time"] for e in executable) / len(executable)
        avg_mp = sum(e["max_memory_peak"] for e in executable) / len(executable)
        print(
            f"Baseline — {len(executable)} executable  "
            f"ET={avg_et:.3f}s  MU={avg_mu:.3f}MB*s  MaxMem={avg_mp:.1f}MB"
        )
        overhead_dict["overhead"].append(
            f"ET={avg_et:.3f}s MU={avg_mu:.3f}MB*s MaxMem={avg_mp:.1f}MB"
        )
        overhead_dict["memory_usage"].append(round(avg_mu, 3))
        overhead_dict["execution_time"].append(round(avg_et, 3))
        overhead_dict["max_memory_peak"].append(round(avg_mp, 3))
        overhead_dict["correct"].append(len(executable))
    else:
        print("Baseline — no executable entries found.")

    dataset = [e for e in dataset if e.get("executable")]

    print(f"Cooling down {PROFILE_COOLDOWN_S}s before epoch 1 profiling ...")
    time.sleep(PROFILE_COOLDOWN_S)

    for epoch in range(1, args.epoch + 1):
        print(
            f"\n=== Epoch {epoch}/{args.epoch} — generating optimized code "
            f"(concurrency={args.concurrency}) ==="
        )
        results = await run_epoch(dataset, args.checkpoint, client, semaphore,
                                  args.max_tokens, args.temperature)

        print(f"=== Epoch {epoch} — profiling optimized code ({args.profile_runs} runs averaged) ===")
        tmp_completions = [
            (results[i].get("tmp_completion", "") if results[i] else "")
            for i in range(len(dataset))
        ]

        candidate_items = []
        for i, entry in enumerate(dataset):
            tmp = tmp_completions[i]
            if not tmp:
                candidate_items.append((i, {"completion": ""}))  # empty triggers skip
                continue
            cand = copy.deepcopy(entry)
            cand["completion"] = tmp
            candidate_items.append((i, cand))

        merged = profile_n_times(candidate_items, tmp_path, f"epoch_{epoch}",
                                 args.workers, args.profile_runs)

        correct = 0
        total_mu = total_et = total_mp = 0
        for i, entry in enumerate(dataset):
            m = merged[i]
            tmp = tmp_completions[i]
            if m["executable"]:
                if (
                    "memory_usage" not in entry
                    or m["memory_usage"] < entry["memory_usage"]
                ):
                    entry.update({
                        "overhead":        m["overhead"],
                        "memory_usage":    m["memory_usage"],
                        "execution_time":  m["execution_time"],
                        "max_memory_peak": m["max_memory_peak"],
                        "executable":      True,
                        "completion":      tmp,
                    })
            if entry.get("executable"):
                total_mu += entry["memory_usage"]
                total_et += entry["execution_time"]
                total_mp += entry["max_memory_peak"]
                correct += 1

        if correct > 0:
            print(
                f"Epoch {epoch} — {correct} executable  "
                f"ET={total_et/correct:.3f}s  MU={total_mu/correct:.3f}MB*s  "
                f"MaxMem={total_mp/correct:.1f}MB"
            )
            overhead_dict["overhead"].append(
                f"ET={total_et/correct:.3f}s MU={total_mu/correct:.3f}MB*s MaxMem={total_mp/correct:.1f}MB"
            )
            overhead_dict["memory_usage"].append(round(total_mu / correct, 3))
            overhead_dict["execution_time"].append(round(total_et / correct, 3))
            overhead_dict["max_memory_peak"].append(round(total_mp / correct, 3))
            overhead_dict["correct"].append(correct)

        epoch_file = os.path.join(results_dir, f"{args.dataset}_vllm_{model_name}_{epoch}.json")
        with open(epoch_file, "w") as f:
            json.dump(dataset, f, indent=4)
        print(f"Epoch {epoch} results saved to {epoch_file}")

        if epoch < args.epoch:
            print(f"Cooling down {PROFILE_COOLDOWN_S}s before epoch {epoch + 1} profiling ...")
            time.sleep(PROFILE_COOLDOWN_S)

    overhead_file = os.path.join(results_dir, f"overhead_vllm_{model_name}.json")
    with open(overhead_file, "w") as f:
        json.dump(overhead_dict, f, indent=4)

    print(f"\n{'='*50}")
    print("Efficiency progression:")
    for i, (mu, et, mp, n) in enumerate(
        zip(
            overhead_dict["memory_usage"],
            overhead_dict["execution_time"],
            overhead_dict["max_memory_peak"],
            overhead_dict["correct"],
        )
    ):
        label = "Baseline" if i == 0 else f"Epoch {i:2d}"
        print(f"  {label}: ET={et:.3f}s  MU={mu:.3f}MB*s  MaxMem={mp:.1f}MB  (n={n})")
    print(f"\nOverhead track saved to {overhead_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Model name as served by vLLM (e.g. Qwen/Qwen2.5-Coder-32B-Instruct)",
    )
    parser.add_argument(
        "--model-name", type=str, required=True,
        help="Name used for result files (e.g. qwen2.5-coder-32b)",
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["EffiBench", "HumanEval", "MBPP"])
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--concurrency", type=int, default=32,
        help="Max number of in-flight requests to vLLM at once",
    )
    parser.add_argument(
        "--input-file", type=str, default=None,
        help="Path to input JSON (overrides default ../results/{dataset}_{model_name}.json)",
    )
    parser.add_argument(
        "--workers", type=int, default=64,
        help="Number of parallel profiling workers",
    )
    parser.add_argument(
        "--profile-runs", type=int, default=5,
        help="How many independent profiling sweeps to average per phase (default: 5)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048,
        help="Max output tokens per completion (default: 2048)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (default: 0.0 = greedy)",
    )
    parser.add_argument(
        "--context-size", type=int, default=None,
        help="Label only: the --max-model-len set on the vLLM server. "
        "Included in the output folder/filename suffix for tracking.",
    )
    args = parser.parse_args()

    asyncio.run(main_async(args))

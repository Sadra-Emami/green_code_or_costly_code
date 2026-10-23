import json
import copy
import argparse
import os
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from code_efficiency_calculator import calculate_code_execution_efficiency


def profile_pass(problems, dataset_name, tmp_path, workers):
    def _profile_one(problem):
        entry = copy.deepcopy(problem)
        entry["dataset"] = dataset_name
        entry["completion"] = entry["canonical_solution"]
        _overhead, mu, et, mp, executable = calculate_code_execution_efficiency(
            entry, evaluation_code=True, stage="canonical", path=tmp_path, n_runs=1
        )
        return {
            "problem_idx":     problem["problem_idx"],
            "task_name":       problem["task_name"],
            "executable":      executable,
            "execution_time":  et,
            "memory_usage":    mu,
            "max_memory_peak": mp,
        }

    results = [None] * len(problems)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {executor.submit(_profile_one, p): i for i, p in enumerate(problems)}
        for fut in tqdm(as_completed(future_to_idx), total=len(problems)):
            i = future_to_idx[fut]
            results[i] = fut.result()
    return results


def average_runs(per_run_results):
    n_problems = len(per_run_results[0])
    merged = []
    for i in range(n_problems):
        per_run = [r[i] for r in per_run_results]
        ok = [r for r in per_run if r["executable"]]
        if ok:
            merged.append({
                "problem_idx":     per_run[0]["problem_idx"],
                "task_name":       per_run[0]["task_name"],
                "executable":      True,
                "execution_time":  sum(r["execution_time"]  for r in ok) / len(ok),
                "memory_usage":    sum(r["memory_usage"]    for r in ok) / len(ok),
                "max_memory_peak": sum(r["max_memory_peak"] for r in ok) / len(ok),
                "n_runs_used":     len(ok),
            })
        else:
            merged.append({
                "problem_idx":     per_run[0]["problem_idx"],
                "task_name":       per_run[0]["task_name"],
                "executable":      False,
                "execution_time":  0,
                "memory_usage":    0,
                "max_memory_peak": 0,
                "n_runs_used":     0,
            })
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True,
                        help="Match the model-name used in vllm_generation (to pick the right problem IDs)")
    parser.add_argument("--dataset", type=str, default="EffiBench",
                        choices=["EffiBench", "HumanEval", "MBPP"])
    parser.add_argument("--dataset-file", type=str, default="../datasets/dataset.json",
                        help="Path to the full dataset.json with canonical_solution fields")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory for result files (default: ../results/{model_name})")
    parser.add_argument("--workers", type=int, default=64,
                        help="Number of parallel profiling workers (default: 64)")
    parser.add_argument("--n-runs", type=int, default=5,
                        help="How many independent full sweeps to run and average (default: 5)")
    parser.add_argument("--keep-runs", action="store_true",
                        help="Save each individual run's JSON alongside the averaged result")
    args = parser.parse_args()

    results_dir = args.results_dir or f"../results/{args.model_name}"
    os.makedirs(results_dir, exist_ok=True)

    gen_path = os.path.join(results_dir, f"{args.dataset}_vllm_{args.model_name}.json")
    if os.path.exists(gen_path):
        with open(gen_path) as f:
            gen_data = json.load(f)
        target_ids = {e["problem_idx"] for e in gen_data}
        print(f"Profiling canonical solutions for {len(target_ids)} problems from {gen_path} ...")
    else:
        target_ids = None
        print("No generation file found — profiling all problems in dataset ...")

    with open(args.dataset_file) as f:
        full_dataset = json.load(f)
    if target_ids is not None:
        problems = [p for p in full_dataset if p["problem_idx"] in target_ids]
        print(f"Found {len(problems)} matching problems in {args.dataset_file}")
    else:
        problems = full_dataset
        print(f"Using all {len(problems)} problems in {args.dataset_file}")

    tmp_path = f"tmp/{args.model_name}"
    os.makedirs(tmp_path, exist_ok=True)

    per_run = []
    for r in range(1, args.n_runs + 1):
        print(f"\n=== Pass {r}/{args.n_runs} ===")
        results = profile_pass(problems, args.dataset, tmp_path, args.workers)
        per_run.append(results)
        # Kill any orphaned mprof processes left by timed-out runs before next pass
        subprocess.run(['pkill', '-9', '-f', 'mprof run'], capture_output=True)
        if args.keep_runs:
            run_path = os.path.join(results_dir,
                                    f"canonical_{args.dataset}_vllm_{args.model_name}_run{r}.json")
            with open(run_path, "w") as f:
                json.dump(results, f, indent=4)
            print(f"  saved pass {r} to {run_path}")
        ok = sum(1 for x in results if x["executable"])
        print(f"  pass {r}: {ok}/{len(results)} executable")

    print(f"\n=== Averaging {args.n_runs} runs ===")
    merged = average_runs(per_run)
    executable = [r for r in merged if r["executable"]]
    print(f"Executable (≥1 successful run): {len(executable)}/{len(merged)}")
    if executable:
        avg_et  = sum(r["execution_time"]  for r in executable) / len(executable)
        avg_mmu = sum(r["max_memory_peak"] for r in executable) / len(executable)
        avg_mu  = sum(r["memory_usage"]    for r in executable) / len(executable)
        print(f"Canonical (averaged) — ET={avg_et:.3f}s  MU={avg_mmu:.1f}MB  TMU={avg_mu:.3f}MB*s")

    out_path = os.path.join(results_dir, f"canonical_{args.dataset}_vllm_{args.model_name}.json")
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=4)
    print(f"Saved averaged result to {out_path}")


if __name__ == "__main__":
    main()

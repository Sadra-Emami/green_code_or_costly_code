import json
import csv
import os
import argparse
from datetime import datetime


def load_json(path: str) -> list | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def compute_stats(entries: list, total_tasks: int) -> dict:
    executable = [e for e in entries if e.get("executable")]
    n = len(executable)
    if n == 0:
        return None

    avg_et  = sum(e["execution_time"]  for e in executable) / n
    avg_mmu = sum(e["max_memory_peak"] for e in executable) / n
    avg_mu  = sum(e["memory_usage"]    for e in executable) / n
    pass1   = n / total_tasks * 100

    return {
        "n": n,
        "ET":       round(avg_et,  3),
        "MU":       round(avg_mmu, 3),
        "TMU":      round(avg_mu,  3),
        "pass@1(%)": round(pass1,  1),
        # raw sums for normalization
        "_sum_et":  sum(e["execution_time"]  for e in executable),
        "_sum_mmu": sum(e["max_memory_peak"] for e in executable),
        "_sum_mu":  sum(e["memory_usage"]    for e in executable),
        # per-problem dicts for normalization
        "_et":  {e["problem_idx"]: e["execution_time"]  for e in executable},
        "_mmu": {e["problem_idx"]: e["max_memory_peak"] for e in executable},
        "_mu":  {e["problem_idx"]: e["memory_usage"]    for e in executable},
    }


def normalize_against_canonical(stats: dict, canonical: dict, fixed_k: set | None = None) -> dict:
    # Special case: baseline built from overhead summary
    if "_baseline_totals" in stats:
        bt = stats["_baseline_totals"]
        allowed = fixed_k if fixed_k is not None else bt["ids"]
        common = allowed & set(canonical.keys())
        if not common:
            return stats
        # Baseline only has average metrics, not per-problem values.
        # Best approximation: avg_model / avg_canonical = (avg_model * n) / sum_canonical
        avg_c_et  = sum(canonical[k]["execution_time"]  for k in common) / len(common)
        avg_c_mmu = sum(canonical[k]["max_memory_peak"] for k in common) / len(common)
        avg_c_mu  = sum(canonical[k]["memory_usage"]    for k in common) / len(common)
        if avg_c_et > 0:
            stats["NET"]  = round(stats["ET"]  / avg_c_et,  4)
        if avg_c_mmu > 0:
            stats["NMU"]  = round(stats["MU"]  / avg_c_mmu, 4)
        if avg_c_mu > 0:
            stats["NTMU"] = round(stats["TMU"] / avg_c_mu,  4)
        return stats

    allowed = fixed_k if fixed_k is not None else set(stats["_et"].keys())
    common = allowed & set(stats["_et"].keys()) & set(canonical.keys())
    if not common:
        return stats

    net_vals  = [stats["_et"][k]  / canonical[k]["execution_time"]  for k in common if canonical[k]["execution_time"]  > 0]
    nmu_vals  = [stats["_mmu"][k] / canonical[k]["max_memory_peak"] for k in common if canonical[k]["max_memory_peak"] > 0]
    ntmu_vals = [stats["_mu"][k]  / canonical[k]["memory_usage"]    for k in common if canonical[k]["memory_usage"]    > 0]

    if net_vals:
        stats["NET"]  = round(sum(net_vals)  / len(net_vals),  4)
    if nmu_vals:
        stats["NMU"]  = round(sum(nmu_vals)  / len(nmu_vals),  4)
    if ntmu_vals:
        stats["NTMU"] = round(sum(ntmu_vals) / len(ntmu_vals), 4)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True,
                        help="Name used in result file names (e.g. qwen3.5-9b-vllm-test50)")
    parser.add_argument("--dataset", type=str, default="EffiBench",
                        choices=["EffiBench", "HumanEval", "MBPP"])
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs; auto-detected if omitted")
    parser.add_argument("--canonical-file", type=str, default=None,
                        help="Optional JSON with canonical solution profiling for normalization")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory for result files (default: ../results/{model_name})")
    parser.add_argument("--total-tasks", type=int, default=1000,
                        help="Total number of tasks in the benchmark (default: 1000)")
    parser.add_argument("--pin-k-to-last", action="store_true",
                        help="Fix the normalization problem subset to the last epoch's common_k "
                             "so NET/NMU/NTMU are comparable across all epochs")
    args = parser.parse_args()

    model_name  = args.model_name
    results_dir = args.results_dir or f"../results/{model_name}"
    dataset     = args.dataset

    baseline_path = os.path.join(results_dir, f"{dataset}_vllm_{model_name}.json")
    baseline = load_json(baseline_path)
    if baseline is None:
        print(f"ERROR: baseline file not found: {baseline_path}")
        return
    total_tasks = args.total_tasks
    print(f"Baseline file: {baseline_path}  ({len(baseline)} entries, pass@1 denominator={total_tasks})")

    if args.epochs is not None:
        num_epochs = args.epochs
    else:
        num_epochs = 0
        while os.path.exists(
            os.path.join(results_dir, f"{dataset}_vllm_{model_name}_{num_epochs + 1}.json")
        ):
            num_epochs += 1
    print(f"Epochs detected: {num_epochs}")

    canonical = {}
    if args.canonical_file and os.path.exists(args.canonical_file):
        canon_data = load_json(args.canonical_file)
        canonical = {
            e["problem_idx"]: {
                "execution_time":  e["execution_time"],
                "max_memory_peak": e["max_memory_peak"],
                "memory_usage":    e["memory_usage"],
            }
            for e in canon_data
            if e.get("executable")
        }
        print(f"Canonical solution entries loaded: {len(canonical)}")
    has_canonical = len(canonical) > 0

    overhead_path = os.path.join(results_dir, f"overhead_vllm_{model_name}.json")
    overhead = load_json(overhead_path)

    snapshots = [("baseline", baseline)]
    for ep in range(1, num_epochs + 1):
        path = os.path.join(results_dir, f"{dataset}_vllm_{model_name}_{ep}.json")
        data = load_json(path)
        if data is None:
            print(f"  [WARN] Epoch {ep} file not found: {path}")
            continue
        snapshots.append((f"vllm_{model_name}_{ep}", data))

    snapshot_stats: list[tuple[str, dict | None]] = []
    for i, (label, entries) in enumerate(snapshots):
        row_label = f"vllm_{model_name}" if label == "baseline" else label

        stats = compute_stats(entries, total_tasks)

        if stats is None and label == "baseline" and overhead and overhead.get("correct"):
            ep1_path = os.path.join(results_dir, f"{dataset}_vllm_{model_name}_1.json")
            ep1_data = load_json(ep1_path) or []
            ep1_exec = [e for e in ep1_data if e.get("executable")]
            stats = {
                "n":         overhead["correct"][0],
                "ET":        round(overhead["execution_time"][0], 3),
                "MU":        round(overhead["max_memory_peak"][0], 3),
                "TMU":       round(overhead["memory_usage"][0], 3),
                "pass@1(%)": round(overhead["correct"][0] / total_tasks * 100, 1),
                "_et":  {e["problem_idx"]: overhead["execution_time"][0]  for e in ep1_exec},
                "_mmu": {e["problem_idx"]: overhead["max_memory_peak"][0] for e in ep1_exec},
                "_mu":  {e["problem_idx"]: overhead["memory_usage"][0]    for e in ep1_exec},
                "_baseline_totals": {
                    "sum_et":  overhead["execution_time"][0]  * overhead["correct"][0],
                    "sum_mmu": overhead["max_memory_peak"][0] * overhead["correct"][0],
                    "sum_mu":  overhead["memory_usage"][0]    * overhead["correct"][0],
                    "ids": {e["problem_idx"] for e in ep1_exec},
                },
            }

        snapshot_stats.append((row_label, stats))

    fixed_k: set | None = None
    if has_canonical and args.pin_k_to_last:
        last_stats = next(
            (s for _, s in reversed(snapshot_stats) if s is not None and "_et" in s),
            None,
        )
        if last_stats is not None:
            fixed_k = set(last_stats["_et"].keys()) & set(canonical.keys())
            print(f"Pinned common_k to last epoch: {len(fixed_k)} problems")

    rows = []
    for row_label, stats in snapshot_stats:
        if stats is None:
            print(f"  [SKIP] {row_label} — no executable entries")
            continue

        if fixed_k is not None and "_et" in stats:
            active = fixed_k & set(stats["_et"].keys())
            if active:
                stats["ET"]  = round(sum(stats["_et"][k]  for k in active) / len(active), 3)
                stats["MU"]  = round(sum(stats["_mmu"][k] for k in active) / len(active), 3)
                stats["TMU"] = round(sum(stats["_mu"][k]  for k in active) / len(active), 3)
                stats["n"]   = len(active)

        if has_canonical:
            stats = normalize_against_canonical(stats, canonical, fixed_k)

        row = {"model": row_label}
        if has_canonical and "NET" in stats:
            print(
                f"  {row_label:40s}  ET={stats['ET']:.3f}s(NET={stats['NET']:.2f})"
                f"  MU={stats['MU']:.1f}MB(NMU={stats['NMU']:.2f})"
                f"  TMU={stats['TMU']:.3f}MB*s(NTMU={stats['NTMU']:.2f})"
                f"  pass@1={stats['pass@1(%)']:.1f}%  (n={stats['n']})"
            )
            row.update({k: stats[k] for k in ["ET", "NET", "MU", "NMU", "TMU", "NTMU", "pass@1(%)"]})
        else:
            print(
                f"  {row_label:40s}  ET={stats['ET']:.3f}s"
                f"  MU={stats['MU']:.1f}MB"
                f"  TMU={stats['TMU']:.3f}MB*s"
                f"  pass@1={stats['pass@1(%)']:.1f}%  (n={stats['n']})"
            )
            row.update({k: stats[k] for k in ["ET", "MU", "TMU", "pass@1(%)"]})

        rows.append(row)

    datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports", model_name)
    os.makedirs(reports_dir, exist_ok=True)
    csv_path = os.path.join(reports_dir, f"report_{model_name}_{datetime_str}.csv")
    fieldnames = ["model", "ET", "NET", "MU", "NMU", "TMU", "NTMU", "pass@1(%)"] \
        if has_canonical else ["model", "ET", "MU", "TMU", "pass@1(%)"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nReport saved to {csv_path}")


if __name__ == "__main__":
    main()

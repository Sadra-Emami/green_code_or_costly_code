import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from code_efficiency_calculator import import_pkg, TreeNode_text, ListNode_text


DEFAULT_DIFFICULTY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "datasets", "dataset_with_difficulty_repaired.json",
)


def load_difficulty_map(path: str = DEFAULT_DIFFICULTY_FILE) -> dict:
    if not os.path.exists(path):
        print(f"  [WARN] Difficulty file not found: {path}")
        return {}
    with open(path) as f:
        data = json.load(f)
    return {e["problem_idx"]: e.get("difficulty") for e in data if "problem_idx" in e}


def enrich_with_difficulty(entries: list, difficulty_map: dict) -> int:
    if not difficulty_map:
        return 0
    n = 0
    for e in entries:
        diff = difficulty_map.get(e.get("problem_idx"))
        if diff is not None:
            e["difficulty"] = diff
            n += 1
    return n


def extract_code(completion: str) -> str:
    """Strip markdown fences from a completion, return raw Python source."""
    if "```python" in completion:
        start = completion.find("```python") + len("```python")
        rest = completion[start:]
        end = rest.find("```")
        if end != -1:
            return rest[:end].strip()
        return rest.strip()
    if "```" in completion:
        start = completion.find("```") + 3
        rest = completion[start:]
        end = rest.find("```")
        if end != -1:
            return rest[:end].strip()
        return rest.strip()
    return completion.strip()


def is_parseable(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _find_class_names(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]


def assemble_script(entry: dict, code: str) -> str:
    test_case = entry["test_case"]
    test_lines = test_case.split("\n")[:100]
    test_case = "\n".join(test_lines)

    classes = _find_class_names(code)

    first_test_line = test_case.strip().split("\n")[0].strip()
    tests_create_solution = first_test_line.startswith("solution =") or first_test_line.startswith("solution=")

    glue = ""
    if "Solution" in classes:
        if not tests_create_solution:
            glue = "\nsolution = Solution()\n"
    elif classes:
        primary = classes[0]
        if not tests_create_solution:
            glue = f"\nSolution = {primary}\nsolution = Solution()\n"
        else:
            glue = f"\nSolution = {primary}\n"
    else:
        if not tests_create_solution:
            glue = "\n"

    parts = [
        import_pkg,
        TreeNode_text,
        ListNode_text,
        code,
        glue,
        test_case,
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_script(script: str, timeout: int = 20) -> tuple[bool, str]:
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(script)
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, ""
        err = (result.stderr or result.stdout or "").strip()
        return False, err[-500:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Per-entry validation
# ---------------------------------------------------------------------------

def validate_entry(entry: dict, timeout: int = 20) -> dict:
    problem_idx = entry["problem_idx"]
    completion = entry["completion"]
    code = extract_code(completion)

    parseable = is_parseable(code)
    if not parseable:
        return {
            "problem_idx": problem_idx,
            "parseable": False,
            "passes_tests": False,
            "detail": "SyntaxError",
        }

    script = assemble_script(entry, code)
    passed, detail = run_script(script, timeout=timeout)
    return {
        "problem_idx": problem_idx,
        "parseable": True,
        "passes_tests": passed,
        "detail": detail,
    }


def validate_file(
    json_path: str,
    workers: int = 8,
    timeout: int = 20,
) -> list[dict]:
    with open(json_path) as f:
        dataset = json.load(f)

    results = [None] * len(dataset)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(validate_entry, entry, timeout): i
            for i, entry in enumerate(dataset)
        }
        for fut in tqdm(as_completed(futures), total=len(dataset), desc=os.path.basename(json_path)):
            i = futures[fut]
            results[i] = fut.result()

    total = len(results)
    parseable = sum(1 for r in results if r["parseable"])
    passing = sum(1 for r in results if r["passes_tests"])

    print(f"\n  {os.path.basename(json_path)}")
    print(f"    Total:        {total}")
    print(f"    Parseable:    {parseable}/{total} ({100*parseable/total:.1f}%)")
    print(f"    Passes tests: {passing}/{total} ({100*passing/total:.1f}%)")

    return results


def validate_and_save(
    json_path: str,
    output_dir: str,
    workers: int = 8,
    timeout: int = 20,
    difficulty_map: dict | None = None,
) -> list[dict]:
    with open(json_path) as f:
        dataset = json.load(f)

    results = validate_file(json_path, workers=workers, timeout=timeout)

    passing_idxs = {r["problem_idx"] for r in results if r["passes_tests"]}
    filtered = [d for d in dataset if d["problem_idx"] in passing_idxs]

    if difficulty_map:
        n_enriched = enrich_with_difficulty(filtered, difficulty_map)
        print(f"    Enriched with difficulty: {n_enriched}/{len(filtered)}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, os.path.basename(json_path))
    with open(out_path, "w") as f:
        json.dump(filtered, f, indent=4)

    print(f"    Saved: {len(filtered)}/{len(dataset)} -> {out_path}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate EffiBench completions")
    parser.add_argument("json_files", nargs="+", help="Path(s) to result JSON files")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument(
        "--save", action="store_true",
        help="Save filtered files into <model>-validated/ folders",
    )
    parser.add_argument(
        "--difficulty-file", type=str, default=DEFAULT_DIFFICULTY_FILE,
        help="Path to dataset_with_difficulty_repaired.json for enrichment",
    )
    args = parser.parse_args()

    difficulty_map = load_difficulty_map(args.difficulty_file)
    if difficulty_map:
        print(f"Loaded difficulty for {len(difficulty_map)} problems from {args.difficulty_file}")

    from collections import defaultdict
    groups = defaultdict(list)
    for path in args.json_files:
        parent = os.path.dirname(os.path.abspath(path))
        groups[parent].append(path)

    for parent, paths in groups.items():
        model_dir = os.path.basename(parent)
        output_dir = os.path.join(os.path.dirname(parent), model_dir + "-validated")

        base_path = None
        epoch_paths = []
        for p in sorted(paths):
            fname = os.path.basename(p)
            stem = fname.rsplit(".json", 1)[0]
            if stem[-1].isdigit() and "_" in stem and stem.rsplit("_", 1)[1].isdigit():
                epoch_paths.append(p)
            else:
                base_path = p

        base_passing_idxs = None
        if base_path:
            if args.save:
                results = validate_and_save(base_path, output_dir, workers=args.workers, timeout=args.timeout, difficulty_map=difficulty_map)
                base_passing_idxs = {r["problem_idx"] for r in results if r["passes_tests"]}
            else:
                results = validate_file(base_path, workers=args.workers, timeout=args.timeout)
                base_passing_idxs = {r["problem_idx"] for r in results if r["passes_tests"]}

        for ep_path in epoch_paths:
            if args.save:
                with open(ep_path) as f:
                    dataset = json.load(f)

                if base_passing_idxs is not None:
                    before = len(dataset)
                    dataset = [d for d in dataset if d["problem_idx"] in base_passing_idxs]
                    if before != len(dataset):
                        print(f"    Filtered to base-passing problems: {len(dataset)}/{before}")

                results = [None] * len(dataset)
                from concurrent.futures import ThreadPoolExecutor, as_completed
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futures = {
                        pool.submit(validate_entry, entry, args.timeout): i
                        for i, entry in enumerate(dataset)
                    }
                    for fut in tqdm(as_completed(futures), total=len(dataset), desc=os.path.basename(ep_path)):
                        i = futures[fut]
                        results[i] = fut.result()

                total = len(results)
                passing = sum(1 for r in results if r["passes_tests"])
                passing_idxs = {r["problem_idx"] for r in results if r["passes_tests"]}
                filtered = [d for d in dataset if d["problem_idx"] in passing_idxs]

                print(f"\n  {os.path.basename(ep_path)}")
                print(f"    Total:        {total}")
                print(f"    Passes tests: {passing}/{total} ({100*passing/total:.1f}%)")

                if difficulty_map:
                    n_enriched = enrich_with_difficulty(filtered, difficulty_map)
                    print(f"    Enriched with difficulty: {n_enriched}/{len(filtered)}")

                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, os.path.basename(ep_path))
                with open(out_path, "w") as f:
                    json.dump(filtered, f, indent=4)
                print(f"    Saved: {len(filtered)}/{total} -> {out_path}")
            else:
                validate_file(ep_path, workers=args.workers, timeout=args.timeout)

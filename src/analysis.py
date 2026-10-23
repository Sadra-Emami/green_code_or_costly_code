"""Post-sweep analysis utilities.

Subcommands
-----------
variants
    Compare variant runs (default / t0.5 / ctx65536 / mt1024) for each base
    model. Reads the latest report CSV from reports/<variant>/ and writes:
      reports/_compare/<base>.csv            long-form: variant x epoch x metric
      reports/_compare/<base>_pivot.csv      wide-form: epoch x metric, columns per variant
      reports/_compare/_summary.csv          best-NTMU and pass@1 per (base, variant)

energy-report
    Aggregate per-variant energy_stats_*.json into a single comparison CSV.
    Reads:  energy_results/<variant>/energy_stats_*.json
    Writes: reports/_compare/_energy.csv

canonical-energy
    Compare each variant's base-generation and last-epoch energy results against
    the canonical solution. Reads energy_results from results/<variant>-validated/
    and writes:
      reports/_compare/canonical_vs_variants_energy.csv

Usage
-----
    python analysis.py variants
    python analysis.py variants --base qwen3.5-4b
    python analysis.py energy-report
    python analysis.py canonical-energy
    python analysis.py canonical-energy --canonical path/to/canonical.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
RESULTS = ROOT / "results"
ENERGY_DIR = ROOT / "energy_results"

KNOWN_SUFFIXES = ("_t0.5", "_ctx65536", "_mt1024")
METRICS = ["ET", "NET", "MU", "NMU", "TMU", "NTMU", "pass@1(%)"]


def split_variant(variant: str) -> tuple[str, str]:
    """'qwen3.5-4b_t0.5' -> ('qwen3.5-4b', 't0.5'). Default tag = 'default'."""
    for s in KNOWN_SUFFIXES:
        if variant.endswith(s):
            return variant[: -len(s)], s.lstrip("_")
    return variant, "default"


_REPEAT_RE_V = re.compile(r"^(.+)_r(\d+)$")
_TAG_SUFFIXES_V = KNOWN_SUFFIXES


def _split_variant_repeat(variant: str) -> tuple[str, str, int | None]:
    for s in _TAG_SUFFIXES_V:
        if variant.endswith(s):
            stem = variant[: -len(s)]
            m = _REPEAT_RE_V.match(stem)
            if m:
                base = m.group(1)
                if base.endswith(s):
                    base = base[: -len(s)]
                return base, s.lstrip("_"), int(m.group(2))
            return stem, s.lstrip("_"), None
    m = _REPEAT_RE_V.match(variant)
    if m:
        return m.group(1), "default", int(m.group(2))
    return variant, "default", None


def _avg_rows(repeat_row_sets: list[list[dict]]) -> list[dict]:
    if len(repeat_row_sets) == 1:
        return repeat_row_sets[0]

    RAW_NORM = [("ET", "NET"), ("MU", "NMU"), ("TMU", "NTMU")]
    all_epochs = list(dict.fromkeys(r["epoch"] for rows in repeat_row_sets for r in rows))
    averaged = []
    for epoch in all_epochs:
        candidates = [r for rows in repeat_row_sets for r in rows if r["epoch"] == epoch]
        row: dict = {"epoch": epoch}
        for m in ("ET", "MU", "TMU", "pass@1(%)"):
            vals = [c[m] for c in candidates if c[m] is not None]
            row[m] = round(sum(vals) / len(vals), 4) if vals else None
        for raw_key, norm_key in RAW_NORM:
            canonicals = [
                c[raw_key] / c[norm_key]
                for c in candidates
                if c[raw_key] is not None and c[norm_key] is not None and c[norm_key] != 0
            ]
            if canonicals and row[raw_key] is not None:
                row[norm_key] = round(row[raw_key] / (sum(canonicals) / len(canonicals)), 4)
            else:
                row[norm_key] = None
        averaged.append(row)
    return averaged


def _latest_report(variant_dir: Path) -> Path | None:
    candidates = sorted(variant_dir.glob(f"report_{variant_dir.name}_*.csv"))
    return candidates[-1] if candidates else None


def _epoch_label(model_col: str, variant: str) -> str:
    if model_col == f"vllm_{variant}":
        return "base"
    m = re.match(rf"^vllm_{re.escape(variant)}_(\d+)$", model_col)
    return m.group(1) if m else model_col


def _load_report(path: Path, variant: str) -> list[dict]:
    out = []
    with path.open() as f:
        for r in csv.DictReader(f):
            row = {"epoch": _epoch_label(r["model"], variant)}
            for m in METRICS:
                v = r.get(m, "")
                row[m] = float(v) if v not in ("", None) else None
            out.append(row)
    return out


def _run_variants(args: argparse.Namespace) -> None:
    out_dir = REPORTS / "_compare"
    variants = sorted(d.name for d in REPORTS.iterdir() if d.is_dir() and d.name != "_compare")

    repeats_buf: dict[str, dict[str, list[list[dict]]]] = defaultdict(lambda: defaultdict(list))
    for v in variants:
        rpt = _latest_report(REPORTS / v)
        if rpt is None:
            print(f"  [skip] no report under reports/{v}/")
            continue
        base, tag, repeat_idx = _split_variant_repeat(v)
        if args.primary_only and repeat_idx is not None:
            continue
        if args.base and base != args.base:
            continue
        rows = _load_report(rpt, v)
        repeats_buf[base][tag].append(rows)
        label = f"r{repeat_idx}" if repeat_idx is not None else "primary"
        print(f"  {base:40s} {tag:9s} {label:7s} <- {rpt.name} ({len(rows)} rows)")

    grouped: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for base, tag_sets in repeats_buf.items():
        for tag, row_sets in tag_sets.items():
            grouped[base][tag] = _avg_rows(row_sets)
            if len(row_sets) > 1:
                print(f"  {base:40s} {tag:9s} averaged {len(row_sets)} repeat(s)")

    if not grouped:
        print("No reports found. Run report_all.sh first.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for base, tag_rows in sorted(grouped.items()):
        long_path = out_dir / f"{base}{args.suffix}.csv"
        with long_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["base_model", "variant", "epoch", *METRICS])
            for tag in sorted(tag_rows):
                for r in tag_rows[tag]:
                    w.writerow([base, tag, r["epoch"], *(r[m] for m in METRICS)])
        print(f"  wrote {long_path}")

        all_epochs = sorted(
            {r["epoch"] for rows in tag_rows.values() for r in rows},
            key=lambda e: (e != "base", int(e) if e.isdigit() else -1),
        )
        all_tags = sorted(tag_rows.keys())

        pivot_path = out_dir / f"{base}_pivot{args.suffix}.csv"
        with pivot_path.open("w", newline="") as f:
            w = csv.writer(f)
            header = ["epoch", "metric", *all_tags]
            if "default" in all_tags:
                header += [f"Δ_{t}_vs_default" for t in all_tags if t != "default"]
            w.writerow(header)
            for epoch in all_epochs:
                for metric in METRICS:
                    vals = {}
                    for tag in all_tags:
                        match = next((r for r in tag_rows[tag] if r["epoch"] == epoch), None)
                        vals[tag] = match[metric] if match else None
                    row = [epoch, metric, *(vals[t] for t in all_tags)]
                    if "default" in all_tags and vals["default"] is not None:
                        for t in all_tags:
                            if t == "default":
                                continue
                            other = vals[t]
                            row.append(round(other - vals["default"], 4) if other is not None else None)
                    w.writerow(row)
        print(f"  wrote {pivot_path}")

        for tag, rows in tag_rows.items():
            entry = {"base_model": base, "variant": tag, "n_epochs": len(rows)}
            if args.best_worst:
                for m in METRICS:
                    cands = [(r["epoch"], r[m]) for r in rows if r[m] is not None]
                    if not cands:
                        entry[f"best_{m}"] = entry[f"best_{m}_iter"] = ""
                        entry[f"worst_{m}"] = entry[f"worst_{m}_iter"] = ""
                        continue
                    higher_better = m == "pass@1(%)"
                    best = (max if higher_better else min)(cands, key=lambda c: c[1])
                    worst = (min if higher_better else max)(cands, key=lambda c: c[1])
                    entry[f"best_{m}"] = best[1]
                    entry[f"best_{m}_iter"] = best[0]
                    entry[f"worst_{m}"] = worst[1]
                    entry[f"worst_{m}_iter"] = worst[0]
            else:
                base_row = next((r for r in rows if r["epoch"] == "base"), None)
                numeric = [r for r in rows if r["epoch"].isdigit()]
                last_row = max(numeric, key=lambda r: int(r["epoch"])) if numeric else None
                entry["last_epoch"] = last_row["epoch"] if last_row else ""
                for m in METRICS:
                    entry[f"base_{m}"] = base_row[m] if base_row else ""
                    entry[f"last_{m}"] = last_row[m] if last_row else ""
            summary_rows.append(entry)

    summary_path = out_dir / f"_summary{args.suffix}.csv"
    if args.best_worst:
        fieldnames = ["base_model", "variant", "n_epochs"]
        for m in METRICS:
            fieldnames += [f"best_{m}", f"best_{m}_iter", f"worst_{m}", f"worst_{m}_iter"]
    else:
        fieldnames = ["base_model", "variant", "n_epochs", "last_epoch"]
        for m in METRICS:
            fieldnames += [f"base_{m}", f"last_{m}"]
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(sorted(summary_rows, key=lambda r: (r["base_model"], r["variant"])))
    print(f"\nGlobal summary -> {summary_path}")


_REPEAT_SUFFIX_RE_E = re.compile(r"_r\d+$")


def _latest_stats(variant_dir: Path) -> Path | None:
    averaged = variant_dir / "energy_stats_averaged.json"
    if averaged.exists():
        return averaged
    files = sorted(variant_dir.glob("energy_stats_*.json"))
    return files[-1] if files else None


def _row_from_stats(variant: str, path: Path) -> dict:
    d = json.loads(path.read_text())
    dt = d.get("device_totals", {})
    summ = d.get("summary", {})

    cpu  = dt.get("CPU",  {})
    gpu  = dt.get("GPU",  {})
    dram = dt.get("DRAM", {})

    cpu_t  = cpu.get("total",  0.0) or 0.0
    gpu_t  = gpu.get("total",  0.0) or 0.0
    dram_t = dram.get("total", 0.0) or 0.0
    total  = summ.get("total_energy") or (cpu_t + gpu_t + dram_t)

    base, tag = split_variant(variant)
    return {
        "variant":       variant,
        "base_model":    base,
        "tag":           tag,
        "total_J":       round(total, 2),
        "cpu_J":         round(cpu_t,  2),
        "gpu_J":         round(gpu_t,  2),
        "dram_J":        round(dram_t, 2),
        "cpu_pct":       round(100 * cpu_t  / total, 2) if total else "",
        "gpu_pct":       round(100 * gpu_t  / total, 2) if total else "",
        "dram_pct":      round(100 * dram_t / total, 2) if total else "",
        "cpu_dynamic_J": round(cpu.get("dynamic", 0.0) or 0.0, 2),
        "gpu_dynamic_J": round(gpu.get("dynamic", 0.0) or 0.0, 2),
        "avg_power_w":   round(summ.get("average_power_w", 0.0) or 0.0, 4),
        "source":        path.name,
    }


def _run_energy_report(_args: argparse.Namespace) -> None:
    out = REPORTS / "_compare" / "_energy.csv"
    rows = []
    missing = []
    for d in sorted(ENERGY_DIR.iterdir()):
        if not d.is_dir():
            continue
        if _REPEAT_SUFFIX_RE_E.search(d.name):
            continue
        latest = _latest_stats(d)
        if latest is None:
            missing.append(d.name)
            continue
        rows.append(_row_from_stats(d.name, latest))

    if not rows:
        print(f"No energy_stats_*.json found under {ENERGY_DIR}")
        return

    rows.sort(key=lambda r: (r["base_model"], r["tag"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {out}")
    if missing:
        print(f"\nVariants missing energy_stats: {missing}")
    print(f"\n{'variant':50s}{'total_J':>14s}{'cpu_J':>14s}{'gpu_J':>14s}{'cpu_%':>8s}{'gpu_%':>8s}")
    for r in rows:
        print(f"{r['variant']:50s}{r['total_J']:>14,.0f}{r['cpu_J']:>14,.0f}"
              f"{r['gpu_J']:>14,.0f}{r['cpu_pct']:>8}{r['gpu_pct']:>8}")


_CANONICAL_RESULTS = ROOT / "datasets" / "canonical_input_energy_results.json"
_TAG_ALT = "|".join(re.escape(s.lstrip("_")) for s in KNOWN_SUFFIXES)
_REPEAT_RE_C = re.compile(rf"^(.*)_r(\d+)((?:_(?:{_TAG_ALT}))?)$")


def _total_energy(entry: dict) -> float:
    stats = entry.get("energy_stats") or {}
    energy = entry.get("energy") or {}
    runs_total = entry.get("runs_total") or 1

    def field(name: str) -> float:
        mean = (stats.get(name) or {}).get("mean")
        if mean is not None:
            return mean
        return (energy.get(name) or 0.0) / runs_total

    return field("cpu_total_energy_j") + field("dram_total_energy_j") + field("gpu_total_energy_j")


def _total_cycles(entry: dict) -> float:
    mean = (entry.get("performance_stats") or {}).get("unhalted_cycles", {}).get("mean")
    if mean is not None:
        return mean
    perf = entry.get("performance") or {}
    runs_total = entry.get("runs_total") or 1
    return (perf.get("unhalted_cycles") or 0.0) / runs_total


def _load_canonical(canonical_path: Path) -> dict[int, dict[str, float]]:
    data = json.loads(canonical_path.read_text())
    lookup = {}
    for entry in data:
        energy = entry.get("energy") or {}
        if energy.get("cpu_total_energy_j") is None:
            continue
        lookup[entry["problem_idx"]] = {
            "energy": _total_energy(entry),
            "cycles": _total_cycles(entry),
        }
    return lookup


def _discover_base_and_last_epoch(variant: str, dataset: str) -> dict[str, Path]:
    """Find the base-generation file and the highest-numbered epoch file for a
    variant under results/<variant>-validated/, restricted to files that have
    a matching *_energy_results.json."""
    vdir = RESULTS / f"{variant}-validated"
    if not vdir.is_dir():
        return {}

    def _has_energy(fp: Path) -> bool:
        return fp.with_name(fp.stem + "_energy_results.json").exists()

    labels: dict[str, Path] = {}

    base_fp = vdir / f"{dataset}_vllm_{variant}.json"
    if base_fp.exists() and _has_energy(base_fp):
        labels["base"] = base_fp

    epoch_re = re.compile(rf"^{re.escape(dataset)}_vllm_{re.escape(variant)}_(\d+)\.json$")
    epoch_files: dict[int, Path] = {}
    for fp in vdir.glob(f"{dataset}_vllm_{variant}_*.json"):
        m = epoch_re.match(fp.name)
        if m and _has_energy(fp):
            epoch_files[int(m.group(1))] = fp
    if epoch_files:
        last_epoch = max(epoch_files)
        labels["last"] = epoch_files[last_epoch]

    return labels


def _load_problem_metrics(fp: Path, canonical: dict[int, dict]) -> dict[int, dict[str, float]]:
    energy_results_path = fp.with_name(fp.stem + "_energy_results.json")
    entries = json.loads(energy_results_path.read_text())
    out = {}
    for e in entries:
        pidx = e.get("problem_idx")
        energy = e.get("energy") or {}
        if energy.get("cpu_total_energy_j") is None or pidx not in canonical:
            continue
        out[pidx] = {"energy": _total_energy(e), "cycles": _total_cycles(e)}
    return out


def _split_repeat_c(variant: str) -> tuple[str, str, int]:
    m = _REPEAT_RE_C.match(variant)
    if m:
        prefix, run_num, trailing_tag = m.group(1), int(m.group(2)), m.group(3)
        if trailing_tag:
            base = prefix[: -len(trailing_tag)]
            tag = trailing_tag.lstrip("_")
        else:
            base = prefix
            tag = ""
        return base, tag, run_num
    for suf in KNOWN_SUFFIXES:
        if variant.endswith(suf):
            return variant[: -len(suf)], suf.lstrip("_"), 0
    return variant, "", 0


def _remove_outliers_iqr(data: list[float]) -> list[float]:
    if len(data) < 4:
        return data
    s = sorted(data)
    q1 = statistics.quantiles(s, n=4)[0]
    q3 = statistics.quantiles(s, n=4)[2]
    iqr = q3 - q1
    return [x for x in data if q1 - 1.5 * iqr <= x <= q3 + 1.5 * iqr]


def _run_canonical_energy(args: argparse.Namespace) -> None:
    canonical_path = args.canonical
    out_path = args.out
    dataset = args.dataset
    canonical = _load_canonical(canonical_path)

    variant_dirs = sorted(
        p.name[: -len("-validated")]
        for p in RESULTS.glob("*-validated")
        if p.is_dir()
    )

    variant_metrics: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    for variant in variant_dirs:
        labels = _discover_base_and_last_epoch(variant, dataset)
        if not labels:
            continue
        per_label: dict[str, dict[int, dict[str, float]]] = {
            label: _load_problem_metrics(fp, canonical) for label, fp in labels.items()
        }
        variant_metrics[variant] = per_label

    if not variant_metrics:
        print(f"No variants with base/last-epoch energy results found under {RESULTS}/*-validated/")
        return

    groups: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for variant in variant_metrics:
        base, tag, run = _split_repeat_c(variant)
        groups[(base, tag)].append((run, variant))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "variant", "label", "num_runs", "num_common_problems",
            "variant_total_energy_j", "variant_total_energy_j_stdev", "CV_pct",
            "canonical_total_energy_j", "diff_j", "pct_overhead", "NEU",
            "canonical_unhalted_cycles", "variant_unhalted_cycles",
        ])
        for (base, tag), runs in sorted(groups.items()):
            runs.sort()
            name = f"{base}_{tag}" if tag else base
            keysets = [
                set(metrics.keys())
                for _, variant in runs
                for label, metrics in variant_metrics[variant].items()
                if label in ("base", "last")
            ]
            common = set.intersection(*keysets) if keysets else set()
            canonical_total_energy = sum(canonical[p]["energy"] for p in common)
            canonical_total_cycles = sum(canonical[p]["cycles"] for p in common)

            for label in ("base", "last"):
                run_energy_totals = [
                    sum(variant_metrics[v][label][p]["energy"] for p in common)
                    for _, v in runs if label in variant_metrics[v]
                ]
                run_cycle_totals = [
                    sum(variant_metrics[v][label][p]["cycles"] for p in common)
                    for _, v in runs if label in variant_metrics[v]
                ]
                if not run_energy_totals:
                    continue
                run_energy_totals = _remove_outliers_iqr(run_energy_totals)
                run_cycle_totals  = _remove_outliers_iqr(run_cycle_totals)
                energy_ratios = [
                    variant_metrics[v][label][p]["energy"] / canonical[p]["energy"]
                    for _, v in runs if label in variant_metrics[v]
                    for p in common if canonical[p]["energy"] > 0
                ]
                avg_total   = statistics.mean(run_energy_totals)
                stdev_total = statistics.stdev(run_energy_totals) if len(run_energy_totals) > 1 else 0.0
                avg_cycles  = statistics.mean(run_cycle_totals) if run_cycle_totals else float("nan")
                diff = avg_total - canonical_total_energy
                pct  = (diff / canonical_total_energy * 100.0) if canonical_total_energy else float("nan")
                neu  = statistics.mean(energy_ratios) if energy_ratios else float("nan")
                cv   = (stdev_total / avg_total * 100.0) if avg_total else float("nan")
                writer.writerow([
                    name, label, len(run_energy_totals), len(common),
                    f"{avg_total:.6f}", f"{stdev_total:.6f}", f"{cv:.2f}",
                    f"{canonical_total_energy:.6f}", f"{diff:.6f}", f"{pct:.2f}", f"{neu:.4f}",
                    f"{canonical_total_cycles:.2f}", f"{avg_cycles:.2f}",
                ])
    print(f"wrote -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # variants
    p_v = sub.add_parser("variants", help="Compare parameter variants across models.")
    p_v.add_argument("--base", default=None, help="Restrict to one base model.")
    p_v.add_argument("--suffix", default="", help="Suffix appended to output filenames.")
    p_v.add_argument("--primary-only", action="store_true",
                     help="Skip _rN repeat variants.")
    p_v.add_argument("--best-worst", action="store_true",
                     help="Summary shows best and worst iteration per metric.")

    # energy-report
    sub.add_parser("energy-report", help="Aggregate energy_stats JSON files into a CSV.")

    # canonical-energy
    p_c = sub.add_parser("canonical-energy",
                          help="Compare variant energy against canonical solution.")
    p_c.add_argument("--canonical", type=Path, default=_CANONICAL_RESULTS,
                     help="Path to canonical energy results JSON.")
    p_c.add_argument("--out", type=Path,
                     default=REPORTS / "_compare" / "canonical_vs_variants_energy.csv",
                     help="Output CSV path.")
    p_c.add_argument("--dataset", default="EffiBench",
                     help="Dataset name used in result filenames (default: EffiBench).")

    args = parser.parse_args()
    {"variants": _run_variants,
     "energy-report": _run_energy_report,
     "canonical-energy": _run_canonical_energy}[args.cmd](args)


if __name__ == "__main__":
    main()

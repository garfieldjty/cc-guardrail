#!/usr/bin/env python3
"""
Run CC-BOS attack + eval across multiple target model configurations,
then aggregate all results into a side-by-side comparison table.

Usage:
  python run_experiments.py                     # run all targets, mode=all
  python run_experiments.py --mode eval         # only eval phase (reuse existing attack results)
  python run_experiments.py --targets cf_llama  # run specific tag(s) only
  python run_experiments.py --compare-only      # skip runs, just print comparison
  python run_experiments.py --limit 5           # pass --limit to main.py
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RESULTS_BASE = Path("results")

TARGETS = [
    {
        "tag": "or_gemma-4-26b",
        "provider": "openrouter",
        "model": "google/gemma-4-26b-a4b-it",
    },
    {
        "tag": "or_qwen3.5-flash",
        "provider": "openrouter",
        "model": "qwen/qwen3.5-flash-02-23",
    },
    {
        "tag": "or_deepseek-v3.2",
        "provider": "openrouter",
        "model": "deepseek/deepseek-v3.2",
    },
]


def run_target(target: dict, mode: str = "all", limit: int | None = None) -> bool:
    tag = target["tag"]
    results_dir = RESULTS_BASE / tag
    results_dir.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "TARGET_PROVIDER": target["provider"],
        "TARGET_MODEL": target["model"],
        "RESULTS_DIR": str(results_dir),
    }

    cmd = [sys.executable, "main.py", "--mode", mode]
    if limit:
        cmd += ["--limit", str(limit)]

    log_file = results_dir / "run.log"
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"[{tag}]  {target['provider']} / {target['model']}")
    print(f"  results → {results_dir}    log → {log_file}")
    print(sep)

    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    lines = []
    with open(log_file, "w") as fh:
        for line in proc.stdout:
            print(line, end="")
            fh.write(line)
            lines.append(line)
    proc.wait()
    elapsed = time.monotonic() - t0

    if proc.returncode != 0:
        print(f"\n[FAIL] {tag} exited {proc.returncode} after {elapsed:.0f}s — see {log_file}")
        return False

    print(f"\n[OK] {tag} finished in {elapsed:.0f}s")
    return True


def _pct(v) -> str:
    return f"{v:.1%}" if v is not None else "N/A"


def _ms(v) -> str:
    return f"{v:.0f}" if v is not None else "N/A"


def collect_results() -> list[dict]:
    rows = []
    for t in TARGETS:
        path = RESULTS_BASE / t["tag"] / "summary.json"
        if not path.exists():
            rows.append({"tag": t["tag"], "model": t["model"], "missing": True})
            continue
        s = json.loads(path.read_text())
        rows.append({
            "tag": t["tag"],
            "provider": t["provider"],
            "model": t["model"],
            "baseline_asr": s.get("overall_baseline_asr"),
            "defended_asr": s.get("overall_defended_asr"),
            "fp_rate": s.get("false_positive_rate"),
            "latency_ms": s.get("avg_guardrail_latency_ms"),
            "v1_asr": s.get("v1_overall_defended_asr"),
            "v1_fp": s.get("v1_false_positive_rate"),
            "v1_latency_ms": s.get("v1_avg_latency_ms"),
            "lg_asr": s.get("lg_overall_defended_asr"),
            "lg_fp": s.get("lg_false_positive_rate"),
            "lg_latency_ms": s.get("lg_avg_latency_ms"),
            "per_category": s.get("per_category", []),
        })
    return rows


def print_comparison(rows: list[dict]):
    W = 150
    print(f"\n{'='*W}")
    print("EXPERIMENT COMPARISON")
    print(f"{'='*W}")

    # Main table
    hdr = (f"{'Tag':<22} {'Model':<36} {'Baseline':>8} "
           f"{'V2(T+J)↓':>9} {'V2-FP':>6} {'V2-Lat':>7} "
           f"{'V1(Dir)↓':>9} {'V1-FP':>6} {'V1-Lat':>7} "
           f"{'LGuard↓':>8} {'LG-FP':>6}")
    print(hdr)
    print("-" * W)

    for r in rows:
        if r.get("missing"):
            print(f"{r['tag']:<22} {r['model']:<36}  (no results yet)")
            continue
        print(
            f"{r['tag']:<22} {r['model']:<36} "
            f"{_pct(r['baseline_asr']):>8} "
            f"{_pct(r['defended_asr']):>9} "
            f"{_pct(r['fp_rate']):>6} "
            f"{_ms(r['latency_ms']):>7} "
            f"{_pct(r['v1_asr']):>9} "
            f"{_pct(r['v1_fp']):>6} "
            f"{_ms(r['v1_latency_ms']):>7} "
            f"{_pct(r['lg_asr']):>8} "
            f"{_pct(r['lg_fp']):>6}"
        )

    print(f"{'='*W}")
    print("Baseline = no guardrail  |  V2(T+J) = translate+judge (new)  |  "
          "V1(Dir) = direct-classify (old)  |  LGuard = Llama Guard  |  FP = false positive rate")

    # Per-category breakdown
    all_cats = sorted({row["category"] for r in rows if not r.get("missing")
                       for row in r["per_category"]})
    if all_cats:
        print(f"\n{'--- Per-category defended ASR (CC-BOS guardrail) ---':^{W}}")
        cat_hdr = f"{'Category':<22}" + "".join(f"{r['tag']:>22}" for r in rows if not r.get("missing"))
        print(cat_hdr)
        print("-" * W)
        cat_map = {
            r["tag"]: {row["category"]: row for row in r["per_category"]}
            for r in rows if not r.get("missing")
        }
        for cat in all_cats:
            line = f"{cat:<22}"
            for r in rows:
                if r.get("missing"):
                    continue
                entry = cat_map[r["tag"]].get(cat)
                val = _pct(entry["defended_asr"]) if entry else "N/A"
                line += f"{val:>22}"
            print(line)
        print(f"{'='*W}")


def save_comparison(rows: list[dict]):
    out = RESULTS_BASE / "comparison.json"
    RESULTS_BASE.mkdir(exist_ok=True)
    payload = [{k: v for k, v in r.items() if k != "per_category"} for r in rows]
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nComparison saved → {out}")


def main():
    parser = argparse.ArgumentParser(description="Run CC-BOS experiments across multiple targets")
    parser.add_argument("--mode", choices=["attack", "eval", "all"], default="all")
    parser.add_argument("--targets", nargs="+", metavar="TAG",
                        help="Run only these tags (e.g. cf_llama-3.1-8b or_gemma-4-26b)")
    parser.add_argument("--limit", type=int, default=None, help="Limit behaviors per run")
    parser.add_argument("--compare-only", action="store_true",
                        help="Skip running, just aggregate and print existing results")
    args = parser.parse_args()

    targets = TARGETS
    if args.targets:
        targets = [t for t in TARGETS if t["tag"] in args.targets]
        if not targets:
            sys.exit(f"No targets matched. Available tags: {[t['tag'] for t in TARGETS]}")

    if not args.compare_only:
        failed = []
        for target in targets:
            ok = run_target(target, mode=args.mode, limit=args.limit)
            if not ok:
                failed.append(target["tag"])
        if failed:
            print(f"\n[WARN] Failed: {failed}")

    rows = collect_results()
    print_comparison(rows)
    save_comparison(rows)


if __name__ == "__main__":
    main()

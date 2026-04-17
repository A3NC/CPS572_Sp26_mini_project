"""
Grid search over hyperparameters for train_and_publish.py.

Trains multiple models with different hyperparameter combinations and evaluates
each on IFEval, GSM8K, and HumanEval. Stops early if a config meets all baselines.

Baseline targets:
    IFEval:    45.0%
    GSM8K:     50.0%
    HumanEval: 30.0%

Usage:
    python grid_search.py
    python grid_search.py --limit 50          # quick eval (fewer samples per task)
    python grid_search.py --no_publish        # skip publishing checkpoints
    python grid_search.py --output results/grid_search_results.json
    python grid_search.py --resume            # skip configs already in output file
"""

import argparse
import asyncio
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Baseline targets 
BASELINES = {
    "google/IFEval/final_acc":   0.45,
    "openai/gsm8k/accuracy":             0.50,
    "openai/openai_humaneval/accuracy":  0.30,
}

#  Search space to search over
SEARCH_SPACE = {
    "lr":         [1e-4, 3e-4],
    "rank":       [32, 64],
    "batch_size": [4, 8],
    "num_steps":  [20, 50],
}

# Generation settings to use during evaluation
EVAL_TEMPERATURE = 0.0
EVAL_TOP_P       = 1.0
 
BASE_MODEL = "meta-llama/Llama-3.1-8B"
 
# grid_search.py lives inside the evaluation/ directory alongside the scripts.
# Use __file__'s own directory for sibling scripts; parent dir for output JSON.
EVAL_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(EVAL_DIR)   # one level up (the project root)
TRAIN_SCRIPT = os.path.join(EVAL_DIR, "train_and_publish.py")
EVAL_SCRIPT  = os.path.join(EVAL_DIR, "eval_all.py")
 
# Helpers
def all_configs():
    keys   = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]
 
 
def config_key(cfg):
    """Stable string key for a config dict."""
    return json.dumps(cfg, sort_keys=True)
 
 
def passes_baselines(metrics: dict) -> dict:
    """
    Returns a dict of {metric: (score, passed)} for the three baseline metrics.
    """
    results = {}
    for metric, threshold in BASELINES.items():
        score = metrics.get(metric)
        results[metric] = {
            "score":    score,
            "target":   threshold,
            "passed":   score is not None and score >= threshold,
        }
    return results
 
 
def all_passed(baseline_check: dict) -> bool:
    return all(v["passed"] for v in baseline_check.values())
 
 
def run_training(cfg: dict, checkpoint_name: str, no_publish: bool) -> str | None:
    """
    Run train_and_publish.py with the given hyperparameters.
    Returns the checkpoint path on success, or None on failure.
    """
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--lr",           str(cfg["lr"]),
        "--rank",         str(cfg["rank"]),
        "--batch_size",   str(cfg["batch_size"]),
        "--num_steps",    str(cfg["num_steps"]),
        "--checkpoint_name", checkpoint_name,
    ]
    if no_publish:
        cmd.append("--no_publish")
 
    print(f"\n  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)   # stream output live
    if result.returncode != 0:
        print(f"  [ERROR] Training failed (exit code {result.returncode})")
        return None
 
    # Read checkpoint path from the JSON written by train_and_publish.py
    info_path = os.path.join(EVAL_DIR, "checkpoint_info.json")
    if not os.path.exists(info_path):
        print("  [ERROR] checkpoint_info.json not found after training.")
        return None
 
    with open(info_path) as f:
        info = json.load(f)
    return info.get("checkpoint_path")
 
 
def run_evaluation(checkpoint_path: str, limit: int | None) -> dict:
    """
    Run eval_all.py on a checkpoint. Returns the metrics dict.
    """
    cmd = [
        sys.executable, EVAL_SCRIPT,
        "--checkpoint_path", checkpoint_path,
        "--base_model",      BASE_MODEL,
        "--temperature",     str(EVAL_TEMPERATURE),
        "--top_p",           str(EVAL_TOP_P),
    ]
    if limit:
        cmd += ["--limit", str(limit)]
 
    print(f"\n  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  [ERROR] Evaluation failed (exit code {result.returncode})")
        return {}
 
    # eval_all.py writes submission.json
    submission_path = os.path.join(EVAL_DIR, "submission.json")
    if not os.path.exists(submission_path):
        print("  [ERROR] submission.json not found after evaluation.")
        return {}
 
    with open(submission_path) as f:
        submission = json.load(f)
 
    # Metrics are stored flat at the top level of submission.json.
    # Also check nested per-task dicts as a fallback.
    metrics = {}
    for key, val in submission.items():
        if isinstance(val, (int, float)):
            metrics[key] = val
    # Fallback: also unpack any nested task dicts that have a "metrics" sub-key
    for task in ("ifeval", "gsm8k", "humaneval"):
        task_data = submission.get(task, {})
        if isinstance(task_data, dict):
            metrics.update(task_data.get("metrics", {}))
    return metrics
 
 
def print_run_summary(idx: int, total: int, cfg: dict,
                      checkpoint_path: str, metrics: dict,
                      baseline_check: dict, elapsed: float):
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  Run {idx}/{total}  |  elapsed {elapsed:.0f}s")
    print(f"  Config:     {cfg}")
    print(f"  Checkpoint: {checkpoint_path or 'FAILED'}")
    print(f"  Metrics:")
    for metric, info in baseline_check.items():
        score_str = f"{info['score']:.4f}" if info['score'] is not None else "N/A"
        status    = "✓ PASS" if info["passed"] else "✗ FAIL"
        print(f"    {metric:<50} {score_str:>8}  (target ≥ {info['target']:.2f})  {status}")
    if all_passed(baseline_check):
        print("  ★ ALL BASELINES MET ★")
    print(bar)
 
 
def save_results(results: list, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "baselines":    BASELINES,
                "search_space": SEARCH_SPACE,
                "base_model":   BASE_MODEL,
                "runs":         results,
            },
            f,
            indent=2,
        )
    print(f"\n  Results saved → {output_path}")
 
 
def print_final_leaderboard(results: list):
    """Print all runs sorted by a simple average of the three baseline metrics."""
    print("\n" + "=" * 80)
    print("GRID SEARCH LEADERBOARD")
    print("=" * 80)
    header = f"{'Rank':<5} {'IFEval':>9} {'GSM8K':>9} {'HumanEval':>11} {'Avg':>8}  Config"
    print(header)
    print("-" * len(header))
 
    def avg_score(run):
        m = run.get("metrics", {})
        scores = [
            m.get("google/IFEval/prompt_strict_acc", 0.0),
            m.get("openai/gsm8k/accuracy", 0.0),
            m.get("openai/openai_humaneval/accuracy", 0.0),
        ]
        return sum(scores) / len(scores)
 
    sorted_runs = sorted(
        [r for r in results if r.get("metrics")],
        key=avg_score,
        reverse=True,
    )
 
    for rank, run in enumerate(sorted_runs, 1):
        m   = run["metrics"]
        cfg = run["config"]
        ie  = m.get("google/IFEval/prompt_strict_acc", float("nan"))
        gsm = m.get("openai/gsm8k/accuracy", float("nan"))
        hev = m.get("openai/openai_humaneval/accuracy", float("nan"))
        av  = avg_score(run)
        met = "★" if run.get("all_passed") else " "
        cfg_str = json.dumps(cfg, sort_keys=True)
        print(f"{rank:<5} {ie:>9.4f} {gsm:>9.4f} {hev:>11.4f} {av:>8.4f} {met} {cfg_str}")
 
    print("=" * 80)
 
    if sorted_runs:
        best = sorted_runs[0]
        print(f"\nBest config (by avg score):")
        print(f"  {json.dumps(best['config'], sort_keys=True, indent=2)}")
        print(f"  Checkpoint: {best.get('checkpoint_path', 'N/A')}")
 
 
# Main function
def main():
    parser = argparse.ArgumentParser(description="Grid search over training hyperparameters")
    parser.add_argument("--limit",      type=int,  default=None,
                        help="Max samples per eval task (use small number for fast iteration)")
    parser.add_argument("--no_publish", action="store_true",
                        help="Skip publishing checkpoints (saves time)")
    parser.add_argument("--output",     type=str,
                        default=os.path.join(EVAL_DIR, "grid_search_results.json"),
                        help="Where to write results JSON")
    parser.add_argument("--resume",     action="store_true",
                        help="Skip configs already present in the output file")
    parser.add_argument("--stop_on_pass", action="store_true", default=True,
                        help="Stop the search as soon as a config passes all baselines")
    args = parser.parse_args()
 
    configs = all_configs()
    total   = len(configs)
    print(f"\nGrid search: {total} configs over {BASE_MODEL}")
    print(f"Search space: {json.dumps(SEARCH_SPACE, indent=2)}")
    print(f"Baselines:    {BASELINES}")
    print(f"Eval limit:   {args.limit or 'full'}")
    print(f"Output:       {args.output}")
 
    # Optionally resume from a previous run
    completed_keys: set[str] = set()
    results: list = []
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            prev = json.load(f)
        results = prev.get("runs", [])
        completed_keys = {config_key(r["config"]) for r in results}
        print(f"\nResuming — {len(completed_keys)} configs already done.")
 
    overall_start = time.perf_counter()
    found_passing = False
 
    for idx, cfg in enumerate(configs, 1):
        key = config_key(cfg)
        if key in completed_keys:
            print(f"\n[{idx}/{total}] SKIP (already evaluated): {cfg}")
            continue
 
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_name = f"gs_{timestamp}_lr{cfg['lr']}_r{cfg['rank']}_bs{cfg['batch_size']}_s{cfg['num_steps']}"
 
        print(f"\n{'#'*70}")
        print(f"# Run {idx}/{total}: {cfg}")
        print(f"{'#'*70}")
 
        run_start = time.perf_counter()
 
        # Train:
        checkpoint_path = run_training(cfg, ckpt_name, args.no_publish)
        if checkpoint_path is None:
            record = {
                "run":             idx,
                "config":          cfg,
                "checkpoint_name": ckpt_name,
                "checkpoint_path": None,
                "metrics":         {},
                "baseline_check":  {},
                "all_passed":      False,
                "error":           "training_failed",
                "timestamp":       timestamp,
            }
            results.append(record)
            save_results(results, args.output)
            continue
 
        # Evaluate:
        metrics = run_evaluation(checkpoint_path, args.limit)
        baseline_check = passes_baselines(metrics)
        passed = all_passed(baseline_check)
 
        elapsed = time.perf_counter() - run_start
        print_run_summary(idx, total, cfg, checkpoint_path, metrics,
                          baseline_check, elapsed)
 
        record = {
            "run":             idx,
            "config":          cfg,
            "checkpoint_name": ckpt_name,
            "checkpoint_path": checkpoint_path,
            "metrics":         metrics,
            "baseline_check":  baseline_check,
            "all_passed":      passed,
            "elapsed_seconds": round(elapsed, 1),
            "timestamp":       timestamp,
        }
        results.append(record)
        save_results(results, args.output)
 
        if passed and args.stop_on_pass:
            print(f"\n★ STOPPING EARLY — all baselines met with config: {cfg}")
            print(f"  Checkpoint: {checkpoint_path}")
            found_passing = True
            break
 
    total_elapsed = time.perf_counter() - overall_start
    print(f"\nGrid search complete in {total_elapsed/60:.1f} minutes.")
    if not found_passing:
        print("No single config met all baselines. See leaderboard below.")
 
    print_final_leaderboard(results)
 
 
if __name__ == "__main__":
    main()
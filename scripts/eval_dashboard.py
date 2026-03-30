#!/usr/bin/env python3
"""Evaluation dashboard — compares metrics across evaluation runs.

Usage:
    python scripts/eval_dashboard.py                    # Compare all CSVs in data/
    python scripts/eval_dashboard.py data/eval_v1.csv data/eval_v2.csv  # Compare specific files
"""
import csv
import os
import re
import sys
from pathlib import Path
from typing import List, Dict


def load_evaluation(csv_path: str) -> List[Dict]:
    """Load evaluation results from a CSV file."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def compute_metrics(results: List[Dict]) -> Dict:
    """Compute summary metrics from evaluation results."""
    valid = [r for r in results if r.get("score") and int(r["score"]) != -1]
    if not valid:
        return {"total": len(results), "valid": 0}

    scores = [int(r["score"]) for r in valid]
    latencies = [float(r["latency_sec"]) for r in results if r.get("latency_sec")]

    metrics = {
        "total": len(results),
        "valid": len(valid),
        "avg_score": sum(scores) / len(scores),
        "median_score": sorted(scores)[len(scores) // 2],
        "perfect_10s": sum(1 for s in scores if s == 10),
        "zero_scores": sum(1 for s in scores if s == 0),
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
    }

    # Citation metrics (if available in CSV)
    if "num_citations" in valid[0]:
        citations = [int(r.get("num_citations", 0)) for r in valid]
        metrics["avg_citations"] = sum(citations) / len(citations)
        metrics["citation_presence"] = sum(1 for c in citations if c > 0) / len(valid) * 100

    if "source_pmid_retrieved" in valid[0]:
        recalls = [r.get("source_pmid_retrieved", "False") for r in valid]
        metrics["retrieval_recall"] = sum(1 for r in recalls if r == "True") / len(valid) * 100

    # Score distribution
    buckets = {"0": 0, "1-4": 0, "5-7": 0, "8-9": 0, "10": 0}
    for s in scores:
        if s == 0: buckets["0"] += 1
        elif s <= 4: buckets["1-4"] += 1
        elif s <= 7: buckets["5-7"] += 1
        elif s <= 9: buckets["8-9"] += 1
        else: buckets["10"] += 1
    metrics["score_distribution"] = buckets

    return metrics


def format_metric(value, fmt=".1f", suffix=""):
    """Format a metric value, handling None."""
    if value is None:
        return "N/A"
    return f"{value:{fmt}}{suffix}"


def print_single_report(name: str, metrics: Dict):
    """Print a summary report for a single evaluation run."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Questions: {metrics['total']} total, {metrics['valid']} scored")
    print(f"  Avg Score: {format_metric(metrics.get('avg_score'), '.2f')} / 10.00")
    print(f"  Median:    {metrics.get('median_score', 'N/A')}")
    print(f"  Perfect:   {metrics.get('perfect_10s', 0)}  |  Zeros: {metrics.get('zero_scores', 0)}")
    print(f"  Latency:   {format_metric(metrics.get('avg_latency'), '.2f', 's')}")

    if "avg_citations" in metrics:
        print(f"  Citations: {format_metric(metrics.get('avg_citations'), '.1f')} avg/answer")
        print(f"  Citation Rate: {format_metric(metrics.get('citation_presence'), '.1f', '%')}")

    if "retrieval_recall" in metrics:
        print(f"  Retrieval Recall: {format_metric(metrics.get('retrieval_recall'), '.1f', '%')}")

    dist = metrics.get("score_distribution", {})
    if dist:
        print(f"\n  Score Distribution:")
        max_count = max(dist.values()) if dist.values() else 1
        for bucket, count in dist.items():
            bar_len = int(count / max(max_count, 1) * 20)
            bar = "█" * bar_len
            print(f"    {bucket:>4}: {count:>3} {bar}")


def print_comparison(runs: List[tuple]):
    """Print a side-by-side comparison of multiple runs."""
    if len(runs) < 2:
        return

    print(f"\n{'='*60}")
    print("  TREND COMPARISON")
    print(f"{'='*60}")

    header = f"  {'Metric':<25}"
    for name, _ in runs:
        short = Path(name).stem[:15]
        header += f" {short:>15}"
    print(header)
    print(f"  {'-'*25}" + (" " + "-"*15) * len(runs))

    comparisons = [
        ("Avg Score", "avg_score", ".2f"),
        ("Median Score", "median_score", "d"),
        ("Perfect 10s", "perfect_10s", "d"),
        ("Zero Scores", "zero_scores", "d"),
        ("Avg Latency (s)", "avg_latency", ".2f"),
        ("Avg Citations", "avg_citations", ".1f"),
        ("Citation Rate (%)", "citation_presence", ".1f"),
        ("Retrieval Recall (%)", "retrieval_recall", ".1f"),
    ]

    for label, key, fmt in comparisons:
        row = f"  {label:<25}"
        values = []
        for _, metrics in runs:
            val = metrics.get(key)
            if val is not None:
                row += f" {val:>15{fmt}}"
                values.append(val)
            else:
                row += f" {'N/A':>15}"

        # Add trend arrow if we have at least 2 values
        if len(values) >= 2:
            diff = values[-1] - values[-2]
            if key in ("zero_scores", "avg_latency"):  # lower is better
                arrow = " ↓ better" if diff < 0 else " ↑ worse" if diff > 0 else ""
            else:  # higher is better
                arrow = " ↑ better" if diff > 0 else " ↓ worse" if diff < 0 else ""
            row += arrow

        print(row)


def main():
    if len(sys.argv) > 1:
        csv_files = sys.argv[1:]
    else:
        data_dir = Path("data")
        csv_files = sorted(data_dir.glob("evaluation_results*.csv"), key=os.path.getmtime)
        if not csv_files:
            print("No evaluation CSV files found in data/. Run an evaluation first:")
            print("  python scripts/run_evaluation.py evaluate")
            sys.exit(1)
        csv_files = [str(f) for f in csv_files]

    runs = []
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"File not found: {csv_file}")
            continue
        results = load_evaluation(csv_file)
        metrics = compute_metrics(results)
        runs.append((csv_file, metrics))
        print_single_report(csv_file, metrics)

    if len(runs) >= 2:
        print_comparison(runs)

    print(f"\n  Processed {len(runs)} evaluation run(s).")


if __name__ == "__main__":
    main()

"""
Aggregate evaluation results across all experiments.

Reads JSON result files from offline_data/results/ and produces:
  - Summary table (printed + LaTeX)
  - wandb summary (if available)
  - Bar charts (SR, SPL by degradation preset)

Usage:
    python offline/aggregate.py
    python offline/aggregate.py --results-dir offline_data/results --output report.md
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from offline.configs import RESULTS_DIR


def load_all_results(results_dir):
    """Load all JSON result files from directory."""
    files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    results = []
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        data["_filename"] = os.path.basename(fp)
        results.append(data)
    return results


def build_table(results):
    """Build a summary table from results."""
    rows = []
    for r in results:
        s = r.get("summary", {})
        rows.append({
            "Tag": s.get("tag", ""),
            "Method": s.get("method", ""),
            "SR": s.get("success_rate", 0),
            "SR±": s.get("success_std", 0),
            "SPL": s.get("spl", 0),
            "SPL±": s.get("spl_std", 0),
            "SoftSPL": s.get("soft_spl", 0),
            "AvgDTG": s.get("avg_distance_to_goal", 0),
            "AvgSteps": s.get("avg_steps", 0),
            "File": r.get("_filename", ""),
        })

    # Sort by tag, then method
    rows.sort(key=lambda x: (x["Tag"], x["Method"]))
    return rows


def print_table(rows):
    """Print formatted ASCII table."""
    if not rows:
        print("No results found.")
        return

    header = f"{'Tag':<20} {'Method':<8} {'SR':>7} {'SPL':>7} {'SoftSPL':>8} {'DTG':>6} {'Steps':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['Tag']:<20} {r['Method']:<8} "
              f"{r['SR']:>6.3f}  {r['SPL']:>6.3f}  "
              f"{r['SoftSPL']:>7.3f}  {r['AvgDTG']:>5.2f}  "
              f"{r['AvgSteps']:>5.1f}")


def to_latex(rows):
    """Generate LaTeX table."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Offline RL results under data degradation.}",
        r"\label{tab:offline-results}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Degradation & Method & SR $\uparrow$ & SPL $\uparrow$ & SoftSPL $\uparrow$ & DTG $\downarrow$ \\",
        r"\midrule",
    ]

    prev_tag = None
    for r in rows:
        tag = r["Tag"] or "Full data"
        if tag != prev_tag and prev_tag is not None:
            lines.append(r"\midrule")
        prev_tag = tag

        lines.append(
            f"  {tag} & {r['Method']} & "
            f"{r['SR']:.3f} $\\pm$ {r['SR±']:.3f} & "
            f"{r['SPL']:.3f} $\\pm$ {r['SPL±']:.3f} & "
            f"{r['SoftSPL']:.3f} & "
            f"{r['AvgDTG']:.2f} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def to_markdown(rows):
    """Generate Markdown table."""
    lines = [
        "# Offline RL Results",
        "",
        "| Degradation | Method | SR ↑ | SPL ↑ | SoftSPL ↑ | DTG ↓ | Steps |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        tag = r["Tag"] or "Full data"
        lines.append(
            f"| {tag} | {r['Method']} | "
            f"{r['SR']:.3f} ± {r['SR±']:.3f} | "
            f"{r['SPL']:.3f} ± {r['SPL±']:.3f} | "
            f"{r['SoftSPL']:.3f} | "
            f"{r['AvgDTG']:.2f} | "
            f"{r['AvgSteps']:.0f} |"
        )
    return "\n".join(lines)


def generate_plots(rows, output_dir):
    """Generate bar charts comparing methods across degradation presets."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Group by tag
    tags = sorted(set(r["Tag"] for r in rows))
    methods = sorted(set(r["Method"] for r in rows))
    colors = {"iql": "#2196F3", "bc": "#FF9800", "expert": "#4CAF50",
              "random": "#9E9E9E"}

    for metric, label in [("SR", "Success Rate"), ("SPL", "SPL")]:
        fig, ax = plt.subplots(figsize=(max(len(tags) * 1.5, 6), 4))
        x = np.arange(len(tags))
        width = 0.8 / max(len(methods), 1)

        for i, method in enumerate(methods):
            vals = []
            errs = []
            for tag in tags:
                matching = [r for r in rows
                           if r["Tag"] == tag and r["Method"] == method]
                if matching:
                    vals.append(matching[0][metric])
                    errs.append(matching[0].get(f"{metric}±", 0))
                else:
                    vals.append(0)
                    errs.append(0)

            ax.bar(x + i * width, vals, width, yerr=errs,
                   label=method.upper(), color=colors.get(method, "#607D8B"),
                   alpha=0.85, capsize=3)

        ax.set_xlabel("Degradation Preset")
        ax.set_ylabel(label)
        ax.set_title(f"{label} by Degradation and Method")
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels([t or "Full" for t in tags], rotation=30, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"{metric.lower()}_comparison.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate results")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--output", type=str, default=None,
                       help="Output markdown report path")
    parser.add_argument("--plots-dir", type=str, default=None,
                       help="Directory for plot images")
    args = parser.parse_args()

    print(f"Loading results from: {args.results_dir}")
    results = load_all_results(args.results_dir)
    print(f"Found {len(results)} result files\n")

    if not results:
        print("No results found. Run evaluate.py first.")
        return

    rows = build_table(results)

    # Print ASCII table
    print_table(rows)

    # Generate LaTeX
    latex = to_latex(rows)
    latex_path = os.path.join(args.results_dir, "results_table.tex")
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"\nLaTeX table saved to: {latex_path}")

    # Generate Markdown
    md = to_markdown(rows)
    md_path = args.output or os.path.join(args.results_dir, "results.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Markdown report saved to: {md_path}")

    # Generate plots
    plots_dir = args.plots_dir or os.path.join(args.results_dir, "plots")
    generate_plots(rows, plots_dir)

    # wandb summary
    try:
        import wandb
        if wandb.run is not None:
            for r in rows:
                wandb.run.summary[f"{r['Tag']}_{r['Method']}_SR"] = r["SR"]
                wandb.run.summary[f"{r['Tag']}_{r['Method']}_SPL"] = r["SPL"]
    except (ImportError, Exception):
        pass

    print("\nDone.")


if __name__ == "__main__":
    main()

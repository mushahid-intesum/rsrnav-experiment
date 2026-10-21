"""
Aggregate experiment results from individual JSON files into summary tables.
Usage: python -m eval.aggregate_results
"""
import os, sys, json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ===================== CONFIG =====================
RESULTS_DIR = "results"
# ==================================================


def load_all_results(results_dir):
    results = {}
    for f in sorted(Path(results_dir).glob("*.json")):
        with open(f) as fp:
            results[f.stem] = json.load(fp)
    return results


def print_table(title, headers, rows, col_widths=None):
    if col_widths is None:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
                      for i, h in enumerate(headers)]
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_row = "|" + "|".join(f" {h:<{w}} " for h, w in zip(headers, col_widths)) + "|"
    print(f"\n{title}")
    print(sep)
    print(header_row)
    print(sep)
    for row in rows:
        cells = "|".join(f" {str(v):<{w}} " for v, w in zip(row, col_widths))
        print(f"|{cells}|")
    print(sep)


def aggregate(results_dir):
    results = load_all_results(results_dir)
    if not results:
        print(f"No results found in {results_dir}/")
        return
    print(f"\nLoaded {len(results)} result files from {results_dir}/")

    # Experiment 1
    exp1 = {k: v for k, v in results.items() if k.startswith("exp1_")}
    if exp1:
        rows = []
        for name, data in sorted(exp1.items()):
            m = data.get("metrics", {})
            rows.append([data.get("correlation_type", "?"),
                         f"{m.get('SR',0):.3f}", f"{m.get('SPL',0):.3f}",
                         f"{m.get('SoftSPL',0):.3f}", f"{m.get('DTS',0):.2f}"])
        print_table("Experiment 1: Does Correlation Help Offline? (IQL, Ideal Data)",
                    ["Model", "SR", "SPL", "SoftSPL", "DTS(m)"], rows)

    # Experiment 2
    exp2 = {k: v for k, v in results.items() if k.startswith("exp2_")}
    if exp2:
        for model_key, label in [("da_", "RSRNav-DA"), ("semantic_", "Semantic Baseline")]:
            rows = []
            for name, data in sorted(exp2.items()):
                if model_key not in name:
                    continue
                m = data.get("metrics", {})
                vol = name.split("vol")[-1] if "vol" in name else "?"
                rows.append([vol, f"{m.get('SR',0):.3f}", f"{m.get('SPL',0):.3f}"])
            if rows:
                print_table(f"Experiment 2: Volume Curve — {label}", ["Episodes", "SR", "SPL"], rows)

    # Experiment 3
    exp3 = {k: v for k, v in results.items() if k.startswith("exp3_")}
    if exp3:
        rows = []
        for name, data in sorted(exp3.items()):
            m = data.get("metrics", {})
            qual = name.replace("exp3_", "").replace("da_", "DA-").replace("semantic_", "Sem-")
            rows.append([qual, f"{m.get('SR',0):.3f}", f"{m.get('SPL',0):.3f}"])
        print_table("Experiment 3: Demonstrator Quality Sensitivity", ["Config", "SR", "SPL"], rows)

    # Experiment 4
    exp4 = {k: v for k, v in results.items() if k.startswith("exp4_")}
    if exp4:
        presets = ["mild", "moderate", "severe"]
        models = ["da", "minimalist", "semantic"]
        model_labels = {"da": "RSRNav-DA", "minimalist": "RSRNav-Min", "semantic": "Semantic"}
        rows = []
        for preset in presets:
            row = [preset.capitalize()]
            for model in models:
                key = f"exp4_{model}_{preset}"
                if key in exp4:
                    m = exp4[key].get("metrics", {})
                    row.append(f"{m.get('SR',0):.3f}/{m.get('SPL',0):.3f}")
                else:
                    row.append("—")
            rows.append(row)
        print_table("Experiment 4: Combined Realistic Degradation (SR/SPL)",
                    ["Preset"] + [model_labels[m] for m in models], rows)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Best SPL per experiment")
    print("=" * 60)
    for prefix in ["exp1", "exp2", "exp3", "exp4"]:
        group = {k: v for k, v in results.items() if k.startswith(prefix)}
        if group:
            best_name = max(group, key=lambda k: group[k].get("metrics", {}).get("SPL", 0))
            best_spl = group[best_name]["metrics"]["SPL"]
            best_sr = group[best_name]["metrics"]["SR"]
            corr = group[best_name].get("correlation_type", "?")
            print(f"  {prefix}: {best_name} ({corr}) — SR={best_sr:.3f} SPL={best_spl:.3f}")


if __name__ == "__main__":
    aggregate(RESULTS_DIR)

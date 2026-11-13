import os
import glob
import csv
import statistics

try:
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("matplotlib not installed, skipping charts (pip install matplotlib)")

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("scipy not installed, skipping stats tests (pip install scipy)")


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
LAST_N_ROUNDS = 10


def load_all_runs():
    results = []
    files = glob.glob(os.path.join(OUTPUT_DIR, "*.csv"))

    if not files:
        print(f"no CSV files found in {OUTPUT_DIR}, run run_experiment.py first")
        return results

    for filepath in files:
        filename = os.path.basename(filepath)
        parts = filename.replace(".csv", "").split("_")
        prompt_type, language, run_id = parts[0], parts[1], parts[2].replace("run", "")

        with open(filepath, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue

        last_rows = rows[-LAST_N_ROUNDS:]
        prices = []
        for row in last_rows:
            prices.append(float(row["price_A"]))
            prices.append(float(row["price_B"]))

        results.append({
            "prompt_type": prompt_type,
            "language": language,
            "run_id": run_id,
            "avg_price": sum(prices) / len(prices),
        })

    return results


def group_by(results, keys):
    groups = {}
    for r in results:
        key = tuple(r[k] for k in keys)
        groups.setdefault(key, []).append(r["avg_price"])
    return groups


def print_summary_table(results):
    print("\naverage price by (prompt, language):")
    groups = group_by(results, ["prompt_type", "language"])
    print(f"{'prompt':<8}{'lang':<8}{'n runs':<10}{'avg price':<12}{'std dev':<10}")
    for (prompt_type, language), prices in sorted(groups.items()):
        mean = statistics.mean(prices)
        std = statistics.stdev(prices) if len(prices) > 1 else 0.0
        print(f"{prompt_type:<8}{language:<8}{len(prices):<10}{mean:<12.4f}{std:<10.4f}")


def run_statistical_tests(results):
    if not HAS_SCIPY:
        return

    print("\nP1 vs P2 (per language):")
    for language in ["en", "uk", "tr"]:
        p1 = [r["avg_price"] for r in results if r["prompt_type"] == "P1" and r["language"] == language]
        p2 = [r["avg_price"] for r in results if r["prompt_type"] == "P2" and r["language"] == language]
        if len(p1) >= 2 and len(p2) >= 2:
            t, p = scipy_stats.ttest_ind(p1, p2)
            flag = "  <-- significant" if p < 0.05 else ""
            print(f"  {language}: P1={statistics.mean(p1):.3f} P2={statistics.mean(p2):.3f} p={p:.5f}{flag}")
        else:
            print(f"  {language}: not enough data")

    print("\nlanguage comparison (per prompt, ANOVA):")
    for prompt_type in ["P1", "P2"]:
        groups, labels = [], []
        for language in ["en", "uk", "tr"]:
            prices = [r["avg_price"] for r in results if r["prompt_type"] == prompt_type and r["language"] == language]
            if len(prices) >= 2:
                groups.append(prices)
                labels.append(language)
        if len(groups) >= 2:
            f, p = scipy_stats.f_oneway(*groups)
            flag = "  <-- significant" if p < 0.05 else ""
            print(f"  {prompt_type} ({labels}): p={p:.5f}{flag}")
        else:
            print(f"  {prompt_type}: not enough data")


def make_plot(results):
    if not HAS_PLOTTING:
        return

    groups = group_by(results, ["prompt_type", "language"])
    labels, means = [], []
    for (prompt_type, language), prices in sorted(groups.items()):
        labels.append(f"{prompt_type}\n{language}")
        means.append(statistics.mean(prices))

    plt.figure(figsize=(10, 6))
    plt.bar(labels, means, color="steelblue")
    plt.ylabel("average price (final rounds)")
    plt.title("average price by prompt x language")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "summary_chart.png")
    plt.savefig(out_path)
    print(f"\nchart saved: {out_path}")


def main():
    results = load_all_runs()
    if not results:
        return
    print(f"loaded {len(results)} runs")
    print_summary_table(results)
    run_statistical_tests(results)
    make_plot(results)


if __name__ == "__main__":
    main()

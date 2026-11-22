"""Generate publication-style figures from the checked-in experiment summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "docs" / "figures"

METHODS = ["bm25", "dense", "hybrid", "hybrid_reranked"]
METHOD_LABELS = {
    "bm25": "BM25",
    "dense": "Dense",
    "hybrid": "Hybrid",
    "hybrid_reranked": "Hybrid + reranker",
}
STAT_KEYS = {"recall": "evidence_recall", "mrr": "mrr", "hit": "hit_rate"}
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUTPUT / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def method_stats(summary: dict, method: str, metric: str) -> dict:
    return summary["methods"][f"{method} (token-aware)"][STAT_KEYS[metric]]


def plot_main_metrics(summary: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, metric, title in zip(
        axes,
        ("recall", "mrr"),
        ("Evidence Recall@5", "MRR@5"),
        strict=True,
    ):
        for i, (method, color) in enumerate(zip(METHODS, COLORS, strict=True)):
            result = method_stats(summary, method, metric)
            mean = result["mean"]
            ax.errorbar(
                i,
                mean,
                yerr=[[mean - result["ci_low"]], [result["ci_high"] - mean]],
                fmt="o",
                color=color,
                capsize=4,
                markersize=7,
            )
        ax.set_title(title)
        ax.set_xticks(range(len(METHODS)), [METHOD_LABELS[m] for m in METHODS], rotation=20)
        ax.set_ylim(0, 0.78)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("Score")
    fig.suptitle("Held-out retrieval results (95% paper-cluster bootstrap CI)")
    fig.tight_layout()
    save_figure(fig, "retrieval_metrics")


def plot_paired_effects(summary: dict) -> None:
    pairs = [
        "bm25 (token-aware) -> hybrid (token-aware)",
        "dense (token-aware) -> hybrid (token-aware)",
        "hybrid (token-aware) -> hybrid_reranked (token-aware)",
    ]
    pair_labels = ["Hybrid − BM25", "Hybrid − Dense", "Reranker − Hybrid"]
    keys = ["evidence_recall", "mrr", "hit_rate", "complete_evidence_coverage"]
    titles = ["Recall", "MRR", "Hit rate", "Complete coverage"]
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False)
    for ax, key, title in zip(axes.flat, keys, titles, strict=True):
        for i, (pair, label, color) in enumerate(zip(pairs, pair_labels, colors, strict=True)):
            result = summary["paired_differences_between_methods"][pair][key]
            mean = result["mean_difference"]
            ax.errorbar(
                mean,
                i,
                xerr=[[mean - result["ci_low"]], [result["ci_high"] - mean]],
                fmt="o",
                capsize=4,
                color=color,
            )
        ax.axvline(0, color="#555555", linewidth=1, linestyle="--")
        ax.set_yticks(range(len(pair_labels)), pair_labels)
        ax.set_title(title)
        ax.set_xlabel("Paired score difference (95% CI)")
        ax.grid(axis="x", alpha=0.25)
    n_clusters = summary["methods"][f"{METHODS[0]} (token-aware)"]["evidence_recall"]["n_clusters"]
    fig.suptitle(f"Paired differences on the {n_clusters}-paper evaluation sample")
    fig.tight_layout()
    save_figure(fig, "paired_effects")


def plot_latency_tradeoff(summary: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    latency = summary["latency_summary"]
    for method, color in zip(METHODS, COLORS, strict=True):
        key = f"{method} (token-aware)"
        x = latency[key]["mean_ms"]
        y = method_stats(summary, method, "recall")["mean"]
        ax.scatter(x, y, s=85, color=color, label=METHOD_LABELS[method], zorder=3)
        ax.annotate(METHOD_LABELS[method], (x, y), xytext=(7, 6), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Mean retrieval latency per question (ms, log scale)")
    ax.set_ylabel("Evidence Recall@5")
    ax.set_title("Retrieval quality and latency trade-off (local CPU)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "latency_tradeoff")


def load_ablation_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def aggregate(rows: list[dict[str, str]], study: str, method: str, parameter: str) -> dict:
    selected = [
        row
        for row in rows
        if row["study"] == study and row["method"] == method and row["parameter"] == parameter
    ]
    if not selected:
        raise ValueError(f"No ablation records for {study}/{method}/{parameter}")
    return {
        metric: float(np.mean([float(row[metric]) for row in selected]))
        for metric in ("evidence_recall", "mrr", "hit_rate", "complete_evidence_coverage")
    }


def plot_ablations(rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    k_values = ["3", "5", "10"]
    for method, color in zip(METHODS, COLORS, strict=True):
        values = [aggregate(rows, "top_k", method, k) for k in k_values]
        axes[0].plot(
            k_values,
            [v["evidence_recall"] for v in values],
            marker="o",
            color=color,
            label=METHOD_LABELS[method],
        )
        axes[1].plot(
            k_values,
            [v["mrr"] for v in values],
            marker="o",
            color=color,
            label=METHOD_LABELS[method],
        )
    axes[0].set_title("Evidence Recall by final K")
    axes[1].set_title("MRR by final K")
    for ax in axes[:2]:
        ax.set_xlabel("Final context K (development subset)")
        ax.set_xticks(range(3), k_values)
        ax.set_ylim(0, 0.85)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Score")
    axes[1].legend(frameon=False, fontsize=8)

    chunk_methods = ["bm25", "dense"]
    chunk_labels = ["BM25", "Dense"]
    chunk_modes = [("paragraph", "Original paragraph"), ("token-aware", "Token-aware")]
    x = np.arange(len(chunk_methods))
    width = 0.34
    for j, (mode, label) in enumerate(chunk_modes):
        values = [
            aggregate(rows, "chunking", method, mode)["evidence_recall"] for method in chunk_methods
        ]
        axes[2].bar(
            x + (j - 0.5) * width, values, width, label=label, color=("#9ECAE9", "#3182BD")[j]
        )
    axes[2].set_title("Chunking ablation (Recall@5)")
    axes[2].set_xticks(x, chunk_labels)
    axes[2].set_ylim(0, 0.85)
    axes[2].set_ylabel("Evidence Recall@5")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle("Development-subset ablations (36 eligible questions)")
    fig.tight_layout()
    save_figure(fig, "ablations")


def plot_question_type(summary: dict) -> None:
    categories = ["extractive", "free_form", "yes_no", "mixed_answerability"]
    labels = ["Extractive", "Free-form", "Yes/no", "Mixed answerability"]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    offsets = np.linspace(-0.27, 0.27, len(METHODS))
    for method_i, (method, color) in enumerate(zip(METHODS, COLORS, strict=True)):
        xs, means, lows, highs = [], [], [], []
        for cat_i, category in enumerate(categories):
            values = summary["by_question_type_nonexclusive"][category][f"{method} (token-aware)"][
                "evidence_recall"
            ]
            mean = values["mean"]
            xs.append(cat_i + offsets[method_i])
            means.append(mean)
            lows.append(mean - values["ci_low"])
            highs.append(values["ci_high"] - mean)
        ax.errorbar(
            xs,
            means,
            yerr=[lows, highs],
            fmt="o",
            capsize=3,
            color=color,
            label=METHOD_LABELS[method],
        )
    ns = [
        summary["by_question_type_nonexclusive"][category][f"{METHODS[0]} (token-aware)"][
            "evidence_recall"
        ]["n_questions"]
        for category in categories
    ]
    ax.set_xticks(
        range(len(categories)), [f"{name}\n(n={n})" for name, n in zip(labels, ns, strict=True)]
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Evidence Recall@5")
    ax.set_title("Recall by non-exclusive question annotation type (95% paper-cluster CI)")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "question_type_recall")


def plot_generation(provider: dict, local: dict) -> None:
    methods = ["closed_book", *METHODS[1:]]
    labels = ["Closed book", *(METHOD_LABELS[method] for method in methods[1:])]
    provider_ids = {
        row["question_id"] for row in provider["per_question_metrics"]["closed_book"]
    }
    local_ids = {row["question_id"] for row in local["per_question_metrics"]["closed_book"]}
    if provider_ids != local_ids:
        raise ValueError("Generation comparisons must use the same question IDs")

    fig, ax = plt.subplots(figsize=(8.8, 4.6), layout="constrained")
    positions = np.arange(len(methods))
    for comparison, label, offset, color in (
        (provider, "GPT-6 Luna (API)", -0.18, COLORS[0]),
        (local, "Qwen2.5-1.5B-Instruct (local)", 0.18, COLORS[2]),
    ):
        scores = [
            comparison["method_metrics"][method]["answer_metrics"]["token_f1_macro"]
            for method in methods
        ]
        bars = ax.bar(positions + offset, scores, width=0.32, color=color, label=label)
        ax.bar_label(bars, fmt="%.3f", padding=4)
    scored = provider["method_metrics"]["closed_book"]["counts"]["answer_scored_count"]
    ax.set_title(f"Answer token F1 on {scored} scored questions\n"
                 f"from the {len(provider_ids)} questions with valid results in every arm")
    ax.set_ylabel("Mean answer token F1")
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, "generation_answer_f1")


def main() -> None:
    global OUTPUT
    parser = argparse.ArgumentParser(description="Generate figures from saved experiment results.")
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=RESULTS / "experiments" / "retrieval_statistical_summary.json",
    )
    parser.add_argument("--ablation-file", type=Path, default=RESULTS / "ablation_results.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--generation-comparisons", type=Path, nargs=2, metavar=("PROVIDER", "LOCAL"))
    args = parser.parse_args()

    OUTPUT = args.output_dir
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    if args.generation_comparisons:
        provider, local = map(read_json, args.generation_comparisons)
        plot_generation(provider, local)
        print(f"Generated the answer F1 figure (PNG and PDF) in {OUTPUT}")
        return

    summary = read_json(args.summary_file)
    plot_main_metrics(summary)
    plot_paired_effects(summary)
    plot_latency_tradeoff(summary)
    if not args.skip_ablations:
        plot_ablations(load_ablation_rows(args.ablation_file))
    plot_question_type(summary)
    figure_count = 4 if args.skip_ablations else 5
    print(f"Generated {figure_count} figures (PNG and PDF) in {OUTPUT}")


if __name__ == "__main__":
    main()

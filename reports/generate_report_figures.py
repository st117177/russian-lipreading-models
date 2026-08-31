#!/usr/bin/env python3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "blue": "#2463A8",
    "orange": "#D9772A",
    "green": "#2D8A56",
    "gray": "#707782",
    "red": "#B84A4A",
}


def finish(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def dataset_growth() -> None:
    labels = ["Семестр 3\nдемо", "Dataset v1", "Dataset v2"]
    clips = [182, 955, 1355]
    speakers = [5, 6, 8]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    bars = axes[0].bar(labels, clips, color=[COLORS["gray"], COLORS["blue"], COLORS["green"]])
    axes[0].set_title("Рост числа клипов")
    axes[0].set_ylabel("Количество клипов")
    axes[0].bar_label(bars, padding=3)

    bars = axes[1].bar(labels, speakers, color=[COLORS["gray"], COLORS["blue"], COLORS["green"]])
    axes[1].set_title("Рост числа спикеров")
    axes[1].set_ylabel("Количество спикеров")
    axes[1].bar_label(bars, padding=3)
    axes[1].set_ylim(0, 9)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    finish(fig, "dataset_growth.png")


def dataset_split() -> None:
    splits = ["Train", "Validation", "Test"]
    clips = [1100, 105, 150]
    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(splits, clips, color=colors, width=0.62)
    ax.bar_label(bars, labels=["1100 клипов\n5 спикеров", "105 клипов\n1 спикер", "150 клипов\n2 спикера"], padding=4)
    ax.set_title("Speaker-based split dataset v2")
    ax.set_ylabel("Количество клипов")
    ax.set_ylim(0, 1250)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    finish(fig, "dataset_v2_split.png")


def v1_v2_comparison() -> None:
    metrics = ["Accuracy", "Macro-F1", "Balanced\naccuracy"]
    v1 = np.array([0.1778, 0.1311, 0.1471])
    v2 = np.array([0.1587, 0.1416, 0.1715])
    v1_std = np.array([0.0574, 0.0312, 0.0258])
    v2_std = np.array([0.0440, 0.0458, 0.0104])
    x = np.arange(len(metrics))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.bar(x - width / 2, v1, width, yerr=v1_std, capsize=4, label="Dataset v1", color=COLORS["gray"])
    ax.bar(x + width / 2, v2, width, yerr=v2_std, capsize=4, label="Dataset v2", color=COLORS["blue"])
    ax.set_xticks(x, metrics)
    ax.set_ylabel("Validation metric")
    ax.set_ylim(0, 0.24)
    ax.set_title("Влияние расширения train-данных, 3 seed")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    finish(fig, "dataset_v1_v2_metrics.png")


def model_comparison() -> None:
    models = [
        "ImageNet\nResNet18 + BiGRU",
        "ImageNet\nResNet18 + TCN",
        "LRW + BiGRU\nold crop",
        "LRW + BiGRU\nlandmark v3",
    ]
    macro_f1 = [0.1416, 0.1182, 0.2736, 0.5359]
    std = [0.0458, 0.0287, 0.0290, 0.0139]
    colors = [COLORS["gray"], COLORS["orange"], COLORS["blue"], COLORS["green"]]

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    bars = ax.bar(models, macro_f1, yerr=std, capsize=5, color=colors, width=0.62)
    ax.bar_label(bars, labels=[f"{value:.4f}" for value in macro_f1], padding=7)
    ax.set_title("Сравнение моделей и preprocessing на validation")
    ax.set_ylabel("Macro-F1, mean +/- std по 3 seed")
    ax.set_ylim(0, 0.62)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    finish(fig, "model_validation_macro_f1.png")


def validation_test_gap() -> None:
    metrics = ["Accuracy", "Macro-F1", "Balanced\naccuracy"]
    validation = np.array([0.2857, 0.2736, 0.3157])
    test = np.array([0.1822, 0.1530, 0.1804])
    val_std = np.array([0.0530, 0.0290, 0.0039])
    test_std = np.array([0.0278, 0.0206, 0.0250])
    x = np.arange(len(metrics))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.bar(x - width / 2, validation, width, yerr=val_std, capsize=4, label="Validation: spk06", color=COLORS["blue"])
    ax.bar(x + width / 2, test, width, yerr=test_std, capsize=4, label="Test: spk07, spk08", color=COLORS["green"])
    ax.set_xticks(x, metrics)
    ax.set_ylabel("Metric")
    ax.set_ylim(0, 0.38)
    ax.set_title("LRW-модель со старым crop: исторические validation и test")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    finish(fig, "old_crop_validation_test_metrics.png")


def per_class_test_f1() -> None:
    words = ["будет", "есть", "когда", "значит", "человек", "может", "просто", "чтобы", "время", "потом"]
    f1 = [0.287, 0.255, 0.202, 0.194, 0.188, 0.134, 0.120, 0.113, 0.037, 0.000]
    support = [18, 25, 20, 6, 8, 10, 25, 28, 7, 3]

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    y = np.arange(len(words))
    bars = ax.barh(y, f1, color=COLORS["blue"])
    ax.set_yticks(y, [f"{word} (n={count})" for word, count in zip(words, support)])
    ax.invert_yaxis()
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in f1], padding=3)
    ax.set_xlim(0, 0.33)
    ax.set_xlabel("Mean test F1 по 3 seed")
    ax.set_title("Старый crop: результаты по словам на историческом test")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    finish(fig, "old_crop_test_per_class_f1.png")


def main() -> None:
    dataset_growth()
    dataset_split()
    v1_v2_comparison()
    model_comparison()
    validation_test_gap()
    per_class_test_f1()
    print(f"Saved report figures to {OUT_DIR}")


if __name__ == "__main__":
    main()

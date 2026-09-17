#!/usr/bin/env python3
"""2D PCA biplot of strain-mean terpene profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "data" / "processed" / "flower_terpenes.parquet"
CANONICAL = ROOT / "data" / "raw" / "terpenes.json"

LABELS = {
    "beta_myrcene": "β-myrcene",
    "d_limonene": "limonene",
    "beta_caryophyllene": "β-caryophyllene",
    "alpha_pinene": "α-pinene",
    "beta_pinene": "β-pinene",
    "humulene": "humulene",
    "linalool": "linalool",
    "terpinolene": "terpinolene",
    "ocimene": "ocimene",
    "alpha_bisabolol": "α-bisabolol",
    "caryophyllene_oxide": "caryophyllene oxide",
    "nerolidol": "nerolidol",
    "camphene": "camphene",
    "guaiol": "guaiol",
    "eucalyptol": "eucalyptol",
    "geraniol": "geraniol",
    "carene": "3-carene",
    "gamma_terpinene": "γ-terpinene",
    "alpha_terpinene": "α-terpinene",
    "p_cymene": "p-cymene",
    "isopulegol": "isopulegol",
}

CHEMOTYPE_COLORS = {
    "beta_caryophyllene": "#4c78a8",
    "d_limonene": "#f58518",
    "beta_myrcene": "#54a24b",
    "terpinolene": "#b279a2",
    "linalool": "#eeca3b",
    "alpha_pinene": "#72b7b2",
    "other": "#9d755d",
}


def load_keys() -> list[str]:
    return [row["key"] for row in json.loads(CANONICAL.read_text())]


def tidy_label(name: str) -> str:
    text = str(name).replace("CBX - ", "").replace(" - Manicured Flower", "")
    return text if len(text) <= 22 else text[:21] + "…"


def strain_means(df: pd.DataFrame, keys: list[str], min_samples: int) -> pd.DataFrame:
    counts = df["strain_name"].value_counts()
    eligible = counts[counts >= min_samples].index
    subset = df[df["strain_name"].isin(eligible)]
    means = subset.groupby("strain_name")[keys].mean().fillna(0.0)
    means["n_samples"] = counts.reindex(means.index).astype(int)
    return means


def pca_scores(shares: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = shares - shares.mean(axis=0)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    scores = u[:, :2] * s[:2]
    loadings = vt[:2].T
    explained = (s[:2] ** 2) / (s ** 2).sum()
    return scores, loadings, explained


def labels_to_keep(means: pd.DataFrame, scores: np.ndarray, top_n: int) -> list[str]:
    radius = np.sqrt((scores ** 2).sum(axis=1))
    ranked = pd.DataFrame(
        {"n": means["n_samples"].to_numpy(), "r": radius},
        index=means.index,
    )
    candidates = list(
        dict.fromkeys(
            ranked.sort_values("n", ascending=False).head(top_n).index.tolist()
            + ranked.sort_values("r", ascending=False).head(8).index.tolist()
        )
    )
    kept: list[str] = []
    pts = {name: scores[i] for i, name in enumerate(means.index)}
    span = max(np.ptp(scores[:, 0]), np.ptp(scores[:, 1]), 1e-9)
    min_dist = 0.055 * span
    for name in candidates:
        xy = pts[name]
        if any(np.linalg.norm(xy - pts[other]) < min_dist for other in kept):
            continue
        kept.append(name)
    return kept


def plot_biplot(
    means: pd.DataFrame,
    keys: list[str],
    scores: np.ndarray,
    loadings: np.ndarray,
    explained: np.ndarray,
    label_names: list[str],
    out: Path,
) -> None:
    shares = means[keys].div(means[keys].sum(axis=1), axis=0)
    dominant = shares.idxmax(axis=1)
    dominant = dominant.where(dominant.isin(CHEMOTYPE_COLORS), "other")

    fig, ax = plt.subplots(figsize=(11.5, 8.6))
    sizes = 18 + 2.4 * np.sqrt(means["n_samples"].to_numpy())

    for chemotype, color in CHEMOTYPE_COLORS.items():
        mask = dominant.eq(chemotype).to_numpy()
        if not mask.any():
            continue
        label = "other" if chemotype == "other" else LABELS.get(chemotype, chemotype)
        ax.scatter(
            scores[mask, 0],
            scores[mask, 1],
            s=sizes[mask],
            c=color,
            alpha=0.78,
            linewidths=0.3,
            edgecolors="white",
            label=f"{label}  n={int(mask.sum())}",
            zorder=3,
        )

    name_to_i = {name: i for i, name in enumerate(means.index)}
    for name in label_names:
        i = name_to_i[name]
        ax.annotate(
            tidy_label(name),
            (scores[i, 0], scores[i, 1]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
            color="#222222",
            zorder=4,
        )

    arrow_scale = 0.82 * np.max(np.abs(scores)) / max(np.max(np.abs(loadings)), 1e-9)
    loading_len = np.linalg.norm(loadings, axis=1)
    keep = np.argsort(loading_len)[-4:]
    for j in keep:
        dx, dy = loadings[j] * arrow_scale
        ax.annotate(
            "",
            xy=(dx, dy),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "-|>", "color": "#333333", "lw": 1.1},
            zorder=5,
        )
        ax.text(
            dx * 1.08,
            dy * 1.08,
            LABELS.get(keys[j], keys[j]),
            ha="center",
            va="center",
            fontsize=9,
            color="#111111",
            fontweight="bold",
            zorder=5,
        )

    ax.axhline(0, color="#cccccc", lw=0.8, zorder=1)
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=1)
    ax.set_xlabel(f"PC1  ({100 * explained[0]:.1f}% of variance)")
    ax.set_ylabel(f"PC2  ({100 * explained[1]:.1f}% of variance)")
    ax.set_title(
        "PCA of strain terpene vectors\n"
        "row-normalised profiles, unscaled; arrows are terpene loadings",
        fontsize=12,
        pad=12,
    )
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92, title="dominant terpene")
    ax.set_aspect("equal", adjustable="datalim")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(MATRIX))
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--label-top", type=int, default=20)
    parser.add_argument("--out", default=str(ROOT / "figures" / "strain_pca.png"))
    args = parser.parse_args()

    keys = load_keys()
    df = pd.read_parquet(args.matrix)
    means = strain_means(df, keys, args.min_samples)
    shares = means[keys].div(means[keys].sum(axis=1), axis=0).to_numpy()
    scores, loadings, explained = pca_scores(shares)
    label_names = labels_to_keep(means, scores, args.label_top)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_biplot(means, keys, scores, loadings, explained, label_names, out)

    coords = means[["n_samples"]].copy()
    coords["pc1"] = scores[:, 0]
    coords["pc2"] = scores[:, 1]
    coords["dominant"] = means[keys].idxmax(axis=1)
    coords.sort_values("n_samples", ascending=False).to_csv(
        out.with_name("strain_pca_coords.csv")
    )
    print(f"strains in PCA: {len(means)}")
    print(f"PC1 {100 * explained[0]:.1f}%   PC2 {100 * explained[1]:.1f}%")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

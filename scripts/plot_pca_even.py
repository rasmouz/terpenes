#!/usr/bin/env python3
"""PCA of strain terpene profiles in centered log-ratio space.

Raw shares live in a simplex corner, so a plain PCA piles most strains into one
blob. CLR plus whitening spreads the same strains over the plane without
inventing structure: the ordering and neighbourhoods are unchanged.
"""

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
    keep = counts[counts >= min_samples].index
    subset = df[df["strain_name"].isin(keep)]
    means = subset.groupby("strain_name")[keys].mean().fillna(0.0)
    means["n_samples"] = counts.reindex(means.index).astype(int)
    return means


def select_terpenes(shares: pd.DataFrame, min_prevalence: float) -> list[str]:
    prevalence = (shares > 0).mean()
    return prevalence[prevalence >= min_prevalence].index.tolist()


def clr(shares: pd.DataFrame, floor: float) -> pd.DataFrame:
    """Centered log-ratio with a detection floor for zeros."""
    x = shares.clip(lower=floor)
    x = x.div(x.sum(axis=1), axis=0)
    log_x = np.log(x)
    return log_x.sub(log_x.mean(axis=1), axis=0)


def pca(values: np.ndarray, whiten: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = values - values.mean(axis=0)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    explained = (s[:2] ** 2) / (s ** 2).sum()
    scores = u[:, :2] * np.sqrt(len(values) - 1) if whiten else u[:, :2] * s[:2]
    return scores, vt[:2].T, explained


def evenness(scores: np.ndarray, grid: int = 16) -> tuple[float, float]:
    """Return (nearest-neighbour distance CV, share of occupied grid cells)."""
    span = float((scores.max(axis=0) - scores.min(axis=0)).max())
    unit = (scores - scores.min(axis=0)) / (span if span > 0 else 1.0)
    diff = unit[:, None, :] - unit[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    nn = dist.min(axis=1)
    cells = {(min(int(x * grid), grid - 1), min(int(y * grid), grid - 1)) for x, y in unit}
    return float(nn.std() / nn.mean()), len(cells) / (grid * grid)


def pick_labels(means: pd.DataFrame, scores: np.ndarray, top_n: int, max_labels: int) -> list[str]:
    radius = np.sqrt((scores ** 2).sum(axis=1))
    ranked = pd.DataFrame({"n": means["n_samples"].to_numpy(), "r": radius}, index=means.index)
    candidates = list(
        dict.fromkeys(
            ranked.sort_values("n", ascending=False).head(top_n).index.tolist()
            + ranked.sort_values("r", ascending=False).head(10).index.tolist()
        )
    )
    span = scores.max(axis=0) - scores.min(axis=0)
    idx = {name: i for i, name in enumerate(means.index)}
    kept: list[str] = []
    for name in candidates:
        xy = scores[idx[name]]
        if any(np.linalg.norm((xy - scores[idx[o]]) / span) < 0.06 for o in kept):
            continue
        kept.append(name)
        if len(kept) >= max_labels:
            break
    return kept


def dominant_series(shares: pd.DataFrame) -> pd.Series:
    dominant = shares.idxmax(axis=1)
    return dominant.where(dominant.isin(CHEMOTYPE_COLORS), "other")


def scatter(ax, scores: np.ndarray, means: pd.DataFrame, dominant: pd.Series, size_scale: float) -> None:
    sizes = size_scale * (6 + np.sqrt(means["n_samples"].to_numpy()))
    for chemotype, color in CHEMOTYPE_COLORS.items():
        mask = dominant.eq(chemotype).to_numpy()
        if not mask.any():
            continue
        ax.scatter(
            scores[mask, 0],
            scores[mask, 1],
            s=sizes[mask],
            c=color,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.3,
            label=f"{LABELS.get(chemotype, chemotype)}  n={int(mask.sum())}",
            zorder=3,
        )


def plot_main(
    means: pd.DataFrame,
    cols: list[str],
    scores: np.ndarray,
    loadings: np.ndarray,
    explained: np.ndarray,
    dominant: pd.Series,
    labels: list[str],
    metrics: tuple[float, float],
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 8.8))
    scatter(ax, scores, means, dominant, size_scale=3.2)

    idx = {name: i for i, name in enumerate(means.index)}
    for name in labels:
        i = idx[name]
        ax.annotate(
            tidy_label(name),
            (scores[i, 0], scores[i, 1]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
            color="#222222",
            zorder=4,
        )

    scale = 0.8 * np.abs(scores).max() / max(np.abs(loadings).max(), 1e-9)
    order = np.argsort(np.linalg.norm(loadings, axis=1))[-6:]
    tips = []
    for j in order:
        dx, dy = loadings[j] * scale
        tips.append((dx * 1.14, dy * 1.14))
        ax.annotate(
            "",
            xy=(dx, dy),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "-|>", "color": "#444444", "lw": 1.1},
            zorder=5,
        )
        ax.text(
            dx * 1.07,
            dy * 1.07,
            LABELS.get(cols[j], cols[j]),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#111111",
            zorder=5,
        )

    extent = np.vstack([scores, np.array(tips)])
    pad = 0.06 * (extent.max(axis=0) - extent.min(axis=0))
    ax.set_xlim(extent[:, 0].min() - pad[0], extent[:, 0].max() + pad[0])
    ax.set_ylim(extent[:, 1].min() - pad[1], extent[:, 1].max() + pad[1])

    ax.axhline(0, color="#dddddd", lw=0.8, zorder=1)
    ax.axvline(0, color="#dddddd", lw=0.8, zorder=1)
    ax.set_xlabel(f"PC1  ({100 * explained[0]:.1f}% of CLR variance, whitened)")
    ax.set_ylabel(f"PC2  ({100 * explained[1]:.1f}% of CLR variance, whitened)")
    ax.set_title(
        "Strain terpene map — CLR-PCA with whitened axes\n"
        f"{len(means)} strains · grid occupancy {metrics[1]:.0%} · nearest-neighbour CV {metrics[0]:.2f}",
        fontsize=12,
        pad=12,
    )
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92, title="dominant terpene")
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_comparison(
    means: pd.DataFrame,
    variants: list[tuple[str, np.ndarray]],
    dominant: pd.Series,
    out: Path,
) -> None:
    fig, axes = plt.subplots(1, len(variants), figsize=(4.6 * len(variants), 4.9))
    for ax, (title, scores) in zip(axes, variants):
        scatter(ax, scores, means, dominant, size_scale=1.0)
        cv, occ = evenness(scores)
        ax.set_title(f"{title}\noccupancy {occ:.0%} · NN CV {cv:.2f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        for spine in ax.spines.values():
            spine.set_color("#cccccc")
    fig.suptitle("How the transform changes the spread (same strains, same colours)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(MATRIX))
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--min-prevalence", type=float, default=0.7)
    parser.add_argument("--floor", type=float, default=1e-3, help="share floor for zeros")
    parser.add_argument("--label-top", type=int, default=22)
    parser.add_argument("--max-labels", type=int, default=24)
    parser.add_argument("--out", default=str(ROOT / "figures" / "strain_pca_clr.png"))
    args = parser.parse_args()

    keys = load_keys()
    df = pd.read_parquet(args.matrix)
    means = strain_means(df, keys, args.min_samples)
    shares_all = means[keys].div(means[keys].sum(axis=1), axis=0)
    cols = select_terpenes(shares_all, args.min_prevalence)
    print(f"terpenes kept ({len(cols)}): {', '.join(cols)}")

    shares = shares_all[cols].div(shares_all[cols].sum(axis=1), axis=0)
    dominant = dominant_series(shares)

    clr_values = clr(shares, args.floor)
    scores, loadings, explained = pca(clr_values.to_numpy(), whiten=True)
    raw_scores, _, _ = pca(shares.to_numpy(), whiten=False)
    clr_plain, _, _ = pca(clr_values.to_numpy(), whiten=False)

    metrics = evenness(scores)
    labels = pick_labels(means, scores, args.label_top, args.max_labels)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_main(means, cols, scores, loadings, explained, dominant, labels, metrics, out)
    plot_comparison(
        means,
        [
            ("raw shares, plain PCA", raw_scores),
            ("CLR, plain PCA", clr_plain),
            ("CLR, whitened axes", scores),
        ],
        dominant,
        out.with_name("strain_pca_spread_comparison.png"),
    )

    coords = means[["n_samples"]].copy()
    coords["pc1"] = scores[:, 0]
    coords["pc2"] = scores[:, 1]
    coords["dominant"] = dominant
    coords.sort_values("n_samples", ascending=False).to_csv(out.with_name("strain_pca_clr_coords.csv"))

    for name, sc in [("raw shares", raw_scores), ("CLR", clr_plain), ("CLR whitened", scores)]:
        cv, occ = evenness(sc)
        print(f"{name:14s} occupancy {occ:5.1%}   nearest-neighbour CV {cv:.2f}")
    print(f"PC1 {100 * explained[0]:.1f}%   PC2 {100 * explained[1]:.1f}%")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

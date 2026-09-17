#!/usr/bin/env python3
"""Plot terpene vectors for the best-sampled strains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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


def load_keys() -> list[str]:
    return [row["key"] for row in json.loads(CANONICAL.read_text())]


def tidy_label(name: str) -> str:
    text = str(name).replace("CBX - ", "").replace(" - Manicured Flower", "")
    return text if len(text) <= 28 else text[:27] + "…"


def strain_means(df: pd.DataFrame, keys: list[str], top: int, min_samples: int) -> pd.DataFrame:
    counts = df["strain_name"].value_counts()
    eligible = counts[counts >= min_samples].head(top).index
    subset = df[df["strain_name"].isin(eligible)]
    means = subset.groupby("strain_name")[keys].mean()
    means["n_samples"] = counts.reindex(means.index)
    return means.loc[eligible]


def order_axes(means: pd.DataFrame, keys: list[str]) -> tuple[list[str], list[str]]:
    col_order = means[keys].mean().sort_values(ascending=False).index.tolist()
    shares = means[col_order].div(means[col_order].sum(axis=1), axis=0)
    dominant = shares.idxmax(axis=1).map(col_order.index)
    row_order = (
        pd.DataFrame({"dom": dominant, "share": shares.max(axis=1)})
        .sort_values(["dom", "share"], ascending=[True, False])
        .index.tolist()
    )
    return row_order, col_order


def plot_heatmap(means: pd.DataFrame, rows: list[str], cols: list[str], out: Path) -> None:
    shares = means.loc[rows, cols].div(means.loc[rows, cols].sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad("#d9d9d9")
    im = ax.imshow(shares.values, cmap=cmap, vmin=0, vmax=60, aspect="auto")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([LABELS.get(c, c) for c in cols], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [f"{tidy_label(r)}  (n={int(means.loc[r, 'n_samples'])})" for r in rows], fontsize=9
    )

    for i in range(shares.shape[0]):
        for j in range(shares.shape[1]):
            value = shares.iat[i, j]
            if value >= 1:
                ax.text(
                    j,
                    i,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if value > 32 else "#333333",
                )

    ax.set_xticks([x - 0.5 for x in range(1, len(cols))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(rows))], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)

    fig.colorbar(im, ax=ax, shrink=0.8, label="share of terpene profile (%)")
    ax.set_title(
        "Terpene vectors of 20 best-sampled flower strains\n"
        "each row is a mean profile, normalised to 100% (grey = analyte not reported)",
        fontsize=12,
        pad=14,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_stacked(means: pd.DataFrame, rows: list[str], cols: list[str], out: Path) -> None:
    top_cols = cols[:10]
    data = means.loc[rows, cols]
    fig, ax = plt.subplots(figsize=(11, 8.2))
    colors = plt.get_cmap("tab10").colors
    y = range(len(rows))
    left = pd.Series(0.0, index=rows)

    for idx, col in enumerate(top_cols):
        ax.barh(list(y), data[col].values, left=left.values, color=colors[idx],
                label=LABELS.get(col, col), height=0.72)
        left = left.add(data[col], fill_value=0)

    rest = data[cols[len(top_cols):]].sum(axis=1)
    rest_label = f"other {len(cols) - len(top_cols)}"
    ax.barh(list(y), rest.values, left=left.values, color="#bdbdbd", label=rest_label, height=0.72)
    totals = left.add(rest, fill_value=0)

    ax.set_yticks(list(y))
    ax.set_yticklabels([tidy_label(r) for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("terpene content (% w/w of flower)")
    ax.set_title("Same 20 strains, absolute terpene load", fontsize=12, pad=12)
    for i, total in enumerate(totals.values):
        ax.text(total + 0.03, i, f"{total:.2f}%", va="center", fontsize=8, color="#444444")
    ax.legend(
        ncol=6,
        fontsize=8.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        frameon=False,
    )
    ax.margins(x=0.12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(MATRIX))
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--out-dir", default=str(ROOT / "figures"))
    args = parser.parse_args()

    keys = load_keys()
    df = pd.read_parquet(args.matrix)
    means = strain_means(df, keys, args.top, args.min_samples)
    rows, cols = order_axes(means, keys)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    heatmap = out_dir / "strain_terpene_heatmap.png"
    stacked = out_dir / "strain_terpene_stacked.png"
    plot_heatmap(means, rows, cols, heatmap)
    plot_stacked(means, rows, cols, stacked)

    means.loc[rows, ["n_samples"] + cols].round(4).to_csv(out_dir / "strain_means.csv")
    print(f"strains plotted: {len(rows)}")
    print(f"wrote {heatmap}")
    print(f"wrote {stacked}")


if __name__ == "__main__":
    main()

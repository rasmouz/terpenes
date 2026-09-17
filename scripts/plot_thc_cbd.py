#!/usr/bin/env python3
"""Plot strains in THC vs CBD space using lab COA values."""

from __future__ import annotations

import argparse
import json
import math
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
    "terpinolene": "terpinolene",
    "linalool": "linalool",
    "alpha_pinene": "α-pinene",
    "other": "other",
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
NONDETECT = {"", "nd", "n/d", "na", "n/a", "none", "nan", "nt", "lod", "not detected"}


def load_keys() -> list[str]:
    return [row["key"] for row in json.loads(CANONICAL.read_text())]


def parse_number(value) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if text.lower() in NONDETECT:
        return 0.0
    if text.startswith("<"):
        return 0.0
    try:
        return float(text)
    except ValueError:
        return None


def tidy_label(name: str) -> str:
    text = str(name).replace("CBX - ", "").replace(" - Manicured Flower", "")
    return text if len(text) <= 22 else text[:21] + "…"


def read_cannabinoids(path: Path) -> pd.DataFrame:
    needed = {"sample_id", "total_thc", "total_cbd"}
    if path.suffix.lower() in {".xlsx", ".xls"}:
        header = pd.read_excel(path, nrows=0).columns
        if not needed.issubset(set(header)):
            return pd.DataFrame(columns=list(needed))
        df = pd.read_excel(path, usecols=lambda c: c in needed)
    else:
        header = pd.read_csv(path, encoding="latin-1", nrows=0).columns
        if not needed.issubset(set(header)):
            return pd.DataFrame(columns=list(needed))
        df = pd.read_csv(path, encoding="latin-1", usecols=lambda c: c in needed, low_memory=False)
    df["total_thc"] = df["total_thc"].map(parse_number)
    df["total_cbd"] = df["total_cbd"].map(parse_number)
    df = df.dropna(subset=["sample_id"])
    df["sample_id"] = df["sample_id"].astype(str)
    return df.drop_duplicates("sample_id")


def attach_cannabinoids(matrix: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    if {"total_thc", "total_cbd"}.issubset(matrix.columns) and matrix["total_thc"].notna().any():
        return matrix
    frames = []
    for path in sorted(list(raw_dir.glob("*results*.csv")) + list(raw_dir.glob("*results*.xlsx"))):
        print(f"reading cannabinoids from {path.name}")
        frame = read_cannabinoids(path)
        if frame.empty:
            print(f"  skip {path.name} (no total_thc/total_cbd)")
            continue
        frames.append(frame)
    cann = pd.concat(frames, ignore_index=True).drop_duplicates("sample_id")
    merged = matrix.copy()
    merged["sample_id"] = merged["sample_id"].astype(str)
    merged = merged.merge(cann, on="sample_id", how="left")
    print(f"THC present: {merged['total_thc'].notna().mean():.0%}  CBD present: {merged['total_cbd'].notna().mean():.0%}")
    return merged


def strain_table(df: pd.DataFrame, keys: list[str], min_samples: int) -> pd.DataFrame:
    subset = df.copy()
    subset.loc[subset["total_thc"] > 40, "total_thc"] = np.nan
    subset.loc[subset["total_cbd"] > 30, "total_cbd"] = np.nan
    shares = subset[keys].fillna(0).div(subset[keys].fillna(0).sum(axis=1).replace(0, np.nan), axis=0)
    subset["dominant"] = shares.idxmax(axis=1)
    grouped = subset.groupby("strain_name").agg(
        n_samples=("strain_name", "size"),
        total_thc=("total_thc", "median"),
        total_cbd=("total_cbd", "median"),
        dominant=("dominant", lambda s: s.mode().iloc[0] if not s.mode().empty else "other"),
    )
    grouped = grouped.dropna(subset=["total_thc", "total_cbd"])
    popular = grouped["n_samples"] >= min_samples
    cbd_rich = (grouped["total_cbd"] >= 1.0) & (grouped["n_samples"] >= 2)
    grouped = grouped[popular | cbd_rich]
    grouped["dominant"] = grouped["dominant"].where(grouped["dominant"].isin(CHEMOTYPE_COLORS), "other")
    return grouped


def pick_labels(means: pd.DataFrame, top_n: int) -> list[str]:
    high_thc = means.sort_values("total_thc", ascending=False).head(4).index.tolist()
    high_cbd = means.sort_values("total_cbd", ascending=False).head(6).index.tolist()
    popular = means.sort_values("n_samples", ascending=False).head(top_n).index.tolist()
    candidates = list(dict.fromkeys(high_cbd + high_thc + popular))
    kept: list[str] = []
    pts = means[["total_thc", "total_cbd"]].to_numpy()
    idx = {name: i for i, name in enumerate(means.index)}
    span = np.array([means["total_thc"].max(), max(means["total_cbd"].max(), 1.0)])
    min_dist = 0.06 * np.linalg.norm(span)
    for name in candidates:
        xy = pts[idx[name]]
        if any(np.linalg.norm((xy - pts[idx[other]]) / span) < 0.07 for other in kept):
            continue
        kept.append(name)
        if len(kept) >= 18:
            break
    return kept


def plot_axes(means: pd.DataFrame, labels: list[str], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.4), sharey=False)

    for ax, stretch in zip(axes, (False, True)):
        for chemotype, color in CHEMOTYPE_COLORS.items():
            mask = means["dominant"].eq(chemotype)
            if not mask.any():
                continue
            ax.scatter(
                means.loc[mask, "total_thc"],
                means.loc[mask, "total_cbd"],
                s=18 + 2.2 * np.sqrt(means.loc[mask, "n_samples"]),
                c=color,
                alpha=0.8,
                edgecolors="white",
                linewidths=0.3,
                label=f"{LABELS.get(chemotype, chemotype)}  n={int(mask.sum())}",
                zorder=3,
            )
        for name in labels:
            ax.annotate(
                tidy_label(name),
                (means.loc[name, "total_thc"], means.loc[name, "total_cbd"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
                color="#222222",
            )
        ax.set_xlabel("total THC (% w/w)")
        ax.axhline(0, color="#dddddd", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        if stretch:
            ax.set_yscale("symlog", linthresh=0.3)
            ax.set_ylabel("total CBD (% w/w, stretched)")
            ax.set_title("Same points, CBD axis stretched\nso Type II / III strains are visible")
        else:
            ax.set_ylabel("total CBD (% w/w)")
            ax.set_title("Lab THC vs CBD\nmost modern flower sits on the THC axis")
            ax.legend(fontsize=7.5, loc="upper right", framealpha=0.92, title="dominant terpene")

        xs = np.linspace(0.5, 35, 80)
        ax.plot(xs, xs, color="#bbbbbb", lw=0.9, ls="--", zorder=1)
        ax.text(18, 19.2, "THC = CBD", fontsize=8, color="#888888", rotation=45)

    fig.suptitle(
        "2D strain map with cannabinoid axes (median lab values per strain)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(MATRIX))
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--label-top", type=int, default=12)
    parser.add_argument("--out", default=str(ROOT / "figures" / "strain_thc_cbd.png"))
    args = parser.parse_args()

    keys = load_keys()
    df = pd.read_parquet(args.matrix)
    df = attach_cannabinoids(df, ROOT / "data" / "raw")
    Path(args.matrix).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.matrix, index=False)
    df.to_csv(Path(args.matrix).with_suffix(".csv"), index=False)

    means = strain_table(df, keys, args.min_samples)
    labels = pick_labels(means, args.label_top)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_axes(means, labels, out)
    means.sort_values("n_samples", ascending=False).to_csv(out.with_name("strain_thc_cbd.csv"))
    print(f"strains plotted: {len(means)}")
    print(f"median THC {means['total_thc'].median():.1f}%   median CBD {means['total_cbd'].median():.2f}%")
    print(f"CBD ≥ 1%: {(means['total_cbd'] >= 1).mean():.1%}   CBD ≥ 5%: {(means['total_cbd'] >= 5).mean():.1%}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

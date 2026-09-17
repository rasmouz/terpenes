#!/usr/bin/env python3
"""Build a flower x 21-terpene matrix from Cannlytics COA files."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PATH = ROOT / "data" / "raw" / "terpenes.json"
ALIASES_PATH = ROOT / "config" / "aliases.json"

FLOWER_INCLUDE = re.compile(r"flower|cured", re.I)
FLOWER_EXCLUDE = re.compile(
    r"infused|pre-?roll|preroll|hemp|edible|concentrate|vape|"
    r"extract|hash|kief|topical|lozenge|gummy|capsule|tincture",
    re.I,
)
TERPENE_ANALYSIS = re.compile(r"terp", re.I)
NONDETECT = {
    "",
    "nd",
    "n/d",
    "na",
    "n/a",
    "none",
    "nan",
    "nt",
    "lod",
    "not detected",
    "not tested",
}


def load_canonical_keys() -> list[str]:
    analytes = json.loads(CANONICAL_PATH.read_text())
    return [row["key"] for row in analytes]


def load_aliases() -> dict[str, str]:
    return json.loads(ALIASES_PATH.read_text())


def parse_results(raw) -> list[dict]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if not isinstance(raw, str) or not raw.strip():
        return []
    text = raw.strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            obj = loader(text)
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, dict)]
        except (TypeError, ValueError, json.JSONDecodeError, SyntaxError, MemoryError):
            continue
    return []


def parse_number(value) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if text.lower() in NONDETECT:
        return 0.0
    if text.startswith("<"):
        rest = re.sub(r"(?i)lo[qd]", "", text[1:]).strip()
        try:
            return float(rest) / 2.0
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return None


def to_percent(value: float, units: str | None) -> float:
    if units is None:
        return value
    u = str(units).strip().lower()
    if u in NONDETECT or u in {"<loq", "none"}:
        return value
    if u in {"percent", "%", "wt%", "w/w%", "pct"}:
        return value
    if u in {"mg/g", "mg_g", "mg/gram"}:
        return value / 10.0
    if u in {"ug/g", "µg/g", "ppm"}:
        return value / 10000.0
    return value


def is_flower(row: pd.Series) -> bool:
    standard = str(row.get("standard_product_type") or "")
    if standard.lower() == "flower":
        return True
    blob = " ".join(
        str(row.get(col) or "")
        for col in ("product_type", "product_category", "sample_matrix", "classification")
    )
    return bool(FLOWER_INCLUDE.search(blob) and not FLOWER_EXCLUDE.search(blob))


def terpenes_from_results(raw, aliases: dict[str, str], canonical: set[str]) -> dict[str, float]:
    found: dict[str, list[float]] = {}
    for item in parse_results(raw):
        analysis = str(item.get("analysis") or "")
        key = str(item.get("key") or "").strip()
        if key in {"total", "total_terpenes"}:
            continue
        mapped = aliases.get(key, key)
        if mapped not in canonical:
            continue
        if analysis and not TERPENE_ANALYSIS.search(analysis):
            # Some COAs omit analysis or mis-tag terpenes; keep mapped canonical keys.
            if key not in aliases and mapped == key and analysis.lower() not in {"", "none", "nan"}:
                continue
        value = parse_number(item.get("value"))
        if value is None:
            continue
        value = to_percent(value, item.get("units"))
        found.setdefault(mapped, []).append(value)
    return {key: sum(vals) for key, vals in found.items()}


def terpenes_from_wide(row: pd.Series, aliases: dict[str, str], canonical: set[str]) -> dict[str, float]:
    found: dict[str, list[float]] = {}
    for col, raw in row.items():
        mapped = aliases.get(str(col), str(col))
        if mapped not in canonical:
            continue
        value = parse_number(raw)
        if value is None:
            continue
        found.setdefault(mapped, []).append(value)
    return {key: sum(vals) for key, vals in found.items()}


def infer_state(path: Path, df: pd.DataFrame) -> str:
    if "state" in df.columns and df["state"].notna().any():
        return str(df["state"].dropna().iloc[0])
    name = path.name.lower()
    for code in ("ca", "co", "ct", "fl", "hi", "ma", "md", "mi", "nv", "ny", "or", "ri", "ut", "wa"):
        if name.startswith(f"{code}-") or f"_{code}_" in name:
            return code.upper()
    return path.stem.split("-")[0].upper()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin-1", low_memory=False)


def process_file(path: Path, canonical: list[str], aliases: dict[str, str]) -> pd.DataFrame:
    df = read_table(path)
    canonical_set = set(canonical)
    state = infer_state(path, df)
    rows = []
    has_results = "results" in df.columns
    for _, row in df.iterrows():
        if not is_flower(row):
            continue
        vector = {}
        if has_results:
            vector = terpenes_from_results(row.get("results"), aliases, canonical_set)
        if not vector:
            vector = terpenes_from_wide(row, aliases, canonical_set)
        if not vector:
            continue
        total = parse_number(row.get("total_terpenes")) if "total_terpenes" in df.columns else None
        out = {
            "sample_id": row.get("sample_id"),
            "strain_name": row.get("strain_name"),
            "product_name": row.get("product_name"),
            "producer": row.get("producer"),
            "lab": row.get("lab"),
            "product_type": row.get("product_type"),
            "date_tested": row.get("date_tested"),
            "source_file": path.name,
            "state": state,
            "total_terpenes": total,
            "total_thc": parse_number(row.get("total_thc")) if "total_thc" in df.columns else None,
            "total_cbd": parse_number(row.get("total_cbd")) if "total_cbd" in df.columns else None,
        }
        for key in canonical:
            out[key] = vector.get(key)
        rows.append(out)
    return pd.DataFrame(rows)


def scale_series(values: pd.Series) -> pd.Series:
    scale = pd.Series(1.0, index=values.index)
    scale = scale.mask(values > 250, 10000.0)
    scale = scale.mask((values > 25) & (values <= 250), 10.0)
    return values / scale


def normalize_units(df: pd.DataFrame, canonical: list[str]) -> pd.DataFrame:
    terp = df[canonical].apply(pd.to_numeric, errors="coerce")
    row_sum = terp.sum(axis=1, skipna=True)
    scale = pd.Series(1.0, index=df.index)
    scale = scale.mask(row_sum > 250, 10000.0)
    scale = scale.mask((row_sum > 25) & (row_sum <= 250), 10.0)
    df[canonical] = terp.div(scale, axis=0)
    if "total_terpenes" in df.columns:
        df["total_terpenes"] = scale_series(pd.to_numeric(df["total_terpenes"], errors="coerce"))
    return df


def drop_implausible(df: pd.DataFrame, canonical: list[str]) -> pd.DataFrame:
    terp = df[canonical].apply(pd.to_numeric, errors="coerce")
    row_sum = terp.sum(axis=1, skipna=True)
    too_high = (terp.max(axis=1) > 15) | (row_sum > 20)
    has_signal = terp.gt(0).any(axis=1)
    return df.loc[~too_high & has_signal].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="COA csv/xlsx files. Defaults to data/raw/*results*",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / "processed" / "flower_terpenes.parquet"),
    )
    args = parser.parse_args()

    canonical = load_canonical_keys()
    aliases = load_aliases()
    aliases.update({key: key for key in canonical})

    if args.inputs:
        paths = [Path(p) for p in args.inputs]
    else:
        raw = ROOT / "data" / "raw"
        paths = sorted(
            p
            for p in list(raw.glob("*results*.csv")) + list(raw.glob("*results*.xlsx"))
            if p.is_file()
        )
    if not paths:
        raise SystemExit("No input files found.")

    frames = []
    for path in paths:
        print(f"processing {path.name} ...")
        frame = process_file(path, canonical, aliases)
        print(f"  flower rows with terpenes: {len(frame)}")
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise SystemExit("No flower terpene rows found.")

    df = pd.concat(frames, ignore_index=True)
    df = normalize_units(df, canonical)
    before = len(df)
    df = drop_implausible(df, canonical)
    print(f"dropped implausible rows: {before - len(df)}")

    strain = df["strain_name"].astype("string")
    missing_strain = (
        strain.isna()
        | strain.str.strip().eq("")
        | strain.str.lower().isin(["none", "nan", "null"])
    )
    df.loc[missing_strain, "strain_name"] = df.loc[missing_strain, "product_name"]

    id_cols = [
        "sample_id",
        "state",
        "lab",
        "producer",
        "strain_name",
        "product_name",
        "product_type",
        "date_tested",
        "source_file",
        "total_terpenes",
        "total_thc",
        "total_cbd",
    ]
    df = df[id_cols + canonical]
    df["terpene_sum"] = df[canonical].sum(axis=1, skipna=True)
    df["n_detected"] = df[canonical].gt(0).sum(axis=1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    csv_out = out.with_suffix(".csv")
    df.to_csv(csv_out, index=False)

    print(f"wrote {out} and {csv_out}")
    print(f"samples: {len(df):,}  unique strain_name: {df['strain_name'].nunique(dropna=True):,}")
    print("detection rate (share > 0):")
    rates = (df[canonical].gt(0).mean() * 100).sort_values(ascending=False)
    for key, rate in rates.items():
        print(f"  {key:22s} {rate:5.1f}%")
    print(df[["terpene_sum", "n_detected", "total_terpenes"]].describe().to_string())


if __name__ == "__main__":
    main()

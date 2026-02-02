#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EPW morphing utilities for Task 4 (MORPH-1).
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from . import ensure_src_on_path
from .config_task4 import MORPH_COLS

ensure_src_on_path()

from io_epw import read_epw, write_epw_with_updates
from solar_geometry import calculate_solar_position


def _month_values(value, default=0.0) -> np.ndarray:
    if value is None:
        return np.full(12, default, dtype=float)
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 12:
        return np.array(value, dtype=float)
    return np.full(12, float(value), dtype=float)


def morph_epw_monthly_shift_scale(
    epw_df: pd.DataFrame,
    deltas: Dict[str, Iterable[float]],
    scales: Dict[str, Iterable[float]],
    cols: Iterable[str],
    meta: Dict | None = None,
    recompute_ghi: bool = True,
) -> pd.DataFrame:
    """
    Apply MORPH-1 monthly shift + scale to EPW dataframe.

    x_future(t,m) = (x0(t,m) + Δx_m) * (1 + a_m)
    """
    df = epw_df.copy()
    if "datetime" in df.columns:
        times = pd.DatetimeIndex(df["datetime"])
    else:
        times = pd.DatetimeIndex(df.index)
    months = times.month.values

    for col in cols:
        if col not in df.columns:
            continue
        delta_m = _month_values(deltas.get(col, 0.0))
        scale_m = _month_values(scales.get(col, 0.0))
        values = df[col].astype(float).to_numpy()
        out = np.zeros_like(values, dtype=float)
        for m in range(1, 13):
            mask = months == m
            out[mask] = (values[mask] + delta_m[m - 1]) * (1.0 + scale_m[m - 1])
        df[col] = out

    # Recompute GHI for consistency when possible
    if recompute_ghi and {"DNI", "DHI", "GHI"}.issubset(df.columns) and meta:
        solar_pos = calculate_solar_position(
            times=times,
            latitude=meta.get("latitude", 0.0),
            longitude=meta.get("longitude", 0.0),
            altitude=meta.get("altitude", 0.0),
        )
        cos_zenith = np.cos(np.radians(solar_pos["zenith"].values))
        cos_zenith = np.clip(cos_zenith, 0.0, 1.0)
        df["GHI"] = df["DHI"].to_numpy() + df["DNI"].to_numpy() * cos_zenith

    # Clip irradiance columns to >= 0
    for col in ("DNI", "DHI", "GHI"):
        if col in df.columns:
            df[col] = df[col].clip(lower=0.0)

    return df


def compute_climate_stats(epw_df: pd.DataFrame, base_temp_c: float = 18.0) -> Dict[str, float]:
    """
    Compute climate statistics for reporting.
    """
    temp = epw_df["Temp"].astype(float).to_numpy()
    cdd = np.sum(np.maximum(0.0, temp - base_temp_c))
    return {
        "temp_mean_C": float(np.mean(temp)),
        "temp_peak_C": float(np.max(temp)),
        "CDD_base18": float(cdd),
        "DNI_mean": float(np.mean(epw_df["DNI"].astype(float))),
        "DHI_mean": float(np.mean(epw_df["DHI"].astype(float))),
    }


def generate_future_epws(
    epw_path: str,
    scenarios: Dict[str, Dict],
    results_dir: str,
    recompute_ghi: bool = True,
) -> Tuple[Dict[str, Dict], Dict[str, str]]:
    """
    Generate future EPW files and climate stats.
    """
    os.makedirs(results_dir, exist_ok=True)
    epw_out_dir = os.path.join(results_dir, "epw_future")
    os.makedirs(epw_out_dir, exist_ok=True)

    epw_df, meta = read_epw(epw_path)
    row_count = len(epw_df)
    if row_count not in (8760, 8784):
        raise ValueError(f"Unexpected EPW row count: {row_count}")

    stats = {}
    epw_paths = {"now": epw_path}
    stats["now"] = compute_climate_stats(epw_df)

    for scenario, cfg in scenarios.items():
        if scenario == "now":
            continue
        df_future = morph_epw_monthly_shift_scale(
            epw_df=epw_df,
            deltas=cfg.get("deltas", {}),
            scales=cfg.get("scales", {}),
            cols=MORPH_COLS,
            meta=meta,
            recompute_ghi=recompute_ghi,
        )
        year = cfg.get("year", scenario)
        out_path = os.path.join(epw_out_dir, f"EPW_future_{year}.epw")
        write_epw_with_updates(
            base_epw_path=epw_path,
            output_epw_path=out_path,
            updates={
                "Temp": df_future["Temp"].to_numpy(),
                "DNI": df_future["DNI"].to_numpy(),
                "DHI": df_future["DHI"].to_numpy(),
                "GHI": df_future["GHI"].to_numpy(),
            },
        )
        epw_paths[scenario] = out_path
        stats[scenario] = compute_climate_stats(df_future)

    stats_df = pd.DataFrame.from_dict(stats, orient="index")
    stats_df.to_csv(os.path.join(results_dir, "climate_stats.csv"), index=True)

    return stats, epw_paths

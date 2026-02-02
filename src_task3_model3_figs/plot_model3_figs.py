#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Model 3 figures (Section 6) for the draft paper.

Figures:
 - 6-1: Mask_target angle-domain comparison (Sungrove vs Borealis).
 - 6-2: Winter representative week time series (Q_sol, T_a, T_m, Q_mass).
 - 6-3: Borealis winter shading openness heatmap (occupied vs unoccupied).
 - 6-4: Section schematic comparison (Sungrove summer vs Borealis winter).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_src_on_path() -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(repo_root, "src")
    if src_dir not in sys.path:
        sys.path.append(src_dir)
    return repo_root


REPO_ROOT = _ensure_src_on_path()

from io_epw import read_epw
from solar_geometry import calculate_solar_position, incidence_cos_window
from irradiance_perez import tilted_irradiance
from thermal_loads import simulate_2R2C, split_solar_to_air_and_mass


def _load_epw(epw_path: str) -> Tuple[pd.DataFrame, dict]:
    df, meta = read_epw(epw_path)
    if "datetime" in df.columns:
        df = df.set_index("datetime")
    df = df.sort_index()
    return df, meta


def _bin_mask_fraction(df, meta, weight_heat: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = df.index
    solar_pos = calculate_solar_position(
        times=times,
        latitude=meta.get("latitude", 0.0),
        longitude=meta.get("longitude", 0.0),
        altitude=meta.get("altitude", 0.0),
    )
    altitude = 90.0 - solar_pos["zenith"].to_numpy()
    azimuth = solar_pos["azimuth"].to_numpy()
    az_s = ((azimuth - 180.0 + 180.0) % 360.0) - 180.0  # [-180, 180], 0=south

    temp_out = df["Temp"].to_numpy()
    t_cool = 24.0
    t_heat = 20.0
    cooling = np.maximum(0.0, temp_out - t_cool)
    heating = np.maximum(0.0, t_heat - temp_out)
    sdi = cooling - weight_heat * heating
    mask = sdi > 0.0

    sun_mask = altitude > 0.0
    altitude = altitude[sun_mask]
    az_s = az_s[sun_mask]
    mask = mask[sun_mask]

    alt_bins = np.linspace(0.0, 90.0, 19)
    az_bins = np.linspace(-180.0, 180.0, 37)

    total, _, _ = np.histogram2d(altitude, az_s, bins=[alt_bins, az_bins])
    shaded, _, _ = np.histogram2d(altitude, az_s, bins=[alt_bins, az_bins], weights=mask.astype(float))

    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(total > 0, shaded / total, np.nan)

    return alt_bins, az_bins, frac


def plot_6_1(epw_sungrove: str, epw_borealis: str, outdir: str) -> str:
    df_s, meta_s = _load_epw(epw_sungrove)
    df_b, meta_b = _load_epw(epw_borealis)

    alt_bins_s, az_bins_s, frac_s = _bin_mask_fraction(df_s, meta_s, weight_heat=0.2)
    alt_bins_b, az_bins_b, frac_b = _bin_mask_fraction(df_b, meta_b, weight_heat=1.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, frac, title in [
        (axes[0], frac_s, "Sungrove (Cooling-dominant)"),
        (axes[1], frac_b, "Borealis (Heating-dominant)"),
    ]:
        im = ax.pcolormesh(az_bins_s, alt_bins_s, frac, cmap="magma", vmin=0, vmax=1, shading="auto")
        ax.set_title(title)
        ax.set_xlabel("Azimuth rel. South (deg)")
        ax.set_ylabel("Solar altitude (deg)")
        ax.set_ylim(0, 90)
        ax.set_xlim(-180, 180)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9, label="Shading demand fraction")
    fig.suptitle("Figure 6-1  Mask_target angle-domain comparison", y=1.02, fontsize=12)
    fig.tight_layout()

    out_path = os.path.join(outdir, "6-1.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _select_winter_week(df: pd.DataFrame) -> pd.DataFrame:
    winter = df[df.index.month.isin([12, 1, 2])].copy()
    if winter.empty:
        return df.iloc[: 24 * 7]
    daily = winter["Temp"].resample("D").mean()
    if len(daily) < 7:
        return winter
    roll = daily.rolling(7).mean().dropna()
    start_day = roll.idxmin()
    start_loc = winter.index.get_indexer([start_day], method="nearest")[0]
    start_loc = max(0, int(start_loc))
    end_loc = start_loc + 24 * 7
    return winter.iloc[start_loc:end_loc]


def plot_6_2(epw_borealis: str, outdir: str) -> str:
    df, meta = _load_epw(epw_borealis)
    week = _select_winter_week(df)
    times = week.index

    solar_pos = calculate_solar_position(
        times=times,
        latitude=meta.get("latitude", 0.0),
        longitude=meta.get("longitude", 0.0),
        altitude=meta.get("altitude", 0.0),
    )
    zen = solar_pos["zenith"].to_numpy()
    az = solar_pos["azimuth"].to_numpy()

    dni = week["DNI"].to_numpy()
    dhi = week["DHI"].to_numpy()
    ghi = week["GHI"].to_numpy()

    direct, diffuse, reflected = tilted_irradiance(
        surface_tilt=90.0,
        surface_azimuth=180.0,
        solar_zenith=zen,
        solar_azimuth=az,
        dni=dni,
        dhi=dhi,
        ghi=ghi,
        albedo=0.2,
    )
    total_irr = direct + diffuse + reflected
    shgc = 0.55
    window_area = 40.0
    q_sol = shgc * window_area * total_irr

    q_solar_a, q_solar_m = split_solar_to_air_and_mass(q_sol, fraction_to_air=0.35)
    q_internal = np.full_like(q_sol, 2500.0)

    temp_air, temp_mass, q_mass = simulate_2R2C(
        temp_outdoor=week["Temp"].to_numpy(),
        q_solar_a=q_solar_a,
        q_solar_m=q_solar_m,
        q_internal=q_internal,
        h_in=3.0,
        h_out=25.0,
        c_air=1.5e6,
        c_mass=2.0e8,
        r_env=0.35,
        r_am=0.15,
        area=240.0,
    )

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(times, temp_air, label="T_a (air)", color="#3182bd")
    ax1.plot(times, temp_mass, label="T_m (mass)", color="#e6550d")
    ax1.set_ylabel("Temperature (C)")
    ax1.set_title("Figure 6-2  Winter representative week (Borealis)")
    ax1.grid(True, alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(times, q_sol, label="Q_sol", color="#31a354", alpha=0.7)
    ax2.plot(times, q_mass, label="Q_mass", color="#756bb1", alpha=0.7)
    ax2.set_ylabel("Heat flow (W)")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left", ncol=2)
    fig.tight_layout()

    out_path = os.path.join(outdir, "6-2.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _occupancy_mask(times: pd.DatetimeIndex) -> np.ndarray:
    hours = times.hour
    weekdays = times.weekday
    weekday_mask = (weekdays < 5) & (hours >= 8) & (hours < 18)
    weekend_mask = (weekdays >= 5) & (hours >= 10) & (hours < 16)
    return weekday_mask | weekend_mask


def plot_6_3(epw_borealis: str, outdir: str) -> str:
    df, meta = _load_epw(epw_borealis)
    month_df = df[df.index.month == 1].copy()
    if month_df.empty:
        month_df = df.iloc[: 24 * 31].copy()

    times = month_df.index
    occ = _occupancy_mask(times)

    solar_pos = calculate_solar_position(
        times=times,
        latitude=meta.get("latitude", 0.0),
        longitude=meta.get("longitude", 0.0),
        altitude=meta.get("altitude", 0.0),
    )
    zen = solar_pos["zenith"].to_numpy()
    az = solar_pos["azimuth"].to_numpy()
    cos_theta = incidence_cos_window(az, zen, window_azimuth=180.0, window_tilt=90.0)

    vt = 0.65
    e_dir_in = month_df["DNI"].to_numpy() * cos_theta * vt

    s_open = np.ones_like(e_dir_in)
    glare = e_dir_in > 1000.0
    s_open[(occ) & glare] = 0.2

    days = month_df.index.day
    day_count = int(days.max())
    occ_map = np.full((24, day_count), np.nan)
    s_map = np.full((24, day_count), np.nan)

    for t, day, hour, o, s in zip(times, days, times.hour, occ, s_open):
        occ_map[hour, day - 1] = 1.0 if o else 0.0
        s_map[hour, day - 1] = s

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    im0 = axes[0].imshow(occ_map, aspect="auto", origin="lower", cmap="Greys", vmin=0, vmax=1)
    axes[0].set_title("Occupancy (Jan)")
    axes[0].set_xlabel("Day")
    axes[0].set_ylabel("Hour")

    im1 = axes[1].imshow(s_map, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("Shading openness $S_b$")
    axes[1].set_xlabel("Day")

    fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.9, label="S_b (0=closed,1=open)")
    fig.suptitle("Figure 6-3  Borealis winter shading openness", y=1.02, fontsize=12)
    fig.tight_layout()

    out_path = os.path.join(outdir, "6-3.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_6_4(outdir: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")

    # Left: Sungrove summer shading priority
    ax = axes[0]
    ax.add_patch(plt.Rectangle((1, 1), 4, 4, fill=False, lw=2))
    ax.add_patch(plt.Rectangle((4.8, 3.8), 2.5, 0.3, color="#555555"))
    ax.add_patch(plt.Rectangle((4.8, 2.0), 0.2, 1.8, color="#999999"))
    ax.plot([8, 5], [5, 3.7], color="#e34a33", lw=2)
    ax.plot([8, 5], [4, 3.7], color="#e34a33", lw=2)
    ax.text(1.0, 5.4, "Sungrove (Summer)", fontsize=10, weight="bold")
    ax.text(5.1, 4.3, "Overhang", fontsize=8)
    ax.text(6.8, 4.6, "High sun\nblocked", fontsize=8, color="#e34a33")

    # Right: Borealis winter sun + thermal mass
    ax = axes[1]
    ax.add_patch(plt.Rectangle((1, 1), 4, 4, fill=False, lw=2))
    ax.add_patch(plt.Rectangle((4.8, 2.0), 0.2, 1.8, color="#999999"))
    ax.add_patch(plt.Rectangle((2.0, 1.0), 1.2, 0.6, color="#a1d99b"))
    ax.plot([8, 2.6], [2, 1.2], color="#e34a33", lw=2)
    ax.plot([8, 2.8], [2.6, 1.4], color="#e34a33", lw=2)
    ax.annotate("L_pen", xy=(3.0, 1.5), xytext=(4.2, 2.2),
                arrowprops=dict(arrowstyle="->", lw=1), fontsize=8)
    ax.annotate("D_mass", xy=(2.6, 1.2), xytext=(1.2, 2.2),
                arrowprops=dict(arrowstyle="->", lw=1), fontsize=8)
    ax.text(1.0, 5.4, "Borealis (Winter)", fontsize=10, weight="bold")
    ax.text(2.0, 0.6, "Thermal mass", fontsize=8)
    ax.text(6.8, 2.6, "Low sun\nadmitted", fontsize=8, color="#e34a33")
    ax.text(2.1, 1.7, "eta_mass up", fontsize=8)

    fig.suptitle("Figure 6-4  Section schematic comparison", y=1.02, fontsize=12)
    fig.tight_layout()

    out_path = os.path.join(outdir, "6-4.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Model 3 figures (Section 6).")
    parser.add_argument(
        "--epw_sungrove",
        default=os.path.join("data", "epw", "HKG_NT_Lau.Fau.Shan.450350_TMYx.2009-2023.epw"),
        help="EPW file for Sungrove (HKG).",
    )
    parser.add_argument(
        "--epw_borealis",
        default=os.path.join("data", "epw", "NOR_OS_Oslo.Blindern.014920_TMYx.2009-2023.epw"),
        help="EPW file for Borealis (Oslo).",
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join("results", "model3_figs"),
        help="Output directory for Model 3 figures.",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    plot_6_1(args.epw_sungrove, args.epw_borealis, args.outdir)
    plot_6_2(args.epw_borealis, args.outdir)
    plot_6_3(args.epw_borealis, args.outdir)
    plot_6_4(args.outdir)
    print(f"Model 3 figures saved to: {args.outdir}")


if __name__ == "__main__":
    main()

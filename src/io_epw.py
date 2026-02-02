#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IO module for reading EnergyPlus Weather (EPW) files.

This module provides functions to read EPW files and extract relevant data
for building performance simulation, including solar irradiance, temperature,
and other weather parameters.
"""

import csv
import pandas as pd
import pvlib
from typing import Tuple, Optional, Dict, Iterable, List


def read_epw(file_path: str) -> Tuple[pd.DataFrame, dict]:
    """
    Read an EnergyPlus Weather (EPW) file and extract relevant data.

    Args:
        file_path: Path to the EPW file.

    Returns:
        A tuple containing:
        - A pandas DataFrame with the following columns:
            - 'datetime': Datetime index
            - 'DNI': Direct Normal Irradiance (W/m²)
            - 'DHI': Diffuse Horizontal Irradiance (W/m²)
            - 'GHI': Global Horizontal Irradiance (W/m²)
            - 'Temp': Dry bulb temperature (°C)
        - A dictionary with metadata about the location.

    Notes:
        Uses pvlib's read_epw function and then selects only the required columns.
        This implementation follows the requirements specified in Section 5.4 of the v3 document.
    """
    # Read EPW file using pvlib
    data, meta = pvlib.iotools.read_epw(file_path)
    
    # Print available columns for debugging
    print(f"Available columns in EPW data: {list(data.columns)}")
    
    # Extract only the required columns with fallback for different naming conventions
    df = pd.DataFrame(index=data.index)
    df['datetime'] = data.index
    
    # Handle different possible column names for irradiance data
    if 'DNI' in data.columns:
        df['DNI'] = data['DNI']
    elif 'dni' in data.columns:
        df['DNI'] = data['dni']
    else:
        # If no direct normal irradiance found, calculate from GHI and DHI if available
        if 'GHI' in data.columns and 'DHI' in data.columns:
            df['DNI'] = data['GHI'] - data['DHI']
        elif 'ghi' in data.columns and 'dhi' in data.columns:
            df['DNI'] = data['ghi'] - data['dhi']
        else:
            df['DNI'] = 0.0
    
    if 'DHI' in data.columns:
        df['DHI'] = data['DHI']
    elif 'dhi' in data.columns:
        df['DHI'] = data['dhi']
    else:
        df['DHI'] = 0.0
    
    if 'GHI' in data.columns:
        df['GHI'] = data['GHI']
    elif 'ghi' in data.columns:
        df['GHI'] = data['ghi']
    else:
        df['GHI'] = 0.0
    
    if 'temp_air' in data.columns:
        df['Temp'] = data['temp_air']
    elif 'Temperature' in data.columns:
        df['Temp'] = data['Temperature']
    elif 'temp' in data.columns:
        df['Temp'] = data['temp']
    else:
        df['Temp'] = 20.0  # Default temperature

    # Optional: keep humidity for climate feature extraction
    if 'relative_humidity' in data.columns:
        df['relative_humidity'] = data['relative_humidity']
    elif 'RH' in data.columns:
        df['relative_humidity'] = data['RH']
    
    return df, meta


EPW_COLUMN_INDEX = {
    "year": 0,
    "month": 1,
    "day": 2,
    "hour": 3,
    "minute": 4,
    "data_source": 5,
    "dry_bulb": 6,
    "dew_point": 7,
    "relative_humidity": 8,
    "atmospheric_pressure": 9,
    "extraterrestrial_horizontal": 10,
    "extraterrestrial_direct": 11,
    "horizontal_infrared": 12,
    "GHI": 13,
    "DNI": 14,
    "DHI": 15,
    "global_horizontal_illuminance": 16,
    "direct_normal_illuminance": 17,
    "diffuse_horizontal_illuminance": 18,
    "zenith_luminance": 19,
    "wind_direction": 20,
    "wind_speed": 21,
    "total_sky_cover": 22,
    "opaque_sky_cover": 23,
    "visibility": 24,
    "ceiling_height": 25,
    "present_weather_observation": 26,
    "present_weather_codes": 27,
    "precipitable_water": 28,
    "aerosol_optical_depth": 29,
    "snow_depth": 30,
    "days_since_last_snow": 31,
    "albedo": 32,
    "liquid_precipitation_depth": 33,
    "liquid_precipitation_quantity": 34,
}


def read_epw_raw(file_path: str) -> Tuple[List[str], List[List[str]]]:
    """
    Read EPW file as raw header lines and data rows.

    Args:
        file_path: Path to EPW file.

    Returns:
        header_lines: List of header lines (including DATA PERIODS line).
        data_rows: List of data rows (list of strings per row).
    """
    header_lines: List[str] = []
    data_rows: List[List[str]] = []
    in_data = False
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not in_data:
                header_lines.append(line.rstrip("\n"))
                if line.startswith("DATA PERIODS"):
                    in_data = True
                continue
            row = next(csv.reader([line.rstrip("\n")]))
            data_rows.append(row)
    return header_lines, data_rows


def write_epw_from_rows(
    header_lines: Iterable[str],
    data_rows: Iterable[Iterable[str]],
    output_path: str
) -> None:
    """
    Write EPW file from header lines and data rows.

    Args:
        header_lines: Iterable of header lines (strings).
        data_rows: Iterable of data rows (iterables of strings).
        output_path: Output EPW path.
    """
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        for line in header_lines:
            f.write(f"{line}\n")
        writer = csv.writer(f, lineterminator="\n")
        for row in data_rows:
            writer.writerow(row)


def write_epw_with_updates(
    base_epw_path: str,
    output_epw_path: str,
    updates: Dict[str, Iterable[float]],
    value_formats: Optional[Dict[str, str]] = None
) -> None:
    """
    Write an EPW file by updating selected columns from a base EPW.

    Args:
        base_epw_path: Path to the source EPW file.
        output_epw_path: Destination EPW file path.
        updates: Mapping from column names (e.g., "Temp", "DNI") to sequences.
        value_formats: Optional per-column format strings.
    """
    header_lines, data_rows = read_epw_raw(base_epw_path)
    row_count = len(data_rows)

    default_formats = {
        "Temp": "{:.2f}",
        "dry_bulb": "{:.2f}",
        "DNI": "{:.0f}",
        "DHI": "{:.0f}",
        "GHI": "{:.0f}",
    }
    if value_formats:
        default_formats.update(value_formats)

    for col_name, values in updates.items():
        if col_name == "Temp":
            col_key = "dry_bulb"
        else:
            col_key = col_name
        if col_key not in EPW_COLUMN_INDEX:
            continue
        col_idx = EPW_COLUMN_INDEX[col_key]
        values = list(values)
        if len(values) != row_count:
            raise ValueError(f"EPW update length mismatch for {col_name}: {len(values)} vs {row_count}")
        fmt = default_formats.get(col_name, "{:.2f}")
        for i, val in enumerate(values):
            data_rows[i][col_idx] = fmt.format(float(val))

    write_epw_from_rows(header_lines, data_rows, output_epw_path)

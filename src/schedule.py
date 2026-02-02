#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schedule helpers for occupancy patterns.

This module centralizes occupancy schedules so that Task 4 can reuse them
without affecting Tasks 1-3.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _to_datetime_index(times: Iterable) -> pd.DatetimeIndex:
    if isinstance(times, pd.DatetimeIndex):
        return times
    return pd.DatetimeIndex(times)


def occupancy_mask(
    times: Iterable,
    weekday_start: int,
    weekday_end: int,
    weekend_start: int,
    weekend_end: int,
    holiday_dates: Optional[Iterable[pd.Timestamp]] = None,
) -> np.ndarray:
    """
    Build an occupancy mask for a recurring daily schedule.

    Args:
        times: Iterable of timestamps.
        weekday_start: Hour (0-23) for weekday occupancy start (inclusive).
        weekday_end: Hour (0-23) for weekday occupancy end (exclusive).
        weekend_start: Hour (0-23) for weekend occupancy start (inclusive).
        weekend_end: Hour (0-23) for weekend occupancy end (exclusive).
        holiday_dates: Optional iterable of dates to treat as unoccupied.

    Returns:
        Boolean NumPy array, True for occupied hours.
    """
    dt_index = _to_datetime_index(times)
    hours = dt_index.hour
    weekdays = dt_index.weekday  # 0=Mon

    weekday_mask = (weekdays < 5) & (hours >= weekday_start) & (hours < weekday_end)
    weekend_mask = (weekdays >= 5) & (hours >= weekend_start) & (hours < weekend_end)
    mask = weekday_mask | weekend_mask

    if holiday_dates:
        holiday_dates = {pd.Timestamp(d).date() for d in holiday_dates}
        mask &= ~np.array([ts.date() in holiday_dates for ts in dt_index])

    return mask


def student_union_schedule(
    times: Iterable,
    weekday_start: int = 8,
    weekday_end: int = 22,
    weekend_start: int = 10,
    weekend_end: int = 20,
) -> np.ndarray:
    """
    Default occupancy schedule for Student Union: day + evening.

    Args:
        times: Iterable of timestamps.
        weekday_start: Hour for weekday occupancy start.
        weekday_end: Hour for weekday occupancy end.
        weekend_start: Hour for weekend occupancy start.
        weekend_end: Hour for weekend occupancy end.

    Returns:
        Boolean NumPy array, True for occupied hours.
    """
    return occupancy_mask(
        times=times,
        weekday_start=weekday_start,
        weekday_end=weekday_end,
        weekend_start=weekend_start,
        weekend_end=weekend_end,
    )

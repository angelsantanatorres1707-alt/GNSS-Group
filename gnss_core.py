"""Pure GNSS analysis and map-selection helpers used by the notebook.

This module deliberately contains no widget, map, file-system, or network work.
Keeping these rules separate makes them fast to test and safe to reuse.
"""

import ast
import math
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from geopy.distance import geodesic


StationKey = tuple[str, str]


@dataclass(frozen=True)
class MapRenderPlan:
    """Describes the stations, vectors, and circles that a map shell should draw."""

    marker_keys: tuple[StationKey, ...]
    vector_keys: tuple[StationKey, ...]
    circle_keys: tuple[StationKey, ...]


def number_list(text: str, default: Sequence[float]) -> list[float]:
    """Reads a numeric tuple/list or returns a copy of the fallback values."""
    try:
        values = list(ast.literal_eval(text))
        if len(values) != len(default):
            raise ValueError
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise ValueError
        return [float(value) for value in values]
    except (ValueError, SyntaxError, TypeError):
        return list(default)


def filter_sigmas(
    dataframe: pd.DataFrame,
    columns: Sequence[int],
    limits: Sequence[float],
    conversion: float = 1.0,
) -> pd.DataFrame:
    """Returns rows whose enabled uncertainty components are below their limits."""
    filtered = dataframe
    for column, limit in zip(columns, limits):
        if limit > 0:
            filtered = filtered[filtered.iloc[:, column] < limit / conversion]
    return filtered


def fit_ts(xdata: np.ndarray, ydata: np.ndarray, sig: np.ndarray) -> tuple:
    """Fits a weighted linear GNSS trend and returns fit, velocity, and statistics."""
    years = xdata / 365.25
    weights = 1.0 / sig**2
    point_count = int(xdata.size)
    design = np.transpose([np.ones(point_count), years])
    normal_equations = np.matmul(np.transpose(design) * weights, design)
    right_hand_side = np.matmul(np.transpose(design) * weights, np.transpose(ydata))
    covariance = np.linalg.inv(normal_equations)
    estimate = np.matmul(covariance, right_hand_side)
    residuals = ydata - np.matmul(design, estimate)
    chi = np.sqrt(np.dot(np.transpose(residuals), residuals * weights) / (point_count - 2))
    wrms = np.sqrt(point_count / np.sum(weights)) * chi
    fit = np.flip(estimate)
    return fit, [estimate[1], np.sqrt(covariance[1, 1])], [wrms, chi, point_count]


def detrended(xdata: np.ndarray, ydata: np.ndarray, sig: np.ndarray) -> tuple:
    """Removes the weighted linear trend from one GNSS component."""
    fit, velocity, statistics = fit_ts(xdata, ydata, sig)
    return ydata - (fit[0] * xdata / 365.25 + fit[1]), velocity, statistics


def vec_add(first: Sequence[float], second: Sequence[float]) -> list[float]:
    """Adds two equally sized vectors component by component."""
    return [first[index] + second[index] for index in range(len(first))]


def vec_sub(first: Sequence[float], second: Sequence[float]) -> list[float]:
    """Subtracts the second vector from the first component by component."""
    return [first[index] - second[index] for index in range(len(first))]


def velocity_endpoint(center: Sequence[float], direction: Sequence[float], scale: float) -> list[float]:
    """Calculates the geographic endpoint for a North/East velocity vector."""
    north, east = direction[0], direction[1]
    length = math.hypot(north, east) * scale
    bearing = math.degrees(math.atan2(east, north))
    endpoint = geodesic(kilometers=length).destination(center, bearing)
    return [endpoint.latitude, endpoint.longitude]


def remove_brac(label: str) -> tuple[str, str]:
    """Splits a display label such as ``P123 (UNR)`` into station and source."""
    station, source = label.strip().rsplit("(", 1)
    return station.strip().upper(), source.rstrip(")").strip().upper()


def parse_bulk_selection(text: str) -> set[int]:
    """Parses individual numbers and inclusive/exclusive bracketed ranges."""
    selected_numbers: set[int] = set()

    def add_range(match: re.Match[str]) -> str:
        opening, start, end, closing = match.groups()
        first_number = int(start) + (opening == "(")
        last_number = int(end) - (closing == ")")
        if first_number <= last_number:
            selected_numbers.update(range(first_number, last_number + 1))
        return " "

    remaining_text = re.sub(r"(\[|\()\s*(\d+)\s*,\s*(\d+)\s*(\]|\))", add_range, text)
    selected_numbers.update(int(number) for number in re.findall(r"\d+", remaining_text))
    return selected_numbers


def parse_map_selection(options: Iterable[str]) -> dict[StationKey, int]:
    """Converts selected widget labels into unique source/station radius records."""
    selections: dict[StationKey, int] = {}
    for option in options:
        station, source, radius_text = option.split(", ")
        selections[(source, station)] = int(radius_text.removesuffix("km"))
    return selections


def build_map_render_plan(
    selections: Mapping[StationKey, int],
    nearby_records: Mapping[StationKey, Sequence[Sequence[object]]],
    plot_vectors: bool,
    decimation: int,
) -> MapRenderPlan:
    """Builds a deduplicated map plan without creating map layers.

    ``nearby_records`` contains rows shaped like
    ``(station_id, latitude, longitude, distance_km)`` for each selected station.
    """
    marker_keys: dict[StationKey, None] = {}
    vector_keys: dict[StationKey, None] = {}
    circle_keys: list[StationKey] = []
    step = max(1, int(decimation))

    for key, radius in selections.items():
        source, station = key
        marker_keys[key] = None
        if plot_vectors:
            vector_keys[key] = None
        if radius <= 0:
            continue

        circle_keys.append(key)
        neighbors = [row for row in nearby_records.get(key, ()) if row[0] != station]
        for index, row in enumerate(neighbors):
            neighbor_key = (source, str(row[0]))
            marker_keys[neighbor_key] = None
            if plot_vectors and index % step == 0:
                vector_keys[neighbor_key] = None

    return MapRenderPlan(
        marker_keys=tuple(marker_keys),
        vector_keys=tuple(vector_keys),
        circle_keys=tuple(circle_keys),
    )

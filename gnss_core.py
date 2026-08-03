"""Pure GNSS analysis and map-selection helpers used by the notebook.

This module deliberately contains no widget, map, file-system, or network work.
Keeping these rules separate makes them fast to test and safe to reuse.
"""

import ast
from html import escape as html_escape
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from geopy.distance import geodesic


StationKey = tuple[str, str]

# Leaflet/Google maps use Web Mercator tiles. At zoom 0, one 256 px tile
# spans this many metres at the equator; keeping the constant here makes the
# screen-space velocity scale explicit and testable.
WEB_MERCATOR_METERS_PER_PIXEL_ZOOM_0 = 156543.03392804097
VELOCITY_REFERENCE_ZOOM = 4
VELOCITY_REFERENCE_LATITUDE = 40.0


@dataclass(frozen=True)
class MapRenderPlan:
    """Describes the stations, vectors, and circles that a map shell should draw."""

    marker_keys: tuple[StationKey, ...]
    vector_keys: tuple[StationKey, ...]
    circle_keys: tuple[StationKey, ...]


@dataclass(frozen=True)
class VectorRenderSpec:
    """Immutable render data for one absolute velocity vector.

    The notebook creates the ipyleaflet ``DivIcon`` and ``Marker`` objects from
    this value for each map render.  Keeping those mutable widget objects out of
    the cache means a staged layer group or a newly-created map never shares a
    live Leaflet object with another owner.
    """

    org: str
    site: str
    location: tuple[float, float]
    velocity: tuple[float, float, float]
    sigma: tuple[float, float, float]
    siglim: tuple[float, float, float]
    up_color_factor: float
    velocity_scale: float
    color: str
    label: str | None
    svg: str
    icon_size: tuple[int, int]
    icon_anchor: tuple[int, int]
    label_html: str | None
    label_icon_size: tuple[int, int]
    label_icon_anchor: tuple[int, int]
    geometry_version: str


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


def web_mercator_meters_per_pixel(latitude: float, zoom: float) -> float:
    """Return Web Mercator metres represented by one screen pixel."""
    latitude_radians = math.radians(max(-85.05112878, min(85.05112878, float(latitude))))
    return (
        WEB_MERCATOR_METERS_PER_PIXEL_ZOOM_0
        * math.cos(latitude_radians)
        / (2.0 ** float(zoom))
    )


def velocity_pixels_per_mm_per_year(
    scale_km_per_mm_per_year: float,
    latitude: float = VELOCITY_REFERENCE_LATITUDE,
    reference_zoom: float = VELOCITY_REFERENCE_ZOOM,
) -> float:
    """Convert the notebook's km/(mm/yr) scale into reference-zoom pixels."""
    metres_per_pixel = web_mercator_meters_per_pixel(latitude, reference_zoom)
    return float(scale_km_per_mm_per_year) * 1000.0 / metres_per_pixel


def velocity_arrow_geometry(
    direction: Sequence[float],
    scale_km_per_mm_per_year: float,
    latitude: float = VELOCITY_REFERENCE_LATITUDE,
    reference_zoom: float = VELOCITY_REFERENCE_ZOOM,
    arrowhead_length_px: float = 8.0,
    arrowhead_half_width_px: float = 4.0,
) -> dict[str, float]:
    """Return pixel geometry for an SVG arrow anchored at a station.

    ``direction`` is North, East, Up. SVG's positive y axis points down, so
    North is negated when constructing the arrow in the notebook. Arrowhead
    dimensions are fixed pixels and therefore do not change with map zoom.
    """
    north, east = float(direction[0]), float(direction[1])
    horizontal_speed = math.hypot(north, east)
    length_px = horizontal_speed * velocity_pixels_per_mm_per_year(
        scale_km_per_mm_per_year, latitude, reference_zoom
    )
    if horizontal_speed:
        unit_x = east / horizontal_speed
        unit_y = -north / horizontal_speed
    else:
        unit_x, unit_y = 0.0, 0.0
    head_length = float(arrowhead_length_px)
    half_width = float(arrowhead_half_width_px)
    canvas_length = max(1.0, length_px)
    padding = max(head_length, half_width) + 2.0
    return {
        "length_px": length_px,
        "canvas_width_px": canvas_length + 2.0 * padding,
        "canvas_height_px": 2.0 * padding,
        "anchor_x_px": padding,
        "anchor_y_px": padding,
        "unit_x": unit_x,
        "unit_y": unit_y,
        "arrowhead_length_px": head_length,
        "arrowhead_half_width_px": half_width,
    }


VECTOR_RENDER_GEOMETRY_VERSION = "absolute-svg-v1"
# Keep enough immutable specs for the notebook's ~2k-station field renders,
# while retaining a hard bound for long-lived kernels and repeated styles.
VECTOR_RENDER_CACHE_SIZE = 8192


def _vector_svg_markup(
    geometry: Mapping[str, float],
    color: str,
) -> str:
    """Build the transparent, pointer-through SVG used by a DivIcon."""
    width = geometry["canvas_width_px"]
    height = geometry["canvas_height_px"]
    x0 = geometry["anchor_x_px"]
    y0 = geometry["anchor_y_px"]
    ux = geometry["unit_x"]
    uy = geometry["unit_y"]
    length = geometry["length_px"]
    head = geometry["arrowhead_length_px"]
    half = geometry["arrowhead_half_width_px"]
    x1, y1 = x0 + ux * length, y0 + uy * length
    bx, by = x1 - ux * head, y1 - uy * head
    px, py = -uy, ux
    left = (bx + px * half, by + py * half)
    right = (bx - px * half, by - py * half)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width:.1f}" height="{height:.1f}" viewBox="0 0 {width:.1f} {height:.1f}" '
        'overflow="visible" aria-label="velocity arrow" style="background:transparent;'
        'pointer-events:none">'
        f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
        f'stroke="{color}" stroke-width="2" stroke-linecap="round"/>'
        f'<polygon points="{x1:.2f},{y1:.2f} {left[0]:.2f},{left[1]:.2f} '
        f'{right[0]:.2f},{right[1]:.2f}" fill="{color}"/>'
        '</svg>'
    )


@lru_cache(maxsize=VECTOR_RENDER_CACHE_SIZE)
def _absolute_vector_render_spec_cached(
    org: str,
    site: str,
    location: tuple[float, float],
    velocity: tuple[float, float, float],
    sigma: tuple[float, float, float],
    siglim: tuple[float, float, float],
    up_color_factor: float,
    velocity_scale: float,
    color: str,
    label: str | None = None,
    latitude: float = VELOCITY_REFERENCE_LATITUDE,
    reference_zoom: float = VELOCITY_REFERENCE_ZOOM,
    arrowhead_length_px: float = 8.0,
    arrowhead_half_width_px: float = 4.0,
    geometry_version: str = VECTOR_RENDER_GEOMETRY_VERSION,
    mode: str = "absolute",
) -> VectorRenderSpec:
    """Return a bounded immutable SVG render spec for an absolute vector.

    All source and control inputs are part of the cache key, including sigma
    limits and the up-colour factor.  Relative vectors deliberately bypass this
    function in the notebook because their reference station can change between
    renders.  Sequence inputs are normalized to tuples before lookup, while the
    public ``cache_info`` and ``cache_clear`` hooks retain the normal
    ``functools.lru_cache`` API.
    """
    if mode != "absolute":
        raise ValueError("absolute_vector_render_spec only accepts mode='absolute'")
    location = (float(location[0]), float(location[1]))
    velocity = tuple(float(value) for value in velocity[:3])
    sigma = tuple(float(value) for value in sigma[:3])
    siglim = tuple(float(value) for value in siglim[:3])
    if len(velocity) != 3 or len(sigma) != 3 or len(siglim) != 3:
        raise ValueError("velocity, sigma, and siglim must contain three values")
    geometry = velocity_arrow_geometry(
        velocity,
        float(velocity_scale),
        float(latitude),
        reference_zoom=float(reference_zoom),
        arrowhead_length_px=float(arrowhead_length_px),
        arrowhead_half_width_px=float(arrowhead_half_width_px),
    )
    width = int(round(geometry["canvas_width_px"]))
    height = int(round(geometry["canvas_height_px"]))
    anchor = (int(round(geometry["anchor_x_px"])), int(round(geometry["anchor_y_px"])))
    label_text = None if label in (None, "None", "") else str(label)
    label_html = None
    if label_text is not None:
        # Keep labels in a separate, transparent class so their marker cannot
        # recreate the default white DivIcon card.
        label_html = (
            '<div style="background:transparent;border:0;'
            'padding:0;margin:0;pointer-events:none;white-space:nowrap;'
            'font-size:12pt;line-height:1.1;transform:translate(8px,-18px)">'
            f"{html_escape(label_text)}</div>"
        )
    return VectorRenderSpec(
        org=str(org),
        site=str(site),
        location=location,
        velocity=velocity,
        sigma=sigma,
        siglim=siglim,
        up_color_factor=float(up_color_factor),
        velocity_scale=float(velocity_scale),
        color=str(color),
        label=label_text,
        svg=_vector_svg_markup(geometry, str(color)),
        icon_size=(width, height),
        icon_anchor=anchor,
        label_html=label_html,
        label_icon_size=(250, 36),
        label_icon_anchor=(0, 0),
        geometry_version=str(geometry_version),
    )


def absolute_vector_render_spec(
    org: str,
    site: str,
    location: Sequence[float],
    velocity: Sequence[float],
    sigma: Sequence[float],
    siglim: Sequence[float],
    up_color_factor: float,
    velocity_scale: float,
    color: str,
    label: str | None = None,
    latitude: float = VELOCITY_REFERENCE_LATITUDE,
    reference_zoom: float = VELOCITY_REFERENCE_ZOOM,
    arrowhead_length_px: float = 8.0,
    arrowhead_half_width_px: float = 4.0,
    geometry_version: str = VECTOR_RENDER_GEOMETRY_VERSION,
    mode: str = "absolute",
) -> VectorRenderSpec:
    """Normalize sequence inputs before looking up the hashable LRU key."""
    if mode != "absolute":
        raise ValueError("absolute_vector_render_spec only accepts mode='absolute'")
    return _absolute_vector_render_spec_cached(
        str(org), str(site), tuple(float(value) for value in location[:2]),
        tuple(float(value) for value in velocity[:3]), tuple(float(value) for value in sigma[:3]),
        tuple(float(value) for value in siglim[:3]), float(up_color_factor), float(velocity_scale),
        str(color), label, float(latitude), float(reference_zoom), float(arrowhead_length_px),
        float(arrowhead_half_width_px), str(geometry_version), mode,
    )


# Preserve the normal functools cache introspection/clear API on the forgiving
# sequence-normalizing facade.  There is still exactly one bounded cache.
absolute_vector_render_spec.cache_info = _absolute_vector_render_spec_cached.cache_info
absolute_vector_render_spec.cache_clear = _absolute_vector_render_spec_cached.cache_clear
absolute_vector_render_spec.cache_parameters = _absolute_vector_render_spec_cached.cache_parameters


# A descriptive alias keeps the cache hook discoverable without introducing a
# second cache (which could otherwise double memory and diverge statistics).
cached_absolute_vector_render_spec = absolute_vector_render_spec


def relative_vector_render_spec(
    location: Sequence[float],
    velocity: Sequence[float],
    sigma: Sequence[float],
    siglim: Sequence[float],
    up_color_factor: float,
    velocity_scale: float,
    color: str,
    label: str | None = None,
    latitude: float = VELOCITY_REFERENCE_LATITUDE,
    reference_zoom: float = VELOCITY_REFERENCE_ZOOM,
    arrowhead_length_px: float = 8.0,
    arrowhead_half_width_px: float = 4.0,
    geometry_version: str = VECTOR_RENDER_GEOMETRY_VERSION,
) -> VectorRenderSpec:
    """Build a relative render spec without touching the absolute LRU cache."""
    location_values = (float(location[0]), float(location[1]))
    velocity_values = tuple(float(value) for value in tuple(velocity)[:3])
    sigma_values = tuple(float(value) for value in tuple(sigma)[:3])
    siglim_values = tuple(float(value) for value in tuple(siglim)[:3])
    geometry = velocity_arrow_geometry(
        velocity_values, float(velocity_scale), float(latitude),
        reference_zoom=float(reference_zoom),
        arrowhead_length_px=float(arrowhead_length_px),
        arrowhead_half_width_px=float(arrowhead_half_width_px),
    )
    label_text = None if label in (None, "None", "") else str(label)
    label_html = None
    if label_text is not None:
        label_html = (
            '<div style="background:transparent;border:0;'
            'padding:0;margin:0;pointer-events:none;white-space:nowrap;'
            'font-size:12pt;line-height:1.1;transform:translate(8px,-18px)">'
            f"{html_escape(label_text)}</div>"
        )
    return VectorRenderSpec(
        org="relative", site="relative", location=location_values,
        velocity=velocity_values, sigma=sigma_values, siglim=siglim_values,
        up_color_factor=float(up_color_factor), velocity_scale=float(velocity_scale),
        color=str(color), label=label_text, svg=_vector_svg_markup(geometry, str(color)),
        icon_size=(int(round(geometry["canvas_width_px"])), int(round(geometry["canvas_height_px"]))),
        icon_anchor=(int(round(geometry["anchor_x_px"])), int(round(geometry["anchor_y_px"]))),
        label_html=label_html, label_icon_size=(250, 36), label_icon_anchor=(0, 0),
        geometry_version=str(geometry_version),
    )


def clear_absolute_vector_render_cache() -> None:
    """Invalidate immutable vector specs after source metadata/control changes."""
    absolute_vector_render_spec.cache_clear()


def velocity_guide_metrics(
    guide_velocity_mm_per_year: float,
    scale_km_per_mm_per_year: float,
    latitude: float,
    current_zoom: float,
    reference_zoom: float = VELOCITY_REFERENCE_ZOOM,
    reference_latitude: float = VELOCITY_REFERENCE_LATITUDE,
) -> dict[str, float]:
    """Return fixed screen bar length and its current geographic equivalent.

    The pixel bar uses the same fixed reference latitude as station arrows;
    only the reported geographic equivalent uses the current map center.
    """
    bar_px = float(guide_velocity_mm_per_year) * velocity_pixels_per_mm_per_year(
        scale_km_per_mm_per_year, reference_latitude, reference_zoom
    )
    current_km = bar_px * web_mercator_meters_per_pixel(latitude, current_zoom) / 1000.0
    return {"bar_px": bar_px, "current_km": current_km}


def format_display_number(value: object, missing: str = "—") -> str:
    """Format a scalar for UI output with exactly two decimal places."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return missing
    if not math.isfinite(number):
        return missing
    return f"{number:.2f}"


def format_neu_vector(values: object, missing: str = "—") -> str:
    """Format a NEU vector as comma-separated, hundredths-precision text."""
    if values is None or isinstance(values, (str, bytes)):
        return missing
    try:
        values = list(values)
    except TypeError:
        return missing
    if len(values) < 3:
        return missing
    formatted = [format_display_number(value, missing) for value in values[:3]]
    if any(item == missing for item in formatted):
        return missing
    return ", ".join(formatted)


def neighbor_velocity_row(
    station: str,
    source: str,
    distance_km: object,
    neighbor_info: Mapping[str, object] | None,
    reference_info: Mapping[str, object] | None,
) -> dict[str, object]:
    """Build one safe nearest-neighbor model shared by map and graph displays."""
    neighbor_info = neighbor_info or {}
    reference_info = reference_info or {}
    velocity = neighbor_info.get("velocity")
    sigma = neighbor_info.get("velsig")
    reference_velocity = reference_info.get("velocity")
    try:
        velocity_values = [float(value) for value in list(velocity)[:3]]
        if len(velocity_values) != 3 or not all(math.isfinite(value) for value in velocity_values):
            raise ValueError
    except (TypeError, ValueError):
        velocity_values = None
    try:
        sigma_values = [float(value) for value in list(sigma)[:3]]
        if len(sigma_values) != 3 or not all(math.isfinite(value) for value in sigma_values):
            raise ValueError
    except (TypeError, ValueError):
        sigma_values = None
    if velocity_values is not None:
        try:
            reference_values = [float(value) for value in list(reference_velocity)[:3]]
            if len(reference_values) != 3 or not all(math.isfinite(value) for value in reference_values):
                raise ValueError
            differential = [value - reference_values[index] for index, value in enumerate(velocity_values)]
        except (TypeError, ValueError):
            differential = None
    else:
        differential = None
    return {
        "station": station,
        "source": source,
        "distance_km": distance_km,
        "velocity": velocity_values,
        "sigma": sigma_values,
        "differential": differential,
    }


def resolve_popup_owner(render_target: object, live_owner: object | None = None) -> object:
    """Choose the live layer owner for lazily-created popup layers."""
    return render_target if live_owner is None else live_owner


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

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
    # Ellipse inputs that were part of the cache key, plus the derived pixel
    # geometry, so a spec can be asserted on without re-parsing its SVG.
    # ``ellipse_confidence is None`` means "no ellipse", and the two derived
    # fields are then None as well.
    ellipse_covariance: tuple[float, float, float] | None = None
    ellipse_confidence: float | None = None
    ellipse_scale: float = 1.0
    ellipse_semi_axes_px: tuple[float, float] | None = None
    ellipse_rotation_deg: float | None = None


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


VECTOR_RENDER_GEOMETRY_VERSION = "absolute-svg-v2"
# Keep enough immutable specs for the notebook's ~2k-station field renders,
# while retaining a hard bound for long-lived kernels and repeated styles.
VECTOR_RENDER_CACHE_SIZE = 8192


def _vector_svg_markup(
    geometry: Mapping[str, float],
    color: str,
    ellipse: Mapping[str, float] | None = None,
    bounds: Mapping[str, float] | None = None,
) -> str:
    """Build the transparent, pointer-through SVG used by a DivIcon.

    ``bounds`` comes from :func:`vector_icon_bounds`; the caller passes the same
    dict it used for ``icon_size``/``icon_anchor`` so the drawing origin and the
    marker anchor cannot disagree.  The optional ellipse is the uncertainty of
    THIS arrow, so it is drawn at the arrow tip in the arrow's own pixel frame,
    never as a separate geographic layer that would agree at one zoom only.
    With ``ellipse=None`` the markup is byte-identical to the pre-ellipse trunk.
    """
    bounds = vector_icon_bounds(geometry, ellipse) if bounds is None else bounds
    width = bounds["canvas_width_px"]
    height = bounds["canvas_height_px"]
    x0 = bounds["anchor_x_px"]
    y0 = bounds["anchor_y_px"]
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
    ellipse_markup = ''
    aria_label = 'velocity arrow'
    if ellipse is not None:
        aria_label = 'velocity arrow and confidence ellipse'
        # Drawn last so a small ellipse is not buried under the arrowhead, and
        # unfilled so it never hides the vector it annotates.  pointer-events is
        # inherited from the svg style, so this element is click-through too.
        dash = (
            '' if ellipse["ellipse_scale"] == 1.0
            else f' stroke-dasharray="{ELLIPSE_EXAGGERATED_DASH}"'
        )
        ellipse_markup = (
            f'<ellipse cx="{x1:.2f}" cy="{y1:.2f}" '
            f'rx="{ellipse["semi_major_px"]:.2f}" ry="{ellipse["semi_minor_px"]:.2f}" '
            f'transform="rotate({ellipse["rotation_deg"]:.2f} {x1:.2f} {y1:.2f})" '
            f'fill="none" stroke="{color}" '
            f'stroke-width="{ELLIPSE_STROKE_WIDTH_PX:g}"{dash}/>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width:.1f}" height="{height:.1f}" viewBox="0 0 {width:.1f} {height:.1f}" '
        f'overflow="visible" aria-label="{aria_label}" style="background:transparent;'
        'pointer-events:none">'
        f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
        f'stroke="{color}" stroke-width="2" stroke-linecap="round"/>'
        f'<polygon points="{x1:.2f},{y1:.2f} {left[0]:.2f},{left[1]:.2f} '
        f'{right[0]:.2f},{right[1]:.2f}" fill="{color}"/>'
        f'{ellipse_markup}'
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
    ellipse_covariance: tuple[float, float, float] | None = None,
    ellipse_confidence: float | None = None,
    ellipse_scale: float = 1.0,
    geometry_version: str = VECTOR_RENDER_GEOMETRY_VERSION,
    mode: str = "absolute",
) -> VectorRenderSpec:
    """Return a bounded immutable SVG render spec for an absolute vector.

    All source and control inputs are part of the cache key, including sigma
    limits, the up-colour factor and the three ellipse inputs.  Omitting an
    ellipse control from the key would serve a stale spec: changing the
    confidence dropdown would hit the existing entry and redraw the previous
    ellipse while the UI reported the new level.  ``ellipse_confidence = None``
    is the canonical "ellipse off" value and reproduces the arrow-only markup
    exactly, so toggling the ellipse off is a cache hit rather than a new family
    of entries.  Relative vectors deliberately bypass this function in the
    notebook because their reference station can change between renders.
    Sequence inputs are normalized to tuples before lookup, while the public
    ``cache_info`` and ``cache_clear`` hooks retain the normal
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
    ellipse = velocity_ellipse_geometry(
        ellipse_covariance,
        ellipse_confidence,
        float(velocity_scale),
        float(ellipse_scale),
        float(latitude),
        float(reference_zoom),
    )
    bounds = vector_icon_bounds(geometry, ellipse)
    width = int(round(bounds["canvas_width_px"]))
    height = int(round(bounds["canvas_height_px"]))
    anchor = (int(round(bounds["anchor_x_px"])), int(round(bounds["anchor_y_px"])))
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
        svg=_vector_svg_markup(geometry, str(color), ellipse, bounds),
        icon_size=(width, height),
        icon_anchor=anchor,
        label_html=label_html,
        label_icon_size=(250, 36),
        label_icon_anchor=(0, 0),
        geometry_version=str(geometry_version),
        ellipse_covariance=ellipse_covariance,
        ellipse_confidence=ellipse_confidence,
        ellipse_scale=float(ellipse_scale),
        ellipse_semi_axes_px=(
            None if ellipse is None
            else (ellipse["semi_major_px"], ellipse["semi_minor_px"])
        ),
        ellipse_rotation_deg=None if ellipse is None else ellipse["rotation_deg"],
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
    ellipse_covariance: Sequence[float] | None = None,
    ellipse_confidence: float | None = None,
    ellipse_scale: float = 1.0,
    geometry_version: str = VECTOR_RENDER_GEOMETRY_VERSION,
    mode: str = "absolute",
) -> VectorRenderSpec:
    """Normalize sequence inputs before looking up the hashable LRU key.

    Everything is forwarded positionally, so a caller that passes the ellipse
    arguments by keyword and one that omits them land on the same key tuple.
    """
    if mode != "absolute":
        raise ValueError("absolute_vector_render_spec only accepts mode='absolute'")
    return _absolute_vector_render_spec_cached(
        str(org), str(site), tuple(float(value) for value in location[:2]),
        tuple(float(value) for value in velocity[:3]), tuple(float(value) for value in sigma[:3]),
        tuple(float(value) for value in siglim[:3]), float(up_color_factor), float(velocity_scale),
        str(color), label, float(latitude), float(reference_zoom), float(arrowhead_length_px),
        float(arrowhead_half_width_px), _normalized_ellipse_covariance(ellipse_covariance),
        None if ellipse_confidence is None else float(ellipse_confidence),
        float(ellipse_scale), str(geometry_version), mode,
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
    ellipse_covariance: Sequence[float] | None = None,
    ellipse_confidence: float | None = None,
    ellipse_scale: float = 1.0,
    geometry_version: str = VECTOR_RENDER_GEOMETRY_VERSION,
) -> VectorRenderSpec:
    """Build a relative render spec without touching the absolute LRU cache.

    ``ellipse_covariance`` describes the PLOTTED vector, which in relative mode is a
    difference, so the caller passes the combined covariance from
    :func:`velocity_ellipse_inputs` rather than the site's own velsig.
    """
    location_values = (float(location[0]), float(location[1]))
    velocity_values = tuple(float(value) for value in tuple(velocity)[:3])
    sigma_values = tuple(float(value) for value in tuple(sigma)[:3])
    siglim_values = tuple(float(value) for value in tuple(siglim)[:3])
    ellipse_covariance = _normalized_ellipse_covariance(ellipse_covariance)
    ellipse_confidence = None if ellipse_confidence is None else float(ellipse_confidence)
    geometry = velocity_arrow_geometry(
        velocity_values, float(velocity_scale), float(latitude),
        reference_zoom=float(reference_zoom),
        arrowhead_length_px=float(arrowhead_length_px),
        arrowhead_half_width_px=float(arrowhead_half_width_px),
    )
    ellipse = velocity_ellipse_geometry(
        ellipse_covariance, ellipse_confidence, float(velocity_scale),
        float(ellipse_scale), float(latitude), float(reference_zoom),
    )
    bounds = vector_icon_bounds(geometry, ellipse)
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
        color=str(color), label=label_text,
        svg=_vector_svg_markup(geometry, str(color), ellipse, bounds),
        icon_size=(int(round(bounds["canvas_width_px"])), int(round(bounds["canvas_height_px"]))),
        icon_anchor=(int(round(bounds["anchor_x_px"])), int(round(bounds["anchor_y_px"]))),
        label_html=label_html, label_icon_size=(250, 36), label_icon_anchor=(0, 0),
        geometry_version=str(geometry_version),
        ellipse_covariance=ellipse_covariance,
        ellipse_confidence=ellipse_confidence,
        ellipse_scale=float(ellipse_scale),
        ellipse_semi_axes_px=(
            None if ellipse is None
            else (ellipse["semi_major_px"], ellipse["semi_minor_px"])
        ),
        ellipse_rotation_deg=None if ellipse is None else ellipse["rotation_deg"],
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


# --- horizontal-velocity confidence ellipses ------------------------------
#
# Two-dimensional confidence levels for the horizontal velocity error ellipse.
# These are NOT the one-dimensional 68/95/99.7 numbers: a bivariate normal
# encloses 1 - exp(-k**2 / 2) of its probability inside the k-sigma contour, so
# k = 1, 2, 3 give 39.35%, 86.47% and 98.89%.  The three-sigma constant is
# 1 - exp(-9/2) = 0.98889...; writing 0.9989 instead yields k = 3.6912 and draws
# every ellipse 23% too large.  The levels live here, not in the notebook, so
# that mistake is caught by a unit test instead of by eye.
VELOCITY_ELLIPSE_CONFIDENCE_LEVELS = (
    ("1-sigma (39.35% 2D)", 0.3935),
    ("2-sigma (86.47% 2D)", 0.8647),
    ("95% (2D)", 0.95),
    ("3-sigma (98.89% 2D)", 0.98889),
)
VELOCITY_ELLIPSE_DEFAULT_CONFIDENCE = 0.95
# Below half a pixel only the 1 px stroke is visible, so every station would
# draw the same dot whatever its sigma: the ellipse would read as informative
# when it is not.  Nothing is drawn instead.  The upper bound is not decoration
# either - the 9999.99 mm/yr sentinel present in the real NGF and UNR
# catalogues produces a 32,658 px semi-axis, i.e. a DivIcon the size of a city
# block, if it ever slips past the sigma limits.
VELOCITY_ELLIPSE_MIN_SEMI_MAJOR_PX = 0.5
VELOCITY_ELLIPSE_MAX_SEMI_MAJOR_PX = 2048.0
# A published correlation may overshoot 1 by float noise; past this it is a
# corrupt record rather than a rounding artefact.
VELOCITY_CORRELATION_TOLERANCE = 1e-9


def horizontal_confidence_scale(confidence: float) -> float:
    """Return the k whose k-sigma ellipse encloses ``confidence`` in two dimensions.

    Uses the psvelo convention ``k = sqrt(-2 * ln(1 - confidence))``, the inverse
    of the bivariate-normal coverage ``1 - exp(-k**2 / 2)``.  ``confidence`` is a
    fraction strictly between 0 and 1; 0.3935, 0.8647 and 0.98889 return 1, 2 and
    3 to within 1.4e-4, which is the rounding of the tabulated levels themselves
    and not of this formula.

    Raises ``ValueError`` outside ``(0, 1)``.  Callers that take a confidence
    from data rather than from the fixed dropdown should use
    :func:`velocity_ellipse_geometry`, which treats an unusable confidence as
    "draw nothing" instead of raising.
    """
    value = float(confidence)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("confidence must be a finite fraction strictly between 0 and 1")
    return math.sqrt(-2.0 * math.log(1.0 - value))


def velocity_correlation(record: Mapping[str, object] | None) -> float:
    """Read a station's North/East velocity correlation, defaulting to zero.

    ``record`` is one ``data_of[org][site]`` mapping.  NGF ``.vel`` files publish
    ``Rne``, ``Rnu`` and ``Reu``, stored by the notebook as ``velcorr``;
    ``velcorr[0]`` is ``Rne``, the only one a horizontal ellipse uses.  UNR and
    JPL publish no correlation at all, and every record cached before ``velcorr``
    was captured has no such key, so this must never raise: a missing, empty,
    non-numeric or non-finite value all mean the same thing for a horizontal
    ellipse - no correlation is known - and return ``0.0``.  With ``rho = 0`` the
    covariance eigenvalues are exactly ``sigma_n**2`` and ``sigma_e**2``, so the
    ellipse is simply axis-aligned, which is the honest depiction of that.
    """
    if not isinstance(record, Mapping):
        return 0.0
    values = record.get("velcorr")
    if values is None or isinstance(values, (str, bytes)):
        return 0.0
    try:
        correlation = float(values[0])
    except (TypeError, ValueError, IndexError, KeyError):
        return 0.0
    return correlation if math.isfinite(correlation) else 0.0


def _horizontal_sigma(sigma: Sequence[float] | None) -> tuple[float, float] | None:
    """Return finite, non-negative ``(sigma_n, sigma_e)``, or None if unusable."""
    # Duck-typed rather than ``isinstance(..., Sequence)``: a numpy array is not
    # a ``collections.abc.Sequence`` but indexes perfectly well.
    if sigma is None or isinstance(sigma, (str, bytes)):
        return None
    try:
        sigma_n, sigma_e = float(sigma[0]), float(sigma[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if not (math.isfinite(sigma_n) and math.isfinite(sigma_e)):
        return None
    # A negative published sigma is not a small sigma; it means the source row
    # was parsed wrong (NGF station VABG does exactly this today).  Refuse it
    # instead of squaring the sign away.
    if sigma_n < 0.0 or sigma_e < 0.0:
        return None
    return sigma_n, sigma_e


def _clamped_correlation(correlation: float) -> float | None:
    """Return rho inside [-1, 1], or None when the record is beyond float noise."""
    if not math.isfinite(correlation):
        return 0.0
    if abs(correlation) > 1.0 + VELOCITY_CORRELATION_TOLERANCE:
        return None
    return max(-1.0, min(1.0, correlation))


def velocity_ellipse_inputs(
    sigma: Sequence[float] | None,
    correlation: float = 0.0,
    reference_sigma: Sequence[float] | None = None,
    reference_correlation: float = 0.0,
) -> tuple[float, float, float] | None:
    """Return ``(sigma_n, sigma_e, correlation)`` for one confidence ellipse.

    Without a reference station this is the station's own horizontal covariance,
    validated.  With one, the plotted vector is a DIFFERENCE of two velocity
    estimates, so the two covariances add::

        var_n = sigma_n_site**2 + sigma_n_reference**2
        var_e = sigma_e_site**2 + sigma_e_reference**2
        cov   = rho_site*sigma_n_site*sigma_e_site
              + rho_reference*sigma_n_reference*sigma_e_reference

    and the combined correlation is ``cov / (sqrt(var_n) * sqrt(var_e))``, which
    Cauchy-Schwarz keeps inside [-1, 1] whenever both inputs are.  This is the
    donor notebook's ``pltsig``/``pltcorr`` arithmetic, since
    ``hypot(a, b) == sqrt(a**2 + b**2)``.  The two stations are assumed
    independent; neighbouring stations share common-mode error, so a relative
    ellipse is an upper bound.  Only the horizontal covariance is combined - the
    map never differences the vertical component, and break-offset sigmas are
    deliberately not propagated.

    Returns ``None`` when no ellipse should be drawn: missing, non-numeric,
    non-finite or negative sigmas; a correlation outside [-1, 1] by more than
    float noise; or a covariance with no area at all (both sigmas zero).
    """
    horizontal = _horizontal_sigma(sigma)
    if horizontal is None:
        return None
    sigma_n, sigma_e = horizontal
    rho = _clamped_correlation(float(correlation))
    if rho is None:
        return None
    variance_n = sigma_n * sigma_n
    variance_e = sigma_e * sigma_e
    covariance_ne = rho * sigma_n * sigma_e

    if reference_sigma is not None:
        reference = _horizontal_sigma(reference_sigma)
        if reference is None:
            return None
        reference_n, reference_e = reference
        reference_rho = _clamped_correlation(float(reference_correlation))
        if reference_rho is None:
            return None
        variance_n += reference_n * reference_n
        variance_e += reference_e * reference_e
        covariance_ne += reference_rho * reference_n * reference_e

    combined_n = math.sqrt(variance_n)
    combined_e = math.sqrt(variance_e)
    if combined_n <= 0.0 and combined_e <= 0.0:
        return None
    denominator = combined_n * combined_e
    return combined_n, combined_e, (covariance_ne / denominator if denominator else 0.0)


def velocity_ellipse_geometry(
    ellipse_covariance: Sequence[float] | None,
    confidence: float | None,
    scale_km_per_mm_per_year: float,
    ellipse_scale: float = 1.0,
    latitude: float = VELOCITY_REFERENCE_LATITUDE,
    reference_zoom: float = VELOCITY_REFERENCE_ZOOM,
    minimum_semi_major_px: float = VELOCITY_ELLIPSE_MIN_SEMI_MAJOR_PX,
    maximum_semi_major_px: float = VELOCITY_ELLIPSE_MAX_SEMI_MAJOR_PX,
) -> dict[str, float] | None:
    """Return pixel geometry for the confidence ellipse of one plotted vector.

    ``ellipse_covariance`` is ``(sigma_n, sigma_e, rho)`` from
    :func:`velocity_ellipse_inputs` - the uncertainty OF THE PLOTTED VECTOR, so
    the station's own covariance in absolute mode and the combined covariance of
    the difference in relative mode.  The ellipse is the k-sigma contour of
    ``C = [[sigma_n**2, rho*sigma_n*sigma_e], [rho*sigma_n*sigma_e, sigma_e**2]]``
    with ``k`` from :func:`horizontal_confidence_scale`; the semi-axes are
    ``k * sqrt(eigenvalue)`` and the major axis lies along the major eigenvector,
    so the ellipse tilts whenever a correlation is supplied.  With ``rho = 0``
    the eigenvalues are ``sigma_n**2`` and ``sigma_e**2`` and the axes lie along
    North and East, which is what UNR and JPL records get since neither source
    publishes a correlation.  The covariance algebra is the donor notebook's,
    unchanged.

    The only change from the donor is the unit.  The donor multiplied by
    km/(mm/yr) and placed 37 geodesic points, which rescales with zoom; the
    trunk's arrows are screen-fixed, so this multiplies by
    :func:`velocity_pixels_per_mm_per_year` - the exact conversion
    :func:`velocity_arrow_geometry` uses, at the same fixed
    ``VELOCITY_REFERENCE_LATITUDE`` - and the ellipse is welded to the arrow at
    every zoom instead of coinciding at exactly one.  Passing a station's own
    latitude here would silently desynchronize the two.

    ``rotation_deg`` is ready for an SVG ``rotate()`` on an ``<ellipse>`` whose
    ``rx`` is the semi-major axis.  SVG's positive y axis points down and the
    arrow encodes ``x = east, y = -north``, so an azimuth measured from North
    towards East becomes exactly ``rotation = azimuth - 90``: ``rotate(a - 90)``
    sends the ``+x`` axis to ``(cos(a-90), sin(a-90)) = (sin a, -cos a)``, which
    is the unit vector at azimuth ``a`` in that same screen frame.  It is not
    normalized into a positive range on purpose - unmodified, the browser's
    ``rotate(rx cos u, ry sin u)`` locus reproduces the donor's parametric
    ellipse point for point at every ``u``.

    ``ellipse_scale`` is a presentation-only exaggeration.  Any value other than
    1 breaks the one property that makes the ellipse readable - that it is the
    uncertainty of the arrow it sits on, drawn to the arrow's own scale - which
    is why the SVG dashes an exaggerated ellipse.

    Returns ``None`` whenever no honest ellipse can be drawn, which every caller
    treats as "omit the markup": no covariance, a confidence outside ``(0, 1)``
    (``None`` is the canonical "ellipse off"), a non-positive scale or
    exaggeration, or a semi-major axis outside the drawable pixel range.

    One degeneracy returns honest numbers rather than ``None`` and callers should
    know about it: ``|rho| = 1`` collapses the minor axis to exactly 0.0, and per
    the SVG specification an ``<ellipse ry="0">`` is not rendered at all, so such
    an ellipse self-suppresses rather than degrading to a line.
    """
    if ellipse_covariance is None or confidence is None:
        return None
    horizontal = _horizontal_sigma(ellipse_covariance)
    if horizontal is None:
        return None
    sigma_north, sigma_east = horizontal
    try:
        rho = _clamped_correlation(float(ellipse_covariance[2]))
        confidence_value = float(confidence)
        exaggeration = float(ellipse_scale)
        kilometres = float(scale_km_per_mm_per_year)
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if rho is None:
        return None
    if not all(math.isfinite(value) for value in (confidence_value, exaggeration, kilometres)):
        return None
    if not 0.0 < confidence_value < 1.0 or exaggeration <= 0.0 or kilometres <= 0.0:
        return None

    confidence_scale = horizontal_confidence_scale(confidence_value)
    pixels_per_mm_per_year = velocity_pixels_per_mm_per_year(kilometres, latitude, reference_zoom)

    # Closed-form eigen-decomposition of the symmetric covariance ordered
    # (North, East).  max(..., 0.0) guards round-off in the minor eigenvalue.
    variance_n = sigma_north**2
    variance_e = sigma_east**2
    covariance_ne = rho * sigma_north * sigma_east
    mean_variance = 0.5 * (variance_n + variance_e)
    root = math.hypot(0.5 * (variance_n - variance_e), covariance_ne)
    semi_major = math.sqrt(max(mean_variance + root, 0.0))
    semi_minor = math.sqrt(max(mean_variance - root, 0.0))
    # Azimuth of the major axis from North towards East.  Positive correlation
    # tilts the ellipse into the NE-SW quadrant, negative into NW-SE.
    azimuth_deg = math.degrees(0.5 * math.atan2(2.0 * covariance_ne, variance_n - variance_e))

    pixels_per_sigma = confidence_scale * exaggeration * pixels_per_mm_per_year
    semi_major_px = semi_major * pixels_per_sigma
    semi_minor_px = semi_minor * pixels_per_sigma
    if not math.isfinite(semi_major_px):
        return None
    if not float(minimum_semi_major_px) <= semi_major_px <= float(maximum_semi_major_px):
        return None
    return {
        "semi_major_px": semi_major_px,
        "semi_minor_px": semi_minor_px,
        "azimuth_deg": azimuth_deg,
        # rotate(a - 90) sends SVG +x to (cos(a-90), sin(a-90)) = (sin a, -cos a),
        # which is exactly (east, -north) for the unit vector at azimuth a - the
        # same axis convention _vector_svg_markup uses for the arrow.
        "rotation_deg": azimuth_deg - 90.0,
        # Half-extents of the axis-aligned box.  The support of a k-sigma
        # covariance ellipse in direction u is k*sqrt(u'Cu), so the box is
        # k*sigma_e wide and k*sigma_n tall whatever the correlation is.  They
        # exist so the DivIcon canvas can grow to contain the ellipse.
        "half_width_px": sigma_east * pixels_per_sigma,
        "half_height_px": sigma_north * pixels_per_sigma,
        "confidence_scale": confidence_scale,
        "ellipse_scale": exaggeration,
    }


ELLIPSE_STROKE_WIDTH_PX = 1.0
# An exaggerated ellipse is no longer the uncertainty of the arrow it sits on,
# so it is dashed: a reader can never mistake one for a true-scale ellipse.
ELLIPSE_EXAGGERATED_DASH = "3 2"
# Edge margin for the icon canvas: 1 px of stroke half-width plus 1 px of
# antialiasing slack.  Every drawn point is placed explicitly, so no
# direction-dependent padding is needed.
ICON_EDGE_PADDING_PX = 2.0


def vector_icon_bounds(
    geometry: Mapping[str, float],
    ellipse: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Return the DivIcon canvas and the station anchor inside it.

    This is the single source of truth for both the SVG's drawing origin and the
    marker's ``icon_anchor``.  Leaflet places the icon div's top-left at
    ``station pixel - icon_anchor`` and the SVG's (0,0) is that top-left, so the
    station lands where ``anchor_x_px``/``anchor_y_px`` says only while
    ``icon_anchor`` is derived from this same dict.  Growing the canvas without
    moving the anchor moves every arrow off its station by a direction-dependent
    amount, which looks plausible rather than obviously broken.

    Without an ellipse the arrow's own values are returned unchanged, so
    ellipse-off icons and markup stay byte-identical to the pre-ellipse trunk.
    With one, the box is the true hull of the shaft, the arrowhead and the
    ellipse at the tip.  Both anchor components come out integral, so the
    ``int(round(...))`` the spec builders apply is exact.
    """
    if ellipse is None:
        return {
            "canvas_width_px": geometry["canvas_width_px"],
            "canvas_height_px": geometry["canvas_height_px"],
            "anchor_x_px": geometry["anchor_x_px"],
            "anchor_y_px": geometry["anchor_y_px"],
        }
    length = geometry["length_px"]
    unit_x, unit_y = geometry["unit_x"], geometry["unit_y"]
    head = geometry["arrowhead_length_px"]
    half = geometry["arrowhead_half_width_px"]
    # Work in a frame whose origin is the station, then shift it into the canvas.
    tip_x, tip_y = unit_x * length, unit_y * length
    base_x, base_y = tip_x - unit_x * head, tip_y - unit_y * head
    normal_x, normal_y = -unit_y, unit_x
    xs = (0.0, tip_x, base_x + normal_x * half, base_x - normal_x * half,
          tip_x - ellipse["half_width_px"], tip_x + ellipse["half_width_px"])
    ys = (0.0, tip_y, base_y + normal_y * half, base_y - normal_y * half,
          tip_y - ellipse["half_height_px"], tip_y + ellipse["half_height_px"])
    anchor_x = float(math.ceil(ICON_EDGE_PADDING_PX - min(xs)))
    anchor_y = float(math.ceil(ICON_EDGE_PADDING_PX - min(ys)))
    return {
        "canvas_width_px": anchor_x + math.ceil(max(xs) + ICON_EDGE_PADDING_PX),
        "canvas_height_px": anchor_y + math.ceil(max(ys) + ICON_EDGE_PADDING_PX),
        "anchor_x_px": anchor_x,
        "anchor_y_px": anchor_y,
    }


def _normalized_ellipse_covariance(
    ellipse_covariance: Sequence[float] | None,
) -> tuple[float, float, float] | None:
    """Return a hashable ``(sigma_n, sigma_e, rho)`` tuple, or None.

    The render specs are cached on their arguments, so the ellipse covariance
    has to reach the key as a tuple however the caller spelled it.
    """
    if ellipse_covariance is None:
        return None
    values = tuple(float(value) for value in tuple(ellipse_covariance)[:3])
    if len(values) != 3:
        raise ValueError("ellipse_covariance must contain sigma_n, sigma_e, and a correlation")
    return values

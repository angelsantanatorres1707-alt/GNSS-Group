"""Confidence-ellipse tests.

Kept in their own module rather than appended into test_gnss.py so the ellipse
work can be reviewed on its own; the imports below are the ones the spec's
header asked to be added to that file.
"""
import json
import math
import re
from pathlib import Path

import nbformat
import numpy as np
import pytest

from gnss_core import (
    VELOCITY_ELLIPSE_CONFIDENCE_LEVELS,
    VELOCITY_ELLIPSE_DEFAULT_CONFIDENCE,
    VELOCITY_ELLIPSE_MIN_SEMI_MAJOR_PX,
    VELOCITY_REFERENCE_LATITUDE,
    VELOCITY_REFERENCE_ZOOM,
    absolute_vector_render_spec,
    clear_absolute_vector_render_cache,
    horizontal_confidence_scale,
    relative_vector_render_spec,
    vector_icon_bounds,
    velocity_arrow_geometry,
    velocity_correlation,
    velocity_ellipse_geometry,
    velocity_ellipse_inputs,
    velocity_pixels_per_mm_per_year,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "GNSS_Analysis_ipyleaflet.ipynb"
CORE_PATH = REPO_ROOT / "gnss_core.py"


def _uncommented(source):
    """Source with whole-line comments removed, so a warning may name a wrong value."""
    return "\n".join(line for line in source.splitlines() if not line.strip().startswith("#"))


def _svg_ellipse_points(ellipse, angles):
    """The (east, north) pixel offsets a browser draws for this <ellipse>.

    ``rotate(t)`` sends (dx, dy) to (dx cos t - dy sin t, dx sin t + dy cos t) in
    SVG user space, whose y axis points down - the same frame the arrow is built
    in - so north is the negated y offset.
    """
    rotation = math.radians(ellipse["rotation_deg"])
    dx = ellipse["semi_major_px"] * np.cos(angles)
    dy = ellipse["semi_minor_px"] * np.sin(angles)
    return (dx * math.cos(rotation) - dy * math.sin(rotation),
            -(dx * math.sin(rotation) + dy * math.cos(rotation)))


def test_two_dimensional_confidence_levels_match_their_labels():
    scales = {value: horizontal_confidence_scale(value)
              for _label, value in VELOCITY_ELLIPSE_CONFIDENCE_LEVELS}

    assert np.isclose(scales[0.3935], 1.0, atol=5e-4)
    assert np.isclose(scales[0.8647], 2.0, atol=5e-4)
    assert np.isclose(scales[0.98889], 3.0, atol=5e-4)
    assert np.isclose(scales[0.95], 2.447746830681)
    # every label states its own value, so a relabelled dropdown cannot lie
    for label, value in VELOCITY_ELLIPSE_CONFIDENCE_LEVELS:
        percent = float(re.search(r"([\d.]+)%", label).group(1))
        assert round(value * 100.0, 2) == percent
        assert "2D" in label
    assert VELOCITY_ELLIPSE_DEFAULT_CONFIDENCE == 0.95
    with pytest.raises(ValueError):
        horizontal_confidence_scale(0.0)
    with pytest.raises(ValueError):
        horizontal_confidence_scale(1.0)


def test_one_dimensional_three_sigma_constant_never_returns():
    # 1 - exp(-9/2) = 0.98889 is 2-D coverage. The 1-D 99.89% figure gives
    # k = 3.6912 instead of 3, which draws every ellipse 23% too big; that was a
    # real shipped bug, so it is pinned here as a number and as a source rule.
    assert round(1 - math.exp(-4.5), 5) == 0.98889
    assert not any(value == 0.9989 for _label, value in VELOCITY_ELLIPSE_CONFIDENCE_LEVELS)
    inflation = horizontal_confidence_scale(0.9989) / horizontal_confidence_scale(0.98889)
    assert round(inflation, 4) == 1.2304

    core_source = _uncommented(CORE_PATH.read_text(encoding="utf-8"))
    assert "0.98889" in core_source
    assert "0.9989" not in core_source
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    for cell in notebook.cells:
        assert "0.9989" not in _uncommented(cell.get("source", "")), cell.get("source", "")[:60]


def test_velocity_ellipse_geometry_keeps_the_donor_covariance_math_in_pixel_space():
    sigma_n, sigma_e, rho, scale = 1.0, 0.4, 0.5, 30.0
    ellipse = velocity_ellipse_geometry((sigma_n, sigma_e, rho), 0.95, scale)

    confidence_scale = horizontal_confidence_scale(0.95)
    pixels = velocity_pixels_per_mm_per_year(scale, VELOCITY_REFERENCE_LATITUDE, VELOCITY_REFERENCE_ZOOM)
    variance_n, variance_e = sigma_n**2, sigma_e**2
    covariance = rho * sigma_n * sigma_e
    mean_variance = 0.5 * (variance_n + variance_e)
    root = math.hypot(0.5 * (variance_n - variance_e), covariance)
    assert np.isclose(ellipse["semi_major_px"],
                      confidence_scale * pixels * math.sqrt(mean_variance + root))
    assert np.isclose(ellipse["semi_minor_px"],
                      confidence_scale * pixels * math.sqrt(mean_variance - root))
    assert np.isclose(ellipse["azimuth_deg"],
                      math.degrees(0.5 * math.atan2(2 * covariance, variance_n - variance_e)))
    # closed form against a general eigen-solver, and against the axis-aligned case
    values, _vectors = np.linalg.eigh([[variance_n, covariance], [covariance, variance_e]])
    assert np.isclose(ellipse["semi_major_px"], confidence_scale * pixels * math.sqrt(values[1]))
    assert np.isclose(ellipse["semi_minor_px"], confidence_scale * pixels * math.sqrt(values[0]))
    uncorrelated = velocity_ellipse_geometry((sigma_n, sigma_e, 0.0), 0.95, scale)
    assert np.isclose(uncorrelated["semi_major_px"], confidence_scale * pixels * sigma_n)
    assert np.isclose(uncorrelated["semi_minor_px"], confidence_scale * pixels * sigma_e)


def test_velocity_ellipse_rotation_maps_a_north_azimuth_onto_svg_screen_axes():
    north_major = velocity_ellipse_geometry((3.0, 0.5, 0.0), 0.95, 30.0)
    east_major = velocity_ellipse_geometry((0.5, 3.0, 0.0), 0.95, 30.0)
    north_east = velocity_ellipse_geometry((2.0, 2.0, 0.8), 0.95, 30.0)
    north_west = velocity_ellipse_geometry((2.0, 2.0, -0.8), 0.95, 30.0)

    assert np.isclose(north_major["rotation_deg"], -90.0)
    assert np.isclose(east_major["rotation_deg"], 0.0)
    assert np.isclose(north_east["azimuth_deg"], 45.0)
    assert np.isclose(north_west["azimuth_deg"], -45.0)
    for ellipse in (north_major, east_major, north_east, north_west):
        # rotate(a - 90) puts the semi-major endpoint on the azimuth itself
        assert np.isclose(ellipse["rotation_deg"], ellipse["azimuth_deg"] - 90.0)
        east, north = _svg_ellipse_points(ellipse, np.array([0.0]))
        assert np.isclose(math.degrees(math.atan2(east[0], north[0])), ellipse["azimuth_deg"])
    # the drawn locus is the donor's parametric ellipse, point for point
    angles = np.radians(np.arange(0, 361, 10))
    azimuth = math.radians(north_east["azimuth_deg"])
    major, minor = north_east["semi_major_px"], north_east["semi_minor_px"]
    donor_north = major * np.cos(angles) * math.cos(azimuth) - minor * np.sin(angles) * math.sin(azimuth)
    donor_east = major * np.cos(angles) * math.sin(azimuth) + minor * np.sin(angles) * math.cos(azimuth)
    east, north = _svg_ellipse_points(north_east, angles)
    assert np.allclose(east, donor_east) and np.allclose(north, donor_north)


def test_velocity_ellipse_is_drawn_in_the_same_pixel_space_as_its_arrow():
    scale, sigma_n = 30.0, 2.0
    ellipse = velocity_ellipse_geometry((sigma_n, 0.5, 0.0), 0.95, scale)
    arrow = velocity_arrow_geometry((sigma_n, 0.0, 0.0), scale, VELOCITY_REFERENCE_LATITUDE)

    # a one-sigma-long arrow and the one-sigma semi-axis are the same length
    assert np.isclose(ellipse["semi_major_px"] / horizontal_confidence_scale(0.95), arrow["length_px"])
    # both default to the fixed reference latitude and zoom, so neither rescales
    # with the map; passing a station's own latitude would desynchronize them
    assert np.isclose(
        velocity_ellipse_geometry((sigma_n, 0.5, 0.0), 0.95, scale,
                                  latitude=VELOCITY_REFERENCE_LATITUDE,
                                  reference_zoom=VELOCITY_REFERENCE_ZOOM)["semi_major_px"],
        ellipse["semi_major_px"])
    doubled = velocity_ellipse_geometry((sigma_n, 0.5, 0.0), 0.95, 2 * scale)
    assert np.isclose(doubled["semi_major_px"], 2 * ellipse["semi_major_px"])
    exaggerated = velocity_ellipse_geometry((sigma_n, 0.5, 0.0), 0.95, scale, 20.0)
    assert np.isclose(exaggerated["semi_major_px"], 20.0 * ellipse["semi_major_px"])
    assert exaggerated["ellipse_scale"] == 20.0 and ellipse["ellipse_scale"] == 1.0
    # the bounding half extents are k*sigma_e wide and k*sigma_n tall whatever rho is
    for correlation in (-0.9, 0.0, 0.6):
        tilted = velocity_ellipse_geometry((sigma_n, 0.5, correlation), 0.95, scale)
        east, north = _svg_ellipse_points(tilted, np.radians(np.arange(0, 360, 0.25)))
        assert np.isclose(np.max(np.abs(east)), tilted["half_width_px"], rtol=1e-5)
        assert np.isclose(np.max(np.abs(north)), tilted["half_height_px"], rtol=1e-5)


def test_velocity_ellipse_is_omitted_rather_than_drawn_dishonestly():
    # NGF's median sigma at the default Vel Scale is a third of a pixel, where a
    # stroked outline only shows its own stroke width; nothing is drawn instead.
    assert velocity_ellipse_geometry((0.09, 0.09, 0.0), 0.95, 10.0) is None
    assert velocity_ellipse_geometry((2.0, 2.0, 0.0), None, 10.0) is None
    assert velocity_ellipse_geometry(None, 0.95, 10.0) is None
    assert velocity_ellipse_geometry((2.0, 2.0, 0.0), 0.0, 10.0) is None
    assert velocity_ellipse_geometry((2.0, 2.0, 0.0), 1.0, 10.0) is None
    assert velocity_ellipse_geometry((float("nan"), 2.0, 0.0), 0.95, 10.0) is None
    assert velocity_ellipse_geometry((-1.38, 0.26, 0.0), 0.95, 30.0) is None
    assert velocity_ellipse_geometry((2.0, 2.0, 1.5), 0.95, 30.0) is None
    assert velocity_ellipse_geometry((2.0, 2.0, 0.0), 0.95, 0.0) is None
    assert velocity_ellipse_geometry((2.0, 2.0, 0.0), 0.95, 30.0, 0.0) is None
    # the 9999.99 mm/yr sentinel in the real catalogues is a 32,658 px semi-axis
    assert velocity_ellipse_geometry((9999.99, 9999.99, 0.0), 0.95, 30.0) is None
    assert VELOCITY_ELLIPSE_MIN_SEMI_MAJOR_PX == 0.5
    # |rho| = 1 is a real degeneracy, so it reports an honest zero minor axis
    # (which an SVG <ellipse ry="0"> then declines to render) instead of a lie
    degenerate = velocity_ellipse_geometry((2.0, 2.0, 1.0 + 1e-12), 0.95, 30.0)
    assert degenerate["semi_minor_px"] == 0.0


def test_velocity_correlation_defaults_to_zero_when_a_source_publishes_none():
    assert velocity_correlation({"velcorr": [-0.072, 0.011, 0.004]}) == -0.072
    # UNR and JPL publish no correlation, and every record cached before
    # velcorr was captured has no such key: none of these may raise
    for record in ({}, None, {"velsig": [0.1, 0.1, 0.3]}, {"velcorr": None},
                   {"velcorr": []}, {"velcorr": "0.5"}, {"velcorr": [None]},
                   {"velcorr": [float("nan")]}):
        assert velocity_correlation(record) == 0.0
    # rho = 0 is not a fudge: the eigenvalues are then exactly the sigmas
    axis_aligned = velocity_ellipse_geometry((2.0, 1.0, velocity_correlation({})), 0.95, 30.0)
    assert np.isclose(axis_aligned["rotation_deg"], -90.0)


def test_velocity_ellipse_inputs_add_both_covariances_for_a_relative_vector():
    site, reference = (0.30, 0.40, 9.9), (0.20, 0.10, 9.9)

    absolute = velocity_ellipse_inputs(site, 0.5)
    relative = velocity_ellipse_inputs(site, 0.5, reference, -0.25)

    assert absolute == (0.30, 0.40, 0.5)
    assert velocity_ellipse_inputs(site) == (0.30, 0.40, 0.0)
    # the donor's pltsig/pltcorr, exactly: hypot(a, b) == sqrt(a**2 + b**2)
    expected_n = math.hypot(0.30, 0.20)
    expected_e = math.hypot(0.40, 0.10)
    expected_cov = 0.5 * 0.30 * 0.40 + -0.25 * 0.20 * 0.10
    assert np.allclose(relative, (expected_n, expected_e, expected_cov / (expected_n * expected_e)))
    # a station differenced against an identical one: sqrt(2) on both axes,
    # and the tilt is unchanged because scaling a covariance keeps its axes
    doubled = velocity_ellipse_inputs(site, 0.5, site, 0.5)
    assert np.allclose(doubled[:2], (0.30 * math.sqrt(2), 0.40 * math.sqrt(2)))
    assert np.isclose(doubled[2], 0.5)
    # Cauchy-Schwarz keeps the combined correlation a legal one
    generator = np.random.default_rng(20260824)
    for _ in range(4096):
        sigmas = generator.uniform(0.0, 4.0, 4)
        first, second = generator.uniform(-1.0, 1.0, 2)
        combined = velocity_ellipse_inputs(
            (sigmas[0], sigmas[1], 0.0), first, (sigmas[2], sigmas[3], 0.0), second)
        assert combined is None or abs(combined[2]) <= 1.0


def test_velocity_ellipse_inputs_refuse_a_covariance_they_cannot_draw():
    # NGF station VABG publishes velsig [-1.38, 0.26, 0.15] because its two-word
    # station name shifts every field by one; a negative sigma is a parse fault,
    # not a small uncertainty, so it must not be squared into a plausible ellipse
    assert velocity_ellipse_inputs((-1.38, 0.26, 0.15)) is None
    assert velocity_ellipse_inputs((0.3, 0.4, 0.0), 0.0, (-1.0, 0.2, 0.0)) is None
    assert velocity_ellipse_inputs(None) is None
    assert velocity_ellipse_inputs((0.3,)) is None
    assert velocity_ellipse_inputs((float("inf"), 0.4, 0.0)) is None
    assert velocity_ellipse_inputs((0.0, 0.0, 0.0)) is None
    assert velocity_ellipse_inputs((0.3, 0.4, 0.0), 1.5) is None
    assert velocity_ellipse_inputs((0.3, 0.4, 0.0), 0.0, (0.2, 0.1, 0.0), -1.5) is None
    # float noise on a published correlation is a rounding artefact, not corruption
    assert velocity_ellipse_inputs((0.3, 0.4, 0.0), 1.0 + 1e-12) == (0.3, 0.4, 1.0)
    # one zero axis is a genuine degeneracy and survives; numpy arrays index fine
    assert velocity_ellipse_inputs(np.array([0.0, 0.4, 0.1])) == (0.0, 0.4, 0.0)


def test_vector_icon_bounds_contain_the_ellipse_and_keep_the_station_anchored():
    arrow = velocity_arrow_geometry((28.0, -19.0, 1.0), 10.0, VELOCITY_REFERENCE_LATITUDE)
    ellipse = velocity_ellipse_geometry((0.42, 0.31, 0.9), 0.98889, 10.0, 20.0)

    without = vector_icon_bounds(arrow, None)
    with_ellipse = vector_icon_bounds(arrow, ellipse)

    # the arrow-only box is untouched, so ellipse-off icons are unchanged
    for key in ("canvas_width_px", "canvas_height_px", "anchor_x_px", "anchor_y_px"):
        assert without[key] == arrow[key]
        # integral, so int(round(...)) cannot drift the icon anchor half a pixel
        # away from the origin the SVG actually draws from
        assert with_ellipse[key] == int(with_ellipse[key])
    assert with_ellipse["canvas_height_px"] > without["canvas_height_px"]
    # the station, the tip and the whole ellipse sit inside the canvas
    x0, y0 = with_ellipse["anchor_x_px"], with_ellipse["anchor_y_px"]
    tip_x = x0 + arrow["unit_x"] * arrow["length_px"]
    tip_y = y0 + arrow["unit_y"] * arrow["length_px"]
    east, north = _svg_ellipse_points(ellipse, np.radians(np.arange(0, 360, 1.0)))
    xs = np.concatenate([tip_x + east, [x0, tip_x]])
    ys = np.concatenate([tip_y - north, [y0, tip_y]])
    assert xs.min() >= 0.0 and ys.min() >= 0.0
    assert xs.max() <= with_ellipse["canvas_width_px"]
    assert ys.max() <= with_ellipse["canvas_height_px"]


def test_ellipse_inputs_join_the_cache_key_without_disturbing_arrow_only_specs():
    clear_absolute_vector_render_cache()
    args = ("NGF", "P049", (40.0, -75.0), (12.5, -20.3, 1.2), (0.42, 0.31, 0.9),
            (1.0, 1.0, 2.0), 5.0, 10.0, "#0000ff", "P049")
    covariance = (0.42, 0.31, -0.35)

    plain = absolute_vector_render_spec(*args)
    disabled = absolute_vector_render_spec(*args, ellipse_covariance=None, ellipse_confidence=None)
    enabled = absolute_vector_render_spec(*args, ellipse_covariance=covariance,
                                          ellipse_confidence=0.95, ellipse_scale=20.0)
    stricter = absolute_vector_render_spec(*args, ellipse_covariance=covariance,
                                           ellipse_confidence=0.98889, ellipse_scale=20.0)

    # "off" is one canonical key, so toggling the ellipse off is a cache hit and
    # invalidates nothing; the notebook therefore never clears the cache for it
    assert plain is disabled
    assert "<ellipse" not in plain.svg and 'aria-label="velocity arrow"' in plain.svg
    assert "<ellipse" in enabled.svg and 'transform="rotate(' in enabled.svg
    assert 'fill="none"' in enabled.svg and enabled.svg.count("pointer-events:none") == 1
    # confidence is in the key; without it this would serve the 95% ellipse
    assert stricter.svg != enabled.svg
    assert stricter.ellipse_semi_axes_px[0] > enabled.ellipse_semi_axes_px[0]
    assert enabled.ellipse_covariance == covariance and enabled.ellipse_confidence == 0.95
    assert np.isclose(enabled.ellipse_rotation_deg, enabled.ellipse_rotation_deg)
    # the canvas grew for the ellipse, so the anchor moved with it
    assert enabled.icon_size != plain.icon_size
    assert enabled.icon_anchor != plain.icon_anchor
    # an exaggerated ellipse is dashed, a true-scale one is not
    true_scale = absolute_vector_render_spec(*args, ellipse_covariance=(2.0, 1.5, 0.0),
                                             ellipse_confidence=0.95)
    assert "stroke-dasharray" in enabled.svg and "stroke-dasharray" not in true_scale.svg
    assert absolute_vector_render_spec.cache_info().currsize == 4


def test_relative_render_spec_carries_the_combined_covariance_without_caching():
    clear_absolute_vector_render_cache()
    combined = velocity_ellipse_inputs((0.30, 0.40, 9.9), 0.5, (0.20, 0.10, 9.9), -0.25)

    spec = relative_vector_render_spec(
        [40.0, -75.0], [1.0, 2.0, 3.0], [0.30, 0.40, 0.9], [1.0, 1.0, 2.0], 5.0, 30.0, "#000000",
        ellipse_covariance=combined, ellipse_confidence=0.95, ellipse_scale=20.0)

    assert absolute_vector_render_spec.cache_info().currsize == 0
    # spec.sigma stays the source velsig; the ellipse carries the differenced one
    assert spec.sigma == (0.30, 0.40, 0.9)
    assert spec.ellipse_covariance == combined
    assert "<ellipse" in spec.svg
    expected = velocity_ellipse_geometry(combined, 0.95, 30.0, 20.0)
    assert np.isclose(spec.ellipse_semi_axes_px[0], expected["semi_major_px"])
    assert np.isclose(spec.ellipse_rotation_deg, expected["rotation_deg"])


def test_notebook_captures_ngf_velocity_correlations_without_breaking_old_caches():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    fetcher = next(cell.source for cell in notebook.cells if "def UpdateNGF" in cell.get("source", ""))

    assert 'for_json[siteid]["velcorr"] = [float(pt) for pt in info[25:28]]' in fetcher
    # Rne, Rnu and Reu are dimensionless, unlike the two lines above them
    velcorr_line = next(line for line in fetcher.splitlines() if '["velcorr"] =' in line)
    assert "*1000" not in velcorr_line
    assert '[float(pt)*1000 for pt in info[19:22]]' in fetcher
    assert '[float(pt)*1000 for pt in info[22:25]]' in fetcher
    # every read goes through the defaulting accessor, because no shipped cache
    # has the key and a bare subscript would raise mid-render for every station
    others = "\n".join(cell.source for cell in notebook.cells
                       if cell.cell_type == "code" and cell.source != fetcher)
    assert '"velcorr"' not in others and "'velcorr'" not in others
    assert others.count("velocity_correlation(") >= 2


def test_notebook_relative_ellipse_combines_the_reference_station_covariance():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    source = next(cell.source for cell in notebook.cells if "def plot_velocity" in cell.get("source", ""))
    body = source.split("def plot_velocity", 1)[1].split("def draw_length", 1)[0]

    assert 'reference_sigma = reference_record.get("velsig", [0,0,0])' in body
    assert "reference_correlation = velocity_correlation(reference_record)" in body
    # the exact argument pair, so dropping the reference cannot pass silently
    assert ("velocity_ellipse_inputs(\n"
            "                velsig, velocity_correlation(data_of[org][siteid]),\n"
            "                reference_sigma, reference_correlation,\n"
            "            )") in body
    # refvel is bound before the loop; an empty selection used to be an
    # UnboundLocalError on the next line
    assert body.index("refvel = [0,0,0]") < body.index("for option in list(map_radius_list.value)")
    # the ellipse is built inside the sigma-limit branch, so it can never
    # outlive the arrow whose uncertainty it claims to be
    assert body.index("if all(velsig[i] <= siglim[i] for i in range(3)):") < body.index("ellipse_covariance = velocity_ellipse_inputs")
    assert "ellipse_confidence=ellipse_confidence.value if ellipse_covariance is not None else None" in body
    assert "ellipse_scale=ellipse_scale_input.value" in body
    # the ellipse controls are in the cache key, so nothing is invalidated here
    assert "clear_absolute_vector_render_cache" not in body

    vector = next(cell.source for cell in notebook.cells if "def draw_vector" in cell.get("source", ""))
    vector = vector.split("def draw_vector", 1)[1]
    for keyword in ("ellipse_covariance=", "ellipse_confidence=", "ellipse_scale="):
        assert vector.count(keyword) >= 2      # the absolute and relative branches


def test_notebook_ellipse_controls_follow_the_map_column_row_idiom():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    declarations = next(cell.source for cell in notebook.cells if "site_table          = VBox" in cell.get("source", ""))
    dashboard = next(cell.source for cell in notebook.cells if "station_selection_section = HBox" in cell.get("source", ""))

    # the levels come from gnss_core, so the dropdown cannot be retyped wrongly
    assert "options=list(VELOCITY_ELLIPSE_CONFIDENCE_LEVELS)" in declarations
    assert "value=VELOCITY_ELLIPSE_DEFAULT_CONFIDENCE" in declarations
    assert "plot_ellipse_check     = wg.Checkbox" in declarations
    assert "ellipse_scale_input    = wg.Dropdown" in declarations
    for row, control in (("ellipse_toggle_row", "plot_ellipse_check"),
                         ("ellipse_level_row", "ellipse_confidence"),
                         ("ellipse_scale_row", "ellipse_scale_input")):
        assert f"{row} = HBox([" in dashboard
        assert f"{row}," in dashboard.split("vector_preview_section = VBox([", 1)[1]
        assert control in dashboard
        assert f"{control}.layout = vector_field_layout" in dashboard
    for label in ("'Error Ellipse:'", "'Ellipse Level:'", "'Ellipse x:'"):
        assert f"wg.Label({label}, layout=vector_label_layout)" in dashboard
    assert dashboard.count("layout=vector_row_layout)") >= 9
    # the level and the exaggeration are meaningless while nothing is drawn
    assert "ellipse_confidence.disabled = not plot_ellipse_check.value" in dashboard
    assert "plot_ellipse_check.observe(update_ellipse_enabled, names='value')" in dashboard
    # the three-column dashboard was not re-proportioned
    widths = [int(value) for value in re.findall(r"width='(\d+)%', flex='0 0 \d+%'", dashboard)]
    assert widths[:3] == [29, 42, 29] and sum(widths[:3]) == 100
    assert dashboard.count("wg.jslink(") == 4

    map_functions = next(cell.source for cell in notebook.cells if "def map_setting_changed" in cell.get("source", ""))
    for control in ("plot_ellipse_check", "ellipse_confidence", "ellipse_scale_input"):
        assert f"{control}.observe(map_setting_changed, names='value')" in map_functions
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import nbformat
import numpy as np
import pandas as pd
import pytest
import ipywidgets as wg
from geopy.distance import geodesic
from ipyleaflet import CircleMarker, LayerGroup, Map, Marker
from nbclient import NotebookClient

from gnss_core import (
    build_map_render_plan,
    detrended,
    filter_sigmas,
    fit_ts,
    number_list,
    parse_bulk_selection,
    parse_map_selection,
    remove_brac,
    vec_add,
    vec_sub,
    velocity_endpoint,
    VELOCITY_REFERENCE_LATITUDE,
    format_display_number,
    format_neu_vector,
    neighbor_velocity_row,
    velocity_arrow_geometry,
    absolute_vector_render_spec,
    clear_absolute_vector_render_cache,
    relative_vector_render_spec,
    velocity_guide_metrics,
    velocity_pixels_per_mm_per_year,
    web_mercator_meters_per_pixel,
    resolve_popup_owner,
)
from gnss_map_runtime import stage_layer_update


NOTEBOOK_PATH = Path(__file__).parent / "GNSS_Analysis_ipyleaflet.ipynb"


def test_notebook_neighbor_table_is_responsive_and_stacked():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    neighbor_source = next(
        cell["source"] for cell in notebook.cells if "def render_neighbor_table" in cell.get("source", "")
    )
    assert "neighbor_table_cell" not in neighbor_source
    assert "neighbor_cell_widget" not in neighbor_source
    assert "import html as html_module" in neighbor_source
    assert "<table" in neighbor_source and "<colgroup>" in neighbor_source
    assert "table-layout: fixed" in neighbor_source
    assert "width: 100%" in neighbor_source and "max-width: 100%" in neighbor_source
    assert "overflow: hidden" in neighbor_source
    assert "overflow: auto" not in neighbor_source and "overflow: scroll" not in neighbor_source
    assert "Distance<br>(km)" in neighbor_source
    assert "Velocity<br>NEU" in neighbor_source
    assert "Sigma<br>NEU" in neighbor_source
    assert "Differential<br>NEU" in neighbor_source
    assert "html_module.escape" in neighbor_source
    assert "format_display_number(components[index]" in neighbor_source
    assert "gnss-neighbor-neu-component" in neighbor_source
    assert "white-space: nowrap" in neighbor_source
    assert "—" in neighbor_source
    assert "range(1, 11)" in neighbor_source

    render_source = neighbor_source.split("def render_neighbor_table", 1)[1]
    widths_match = re.search(r"widths = \[(.*?)\]", render_source, flags=re.DOTALL)
    assert widths_match
    widths = [int(value) for value in re.findall(r"'(\d+)%'", widths_match.group(1))]
    assert widths == [7, 12, 11, 13, 19, 19, 19]
    assert sum(widths) == 100

    dashboard_source = next(
        cell["source"] for cell in notebook.cells if "station_selection_section = HBox" in cell.get("source", "")
    )
    dashboard_widths = [
        int(value) for value in re.findall(r"width='(\d+)%', flex='0 0 \d+%'", dashboard_source)
    ]
    assert dashboard_widths[:3] == [29, 42, 29]
    assert sum(dashboard_widths[:3]) == 100
    assert "align_items='stretch'" in dashboard_source

    site_source = next(cell["source"] for cell in notebook.cells if "def render_site_table" in cell.get("source", ""))
    assert "width='100%', min_width='0'" in site_source
    assert "box_sizing=" not in site_source and "gap=" not in site_source
    site_widths = [int(value) for value in re.findall(r"'(\d+)%'", re.search(r"widths = \[(.*?)\]", site_source, flags=re.DOTALL).group(1))]
    assert site_widths == [7, 30, 25, 38] and sum(site_widths) == 100
    for label in ("No.", "Station ID", "Source", "Radius (km)"):
        assert f"table_header('{label}')" in site_source
    assert "padding: 0.12em 0.16em" in render_source
    assert "line-height: 0.98" in render_source
    assert "height='8rem'" in site_source
    assert "overflow='hidden auto' if len(map_radius_list.options) > 5 else 'hidden'" in site_source
    body_segment = site_source.split("body_layout = wg.Layout", 1)[1].split("data_body = VBox", 1)[0]
    assert "overflow_y=" not in body_segment and "overflow_x=" not in body_segment
    assert "height='1.6rem', min_height='1.6rem', max_height='1.6rem', flex='0 0 1.6rem'" in site_source
    assert "flex='0 0 2.2rem'" in site_source and "flex='0 0 2.0rem'" in site_source
    assert "data_body = VBox(data_rows" in site_source
    assert "site_table.children = (title_row, header, data_body, map_site_scroll_output)" in site_source
    assert "row_count = max(5, len(map_radius_list.options))" in site_source
    assert "height='2.2rem'" in site_source and "height='2.0rem'" in site_source
    assert "range(1, row_count + 1)" in site_source
    assert "flex='0 0 auto'" in render_source
    declarations = next(cell["source"] for cell in notebook.cells if "site_table          = VBox" in cell.get("source", ""))
    assert "site_table          = VBox" in declarations
    assert "flex='0 0 auto'" in declarations
    assert "neighbor_table = VBox" in declarations
    assert "overflow_y='hidden'" in declarations
    assert "width=table_content_width" in declarations
    assert "margin='0 0 0.5% 0'" in declarations


def test_station_viewport_scrolls_only_after_successful_sixth_map_add():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    source = next(cell["source"] for cell in notebook.cells if "def update_map_list" in cell.get("source", ""))
    assert "previous_count = len(map_radius_list.options)" in source
    assert "if previous_count >= 5:" in source
    assert "scroll_new_map_station_into_view(len(map_radius_list.options))" in source
    assert "gnss-selected-stations-viewport" in source
    bridge = next(cell["source"] for cell in notebook.cells if "def scroll_new_map_station_into_view" in cell.get("source", ""))
    assert "map_site_scroll_output.clear_output(wait=True)" in bridge
    assert "querySelectorAll('.gnss-selected-stations-viewport')" in bridge
    assert "expected_count: int" in bridge
    assert "expectedCount" in bridge
    assert "querySelectorAll('.gnss-selected-station-row').length >= expectedCount" in bridge
    assert "isConnected" in bridge and "offsetParent" in bridge
    assert "requestAnimationFrame(sync)" in bridge
    assert "attempts < 12" in bridge
    assert "viewport.scrollTop = viewport.scrollHeight" in bridge
    assert "scrollIntoView" not in bridge
    render = next(cell["source"] for cell in notebook.cells if "def render_site_table" in cell.get("source", ""))
    assert "overflow='hidden auto' if len(map_radius_list.options) > 5 else 'hidden'" in render
    assert "data_body.add_class('gnss-selected-stations-viewport')" in render
    assert "gnss-selected-station-row" in render
    assert "site_table.children = (title_row, header, data_body, map_site_scroll_output)" in render


def test_velocity_guide_is_compact_magnitude_bracket_with_separate_ground_equivalent():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    guide_source = next(cell.source for cell in notebook.cells if "def update_vector_guide" in cell.get("source", ""))
    assert "gnss-velocity-guide" in guide_source
    assert "mm/yr vector magnitude" in guide_source
    assert "border-top:3px solid #111" in guide_source
    assert "border-left:2px solid #111" in guide_source
    assert "border-right:2px solid #111" in guide_source
    assert "Geographic equivalent:" in guide_source
    assert "current_km" in guide_source and "guide_vel" in guide_source and "scale" in guide_source
    assert 'guide_width_px = metrics["bar_px"]' in guide_source
    assert "max(48.0" not in guide_source and "0.32 * guide_vel * scale" not in guide_source
    assert "multicolor" not in guide_source.lower()

    controls = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert "Guide (mm/yr):" in controls
    assert "Arrow (mm/yr):" not in controls
    assert "Magnitude represented by the screen-fixed velocity reference bracket" in controls
    assert "Controls plotted-vector and guide length conversion" in controls
    help_source = "\n".join(cell.source for cell in notebook.cells if "Plot Velocities:" in cell.get("source", ""))
    assert "Guide: Sets the velocity magnitude represented by the screen-fixed reference bracket" in help_source
    assert "separate geographic-equivalent readout changes with the live map zoom and center latitude" in help_source

    control_declarations = next(
        cell["source"] for cell in notebook.cells if "bulk_selection_control_height" in cell.get("source", "")
    )
    assert "table_content_width = '99%'" in control_declarations
    assert "bulk_selection_control_height = '2rem'" in control_declarations
    assert control_declarations.count("height=bulk_selection_control_height") >= 3

    dashboard_source = next(
        cell["source"] for cell in notebook.cells if "bulk_selection_section = VBox" in cell.get("source", "")
    )
    assert "width='100%', min_width='0', height=bulk_selection_control_height" in dashboard_source
    assert "margin='0 0 0.2rem 0'" in dashboard_source
    assert "width=table_content_width, min_width='0', height='4.2rem', min_height='4.2rem'" in dashboard_source
    assert "flex='0 0 auto', overflow='visible'" in dashboard_source
    selection_layout_source = dashboard_source.split("bulk_selection_row = HBox", 1)[1].split("center_map_column = VBox", 1)[0]
    assert "gap=" not in selection_layout_source
    assert "box_sizing=" not in selection_layout_source
    assert "align_items='center'" in dashboard_source
    assert "height='clamp(610px, 34vw, 630px)'" in dashboard_source

    serialized_layout = wg.Layout(
        width='100%', height='2rem', min_height='2rem',
        margin='0 0 0.2rem 0', flex='0 0 auto', overflow='visible',
    ).get_state()
    assert serialized_layout['height'] == '2rem'
    assert serialized_layout['min_height'] == '2rem'
    assert serialized_layout['margin'] == '0 0 0.2rem 0'
    assert serialized_layout['flex'] == '0 0 auto'
    assert serialized_layout['overflow'] == 'visible'
    serialized_scroll = wg.Layout(overflow='hidden auto').get_state()
    assert serialized_scroll['overflow'] == 'hidden auto'


def test_rebuilt_station_row_observers_update_selection_and_neighbor_table():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    cell_source = next(cell.source for cell in notebook.cells if "def sync_map_site_selection" in cell.get("source", ""))
    fragment = cell_source[cell_source.index("def selected_map_options_from_rows"):cell_source.index("def nearest_station_rows")]
    neighbor_calls = []
    options = ("AAAA, NGF, 10km", "BBBB, NGF, 20km")
    namespace = {
        "wg": wg,
        "HBox": wg.HBox,
        "VBox": wg.VBox,
        "map_site_checks": {},
        "map_radius_list": wg.SelectMultiple(options=options, value=()),
        "render_neighbor_table": lambda: neighbor_calls.append(True),
        "site_table": wg.VBox(),
        "map_site_scroll_output": wg.Output(),
    }
    exec(fragment, namespace)

    namespace["render_site_table"]()
    namespace["map_site_checks"][options[1]].value = True
    assert namespace["map_radius_list"].value == (options[1],)
    assert len(neighbor_calls) == 1

    namespace["render_site_table"]()
    namespace["map_site_checks"][options[0]].value = True
    assert namespace["map_radius_list"].value == options
    assert len(neighbor_calls) == 2


def test_add_station_ignores_exact_duplicate_but_allows_different_radius():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    cell_source = next(cell.source for cell in notebook.cells if "def update_map_list" in cell.get("source", ""))
    fragment = cell_source[cell_source.index("def update_map_list"):cell_source.index("add_sites_map_button.on_click")]
    original = (
        "AAAA, NGF, 10km", "BBBB, NGF, 10km", "CCCC, NGF, 10km",
        "DDDD, NGF, 10km", "EEEE, NGF, 10km",
    )
    render_calls = []
    scroll_calls = []
    namespace = {
        "site_id": wg.Text(value="AAAA"),
        "site_radius": wg.IntText(value=10),
        "org_map_select": wg.Dropdown(options=("NGF",), value="NGF"),
        "data_of": {"NGF": {station: {} for station in ("AAAA", "BBBB", "CCCC", "DDDD", "EEEE")}},
        "map_radius_list": wg.SelectMultiple(options=original, value=(original[0],)),
        "render_site_table": lambda: render_calls.append("site"),
        "render_neighbor_table": lambda: render_calls.append("neighbor"),
        "scroll_new_map_station_into_view": lambda count: scroll_calls.append(count),
    }
    exec(fragment, namespace)

    namespace["update_map_list"](None)
    assert namespace["map_radius_list"].options == original
    assert namespace["map_radius_list"].value == (original[0],)
    assert render_calls == [] and scroll_calls == []

    namespace["site_radius"].value = 20
    namespace["update_map_list"](None)
    assert namespace["map_radius_list"].options[-1] == "AAAA, NGF, 20km"
    assert namespace["map_radius_list"].value == (original[0],)
    assert render_calls == ["site", "neighbor"]
    assert scroll_calls == [6]


def test_live_dynamic_layer_resolver_rejects_detached_staging_group():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    cell_source = next(cell.source for cell in notebook.cells if "def ensure_live_dynamic_layers" in cell.get("source", ""))
    fragment = cell_source[cell_source.index("def ensure_live_dynamic_layers"):cell_source.index("def new_map")]
    map_widget = Map()
    visible = LayerGroup(name="Station and Velocity Layers")
    detached = LayerGroup(name="Staged Station and Velocity Layers")
    map_widget.add(visible)
    namespace = {
        "LayerGroup": LayerGroup,
        "map_dynamic_layers": [detached],
        "add_map_layer": lambda owner, layer: owner.add(layer),
    }
    exec(fragment, namespace)

    resolved = namespace["ensure_live_dynamic_layers"](map_widget)
    assert resolved is visible
    assert namespace["map_dynamic_layers"][0] is visible


def test_plot_sites_reconciles_live_row_selection_and_reloads_visible_map():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    cell_source = next(cell.source for cell in notebook.cells if "def site_circle" in cell.get("source", ""))
    fragment = cell_source[cell_source.index("def site_circle"):cell_source.index("site_circle_submit.on_click")]
    option = "AAAA, NGF, 10km"
    map_holder = ["Not initialized yet!"]
    selected_widget = wg.ToggleButton(value=True)
    events = []

    class DummyMap:
        layers = ()

    def new_map(_button):
        events.append("new")
        map_holder[0] = DummyMap()

    namespace = {
        "map": map_holder,
        "map_site_checks": {option: selected_widget},
        "map_radius_list": wg.SelectMultiple(options=(option,), value=()),
        "selected_map_options_from_rows": lambda: (option,),
        "new_map": new_map,
        "ensure_live_dynamic_layers": lambda _map: events.append("resolve"),
        "reload_map": lambda _button: events.append("reload"),
    }
    exec(fragment, namespace)
    namespace["site_circle"](None)

    assert namespace["map_radius_list"].value == (option,)
    assert events == ["new", "resolve", "reload"]
    assert "site_circle_submit.on_click(site_circle)" in cell_source


VALIDATION_DIR = Path(__file__).parent / "validation"
VALIDATION_MANIFEST_PATH = VALIDATION_DIR / "manifest.json"
REVIEWED_MANIFEST_SHA256 = "c88e1767cd9afbadc02e5b5f5be32da906394ea08803792536c74d150fb48a41"
REQUIRED_FIXTURE_IDENTITY_FIELDS = {
    "fixture_id",
    "file",
    "origin_type",
    "provider",
    "source_url",
    "retrieved_at",
    "retrieval_timestamp_basis",
    "release_id",
    "reference_frame",
    "capture_method",
    "purpose",
    "sha256",
    "byte_count",
    "data_row_count",
    "units",
}
REQUIRED_TEXT_IDENTITY_FIELDS = REQUIRED_FIXTURE_IDENTITY_FIELDS - {
    "byte_count",
    "data_row_count",
    "units",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def load_validation_manifest():
    return json.loads(VALIDATION_MANIFEST_PATH.read_text(encoding="utf-8"))


REGISTERED_FIXTURE_IDS = tuple(
    fixture["fixture_id"] for fixture in load_validation_manifest()["fixtures"]
)
EXTERNAL_FIXTURE_IDS = tuple(
    fixture["fixture_id"]
    for fixture in load_validation_manifest()["fixtures"]
    if fixture["origin_type"] == "external"
)
SYNTHETIC_FIXTURE_IDS = tuple(
    fixture["fixture_id"]
    for fixture in load_validation_manifest()["fixtures"]
    if fixture["origin_type"] == "synthetic"
)


def validation_fixture(manifest, fixture_id):
    return next(fixture for fixture in manifest["fixtures"] if fixture["fixture_id"] == fixture_id)


def load_weighted_fit_oracle():
    manifest = load_validation_manifest()
    fixture = validation_fixture(manifest, "weighted_fit_oracle_v1")
    table = pd.read_csv(VALIDATION_DIR / fixture["file"])
    return table, fixture["expected"]


def test_valid_numeric_input_should_return_float_values():
    text = "(10, 20, 30)"

    result = number_list(text, [1, 1, 2])

    assert result == [10.0, 20.0, 30.0]


def test_invalid_numeric_input_should_return_a_new_default_list():
    fallback = [1, 1, 2]

    result = number_list("not a number list", fallback)

    assert result == fallback


def test_boolean_numeric_input_should_return_default_values():
    fallback = [1, 1, 2]

    result = number_list("(True, 1, 2)", fallback)

    assert result == fallback


def test_north_sigma_limit_should_filter_only_north_failures():
    table = pd.DataFrame([[0.001, 0.020, 0.030], [0.020, 0.002, 0.003]])

    result = filter_sigmas(table, [0, 1, 2], (10, 0, 0), 1000)

    assert result.index.tolist() == [0]


def test_east_sigma_limit_should_filter_only_east_failures():
    table = pd.DataFrame([[0.001, 0.020, 0.030], [0.020, 0.002, 0.003]])

    result = filter_sigmas(table, [0, 1, 2], (0, 10, 0), 1000)

    assert result.index.tolist() == [1]


def test_linear_time_series_should_return_expected_velocity():
    xdata = np.array([0.0, 365.25, 730.5])
    ydata = np.array([2.0, 5.0, 8.0])

    _fit, velocity, _statistics = fit_ts(xdata, ydata, np.ones(3))

    assert np.isclose(velocity[0], 3.0)


def test_linear_time_series_should_detrend_to_zero_residuals():
    xdata = np.array([0.0, 365.25, 730.5])
    ydata = np.array([2.0, 5.0, 8.0])

    residuals, _velocity, _statistics = detrended(xdata, ydata, np.ones(3))

    assert np.allclose(residuals, 0.0)


def test_weighted_oracle_should_return_independently_derived_velocity():
    table, expected = load_weighted_fit_oracle()

    _fit, velocity, _statistics = fit_ts(
        table["days_since_2000"].to_numpy(),
        table["position_mm"].to_numpy(),
        table["sigma_mm"].to_numpy(),
    )

    assert np.isclose(velocity[0], expected["velocity_mm_per_year"], rtol=1e-12, atol=1e-12)


def test_weighted_oracle_should_return_independently_derived_intercept():
    table, expected = load_weighted_fit_oracle()

    fit, _velocity, _statistics = fit_ts(
        table["days_since_2000"].to_numpy(),
        table["position_mm"].to_numpy(),
        table["sigma_mm"].to_numpy(),
    )

    assert np.isclose(fit[1], expected["intercept_mm"], rtol=1e-12, atol=1e-12)


def test_weighted_oracle_should_return_independently_derived_velocity_sigma():
    table, expected = load_weighted_fit_oracle()

    _fit, velocity, _statistics = fit_ts(
        table["days_since_2000"].to_numpy(),
        table["position_mm"].to_numpy(),
        table["sigma_mm"].to_numpy(),
    )

    assert np.isclose(velocity[1], expected["velocity_sigma_mm_per_year"], rtol=1e-12, atol=1e-12)


def test_weighted_oracle_should_return_independently_derived_chi():
    table, expected = load_weighted_fit_oracle()

    _fit, _velocity, statistics = fit_ts(
        table["days_since_2000"].to_numpy(),
        table["position_mm"].to_numpy(),
        table["sigma_mm"].to_numpy(),
    )

    assert np.isclose(statistics[1], expected["chi"], rtol=1e-12, atol=1e-12)


def test_weighted_oracle_should_return_independently_derived_wrms():
    table, expected = load_weighted_fit_oracle()

    _fit, _velocity, statistics = fit_ts(
        table["days_since_2000"].to_numpy(),
        table["position_mm"].to_numpy(),
        table["sigma_mm"].to_numpy(),
    )

    assert np.isclose(statistics[0], expected["wrms_mm"], rtol=1e-12, atol=1e-12)


def test_weighted_oracle_should_return_independently_derived_residuals():
    table, expected = load_weighted_fit_oracle()

    residuals, _velocity, _statistics = detrended(
        table["days_since_2000"].to_numpy(),
        table["position_mm"].to_numpy(),
        table["sigma_mm"].to_numpy(),
    )

    assert np.allclose(residuals, expected["residuals_mm"], rtol=1e-12, atol=1e-12)


def test_weighted_oracle_should_report_fixture_observation_count():
    table, _expected = load_weighted_fit_oracle()

    _fit, _velocity, statistics = fit_ts(
        table["days_since_2000"].to_numpy(),
        table["position_mm"].to_numpy(),
        table["sigma_mm"].to_numpy(),
    )

    assert statistics[2] == len(table)


def test_validation_manifest_should_match_reviewed_digest():
    result = file_sha256(VALIDATION_MANIFEST_PATH)

    assert result == REVIEWED_MANIFEST_SHA256


@pytest.mark.parametrize("fixture_id", REGISTERED_FIXTURE_IDS)
def test_registered_fixture_should_have_complete_identity(fixture_id):
    manifest = load_validation_manifest()
    fixture = validation_fixture(manifest, fixture_id)

    missing_fields = REQUIRED_FIXTURE_IDENTITY_FIELDS - fixture.keys()

    assert missing_fields == set()


@pytest.mark.parametrize("fixture_id", REGISTERED_FIXTURE_IDS)
def test_registered_fixture_should_not_have_blank_text_identity(fixture_id):
    manifest = load_validation_manifest()
    fixture = validation_fixture(manifest, fixture_id)

    result = all(
        isinstance(fixture[field], str) and bool(fixture[field].strip())
        for field in REQUIRED_TEXT_IDENTITY_FIELDS
    )

    assert result


@pytest.mark.parametrize("fixture_id", REGISTERED_FIXTURE_IDS)
def test_registered_fixture_should_use_timezone_aware_retrieval_time(fixture_id):
    manifest = load_validation_manifest()
    fixture = validation_fixture(manifest, fixture_id)
    timestamp = fixture["retrieved_at"]

    parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

    assert parsed_timestamp.utcoffset() is not None


@pytest.mark.parametrize("fixture_id", EXTERNAL_FIXTURE_IDS)
def test_registered_external_fixture_should_identify_https_source(fixture_id):
    manifest = load_validation_manifest()
    fixture = validation_fixture(manifest, fixture_id)
    source_url = fixture["source_url"]

    parsed_url = urlparse(source_url)

    assert parsed_url.scheme == "https" and bool(parsed_url.netloc)


@pytest.mark.parametrize("fixture_id", SYNTHETIC_FIXTURE_IDS)
def test_registered_synthetic_fixture_should_separate_method_reference_from_source(fixture_id):
    manifest = load_validation_manifest()
    fixture = validation_fixture(manifest, fixture_id)
    method_url = urlparse(fixture["method_reference_url"])

    result = (
        fixture["source_url"] == "not-applicable-synthetic"
        and method_url.scheme == "https"
        and bool(method_url.netloc)
    )

    assert result


@pytest.mark.parametrize("fixture_id", REGISTERED_FIXTURE_IDS)
def test_registered_fixture_should_remain_inside_frozen_fixture_directory(fixture_id):
    manifest = load_validation_manifest()
    fixture = validation_fixture(manifest, fixture_id)
    fixture_path = (VALIDATION_DIR / fixture["file"]).resolve()
    frozen_directory = (VALIDATION_DIR / "fixtures").resolve()

    assert fixture_path.is_relative_to(frozen_directory)


@pytest.mark.parametrize("fixture_id", REGISTERED_FIXTURE_IDS)
def test_registered_fixture_should_match_reviewed_digest(fixture_id):
    manifest = load_validation_manifest()
    fixture = validation_fixture(manifest, fixture_id)
    fixture_path = VALIDATION_DIR / fixture["file"]

    result = file_sha256(fixture_path)

    assert result == fixture["sha256"]


@pytest.mark.parametrize("fixture_id", REGISTERED_FIXTURE_IDS)
def test_registered_fixture_should_match_reviewed_byte_count(fixture_id):
    manifest = load_validation_manifest()
    fixture = validation_fixture(manifest, fixture_id)
    fixture_path = VALIDATION_DIR / fixture["file"]

    result = fixture_path.stat().st_size

    assert result == fixture["byte_count"]


@pytest.mark.parametrize("fixture_id", REGISTERED_FIXTURE_IDS)
def test_registered_fixture_should_match_reviewed_row_count(fixture_id):
    manifest = load_validation_manifest()
    fixture = validation_fixture(manifest, fixture_id)
    fixture_path = VALIDATION_DIR / fixture["file"]

    result = len(pd.read_csv(fixture_path))

    assert result == fixture["data_row_count"]


def test_northeast_velocity_should_end_northeast_of_station():
    center = [40.0, -75.0]

    endpoint = velocity_endpoint(center, [3.0, 4.0, 0.0], 10.0)

    assert endpoint[0] > center[0] and endpoint[1] > center[1]


def test_velocity_endpoint_should_match_scaled_horizontal_velocity_distance():
    center = [40.0, -75.0]

    endpoint = velocity_endpoint(center, [3.0, 4.0, 0.0], 10.0)

    assert np.isclose(geodesic(center, endpoint).km, 50.0, atol=0.01)


def test_velocity_pixel_scale_uses_reference_zoom_and_is_zoom_invariant():
    pixels = velocity_pixels_per_mm_per_year(10.0, 40.0, reference_zoom=4)

    assert np.isclose(pixels, 10_000.0 / web_mercator_meters_per_pixel(40.0, 4))
    assert np.isclose(pixels, velocity_pixels_per_mm_per_year(10.0, 40.0, reference_zoom=4))


def test_velocity_arrow_geometry_preserves_direction_and_fixed_arrowhead():
    north = velocity_arrow_geometry([3.0, 4.0, 0.0], 10.0, 40.0)
    south = velocity_arrow_geometry([-3.0, -4.0, 0.0], 10.0, 40.0)

    assert north["unit_x"] > 0 and north["unit_y"] < 0
    assert south["unit_x"] < 0 and south["unit_y"] > 0
    assert north["arrowhead_length_px"] == south["arrowhead_length_px"] == 8.0
    assert north["arrowhead_half_width_px"] == south["arrowhead_half_width_px"] == 4.0


def test_absolute_vector_render_spec_is_bounded_and_keyed_by_source_inputs():
    clear_absolute_vector_render_cache()
    args = (
        "NGF", "P049", (40.0, -75.0), (1.0, 2.0, 3.0),
        (0.1, 0.2, 0.3), (1.0, 1.0, 2.0), 5.0, 10.0, "#000000", "P049",
    )
    first = absolute_vector_render_spec(*args)
    second = absolute_vector_render_spec(*args)
    changed = absolute_vector_render_spec(*args[:-2], "#000001", "P049")

    info = absolute_vector_render_spec.cache_info()
    assert first is second
    assert info.hits == 1 and info.misses == 2
    assert first.svg != changed.svg
    assert "leaflet-div-icon" not in first.svg and ":has(" not in first.svg
    assert "background:transparent" in first.label_html
    escaped = absolute_vector_render_spec(*args[:-1], "<P049>").label_html
    assert "&lt;P049&gt;" in escaped and "<P049>" not in escaped
    assert info.maxsize == 8192


def test_relative_vector_render_spec_bypasses_absolute_cache():
    clear_absolute_vector_render_cache()
    relative_vector_render_spec([40.0, -75.0], [1.0, 2.0, 3.0], [0.1, 0.2, 0.3], [1.0, 1.0, 2.0], 5.0, 10.0, "#000000")
    assert absolute_vector_render_spec.cache_info().currsize == 0


def test_notebook_vector_path_uses_explicit_transparent_divicon_classes_without_popups():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    source = next(cell.source for cell in notebook.cells if "def draw_vector" in cell.get("source", ""))
    source = source.split("def draw_vector", 1)[1]
    assert "class_name='gnss-vector-div-icon'" in source
    assert "class_name='gnss-vector-label-div-icon'" in source
    assert "Popup" not in source and "station_popup" not in source

    imports = next(cell.source for cell in notebook.cells if "class GNSSDivIcon" in cell.get("source", ""))
    assert 'class_name = Unicode("").tag(sync=True, o=True)' in imports
    assert "className" not in imports
    assert "class_name=class_name" in imports


def test_vector_pane_is_lower_and_noninteractive_while_station_markers_stay_above():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert "VECTOR_PANE_NAME = 'gnss-vector-pane'" in source
    assert "STATION_PANE_NAME = 'gnss-station-pane'" in source
    assert "VECTOR_PANE_NAME: {'zIndex': 410, 'pointerEvents': 'none'}" in source
    assert "STATION_PANE_NAME: {'zIndex': 620, 'pointerEvents': 'auto'}" in source
    assert source.count("pane=VECTOR_PANE_NAME") >= 2
    assert source.count("pane=STATION_PANE_NAME") >= 2
    assert "panes=dict(GNSS_MAP_PANES)" in source
    assert ".gnss-vector-div-icon,.gnss-vector-label-div-icon{pointer-events:none!important" in source

    panes = {
        "gnss-vector-pane": {"zIndex": 410, "pointerEvents": "none"},
        "gnss-station-pane": {"zIndex": 620, "pointerEvents": "auto"},
    }
    map_widget = Map(panes=panes)
    vector = Marker(location=(40.0, -75.0), pane="gnss-vector-pane", keyboard=False)
    station = CircleMarker(location=(40.0, -75.0), pane="gnss-station-pane")
    assert map_widget.panes[vector.pane]["pointerEvents"] == "none"
    assert map_widget.panes[vector.pane]["zIndex"] < map_widget.panes[station.pane]["zIndex"]


def test_velocity_guide_stays_screen_relative_but_reports_zoom_equivalent():
    zoom4 = velocity_guide_metrics(10.0, 10.0, 40.0, current_zoom=4)
    zoom5 = velocity_guide_metrics(10.0, 10.0, 40.0, current_zoom=5)

    assert np.isclose(zoom4["bar_px"], zoom5["bar_px"])
    assert np.isclose(zoom5["current_km"], zoom4["current_km"] / 2.0)


def test_velocity_guide_bar_is_pan_invariant_but_current_equivalent_uses_center_latitude():
    low_latitude = velocity_guide_metrics(10.0, 10.0, 10.0, current_zoom=4)
    high_latitude = velocity_guide_metrics(10.0, 10.0, 60.0, current_zoom=4)

    assert np.isclose(low_latitude["bar_px"], high_latitude["bar_px"])
    assert not np.isclose(low_latitude["current_km"], high_latitude["current_km"])
    assert VELOCITY_REFERENCE_LATITUDE == 40.0


def test_display_formatting_rounds_and_preserves_trailing_zeros():
    assert format_display_number(1.235) == "1.24"
    assert format_display_number(1) == "1.00"
    assert format_display_number(None) == "—"
    assert format_neu_vector([1, -2.5, 3.456]) == "1.00, -2.50, 3.46"
    assert format_neu_vector(None) == "—"


def test_neighbor_velocity_row_calculates_relative_neu_and_handles_missing_data():
    row = neighbor_velocity_row(
        "AB01", "NGF", 12.345,
        {"velocity": [4.0, 5.0, 6.0], "velsig": [0.1, 0.2, 0.3]},
        {"velocity": [1.0, 2.0, 3.0]},
    )
    missing = neighbor_velocity_row("AB02", "NGF", 20.0, {}, {"velocity": [1.0, 2.0, 3.0]})

    assert row["differential"] == [3.0, 3.0, 3.0]
    assert row["velocity"] == [4.0, 5.0, 6.0]
    assert row["sigma"] == [0.1, 0.2, 0.3]
    assert missing["velocity"] is None and missing["sigma"] is None and missing["differential"] is None


def test_notebook_popup_path_caches_layer_and_explicitly_opens_on_click():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert "Popup(" in source
    assert "marker._gnss_popup_layer" in source
    assert "popup_layer.open_popup(marker.location)" in source
    assert "popup_target = resolve_popup_owner(map_obj, popup_owner)" in source
    assert "popup_owner=resolve_popup_owner(map_obj, map_dynamic_layers[0])" in source
    assert "event_type = _kwargs.get(\"type\")" in source
    assert "if event_type == \"click\":" in source


def test_station_popup_metadata_is_bounded_wrapping_and_escaped():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    cell_source = next(cell.source for cell in notebook.cells if "def station_popup" in cell.get("source", ""))
    popup_body = cell_source.split("def station_popup", 1)[1].split("def plot_site_locations", 1)[0]
    popup_path = cell_source.split("def plot_site_locations", 1)[1]

    assert "wg.Textarea" not in popup_body and "<pre" not in popup_body.lower()
    assert "white-space:nowrap" not in popup_body.replace(" ", "")
    assert "overflow-wrap:anywhere" in popup_body and "word-break:break-word" in popup_body
    assert "white-space:normal" in popup_body
    assert "grid-template-columns:minmax(7rem,auto) minmax(0,1fr)" in popup_body
    assert "html_module.escape" in popup_body
    assert "width='100%', max_width='100%', overflow='hidden'" in popup_body
    assert "min_width=280" in popup_path
    assert "max_width=460" in popup_path
    assert "max_height=360" in popup_path
    assert "popup_layer.open_popup(marker.location)" in popup_path


def test_popup_owner_should_follow_visible_group_when_rendering_is_staged():
    staged_target = object()
    visible_target = object()

    assert resolve_popup_owner(staged_target, visible_target) is visible_target
    assert resolve_popup_owner(staged_target) is staged_target


def test_vector_addition_should_add_matching_components():
    first = [1.0, 2.0, 3.0]
    second = [4.0, 5.0, 6.0]

    result = vec_add(first, second)

    assert result == [5.0, 7.0, 9.0]


def test_vector_subtraction_should_subtract_matching_components():
    first = [5.0, 7.0, 9.0]
    second = [4.0, 5.0, 6.0]

    result = vec_sub(first, second)

    assert result == [1.0, 2.0, 3.0]


def test_station_display_label_should_return_normalized_station_and_source():
    label = " p049 (unr) "

    result = remove_brac(label)

    assert result == ("P049", "UNR")


def test_closed_range_should_select_both_endpoints():
    text = "[1, 4]"

    result = parse_bulk_selection(text)

    assert result == {1, 2, 3, 4}


def test_open_range_should_exclude_both_endpoints():
    text = "(1, 4)"

    result = parse_bulk_selection(text)

    assert result == {2, 3}


def test_mixed_batch_selection_should_combine_ranges_and_individual_numbers():
    text = "[1, 8], 12"

    result = parse_bulk_selection(text)

    assert result == set(range(1, 9)) | {12}


def test_selection_options_should_keep_same_station_from_different_sources():
    options = ["P049, NGF, 1000km", "P049, UNR, 0km"]

    result = parse_map_selection(options)

    assert result == {("NGF", "P049"): 1000, ("UNR", "P049"): 0}


def test_map_plan_should_exclude_radius_center_from_neighbor_vector_requests():
    selections = {("NGF", "P049"): 1000}
    nearby = {("NGF", "P049"): [("P049", 0.0, 0.0, 0.0), ("MDR6", 0.0, 0.0, 1.0)]}

    result = build_map_render_plan(selections, nearby, plot_vectors=True, decimation=1)

    assert result.vector_keys == (("NGF", "P049"), ("NGF", "MDR6"))


def test_map_plan_should_deduplicate_overlapping_station_requests():
    selections = {("NGF", "P049"): 1000, ("NGF", "P050"): 1000}
    nearby = {
        ("NGF", "P049"): [("MDR6", 0.0, 0.0, 1.0)],
        ("NGF", "P050"): [("MDR6", 0.0, 0.0, 1.0)],
    }

    result = build_map_render_plan(selections, nearby, plot_vectors=True, decimation=1)

    assert result.vector_keys.count(("NGF", "MDR6")) == 1


def test_staged_update_should_keep_visible_layers_until_render_finishes():
    visible_layers = LayerGroup()
    old_marker = CircleMarker(location=[40.0, -100.0])
    new_marker = CircleMarker(location=[41.0, -101.0])
    visible_layers_during_render = []
    visible_layers.layers = (old_marker,)

    def render(staged_layers):
        visible_layers_during_render.append(visible_layers.layers)
        staged_layers.add(new_marker)

    stage_layer_update(visible_layers, render)

    assert visible_layers_during_render == [(old_marker,)]


def test_completed_staged_update_should_replace_visible_layers():
    visible_layers = LayerGroup()
    old_marker = CircleMarker(location=[40.0, -100.0])
    new_marker = CircleMarker(location=[41.0, -101.0])
    visible_layers.layers = (old_marker,)

    def render(staged_layers):
        staged_layers.add(new_marker)

    stage_layer_update(visible_layers, render)

    assert visible_layers.layers == (new_marker,)


def test_many_staged_markers_should_trigger_one_visible_layer_update():
    visible_layers = LayerGroup()
    visible_updates = []
    marker_count = 300
    visible_layers.observe(lambda change: visible_updates.append(change["new"]), names="layers")

    def render(staged_layers):
        for number in range(marker_count):
            staged_layers.add(CircleMarker(location=[40.0 + number / 10000, -100.0]))

    stage_layer_update(visible_layers, render)

    assert len(visible_updates) == 1


@pytest.mark.e2e
def test_notebook_when_executed_should_complete_without_error():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    oracle_table, oracle_expected = load_weighted_fit_oracle()
    oracle_probe = f"""
_probe_xdata = np.array({oracle_table['days_since_2000'].tolist()!r}, dtype=float)
_probe_ydata = np.array({oracle_table['position_mm'].tolist()!r}, dtype=float)
_probe_sig = np.array({oracle_table['sigma_mm'].tolist()!r}, dtype=float)
_probe_fit, _probe_velocity, _probe_statistics = FitTS(_probe_xdata, _probe_ydata, _probe_sig)
_probe_residuals, _probe_detrended_velocity, _probe_detrended_statistics = detrended(
    _probe_xdata, _probe_ydata, _probe_sig
)
assert np.isclose(_probe_fit[1], {oracle_expected['intercept_mm']!r}, rtol=1e-12, atol=1e-12)
assert np.isclose(_probe_fit[0], {oracle_expected['velocity_mm_per_year']!r}, rtol=1e-12, atol=1e-12)
assert np.isclose(_probe_velocity[0], {oracle_expected['velocity_mm_per_year']!r}, rtol=1e-12, atol=1e-12)
assert np.isclose(_probe_velocity[1], {oracle_expected['velocity_sigma_mm_per_year']!r}, rtol=1e-12, atol=1e-12)
assert np.isclose(_probe_statistics[0], {oracle_expected['wrms_mm']!r}, rtol=1e-12, atol=1e-12)
assert np.isclose(_probe_statistics[1], {oracle_expected['chi']!r}, rtol=1e-12, atol=1e-12)
assert _probe_statistics[2] == {len(oracle_table)}
assert np.allclose(_probe_residuals, {oracle_expected['residuals_mm']!r}, rtol=1e-12, atol=1e-12)
assert np.allclose(_probe_residuals, _probe_ydata - (_probe_fit[0] * _probe_xdata / 365.25 + _probe_fit[1]))
print('FITTS_ORACLE_PROBE_OK')
"""
    notebook.cells.append(nbformat.v4.new_code_cell(oracle_probe))
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )

    client.execute()

    assert any(
        output.get("text", "").strip() == "FITTS_ORACLE_PROBE_OK"
        for output in notebook.cells[-1].get("outputs", [])
        if output.output_type == "stream"
    )

from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
from geopy.distance import geodesic
from ipyleaflet import CircleMarker, LayerGroup
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
)
from gnss_map_runtime import stage_layer_update


NOTEBOOK_PATH = Path(__file__).parent / "GNSS_Analysis_ipyleaflet.ipynb"


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


def test_northeast_velocity_should_end_northeast_of_station():
    center = [40.0, -75.0]

    endpoint = velocity_endpoint(center, [3.0, 4.0, 0.0], 10.0)

    assert endpoint[0] > center[0] and endpoint[1] > center[1]


def test_velocity_endpoint_should_match_scaled_horizontal_velocity_distance():
    center = [40.0, -75.0]

    endpoint = velocity_endpoint(center, [3.0, 4.0, 0.0], 10.0)

    assert np.isclose(geodesic(center, endpoint).km, 50.0, atol=0.01)


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
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )

    client.execute()

    assert notebook.cells

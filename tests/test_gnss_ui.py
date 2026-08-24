# ---- file preamble for tests/test_gnss_ui.py (module-level imports and helpers) ----
import re
from pathlib import Path

import ipywidgets as wg
import nbformat
import pytest

from gnss_ui import (
    GNSS_ACCENT, GNSS_CSS_TEXT, action_grid, assert_reachable, bind, bind_click,
    break_counts, catalogue_rows, field_row, list_counts, log_well, match_codes,
    panel, plot_preview_svg, section, sigma_state, slider_row, stat_grid_html,
    station_detail_pairs, window_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "GNSS_Analysis_ipyleaflet.ipynb"
GNSS_UI_PATH = REPO_ROOT / "gnss_ui.py"

RESTYLED_CONTROLS = """
update_NGF_butt update_UNR_butt update_JPL_butt fetcher_output breaks_select
add_breaks_site_button remove_breaks_site_button breaks_update_output clear_fetch_log_butt
site_searchbar org_avail_select site_search_submit clear_log_butt availability_output
availability_detail ts_site_form org_ts_select append_butt live_update_check ts_sites
ts_filter_form plot_ts_butt show_breaks_data_button remove_site_button clear_list_butt
close_ts_butt id_button start_year_form end_year_form siglim_form remove_outliers
remove_breaks_checkbox detrend_check error_bar_check error_bar_outline_check
error_bar_opacity thin_error_bars add_breaks_ts_button remove_breaks_ts_button shift_value
update_customization shift_output backend_butt backend_activate breaks_data_output
timeseries_output plot_ts_res ts_preview ts_ribbon ts_results_ribbon
""".split()


def _cell(notebook, needle):
    return next(cell["source"] for cell in notebook.cells if needle in cell.get("source", ""))


def _restyled_slices(windows_source):
    """Returns the fetch/availability region and the timeseries region only.

    The map region between them still legitimately carries '145px', '36px', and
    the clamp height, so every rule below is scoped away from it.
    """
    head = windows_source.split("render_site_table()", 1)[0]
    tail = windows_source.split("ts_list_meter = ui.meter()", 1)[1]
    return head, tail


def test_restyled_windows_are_titled_panels_not_bare_vboxes():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    windows = _cell(notebook, "fetch_window = VBox")
    head, tail = _restyled_slices(windows)
    for title in ("'Database Fetcher'", "'Station Availability'", "'Timeseries'",
                  "'Timeseries Output'"):
        assert "ui.panel(%s" % title in windows
    assert windows.count("ui.stylesheet()") == 4
    assert "<em style=\"color:blue" not in head
    assert "<h1>Availability</h1>" not in head
    assert "<h1>Timeseries</h1>" not in tail
    assert "<ul>" not in head
    assert head.count("ui.section(") >= 3 and tail.count("ui.section(") >= 10
    assert "ui.dashboard(" in tail

def test_restyled_windows_use_relative_units_only():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    windows = _cell(notebook, "fetch_window = VBox")
    head, tail = _restyled_slices(windows)
    declarations = "".join(
        _cell(notebook, needle)
        for needle in ("update_NGF_butt     = wg.Button", "site_searchbar      = wg.Text",
                       "ts_site_form        = wg.Text")
    )
    pattern = r"(?:width|height|min_width|min_height|max_width|max_height)='(\d+)px'"
    for name, source in (("fetch/availability", head), ("timeseries", tail),
                         ("widget declarations", declarations)):
        assert re.findall(pattern, source) == [], name
    assert "layout(\"150px\")" not in declarations
    assert "layout(\"200px\")" not in declarations
    assert "width='100px'" not in declarations

def test_every_restyled_control_survives_the_regroup():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    windows = _cell(notebook, "fetch_window = VBox")
    missing = [name for name in RESTYLED_CONTROLS if not re.search(r"\b%s\b" % name, windows)]
    assert missing == []

def test_restyled_windows_assert_their_own_reachability():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    windows = _cell(notebook, "fetch_window = VBox")
    assert windows.count("ui.assert_reachable(") == 4
    for window in ("'fetch_window'", "'availability_window'", "'timeseries_window'",
                   "'timeseries_output_window'"):
        assert window in windows

def test_lazy_slider_pairs_extend_to_the_timeseries_window():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    windows = _cell(notebook, "fetch_window = VBox")
    declarations = _cell(notebook, "ts_site_form        = wg.Text")
    assert windows.count("wg.jslink(") == 4          # the four map pairs, unchanged
    assert windows.count("ui.slider_row(") == 3      # shift, every-Nth bar, envelope alpha
    for name in ("shift_slider", "thin_error_bars_slider", "error_bar_opacity_slider"):
        assert name in declarations and name in windows
    assert "shift_value        = wg.BoundedFloatText" in declarations
    assert "thin_error_bars    = wg.BoundedIntText" in declarations
    assert "error_bar_opacity  = wg.BoundedFloatText" in declarations

def test_palette_is_applied_and_legacy_button_colors_are_gone():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    declaration_cells = "".join(
        _cell(notebook, needle)
        for needle in ("update_NGF_butt     = wg.Button", "site_searchbar      = wg.Text",
                       "site_table          = VBox", "ts_site_form        = wg.Text")
    )
    for legacy in ("rgb(196,253,196)", "mistyrose", "paleturquoise", "springgreen",
                   "oldlace", "lightgoldenrodyellow"):
        assert legacy not in declaration_cells, legacy
    assert GNSS_ACCENT in declaration_cells
    assert "plot_vec_check.add_class('gnss-accent-toggle')" in declaration_cells
    guide = _cell(notebook, "def update_vector_guide")
    assert "border-top:3px solid #111" in guide      # test-locked; never repainted

def test_windows_cell_never_uses_a_layout_kwarg_ipywidgets_discards():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    windows = _cell(notebook, "fetch_window = VBox")
    head, tail = _restyled_slices(windows)
    for source in (head, tail, GNSS_UI_PATH.read_text()):
        assert "box_sizing=" not in source
        assert "flex_wrap=" not in source
        assert "overflow_x=" not in source and "overflow_y=" not in source
    assert re.search(r"[^d_]gap=", head) is None
    assert re.search(r"[^d_]gap=", tail) is None
    assert wg.Layout(grid_gap="0.4rem").get_state()["grid_gap"] == "0.4rem"
    assert "gap" not in wg.Layout().trait_names()

def test_notebook_source_has_not_been_autoformatted():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    declarations = _cell(notebook, "site_table          = VBox")
    assert "site_table          = VBox" in declarations
    aligned = [line for line in declarations.splitlines() if re.match(r"^\w+\s{2,}= ", line)]
    assert len(aligned) >= 10

def test_gnss_ui_sigma_state_separates_unparseable_text_from_a_default():
    assert sigma_state("", [10, 10, 30]) == ((10, 10, 30), True, "inherited")
    assert sigma_state("(5,5,15)", [10, 10, 30]) == ((5.0, 5.0, 15.0), True, "pending")
    assert sigma_state("(10,10,30)", [10, 10, 30]) == ((10.0, 10.0, 30.0), True, "pending")
    assert sigma_state("garbage", [10, 10, 30]) == ((10, 10, 30), False, "invalid")
    assert sigma_state("(1,2)", [10, 10, 30]) == ((10, 10, 30), False, "invalid")

def test_gnss_ui_window_summary_matches_the_plot_handler_parsing():
    assert window_summary("1999-01-01", "2024-06-30") == ("1999-01-01 to 2024-06-30 (25.49 yr)", True)
    assert window_summary("", "") == ("full record", True)
    assert window_summary("2024-01-01", "1999-01-01")[1] is False
    assert window_summary("1999-13-01", "")[1] is False
    # strptime rejects padding, so the readout must not strip and claim validity.
    assert window_summary(" 2015-01-01", "")[1] is False

def test_gnss_ui_quantifiers_count_the_working_set_and_its_breaks():
    options = ["P040 (NGF)", "P041 (NGF)", "P040 (UNR)"]
    total, chosen, per, per_selected, unresolved = list_counts(
        options, ["P040 (NGF)"], ["NGF", "UNR", "JPL"])
    assert (total, chosen, unresolved) == (3, 1, 0)
    assert per == {"NGF": 2, "UNR": 1, "JPL": 0}
    assert per_selected == {"NGF": 1, "UNR": 0, "JPL": 0}
    data_of = {"NGF": {"P040": {"breaks": {"a": {}, "b": {}}}, "P041": {}}, "UNR": {"P040": {}}}
    assert break_counts(options, ["P040 (NGF)"], data_of) == (1, 3, 1, 2)
    assert catalogue_rows(data_of, ["NGF", "UNR", "JPL"]) == [("NGF", 2, 1), ("UNR", 1, 0), ("JPL", 0, 0)]
    assert match_codes(["P040", "P041", "AB12"], "p04") == (2, ["P040", "P041"], False)
    assert match_codes(["P040"], "P040")[2] is True
    assert match_codes(options, "unr", contains=True)[0] == 1

def test_gnss_ui_data_view_labels_units_and_escapes_catalog_text():
    record = {"location": [40.1234567, -110.7654321], "height": 1234.5678,
              "velocity": [1.234, -3.456, 0.111], "velsig": [0.1, 0.1, 0.3],
              "stntype": "<script>", "breaks": {"a": {}}}
    pairs = dict(station_detail_pairs(record))
    assert pairs["Location (lat, lon) (deg)"] == "40.1234567, -110.7654321"
    assert pairs["Height (m)"] == "1234.5678"
    assert pairs["Velocity N,E,U (mm/yr)"] == "1.23, -3.46, 0.11"
    assert pairs["Breaks (epochs)"] == "1"
    markup = stat_grid_html(station_detail_pairs(record))
    assert "&lt;script&gt;" in markup and "<script>" not in markup
    assert 'class="gnss-kv"' in markup
    assert "No data loaded." in stat_grid_html([])

def test_gnss_ui_plot_preview_reflects_the_live_styling_controls():
    plain = plot_preview_svg(False, 1, False, 0.0, False, True, 0)
    styled = plot_preview_svg(True, 1, True, 0.4, True, False, 5.0)
    assert "viewBox" in plain
    # the 1px hairline is the one justified absolute unit in the whole system
    assert re.findall(r"(\d+)px", plain) == ["1"]
    assert plain.count("<line") < styled.count("<line")      # error bars and break markers
    assert 'fill-opacity="0.4"' in styled and "fill-opacity" not in plain
    assert "detrended" in plain and "shift +5 mm (Up +15)" in styled

def test_gnss_ui_components_emit_only_real_layout_traits():
    grid = action_grid([wg.Button(description="a"), wg.Button(description="b")], columns=2)
    assert grid.layout.get_state()["grid_gap"]
    shell = panel("T", [section("S", [field_row("Label:", wg.Text())])], subtitle="sub")
    states = [widget.layout.get_state() for widget in (shell, shell.header, shell.body)]
    assert all("border_top" in states[0] for _ in [0])
    assert states[1]["border_bottom"]
    assert all("px" not in str(value) for state in states for key, value in state.items()
               if key in ("width", "padding", "margin", "max_width"))

def test_gnss_ui_slider_row_refuses_bounds_that_jslink_would_silently_clamp():
    good = slider_row("Shift (mm):", wg.FloatSlider(min=-5, max=5, step=0.5),
                      wg.BoundedFloatText(min=-5, max=5, step=0.5))
    assert good.slider.readout is False and good.slider.description == ""
    assert good.value_input.layout.width == "20%"
    with pytest.raises(ValueError):
        slider_row("x", wg.IntSlider(min=0, max=5), wg.BoundedIntText(min=0, max=9))

def test_gnss_ui_bind_survives_a_cell_rerun_without_stacking_handlers():
    checkbox = wg.Checkbox(value=False)
    button = wg.Button()
    calls = []
    for _ in range(4):
        def refresh(_event=None):
            calls.append(1)
        bind(checkbox, refresh)
        bind_click(button, refresh)
    checkbox.value = True
    button.click()
    assert calls == [1, 1]

def test_gnss_ui_assert_reachable_names_the_control_that_went_missing():
    kept = wg.Button(description="kept")
    dropped = wg.Button(description="dropped")
    root = wg.VBox([wg.VBox([kept])])
    assert assert_reachable(root, {"kept": kept}, "w") == 1
    with pytest.raises(AssertionError) as error:
        assert_reachable(root, {"kept": kept, "dropped": dropped}, "timeseries_window")
    assert "timeseries_window" in str(error.value) and "dropped" in str(error.value)

def test_gnss_ui_log_well_bounds_an_output_without_replacing_it():
    output = wg.Output()
    same = log_well(output, height="6rem")
    assert same is output                                  # `with output:` keeps working
    assert output.layout.max_height == "6rem"
    assert output.layout.overflow == "auto"
    assert "gnss-log" in output._dom_classes
    assert log_well(wg.Output(), height=None).layout.max_height is None

def test_gnss_ui_stylesheet_is_a_scoped_style_block_per_window():
    carrier = __import__("gnss_ui").stylesheet()
    assert carrier.value.strip().startswith("<style>")
    assert ".gnss-panel" in GNSS_CSS_TEXT and ".gnss-log" in GNSS_CSS_TEXT
    assert ".jupyter-widget-checkbox" in GNSS_CSS_TEXT      # theme ships both class spellings
    assert ".widget-checkbox" in GNSS_CSS_TEXT
    assert __import__("gnss_ui").stylesheet() is not carrier

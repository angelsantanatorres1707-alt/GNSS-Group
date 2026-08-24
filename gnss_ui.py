"""Panel-system palette, stylesheet, and widget components for the GNSS notebook.

This module owns every colour, spacing token, and container helper used by the
Database Fetcher, Availability, Timeseries, and Timeseries Output windows.  It
holds no notebook state: every function either returns a fresh widget or is a
pure function over plain Python values, so the whole system is importable and
unit-testable outside Jupyter.

Two ipywidgets facts shape the implementation and are load-bearing:

* ``gap`` and ``box_sizing`` are **not** ``Layout`` traits.  ``wg.Layout(gap='3%')``
  raises only a ``DeprecationWarning`` and the value never reaches the browser.
  All spacing here is delivered by ``margin`` (a real trait) and by ``grid_gap``
  on a ``GridBox`` (also a real trait).
* ``border`` is a *property* that fans out to the four ``border_*`` traits, so it
  does work and is used verbatim to match the approved map dashboard.
"""

from __future__ import annotations

import html as html_module
from datetime import datetime
from typing import Mapping, Sequence

import ipywidgets as wg
from ipywidgets import HBox, VBox

from gnss_core import format_display_number, format_neu_vector, number_list, remove_brac


# ---------------------------------------------------------------------------
# Palette.  Colour carries meaning: accent for the single primary action and
# active state, danger for non-undoable deletes, neutrals for everything else.
# ---------------------------------------------------------------------------

GNSS_ACCENT = "#1966FF"        # MIT EAPS blue; 4.79:1 on white, AA with white text
GNSS_ACCENT_WEAK = "#E8F0FF"   # selected-row / active-chip wash
GNSS_ACCENT_DEEP = "#0B3FA8"   # accent as legible body text (8.6:1 on the sunk surface)
GNSS_INK = "#1A1D21"           # primary text and data values
GNSS_INK_MUTED = "#5C636A"     # labels, units, hints, meter chrome
GNSS_RULE = "#D0D0D0"          # panel border; byte-identical to the map dashboard
GNSS_RULE_SOFT = "#E4E7EA"     # inner hairlines and table cells
GNSS_SURFACE = "#FFFFFF"       # panel ground
GNSS_SURFACE_SUNK = "#F5F6F8"  # header bars, log wells, ribbon
GNSS_NEUTRAL_BTN = "#ECEEF1"   # every secondary action
GNSS_DANGER = "#B42318"        # destructive text/border only, never a loud fill
GNSS_DANGER_WEAK = "#FDECEA"   # destructive button fill and error wash
GNSS_OK = "#0F7B3F"            # loaded / succeeded
GNSS_WARN = "#8A6100"          # "this control will currently do nothing"

# ``border`` fans out to border_top/right/bottom/left, so these are real CSS.
GNSS_BORDER = "1px solid " + GNSS_RULE
GNSS_BORDER_INNER = "1px solid " + GNSS_RULE_SOFT

# ---------------------------------------------------------------------------
# Spacing and proportion tokens.  Everything is %, rem, fr or flex.  The only
# absolute unit in the system is the 1px hairline inside the two border tokens
# above: 0.0625rem is sub-pixel and disappears at common zoom levels, so a
# hairline is the one place a pixel is the correct unit.
#
# Padding is rem, not the map's padding='3%', deliberately: percentage padding
# resolves against WIDTH on all four edges, so the map's 3% inside a 29% column
# renders ~11-17px while the same token inside a full-width panel renders ~38-58px.
# rem reproduces the map's *rendered* rhythm instead of its literal token.
# ---------------------------------------------------------------------------

GNSS_PANEL_PAD = "0.7rem 0.8rem 0.8rem 0.8rem"
GNSS_HEAD_PAD = "0.4rem 0.8rem"
GNSS_SECTION_PAD = "0.55rem 0.65rem 0.65rem 0.65rem"
GNSS_ROW_MARGIN = "0 0 0.35rem 0"       # replaces the dead gap='3%' between rows
GNSS_SECTION_MARGIN = "0 0 0.55rem 0"
GNSS_PANEL_MARGIN = "0 0 0.9rem 0"
GNSS_GRID_GAP = "0.4rem"                # real trait, unlike `gap`
GNSS_CONTROL_H = "1.9rem"               # replaces the lambda's baked-in 40px
GNSS_WINDOW_MAX = "72rem"               # keeps a 3-button panel off a 3440px monitor

GNSS_LABEL_W = "38%"   # identical to the map's left_label_layout
GNSS_FIELD_W = "58%"   # 38 + 58 = 96%, leaving the slack the map already leaves
GNSS_SLIDER_W = "34%"  # identical to the map's vector_slider_layout
GNSS_VALUE_W = "20%"   # widened from the map's 18%: 18% of a 29% column is 53px
                       # at a 1024px viewport, under the ~60px a spinner needs.

GNSS_TONES = {
    "muted": GNSS_INK_MUTED,
    "ok": GNSS_OK,
    "warn": GNSS_WARN,
    "bad": GNSS_DANGER,
    "accent": GNSS_ACCENT_DEEP,
}


# ---------------------------------------------------------------------------
# Stylesheet.  Carries only what Layout traits cannot express, using the scoped
# <style> escape hatch the notebook already sanctions via table_cell_css.
# ---------------------------------------------------------------------------

GNSS_CSS_TEXT = """
<style>
/* GNSS panel system.  Structural rules (colours, borders, which widgets are
   destructive) are scoped under a gnss- class so nothing leaks into the map
   window. Text size is NOT scoped that way - see the fluid-type block below.

   ipywidgets 8 ships every rule twice, as `.widget-x` and `.jupyter-widget-x`;
   the runtime DOM currently carries the un-prefixed form.  Overrides below are
   written for both so a future bundle cannot silently drop them. */

/* Fluid type: every size below is a CSS container-query length (cqi = 1%% of
   the nearest ancestor with container-type set), not vw/vh and not a bare
   rem. rem is fixed to the root font size no matter how small the notebook's
   OUTPUT CELL gets - shrink that cell and a rem-sized label stays full size
   while its %%-wide box shrinks around it, which is the clipping bug this
   replaces. vw has the opposite problem: it tracks the whole browser
   viewport, not the cell, so two side-by-side notebooks (or a split editor)
   would size their text identically regardless of how much room either one
   actually has. cqi tracks the one thing that should drive this: how much
   width THIS output currently has. body is the container because every
   window here (map included) is a separate top-level display() call with no
   shared wrapper except body - one declaration covers all of them, map
   included, without adding a single element to the DOM. inline-size (not
   full size/both axes) is deliberate: notebooks scroll vertically without
   limit, so a height-based query would keep shrinking text on a merely
   *tall* page, which nothing here wants. Every clamp() keeps its own
   ceiling so the type hierarchy (title > section > body > hint) survives at
   any width instead of converging to one size. */
body { container-type: inline-size; }

/* Each panel is also a container, and it is the one that matters: in JupyterLab
   `body` is the whole browser window, so a notebook in a narrow split pane
   would still resolve cqi against the full window and get maximum-size type -
   the exact case this system exists to handle. Declaring the panel a container
   means cqi inside it measures the panel, which tracks the output cell. `body`
   stays declared as the fallback for the map, which is not a gnss-panel. */
.gnss-panel { container-type: inline-size;
  color: %(ink)s; background: %(surface)s;
  font-size: clamp(0.69rem, 0.59rem + 0.31cqi, 0.9rem); }
.gnss-panel-head { background: %(sunk)s; }
.gnss-panel-title { font-weight: 700; letter-spacing: 0.01em; color: %(ink)s;
  font-size: clamp(0.76rem, 0.65rem + 0.35cqi, 1.0rem); }
.gnss-panel-sub { color: %(muted)s;
  font-size: clamp(0.59rem, 0.51rem + 0.27cqi, 0.78rem); }

.gnss-section-title { font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase; color: %(muted)s;
  font-size: clamp(0.56rem, 0.48rem + 0.25cqi, 0.73rem); }

/* The theme sets .widget-label { overflow:hidden; text-overflow:ellipsis;
   white-space:nowrap } and pins .widget-inline-hbox .widget-label to a fixed
   width AND a fixed height (--jp-widgets-inline-height, 28px) with
   flex-shrink:0.  That is why 'Scale (km/mm/yr):' used to clip to
   'Scale (km/mm...' below ~975px - on the map too, not just the four
   restyled windows, since a narrow VS Code notebook pane hits this on any
   label. white-space:normal alone trades that for a SECOND clip: the label
   wraps to two lines but the fixed height only shows the first one - the
   row's own height doesn't grow because a flex row sizes to its tallest
   child, and every child here is still individually pinned to 28px. height
   has to come off the label too, or the wrap never has anywhere to go. This
   rule is deliberately NOT scoped under .gnss-panel: every other structural
   rule in this sheet is, but a clipped label is a readability bug wherever
   it happens, and this notebook only ever has one map. */
.widget-label, .jupyter-widget-label {
  white-space: normal; overflow: visible; text-overflow: clip;
  height: auto !important; min-height: 1.6em;
  line-height: 1.2; color: %(ink)s;
  /* wrapping between words is not enough at the narrow end: the last
     offenders are single unbreakable tokens ('Resolution:', 'Backend:',
     '(km/mm/yr):') that no amount of white-space:normal can break.
     break-word, NOT anywhere: `anywhere` also shrinks the element's
     min-content contribution, so a flex row is then free to squeeze the
     label to one character wide and 'Station ID:' renders as a vertical
     stack of single letters. break-word breaks the same long tokens but
     leaves min-content at the longest word, which keeps the column honest. */
  overflow-wrap: break-word;
  font-size: clamp(0.66rem, 0.56rem + 0.3cqi, 0.86rem); }

.gnss-hint { line-height: 1.45; color: %(muted)s;
  font-size: clamp(0.61rem, 0.52rem + 0.28cqi, 0.8rem); }
.gnss-hint code { font-size: 0.95em; background: %(sunk)s; padding: 0 0.2em; }

/* .gnss-meter is always an inline <span>; text-align on an inline element is a
   no-op, so the alignment is set on its widget wrapper instead - the .gnss-meter-host
   class the meter() widget itself carries. */
.gnss-meter-host .widget-html-content, .gnss-meter-host .jupyter-widget-html-content {
  text-align: right; }
.gnss-meter { color: %(muted)s; line-height: 1.3;
  font-variant-numeric: tabular-nums;
  font-size: clamp(0.58rem, 0.49rem + 0.26cqi, 0.76rem); }
.gnss-meter b { color: %(accent)s; font-weight: 700; }
.gnss-meter .bad { color: %(danger)s; font-weight: 700; }
.gnss-meter .warn { color: %(warn)s; font-weight: 600; }
.gnss-meter .ok { color: %(ok)s; font-weight: 600; }

.gnss-chip { display: inline-block; font-weight: 600;
  padding: 0.02rem 0.4rem; border: 1px solid currentColor; border-radius: 0.75rem;
  line-height: 1.55; font-size: clamp(0.57rem, 0.49rem + 0.26cqi, 0.75rem); }

.gnss-ribbon { display: flex; flex-wrap: wrap; color: %(muted)s;
  font-variant-numeric: tabular-nums;
  font-size: clamp(0.58rem, 0.5rem + 0.27cqi, 0.77rem); }
.gnss-ribbon span { margin: 0 0.9rem 0.1rem 0; }
.gnss-ribbon b { color: %(ink)s; font-weight: 650; }

/* Data View: the label/value metadata grid.  Same grid idiom the station popup
   already uses, so the two metadata readouts agree visually. */
.gnss-kv { display: grid; column-gap: 0.7rem; row-gap: 0.22rem; margin: 0;
  line-height: 1.35; font-size: clamp(0.63rem, 0.54rem + 0.29cqi, 0.83rem); }
.gnss-kv dt { color: %(muted)s; }
.gnss-kv dd { margin: 0; color: %(ink)s; font-variant-numeric: tabular-nums;
  overflow-wrap: anywhere; word-break: break-word; }

/* Bounded, scrolling log wells.  Every Output in the four non-map windows is
   unbounded today and shoves the page down mid-download.  The table rules also
   pick up the break table and the detrended velocity frame, which are emitted
   as bare markup by the handler cells and carry no styling of their own. */
.gnss-log, .gnss-log .widget-output, .gnss-log .jupyter-widget-output,
.gnss-log .jp-OutputArea-output { font-size: clamp(0.62rem, 0.54rem + 0.28cqi, 0.82rem); }
.gnss-log table { border-collapse: collapse; width: 100%%; table-layout: fixed;
  font-size: clamp(0.56rem, 0.48rem + 0.25cqi, 0.73rem); }
.gnss-log th, .gnss-log td { border: 1px solid %(rule_soft)s; padding: 0.12em 0.35em;
  text-align: right; line-height: 1.25; overflow-wrap: anywhere; }
.gnss-log th { background: %(sunk)s; color: %(muted)s; font-weight: 700; text-align: center; }

/* A boolean is one line, not a 300px block.  The theme sets
   .widget-checkbox { width: var(--jp-widgets-inline-width) }, and five of those
   is the largest single waste of space in the timeseries window. */
.gnss-panel .widget-checkbox, .gnss-panel .jupyter-widget-checkbox {
  width: auto !important; min-width: 0 !important; height: auto !important;
  line-height: 1.3 !important; }
.gnss-panel .widget-checkbox label, .gnss-panel .jupyter-widget-checkbox label {
  white-space: normal; }

/* The radio group is flex-grow:1 by default and would push its column open. */
.gnss-panel .widget-radio-box, .gnss-panel .jupyter-widget-radio-box {
  flex-grow: 0 !important; }
.gnss-panel .widget-radio-box label, .gnss-panel .jupyter-widget-radio-box label {
  margin-right: 0.7rem; }

/* backend_butt uses orientation='horizontal'.  The theme styles radio inputs
   for the VERTICAL stack it assumes - `.widget-radio-box input` is
   display:block - so laid out horizontally the input takes its own line and
   pushes the option text to a second line inside a label the theme has
   pinned to one line's height (20px), clipping the text at every width.
   Making the input inline and letting the label and box size to content is
   what the horizontal orientation needs; scoped to the -horizontal modifier
   so a vertical radio group anywhere else keeps the theme's own layout. */
.widget-radio-box-horizontal, .jupyter-widget-radio-box-horizontal {
  height: auto !important; align-items: center; }
.widget-radio-box-horizontal label, .jupyter-widget-radio-box-horizontal label {
  display: inline-flex !important; align-items: center;
  height: auto !important; white-space: nowrap; }
.widget-radio-box-horizontal input, .jupyter-widget-radio-box-horizontal input {
  display: inline-block !important; flex: 0 0 auto; }

/* Same fixed-box clipping as .widget-label, same fix, and for the same
   reason not scoped to .gnss-panel: 'Add Breaks' truncates to 'Add...' on
   the map at a narrow width exactly like an unfixed label would. Button text
   is short enough that this rarely shows at a comfortable width - it only
   bites at the extreme end this fix exists for. */
/* Text handling is global, because 'Add Breaks' truncating to 'Add...' is a
   readability bug wherever it happens. The BOX is not: `height: auto` here
   would re-size the map's own 40px action buttons to ~22px, which is a layout
   change inside a window that is colors-only. Height is therefore scoped to
   the restyled panels, whose buttons this system sizes anyway. */
.jupyter-button, .widget-button {
  white-space: normal !important; overflow: visible; text-overflow: clip;
  overflow-wrap: break-word;
  font-size: clamp(0.66rem, 0.56rem + 0.3cqi, 0.86rem); }
.gnss-panel .jupyter-button, .gnss-panel .widget-button {
  height: auto !important; min-height: 1.9em; line-height: 1.15; }
.gnss-btn-danger button, button.gnss-btn-danger { border: 1px solid %(danger_edge)s !important; }

/* The velocity toggle is a ToggleButton, whose ButtonStyle carries no
   button_color trait at all, so its active state can only be accented here. */
/* add_class puts the class on the widget's own root element, and for a
   ToggleButton that root IS the <button> - so the descendant forms below never
   match anything on their own. The bare .mod-active form is the one that
   actually fires; the others are kept only in case a future bundle nests. */
.gnss-accent-toggle.mod-active,
.gnss-accent-toggle button.mod-active, .gnss-accent-toggle.mod-active button {
  background: %(accent_solid)s !important; color: #FFFFFF !important; font-weight: 600; }

/* Controls that are a documented no-op in the current state are dimmed, not
   disabled: nothing in the notebook ever disables them, so inventing a real
   disabled state would contradict the shipped documentation. */
.gnss-gated { opacity: 0.5; }
</style>
""" % {
    "ink": GNSS_INK,
    "muted": GNSS_INK_MUTED,
    "surface": GNSS_SURFACE,
    "sunk": GNSS_SURFACE_SUNK,
    "accent": GNSS_ACCENT_DEEP,
    "accent_solid": GNSS_ACCENT,
    "danger": GNSS_DANGER,
    "danger_edge": "#F0C4BF",
    "warn": GNSS_WARN,
    "ok": GNSS_OK,
    "rule_soft": GNSS_RULE_SOFT,
}


def stylesheet() -> wg.HTML:
    """Returns a fresh stylesheet carrier for one window.

    Each of the four windows is a separate display cell, and cell 41 runs before
    the map cell, so every window carries its own copy.  Duplicate identical
    ``<style>`` blocks are idempotent.  This mirrors ``table_cell_css`` exactly.

    Returns:
        A ``wg.HTML`` widget holding the scoped stylesheet.
    """
    return wg.HTML(GNSS_CSS_TEXT, layout=wg.Layout(margin="0"))


# ---------------------------------------------------------------------------
# Small HTML fragments.
# ---------------------------------------------------------------------------

def chip(text: str, tone: str = "muted") -> str:
    """Returns a status-pill HTML fragment.

    Args:
        text: pill text; escaped.
        tone: one of ``muted``, ``ok``, ``warn``, ``bad``, ``accent``.

    Returns:
        An HTML string, for embedding in a meter, hint, or stat grid.
    """
    return '<span class="gnss-chip" style="color:%s">%s</span>' % (
        GNSS_TONES.get(tone, GNSS_INK_MUTED),
        html_module.escape(str(text)),
    )


def hint(text: str, tone: str = "muted") -> wg.HTML:
    """Returns a muted instruction line.

    This is the single replacement for both legacy ``<em style="color:blue">``
    blocks and for the ``<ul>`` nested inside ``<em>``.

    Args:
        text: caller-supplied HTML.  **Not escaped** - pass literals only and use
            :func:`stat_grid` for anything derived from catalog data.
        tone: one of ``muted``, ``ok``, ``warn``, ``bad``, ``accent``.

    Returns:
        A ``wg.HTML`` widget.
    """
    return wg.HTML(
        '<div class="gnss-hint" style="color:%s">%s</div>'
        % (GNSS_TONES.get(tone, GNSS_INK_MUTED), text),
        layout=wg.Layout(width="100%", min_width="0", margin=GNSS_ROW_MARGIN),
    )


def meter(text: str = "") -> wg.HTML:
    """Returns a right-aligned live state readout for a section header.

    Args:
        text: initial HTML fragment.

    Returns:
        A ``wg.HTML`` widget whose value :func:`set_meter` rewrites.
    """
    widget = wg.HTML(
        '<span class="gnss-meter">%s</span>' % text,
        layout=wg.Layout(flex="1 1 0", min_width="0", margin="0"),
    )
    widget.add_class("gnss-meter-host")
    return widget


def set_meter(meter_widget: wg.HTML, text: str) -> None:
    """Replaces a meter's content, preserving its wrapper span.

    Args:
        meter_widget: widget returned by :func:`meter`.
        text: new HTML fragment.
    """
    meter_widget.value = '<span class="gnss-meter">%s</span>' % text


def stat_grid_html(pairs, columns=1, empty="No data loaded."):
    """Renders the Data View as an escaped label/value grid.

    One grid, not two widgets per pair: 20 pairs as 40 Labels would carry 40
    comms and 40 copies of the theme's widget margin.

    Args:
        pairs: iterable of ``(key, value)`` or ``(key, (value, tone))``.
        columns: number of label/value column pairs across.
        empty: text shown when ``pairs`` is empty.

    Returns:
        An HTML string; assign it to the ``.value`` of a ``wg.HTML`` widget.
    """
    pairs = list(pairs)
    if not pairs:
        return '<div class="gnss-hint">%s</div>' % html_module.escape(empty)
    cells = []
    for key, value in pairs:
        tone = "ink"
        if isinstance(value, tuple):
            value, tone = value
        cells.append(
            '<dt>%s</dt><dd style="color:%s">%s</dd>'
            % (
                html_module.escape(str(key)),
                GNSS_TONES.get(tone, GNSS_INK),
                html_module.escape(str(value)),
            )
        )
    return (
        '<dl class="gnss-kv" style="grid-template-columns:'
        'repeat(%d, minmax(6rem,auto) minmax(0,1fr))">%s</dl>'
        % (int(columns), "".join(cells))
    )


def stat_grid(pairs, columns=1, empty="No data loaded."):
    """Widget form of :func:`stat_grid_html`.

    Args:
        pairs: iterable of ``(key, value)`` or ``(key, (value, tone))``.
        columns: number of label/value column pairs across.
        empty: text shown when ``pairs`` is empty.

    Returns:
        A single ``wg.HTML`` widget.
    """
    return wg.HTML(
        stat_grid_html(pairs, columns=columns, empty=empty),
        layout=wg.Layout(width="100%", min_width="0", margin="0"),
    )


def ribbon(pairs: Sequence) -> str:
    """Builds the run-state ribbon shown above the controls and the results.

    Args:
        pairs: iterable of ``(label, value)``.

    Returns:
        An HTML string; assign it to the ``.value`` of two ``wg.HTML`` widgets so
        the control window and the output window always agree.
    """
    body = "".join(
        "<span>%s <b>%s</b></span>" % (html_module.escape(str(k)), html_module.escape(str(v)))
        for k, v in pairs
    )
    return '<div class="gnss-ribbon">%s</div>' % body


# ---------------------------------------------------------------------------
# Containers.
# ---------------------------------------------------------------------------

def panel(title, body, subtitle="", actions=None, flex="0 0 auto", max_width=None):
    """Builds a titled panel: a header bar over a bordered body.

    Args:
        title: panel name shown at the header left.
        body: iterable of widgets stacked inside the panel body.
        subtitle: muted scope/step clause shown at the header right.
        actions: optional iterable of buttons pinned to the header right.
        flex: flex shorthand for the panel itself.
        max_width: optional rem cap so a narrow window is not stretched.

    Returns:
        A ``VBox`` exposing ``.header`` and ``.body``.
    """
    head_children = [
        wg.HTML(
            '<span class="gnss-panel-title">%s</span>' % html_module.escape(title),
            layout=wg.Layout(flex="1 1 auto", min_width="0", margin="0"),
        )
    ]
    if subtitle:
        head_children.append(
            wg.HTML(
                '<span class="gnss-panel-sub">%s</span>' % html_module.escape(subtitle),
                layout=wg.Layout(flex="0 1 auto", min_width="0", margin="0"),
            )
        )
    if actions:
        head_children.append(
            HBox(
                list(actions),
                layout=wg.Layout(
                    flex="0 0 auto", justify_content="flex-end", align_items="center", margin="0"
                ),
            )
        )
    header = HBox(
        head_children,
        layout=wg.Layout(
            width="100%", min_width="0", flex="0 0 auto", justify_content="space-between",
            align_items="center", padding=GNSS_HEAD_PAD, border_bottom=GNSS_BORDER, margin="0",
        ),
    )
    header.add_class("gnss-panel-head")
    body_box = VBox(
        list(body),
        layout=wg.Layout(width="100%", min_width="0", flex="1 1 auto", padding=GNSS_PANEL_PAD, margin="0"),
    )
    shell_layout = wg.Layout(
        width="100%", min_width="0", flex=flex, border=GNSS_BORDER,
        margin=GNSS_PANEL_MARGIN, overflow="visible",
    )
    if max_width is not None:
        shell_layout.max_width = max_width
    shell = VBox([header, body_box], layout=shell_layout)
    shell.add_class("gnss-panel")
    shell.header = header
    shell.body = body_box
    return shell


def section(title, rows, meter_widget=None, flex=None, bordered=True):
    """Groups related rows under an uppercase label with an optional live meter.

    Args:
        title: section label, or ``None`` for an unlabelled group.
        rows: iterable of row widgets.
        meter_widget: optional widget from :func:`meter`, right-aligned in the head.
        flex: optional flex shorthand so a section can absorb leftover height.
        bordered: draws the inner hairline when ``True``.

    Returns:
        A ``VBox`` exposing ``.head``.
    """
    children = []
    head = None
    if title is not None or meter_widget is not None:
        head_kids = []
        if title is not None:
            head_kids.append(
                wg.HTML(
                    '<span class="gnss-section-title">%s</span>' % html_module.escape(title),
                    layout=wg.Layout(flex="0 0 auto", min_width="0", margin="0"),
                )
            )
        if meter_widget is not None:
            head_kids.append(meter_widget)
        head = HBox(
            head_kids,
            layout=wg.Layout(
                width="100%", min_width="0", justify_content="space-between",
                align_items="baseline", margin="0 0 0.3rem 0",
            ),
        )
        children.append(head)
    children.extend(rows)
    layout = wg.Layout(width="100%", min_width="0", margin=GNSS_SECTION_MARGIN)
    if bordered:
        layout.border = GNSS_BORDER_INNER
        layout.padding = GNSS_SECTION_PAD
    if flex is not None:
        layout.flex = flex
        layout.min_height = "0"
    box = VBox(children, layout=layout)
    box.add_class("gnss-section")
    box.head = head
    return box


def column(width, children, align_items="stretch"):
    """Builds one dashboard column.

    ``flex`` is written before ``width`` on purpose.  A notebook test takes the
    first three matches of ``width='(\\d+)%', flex='0 0 \\d+%'`` in the windows
    cell and requires ``[29, 42, 29]``; emitting flex first makes this helper
    structurally unable to add a match, wherever it is called from.

    Args:
        width: relative column width, e.g. ``'29%'``.
        children: widgets stacked in the column.
        align_items: cross-axis alignment.

    Returns:
        A ``VBox`` column.
    """
    return VBox(
        list(children),
        layout=wg.Layout(
            flex="0 0 " + width, width=width, min_width="0", min_height="0",
            justify_content="flex-start", align_items=align_items, margin="0",
        ),
    )


def dashboard(columns):
    """Rows the dashboard columns, wrapping instead of crushing when narrow.

    ``flex_wrap`` is not a Layout trait; ``flex_flow`` is, so ``'row wrap'`` is
    how the columns stack on a narrow or portrait screen.  No height is set: a
    definite height is what makes a fixed dashboard clip its own contents.

    Args:
        columns: iterable of widgets from :func:`column`.

    Returns:
        An ``HBox``.
    """
    return HBox(
        list(columns),
        layout=wg.Layout(
            width="100%", min_width="0", flex_flow="row wrap", align_items="stretch",
            justify_content="space-between", margin="0 0 0.7rem 0", overflow="visible",
        ),
    )


# ---------------------------------------------------------------------------
# Rows and controls.
# ---------------------------------------------------------------------------

def field_row(label, field, label_width=GNSS_LABEL_W, field_width=GNSS_FIELD_W, tooltip=None):
    """Pairs a fixed-width label with a field, mirroring the map dashboard.

    The widget's own ``description`` is cleared: the theme pins every inline
    description to a fixed 80px *inside* the widget's declared width, which is
    what currently leaves ~62px of usable space in the 150px timeseries fields.

    Replaces ``field.layout`` outright rather than mutating it, so a Layout
    object shared by reference can never be edited through this call.

    Args:
        label: label text including its trailing colon and units.
        field: the control placed to the right of the label.
        label_width: relative width of the label column.
        field_width: relative width of the field, or ``None`` to leave it alone.
        tooltip: optional hover text on the label.

    Returns:
        An ``HBox`` row.
    """
    lab = wg.Label(label, layout=wg.Layout(flex="0 0 " + label_width, width=label_width, min_width="0"))
    if tooltip:
        lab.tooltip = tooltip
    if "description" in field.trait_names():
        field.description = ""
    if field_width is not None:
        field.layout = wg.Layout(width=field_width, min_width="0")
    return HBox(
        [lab, field],
        layout=wg.Layout(
            width="100%", min_width="0", justify_content="space-between",
            align_items="center", margin=GNSS_ROW_MARGIN,
        ),
    )


def slider_row(label, slider, value_widget, label_width=GNSS_LABEL_W):
    """Builds the lazy-slider row: a coarse slider jslinked to a precision box.

    ``value_widget`` is the widget the notebook handlers already read, and the
    slider is the new companion; ``wg.jslink`` is symmetric, so no handler changes.

    Both members must share ``min``/``max``/``step`` or the link clamps on the
    slider and pushes the clamped value back into the box, silently rewriting a
    typed entry.  This raises rather than allowing that.

    Args:
        label: label text including units.
        slider: an ``IntSlider`` or ``FloatSlider``.
        value_widget: the matching ``BoundedIntText`` / ``BoundedFloatText``.
        label_width: relative width of the label column.

    Returns:
        An ``HBox`` exposing ``.slider`` and ``.value_input``.

    Raises:
        ValueError: if the two controls do not share bounds.
    """
    if (slider.min, slider.max) != (value_widget.min, value_widget.max):
        raise ValueError("slider and value box bounds must match for %r" % label)
    slider.description = ""
    slider.readout = False
    slider.layout = wg.Layout(width=GNSS_SLIDER_W, min_width="0")
    value_widget.description = ""
    value_widget.layout = wg.Layout(width=GNSS_VALUE_W, min_width="0")
    lab = wg.Label(label, layout=wg.Layout(flex="0 0 " + label_width, width=label_width, min_width="0"))
    wg.jslink((slider, "value"), (value_widget, "value"))
    row = HBox(
        [lab, slider, value_widget],
        layout=wg.Layout(
            width="100%", min_width="0", justify_content="space-between",
            align_items="center", margin=GNSS_ROW_MARGIN,
        ),
    )
    row.slider = slider
    row.value_input = value_widget
    return row


def flag(checkbox, note="", tone="muted"):
    """Turns a 300px-wide themed Checkbox into one full-width labelled line.

    Args:
        checkbox: an existing ``wg.Checkbox``; identity is preserved so handlers
            reading ``.value`` keep working.
        note: muted trailing HTML, e.g. an explanation of a cross-window effect.
        tone: tone for the note.

    Returns:
        An ``HBox`` row.
    """
    checkbox.indent = False
    checkbox.layout = wg.Layout(width="auto", min_width="0", margin="0")
    kids = [checkbox]
    if note:
        kids.append(
            wg.HTML(
                '<span class="gnss-hint" style="color:%s">%s</span>'
                % (GNSS_TONES.get(tone, GNSS_INK_MUTED), note),
                layout=wg.Layout(flex="1 1 0", min_width="0", margin="0 0 0 0.45rem"),
            )
        )
    return HBox(
        kids,
        layout=wg.Layout(width="100%", min_width="0", align_items="center", margin=GNSS_ROW_MARGIN),
    )


def action_grid(buttons, columns=2):
    """Lays buttons out as an even grid with spacing that actually renders.

    ``grid_gap`` is a real Layout trait and only applies to a ``GridBox``, which
    is exactly why this is a GridBox rather than an HBox: the map's equivalent
    uses ``gap='3%'``, which is discarded, so its buttons sit flush today.

    Args:
        buttons: iterable of already-styled buttons.
        columns: number of equal columns.

    Returns:
        A ``wg.GridBox``.
    """
    buttons = list(buttons)
    for button in buttons:
        button.layout = wg.Layout(width="100%", height=GNSS_CONTROL_H, min_width="0", margin="0")
    return wg.GridBox(
        buttons,
        layout=wg.Layout(
            width="100%", min_width="0",
            grid_template_columns=" ".join(["minmax(0, 1fr)"] * int(columns)),
            grid_gap=GNSS_GRID_GAP, margin=GNSS_ROW_MARGIN,
        ),
    )


def primary(button, width="100%"):
    """Marks a button as the single primary action of its window."""
    button.style.button_color = GNSS_ACCENT
    button.style.text_color = "#FFFFFF"
    button.style.font_weight = "600"
    button.layout = wg.Layout(width=width, height=GNSS_CONTROL_H, min_width="0", margin="0")
    return button


def neutral(button, width="100%"):
    """Marks a button as a secondary, non-destructive action."""
    button.style.button_color = GNSS_NEUTRAL_BTN
    button.style.text_color = GNSS_INK
    button.layout = wg.Layout(width=width, height=GNSS_CONTROL_H, min_width="0", margin="0")
    return button


def destructive(button, width="100%"):
    """Marks a button as a non-undoable delete.

    Danger is carried by text and border, not by a loud fill: a filled red button
    outshouts the primary action and trains the eye to ignore it.
    """
    button.style.button_color = GNSS_DANGER_WEAK
    button.style.text_color = GNSS_DANGER
    button.style.font_weight = "600"
    button.layout = wg.Layout(width=width, height=GNSS_CONTROL_H, min_width="0", margin="0")
    button.add_class("gnss-btn-danger")
    return button


def log_well(output, height="11rem"):
    """Bounds an Output so a long log stops shoving the page down.

    The Output object is never replaced, so every ``with <output>:`` block in the
    handler cells keeps working: capture is by name, not by position.

    Args:
        output: an existing ``wg.Output``.
        height: rem cap before the well scrolls; ``None`` grows freely, which is
            what the figure pane needs.

    Returns:
        The same Output, restyled.
    """
    layout = wg.Layout(
        width="100%", min_width="0", flex="0 0 auto", min_height="2.2rem",
        overflow="auto", border=GNSS_BORDER_INNER, padding="0.35rem 0.5rem", margin="0",
    )
    if height is not None:
        layout.max_height = height
    output.layout = layout
    output.add_class("gnss-log")
    return output


def log_panel(output, title, height="11rem", meter_widget=None, actions=None):
    """Wraps a bounded Output in a titled section.

    Args:
        output: the existing ``wg.Output``.
        title: section label.
        height: rem cap before the well scrolls, or ``None`` to grow freely.
        meter_widget: optional meter for the section head.
        actions: optional widgets appended under the well.

    Returns:
        A bordered :func:`section` containing the framed Output.
    """
    rows = [log_well(output, height)] + (list(actions) if actions else [])
    return section(title, rows, meter_widget=meter_widget)


# ---------------------------------------------------------------------------
# Wiring helpers.
# ---------------------------------------------------------------------------

_GNSS_BOUND: dict = {}
_GNSS_CLICKS: dict = {}


def bind(widget, handler, names="value", key=None):
    """Registers an observer at most once across cell re-runs.

    traitlets already de-duplicates the *same function object*, but re-running a
    cell rebinds the name to a **new** object, so the old registration survives
    and the handler fires twice.  The windows cell already demonstrates the bug
    on the basemap dropdown.  ``unobserve_all`` is deliberately not used: it
    would also strip observers other cells registered on shared widgets.

    Args:
        widget: the observed widget.
        handler: callable taking a traitlets change dict.
        names: trait name or tuple of names.
        key: explicit identity; defaults to (widget id, names, handler name).

    Returns:
        The handler.
    """
    reg_key = key if key is not None else (
        id(widget), str(names), getattr(handler, "__name__", repr(handler))
    )
    previous = _GNSS_BOUND.get(reg_key)
    if previous is not None:
        try:
            widget.unobserve(previous, names=names)
        except ValueError:
            pass
    widget.observe(handler, names=names)
    _GNSS_BOUND[reg_key] = handler
    return handler


def bind_click(button, handler, key=None):
    """Registers a click handler at most once across cell re-runs.

    ``CallbackDispatcher`` de-duplicates the *same function object*, but
    re-running a cell rebinds the name to a new object and the stale handler
    survives, so a refresh would run once per re-run.

    Args:
        button: the ``wg.Button``.
        handler: callable taking the button.
        key: explicit identity; defaults to (button id, handler name).

    Returns:
        The handler.
    """
    reg_key = key if key is not None else (
        id(button), getattr(handler, "__name__", repr(handler))
    )
    previous = _GNSS_CLICKS.get(reg_key)
    if previous is not None:
        button.on_click(previous, remove=True)
    button.on_click(handler)
    _GNSS_CLICKS[reg_key] = handler
    return handler


def commit_on_enter(text_widget, handler):
    """Fires ``handler`` when the user commits a value with Enter or blur.

    ``Text.on_submit`` still exists but warns: "on_submit is deprecated. Instead,
    set the .continuous_update attribute to False and observe the value changing".
    This is that documented replacement.  It does not re-fire on an unchanged
    value, which is fine for an idempotent read-only lookup and is why it must
    not be used for anything that mutates catalog data.

    Args:
        text_widget: a ``wg.Text`` acting as a search field.
        handler: callable taking one positional argument, matching ``on_click``.
    """
    text_widget.continuous_update = False
    bind(text_widget, lambda _change: handler(None), names="value",
         key=(id(text_widget), "commit_on_enter"))


def walk(widget, seen=None):
    """Yields a widget and every descendant exactly once."""
    seen = set() if seen is None else seen
    if id(widget) in seen:
        return
    seen.add(id(widget))
    yield widget
    for child in getattr(widget, "children", ()) or ():
        yield from walk(child, seen)


def assert_reachable(root, required, where):
    """Fails loudly if a regroup dropped a control out of a window.

    Run at the bottom of the windows cell.  Because it raises, the end-to-end
    notebook test turns "retain all features" from a promise into a build gate.

    Args:
        root: a window container.
        required: mapping of name to widget that must live inside ``root``.
        where: window name, used in the error message.

    Returns:
        The number of verified controls.

    Raises:
        AssertionError: naming every missing control.
    """
    present = {id(widget) for widget in walk(root)}
    missing = sorted(name for name, widget in required.items() if id(widget) not in present)
    if missing:
        raise AssertionError("%s: unreachable controls -> %s" % (where, ", ".join(missing)))
    return len(required)


# ---------------------------------------------------------------------------
# Pure readout state.  No widgets, no I/O: every one of these is unit-testable.
# ---------------------------------------------------------------------------

def list_counts(options, selected, orglist):
    """Counts the timeseries working set overall and per source.

    Args:
        options: the ``'SITE (ORG)'`` labels currently in the list.
        selected: the currently selected labels.
        orglist: known source keys.

    Returns:
        ``(total, selected, per_source, per_source_selected, unresolved)``.
    """
    selected = set(selected)
    per = {org: 0 for org in orglist}
    per_selected = {org: 0 for org in orglist}
    unresolved = 0
    for label in options:
        try:
            _site, org = remove_brac(label)
        except Exception:
            unresolved += 1
            continue
        if org not in per:
            unresolved += 1
            continue
        per[org] += 1
        if label in selected:
            per_selected[org] += 1
    return len(list(options)), len(selected), per, per_selected, unresolved


def break_counts(options, selected, data_of):
    """Counts catalogued break epochs for the working set and the selection.

    Args:
        options: ``'SITE (ORG)'`` labels.
        selected: selected labels.
        data_of: the loaded catalogs, ``data_of[org][site]``.

    Returns:
        ``(with_breaks, resolvable, selected_with_breaks, selected_epochs)``.
    """
    selected = set(selected)
    with_breaks = resolvable = selected_with = selected_epochs = 0
    for label in options:
        try:
            site, org = remove_brac(label)
        except Exception:
            continue
        record = (data_of.get(org) or {}).get(site)
        if not isinstance(record, Mapping):
            continue
        resolvable += 1
        breaks = record.get("breaks")
        if breaks:
            with_breaks += 1
            if label in selected:
                selected_with += 1
                selected_epochs += len(breaks)
    return with_breaks, resolvable, selected_with, selected_epochs


def sigma_state(text, in_force):
    """Distinguishes a valid sigma triple from unparseable text.

    ``number_list`` swallows every parse failure and returns its default, so a
    typo and a deliberate ``(10,10,30)`` are otherwise indistinguishable.  Two
    probes with different sentinels resolve that using nothing but that function.

    Args:
        text: raw sigma-limit entry.
        in_force: the module-global triple currently applied.

    Returns:
        ``(effective, is_valid, status)`` where status is ``inherited``,
        ``pending``, or ``invalid``.
    """
    if not text or not str(text).strip():
        return tuple(in_force), True, "inherited"
    first = tuple(number_list(text, [-1.0, -1.0, -1.0]))
    second = tuple(number_list(text, [-2.0, -2.0, -2.0]))
    if first != second:
        return tuple(in_force), False, "invalid"
    return first, True, "pending"


def window_summary(start_text, end_text):
    """Validates and summarises the epoch window before Plot is pressed.

    Deliberately does not strip: the plot handler calls ``strptime`` on the raw
    value, which rejects a leading or trailing space, so a stripping validator
    would show valid for an entry that then raises.

    Args:
        start_text: raw start entry.
        end_text: raw end entry.

    Returns:
        ``(summary, is_valid)``.
    """
    bad = []
    low = high = None
    if start_text:
        try:
            low = datetime.strptime(start_text, "%Y-%m-%d")
        except ValueError:
            bad.append("start")
    if end_text:
        try:
            high = datetime.strptime(end_text, "%Y-%m-%d")
        except ValueError:
            bad.append("end")
    if bad:
        return "invalid %s - use YYYY-MM-DD" % " and ".join(bad), False
    if low is None and high is None:
        return "full record", True
    if low is not None and high is not None:
        if high < low:
            return "end precedes start", False
        return "%s to %s (%.2f yr)" % (start_text, end_text, (high - low).days / 365.25), True
    if low is not None:
        return "%s to end of record" % start_text, True
    return "start of record to %s" % end_text, True


def catalogue_rows(data_of, orglist):
    """Per-source inventory for the fetcher window.

    Args:
        data_of: the loaded catalogs.
        orglist: known source keys.

    Returns:
        A list of ``(org, station_count, stations_with_breaks)``.
    """
    rows = []
    for org in orglist:
        record = data_of.get(org) or {}
        with_breaks = sum(
            1 for site in record
            if isinstance(record[site], Mapping) and record[site].get("breaks")
        )
        rows.append((org, len(record), with_breaks))
    return rows


def match_codes(keys, text, cap=8, contains=False):
    """Filters a catalog or a working set, for the quantifier readouts.

    Args:
        keys: iterable of station codes or ``'SITE (ORG)'`` labels.
        text: raw search entry; normalised exactly as the stored lookup does,
            with ``.strip().upper()``.
        cap: how many example matches to return.
        contains: substring match instead of prefix match, so a working-set
            filter can also match on the source, e.g. ``UNR``.

    Returns:
        ``(match_count, sample, is_exact)``.
    """
    keys = [str(key) for key in keys]
    query = (text or "").strip().upper()
    if not query:
        return len(keys), [], False
    if contains:
        hits = [key for key in keys if query in key.upper()]
    else:
        hits = [key for key in keys if key.startswith(query)]
    return len(hits), sorted(hits)[:cap], query in set(keys)


GNSS_FIELD_LABELS = (
    ("location", "Location (lat, lon)", "deg"),
    ("height", "Height", "m"),
    ("elev", "Elevation", "m"),
    ("region", "Region", ""),
    ("regions", "Regions", ""),
    ("stntype", "Station type", ""),
    ("velocity", "Velocity N,E,U", "mm/yr"),
    ("velsig", "Sigma N,E,U", "mm/yr"),
)


def station_detail_pairs(record):
    """Formats one station record as Data View pairs with real units.

    Replaces ``str(record)`` hard-wrapped at a fixed 150 characters, which fights
    every responsive layout and buries every field in one run-on blob.

    Args:
        record: ``data_of[org][site]``.

    Returns:
        A list of ``(label, value)`` pairs; unknown keys are appended verbatim so
        no stored field is ever hidden.
    """
    pairs = []
    seen = set()
    for key, label, unit in GNSS_FIELD_LABELS:
        if key not in record:
            continue
        seen.add(key)
        value = record[key]
        if key == "location" and isinstance(value, (list, tuple)) and len(value) >= 2:
            text = "%.7f, %.7f" % (float(value[0]), float(value[1]))
        elif key in ("height", "elev"):
            try:
                text = "%.4f" % float(value)
            except (TypeError, ValueError):
                text = format_display_number(value)
        elif key in ("velocity", "velsig"):
            text = format_neu_vector(value)
        elif isinstance(value, (list, tuple)):
            text = ", ".join(str(item) for item in value)
        else:
            text = str(value)
        pairs.append(("%s (%s)" % (label, unit) if unit else label, text))
    breaks = record.get("breaks") or {}
    seen.add("breaks")
    pairs.append(("Breaks (epochs)", str(len(breaks))))
    for key in record:
        if key not in seen:
            pairs.append((str(key), str(record[key])))
    return pairs


def plot_preview_svg(error_bars, every_nth, outline, alpha, breaks_marked, detrend, shift_mm):
    """Inline SVG showing what the current styling will draw.

    The timeseries counterpart of the map's approved vector legend: three stacked
    N/E/U panels redrawn on every control change, so a researcher sees the effect
    of a checkbox without paying for a multi-second plot.  A ``viewBox`` with
    ``preserveAspectRatio`` makes every font size a user unit that scales with
    the box.

    Args:
        error_bars: whether per-epoch bars are drawn.
        every_nth: error-bar decimation stride.
        outline: whether the sigma envelope is filled.
        alpha: envelope opacity.
        breaks_marked: whether break epochs are still marked.
        detrend: whether the secular trend is removed.
        shift_mm: staged display offset in mm.

    Returns:
        An HTML string.
    """
    stride = max(1, int(every_nth or 1))
    fill = float(alpha or 0.0) if outline else 0.0
    body = ""
    for index, comp in enumerate(("N", "E", "U")):
        base = 26 + index * 46
        slope = 0 if detrend else 7
        body += '<text x="4" y="%d" font-size="9" fill="%s">d%s</text>' % (
            base + 3, GNSS_INK_MUTED, comp)
        if fill > 0:
            body += (
                '<path d="M24,%d L120,%d L120,%d L24,%d Z" fill="%s" fill-opacity="%g"/>'
                % (base - 6, base - 6 - slope, base + 6 - slope, base + 6, GNSS_ACCENT, fill)
            )
        body += (
            '<polyline points="24,%d 56,%d 88,%d 120,%d" fill="none" stroke="%s" stroke-width="1.6"/>'
            % (base, base - slope / 3.0, base - 2 * slope / 3.0, base - slope, GNSS_INK)
        )
        if error_bars:
            for step, x_pos in enumerate(range(24, 121, 12)):
                if step % stride:
                    continue
                body += (
                    '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="1"/>'
                    % (x_pos, base - 9, x_pos, base + 9, GNSS_INK_MUTED)
                )
        if breaks_marked:
            body += (
                '<line x1="72" y1="%d" x2="72" y2="%d" stroke="%s" stroke-width="1" stroke-dasharray="3 2"/>'
                % (base - 14, base + 14, GNSS_DANGER)
            )
    caption = "detrended" if detrend else "raw trend"
    if shift_mm:
        caption += " &middot; shift %+g mm (Up %+g)" % (shift_mm, shift_mm * 3)
    return (
        '<div style="border:%s;background:%s;padding:0.45rem;text-align:center">'
        '<svg viewBox="0 0 132 152" width="100%%" preserveAspectRatio="xMidYMid meet" '
        'aria-label="Timeseries styling preview">%s</svg>'
        '<div class="gnss-hint" style="margin-top:0.2rem">%s</div></div>'
        % (GNSS_BORDER_INNER, GNSS_SURFACE_SUNK, body, caption)
    )

"""Small ipyleaflet adapter for atomically replacing dynamic map layers."""

from collections.abc import Callable

from ipyleaflet import LayerGroup


def stage_layer_update(visible_layers: LayerGroup, render_layers: Callable[[LayerGroup], None]) -> None:
    """Builds map layers off-screen, then replaces the visible group in one update."""
    staged_layers = LayerGroup(name="Staged Station and Velocity Layers")
    render_layers(staged_layers)
    visible_layers.layers = staged_layers.layers

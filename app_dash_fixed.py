
"""
app_dash.py — Dash + dash-leaflet front-end for the sketch-level transit ridership tool.

This replaces the Streamlit + folium UI with an event-driven Dash UI so pan/zoom
doesn't cause full-script reruns and map interactions feel smooth.

Run:
  python app_dash.py
Then open:
  http://127.0.0.1:8050
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update, ALL
import dash_leaflet as dl
import dash_leaflet.express as dlx

# Backend
from core import warm_start, compute_line_metrics, ScenarioParams


# ----------------------------
# Defaults / config
# ----------------------------

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path(
    os.environ.get("TRANSITTOOL_DATA_DIR", REPO_ROOT / "data" / "transit_tool")
).expanduser()

DEFAULT_CENTER = (37.7749, -122.4194)  # (lat, lon) SF-ish
DEFAULT_ZOOM = 12

DEFAULT_PATHS = {
    "graph_path": str(DEFAULT_DATA_DIR / "graph_csr.npz"),
    "bg_path": str(DEFAULT_DATA_DIR / "bg_mapping.npz"),
    "od_path": str(DEFAULT_DATA_DIR / "replica_bg_od.parquet"),
}


# ----------------------------
# Cached asset load (process lifetime)
# ----------------------------

@lru_cache(maxsize=4)
def get_assets_cached(graph_path: str, bg_path: str, od_path: str):
    return warm_start(graph_path, bg_path, od_path)


# ----------------------------
# Small helpers
# ----------------------------

def latlon_to_lonlat_array(stations_latlon: List[Tuple[float, float]]) -> np.ndarray:
    if not stations_latlon:
        return np.zeros((0, 2), dtype=float)
    return np.array([[lon, lat] for (lat, lon) in stations_latlon], dtype=float)


def _extract_points_from_geojson(fc: Optional[Dict[str, Any]]) -> List[Tuple[float, float]]:
    """Return [(lat, lon), ...] from a FeatureCollection, preserving feature order."""
    if not fc or fc.get("type") != "FeatureCollection":
        return []
    out: List[Tuple[float, float]] = []
    for feat in fc.get("features", []) or []:
        try:
            geom = feat.get("geometry", {})
            if geom.get("type") != "Point":
                continue
            coords = geom.get("coordinates", None)  # [lon, lat]
            if not coords or len(coords) < 2:
                continue
            lon, lat = float(coords[0]), float(coords[1])
            out.append((lat, lon))
        except Exception:
            continue
    return out


def _make_polyline(stations_latlon: List[Tuple[float, float]]):
    if len(stations_latlon) < 2:
        return None
    return dl.Polyline(positions=stations_latlon, color="black", weight=4, opacity=0.5)


def _station_markers(stations_latlon: List[Tuple[float, float]]):
    markers = []
    for i, (lat, lon) in enumerate(stations_latlon):
        markers.append(
            dl.Marker(
                position=(lat, lon),
                children=dl.Tooltip(f"Station {i+1}"),
            )
        )
    return markers


def _scenario_params_from_inputs(v: Dict[str, Any]) -> ScenarioParams:
    # Multiselects come back as list[int] or list[str]
    walk = tuple(float(x) for x in sorted(v.get("walk_thresholds") or [5, 10, 15]))
    bike = tuple(float(x) for x in sorted(v.get("bike_thresholds") or [5, 10, 15]))

    restrict_mode = v.get("restrict_mode", "car")
    restrict_to_primary_mode = None if restrict_mode == "all" else restrict_mode

    params = ScenarioParams(
        walk_thresholds_min=walk,
        bike_thresholds_min=bike,
        restrict_to_primary_mode=restrict_to_primary_mode,
        w_distance=v.get("w_type", "exp_saturating"),  # type: ignore
        lambda_km=float(v.get("lambda_km", 6.0)),
        cruise_kmh=float(v.get("cruise_kmh", 75.0)),
        dwell_sec=float(v.get("dwell_sec", 10.0)),
        assign_method=v.get("assign_method", "euclidean"),  # type: ignore
        assign_walk_cutoff_min=float(v.get("assign_walk_cutoff", 30.0)),
        return_catchment_nodes=bool(v.get("return_nodes", False)),
        return_catchment_polygons=bool(v.get("return_polys", True)),
        polygon_min_points=int(v.get("poly_min_points", 40)),
        polygon_buffer_deg=float(v.get("poly_buffer", 0.0005)),
    )
    return params


def _format_int(x: Any) -> str:
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return str(x)


def _metrics_panel(metrics: Dict[str, Any]) -> html.Div:
    rid = metrics.get("ridership", {}) or {}
    acc = metrics.get("accessibility", {}) or {}
    diag = metrics.get("diagnostics", {}) or {}

    # Ridership
    total = rid.get("total_riders", 0.0)
    boardings = rid.get("boardings_by_station", []) or []

    ridership_block = html.Div(
        [
            html.H3("Ridership", style={"marginTop": 0}),
            html.Div("Estimated daily linked riders:", style={"fontWeight": 600}),
            html.Div(_format_int(total), style={"fontSize": "2rem", "fontWeight": 800, "lineHeight": "1.1"}),
            html.Div("Boardings by station", style={"marginTop": "10px", "fontSize": "0.85rem", "opacity": 0.8})
            if boardings
            else html.Div(),
            html.Pre(
                json.dumps({f"{i+1}": round(float(x), 1) for i, x in enumerate(boardings)}, indent=2),
                style={"margin": 0, "display": "block" if boardings else "none"},
            ),
        ],
        style=_card_style(),
    )

    # Accessibility
    walk = acc.get("walk", {}) or {}
    bike = acc.get("bike", {}) or {}

    def _kv_lines(d: Dict[str, Any], prefix: str):
        rows = []
        for k, v in d.items():
            rows.append(html.Div([html.Span(f"{prefix} ≤ {k} min: "), html.B(_format_int(v))]))
        return rows

    accessibility_block = html.Div(
        [
            html.H3("Accessibility", style={"marginTop": 0}),
            html.Div("Population within walk thresholds (unioned)", style={"fontSize": "0.85rem", "opacity": 0.8})
            if walk else html.Div(),
            *(_kv_lines(walk, "Walk") if walk else []),
            html.Hr(style={"margin": "10px 0"}) if walk and bike else html.Div(),
            html.Div("Population within bike thresholds (unioned)", style={"fontSize": "0.85rem", "opacity": 0.8})
            if bike else html.Div(),
            *(_kv_lines(bike, "Bike") if bike else []),
        ],
        style=_card_style(),
    )

    # Diagnostics
    defs = {
        "alpha_mean": ("Mean coverage α", "Average share of each origin BG captured by at least one station catchment (0–1)."),
        "alpha_p50": ("Median coverage α", "50th percentile of BG coverage α across origins (0–1)."),
        "alpha_p90": ("90th pct coverage α", "90th percentile of BG coverage α across origins (0–1)."),
        "w_d_mean": ("Mean distance weight w(d)", "Average OD distance downweight applied to included trips (0–1)."),
        "od_rows_used": ("OD rows used", "Number of OD records used in ridership aggregation (count)."),
        "in_vehicle_time_min_mean": ("Mean in-vehicle time", "Mean in-vehicle time between assigned O→D stations (minutes)."),
        "in_vehicle_time_min_p90": ("90th pct in-vehicle time", "90th percentile in-vehicle time (minutes)."),
    }

    diag_rows = []
    for k, (name, desc) in defs.items():
        if k in diag:
            diag_rows.append(
                html.Div(
                    [
                        html.Div([html.B(name), html.Span(f": {diag.get(k)}")]),
                        html.Div(desc, style={"fontSize": "0.8rem", "opacity": 0.75}),
                        html.Hr(style={"margin": "8px 0"}),
                    ]
                )
            )

    diagnostics_block = html.Div(
        [html.H3("Diagnostics", style={"marginTop": 0}), *diag_rows] if diag_rows else [html.H3("Diagnostics", style={"marginTop": 0}), html.Div("—")],
        style=_card_style(),
    )

    return html.Div(
        [ridership_block, accessibility_block, diagnostics_block],
        style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "12px"},
    )


def _card_style() -> Dict[str, str]:
    return {
        "border": "1px solid rgba(0,0,0,0.12)",
        "borderRadius": "10px",
        "padding": "12px",
        "background": "white",
        "boxShadow": "0 1px 6px rgba(0,0,0,0.06)",
    }


# ----------------------------
# Dash app
# ----------------------------

app = Dash(__name__)
server = app.server

SIDEBAR_W = "360px"

app.layout = html.Div(
    [
        # Stores
        dcc.Store(id="stations_store", data=[]),  # [(lat, lon), ...]
        dcc.Store(id="metrics_store", data=None),  # dict
        dcc.Store(id="view_store", data={"center": list(DEFAULT_CENTER), "zoom": DEFAULT_ZOOM}),
        dcc.Store(id="selected_station_idx", data=None),
        dcc.Download(id="download_json"),

        html.Div(
            className="app-shell",
            children=[
                # Sidebar (inputs)
                html.Div(
                    className="sidebar",
                    children=[
                        html.Div("Transit Line Sketch Tool", className="section-title"),
                        html.Div(
                            "Click to add stations in order. Use the draw toolbar to edit/move/delete markers. Click compute to update metrics.",
                            className="caption",
                        ),

                        html.Hr(),

                        html.Div("Data paths", className="subheader"),
                        html.Div("Graph CSR .npz", className="label"),
                        dcc.Input(id="graph_path", type="text", value=DEFAULT_PATHS["graph_path"], className="input"),
                        html.Div("Block-group mapping .npz", className="label", style={"marginTop": "10px"}),
                        dcc.Input(id="bg_path", type="text", value=DEFAULT_PATHS["bg_path"], className="input"),
                        html.Div("Replica BG→BG OD (parquet/csv)", className="label", style={"marginTop": "10px"}),
                        dcc.Input(id="od_path", type="text", value=DEFAULT_PATHS["od_path"], className="input"),

                        html.Hr(),

                        html.Div("Model params", className="subheader"),
                        html.Div("Walk thresholds (min)", className="label"),
                        dcc.Dropdown(
                            id="walk_thresholds",
                            options=[{"label": str(x), "value": x} for x in [3, 5, 8, 10, 12, 15, 20]],
                            value=[5, 10, 15],
                            multi=True,
                            clearable=False,
                        ),
                        html.Div("Bike thresholds (min)", className="label", style={"marginTop": "10px"}),
                        dcc.Dropdown(
                            id="bike_thresholds",
                            options=[{"label": str(x), "value": x} for x in [3, 5, 8, 10, 12, 15, 20]],
                            value=[5, 10, 15],
                            multi=True,
                            clearable=False,
                        ),

                        html.Div("Restrict OD to primary mode", className="label", style={"marginTop": "10px"}),
                        dcc.Dropdown(
                            id="restrict_mode",
                            options=[{"label": "car", "value": "car"}, {"label": "all", "value": "all"}],
                            value="car",
                            clearable=False,
                        ),

                        html.Div("Distance weight w(d)", className="label", style={"marginTop": "10px"}),
                        dcc.Dropdown(
                            id="w_type",
                            options=[{"label": "exp_saturating", "value": "exp_saturating"}, {"label": "step", "value": "step"}],
                            value="exp_saturating",
                            clearable=False,
                        ),
                        html.Div("lambda (km) for exp_saturating", className="label", style={"marginTop": "10px"}),
                        dcc.Slider(id="lambda_km", min=1.0, max=30.0, step=0.5, value=6.0, tooltip={"placement": "bottom"}),

                        html.Div("Line cruise speed (km/h)", className="label", style={"marginTop": "10px"}),
                        dcc.Slider(id="cruise_kmh", min=30.0, max=110.0, step=1.0, value=75.0, tooltip={"placement": "bottom"}),

                        html.Div("Dwell per stop (sec)", className="label", style={"marginTop": "10px"}),
                        dcc.Slider(id="dwell_sec", min=0.0, max=60.0, step=1.0, value=10.0, tooltip={"placement": "bottom"}),

                        html.Div("BG→station assignment", className="label", style={"marginTop": "10px"}),
                        dcc.Dropdown(
                            id="assign_method",
                            options=[{"label": "euclidean", "value": "euclidean"}, {"label": "walk_to_centroid_node", "value": "walk_to_centroid_node"}],
                            value="euclidean",
                            clearable=False,
                        ),
                        html.Div("Assign walk cutoff (min) (if walk_to_centroid_node)", className="label", style={"marginTop": "10px"}),
                        dcc.Slider(id="assign_walk_cutoff", min=5.0, max=120.0, step=5.0, value=30.0, tooltip={"placement": "bottom"}),

                        html.Hr(),

                        html.Div("Output", className="subheader"),
                        dcc.Checklist(
                            id="return_nodes",
                            options=[{"label": "Return catchment nodes (HUGE payload)", "value": "yes"}],
                            value=[],
                        ),
                        dcc.Checklist(
                            id="return_polys",
                            options=[{"label": "Return catchment polygons", "value": "yes"}],
                            value=["yes"],
                            style={"marginTop": "6px"},
                        ),
                        html.Div("Polygon min points", className="label", style={"marginTop": "10px"}),
                        dcc.Slider(id="poly_min_points", min=10, max=200, step=5, value=40, tooltip={"placement": "bottom"}),

                        html.Div("Polygon buffer (deg, lon/lat)", className="label", style={"marginTop": "10px"}),
                        dcc.Input(id="poly_buffer", type="number", value=0.0005, step=0.0001, className="input"),
                    ],
                ),

                # Main
                html.Div(
                    className="main",
                    children=[
                        html.Div(
                            className="row",
                            children=[
                                # Left column: map + stations
                                html.Div(
                                    className="col-left",
                                    children=[
                                        html.Div("Map", className="subheader"),
                                        html.Div(
                                            [
                                                dcc.Checklist(
                                                    id="click_add",
                                                    options=[{"label": "Click map to add station", "value": "yes"}],
                                                    value=["yes"],
                                                )
                                            ],
                                            style={"marginBottom": "8px"},
                                        ),
                                        html.Div(
                                            id="map_container",
                                            className="card",
                                            style={"padding": "0px", "overflow": "hidden"},
                                        ),
                                        html.Div(style={"height": "10px"}),
                                        html.Div("Stations (ordered)", className="subheader"),
                                        html.Div(
                                            id="stations_list",
                                            className="card",
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Button("Move up", id="move_up", className="btn"),
                                                        html.Button("Move down", id="move_down", className="btn", style={"marginLeft": "8px"}),
                                                        html.Button("Remove selected", id="remove_selected", className="btn", style={"marginLeft": "8px"}),
                                                    ],
                                                    style={"marginTop": "10px"},
                                                ),
                                                html.Hr(style={"margin": "12px 0"}),
                                                html.Div(html.B("Insert infill station")),
                                                html.Div(
                                                    [
                                                        html.Div("Insert position", className="label"),
                                                        dcc.Input(id="ins_idx", type="number", min=1, step=1, value=1, className="input"),
                                                    ],
                                                    style={"width": "32%", "display": "inline-block", "verticalAlign": "top", "marginRight": "2%"},
                                                ),
                                                html.Div(
                                                    [
                                                        html.Div("Lat", className="label"),
                                                        dcc.Input(id="ins_lat", type="number", value=DEFAULT_CENTER[0], className="input"),
                                                    ],
                                                    style={"width": "32%", "display": "inline-block", "verticalAlign": "top", "marginRight": "2%"},
                                                ),
                                                html.Div(
                                                    [
                                                        html.Div("Lon", className="label"),
                                                        dcc.Input(id="ins_lon", type="number", value=DEFAULT_CENTER[1], className="input"),
                                                    ],
                                                    style={"width": "32%", "display": "inline-block", "verticalAlign": "top"},
                                                ),
                                                html.Div(style={"height": "8px"}),
                                                html.Button("Insert station", id="insert_station", className="btn"),
                                                html.Hr(style={"margin": "12px 0"}),
                                                html.Div(
                                                    [
                                                        html.Button("Clear stations", id="clear_stations", className="btn"),
                                                        html.Button("Remove last station", id="remove_last", className="btn", style={"marginLeft": "8px"}),
                                                        html.Button("Reverse order", id="reverse_order", className="btn", style={"marginLeft": "8px"}),
                                                    ]
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            "Tip: Use the draw toolbar’s Edit mode to drag markers, then click Compute.",
                                            className="caption",
                                            style={"marginTop": "10px"},
                                        ),
                                    ],
                                ),

                                # Right column: compute + metrics
                                html.Div(
                                    className="col-right",
                                    children=[
                                        html.Div("Compute", className="subheader"),
                                        html.Div(
                                            "When you click compute, the backend snaps stations to the graph, builds walk/bike catchments, computes coverage α, and estimates ridership from Replica OD.",
                                            className="caption",
                                        ),
                                        html.Button("Compute new line metrics", id="compute_btn", className="btn btn-primary"),
                                        html.Div(id="compute_status", style={"marginTop": "10px"}),
                                        html.Div(style={"height": "10px"}),
                                        html.Div(id="metrics_panel"),
                                        html.Hr(style={"margin": "12px 0"}),
                                        html.Div("Download / Inspect", className="subheader"),
                                        html.Button("Download results JSON", id="download_btn", className="btn"),
                                        html.Details([html.Summary("Show raw JSON"), html.Pre(id="raw_json", style={"whiteSpace": "pre-wrap"})]),
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
            ],
        ),
    ]
)


# ----------------------------
# Map builder callback
# ----------------------------

@app.callback(
    Output("map_container", "children"),
    Input("stations_store", "data"),
    Input("metrics_store", "data"),
    Input("view_store", "data"),
)
def render_map(stations, metrics, view):
    stations = stations or []
    metrics = metrics or {}
    center = (view or {}).get("center", list(DEFAULT_CENTER))
    zoom = (view or {}).get("zoom", DEFAULT_ZOOM)

    # Catchment polygons (show max threshold for walk+bike, same as Streamlit)
    polys_children = []
    try:
        layers = (metrics.get("layers", {}) or {}).get("catchments", {}) or {}
        if layers:
            walk = layers.get("walk", {}) or {}
            bike = layers.get("bike", {}) or {}

            def _fc_to_geojson(fc):
                return fc if isinstance(fc, dict) else {}

            # Use max thresholds in params? We don't have params here, so use max key present.
            if walk:
                tmax = sorted(walk.keys(), key=lambda x: float(x))[-1]
                polys_children.append(
                    dl.GeoJSON(
                        data=_fc_to_geojson(walk.get(tmax, {})),
                        id="walk_polys",
                        options=dict(style=dict(weight=2, fillOpacity=0.12)),
                        hoverStyle=dict(weight=3, fillOpacity=0.18),
                    )
                )
            if bike:
                tmax = sorted(bike.keys(), key=lambda x: float(x))[-1]
                polys_children.append(
                    dl.GeoJSON(
                        data=_fc_to_geojson(bike.get(tmax, {})),
                        id="bike_polys",
                        options=dict(style=dict(weight=2, fillOpacity=0.12)),
                        hoverStyle=dict(weight=3, fillOpacity=0.18),
                    )
                )
    except Exception:
        polys_children = []

    # Editable layer group
    edit = dl.EditControl(
        id="edit_control",
        draw=dict(polyline=False, polygon=False, circle=False, rectangle=False, circlemarker=False, marker=True),
        edit=dict(edit=True, remove=True),
    )

    feature_group = dl.FeatureGroup([edit] + _station_markers(stations), id="stations_fg")

    polyline = _make_polyline(stations)
    map_children = [dl.TileLayer(url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"), feature_group]
    if polyline is not None:
        map_children.append(polyline)
    map_children += polys_children

    m = dl.Map(
        id="map",
        center=center,
        zoom=zoom,
        children=map_children,
        style={"width": "100%", "height": "650px"},
    )
    return m


# ----------------------------
# Persist map view (center/zoom) without triggering full rebuild loops
# ----------------------------

@app.callback(
    Output("view_store", "data"),
    Input("map", "viewport"),
    State("view_store", "data"),
    prevent_initial_call=True,
)
def persist_view(viewport, cur):
    if not viewport:
        return no_update
    c = viewport.get("center")
    z = viewport.get("zoom")
    if not c or z is None:
        return no_update
    out = dict(cur or {})
    out["center"] = [float(c[0]), float(c[1])]
    out["zoom"] = int(z)
    return out


# ----------------------------
# Station edits from draw toolbar (authoritative)
# ----------------------------

@app.callback(
    Output("stations_store", "data", allow_duplicate=True),
    Output("metrics_store", "data", allow_duplicate=True),
    Input("edit_control", "geojson"),
    State("stations_store", "data"),
    prevent_initial_call=True,
)
def stations_from_draw(geojson, stations_cur):
    pts = _extract_points_from_geojson(geojson)
    if not pts:
        # If user deleted everything via draw/remove, allow clearing.
        if geojson and (geojson.get("features") == []):
            return [], None
        return no_update, no_update
    # Use marker order from the feature list.
    return pts, None


# ----------------------------
# Station add from map click
# ----------------------------

@app.callback(
    Output("stations_store", "data", allow_duplicate=True),
    Output("metrics_store", "data", allow_duplicate=True),
    Input("map", "click_lat_lng"),
    State("stations_store", "data"),
    State("click_add", "value"),
    prevent_initial_call=True,
)
def add_station_from_click(click, stations, click_add_value):
    if not click:
        return no_update, no_update
    if not (click_add_value and "yes" in click_add_value):
        return no_update, no_update

    stations = list(stations or [])
    pt = (float(click[0]), float(click[1]))
    if stations and pt == tuple(stations[-1]):
        return no_update, no_update  # debounce exact repeat
    stations.append(pt)
    return stations, None


# ----------------------------
# Stations list render + selection
# ----------------------------

@app.callback(
    Output("stations_list", "children"),
    Output("ins_idx", "value"),
    Input("stations_store", "data"),
    Input("selected_station_idx", "data"),
)
def render_station_list(stations, selected_idx):
    stations = stations or []
    n = len(stations)
    if n == 0:
        return html.Div("No stations yet. Click on the map to add stations (or use the marker draw tool)."), 1

    # Clamp selected idx
    if selected_idx is None or not (0 <= int(selected_idx) < n):
        selected_idx = 0

    items = []
    for i, (lat, lon) in enumerate(stations):
        is_sel = (i == int(selected_idx))
        items.append(
            html.Div(
                [
                    html.Span(f"{i+1}. ", style={"fontWeight": 800}),
                    html.Span(f"({lat:.6f}, {lon:.6f})"),
                    html.Span(" selected", className="pill", style={"marginLeft": "8px", "display": "inline-block" if is_sel else "none"}),
                ],
                n_clicks=0,
                id={"type": "station_row", "index": i},
                style={
                    "padding": "6px 8px",
                    "borderRadius": "8px",
                    "cursor": "pointer",
                    "background": "rgba(31,119,255,0.10)" if is_sel else "transparent",
                },
            )
        )
    return html.Div(items), min(n + 1, int(selected_idx) + 1)


# Initialize / clamp selected station index when stations change
@app.callback(
    Output("selected_station_idx", "data", allow_duplicate=True),
    Input("stations_store", "data"),
    State("selected_station_idx", "data"),
    prevent_initial_call=True,
)
def clamp_selected_idx(stations, cur):
    stations = stations or []
    if not stations:
        return None
    n = len(stations)
    if cur is None:
        return 0
    try:
        i = int(cur)
    except Exception:
        return 0
    if i < 0:
        return 0
    if i >= n:
        return n - 1
    return no_update

@app.callback(
    Output("selected_station_idx", "data", allow_duplicate=True),
    Input({"type": "station_row", "index": ALL}, "n_clicks"),
    State("selected_station_idx", "data"),
    prevent_initial_call=True,
)
def select_station(_clicks, cur):
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
    try:
        idx = json.loads(prop_id)["index"]
        return int(idx)
    except Exception:
        return no_update



# ----------------------------
# Station list actions (move, remove, insert, clear, reverse)
# ----------------------------

@app.callback(
    Output("stations_store", "data", allow_duplicate=True),
    Output("metrics_store", "data", allow_duplicate=True),
    Input("move_up", "n_clicks"),
    Input("move_down", "n_clicks"),
    Input("remove_selected", "n_clicks"),
    Input("insert_station", "n_clicks"),
    Input("clear_stations", "n_clicks"),
    Input("remove_last", "n_clicks"),
    Input("reverse_order", "n_clicks"),
    State("stations_store", "data"),
    State("selected_station_idx", "data"),
    State("ins_idx", "value"),
    State("ins_lat", "value"),
    State("ins_lon", "value"),
    prevent_initial_call=True,
)
def station_actions(n_up, n_down, n_rm, n_ins, n_clear, n_last, n_rev, stations, selected_idx, ins_idx, ins_lat, ins_lon):
    stations = list(stations or [])
    if not stations and not (callback_context.triggered and callback_context.triggered[0]["prop_id"].startswith("clear_stations")):
        return no_update, no_update

    trig = callback_context.triggered[0]["prop_id"].split(".")[0]

    if trig == "clear_stations":
        return [], None

    if trig == "remove_last":
        if stations:
            stations.pop()
            return stations, None
        return no_update, no_update

    if trig == "reverse_order":
        if len(stations) >= 2:
            stations = list(reversed(stations))
            return stations, None
        return no_update, no_update

    # Selected index
    if selected_idx is None:
        selected_idx = 0
    i = int(selected_idx)

    if trig == "move_up":
        if 0 < i < len(stations):
            stations[i - 1], stations[i] = stations[i], stations[i - 1]
            return stations, None
        return no_update, no_update

    if trig == "move_down":
        if 0 <= i < len(stations) - 1:
            stations[i + 1], stations[i] = stations[i], stations[i + 1]
            return stations, None
        return no_update, no_update

    if trig == "remove_selected":
        if 0 <= i < len(stations):
            stations.pop(i)
            return stations, None
        return no_update, no_update

    if trig == "insert_station":
        if ins_lat is None or ins_lon is None:
            return no_update, no_update
        pos = int(ins_idx or (len(stations) + 1)) - 1
        pos = max(0, min(pos, len(stations)))
        stations.insert(pos, (float(ins_lat), float(ins_lon)))
        return stations, None

    return no_update, no_update


# ----------------------------
# Compute
# ----------------------------

@app.callback(
    Output("metrics_store", "data", allow_duplicate=True),
    Output("compute_status", "children"),
    Input("compute_btn", "n_clicks"),
    State("stations_store", "data"),
    State("graph_path", "value"),
    State("bg_path", "value"),
    State("od_path", "value"),
    State("walk_thresholds", "value"),
    State("bike_thresholds", "value"),
    State("restrict_mode", "value"),
    State("w_type", "value"),
    State("lambda_km", "value"),
    State("cruise_kmh", "value"),
    State("dwell_sec", "value"),
    State("assign_method", "value"),
    State("assign_walk_cutoff", "value"),
    State("return_nodes", "value"),
    State("return_polys", "value"),
    State("poly_min_points", "value"),
    State("poly_buffer", "value"),
    prevent_initial_call=True,
)
def compute_cb(_n, stations, graph_path, bg_path, od_path,
               walk_thresholds, bike_thresholds, restrict_mode,
               w_type, lambda_km, cruise_kmh, dwell_sec,
               assign_method, assign_walk_cutoff,
               return_nodes, return_polys, poly_min_points, poly_buffer):
    stations = stations or []
    if len(stations) == 0:
        return no_update, html.Div("Add at least one station first.", style={"color": "#b00020"})

    params_dict = dict(
        walk_thresholds=walk_thresholds,
        bike_thresholds=bike_thresholds,
        restrict_mode=restrict_mode,
        w_type=w_type,
        lambda_km=lambda_km,
        cruise_kmh=cruise_kmh,
        dwell_sec=dwell_sec,
        assign_method=assign_method,
        assign_walk_cutoff=assign_walk_cutoff,
        return_nodes=bool(return_nodes and "yes" in return_nodes),
        return_polys=bool(return_polys and "yes" in return_polys),
        poly_min_points=poly_min_points,
        poly_buffer=poly_buffer,
    )
    params = _scenario_params_from_inputs(params_dict)

    try:
        assets = get_assets_cached(str(graph_path), str(bg_path), str(od_path))
    except Exception as e:
        return no_update, html.Div(f"Failed to load assets: {e}", style={"color": "#b00020"})

    try:
        station_lonlat = latlon_to_lonlat_array([(float(a), float(b)) for a, b in stations])
        out = compute_line_metrics(assets, station_lonlat, params)
        return out, html.Div("Done.", style={"color": "#0b7a0b", "fontWeight": 700})
    except Exception as e:
        return no_update, html.Div(f"Computation failed: {e}", style={"color": "#b00020"})


@app.callback(
    Output("metrics_panel", "children"),
    Output("raw_json", "children"),
    Input("metrics_store", "data"),
)
def render_metrics(metrics):
    if not metrics:
        return html.Div("Compute metrics to see results here.", className="caption"), ""
    return _metrics_panel(metrics), json.dumps(metrics, indent=2)


# Disable compute button when no stations
@app.callback(
    Output("compute_btn", "disabled"),
    Input("stations_store", "data"),
)
def disable_compute(stations):
    return not bool(stations)


# ----------------------------
# Download JSON
# ----------------------------

@app.callback(
    Output("download_json", "data"),
    Input("download_btn", "n_clicks"),
    State("metrics_store", "data"),
    prevent_initial_call=True,
)
def download_json(_n, metrics):
    if not metrics:
        return no_update
    return dict(content=json.dumps(metrics, indent=2), filename="line_metrics.json")


if __name__ == "__main__":
    # Dash default is http://127.0.0.1:8050
    app.run(debug=True, host="127.0.0.1", port=8050)

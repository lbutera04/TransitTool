"""
app.py — Streamlit front-end for the sketch-level transit ridership tool

Features
--------
- Loads heavy assets once (graph CSR, BG mapping, Replica OD) via st.cache_resource
- Interactive map:
    * Add stations by clicking the map (in order)
    * Drag stations (Leaflet Draw "Edit" mode)
    * Delete stations (Draw "Delete" mode)
- "Compute new line metrics" button:
    * snaps stations to nearest graph node
    * computes walk/bike isochrones (per station + thresholds)
    * computes α_i BG access fractions
    * estimates ridership with conservative multiplicative rule + distance weighting
- Visualizes:
    * station markers
    * catchment polygons (convex hull approximation from backend)
    * key metrics (accessibility + ridership + diagnostics)

Dependencies
------------
pip:
  streamlit
  folium
  streamlit-folium
  numpy
  pandas

Plus whatever your core.py needs (scipy, shapely, etc.)

Run
---
streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os

import numpy as np
import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

from core import warm_start, compute_line_metrics, ScenarioParams
from transit_map_component import transit_map


# ----------------------------
# Config
# ----------------------------

st.set_page_config(
    page_title="Transit Line Estimation Tool",
    layout="wide",
)

st.markdown(
    """
    <style>
      /* Pull main content up (removes dead space above first element) */
      [data-testid="stAppViewContainer"] .main .block-container{
          padding-top: 0.4rem !important;
          padding-bottom: 0.4rem !important;
      }

      /* Also remove the extra padding Streamlit sometimes adds around the main section */
      section.main > div {
          padding-top: 0rem !important;
      }

      /* Tighten the first title spacing */
      h1 {
          margin-top: 0rem;
          padding-top: 0rem;
          margin-bottom: 0.25rem;
      }

      /* Optional: slightly reduce space between elements globally */
      .stMarkdown, .stText, .stCaption {
          margin-top: 0rem;
      }
    </style>

    <style>
    /* --- Make metric labels wrap (no ellipsis) --- */
    div[data-testid="stMetricLabel"] * {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        word-break: normal !important;
        overflow-wrap: anywhere !important;
        line-height: 1.15 !important;
    }

    /* --- Slightly shrink subheaders in the 3-column metrics panel
           and prevent mid-word breaking like "Accessibilit\\ny" --- */
    div[data-testid="stHorizontalBlock"] h3 {
        font-size: 1.6rem !important;
        word-break: keep-all !important;
        overflow-wrap: normal !important;
        hyphens: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path(os.environ.get("TRANSITTOOL_DATA_DIR", REPO_ROOT / "data" / "transit_tool")).expanduser()

DEFAULT_CENTER = (37.7749, -122.4194)  # (lat, lon) SF-ish; change for your region
DEFAULT_ZOOM = 12


# ----------------------------
# Helpers: station storage / parsing
# ----------------------------

def _ramp_red_yellow_green(t: float) -> str:
    """
    t in [0,1]
    0.00 -> deep red
    0.35 -> orange
    0.65 -> yellow
    1.00 -> green
    """
    t = max(0.0, min(1.0, float(t)))

    if t <= 0.35:
        # red -> orange
        a = t / 0.35
        r = 255
        g = int(60 + a * (165 - 60))   # 60 -> 165
        b = 0

    elif t <= 0.65:
        # orange -> yellow
        a = (t - 0.35) / (0.65 - 0.35)
        r = 255
        g = int(165 + a * (255 - 165)) # 165 -> 255
        b = 0

    else:
        # yellow -> green
        a = (t - 0.65) / (1.0 - 0.65)
        r = int(255 - a * 255)         # 255 -> 0
        g = 255
        b = 0

    return f"#{r:02x}{g:02x}{b:02x}"

def _opacity_ramp_centered(
    t: float,
    *,
    max_opacity: float,
    min_opacity: float,
    gamma: float = 1.6,
) -> float:
    """
    Opacity decay from center outward.

    t in [0,1]
    gamma > 1 biases opacity toward the center
    """
    t = max(0.0, min(1.0, float(t)))

    # invert so 1.0 = center, 0.0 = edge
    w = 1.0 - t

    # nonlinear falloff (gamma controls steepness)
    w = w ** gamma

    return min_opacity + w * (max_opacity - min_opacity)

def _pick_threshold_layer(layer_dict: dict, thresholds_min: list[int]) -> dict:
    """Pick the max-threshold GeoJSON layer regardless of key type (str/int/float)."""
    if not layer_dict:
        return {}
    tmax = max(thresholds_min)
    return (
        layer_dict.get(str(tmax))
        or layer_dict.get(int(tmax))
        or layer_dict.get(float(tmax))
        or {}
    )

def _as_featurecollection(g: object) -> dict:
    """Coerce various GeoJSON-ish shapes into a FeatureCollection for Leaflet."""
    if not isinstance(g, dict):
        return {"type": "FeatureCollection", "features": []}

    t = g.get("type")
    if t == "FeatureCollection" and isinstance(g.get("features"), list):
        return g

    if t == "Feature":
        return {"type": "FeatureCollection", "features": [g]}

    # Geometry object
    if isinstance(t, str) and t in {
        "Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon", "GeometryCollection"
    }:
        return {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": g}]}

    # Unknown dict shape → treat as empty
    return {"type": "FeatureCollection", "features": []}


def _haversine_miles(lat1, lon1, lat2, lon2) -> float:
    # Earth radius in miles
    R = 3958.7613
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlmb/2)**2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def _ensure_session_state():
    if "stations_latlon" not in st.session_state:
        # List of (lat, lon) in station order
        st.session_state.stations_latlon = []
    if "map_center" not in st.session_state:
        st.session_state.map_center = DEFAULT_CENTER  # (lat, lon)
    if "map_zoom" not in st.session_state:
        st.session_state.map_zoom = DEFAULT_ZOOM
    if "last_metrics" not in st.session_state:
        st.session_state.last_metrics = None
    if "last_drawn_point" not in st.session_state:
        st.session_state.last_drawn_point = None  # (lat, lon) last appended from draw tool
    if "stations_version" not in st.session_state:
        st.session_state.stations_version = 0

def _bump_stations_version() -> None:
    """Force the React map component to accept the Python-side station list."""
    st.session_state.stations_version = int(st.session_state.get("stations_version", 0)) + 1


def _latlon_to_lonlat_array(stations_latlon: List[Tuple[float, float]]) -> np.ndarray:
    # backend expects (lon,lat)
    if not stations_latlon:
        return np.zeros((0, 2), dtype=float)
    arr = np.array([[lon, lat] for (lat, lon) in stations_latlon], dtype=float)
    return arr


def _extract_markers_from_drawings(drawings: Any) -> List[Tuple[float, float]]:
    """
    Extract markers (lat, lon) from st_folium returned 'all_drawings'.
    We preserve the order that Leaflet reports them; typically correlates with creation order.
    """
    out: List[Tuple[float, float]] = []
    if not drawings:
        return out

    # st_folium returns list of GeoJSON-like features
    for feat in drawings:
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


def _sync_stations_from_map(map_state: Dict[str, Any]) -> None:
    lad = map_state.get("last_active_drawing")
    if lad:
        try:
            geom = lad.get("geometry", {})
            if geom.get("type") == "Point":
                coords = geom.get("coordinates")  # [lon, lat]
                if coords and len(coords) >= 2:
                    lon, lat = float(coords[0]), float(coords[1])
                    pt = (lat, lon)
                    if not st.session_state.stations_latlon or pt != st.session_state.stations_latlon[-1]:
                        st.session_state.stations_latlon.append(pt)
                        st.session_state.last_metrics = None
        except Exception:
            pass

    drawings = map_state.get("all_drawings", None)
    if isinstance(drawings, list):
        drawn_markers = _extract_markers_from_drawings(drawings)

        # Treat non-empty marker set as authoritative (edit/reorder/delete via Draw)
        if drawn_markers:
            st.session_state.stations_latlon = drawn_markers
            st.session_state.last_metrics = None

        # IMPORTANT: do NOT clear stations just because drawings == []
        # (click-to-add and other events can yield [] even without deletions)


def _add_station_from_click(map_state: Dict[str, Any]) -> None:
    lc = map_state.get("last_clicked")
    if not lc:
        return

    # Persist current view (so a rerun keeps your pan/zoom)
    c = map_state.get("center")
    z = map_state.get("zoom")
    if c and "lat" in c and "lng" in c:
        st.session_state.map_center = (float(c["lat"]), float(c["lng"]))
    if z is not None:
        st.session_state.map_zoom = int(z)

    lat = float(lc.get("lat", 0.0))
    lon = float(lc.get("lng", 0.0))
    pt = (lat, lon)

    # Debounce: streamlit-folium can "replay" last_clicked across reruns
    # If we've already appended this exact point as the last station, do nothing.
    if st.session_state.stations_latlon and pt == st.session_state.stations_latlon[-1]:
        return

    st.session_state.stations_latlon.append(pt)
    st.session_state.last_metrics = None

    # KEY FIX:
    # The click arrives AFTER the map was rendered in this run.
    # Force an immediate rerun so the map rebuild includes the new marker right away.
    st.rerun()
    lc = map_state.get("last_clicked", None)
    if not lc:
        return

    # Persist current view BEFORE triggering a rerun
    c = map_state.get("center")
    z = map_state.get("zoom")
    if c and "lat" in c and "lng" in c:
        st.session_state.map_center = (float(c["lat"]), float(c["lng"]))
    if z is not None:
        st.session_state.map_zoom = int(z)

    lat = float(lc.get("lat", 0.0))
    lon = float(lc.get("lng", 0.0))

    if st.session_state.stations_latlon and abs(st.session_state.stations_latlon[-1][0] - lat) < 1e-10 and abs(st.session_state.stations_latlon[-1][1] - lon) < 1e-10:
        return

    st.session_state.stations_latlon.append((lat, lon))
    st.session_state.last_metrics = None


def _make_base_map(center_latlon: Tuple[float, float], zoom: int) -> folium.Map:
    m = folium.Map(location=center_latlon, zoom_start=zoom, tiles="cartodbpositron", control_scale=True)
    return m


def _add_station_layers(m: folium.Map, stations_latlon: List[Tuple[float, float]]) -> None:
    # Draw existing stations as draggable markers (visual only; persistence handled by Draw tool)
    for i, (lat, lon) in enumerate(stations_latlon):
        folium.Marker(
            location=(lat, lon),
            tooltip=f"Station {i+1}",
            icon=folium.Icon(color="blue", icon="train", prefix="fa"),
            draggable=False,  # dragging persistence is handled via Draw edit
        ).add_to(m)


    # Connect stations in order (faint black line)
    if len(stations_latlon) >= 2:
        folium.PolyLine(
            locations=stations_latlon,
            color="black",
            weight=4,
            opacity=0.5,
        ).add_to(m)


def _add_geojson_layer(m: folium.Map, geojson_fc: Dict[str, Any], name: str, color: Optional[str] = None) -> None:
    if not geojson_fc or not geojson_fc.get("features"):
        return

    style_fn = None
    if color is not None:
        def style_fn(_):
            return {"color": color, "weight": 2, "fillOpacity": 0.15}

    folium.GeoJson(
        geojson_fc,
        name=name,
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["station_index", "threshold_min", "mode"],
            aliases=["Station", "Threshold (min)", "Mode"],
            localize=True,
            sticky=False,
        ),
    ).add_to(m)


def _render_metrics_panel(metrics: Dict[str, Any]) -> None:
    rid = metrics.get("ridership", {})
    acc = metrics.get("accessibility", {})
    diag = metrics.get("diagnostics", {})

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Ridership")
        st.markdown("**Estimated daily linked riders:**")
        st.markdown(f"<div style='font-size: 2rem; font-weight: 700; line-height: 1.1'>{rid.get('total_riders', 0.0):,.0f}</div>", unsafe_allow_html=True)
        b = rid.get("boardings_by_station", [])
        if b:
            st.caption("Boardings by station")
            st.write({f"{i+1}": round(float(x), 1) for i, x in enumerate(b)})
        total = float(rid.get("total_riders", 0.0))

        stations = st.session_state.stations_latlon
        route_mi = 0.0
        if len(stations) >= 2:
            for (lat1, lon1), (lat2, lon2) in zip(stations[:-1], stations[1:]):
                route_mi += _haversine_miles(lat1, lon1, lat2, lon2)

        if route_mi > 1e-6:
            st.markdown("**Estimated riders per mile:**")
            st.markdown(f"<div style='font-size: 2rem; font-weight: 700; line-height: 1.1'>{total / route_mi:,.0f}</div>", unsafe_allow_html=True)
            st.caption(f"Approx route length: {route_mi:.2f} mi (straight-line station hops)")


    with col2:
        st.subheader("Accessibility")
        walk = acc.get("walk", {})
        bike = acc.get("bike", {})
        if walk:
            st.caption("Population within walk thresholds (unioned)")
            for t, v in walk.items():
                st.write(f"Walk ≤ {t} min: **{v:,.0f}**")
        if bike:
            st.caption("Population within bike thresholds (unioned)")
            for t, v in bike.items():
                st.write(f"Bike ≤ {t} min: **{v:,.0f}**")

    with col3:
        st.subheader("Diagnostics")

        # Friendly names + meaning
        defs = {
            "alpha_mean": ("Mean coverage α", "Average share of each origin BG captured by at least one station catchment (0–1)."),
            "alpha_p50": ("Median coverage α", "50th percentile of BG coverage α across origins (0–1)."),
            "alpha_p90": ("90th pct coverage α", "90th percentile of BG coverage α across origins (0–1)."),
            "w_d_mean": ("Mean distance weight w(d)", "Average OD distance downweight applied to included trips (0–1)."),
            "od_rows_used": ("OD rows used", "Number of OD records used in ridership aggregation (count)."),
            "in_vehicle_time_min_mean": ("Mean in-vehicle time", "Mean in-vehicle time between assigned O→D stations (minutes)."),
            "in_vehicle_time_min_p90": ("90th pct in-vehicle time", "90th percentile in-vehicle time between assigned O→D stations (minutes)."),
        }

        for k, (label, desc) in defs.items():
            if k in diag:
                v = diag[k]
                if isinstance(v, (int, float)):
                    # nicer formatting: counts as integer-ish, others compact
                    if "rows" in k:
                        val = f"{float(v):,.0f}"
                    else:
                        val = f"{float(v):.3f}"
                else:
                    val = str(v)

                st.write(f"**{label}: {val}**")
                st.caption(desc)

        with st.expander("Raw diagnostics"):
            st.json(diag)



# ----------------------------
# Sidebar: paths + params
# ----------------------------

def _sidebar_inputs() -> Tuple[str, str, str, ScenarioParams]:
    st.sidebar.header("Data paths")

    graph_path = st.sidebar.text_input(
        "Graph CSR .npz",
        value=str(DEFAULT_DATA_DIR / "graph_csr.npz"),
    )

    bg_path = st.sidebar.text_input(
        "Block-group mapping .npz",
        value=str(DEFAULT_DATA_DIR / "bg_mapping.npz"),
    )

    od_path = st.sidebar.text_input(
        "Replica BG→BG OD (parquet/csv)",
        value=str(DEFAULT_DATA_DIR / "replica_bg_od.parquet"),  # or whatever your actual filename is
    )

    st.sidebar.header("Model params")

    walk_thresholds = st.sidebar.multiselect("Walk thresholds (min)", [3, 5, 8, 10, 12, 15, 20], default=[5, 10, 15])
    bike_thresholds = st.sidebar.multiselect("Bike thresholds (min)", [3, 5, 8, 10, 12, 15, 20], default=[5, 10, 15])

    restrict_mode = st.sidebar.selectbox("Restrict OD to primary mode", ["car", "all"], index=0)
    restrict_to_primary_mode = None if restrict_mode == "all" else restrict_mode

    w_type = st.sidebar.selectbox("Distance weight w(d)", ["exp_saturating", "step"], index=0)
    lambda_km = st.sidebar.slider("lambda (km) for exp_saturating", 1.0, 30.0, 6.0, 0.5)

    cruise_kmh = st.sidebar.slider("Line cruise speed (km/h)", 30.0, 110.0, 75.0, 1.0)
    dwell_sec = st.sidebar.slider("Dwell per stop (sec)", 0.0, 60.0, 10.0, 1.0)

    assign_method = st.sidebar.selectbox("BG→station assignment", ["euclidean", "walk_to_centroid_node"], index=0)
    assign_walk_cutoff = st.sidebar.slider("Assign walk cutoff (min) (if walk_to_centroid_node)", 5.0, 120.0, 30.0, 5.0)

    st.sidebar.header("Output")
    return_nodes = st.sidebar.checkbox("Return catchment nodes (HUGE payload)", value=False)
    return_polys = st.sidebar.checkbox("Return catchment polygons", value=True)
    poly_min_points = st.sidebar.slider("Polygon min points", 10, 200, 40, 5)
    poly_buffer = st.sidebar.number_input("Polygon buffer (deg, lon/lat)", value=0.0005, format="%.6f")

    params = ScenarioParams(
        walk_thresholds_min=tuple(float(x) for x in sorted(walk_thresholds)),
        bike_thresholds_min=tuple(float(x) for x in sorted(bike_thresholds)),
        restrict_to_primary_mode=restrict_to_primary_mode,
        w_distance=w_type,  # type: ignore
        lambda_km=float(lambda_km),
        cruise_kmh=float(cruise_kmh),
        dwell_sec=float(dwell_sec),
        assign_method=assign_method,  # type: ignore
        assign_walk_cutoff_min=float(assign_walk_cutoff),
        return_catchment_nodes=bool(return_nodes),
        return_catchment_polygons=bool(return_polys),
        polygon_min_points=int(poly_min_points),
        polygon_buffer_deg=float(poly_buffer),
    )

    return graph_path, bg_path, od_path, params


# ----------------------------
# Cached asset load
# ----------------------------

@st.cache_resource(show_spinner=True)
def get_assets_cached(graph_path: str, bg_path: str, od_path: str):
    return warm_start(graph_path, bg_path, od_path)


# ----------------------------
# UI
# ----------------------------

def main():
    _ensure_session_state()

    st.title("Transit Line Estimation Tool")
    st.caption("Click to add stations in order. Use the draw toolbar to edit/move/delete markers. Click compute to update metrics.")

    graph_path, bg_path, od_path, params = _sidebar_inputs()

    # Load assets (once)
    try:
        with st.spinner("Loading assets (cached)..."):
            assets = get_assets_cached(graph_path, bg_path, od_path)
    except Exception as e:
        st.error("Failed to load assets. Check filepaths and formats.")
        st.exception(e)
        st.stop()

    # Map + controls
    left, right = st.columns([1.2, 1.0], gap="large")

    with left:
        st.subheader("Map")

        # # base map
        # if st.session_state.stations_latlon:
        #     center = st.session_state.stations_latlon[-1]
        # else:
        #     center = DEFAULT_CENTER

        # m = _make_base_map(center, DEFAULT_ZOOM)
        m = _make_base_map(st.session_state.map_center, st.session_state.map_zoom)

        # Draw tool: allow marker creation + editing + deleting
        Draw(
            export=False,
            position="topleft",
            draw_options={
                "polyline": False,
                "polygon": False,
                "circle": False,
                "rectangle": False,
                "circlemarker": False,
                "marker": True,
            },
            edit_options={"edit": True, "remove": True},
        ).add_to(m)

        # Show current stations
        _add_station_layers(m, st.session_state.stations_latlon)

        # Add last computed catchment polygons if available
        if st.session_state.last_metrics and "layers" in st.session_state.last_metrics:
            layers = st.session_state.last_metrics["layers"].get("catchments", {})
            # Add a couple of layers: max threshold walk/bike
            if layers:
                try:
                    walk = layers.get("walk", {})
                    bike = layers.get("bike", {})
                    if walk:
                        tmax = str(max(params.walk_thresholds_min))
                        _add_geojson_layer(m, walk.get(tmax, {}), f"Walk catchments ≤ {tmax} min", color=None)
                    if bike:
                        tmax = str(max(params.bike_thresholds_min))
                        _add_geojson_layer(m, bike.get(tmax, {}), f"Bike catchments ≤ {tmax} min", color=None)
                except Exception:
                    pass

        folium.LayerControl(collapsed=True).add_to(m)

        # Build GeoJSON layers list in the format the component expects
        geo_layers = []
        if st.session_state.last_metrics and "layers" in st.session_state.last_metrics:
            catch = st.session_state.last_metrics["layers"].get("catchments", {})
            walk = catch.get("walk", {}) if isinstance(catch, dict) else {}
            bike = catch.get("bike", {}) if isinstance(catch, dict) else {}

            # --- WALK rings ---
            w_th = sorted(params.walk_thresholds_min)
            w_min, w_max = w_th[0], w_th[-1]
            for tmin in w_th:
                g = _as_featurecollection(_pick_threshold_layer(walk, [tmin]))
                if not g["features"]:
                    continue

                frac = 0.0 if w_max == w_min else (tmin - w_min) / (w_max - w_min)
                fill = _ramp_red_yellow_green(frac)

                # Only the OUTERMOST ring gets a strong red outline
                is_outer = (tmin == w_max)

                opacity = _opacity_ramp_centered(
                    frac,
                    max_opacity=0.35,   # center
                    min_opacity=0.15,   # outer edge
                    gamma=1.8,
                )

                geo_layers.append({
                    "name": f"Walk ≤ {tmin} min",
                    "geojson": g,
                    "style": {
                        "fillColor": fill,
                        "fillOpacity": opacity,                 # tune
                        "color": "#b71c1c" if is_outer else "#000000",
                        "opacity": 1.0 if is_outer else 0.0, # hide interior outlines
                        "weight": 2 if is_outer else 0,
                    },
                })

            # --- BIKE rings ---
            b_th = sorted(params.bike_thresholds_min)
            b_min, b_max = b_th[0], b_th[-1]
            for tmin in b_th:
                g = _as_featurecollection(_pick_threshold_layer(bike, [tmin]))
                if not g["features"]:
                    continue

                frac = 0.0 if b_max == b_min else (tmin - b_min) / (b_max - b_min)
                fill = _ramp_red_yellow_green(frac)

                opacity = _opacity_ramp_centered(
                    frac,
                    max_opacity=0.15,
                    min_opacity=0.05,
                    gamma=1.4,
                )

                geo_layers.append({
                    "name": f"Bike ≤ {tmin} min",
                    "geojson": g,
                    "style": {
                        "fillColor": fill,
                        "fillOpacity": opacity,    # << bike more transparent
                        "color": "#1b5e20",
                        "opacity": 0.25,
                        "weight": 0,
                        "dashArray": "3 6",     # optional: make bike visually distinct
                    },
                })

        # Keep your existing UX toggle
        click_add = st.checkbox("Click map to add station", value=True, key="click_add_station")

        station_tooltips = [f"Station {i+1}" for i in range(len(st.session_state.stations_latlon))]

        # Render the React map (NO folium; NO iframe reload per rerun)
        map_state = transit_map(
            stations=st.session_state.stations_latlon,
            center=st.session_state.map_center,
            zoom=st.session_state.map_zoom,
            click_to_add=click_add,
            geojson_layers=geo_layers,
            height=650,
            station_tooltips=station_tooltips,   # <-- add this
            stations_version=int(st.session_state.stations_version),
            key="main_map_react",
        )

        # --- Buttons directly under the map ---
        btn1, btn2, btn3 = st.columns(3)

        with btn1:
            if st.button("Clear All Stations", key="btn_clear_stations"):
                st.session_state.stations_latlon = []
                st.session_state.last_metrics = None
                _bump_stations_version()
                st.rerun()

        with btn2:
            if st.button("Remove Last Station", key="btn_remove_last") and st.session_state.stations_latlon:
                st.session_state.stations_latlon.pop()
                st.session_state.last_metrics = None
                _bump_stations_version()
                st.rerun()

        with btn3:
            if st.button("Reverse Station Order", key="btn_reverse") and len(st.session_state.stations_latlon) >= 2:
                st.session_state.stations_latlon = list(reversed(st.session_state.stations_latlon))
                st.session_state.last_metrics = None
                _bump_stations_version()
                st.rerun()

        st.divider()

        # ---- Sync React map -> Streamlit session state ----
        prev_stations = list(st.session_state.get("stations_latlon", []))

        applied_v = (map_state or {}).get("appliedStationsVersion", None)
        expected_v = int(st.session_state.stations_version)

        # Only trust the component's stations if the frontend says it has applied
        # the current Python stations_version.
        trust_stations = (applied_v is not None) and (int(applied_v) == expected_v)

        raw = (map_state or {}).get("stations") or []
        new_stations = []
        for p in raw:
            try:
                lat = float(p[0])
                lon = float(p[1])
                new_stations.append((lat, lon))
            except Exception:
                pass

        if trust_stations:
            if new_stations != prev_stations:
                st.session_state.stations_latlon = new_stations

                had_metrics = st.session_state.last_metrics is not None
                st.session_state.last_metrics = None

                if had_metrics:
                    st.rerun()  # <-- IMPORTANT: re-render map with geo_layers cleared
        else:
            # Still allow view updates even when we don't trust stations yet
            pass

        # Persist view (optional)
        # c = (map_state or {}).get("center")
        # z = (map_state or {}).get("zoom")
        # if c and len(c) == 2:
        #     st.session_state.map_center = (float(c[0]), float(c[1]))
        # if z is not None:
        #     st.session_state.map_zoom = int(z)
        # -----------------------------------------------

        # Station list editor
        st.markdown("**Stations (ordered):**")
        if st.session_state.stations_latlon:
            for i, (lat, lon) in enumerate(st.session_state.stations_latlon):
                st.write(f"{i+1}. ({lat:.6f}, {lon:.6f})")
        else:
            st.info("No stations yet. Click on the map to add stations (or use the marker draw tool).")

        st.markdown("**Stations (ordered):**")

        stations = st.session_state.stations_latlon
        if stations:
            for i, (lat, lon) in enumerate(stations):
                c1, c2, c3, c4 = st.columns([0.12, 0.68, 0.10, 0.10])
                with c1:
                    st.write(f"**{i+1}.**")
                with c2:
                    st.write(f"({lat:.6f}, {lon:.6f})")
                with c3:
                    up_disabled = (i == 0)
                    if st.button("↑", key=f"up_{i}", disabled=up_disabled):
                        stations[i-1], stations[i] = stations[i], stations[i-1]
                        st.session_state.last_metrics = None
                        _bump_stations_version()
                        st.rerun()
                with c4:
                    down_disabled = (i == len(stations)-1)
                    if st.button("↓", key=f"down_{i}", disabled=down_disabled):
                        stations[i+1], stations[i] = stations[i], stations[i+1]
                        st.session_state.last_metrics = None
                        _bump_stations_version()
                        st.rerun()

            # Optional: infill insert
            st.caption("Insert infill station")
            ins_idx = st.number_input("Insert position", min_value=1, max_value=len(stations)+1, value=len(stations)+1, step=1)
            ins_lat = st.number_input("Lat", value=float(stations[-1][0]), format="%.6f")
            ins_lon = st.number_input("Lon", value=float(stations[-1][1]), format="%.6f")
            if st.button("Insert station"):
                stations.insert(int(ins_idx)-1, (float(ins_lat), float(ins_lon)))
                st.session_state.last_metrics = None
                _bump_stations_version()
                st.rerun()
        else:
            st.info("No stations yet. Click on the map to add stations (or use the marker draw tool).")

        # col_a, col_b, col_c = st.columns(3)
        # with col_a:
        #     if st.button("Clear stations"):
        #         st.session_state.stations_latlon = []
        #         st.session_state.last_metrics = None
        #         _bump_stations_version()
        #         st.rerun()
        # with col_b:
        #     if st.button("Remove last station") and st.session_state.stations_latlon:
        #         st.session_state.stations_latlon.pop()
        #         st.session_state.last_metrics = None
        #         _bump_stations_version()
        #         st.rerun()
        # with col_c:
        #     if st.button("Reverse order") and len(st.session_state.stations_latlon) >= 2:
        #         st.session_state.stations_latlon = list(reversed(st.session_state.stations_latlon))
        #         st.session_state.last_metrics = None
        #         _bump_stations_version()
        #         st.rerun()

    with right:
        st.subheader("Compute")
        st.write("When you click compute, the backend snaps stations to the graph, builds walk/bike catchments, computes coverage α, and estimates ridership from Replica OD.")

        disabled = len(st.session_state.stations_latlon) == 0
        if st.button("Compute new line metrics", type="primary", disabled=disabled):
            station_lonlat = _latlon_to_lonlat_array(st.session_state.stations_latlon)

            try:
                with st.spinner("Computing line metrics..."):
                    out = compute_line_metrics(assets, station_lonlat, params)
                st.session_state.last_metrics = out
                st.success("Done.")
                st.rerun()
            except Exception as e:
                st.error("Computation failed.")
                st.exception(e)

        if st.session_state.last_metrics:
            st.divider()
            _render_metrics_panel(st.session_state.last_metrics)

            st.divider()
            st.subheader("Download / Inspect")
            payload = st.session_state.last_metrics
            st.download_button(
                "Download results JSON",
                data=json.dumps(payload).encode("utf-8"),
                file_name="line_metrics.json",
                mime="application/json",
            )
            with st.expander("Show raw JSON"):
                st.json(payload)
        else:
            st.info("Compute metrics to see results here.")

    st.caption(
        "Tip: For smoother editing, use the draw toolbar's **Edit** mode to drag markers, then click **Compute**."
    )


if __name__ == "__main__":
    main()

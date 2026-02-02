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

import numpy as np
import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

from core import warm_start, compute_line_metrics, ScenarioParams


# ----------------------------
# Config
# ----------------------------

st.set_page_config(
    page_title="Transit Line Sketch Tool",
    layout="wide",
)

DEFAULT_CENTER = (37.7749, -122.4194)  # (lat, lon) SF-ish; change for your region
DEFAULT_ZOOM = 12


# ----------------------------
# Helpers: station storage / parsing
# ----------------------------

def _ensure_session_state():
    if "stations_latlon" not in st.session_state:
        # List of (lat, lon) in station order
        st.session_state.stations_latlon = []

    if "last_metrics" not in st.session_state:
        st.session_state.last_metrics = None


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
    """
    Sync session_state.stations_latlon from map draw/edit/delete.
    Priority:
      - If user used Draw markers, use those.
      - Else keep existing stations.
    """
    drawings = map_state.get("all_drawings", None)
    drawn_markers = _extract_markers_from_drawings(drawings)
    if drawn_markers:
        st.session_state.stations_latlon = drawn_markers


def _add_station_from_click(map_state: Dict[str, Any]) -> None:
    """
    Add a station at last clicked location (if present).
    Note: st_folium provides 'last_clicked' with {'lat':..., 'lng':...}
    """
    lc = map_state.get("last_clicked", None)
    if not lc:
        return
    lat = float(lc.get("lat", 0.0))
    lon = float(lc.get("lng", 0.0))
    # Avoid duplicate adds on rerun: only add if it differs from last station
    if st.session_state.stations_latlon and (abs(st.session_state.stations_latlon[-1][0] - lat) < 1e-10) and (
        abs(st.session_state.stations_latlon[-1][1] - lon) < 1e-10
    ):
        return
    st.session_state.stations_latlon.append((lat, lon))


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
        st.metric("Estimated daily linked riders", f"{rid.get('total_riders', 0.0):,.0f}")
        b = rid.get("boardings_by_station", [])
        if b:
            st.caption("Boardings by station")
            st.write([round(x, 1) for x in b])

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
        for k in ["alpha_mean", "alpha_p50", "alpha_p90", "w_d_mean", "od_rows_used", "in_vehicle_time_min_mean", "in_vehicle_time_min_p90"]:
            if k in diag:
                st.write(f"{k}: **{diag[k]:.4g}**" if isinstance(diag[k], (int, float)) else f"{k}: **{diag[k]}**")


# ----------------------------
# Sidebar: paths + params
# ----------------------------

def _sidebar_inputs() -> Tuple[str, str, str, ScenarioParams]:
    st.sidebar.header("Data paths")

    graph_path = st.sidebar.text_input(
        "Graph CSR .npz",
        value=str(Path.home() / "data" / "transit_tool" / "graph_csr.npz"),
    )
    bg_path = st.sidebar.text_input(
        "Block-group mapping .npz",
        value=str(Path.home() / "data" / "transit_tool" / "bg_mapping.npz"),
    )
    od_path = st.sidebar.text_input(
        "Replica BG→BG OD (parquet/csv)",
        value=str(Path.home() / "data" / "transit_tool" / "replica_bg_od.parquet"),
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

    st.title("Transit Line Sketch Tool")
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

        # base map
        if st.session_state.stations_latlon:
            center = st.session_state.stations_latlon[-1]
        else:
            center = DEFAULT_CENTER

        m = _make_base_map(center, DEFAULT_ZOOM)

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

        # Render and capture map state
        map_state = st_folium(m, height=650, width=None, returned_objects=["all_drawings", "last_clicked"])

        # Sync stations from draw/edit/delete
        _sync_stations_from_map(map_state)

        # Add stations on click (convenient)
        click_add = st.checkbox("Click map to add station", value=True)
        if click_add:
            _add_station_from_click(map_state)

        # Station list editor
        st.markdown("**Stations (ordered):**")
        if st.session_state.stations_latlon:
            for i, (lat, lon) in enumerate(st.session_state.stations_latlon):
                st.write(f"{i+1}. ({lat:.6f}, {lon:.6f})")
        else:
            st.info("No stations yet. Click on the map to add stations (or use the marker draw tool).")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("Clear stations"):
                st.session_state.stations_latlon = []
                st.session_state.last_metrics = None
                st.rerun()
        with col_b:
            if st.button("Remove last station") and st.session_state.stations_latlon:
                st.session_state.stations_latlon.pop()
                st.session_state.last_metrics = None
                st.rerun()
        with col_c:
            if st.button("Reverse order") and len(st.session_state.stations_latlon) >= 2:
                st.session_state.stations_latlon = list(reversed(st.session_state.stations_latlon))
                st.session_state.last_metrics = None
                st.rerun()

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

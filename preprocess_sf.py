"""
preprocess_sf.py — one-shot preprocessing to make app.py run TODAY

Inputs (from Downloads)
-----------------------
1) Geofabrik .osm.pbf covering San Francisco (or California extract; we will clip by bbox/county-ish if needed)
2) Replica OD CSV filtered to SF county (BG→BG or can be mapped to BG GEOIDs)
3) Block group GeoJSON (Census geographies)

Outputs (written to)
--------------------
~/Code/TransitTool/data/transit_tool/
  graph_csr.npz        # CSR graph arrays for walk/bike (minutes)
  bg_mapping.npz       # BG↔node weighted mapping for alpha fractions
  replica_bg_od.parquet# Cleaned OD (o_bg,d_bg,trips,distance_km,primary_mode)

Then app.py should run immediately, pointing to those filepaths.

Install deps (recommended via conda/mamba)
-----------------------------------------
mamba create -n transittool -c conda-forge python=3.12 \
  numpy pandas scipy geopandas shapely pyproj pyarrow folium streamlit streamlit-folium

For PBF reading:
  pip install pyrosm
If pyrosm fails on your machine, fallback instructions are included in comments.

Run
---
python preprocess_sf.py \
  --pbf ~/Downloads/san-francisco-latest.osm.pbf \
  --replica_csv ~/Downloads/replica_sf_od.csv \
  --bg_geojson ~/Downloads/sf_block_groups.geojson \
  --out_dir ~/Code/TransitTool/data/transit_tool \
  --samples_per_bg 25

Notes
-----
- This builds a *walk/bike street graph* from the PBF (using pyrosm).
- α_i is approximated by sampling points uniformly within each BG polygon,
  snapping each sample to nearest graph node, and using equal weights.
  This gives fractional coverage without expensive polygon intersections.
- If your Replica CSV does not have distance_km, we compute it from BG centroid haversine.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import geopandas as gpd
from shapely.geometry import Point
from shapely.prepared import prep

import subprocess
import tempfile
import json
import osmium


# ----------------------------
# Small utilities
# ----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def haversine_km(lon1, lat1, lon2, lat2):
    """
    Vectorized haversine. Inputs can be numpy arrays.
    Returns distance in km.
    """
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)

    R = 6371.0
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlmb = np.deg2rad(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(a))


def choose_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    # try case-insensitive match
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def bg_id_colname(gdf: gpd.GeoDataFrame) -> str:
    c = choose_col(gdf, ["GEOID", "geoid", "bg_geoid", "GEOID20", "GEOID10"])
    if not c:
        raise ValueError("BG geojson is missing GEOID-like column (e.g. GEOID, GEOID20).")
    return c


def pop_colname(gdf: gpd.GeoDataFrame) -> Optional[str]:
    return choose_col(gdf, ["population", "POP", "pop", "totpop", "P0010001", "TOTAL_POP"])


def parse_bbox_sf(bg_gdf: gpd.GeoDataFrame) -> Tuple[float, float, float, float]:
    # bbox in lon/lat: (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = bg_gdf.total_bounds
    # add small buffer
    dx = (maxx - minx) * 0.02
    dy = (maxy - miny) * 0.02
    return (minx - dx, miny - dy, maxx + dx, maxy + dy)


def clip_points_bbox_xy(xy: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
    minx, miny, maxx, maxy = bbox
    m = (xy[:, 0] >= minx) & (xy[:, 0] <= maxx) & (xy[:, 1] >= miny) & (xy[:, 1] <= maxy)
    return xy[m]


# ----------------------------
# Step 1: Build graph CSR from PBF
# ----------------------------

def write_boundary_from_bgs(bg_path: Path, out_geojson: Path, buffer_m: float = 250.0) -> None:
    """
    Creates a single boundary polygon GeoJSON by unioning BG polygons.
    Buffers slightly to avoid cutting off roads at the edge.

    buffer_m is applied in a projected CRS (EPSG:3857) then returned to EPSG:4326.
    """
    gdf = gpd.read_file(bg_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    # Union all BG polygons
    boundary = gdf.geometry.union_all()

    # Buffer in meters
    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary], crs="EPSG:4326").to_crs("EPSG:3857")
    boundary_buf = boundary_gdf.geometry.iloc[0].buffer(buffer_m)
    boundary_buf = gpd.GeoDataFrame(geometry=[boundary_buf], crs="EPSG:3857").to_crs("EPSG:4326").geometry.iloc[0]

    boundary_fc = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {}, "geometry": gpd.GeoSeries([boundary_buf], crs="EPSG:4326").__geo_interface__["features"][0]["geometry"]}],
    }

    out_geojson.write_text(json.dumps(boundary_fc))

def clip_pbf_with_osmium(pbf_in: Path, boundary_geojson: Path, pbf_out: Path) -> None:
    """
    Uses osmium-tool to clip a large PBF to the boundary polygon.

    Requires: osmium (osmium-tool)
      mac: brew install osmium-tool
      conda: mamba install -c conda-forge osmium-tool
    """
    cmd = [
        "osmium", "extract",
        "-p", str(boundary_geojson),
        "-s", "smart",
        "-o", str(pbf_out),
        str(pbf_in),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "osmium-tool not found. Install it:\n"
            "  brew install osmium-tool\n"
            "or\n"
            "  mamba install -c conda-forge osmium-tool"
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"osmium extract failed:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}") from e

def build_graph_csr_from_pbf(
    pbf_path: Path,
    bbox: Tuple[float, float, float, float],
    walk_kmh: float = 5.0,
    bike_kmh: float = 15.0,
):
    """
    Uses pyrosm to load walking/biking networks from a PBF, clipped to bbox.

    Returns:
      node_xy (N,2) lon,lat
      indptr (N+1), indices (M,), w_walk_min (M,), w_bike_min (M,)
    """
    try:
        from pyrosm import OSM
    except Exception as e:
        raise RuntimeError(
            "pyrosm is required to read .osm.pbf. Install with: pip install pyrosm\n"
            "If pyrosm fails, alternative is using OSMnx via Overpass (online)."
        ) from e

    osm = OSM(str(pbf_path))

    # Get nodes/edges for walking. This typically includes footways etc.
    nodes, edges = osm.get_network(network_type="walking", nodes=True)

    if nodes is None or edges is None or len(nodes) == 0 or len(edges) == 0:
        raise RuntimeError("pyrosm returned empty walking network. Check PBF coverage/bbox.")

    # Normalize columns
    # nodes: id, lon, lat
    if "id" not in nodes.columns:
        raise RuntimeError("Unexpected pyrosm nodes schema: missing 'id'.")
    if "lon" not in nodes.columns or "lat" not in nodes.columns:
        raise RuntimeError("Unexpected pyrosm nodes schema: missing lon/lat.")

    # edges: u, v, length
    # pyrosm usually outputs 'u', 'v', 'length'
    if "u" not in edges.columns or "v" not in edges.columns:
        raise RuntimeError("Unexpected pyrosm edges schema: missing u/v.")
    length_col = choose_col(edges, ["length", "length_m", "len", "distance"])
    if not length_col:
        raise RuntimeError("Unexpected pyrosm edges schema: missing edge length column.")

    # Build dense node indexing
    node_ids = nodes["id"].astype(np.int64).to_numpy()
    id2idx = pd.Series(np.arange(node_ids.size, dtype=np.int32), index=node_ids)

    node_xy = nodes[["lon", "lat"]].to_numpy(dtype=np.float64)

    # Convert u/v to dense
    u = id2idx.loc[edges["u"].astype(np.int64)].to_numpy(dtype=np.int32)
    v = id2idx.loc[edges["v"].astype(np.int64)].to_numpy(dtype=np.int32)

    length_m = edges[length_col].to_numpy(dtype=np.float64)
    # Basic speed model (you can later make slope/roadclass dependent)
    walk_mps = (walk_kmh * 1000.0) / 3600.0
    bike_mps = (bike_kmh * 1000.0) / 3600.0
    w_walk_min = (length_m / walk_mps) / 60.0
    w_bike_min = (length_m / bike_mps) / 60.0

    # Build CSR for directed graph. Many walking edges are effectively bidirectional;
    # we add reverse edges too for robustness unless edges already include both.
    # We'll explicitly add both directions.
    uu = np.concatenate([u, v])
    vv = np.concatenate([v, u])
    ww_walk = np.concatenate([w_walk_min, w_walk_min]).astype(np.float32)
    ww_bike = np.concatenate([w_bike_min, w_bike_min]).astype(np.float32)

    N = node_xy.shape[0]
    order = np.argsort(uu, kind="mergesort")
    uu = uu[order]
    vv = vv[order]
    ww_walk = ww_walk[order]
    ww_bike = ww_bike[order]

    indptr = np.zeros(N + 1, dtype=np.int64)
    np.add.at(indptr, uu + 1, 1)
    indptr = np.cumsum(indptr, dtype=np.int64)
    indices = vv.astype(np.int32, copy=False)

    return node_xy, indptr, indices, ww_walk, ww_bike


# ----------------------------
# Step 2: Build BG↔node weights
# ----------------------------

def random_points_in_polygon(poly, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Rejection sampling within polygon bounds.
    Returns (n,2) lon,lat.
    """
    if poly.is_empty:
        return np.zeros((0, 2), dtype=np.float64)

    minx, miny, maxx, maxy = poly.bounds
    ppoly = prep(poly)

    pts = []
    # limit iterations to avoid infinite loops for very skinny shapes
    max_tries = max(5000, n * 500)
    tries = 0
    while len(pts) < n and tries < max_tries:
        tries += 1
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if ppoly.contains(Point(x, y)):
            pts.append((x, y))

    if len(pts) < n:
        # fall back: include representative point
        rp = poly.representative_point()
        pts.append((float(rp.x), float(rp.y)))

    return np.array(pts[:n], dtype=np.float64)


def build_bg_mapping_npz(
    bg_geojson_path: Path,
    node_xy: np.ndarray,
    out_path: Path,
    samples_per_bg: int = 25,
    seed: int = 0,
):
    """
    Reads BG polygons, samples points per BG, snaps to nearest graph nodes,
    and builds a weighted BG↔node mapping for alpha fractions.

    Output keys:
      bg_ids, pop, centroid_node, bg_indptr, bg_nodes, bg_weights
    """
    gdf = gpd.read_file(bg_geojson_path)

    # Ensure lon/lat
    if gdf.crs is None:
        # assume GEOJSON is EPSG:4326
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    id_col = bg_id_colname(gdf)
    pop_col = pop_colname(gdf)

    bg_ids = gdf[id_col].astype(str).to_numpy()
    if pop_col:
        pop = gdf[pop_col].fillna(0).astype(float).to_numpy()
    else:
        # If you don't have population yet, set 1.0 so accessibility returns "BG-equivalents".
        # You can later join ACS population and re-run preprocessing.
        pop = np.ones(len(gdf), dtype=np.float64)

    # KDTree for snapping
    tree = cKDTree(node_xy)

    # centroid nodes
    centroids = gdf.geometry.centroid
    centroid_xy = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])
    _, centroid_node = tree.query(centroid_xy, k=1)
    centroid_node = centroid_node.astype(np.int32)

    rng = np.random.default_rng(seed)

    # Build BG CSR-like mapping arrays
    bg_indptr = np.zeros(len(gdf) + 1, dtype=np.int64)
    bg_nodes_list: List[np.ndarray] = []
    bg_wts_list: List[np.ndarray] = []

    for i, geom in enumerate(gdf.geometry):
        if geom is None or geom.is_empty:
            # no nodes
            bg_nodes_list.append(np.zeros((0,), dtype=np.int32))
            bg_wts_list.append(np.zeros((0,), dtype=np.float32))
            bg_indptr[i + 1] = bg_indptr[i]
            continue

        # sample points, snap to nodes
        pts = random_points_in_polygon(geom, samples_per_bg, rng)  # (S,2) lon,lat
        _, nn = tree.query(pts, k=1)
        nn = nn.astype(np.int32)

        # compress duplicates to keep mapping small
        uniq, counts = np.unique(nn, return_counts=True)
        weights = (counts / counts.sum()).astype(np.float32)

        bg_nodes_list.append(uniq)
        bg_wts_list.append(weights)
        bg_indptr[i + 1] = bg_indptr[i] + uniq.size

    bg_nodes = np.concatenate(bg_nodes_list).astype(np.int32, copy=False) if bg_nodes_list else np.zeros((0,), dtype=np.int32)
    bg_weights = np.concatenate(bg_wts_list).astype(np.float32, copy=False) if bg_wts_list else np.zeros((0,), dtype=np.float32)

    np.savez_compressed(
        out_path,
        bg_ids=bg_ids,
        pop=pop.astype(np.float32),
        centroid_node=centroid_node,
        bg_indptr=bg_indptr.astype(np.int64),
        bg_nodes=bg_nodes,
        bg_weights=bg_weights,
    )


# ----------------------------
# Step 3: Clean Replica OD
# ----------------------------

def clean_replica_od(
    replica_path: Path,
    bg_gdf: gpd.GeoDataFrame,
    out_path: Path,
):
    """
    Accepts either:
      A) trip-level Replica (like your file): origin_bgrp_2020, destination_bgrp_2020, primary_mode, trip_distance_miles
      B) already-aggregated OD: o_bg, d_bg, trips, distance_km, primary_mode

    Produces parquet with columns required by core.py:
      o_bg, d_bg, trips, distance_km, primary_mode
    """
    # ---- read csv/parquet ----
    if str(replica_path).endswith(".parquet"):
        od = pd.read_parquet(replica_path)
    else:
        od = pd.read_csv(replica_path)

    # Helpers
    def choose_col(df: pd.DataFrame, candidates):
        cols = set(df.columns)
        for c in candidates:
            if c in cols:
                return c
        lower_map = {c.lower(): c for c in df.columns}
        for c in candidates:
            if c.lower() in lower_map:
                return lower_map[c.lower()]
        return None

    # Candidate columns for trip-level file
    o_trip = choose_col(od, ["origin_bgrp_2020", "origin_bg", "origin_geoid", "origin"])
    d_trip = choose_col(od, ["destination_bgrp_2020", "destination_bg", "destination_geoid", "destination"])
    mode_col = choose_col(od, ["primary_mode", "mode", "trip_mode"])
    dist_mi_col = choose_col(od, ["trip_distance_miles", "distance_miles", "dist_miles"])

    # Candidate columns for already-aggregated OD
    o_od = choose_col(od, ["o_bg", "origin_bg"])
    d_od = choose_col(od, ["d_bg", "dest_bg", "destination_bg"])
    trips_col = choose_col(od, ["trips", "trip_count", "count", "n_trips", "num_trips"])
    dist_km_col = choose_col(od, ["distance_km", "dist_km", "trip_distance_km", "distance_km_mean"])

    # ---- Case A: already aggregated ----
    if o_od and d_od and trips_col:
        od2 = od[[o_od, d_od, trips_col]].copy()
        od2 = od2.rename(columns={o_od: "o_bg", d_od: "d_bg", trips_col: "trips"})
        od2["o_bg"] = od2["o_bg"].astype(str)
        od2["d_bg"] = od2["d_bg"].astype(str)
        od2["trips"] = od2["trips"].astype(float)

        if mode_col:
            od2["primary_mode"] = od[mode_col].astype(str).str.lower()
        else:
            od2["primary_mode"] = "car"

        if dist_km_col:
            od2["distance_km"] = od[dist_km_col].astype(float)
        else:
            od2["distance_km"] = np.nan  # fill later from BG centroids

    # ---- Case B: trip-level (your file) -> aggregate ----
    elif o_trip and d_trip:
        tmp = pd.DataFrame({
            "o_bg": od[o_trip].astype(str),
            "d_bg": od[d_trip].astype(str),
        })

        if mode_col:
            tmp["primary_mode"] = od[mode_col].astype(str).str.lower()
        else:
            tmp["primary_mode"] = "car"

        # trips = 1 per row, then groupby count
        tmp["trips"] = 1.0

        # distance
        if dist_mi_col:
            # miles -> km
            tmp["distance_km"] = od[dist_mi_col].astype(float) * 1.609344
            # aggregate: mean distance per OD-mode bucket
            agg = tmp.groupby(["o_bg", "d_bg", "primary_mode"], as_index=False).agg(
                trips=("trips", "sum"),
                distance_km=("distance_km", "mean"),
            )
        else:
            # no distance in file; we'll compute from BG centroids later
            agg = tmp.groupby(["o_bg", "d_bg", "primary_mode"], as_index=False).agg(
                trips=("trips", "sum"),
            )
            agg["distance_km"] = np.nan

        od2 = agg

    else:
        raise ValueError(
            "Replica file doesn't match expected schemas.\n"
            f"Found columns: {list(od.columns)}\n"
            "Need either:\n"
            "  - trip-level: origin_bgrp_2020 + destination_bgrp_2020 (and optionally trip_distance_miles)\n"
            "  - aggregated: o_bg + d_bg + trips\n"
        )

    # ---- Fill distance_km from BG centroids if missing ----
    if od2["distance_km"].isna().any():
        # Use projected centroids to avoid the GeoPandas warning
        bg = bg_gdf.copy()
        if bg.crs is None:
            bg = bg.set_crs("EPSG:4326")
        else:
            bg = bg.to_crs("EPSG:4326")

        id_col = choose_col(bg, ["GEOID", "geoid", "GEOID20", "bg_geoid"])
        if not id_col:
            raise ValueError("BG geojson missing GEOID column needed to compute centroid distances.")

        bg_3857 = bg.to_crs("EPSG:3857")
        cent = bg_3857.geometry.centroid.to_crs("EPSG:4326")
        bg_cent = pd.DataFrame({
            "bg": bg[id_col].astype(str).to_numpy(),
            "lon": cent.x.to_numpy(dtype=np.float64),
            "lat": cent.y.to_numpy(dtype=np.float64),
        }).set_index("bg")

        o_lon = bg_cent.reindex(od2["o_bg"])["lon"].to_numpy()
        o_lat = bg_cent.reindex(od2["o_bg"])["lat"].to_numpy()
        d_lon = bg_cent.reindex(od2["d_bg"])["lon"].to_numpy()
        d_lat = bg_cent.reindex(od2["d_bg"])["lat"].to_numpy()

        bad = np.isnan(o_lon) | np.isnan(o_lat) | np.isnan(d_lon) | np.isnan(d_lat)
        # if BG IDs not found, set 0 (rows will be filtered later by bg_index in core anyway)
        o_lon[bad] = 0.0; o_lat[bad] = 0.0; d_lon[bad] = 0.0; d_lat[bad] = 0.0

        # haversine
        def haversine_km(lon1, lat1, lon2, lat2):
            R = 6371.0
            lon1 = np.deg2rad(lon1); lat1 = np.deg2rad(lat1)
            lon2 = np.deg2rad(lon2); lat2 = np.deg2rad(lat2)
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
            return 2*R*np.arcsin(np.sqrt(a))

        fill = haversine_km(o_lon, o_lat, d_lon, d_lat).astype(np.float32)
        od2.loc[od2["distance_km"].isna(), "distance_km"] = fill[od2["distance_km"].isna().to_numpy()]

    # ---- Final columns + save ----
    od2 = od2[["o_bg", "d_bg", "trips", "distance_km", "primary_mode"]].copy()
    od2["trips"] = od2["trips"].astype(float)
    od2["distance_km"] = od2["distance_km"].astype(float)
    od2["primary_mode"] = od2["primary_mode"].astype(str)

    od2.to_parquet(out_path, index=False)

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbf", type=str, required=True, help="Path to geofabrik .osm.pbf")
    ap.add_argument("--replica_parquet", type=str, required=True, help="Path to Replica OD CSV")
    ap.add_argument("--bg_geojson", type=str, required=True, help="Path to block group geojson")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for data/transit_tool")
    ap.add_argument("--samples_per_bg", type=int, default=25, help="Samples per block group for alpha fraction mapping")
    ap.add_argument("--walk_kmh", type=float, default=5.0)
    ap.add_argument("--bike_kmh", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pbf_path = Path(os.path.expanduser(args.pbf))
    replica_path = Path(os.path.expanduser(args.replica_parquet))
    bg_path = Path(os.path.expanduser(args.bg_geojson))
    out_dir = Path(os.path.expanduser(args.out_dir))

    ensure_dir(out_dir)

    graph_out = out_dir / "graph_csr.npz"
    bg_out = out_dir / "bg_mapping.npz"
    od_out = out_dir / "replica_bg_od.parquet"

    # Load BG geojson (for bbox and OD distance)
    bg_gdf = gpd.read_file(bg_path)
    if bg_gdf.crs is None:
        bg_gdf = bg_gdf.set_crs("EPSG:4326")
    else:
        bg_gdf = bg_gdf.to_crs("EPSG:4326")

    bbox = parse_bbox_sf(bg_gdf)

    # --- Build SF boundary polygon from BGs ---
    boundary_geojson = out_dir / "sf_boundary.geojson"
    print("Creating SF boundary polygon from BGs...")
    write_boundary_from_bgs(bg_path, boundary_geojson, buffer_m=250.0)
    print(f"Boundary written to: {boundary_geojson}")

    # --- Clip NorCal PBF down to SF ---
    sf_pbf = out_dir / "sf_clipped.osm.pbf"
    if not sf_pbf.exists():
        print("Clipping PBF to SF boundary (one-time)...")
        clip_pbf_with_osmium(pbf_path, boundary_geojson, sf_pbf)
        print(f"Clipped PBF: {sf_pbf}")
    else:
        print(f"Using existing clipped PBF: {sf_pbf}")

    # Now build CSR from the *clipped* PBF (bbox can just be boundary bounds)
    bg_gdf = gpd.read_file(bg_path)
    if bg_gdf.crs is None:
        bg_gdf = bg_gdf.set_crs("EPSG:4326")
    else:
        bg_gdf = bg_gdf.to_crs("EPSG:4326")

    bbox = parse_bbox_sf(bg_gdf)

    print("Building graph CSR from clipped PBF...")
    node_xy, indptr, indices, w_walk_min, w_bike_min = build_graph_csr_from_pbf(
        sf_pbf,
        bbox=bbox,
        walk_kmh=float(args.walk_kmh),
        bike_kmh=float(args.bike_kmh),
    )

    print(f"Graph nodes: {node_xy.shape[0]:,}  edges(CSR): {indices.shape[0]:,}")
    print(f"Saving {graph_out}")
    np.savez_compressed(
        graph_out,
        indptr=indptr.astype(np.int64),
        indices=indices.astype(np.int32),
        w_walk_min=w_walk_min.astype(np.float32),
        w_bike_min=w_bike_min.astype(np.float32),
        node_xy=node_xy.astype(np.float64),
        directed=True,
    )

    print("Building BG↔node mapping...")
    print(f"Saving {bg_out}")
    build_bg_mapping_npz(
        bg_geojson_path=bg_path,
        node_xy=node_xy,
        out_path=bg_out,
        samples_per_bg=int(args.samples_per_bg),
        seed=int(args.seed),
    )

    print("Cleaning Replica OD...")
    print(f"Saving {od_out}")
    clean_replica_od(
        replica_path=replica_path,
        bg_gdf=bg_gdf,
        out_path=od_out,
    )

    print("\nDone.")
    print("Point app.py to:")
    print(f"  Graph:  {graph_out}")
    print(f"  BGs:    {bg_out}")
    print(f"  OD:     {od_out}")


if __name__ == "__main__":
    main()

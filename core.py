"""
core.py — Sketch-level transit line testing backend (Python + CSR + fast cutoff Dijkstra)

What this file does
-------------------
1) Loads static assets once (graph CSR, BG↔node weights, Replica OD).
2) On each interactive "Compute new line metrics" call:
   - snaps user station locations to nearest graph node
   - runs cutoff-Dijkstra from each station node to build walk/bike isochrone node sets
   - computes BG access fractions α_i using BG↔node weights (unioned catchments, no double count)
   - optionally assigns each BG to a station (euclidean or walk-time to BG centroid node)
   - runs conservative multiplicative ridership model with distance weighting w(d)
   - returns a single JSON-serializable payload for the frontend to visualize

Assumptions / Notes
-------------------
- Graph is directed or undirected CSR; weights are minutes for walk/bike.
- BG mapping uses per-BG node weights that sum to 1 (or less; we clamp α to [0,1]).
- Isochrones are computed ONLY for stations the user placed (snapped to nearest node).
- For visualization, we can return:
    a) station catchment polygons (convex hull approximation) OR
    b) station catchment node points (can be huge)
  Default is polygons (cheap).

Dependencies
------------
numpy, pandas, scipy, shapely
(optional) pyarrow for parquet, sklearn for KDTree fallback (we use scipy.spatial cKDTree)

You will need to adapt:
- load_graph_assets(...)
- load_bg_assets(...)
- load_replica_od(...)
to your file formats.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import math
import heapq
import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, mapping, Point


# ----------------------------
# Data containers
# ----------------------------

@dataclass(frozen=True)
class GraphCSR:
    """
    CSR graph representation.

    indptr, indices define adjacency.
    weights_* are aligned with indices (same length).
    node_xy are planar or lon/lat coords (used for snapping/visualization only).
    """
    indptr: np.ndarray         # shape (N+1,), int32/64
    indices: np.ndarray        # shape (M,), int32/64
    w_walk_min: np.ndarray     # shape (M,), float32/64 (edge travel time minutes)
    w_bike_min: np.ndarray     # shape (M,), float32/64
    node_xy: np.ndarray        # shape (N, 2), float64 (x,y) usually lon,lat
    directed: bool = True

    @property
    def N(self) -> int:
        return int(self.indptr.shape[0] - 1)


@dataclass(frozen=True)
class BlockGroups:
    """
    BG↔node mapping in a CSR-like format per block group:

    bg_indptr: offsets into bg_nodes/bg_weights for each BG index.
    bg_nodes: node indices belonging to BG i (or nearby nodes).
    bg_weights: weights per node for BG i (should sum to 1 per BG if possible).

    bg_ids: external BG IDs matching Replica OD.
    pop: population per BG (optional; used for accessibility counts).
    centroid_node: nearest graph node to BG centroid (used for station assignment).
    """
    bg_ids: np.ndarray              # shape (B,), dtype object or string
    pop: np.ndarray                 # shape (B,), float
    centroid_node: np.ndarray       # shape (B,), int
    bg_indptr: np.ndarray           # shape (B+1,), int
    bg_nodes: np.ndarray            # shape (K,), int
    bg_weights: np.ndarray          # shape (K,), float


@dataclass(frozen=True)
class ReplicaOD:
    """
    Replica OD table already aggregated to BG→BG.
    Required columns:
        o_bg, d_bg, trips, distance_km, primary_mode, start_min (optional)
    """
    df: pd.DataFrame


@dataclass(frozen=True)
class Assets:
    graph: GraphCSR
    bgs: BlockGroups
    od: ReplicaOD
    kdtree: cKDTree
    bg_index: Dict[Any, int]  # bg_id -> row index in bgs arrays


@dataclass(frozen=True)
class ScenarioParams:
    # Catchment thresholds in minutes
    walk_thresholds_min: Tuple[float, ...] = (5.0, 10.0, 15.0)
    bike_thresholds_min: Tuple[float, ...] = (5.0, 10.0, 15.0)

    # Ridership filters
    restrict_to_primary_mode: Optional[str] = "car"  # set None to disable

    # Distance weighting
    w_distance: Literal["exp_saturating", "step"] = "exp_saturating"
    lambda_km: float = 6.0  # for exp_saturating: w(d)=1-exp(-d/lambda)

    # Station assignment for diagnostics / station boardings
    assign_method: Literal["euclidean", "walk_to_centroid_node"] = "euclidean"
    assign_walk_cutoff_min: float = 30.0  # used only if assign_method = walk_to_centroid_node

    # Line travel time diagnostics (not used in adoption)
    cruise_kmh: float = 75.0
    dwell_sec: float = 10.0

    # Output controls for frontend
    return_catchment_nodes: bool = False   # can be huge; default false
    return_catchment_polygons: bool = True # cheap; convex hull approx
    polygon_min_points: int = 40           # require enough points to hull
    polygon_buffer_deg: float = 0.0005     # small buffer to make hull visible in lon/lat

    # Performance
    dijkstra_heap_reserve: int = 0  # no-op (kept for tuning hooks)

    # Access/egress behavior model
    beta_walk_per_min: float = 0.05   # exp(-beta * minutes)
    beta_bike_per_min: float = 0.14   # harsher than walk
    bike_base_multiplier: float = 0.45  # overall downweight of bike+transit relative to walk+transit

    # “bike–ride–bike is silly if rail leg is tiny” penalty
    brb_ratio_floor: float = 0.70     # need in-vehicle >= 0.70 * (access+egress) to avoid big penalty
    brb_ratio_power: float = 3.0      # higher = harsher penalty when ratio is small

    # Competition with direct biking
    direct_bike_kmh: float = 13.0
    bike_comp_scale_min: float = 10.0   # softness of logistic transition (minutes)
    bike_comp_bias_min: float = 1.0    # positive makes biking “more attractive” vs transit


# ----------------------------
# Loaders (YOU WILL EDIT THESE)
# ----------------------------

def load_graph_assets(path: str) -> GraphCSR:
    """
    Load graph CSR arrays from disk.

    Expected: you provide files containing indptr, indices, w_walk_min, w_bike_min, node_xy.

    Replace this with your own loader (npz/parquet/custom).
    """
    data = np.load(path, allow_pickle=False)
    return GraphCSR(
        indptr=data["indptr"],
        indices=data["indices"],
        w_walk_min=data["w_walk_min"],
        w_bike_min=data["w_bike_min"],
        node_xy=data["node_xy"],
        directed=bool(data.get("directed", True)),
    )


def load_bg_assets(path: str) -> BlockGroups:
    """
    Load BG arrays. Replace with your own loader.

    Required keys in npz:
      bg_ids, pop, centroid_node, bg_indptr, bg_nodes, bg_weights
    """
    data = np.load(path, allow_pickle=True)
    return BlockGroups(
        bg_ids=data["bg_ids"],
        pop=data["pop"].astype(float),
        centroid_node=data["centroid_node"].astype(int),
        bg_indptr=data["bg_indptr"].astype(int),
        bg_nodes=data["bg_nodes"].astype(int),
        bg_weights=data["bg_weights"].astype(float),
    )


def load_replica_od(path: str) -> ReplicaOD:
    """
    Load Replica OD aggregated table.

    You should pre-aggregate Replica to BG->BG and store as parquet/csv.
    Expected columns:
      o_bg, d_bg, trips, distance_km, primary_mode
    Optional:
      start_min (minutes since midnight) or similar for peak filtering later.
    """
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    required = {"o_bg", "d_bg", "trips", "distance_km", "primary_mode"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Replica OD missing columns: {sorted(missing)}")

    # Ensure dtypes
    df = df.copy()
    df["trips"] = df["trips"].astype(float)
    df["distance_km"] = df["distance_km"].astype(float)
    df["primary_mode"] = df["primary_mode"].astype(str)

    return ReplicaOD(df=df)


def load_assets(
    graph_npz_path: str,
    bg_npz_path: str,
    replica_od_path: str,
) -> Assets:
    """
    Call this once at app startup (cache it in Streamlit via st.cache_resource).
    """
    graph = load_graph_assets(graph_npz_path)
    bgs = load_bg_assets(bg_npz_path)
    od = load_replica_od(replica_od_path)

    kdtree = cKDTree(graph.node_xy)
    bg_index = {bg_id: i for i, bg_id in enumerate(bgs.bg_ids)}

    return Assets(graph=graph, bgs=bgs, od=od, kdtree=kdtree, bg_index=bg_index)


# ----------------------------
# Core algorithms
# ----------------------------

def min_dist_map_to_stations(
    graph: GraphCSR,
    station_nodes: np.ndarray,
    mode: Literal["walk", "bike"],
    cutoff_min: float,
) -> Dict[int, float]:
    """
    Returns dict: node -> min time-to-any-station within cutoff.
    Only includes nodes reached within cutoff, so memory stays bounded.
    """
    w = graph.w_walk_min if mode == "walk" else graph.w_bike_min
    best: Dict[int, float] = {}
    for s in station_nodes:
        nodes, dists = dijkstra_cutoff_csr(graph.indptr, graph.indices, w, int(s), float(cutoff_min))
        # dists aligned with nodes
        for n, d in zip(nodes, dists):
            n = int(n)
            d = float(d)
            prev = best.get(n)
            if prev is None or d < prev:
                best[n] = d
    return best


def compute_bg_access_alpha_and_time(
    bgs: BlockGroups,
    node2dist: Dict[int, float],
    cutoff_min: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each BG:
      alpha[i] = sum weights of bg_nodes with dist<=cutoff
      t_mean[i] = weighted mean dist among covered nodes (NaN if alpha=0)
    """
    B = len(bgs.bg_ids)
    alpha = np.zeros(B, dtype=np.float64)
    tmean = np.full(B, np.nan, dtype=np.float64)

    indptr = bgs.bg_indptr
    nodes = bgs.bg_nodes
    wts = bgs.bg_weights

    for i in range(B):
        a = 0.0
        td = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            n = int(nodes[p])
            wt = float(wts[p])
            d = node2dist.get(n)
            if d is not None and d <= cutoff_min:
                a += wt
                td += wt * float(d)

        if a > 0:
            alpha[i] = a
            tmean[i] = td / a
        else:
            alpha[i] = 0.0

    # clamp for safety
    np.clip(alpha, 0.0, 1.0, out=alpha)
    return alpha, tmean

def snap_points_to_nodes_xy(
    kdtree: cKDTree,
    node_xy: np.ndarray,
    points_xy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Snap user points to nearest graph node.
    Returns:
      node_idx: (K,) int
      snapped_xy: (K,2) float
    """
    d, idx = kdtree.query(points_xy, k=1)
    idx = idx.astype(int)
    return idx, node_xy[idx]


def dijkstra_cutoff_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    source: int,
    cutoff: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cutoff Dijkstra on CSR graph (non-negative weights).

    Returns:
      reached_nodes: int array of visited nodes within cutoff
      reached_dist: float array of their distances

    Implementation detail:
      - Uses a min-heap and stops when the smallest frontier distance > cutoff.
      - Stores distances only for visited nodes in dict, then materializes arrays.
    """
    # Fast path for cutoff <= 0
    if cutoff <= 0.0:
        return np.array([source], dtype=np.int32), np.array([0.0], dtype=np.float32)

    dist: Dict[int, float] = {int(source): 0.0}
    heap: List[Tuple[float, int]] = [(0.0, int(source))]

    while heap:
        d_u, u = heapq.heappop(heap)
        if d_u > cutoff:
            break
        # stale entry
        if d_u != dist.get(u, float("inf")):
            continue

        start, end = int(indptr[u]), int(indptr[u + 1])
        nbrs = indices[start:end]
        wts = weights[start:end]
        for v, w in zip(nbrs, wts):
            v = int(v)
            nd = d_u + float(w)
            if nd <= cutoff and nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    reached_nodes = np.fromiter(dist.keys(), dtype=np.int32, count=len(dist))
    reached_dist = np.fromiter(dist.values(), dtype=np.float32, count=len(dist))
    return reached_nodes, reached_dist


def union_node_sets(sets: Iterable[np.ndarray]) -> np.ndarray:
    """
    Union a list of node-index arrays into one unique array.
    """
    arrs = [s for s in sets if s is not None and len(s) > 0]
    if not arrs:
        return np.empty(0, dtype=np.int32)
    return np.unique(np.concatenate(arrs).astype(np.int32, copy=False))


def compute_station_catchments(
    graph: GraphCSR,
    station_nodes: np.ndarray,
    thresholds_min: Tuple[float, ...],
    mode: Literal["walk", "bike"],
) -> Dict[float, np.ndarray]:
    """
    For each threshold, returns unioned catchment nodes across all stations.
    Also returns per-station catchments for visualization in compute_line_metrics().
    """
    w = graph.w_walk_min if mode == "walk" else graph.w_bike_min

    union_by_t: Dict[float, List[np.ndarray]] = {t: [] for t in thresholds_min}
    for s in station_nodes:
        for t in thresholds_min:
            nodes, _ = dijkstra_cutoff_csr(graph.indptr, graph.indices, w, int(s), float(t))
            union_by_t[t].append(nodes)

    return {t: union_node_sets(union_by_t[t]) for t in thresholds_min}


def compute_bg_access_fractions(
    bgs: BlockGroups,
    covered_nodes: np.ndarray,
) -> np.ndarray:
    """
    α_i = fraction of BG i covered by the unioned catchment.

    Uses BG↔node mapping weights; each node contributes once.
    """
    covered = np.zeros(int(np.max(bgs.bg_nodes)) + 1, dtype=np.uint8)
    covered[covered_nodes] = 1

    B = bgs.bg_ids.shape[0]
    alpha = np.zeros(B, dtype=np.float32)

    for i in range(B):
        a, b = int(bgs.bg_indptr[i]), int(bgs.bg_indptr[i + 1])
        nodes = bgs.bg_nodes[a:b]
        wts = bgs.bg_weights[a:b]
        if nodes.size == 0:
            alpha[i] = 0.0
            continue
        alpha_i = float(np.sum(wts * covered[nodes]))
        # clamp
        if alpha_i < 0.0:
            alpha_i = 0.0
        elif alpha_i > 1.0:
            alpha_i = 1.0
        alpha[i] = alpha_i

    return alpha


def compute_distance_weight(
    d_km: np.ndarray,
    params: ScenarioParams,
) -> np.ndarray:
    """
    w(d) in [0,1]
    """
    if params.w_distance == "exp_saturating":
        lam = max(1e-6, float(params.lambda_km))
        return 1.0 - np.exp(-d_km / lam)
    else:
        # simple step rule (example)
        w = np.ones_like(d_km, dtype=np.float32)
        w[d_km < 2.0] = 0.2
        w[(d_km >= 2.0) & (d_km < 8.0)] = 0.8
        w[d_km >= 8.0] = 1.0
        return w.astype(np.float32)


def assign_bgs_to_stations(
    assets: Assets,
    station_nodes: np.ndarray,
    station_xy: np.ndarray,
    params: ScenarioParams,
) -> np.ndarray:
    """
    Returns station assignment per BG index: (B,) int in [0, K-1].

    Methods:
      - euclidean: assign by nearest station in xy space (cheap)
      - walk_to_centroid_node: assign by walk time from station to BG centroid node (more expensive)
    """
    B = assets.bgs.bg_ids.shape[0]
    K = station_nodes.shape[0]

    if K == 0:
        return -np.ones(B, dtype=np.int32)

    if params.assign_method == "euclidean":
        # BG centroid node coords -> nearest station by euclidean
        bg_xy = assets.graph.node_xy[assets.bgs.centroid_node]
        st_tree = cKDTree(station_xy)
        _, idx = st_tree.query(bg_xy, k=1)
        return idx.astype(np.int32)

    # walk-time assignment to centroid_node (more faithful)
    # Compute walk cutoff trees for each station up to assign_walk_cutoff_min,
    # then choose station with minimum dist to each BG centroid node (if reachable).
    cutoff = float(params.assign_walk_cutoff_min)
    w = assets.graph.w_walk_min

    centroid_nodes = assets.bgs.centroid_node.astype(int)
    best = np.full(B, np.inf, dtype=np.float32)
    best_k = np.full(B, -1, dtype=np.int32)

    for k, s in enumerate(station_nodes):
        nodes, dist = dijkstra_cutoff_csr(assets.graph.indptr, assets.graph.indices, w, int(s), cutoff)
        # map reached node -> dist (sparse)
        # Build dict for this station; for performance you can build a dense array if cutoff reaches many nodes.
        dmap = {int(n): float(d) for n, d in zip(nodes, dist)}
        for i in range(B):
            cn = int(centroid_nodes[i])
            dcn = dmap.get(cn, float("inf"))
            if dcn < best[i]:
                best[i] = dcn
                best_k[i] = k

    # fallback: if unreachable, use euclidean
    bad = best_k < 0
    if np.any(bad):
        bg_xy = assets.graph.node_xy[assets.bgs.centroid_node]
        st_tree = cKDTree(station_xy)
        _, idx = st_tree.query(bg_xy, k=1)
        best_k[bad] = idx.astype(np.int32)[bad]

    return best_k


def station_line_cumdist_km(station_xy: np.ndarray) -> np.ndarray:
    """
    Approx cumulative distance along station order (great-circle approximation for lon/lat).
    Returns cumdist[k] (km) from station 0 to k.
    """
    if station_xy.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)

    # Haversine in km
    def hav_km(lon1, lat1, lon2, lat2):
        R = 6371.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2)
        return 2 * R * math.asin(math.sqrt(a))

    K = station_xy.shape[0]
    cum = np.zeros(K, dtype=np.float32)
    for i in range(1, K):
        lon1, lat1 = float(station_xy[i - 1, 0]), float(station_xy[i - 1, 1])
        lon2, lat2 = float(station_xy[i, 0]), float(station_xy[i, 1])
        cum[i] = cum[i - 1] + float(hav_km(lon1, lat1, lon2, lat2))
    return cum


def line_in_vehicle_time_min(
    cumdist_km: np.ndarray,
    s_idx_o: np.ndarray,
    s_idx_d: np.ndarray,
    cruise_kmh: float,
    dwell_sec: float,
) -> np.ndarray:
    """
    Diagnostic in-vehicle time in minutes based on station order.
    """
    v = max(1e-6, float(cruise_kmh))
    dwell_min = float(dwell_sec) / 60.0
    dist_km = np.abs(cumdist_km[s_idx_d] - cumdist_km[s_idx_o])
    n_stops = np.abs(s_idx_d - s_idx_o).astype(np.float32)
    return (dist_km / v) * 60.0 + n_stops * dwell_min


# ----------------------------
# Visualization helpers
# ----------------------------

def catchment_polygon_geojson(
    node_xy: np.ndarray,
    catch_nodes: np.ndarray,
    buffer_deg: float,
    min_points: int,
) -> Optional[Dict[str, Any]]:
    """
    Cheap polygon approximation for catchment visualization:
      - takes convex hull of reached node points
      - adds a small buffer so it draws nicely
    Returns GeoJSON geometry or None if insufficient points.
    """
    if catch_nodes.size < min_points:
        return None
    pts = node_xy[catch_nodes]
    mp = MultiPoint([Point(float(x), float(y)) for x, y in pts])
    hull = mp.convex_hull
    if buffer_deg > 0:
        hull = hull.buffer(buffer_deg)
    return mapping(hull)


# ----------------------------
# Main interactive entrypoint
# ----------------------------

def compute_line_metrics(
    assets: Assets,
    station_points_xy: np.ndarray,
    params: ScenarioParams = ScenarioParams(),
) -> Dict[str, Any]:
    """
    Main function to call when the user clicks "Compute new line metrics".

    Inputs:
      station_points_xy: (K,2) array of user-placed station coordinates (lon,lat or x,y).
    Output:
      JSON-serializable dict with:
        - snapped stations
        - catchment layers (polygons or nodes)
        - BG access fractions α
        - accessibility totals
        - ridership totals + station boardings + diagnostics
    """
    station_points_xy = np.asarray(station_points_xy, dtype=np.float64)
    if station_points_xy.ndim != 2 or station_points_xy.shape[1] != 2:
        raise ValueError("station_points_xy must be shape (K,2)")

    # 1) Snap stations to graph nodes
    station_nodes, snapped_xy = snap_points_to_nodes_xy(assets.kdtree, assets.graph.node_xy, station_points_xy)
    K = int(station_nodes.shape[0])

    # Early exit: no stations
    if K == 0:
        return {
            "stations": [],
            "accessibility": {},
            "ridership": {"total_riders": 0.0},
            "diagnostics": {"note": "No stations provided."},
            "layers": {},
        }

    # 2) Compute per-station catchments and unioned catchments for walk/bike thresholds
    #    We keep both:
    #      - per-station (for map)
    #      - unioned (for α and totals)
    graph = assets.graph
    bgs = assets.bgs

    per_station = {
        "walk": {float(t): [] for t in params.walk_thresholds_min},
        "bike": {float(t): [] for t in params.bike_thresholds_min},
    }
    unioned = {
        "walk": {},
        "bike": {},
    }

    # Walk
    for s in station_nodes:
        for t in params.walk_thresholds_min:
            nodes, _ = dijkstra_cutoff_csr(graph.indptr, graph.indices, graph.w_walk_min, int(s), float(t))
            per_station["walk"][float(t)].append(nodes)
    unioned["walk"] = {float(t): union_node_sets(per_station["walk"][float(t)]) for t in params.walk_thresholds_min}

    # Bike
    for s in station_nodes:
        for t in params.bike_thresholds_min:
            nodes, _ = dijkstra_cutoff_csr(graph.indptr, graph.indices, graph.w_bike_min, int(s), float(t))
            per_station["bike"][float(t)].append(nodes)
    unioned["bike"] = {float(t): union_node_sets(per_station["bike"][float(t)]) for t in params.bike_thresholds_min}

    # 3) Compute BG access fractions α for a selected definition of "accessible"
    #    For simplicity, define α using the *largest walk threshold* OR *largest bike threshold* unioned together.
    #    You can change this rule to: walk-only, bike-only, or max(walk,bike) per BG.
    t_walk_max = float(max(params.walk_thresholds_min))
    t_bike_max = float(max(params.bike_thresholds_min))

    covered_nodes = union_node_sets([unioned["walk"][t_walk_max], unioned["bike"][t_bike_max]])
    alpha = compute_bg_access_fractions(bgs, covered_nodes)  # shape (B,)

    # --- NEW: compute separate walk/bike access factors for ridership (not unioned) ---
    walk_node2dist = min_dist_map_to_stations(graph, station_nodes, "walk", t_walk_max)
    bike_node2dist = min_dist_map_to_stations(graph, station_nodes, "bike", t_bike_max)

    alpha_walk, t_walk = compute_bg_access_alpha_and_time(bgs, walk_node2dist, t_walk_max)
    alpha_bike, t_bike = compute_bg_access_alpha_and_time(bgs, bike_node2dist, t_bike_max)

    # Access/egress factors per BG (minutes-based decay)
    walk_factor = alpha_walk * np.exp(-params.beta_walk_per_min * np.nan_to_num(t_walk, nan=1e9))

    bike_factor = (
        params.bike_base_multiplier
        * alpha_bike
        * np.exp(-params.beta_bike_per_min * np.nan_to_num(t_bike, nan=1e9))
    )

    # Combined “effective” access factor used for ridership
    A = walk_factor + bike_factor  # shape (B,)

    # Bike dominance share (0..1), used to intensify bike–ride–bike penalty
    bike_dom = np.divide(bike_factor, A, out=np.zeros_like(A), where=(A > 0))

    # Accessibility totals (population served within each threshold, unioned across stations)
    pop = bgs.pop.astype(np.float64, copy=False)

    def pop_served(covered_nodes_t: np.ndarray) -> float:
        # Approximate: sum_i pop_i * α_i(t), where α_i(t) computed for this threshold union.
        # For speed, recompute α for each threshold union. B is usually manageable; if not, cache per threshold.
        a_t = compute_bg_access_fractions(bgs, covered_nodes_t).astype(np.float64)
        return float(np.sum(pop * a_t))

    accessibility = {
        "walk": {str(t): pop_served(unioned["walk"][float(t)]) for t in params.walk_thresholds_min},
        "bike": {str(t): pop_served(unioned["bike"][float(t)]) for t in params.bike_thresholds_min},
        "alpha_definition": f"alpha = union(walk<= {t_walk_max} min, bike<= {t_bike_max} min)",
    }

    # 4) Station assignment (for station-level boardings/alightings + diagnostics)
    bg_station = assign_bgs_to_stations(assets, station_nodes, snapped_xy, params)  # (B,)

    # 5) Ridership estimation on OD table
    od = assets.od.df

    # Filter OD rows to BGs we know about
    # Map bg_id -> index, drop unknowns
    o_idx = od["o_bg"].map(assets.bg_index)
    d_idx = od["d_bg"].map(assets.bg_index)
    mask_known = o_idx.notna() & d_idx.notna()

    od2 = od.loc[mask_known, ["o_bg", "d_bg", "trips", "distance_km", "primary_mode"]].copy()
    o_i = o_idx[mask_known].astype(int).to_numpy()
    d_i = d_idx[mask_known].astype(int).to_numpy()

    CARLIKE = {
        "car",
        "private_auto",
        "auto_driver",
        "auto_passenger",
        "rideshare",
        "taxi",
    }

    # Optional mode filter

    if params.restrict_to_primary_mode is not None:
        m = str(params.restrict_to_primary_mode).lower()
        if m == "car":
            od2 = od2[od2["primary_mode"].str.lower().isin(CARLIKE)]
        else:
            od2 = od2[od2["primary_mode"].str.lower() == m]
        # Rebuild o_i/d_i aligned to od2
        # (cheap approach: remap from original using keep mask)
        # We'll just rebuild from scratch:
        o_i = od2["o_bg"].map(assets.bg_index).astype(int).to_numpy()
        d_i = od2["d_bg"].map(assets.bg_index).astype(int).to_numpy()

    trips = od2["trips"].to_numpy(dtype=np.float64)
    dist_km = od2["distance_km"].to_numpy(dtype=np.float32)

    w_d = compute_distance_weight(dist_km, params).astype(np.float64)

    # Combined access on each end (walk + downweighted bike)
    Ao = A[o_i].astype(np.float64)
    Ad = A[d_i].astype(np.float64)

    # Station indices for each OD (needed for in-vehicle time)
    o_station = bg_station[o_i]
    d_station = bg_station[d_i]

    # In-vehicle time proxy from station order distances
    cumdist = station_line_cumdist_km(snapped_xy)
    o_st = np.clip(o_station, 0, K - 1)
    d_st = np.clip(d_station, 0, K - 1)
    t_iv = line_in_vehicle_time_min(cumdist, o_st, d_st, params.cruise_kmh, params.dwell_sec).astype(np.float64)

    # Expected access+egress time (blend walk/bike by their factor weights)
    t_walk_o = np.nan_to_num(t_walk[o_i], nan=1e9)
    t_bike_o = np.nan_to_num(t_bike[o_i], nan=1e9)
    t_walk_d = np.nan_to_num(t_walk[d_i], nan=1e9)
    t_bike_d = np.nan_to_num(t_bike[d_i], nan=1e9)

    wf_o = walk_factor[o_i].astype(np.float64)
    bf_o = bike_factor[o_i].astype(np.float64)
    wf_d = walk_factor[d_i].astype(np.float64)
    bf_d = bike_factor[d_i].astype(np.float64)

    to = np.divide(
        wf_o * t_walk_o + bf_o * t_bike_o,
        wf_o + bf_o,
        out=np.full_like(t_walk_o, 1e9, dtype=np.float64),
        where=((wf_o + bf_o) > 0),
    )
    td = np.divide(
        wf_d * t_walk_d + bf_d * t_bike_d,
        wf_d + bf_d,
        out=np.full_like(t_walk_d, 1e9, dtype=np.float64),
        where=((wf_d + bf_d) > 0),
    )

    t_access = to + td  # minutes

    # --- Penalty 1: bike–ride–bike stupidity penalty (only when both ends are bike-dominant) ---
    bike_dom_o = bike_dom[o_i].astype(np.float64)
    bike_dom_d = bike_dom[d_i].astype(np.float64)
    both_bikeish = bike_dom_o * bike_dom_d  # 0..1 intensity

    ratio = np.divide(t_iv, t_access, out=np.zeros_like(t_iv), where=(t_access > 0))
    brb = np.ones_like(t_iv)
    mask = ratio < params.brb_ratio_floor
    brb[mask] = (np.maximum(ratio[mask], 1e-6) / params.brb_ratio_floor) ** params.brb_ratio_power
    brb = (1.0 - both_bikeish) + both_bikeish * brb

    # --- Penalty 2: compete with direct biking ---
    t_bike_direct = (dist_km.astype(np.float64) / params.direct_bike_kmh) * 60.0
    t_transit = t_access + t_iv

    z = (t_bike_direct - t_transit - params.bike_comp_bias_min) / params.bike_comp_scale_min
    p_choose_transit = 1.0 / (1.0 + np.exp(-z))

    share = Ao * Ad * w_d * brb * p_choose_transit
    riders = trips * share
    total_riders = float(np.sum(riders))

    boardings = np.zeros(K, dtype=np.float64)
    alightings = np.zeros(K, dtype=np.float64)

    # accumulate
    for k in range(K):
        boardings[k] = float(np.sum(riders[o_station == k]))
        alightings[k] = float(np.sum(riders[d_station == k]))

    # 6) Diagnostics: α distribution, w(d), in-vehicle time (optional)
    diag: Dict[str, Any] = {
        "alpha_mean": float(np.mean(alpha)),
        "alpha_p50": float(np.quantile(alpha, 0.50)),
        "alpha_p90": float(np.quantile(alpha, 0.90)),
        "w_d_mean": float(np.mean(w_d)) if len(w_d) else 0.0,
        "od_rows_used": int(len(od2)),
        "A_mean": float(np.mean(A)),
        "walk_factor_mean": float(np.mean(walk_factor)),
        "bike_factor_mean": float(np.mean(bike_factor)),
        "bike_dom_mean": float(np.mean(bike_dom)),
    }

    # In-vehicle time diagnostics (based on station order distance)
    cumdist = station_line_cumdist_km(snapped_xy)
    o_st = np.clip(o_station, 0, K - 1)
    d_st = np.clip(d_station, 0, K - 1)
    tiv = line_in_vehicle_time_min(cumdist, o_st, d_st, params.cruise_kmh, params.dwell_sec)
    if tiv.size:
        diag.update({
            "in_vehicle_time_min_mean": float(np.mean(tiv)),
            "in_vehicle_time_min_p90": float(np.quantile(tiv, 0.90)),
        })

    # 7) Build layers for frontend
    layers: Dict[str, Any] = {"catchments": {"walk": {}, "bike": {}}}

    # Per-station catchment visualization
    # NOTE: returning nodes can be huge; polygons are cheap but approximate.
    if params.return_catchment_polygons:
        for mode in ("walk", "bike"):
            thresholds = params.walk_thresholds_min if mode == "walk" else params.bike_thresholds_min
            for t in thresholds:
                t = float(t)
                polys = []
                for nodes in per_station[mode][t]:
                    gj = catchment_polygon_geojson(
                        graph.node_xy, nodes,
                        buffer_deg=float(params.polygon_buffer_deg),
                        min_points=int(params.polygon_min_points),
                    )
                    polys.append(gj)
                layers["catchments"][mode][str(t)] = {
                    "type": "FeatureCollection",
                    "features": [
                        {"type": "Feature", "properties": {"station_index": i, "threshold_min": t, "mode": mode},
                         "geometry": polys[i]}
                        for i in range(K) if polys[i] is not None
                    ],
                }

    if params.return_catchment_nodes:
        # WARNING: large payloads; use only for debugging or tiny graphs.
        for mode in ("walk", "bike"):
            thresholds = params.walk_thresholds_min if mode == "walk" else params.bike_thresholds_min
            for t in thresholds:
                t = float(t)
                feats = []
                for si, nodes in enumerate(per_station[mode][t]):
                    pts = graph.node_xy[nodes]
                    for x, y in pts:
                        feats.append({
                            "type": "Feature",
                            "properties": {"station_index": int(si), "threshold_min": t, "mode": mode},
                            "geometry": {"type": "Point", "coordinates": [float(x), float(y)]},
                        })
                layers["catchments"][mode][str(t)] = {"type": "FeatureCollection", "features": feats}

    # Frontend-friendly stations
    stations_out = [
        {
            "station_index": int(i),
            "input_xy": [float(station_points_xy[i, 0]), float(station_points_xy[i, 1])],
            "snapped_xy": [float(snapped_xy[i, 0]), float(snapped_xy[i, 1])],
            "snapped_node": int(station_nodes[i]),
        }
        for i in range(K)
    ]

    # 8) Package result
    result = {
        "stations": stations_out,
        "accessibility": accessibility,
        "ridership": {
            "total_riders": total_riders,
            "boardings_by_station": boardings.tolist(),
            "alightings_by_station": alightings.tolist(),
            "params": {
                "restrict_to_primary_mode": params.restrict_to_primary_mode,
                "w_distance": params.w_distance,
                "lambda_km": params.lambda_km,
                "alpha_definition": accessibility["alpha_definition"],
            },
        },
        "diagnostics": diag,
        "layers": layers,
    }

    return result


# ----------------------------
# Optional: convenience for Streamlit caching
# ----------------------------

def warm_start(
    graph_npz_path: str,
    bg_npz_path: str,
    replica_od_path: str,
) -> Assets:
    """
    Convenience wrapper you call once from app.py.

    In Streamlit:
        @st.cache_resource
        def get_assets():
            return warm_start(...)

    Then on button click:
        out = compute_line_metrics(assets, station_xy, params)
    """
    return load_assets(graph_npz_path, bg_npz_path, replica_od_path)

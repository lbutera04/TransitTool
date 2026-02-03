from __future__ import annotations

import streamlit.components.v1 as components
from typing import Any, Dict, List, Optional, Tuple

# During dev, you’ll point this to the frontend dev server
# During release, you’ll point it to the built frontend "dist" directory
_RELEASE = False

if not _RELEASE:
    _component = components.declare_component(
        "transit_map",
        url="http://localhost:3001",  # dev server port (we'll set this below)
    )
else:
    import os
    _component = components.declare_component(
        "transit_map",
        path=os.path.join(os.path.dirname(__file__), "frontend", "dist"),
    )


def transit_map(
    *,
    stations: List[Tuple[float, float]],
    stations_version: int = 0,
    center: Tuple[float, float],
    zoom: int,
    click_to_add: bool,
    geojson_layers: Optional[List[Dict[str, Any]]] = None,
    height: int = 650,
    station_tooltips=None,   # <-- ADD (python-style)
    key: str = "transit_map",
) -> Dict[str, Any]:
    """
    Returns dict with:
      - stations: [[lat, lon], ...]
      - center:   [lat, lon]
      - zoom:     int
    """
    res = _component(
        initialStations=[list(x) for x in stations],
        stationsVersion=int(stations_version),
        initialCenter=[float(center[0]), float(center[1])],
        initialZoom=int(zoom),
        clickToAdd=bool(click_to_add),
        geojsonLayers=geojson_layers or [],
        height=int(height),
        stationTooltips=station_tooltips or [],  # <-- FORWARD (JS-style camelCase)
        key=key,
        default={"stations": [list(x) for x in stations], "center": list(center), "zoom": int(zoom)},
    )
    return res or {"stations": [list(x) for x in stations], "center": list(center), "zoom": int(zoom)}

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { MapContainer, TileLayer, Marker, Polyline, GeoJSON, useMapEvents } from "react-leaflet";
import L from "leaflet";
import { Streamlit } from "streamlit-component-lib";

// ---- Leaflet default marker icons via CDN (avoid bundler asset issues) ----
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
    iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
    shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

// ---------- helpers ----------
function clampZoom(z) {
    const zi = Number.isFinite(z) ? Math.round(z) : 12;
    return Math.max(1, Math.min(22, zi));
}

function normalizeStations(stations) {
    if (!Array.isArray(stations)) return [];
    return stations
        .map((p) => {
            if (!p || p.length < 2) return null;
            const lat = parseFloat(p[0]);
            const lon = parseFloat(p[1]);
            if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
            return [lat, lon];
        })
        .filter(Boolean);
}

function stationsEqual(a, b) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
        if (Math.abs(a[i][0] - b[i][0]) > 1e-10) return false;
        if (Math.abs(a[i][1] - b[i][1]) > 1e-10) return false;
    }
    return true;
}

// Captures clicks + view changes
function MapEvents({ onMapClick, onViewChange }) {
    const map = useMapEvents({
        click(e) {
            onMapClick?.(e.latlng);
        },
        moveend() {
            const c = map.getCenter();
            onViewChange?.({ center: [c.lat, c.lng], zoom: map.getZoom() });
        },
        zoomend() {
            const c = map.getCenter();
            onViewChange?.({ center: [c.lat, c.lng], zoom: map.getZoom() });
        },
    });
    return null;
}

export default function TransitMap(props) {
    const {
        initialStations = [],
        initialCenter = [37.7749, -122.4194],
        initialZoom = 12,
        clickToAdd = true,
        geojsonLayers = [],
        height = 650,
    } = props;

    // ---- derived initial values from props ----
    const initStations = useMemo(() => normalizeStations(initialStations), [initialStations]);
    const initCenter = useMemo(() => {
        const c = Array.isArray(initialCenter) && initialCenter.length === 2 ? initialCenter : [37.7749, -122.4194];
        return [parseFloat(c[0]), parseFloat(c[1])];
    }, [initialCenter]);
    const initZoom = useMemo(() => clampZoom(initialZoom), [initialZoom]);

    // ---- local state ----
    const [stations, setStations] = useState(initStations);
    const [center, setCenter] = useState(initCenter);
    const [zoom, setZoom] = useState(initZoom);

    // ---- refs to avoid stale closures + echo loops ----
    const stationsRef = useRef(stations);
    useEffect(() => {
        stationsRef.current = stations;
    }, [stations]);

    const lastSentStationsRef = useRef(initStations);

    // ---- Streamlit ready + sizing ----
    useEffect(() => {
        Streamlit.setComponentReady();
    }, []);

    useEffect(() => {
        Streamlit.setFrameHeight(height || 650);
    }, [height]);

    // ---- emit value to Streamlit (authoritative bridge) ----
    const emit = useCallback((nextStations, nextCenter, nextZoom) => {
        lastSentStationsRef.current = nextStations;
        Streamlit.setComponentValue({
            stations: nextStations,
            center: nextCenter,
            zoom: nextZoom,
        });
    }, []);

    // ---- guard against Streamlit rerun echo overwriting local state ----
    useEffect(() => {
        const incoming = normalizeStations(initialStations);

        // Only adopt incoming stations if they're not merely an echo of what we just sent.
        if (!stationsEqual(incoming, lastSentStationsRef.current) && !stationsEqual(incoming, stations)) {
            setStations(incoming);
        }
    }, [initialStations, stations]);

    // Center/zoom can safely follow props (they don't affect station list logic)
    useEffect(() => {
        const c = Array.isArray(initialCenter) && initialCenter.length === 2 ? initialCenter : initCenter;
        const nextCenter = [parseFloat(c[0]), parseFloat(c[1])];
        if (Number.isFinite(nextCenter[0]) && Number.isFinite(nextCenter[1])) setCenter(nextCenter);
    }, [initialCenter, initCenter]);

    useEffect(() => {
        setZoom(clampZoom(initialZoom));
    }, [initialZoom]);

    // ---- event handlers ----
    const onMapClick = useCallback(
        (latlng) => {
            if (!clickToAdd) return;

            const lat = latlng.lat;
            const lon = latlng.lng;
            if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;

            setStations((prev) => {
                const next = [...prev, [lat, lon]];
                emit(next, center, zoom);
                return next;
            });
        },
        [clickToAdd, emit, center, zoom]
    );

    const onViewChange = useCallback(
        ({ center: c, zoom: z }) => {
            const nextCenter = Array.isArray(c) && c.length === 2 ? [c[0], c[1]] : center;
            const nextZoom = Number.isFinite(z) ? clampZoom(z) : zoom;

            setCenter(nextCenter);
            setZoom(nextZoom);

            emit(stationsRef.current, nextCenter, nextZoom);
        },
        [emit, center, zoom]
    );

    const onMarkerDragEnd = useCallback(
        (idx, e) => {
            const ll = e.target.getLatLng();
            const lat = ll.lat;
            const lon = ll.lng;

            setStations((prev) => {
                const next = prev.slice();
                next[idx] = [lat, lon];
                emit(next, center, zoom);
                return next;
            });
        },
        [emit, center, zoom]
    );

    const onMarkerRightClick = useCallback(
        (idx) => {
            setStations((prev) => {
                const next = prev.filter((_, i) => i !== idx);
                emit(next, center, zoom);
                return next;
            });
        },
        [emit, center, zoom]
    );

    // ---- derived render data ----
    const polylinePositions = useMemo(() => stations.map(([lat, lon]) => [lat, lon]), [stations]);

    const layers = useMemo(() => {
        if (!Array.isArray(geojsonLayers)) return [];
        return geojsonLayers.filter((l) => l && l.geojson && l.geojson.features && l.geojson.features.length > 0);
    }, [geojsonLayers]);

    return (
        <div style={{ width: "100%", height: height || 650 }}>
            <MapContainer center={center} zoom={zoom} style={{ width: "100%", height: "100%" }}>
                <TileLayer
                    attribution='&copy; OpenStreetMap contributors &copy; CARTO'
                    url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
                />

                <MapEvents onMapClick={onMapClick} onViewChange={onViewChange} />

                {layers.map((l, i) => (
                    <GeoJSON
                        key={`${l.name || "layer"}-${i}`}
                        data={l.geojson}
                        style={() => ({
                            color: (l.style && l.style.color) || "#3388ff",
                            weight: (l.style && l.style.weight) || 2,
                            fillOpacity: (l.style && l.style.fillOpacity) ?? 0.12,
                        })}
                    />
                ))}

                {stations.map(([lat, lon], idx) => (
                    <Marker
                        key={`st-${idx}`}
                        position={[lat, lon]}
                        draggable={true}
                        eventHandlers={{
                            dragend: (e) => onMarkerDragEnd(idx, e),
                            contextmenu: () => onMarkerRightClick(idx),
                        }}
                    />
                ))}

                {stations.length >= 2 && (
                    <Polyline positions={polylinePositions} pathOptions={{ color: "black", weight: 4, opacity: 0.5 }} />
                )}
            </MapContainer>
        </div>
    );
}

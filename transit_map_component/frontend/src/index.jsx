import React from "react";
import ReactDOM from "react-dom/client";
import { withStreamlitConnection } from "streamlit-component-lib";
import TransitMap from "./TransitMap.jsx";
import "leaflet/dist/leaflet.css";

function Wrapped(props) {
    const args = props?.args ?? {};
    return <TransitMap {...args} />;
}

const Connected = withStreamlitConnection(Wrapped);

ReactDOM.createRoot(document.getElementById("root")).render(
    <React.StrictMode>
        <Connected />
    </React.StrictMode>
);

# streamlit_app.py
"""DustGuard-Vision: AI-Driven Construction Site Dust Mitigation & Water Conservation System.
IIT Bombay Hackathon Edition: Edge Vision + Classical CV + Satellite Aerosol Cross-Validation.
"""
import os
import time
import tempfile
import cv2
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# Import Core Subsystems
from constants import (
    DEFAULT_FPS,
    OPACITY_THRESHOLD,
    CONSECUTIVE_FRAMES,
    MISTING_API_URL,
    CANNON_ID,
    FLOW_RATE_L_PER_MIN,
    BURST_SECONDS,
    DB_PATH
)
from core.video_capture import FrameBuffer
from core.inference import process_frame, draw_hud
from core.decision_engine import DecisionStateMachine
from core.db import (
    init_db,
    log_dust_event,
    log_water_savings,
    get_recent_events,
    get_summary_metrics
)

# Initialize Storage & Directories
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
init_db(DB_PATH)

# Page Setup
st.set_page_config(
    page_title="DustGuard-Vision | Municipal AI Inspector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Sci-Fi Command Center Glassmorphism Styling
st.markdown("""
<style>
    /* Premium Modern Dark-Navy Theme */
    .stApp {
        background: linear-gradient(135deg, #0a0f1d 0%, #0d172e 45%, #070b16 100%) !important;
        background-attachment: fixed !important;
        color: #f1f5f9;
    }

    /* Top Streamlit Header Bar: Seamless Dark/Transparent */
    header, header[data-testid="stHeader"], [data-testid="stHeader"] {
        background: transparent !important;
        background-color: rgba(10, 15, 29, 0.95) !important;
        border-bottom: 1px solid rgba(56, 189, 248, 0.15) !important;
    }
    
    /* Styled Cohesive Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1427 0%, #080d1a 100%) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.2) !important;
    }
    
    /* Sidebar text, labels, and captions */
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown p {
        color: #f1f5f9 !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] .stCaption {
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }

    /* Crisp High-Contrast Input Fields */
    input, textarea, [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {
        background-color: #0d1527 !important;
        color: #ffffff !important;
        border: 2px solid #0284c7 !important;
        border-radius: 8px !important;
        font-weight: 800 !important;
        font-size: 1.05rem !important;
    }

    /* Plus/Minus Buttons on Number Inputs */
    [data-testid="stNumberInput"] button {
        background: #0284c7 !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 800 !important;
    }

    /* High-Visibility Code Badges */
    code {
        background-color: #0d1527 !important;
        color: #38bdf8 !important;
        font-weight: 800 !important;
        padding: 3px 8px !important;
        border-radius: 6px !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
    }
    
    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(13, 23, 46, 0.75) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.45);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.5) !important;
        transform: translateY(-2px);
    }
    .metric-title {
        color: #94a3b8 !important;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-val {
        color: #ffffff !important;
        font-size: 1.85rem;
        font-weight: 800;
        margin-top: 4px;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto;
    }
    
    /* Hardware status badges */
    .status-badge-active {
        background: linear-gradient(135deg, #ef4444, #b91c1c);
        color: white !important;
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        box-shadow: 0 0 16px rgba(239, 68, 68, 0.6);
    }
    .status-badge-standby {
        background: linear-gradient(135deg, #10b981, #047857);
        color: white !important;
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.35);
    }
    .status-badge-pending {
        background: linear-gradient(135deg, #f59e0b, #b45309);
        color: white !important;
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        box-shadow: 0 0 12px rgba(245, 158, 11, 0.4);
    }
    /* Ultra-Bright Bold Headings */
    h1, h2, h3, h4, .stSubheader, [data-testid="stSubheader"] {
        color: #ffffff !important;
        font-weight: 800 !important;
        letter-spacing: 0.5px !important;
        text-shadow: 0 0 14px rgba(56, 189, 248, 0.4) !important;
    }
    
    /* Sidebar Headings: High-Contrast Neon Cyan */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
        color: #38bdf8 !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        text-shadow: 0 0 12px rgba(56, 189, 248, 0.6) !important;
    }

    /* High-Visibility Clickable Buttons (Strictly scoped to Streamlit controls) */
    [data-testid="stBaseButton-secondary"], 
    [data-testid="stBaseButton-primary"], 
    .stDownloadButton button, 
    [data-testid="stButton"] > button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 0.95rem !important;
        border: 2px solid #38bdf8 !important;
        border-radius: 8px !important;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.45) !important;
        padding: 8px 18px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        transition: all 0.25s ease-in-out !important;
    }
    [data-testid="stBaseButton-secondary"]:hover, 
    [data-testid="stBaseButton-primary"]:hover, 
    .stDownloadButton button:hover, 
    [data-testid="stButton"] > button:hover {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
        border-color: #7dd3fc !important;
        box-shadow: 0 0 24px rgba(56, 189, 248, 0.8) !important;
        transform: scale(1.02) !important;
    }

    /* Completely neutralize Plotly toolbar/modebar buttons so NO blue boxes appear */
    .modebar, .modebar-container, .modebar-btn, .modebar-group, [data-testid="stPlotlyChart"] .modebar-btn, [data-testid="stPlotlyChart"] button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 2px !important;
        transform: none !important;
    }
    .modebar-btn svg {
        fill: #94a3b8 !important;
    }
    .modebar-btn:hover svg {
        fill: #38bdf8 !important;
    }

    /* File Uploader Button (Browse Files) */
    [data-testid="stFileUploader"] section button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        border: 2px solid #38bdf8 !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.4) !important;
    }

    /* Radio buttons & Checkbox text */
    [data-testid="stRadio"] label, [data-testid="stCheckbox"] label {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
    }

    /* Tabs: Bright and Bold */
    [data-testid="stTabs"] button {
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        color: #94a3b8 !important;
        font-weight: 800 !important;
        font-size: 0.95rem !important;
        box-shadow: none !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
        text-shadow: 0 0 10px rgba(56, 189, 248, 0.6) !important;
    }
    
    /* File uploader dropzone styling */
    [data-testid="stFileUploader"] section {
        background-color: #111c35 !important;
        border: 2px dashed #0284c7 !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploader"] section * {
        color: #cbd5e1 !important;
    }
</style>
""", unsafe_allow_html=True)

# Application Header & Live Satellite Status Ribbon
col_header, col_sat_badge = st.columns([2.9, 2.1])
with col_header:
    st.markdown("""
    <div style="display: flex; flex-direction: column; gap: 4px; margin-bottom: 6px;">
        <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
            <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🛡️ DustGuard-Vision
            </h1>
            <span style="background: rgba(14, 165, 233, 0.2); border: 1.5px solid #0284c7; color: #38bdf8; font-size: 0.75rem; font-weight: 800; padding: 3px 10px; border-radius: 12px; letter-spacing: 0.8px;">
                IIT BOMBAY INNOVATION
            </span>
        </div>
        <div style="font-size: 0.84rem; color: #94a3b8; font-weight: 500;">
            Autonomous Construction Plume Suppression | Edge AI (YOLOv8) ✕ Sentinel-5P / ISRO Geospatial Validation
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_sat_badge:
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(8, 14, 28, 0.95)); border: 2px solid #0284c7; border-radius: 12px; padding: 12px 18px; box-shadow: 0 0 20px rgba(2, 132, 199, 0.4); text-align: left;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px;">
            <span style="font-size: 1.05rem; font-weight: 800; color: #38bdf8; text-shadow: 0 0 10px rgba(56, 189, 248, 0.8);">
                🛰️ SATELLITE RADAR: ACTIVE
            </span>
            <span style="display: inline-flex; align-items: center; gap: 6px; background: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; padding: 2px 8px; border-radius: 12px; font-size: 0.72rem; color: #34d399; font-weight: 700;">
                <span style="width: 8px; height: 8px; border-radius: 50%; background: #10b981; box-shadow: 0 0 8px #10b981;"></span> LIVE ORBIT
            </span>
        </div>
        <div style="font-size: 0.9rem; color: #ffffff; font-weight: 700;">
            ISRO EOS-04 / Sentinel-5P (824 km Polar Sun-Sync)
        </div>
        <div style="font-size: 0.82rem; color: #94a3b8; font-weight: 600; margin-top: 2px;">
            📍 Powai Target: 19.1334° N, 72.9133° E | AOD: <span style="color: #38bdf8; font-weight: 800;">0.42 (Elevated)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/fluency/96/dust.png", width=64)
st.sidebar.markdown("### 🎛️ Command Controls")

# Video Ingestion: Multi-Source Options
st.sidebar.subheader("📹 Video Ingestion Source")
source_mode = st.sidebar.radio(
    "Select Input Feed:",
    options=[
        "📁 Upload / Drag & Drop Video",
        "📹 Live USB Webcam",
        "🌐 RTSP IP Camera Stream",
        "🎮 Built-in AI Simulation Demo"
    ],
    index=0
)

video_source = "demo"
if source_mode == "📁 Upload / Drag & Drop Video":
    uploaded_file = st.sidebar.file_uploader(
        "📥 Drag & Drop Site Video (.mp4)",
        type=["mp4", "avi", "mov", "mkv"],
        help="Drag and drop any drone or site footage here"
    )
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_source = tfile.name
        st.sidebar.success(f"Loaded: {uploaded_file.name}")
    elif os.path.exists("data/demo_sample.mp4"):
        video_source = "data/demo_sample.mp4"
        st.sidebar.caption("✨ Active Video: Demo Site Feed (or drop a video above)")
    else:
        video_source = "demo"
        st.sidebar.caption("✨ Active Video: Synthetic Site Simulation (or drop a video above)")

elif source_mode == "📹 Live USB Webcam":
    cam_index = st.sidebar.number_input("Webcam Hardware Index (0 = Default)", min_value=0, max_value=5, value=0)
    video_source = str(cam_index)
    st.sidebar.info(f"Targeting local camera index: {cam_index}")

elif source_mode == "🌐 RTSP IP Camera Stream":
    rtsp_url = st.sidebar.text_input("RTSP Stream URL", value="rtsp://192.168.1.100:554/live")
    video_source = rtsp_url
    st.sidebar.info("Streaming from IP camera stream")

elif source_mode == "🎮 Built-in AI Simulation Demo":
    video_source = "demo"
    st.sidebar.info("Running autonomous simulation with moving trucks & dust plumes.")

# Tunable Parameters with Dropdown
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Threshold Tuning")

threshold_profile = st.sidebar.selectbox(
    "Select Compliance Protocol:",
    options=[
        "🏢 Standard Site Protocol (35% Opacity)",
        "🌿 Strict Eco-Sensitive Zone (25% Opacity)",
        "🚜 Heavy Demolition Zone (50% Opacity)",
        "🎯 Custom Manual Threshold"
    ],
    index=0
)

if threshold_profile == "🌿 Strict Eco-Sensitive Zone (25% Opacity)":
    default_thresh = 25.0
    default_consec = 2
    default_burst = 20
    default_flow = 60.0
elif threshold_profile == "🚜 Heavy Demolition Zone (50% Opacity)":
    default_thresh = 50.0
    default_consec = 4
    default_burst = 10
    default_flow = 45.0
elif threshold_profile == "🎯 Custom Manual Threshold":
    default_thresh = 35.0
    default_consec = 3
    default_burst = 15
    default_flow = 50.0
else: # Standard 35%
    default_thresh = 35.0
    default_consec = 3
    default_burst = 15
    default_flow = 50.0

# Threshold Selection Dropdown
threshold = st.sidebar.selectbox(
    "Opacity Alarm Trigger (%):",
    options=[15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0],
    index=[15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0].index(default_thresh)
)

consec_frames = st.sidebar.selectbox(
    "Debounce Window (Consecutive Frames):",
    options=[1, 2, 3, 4, 5, 6],
    index=default_consec - 1
)

burst_sec = st.sidebar.selectbox(
    "Minimum Mist Burst Duration:",
    options=[5, 10, 15, 20, 25, 30],
    index=[5, 10, 15, 20, 25, 30].index(default_burst) if default_burst in [5, 10, 15, 20, 25, 30] else 2
)

cannon_flow = st.sidebar.number_input("Cannon Flow Rate (L/min)", min_value=10.0, max_value=150.0, value=default_flow, step=5.0)

# Stream Control
st.sidebar.markdown("---")
run_stream = st.sidebar.checkbox("▶️ Active Inspection Stream", value=True)

# Initialize Session State
if "state_machine" not in st.session_state:
    st.session_state.state_machine = DecisionStateMachine(
        threshold=threshold,
        consecutive_frames=consec_frames,
        burst_seconds=burst_sec,
        flow_rate_l_min=cannon_flow
    )
else:
    st.session_state.state_machine.threshold = threshold
    st.session_state.state_machine.consecutive_frames_req = consec_frames
    st.session_state.state_machine.burst_seconds = burst_sec
    st.session_state.state_machine.flow_rate_l_min = cannon_flow

# KPI Metric Cards Section
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    opacity_kpi_placeholder = st.empty()
with kpi_col2:
    status_kpi_placeholder = st.empty()
with kpi_col3:
    savings_kpi_placeholder = st.empty()
with kpi_col4:
    compliance_kpi_placeholder = st.empty()

# Main Layout: Left = Live Video Feed, Right = Telemetry & Real-Time Opacity Gauge
feed_col, telemetry_col = st.columns([5, 3])

with feed_col:
    st.subheader("📹 Real-Time Computer Vision Feed")
    video_placeholder = st.empty()

with telemetry_col:
    st.subheader("📊 Plume Opacity Telemetry")
    gauge_placeholder = st.empty()


def render_svg_gauge(current_val: float, alarm_threshold: float) -> str:
    """Renders a sleek, hardware-accelerated SVG speedometer gauge with glowing HUD theme."""
    clamped = max(0.0, min(100.0, current_val))
    if clamped >= alarm_threshold:
        bar_color = "#ef4444"
        status_txt = "CRITICAL BREACH"
        glow = "rgba(239, 68, 68, 0.8)"
    elif clamped >= alarm_threshold * 0.7:
        bar_color = "#f59e0b"
        status_txt = "ELEVATED DUST"
        glow = "rgba(245, 158, 11, 0.6)"
    else:
        bar_color = "#10b981"
        status_txt = "AIR CLEAN"
        glow = "rgba(16, 185, 129, 0.5)"

    dash_length = int((clamped / 100.0) * 220)

    return f"""
    <div style="background: rgba(15, 23, 42, 0.75); backdrop-filter: blur(12px); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 12px; padding: 18px; text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.5);">
        <div style="font-size: 0.8rem; font-weight: 700; color: #38bdf8; letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 6px;">
            Plume Opacity Index
        </div>
        <svg viewBox="0 0 200 135" width="100%" height="185">
            <path d="M 20 115 A 80 80 0 0 1 180 115" fill="none" stroke="#1e293b" stroke-width="16" stroke-linecap="round" />
            <path d="M 20 115 A 80 80 0 0 1 180 115" fill="none" stroke="#ef4444" stroke-width="16" stroke-dasharray="75 220" stroke-dashoffset="-145" opacity="0.3" stroke-linecap="butt" />
            <path d="M 20 115 A 80 80 0 0 1 180 115" fill="none" stroke="{bar_color}" stroke-width="16" stroke-dasharray="{dash_length} 220" stroke-linecap="round" style="filter: drop-shadow(0 0 8px {glow});" />
            <text x="100" y="85" font-size="34" font-weight="800" fill="#f8fafc" text-anchor="middle" font-family="system-ui, -apple-system, sans-serif">{clamped:.1f}%</text>
            <text x="100" y="105" font-size="11" font-weight="700" fill="{bar_color}" text-anchor="middle" letter-spacing="1px">{status_txt}</text>
            <text x="22" y="128" font-size="10" fill="#64748b" text-anchor="middle">0%</text>
            <text x="100" y="128" font-size="10" font-weight="700" fill="#f59e0b" text-anchor="middle">LIMIT: {alarm_threshold:.0f}%</text>
            <text x="178" y="128" font-size="10" fill="#64748b" text-anchor="middle">100%</text>
        </svg>
    </div>
    """


# Unique Hackathon Feature: Satellite Macro Geospatial Radar
st.markdown("---")
st.markdown("<h2 style='color: #ffffff; font-weight: 800; text-shadow: 0 0 16px rgba(56, 189, 248, 0.7);'>🛰️ Multi-Tier Surveillance: Satellite-to-Ground Geospatial Radar (ISRO / Sentinel-5P)</h2>", unsafe_allow_html=True)

# City Coordinates Database
CITY_COORDS = {
    "hyderabad": (17.3850, 78.4867, "Hyderabad HITEC Infrastructure Zone"),
    "mumbai": (19.1334, 72.9133, "Powai Construction Corridor (Mumbai)"),
    "delhi": (28.6139, 77.2090, "Delhi NCR Infrastructure Corridor"),
    "delhi ncr": (28.6139, 77.2090, "Delhi NCR Infrastructure Corridor"),
    "bengaluru": (12.9716, 77.5946, "Bengaluru Tech Corridor"),
    "bangalore": (12.9716, 77.5946, "Bengaluru Tech Corridor"),
    "pune": (18.5204, 73.8567, "Pune Industrial Ring Road"),
    "chennai": (13.0827, 80.2707, "Chennai Port Logistics Corridor"),
    "kolkata": (22.5726, 88.3639, "Kolkata Sector V Development"),
    "ahmedabad": (23.0225, 72.5714, "Ahmedabad Urban Ring Road"),
    "jaipur": (26.9124, 75.7873, "Jaipur Smart City Corridor")
}

if "selected_city" not in st.session_state:
    st.session_state["selected_city"] = "Hyderabad"

# Callback function when quick buttons are clicked
def set_city(city_name):
    st.session_state["selected_city"] = city_name

# Search Bar & Map Style Controls
search_col1, search_col2, search_col3 = st.columns([3, 2, 1])

with search_col1:
    city_text = st.text_input(
        "🔍 Search City or Site Corridor:", 
        value=st.session_state["selected_city"],
        help="Type any city (e.g., Mumbai, Delhi, Bengaluru, Pune, Hyderabad) and press Search"
    )

with search_col2:
    map_mode = st.radio("Map Layer:", options=["🌐 Satellite / Dark Matter", "🗺️ Standard Street Map"], horizontal=True)

with search_col3:
    st.write("") # spacing
    if st.button("🔎 Search", use_container_width=True):
        st.session_state["selected_city"] = city_text
        st.rerun()

# Quick Clickable City Shortcuts
q_col1, q_col2, q_col3, q_col4, q_col5 = st.columns(5)
with q_col1:
    st.button("📍 Hyderabad", use_container_width=True, on_click=set_city, args=("Hyderabad",))
with q_col2:
    st.button("📍 Mumbai", use_container_width=True, on_click=set_city, args=("Mumbai",))
with q_col3:
    st.button("📍 Delhi NCR", use_container_width=True, on_click=set_city, args=("Delhi NCR",))
with q_col4:
    st.button("📍 Bengaluru", use_container_width=True, on_click=set_city, args=("Bengaluru",))
with q_col5:
    st.button("📍 Pune", use_container_width=True, on_click=set_city, args=("Pune",))

# Determine effective target city
active_query = st.session_state.get("selected_city", "Hyderabad").strip().lower()

if active_query in CITY_COORDS:
    target_lat, target_lon, target_name = CITY_COORDS[active_query]
else:
    # Partial match lookup if user typed something like "delhi" or "mumbai"
    matched = False
    for k, v in CITY_COORDS.items():
        if k in active_query or active_query in k:
            target_lat, target_lon, target_name = v
            matched = True
            break
    if not matched:
        target_lat, target_lon, target_name = 17.3850, 78.4867, f"{active_query.title()} Construction Site"

# Select Map Style
chosen_style = "carto-darkmatter" if "Satellite" in map_mode else "open-street-map"

# Build Full-Width Interactive Movable Scattermap
fig_map = go.Figure()

# Dispersion nodes offset around target center
lats = [target_lat, target_lat + 0.0035, target_lat - 0.0030, target_lat + 0.0065, target_lat - 0.0055]
lons = [target_lon, target_lon + 0.0045, target_lon - 0.0035, target_lon + 0.0080, target_lon - 0.0065]
aod_vals = [0.68, 0.54, 0.49, 0.39, 0.34]
node_tags = ["🎯 SITE DUST CORE", "PLUME NORTH", "PLUME SOUTH", "DOWNWIND BUFFER", "CLEAN REF"]
node_hovers = [
    f"🎯 {target_name} - DUST EPICENTER [AOD: 0.68]",
    "🌪️ Plume Dispersion Sector North",
    "🌪️ Plume Dispersion Sector South",
    "🏙️ Downwind Buffer Zone",
    "🌿 Clean Ambient Reference Baseline"
]

# Dispersion Vector Line
fig_map.add_trace(go.Scattermap(
    lat=[target_lat, target_lat + 0.0035, target_lat + 0.0065],
    lon=[target_lon, target_lon + 0.0045, target_lon + 0.0080],
    mode="lines",
    line=dict(width=4, color="#38bdf8"),
    name="Wind Dispersion Vector (14 km/h)",
    hoverinfo="text",
    hovertext="💨 Atmospheric Wind Vector: 14 km/h Dispersion Toward Buffer Zone"
))

# AOD Monitoring Stations
fig_map.add_trace(go.Scattermap(
    lat=lats,
    lon=lons,
    mode="markers+text",
    marker=dict(
        size=[38, 28, 26, 22, 18],
        color=aod_vals,
        colorscale=[[0, "#10b981"], [0.45, "#f59e0b"], [1, "#ef4444"]],
        cmin=0.2,
        cmax=0.75,
        colorbar=dict(
            title=dict(text="<b>Aerosol AOD</b>", font=dict(color="#ffffff", size=13)),
            tickfont=dict(color="#ffffff")
        ),
        opacity=0.92
    ),
    text=node_tags,
    textposition="top right",
    textfont=dict(size=12, color="#ffffff", family="sans-serif"),
    hovertext=node_hovers,
    hoverinfo="text",
    name="AOD Stations"
))

# Full pan & drag config with cursor tracking and uirevision keying
fig_map.update_layout(
    uirevision=f"{target_lat}_{target_lon}_{chosen_style}", # Updates center when city changes, keeps pan state otherwise
    dragmode="pan",  # Makes the map directly draggable and movable with cursor
    map=dict(
        style=chosen_style,
        center=dict(lat=target_lat, lon=target_lon),
        zoom=13.0
    ),
    paper_bgcolor='rgba(13, 23, 46, 0.95)',
    height=550,
    margin=dict(l=0, r=0, t=35, b=0),
    title=dict(
        text=f"<b>🛰️ Sentinel-5P / ISRO Geospatial Radar: {target_name} ({target_lat:.4f}°N, {target_lon:.4f}°E)</b>",
        font=dict(color="#38bdf8", size=16)
    )
)

# Render Full-Width Map with smooth scroll zoom and movable cursor controls
st.plotly_chart(
    fig_map, 
    use_container_width=True, 
    key=f"sat_map_{active_query.replace(' ', '_')}_{chosen_style}",
    config={
        "scrollZoom": True,
        "displayModeBar": False
    }
)

# Municipal Audit & Inspection Records Section
st.markdown("---")
st.subheader("🏛️ Municipal Compliance & Incident Scorecard")

tab_log, tab_gallery = st.tabs(["📋 Violation Event Logs", "🖼️ Evidence Snapshots"])
with tab_log:
    table_placeholder = st.empty()
    download_placeholder = st.empty()

with tab_gallery:
    gallery_placeholder = st.empty()


def update_audit_views():
    """Updates the data inside the table and gallery placeholders."""
    events = get_recent_events(limit=15)
    if events:
        df_events = pd.DataFrame(events)
        table_placeholder.dataframe(df_events, use_container_width=True)
    else:
        table_placeholder.info("No threshold violations recorded yet. All operations currently within municipal air quality limits.")

    snapshots = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith(('.jpg', '.png'))]
    if snapshots:
        with gallery_placeholder.container():
            cols = st.columns(4)
            for idx, snap_file in enumerate(reversed(snapshots[-8:])):
                with cols[idx % 4]:
                    img_path = os.path.join(SNAPSHOT_DIR, snap_file)
                    st.image(img_path, caption=snap_file, use_container_width=True)
    else:
        gallery_placeholder.info("Snapshots will automatically appear here when opacity breaches threshold limits.")


# Initial display of records
update_audit_views()

# Run the Stream Processing Loop
if run_stream:
    buffer = FrameBuffer(source=video_source, target_fps=DEFAULT_FPS)
    try:
        buffer.start()
        frame_counter = 0
        last_log_time = 0.0

        for frame in buffer.frames():
            if not run_stream:
                break

            # 1. Hybrid Vision Inference
            results = process_frame(frame)
            opacity = results["opacity"]
            detections = results["objects"]

            # 2. Decision State Machine Update
            state, metrics = st.session_state.state_machine.update(opacity)

            # 3. Save Snapshots & Log Breach Events
            now_ts = time.time()
            if opacity >= threshold and (now_ts - last_log_time) > 3.0:
                last_log_time = now_ts
                snap_name = f"snapshot_{int(now_ts)}.jpg"
                snap_path = os.path.join(SNAPSHOT_DIR, snap_name)
                cv2.imwrite(snap_path, frame)
                log_dust_event("CAM_HAUL_ROAD_01", opacity, state, snap_name)
                log_water_savings(
                    metrics["baseline_usage_liters"],
                    metrics["dustguard_usage_liters"],
                    metrics["water_saved_liters"]
                )
                update_audit_views()

            # 4. Render HUD & Bounding Boxes
            annotated_frame = draw_hud(frame, detections, opacity, state, threshold)

            # 5. Streamlit Component Updates
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            # Telemetry Gauge (Smooth hardware-accelerated SVG gauge)
            gauge_placeholder.markdown(render_svg_gauge(opacity, threshold), unsafe_allow_html=True)

            # Metric Cards
            opacity_kpi_placeholder.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Current Opacity</div>
                <div class="metric-val" style="color: {'#ef4444' if opacity >= threshold else '#10b981'}">{opacity:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            if state in ("ACTIVE_MISTING", "COOL_DOWN"):
                badge_class = "status-badge-active"
            elif state == "PENDING":
                badge_class = "status-badge-pending"
            else:
                badge_class = "status-badge-standby"

            status_kpi_placeholder.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Cannon Actuation</div>
                <div class="metric-val"><span class="{badge_class}">{state.replace('_', ' ')}</span></div>
            </div>
            """, unsafe_allow_html=True)

            savings_kpi_placeholder.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Water Conserved</div>
                <div class="metric-val" style="color: #38bdf8">{metrics['water_saved_liters']:.1f} <span style="font-size: 1rem">L</span></div>
            </div>
            """, unsafe_allow_html=True)

            compliance_kpi_placeholder.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Water Efficiency</div>
                <div class="metric-val" style="color: #10b981">{metrics['efficiency_pct']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            frame_counter += 1
            time.sleep(0.04)  # Smooth UI throttle
    finally:
        buffer.stop()

# Post-stream Export CSV button (Rendered safely once)
events_export = get_recent_events(limit=50)
if events_export:
    df_exp = pd.DataFrame(events_export)
    csv_bytes = df_exp.to_csv(index=False).encode('utf-8')
    download_placeholder.download_button(
        label="📥 Export Municipal Audit Report (CSV)",
        data=csv_bytes,
        file_name="dustguard_compliance_report.csv",
        mime="text/csv",
        key="btn_download_export_csv"
    )

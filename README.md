# DustGuard‑Vision Prototype

A **hackathon‑ready** prototype implementing the architecture you described.  It is a **single‑repo micro‑monolith** written in Python and can be built and run in a day.

- **Video ingestion** – RTSP, USB webcam or uploaded MP4 (frame buffer @ ~10 FPS)
- **Hybrid inference** – YOLOv8 (Ultralytics) + OpenCV opacity estimator
- **Decision engine** – FSM with configurable opacity threshold & consecutive‑frame logic, triggers a mist‑cannon REST API and computes water‑saving metrics
- **Persistence** – SQLite (`dustguard.db`) storing events and snapshots
- **Dashboard** – Streamlit UI with live video, gauge, KPI cards and export to CSV/PDF

## Quick start
```bash
# 1. Create a virtual environment
python -m venv .venv && .\.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the YOLOv8 nano model (the script will fetch it on first run)
#    or place `yolov8n.pt` in the `models/` folder.

# 4. Run the app
streamlit run app.py
```

*The prototype is deliberately lightweight – no GPU is required for the demo; the model runs on CPU at ~10 FPS on a typical laptop.*

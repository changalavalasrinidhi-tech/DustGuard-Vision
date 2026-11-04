# constants.py
"""Global configuration for DustGuard‑Vision prototype.
All values are easy to tune from the UI if needed.
"""

# Video
DEFAULT_FPS = 10  # Target processing FPS (down‑sampled)
DEFAULT_SOURCE = ""  # Empty means user must provide via UI (RTSP, webcam index, or file path)

# Opacity detection
OPACITY_THRESHOLD = 35.0  # percent – triggers misting when exceeded for 3 frames
CONSECUTIVE_FRAMES = 3    # frames required above threshold before activation

# Misting cannon API (simulated – replace with real endpoint)
MISTING_API_URL = "http://localhost:5000/api/mist"  # POST JSON {"cannon_id": "c1", "state": "ON"}
CANNON_ID = "cannon_01"

# Water‑saving calculation
FLOW_RATE_L_PER_MIN = 50.0  # L/min of misting cannon when ON
BURST_SECONDS = 15          # Minimum ON time after activation

# Database
DB_PATH = "dustguard.db"

# core/decision_engine.py
"""Decision Engine & Finite State Machine (FSM) for DustGuard-Vision.
Controls misting cannon triggering and real-time water savings computation.
"""
import time
import requests
import logging
from typing import Dict, Any, Tuple

try:
    from constants import (
        OPACITY_THRESHOLD,
        CONSECUTIVE_FRAMES,
        MISTING_API_URL,
        CANNON_ID,
        FLOW_RATE_L_PER_MIN,
        BURST_SECONDS
    )
except ImportError:
    from ..constants import (
        OPACITY_THRESHOLD,
        CONSECUTIVE_FRAMES,
        MISTING_API_URL,
        CANNON_ID,
        FLOW_RATE_L_PER_MIN,
        BURST_SECONDS
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DecisionEngine")


class DecisionStateMachine:
    """Finite State Machine:
    - STANDBY: Opacity < 35%. Misting cannon OFF.
    - PENDING: Opacity >= 35% for < 3 consecutive frames (anti-flicker debouncing).
    - ACTIVE_MISTING: Opacity >= 35% for >= 3 frames. Cannon ON.
    - COOL_DOWN: Keeps cannon ON for a minimum burst duration (e.g. 15s) after threshold drop.
    """
    STANDBY = "STANDBY"
    PENDING = "PENDING"
    ACTIVE_MISTING = "ACTIVE_MISTING"
    COOL_DOWN = "COOL_DOWN"

    def __init__(
        self,
        threshold: float = OPACITY_THRESHOLD,
        consecutive_frames: int = CONSECUTIVE_FRAMES,
        burst_seconds: float = BURST_SECONDS,
        flow_rate_l_min: float = FLOW_RATE_L_PER_MIN,
        cannon_id: str = CANNON_ID,
        api_url: str = MISTING_API_URL
    ):
        self.threshold = threshold
        self.consecutive_frames_req = consecutive_frames
        self.burst_seconds = burst_seconds
        self.flow_rate_l_min = flow_rate_l_min
        self.cannon_id = cannon_id
        self.api_url = api_url

        self.current_state = self.STANDBY
        self.high_opacity_count = 0
        self.burst_start_time = 0.0

        # Timing tracking for water conservation
        self.session_start_time = time.time()
        self.total_misting_seconds = 0.0
        self._last_tick_time = time.time()

    def trigger_mister_api(self, state: str) -> bool:
        """Sends simulated or live HTTP POST to misting controller API."""
        payload = {
            "cannon_id": self.cannon_id,
            "state": state,
            "timestamp": time.time()
        }
        try:
            # We use a short timeout so UI stays fluid even if local endpoint is offline
            res = requests.post(self.api_url, json=payload, timeout=0.1)
            return res.status_code == 200
        except Exception:
            # Simulated mist cannon fallback
            logger.debug(f"[Simulated API] Triggered cannon {self.cannon_id} -> {state}")
            return True

    def update(self, opacity: float) -> Tuple[str, Dict[str, Any]]:
        """Update FSM state based on latest frame opacity score.
        Returns: (current_state, metrics_dict)
        """
        now = time.time()
        dt = now - self._last_tick_time
        self._last_tick_time = now

        # Track active misting duration
        if self.current_state in (self.ACTIVE_MISTING, self.COOL_DOWN):
            self.total_misting_seconds += dt

        # State transition logic
        if self.current_state == self.STANDBY:
            if opacity >= self.threshold:
                self.high_opacity_count = 1
                self.current_state = self.PENDING
            else:
                self.high_opacity_count = 0

        elif self.current_state == self.PENDING:
            if opacity >= self.threshold:
                self.high_opacity_count += 1
                if self.high_opacity_count >= self.consecutive_frames_req:
                    self.current_state = self.ACTIVE_MISTING
                    self.burst_start_time = now
                    self.trigger_mister_api("ON")
            else:
                # Glitch/shadow rejected by debouncer
                self.high_opacity_count = 0
                self.current_state = self.STANDBY

        elif self.current_state == self.ACTIVE_MISTING:
            if opacity >= self.threshold:
                # Still high, refresh burst window
                self.burst_start_time = now
            else:
                # Opacity dropped below threshold -> transition to COOL_DOWN burst hold
                self.current_state = self.COOL_DOWN

        elif self.current_state == self.COOL_DOWN:
            if opacity >= self.threshold:
                # Resurged during cool-down
                self.current_state = self.ACTIVE_MISTING
                self.burst_start_time = now
            elif (now - self.burst_start_time) >= self.burst_seconds:
                # Minimum burst cycle completed safely
                self.current_state = self.STANDBY
                self.high_opacity_count = 0
                self.trigger_mister_api("OFF")

        # Water Conservation Calculation:
        # Water Saved (Liters) = (T_total_operation - T_active_misting) * Flow_Rate (50 L/min)
        total_op_min = max(0.001, (now - self.session_start_time) / 60.0)
        active_misting_min = self.total_misting_seconds / 60.0
        baseline_usage_liters = total_op_min * self.flow_rate_l_min
        dustguard_usage_liters = active_misting_min * self.flow_rate_l_min
        water_saved_liters = max(0.0, baseline_usage_liters - dustguard_usage_liters)
        efficiency_pct = (water_saved_liters / baseline_usage_liters * 100.0) if baseline_usage_liters > 0 else 0.0

        metrics = {
            "state": self.current_state,
            "opacity": opacity,
            "high_opacity_count": self.high_opacity_count,
            "baseline_usage_liters": round(baseline_usage_liters, 2),
            "dustguard_usage_liters": round(dustguard_usage_liters, 2),
            "water_saved_liters": round(water_saved_liters, 2),
            "efficiency_pct": round(efficiency_pct, 1),
            "misting_active": self.current_state in (self.ACTIVE_MISTING, self.COOL_DOWN)
        }

        return self.current_state, metrics

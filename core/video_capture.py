# core/video_capture.py
"""Video capture and frame buffering layer.
Provides smooth frame acquisition with frame-rate downsampling (default 10 FPS).
Supports:
- Uploaded MP4 files
- USB Webcams (indexes like 0, 1)
- RTSP / HTTP video streams
- Synthetic Construction Site Generator (for instant hackathon demo when no camera is attached)
"""
import cv2
import threading
import time
import numpy as np
from collections import deque
from typing import Generator, Optional, Union

try:
    from constants import DEFAULT_FPS
except ImportError:
    from ..constants import DEFAULT_FPS


class FrameBuffer:
    def __init__(self, source: Union[str, int], target_fps: int = DEFAULT_FPS, buffer_size: int = 64):
        self.raw_source = source
        self.target_fps = target_fps
        self.buffer = deque(maxlen=buffer_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_interval = 1.0 / max(1, self.target_fps)
        self.is_synthetic = False
        self._sim_t = 0.0

    def start(self) -> None:
        src = self.raw_source
        if isinstance(src, str) and src.isdigit():
            src = int(src)

        if str(self.raw_source).lower() in ("demo", "synthetic", ""):
            self.is_synthetic = True
        else:
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                # Fallback gracefully to demo generator so the app never crashes
                self.is_synthetic = True

        self.thread.start()

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generates an attractive realistic synthetic construction site frame with moving trucks and dynamic dust clouds."""
        h, w = 480, 720
        frame = np.full((h, w, 3), (120, 160, 180), dtype=np.uint8)  # Sky/distant ground

        # Ground / Haul road (brownish beige)
        road_poly = np.array([[0, int(h * 0.45)], [w, int(h * 0.45)], [w, h], [0, h]], np.int32)
        cv2.fillPoly(frame, [road_poly], (70, 110, 150))

        # Construction site mounds
        cv2.ellipse(frame, (180, 260), (160, 70), 0, 0, 360, (60, 95, 130), -1)
        cv2.ellipse(frame, (560, 280), (140, 60), 0, 0, 360, (55, 90, 125), -1)

        # Dynamic truck motion
        self._sim_t += 0.08
        truck_x = int((self._sim_t * 40) % (w + 200)) - 100
        truck_y = 300

        # Draw vehicle / dump truck
        cv2.rectangle(frame, (truck_x, truck_y), (truck_x + 130, truck_y + 60), (30, 140, 230), -1)  # Yellow cabin
        cv2.rectangle(frame, (truck_x + 60, truck_y - 35), (truck_x + 140, truck_y + 30), (50, 60, 70), -1)  # Dump bed
        cv2.circle(frame, (truck_x + 30, truck_y + 60), 16, (20, 20, 20), -1)
        cv2.circle(frame, (truck_x + 105, truck_y + 60), 16, (20, 20, 20), -1)

        # Periodic dust storm / plume (simulated opacity waves)
        wave = np.sin(self._sim_t * 0.4)
        if wave > -0.2:
            dust_alpha = min(0.75, (wave + 0.2) * 0.7)
            dust_overlay = frame.copy()
            # Draw multi-layered plume near tires and road
            plume_x = truck_x - 30
            plume_y = truck_y + 20
            cv2.ellipse(dust_overlay, (plume_x, plume_y), (140, 80), 0, 0, 360, (140, 190, 215), -1)
            cv2.ellipse(dust_overlay, (plume_x - 60, plume_y - 20), (110, 60), 0, 0, 360, (160, 205, 230), -1)
            # Blend plume with frame for realistic airborne opacity
            cv2.addWeighted(dust_overlay, dust_alpha, frame, 1 - dust_alpha, 0, frame)

        return frame

    def _run(self) -> None:
        last_ts = time.time()
        while not self.stop_event.is_set():
            now = time.time()
            if now - last_ts >= self.frame_interval:
                if self.is_synthetic or self.cap is None:
                    frame = self._generate_synthetic_frame()
                    self.buffer.append(frame)
                else:
                    ret, frame = self.cap.read()
                    if not ret:
                        # Loop video files seamlessly for hackathon booth playback
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.cap.read()
                        if not ret:
                            frame = self._generate_synthetic_frame()
                    self.buffer.append(frame)
                last_ts = now
            time.sleep(0.005)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()

    def frames(self) -> Generator[np.ndarray, None, None]:
        while not self.stop_event.is_set():
            if self.buffer:
                yield self.buffer.popleft()
            else:
                time.sleep(0.01)

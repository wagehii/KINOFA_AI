"""
Kinova - AI Physiotherapy Computer Vision Engine
=================================================
Targeted smart-mirror pose tracking with granular rep telemetry and
backend synchronisation.

Requirements:
    pip install "numpy<2" opencv-python mediapipe requests

Usage:
    python kinova_cv_engine.py [config.json] [camera_index]

Controls:
    N - next exercise      P - previous exercise      Q - quit and sync
"""

from __future__ import annotations

import cv2
import json
import time
import math
import os
import sys
import urllib.request
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum, auto
from datetime import datetime, timezone

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_vision

try:
    import requests
except ImportError:
    requests = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POSE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODEL_FILENAME = "pose_landmarker_lite.task"

API_BASE_URL = os.getenv("KINOVA_API_URL", "http://localhost:8000")
API_TIMEOUT_SECONDS = 10

# BGR colours for the targeted skeleton overlay.
COLOR_CORRECT = (80, 220, 80)      # green: inside tolerance
COLOR_WARNING = (0, 180, 255)      # orange: deviation or overshoot

LANDMARK_LABELS: Dict[int, str] = {
    11: "Left Shoulder", 12: "Right Shoulder",
    13: "Left Elbow", 14: "Right Elbow",
    15: "Left Wrist", 16: "Right Wrist",
    23: "Left Hip", 24: "Right Hip",
    25: "Left Knee", 26: "Right Knee",
    27: "Left Ankle", 28: "Right Ankle",
}


def utc_now_iso() -> str:
    """RFC3339 timestamp with a trailing Z, matching the API contract."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def ensure_pose_model(model_path: str = MODEL_FILENAME) -> str:
    if not os.path.exists(model_path):
        print(f"Downloading pose model to '{model_path}' (one-time)...")
        urllib.request.urlretrieve(POSE_LANDMARKER_MODEL_URL, model_path)
        print("Model ready.")
    return model_path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExerciseConfig:
    exercise_id: str
    exercise_name: str
    target_joint: str
    mediapipe_keypoints: List[int]
    start_angle: float
    target_angle: float
    error_margin: float

    @property
    def direction(self) -> int:
        """+1 when the tracked angle should increase, -1 when it should decrease."""
        return 1 if self.target_angle >= self.start_angle else -1

    @property
    def total_rom(self) -> float:
        return abs(self.target_angle - self.start_angle)

    @property
    def corridor(self) -> Tuple[float, float]:
        """Allowed angular band including tolerance at both ends."""
        low = min(self.start_angle, self.target_angle) - self.error_margin
        high = max(self.start_angle, self.target_angle) + self.error_margin
        return low, high


class ConfigLoader:
    REQUIRED_FIELDS = (
        "exercise_id", "exercise_name", "target_joint",
        "mediapipe_keypoints", "start_angle", "target_angle", "error_margin",
    )

    @classmethod
    def load(cls, path: str) -> List[ExerciseConfig]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Exercise config not found: {path}")

        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)

        exercises: List[ExerciseConfig] = []
        for index, item in enumerate(raw):
            missing = [f for f in cls.REQUIRED_FIELDS if f not in item]
            if missing:
                raise ValueError(f"Exercise #{index} missing fields: {missing}")

            keypoints = list(item["mediapipe_keypoints"])
            if len(keypoints) != 3 or len(set(keypoints)) != 3:
                raise ValueError(f"{item['exercise_id']}: needs 3 distinct keypoints")
            if any(not (0 <= k <= 32) for k in keypoints):
                raise ValueError(f"{item['exercise_id']}: keypoint outside range 0-32")

            config = ExerciseConfig(
                exercise_id=item["exercise_id"],
                exercise_name=item["exercise_name"],
                target_joint=item["target_joint"],
                mediapipe_keypoints=keypoints,
                start_angle=float(item["start_angle"]),
                target_angle=float(item["target_angle"]),
                error_margin=float(item["error_margin"]),
            )

            if config.total_rom <= 2 * config.error_margin:
                raise ValueError(
                    f"{config.exercise_id}: error_margin {config.error_margin} too "
                    f"large for ROM {config.total_rom}; start and target bands overlap"
                )
            exercises.append(config)

        if not exercises:
            raise ValueError("Config contains no exercises.")
        return exercises


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class PoseMath:
    @staticmethod
    def angle_3d(a, b, c) -> float:
        """Angle in degrees at vertex b, using full 3D landmark coordinates."""
        pa = np.array([a.x, a.y, a.z], dtype=np.float64)
        pb = np.array([b.x, b.y, b.z], dtype=np.float64)
        pc = np.array([c.x, c.y, c.z], dtype=np.float64)

        v1, v2 = pa - pb, pc - pb
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0

        cosine = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        return math.degrees(math.acos(cosine))

    @staticmethod
    def visible(landmarks, indices, threshold: float = 0.4) -> bool:
        try:
            return all(landmarks[i].visibility >= threshold for i in indices)
        except (IndexError, AttributeError):
            return False


class AngleSmoother:
    """Exponential moving average to suppress landmark jitter."""

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self._value: Optional[float] = None

    def update(self, raw: float) -> float:
        if self._value is None:
            self._value = raw
        else:
            self._value = self.alpha * raw + (1.0 - self.alpha) * self._value
        return self._value

    def reset(self) -> None:
        self._value = None


# ---------------------------------------------------------------------------
# Telemetry models
# ---------------------------------------------------------------------------

@dataclass
class JointAngleReading:
    jointName: str
    angle: float
    timestamp: str

    def to_dict(self) -> dict:
        return {"jointName": self.jointName,
                "angle": round(self.angle, 1),
                "timestamp": self.timestamp}


@dataclass
class MovementError:
    errorType: str
    bodyPart: str
    severity: str
    deviationValue: float

    def to_dict(self) -> dict:
        return {"errorType": self.errorType,
                "bodyPart": self.bodyPart,
                "severity": self.severity,
                "deviationValue": round(self.deviationValue, 1)}


@dataclass
class RepetitionRecord:
    repNumber: int
    startTime: str
    endTime: str
    isCorrect: bool
    jointAngleReadings: List[JointAngleReading] = field(default_factory=list)
    movementErrors: List[MovementError] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "repNumber": self.repNumber,
            "startTime": self.startTime,
            "endTime": self.endTime,
            "isCorrect": self.isCorrect,
            "jointAngleReadings": [r.to_dict() for r in self.jointAngleReadings],
            "movementErrors": [e.to_dict() for e in self.movementErrors],
        }


class ErrorDetector:
    """Classifies deviations observed during a repetition."""

    @staticmethod
    def severity_for(deviation: float, margin: float) -> str:
        if deviation <= margin * 0.5:
            return "Low"
        if deviation <= margin * 1.5:
            return "Medium"
        return "High"

    @classmethod
    def evaluate(cls, config: ExerciseConfig, angles: List[float],
                 peak_progress: float) -> List[MovementError]:
        errors: List[MovementError] = []
        if not angles:
            return errors

        joint = config.target_joint
        direction = config.direction

        # Overshoot beyond the target end of the corridor.
        if direction > 0:
            overshoot = max(angles) - (config.target_angle + config.error_margin)
        else:
            overshoot = (config.target_angle - config.error_margin) - min(angles)
        if overshoot > 0:
            errors.append(MovementError(
                "Overextension", joint,
                cls.severity_for(overshoot, config.error_margin), overshoot))

        # Movement backwards past the start position.
        if direction > 0:
            regression = (config.start_angle - config.error_margin) - min(angles)
        else:
            regression = max(angles) - (config.start_angle + config.error_margin)
        if regression > 0:
            errors.append(MovementError(
                "PostureDeviation", joint,
                cls.severity_for(regression, config.error_margin), regression))

        # Incomplete range of motion.
        if peak_progress < 100.0:
            shortfall = config.total_rom * (100.0 - peak_progress) / 100.0
            if shortfall > config.error_margin:
                errors.append(MovementError(
                    "InsufficientROM", joint,
                    cls.severity_for(shortfall, config.error_margin), shortfall))

        # Tremor: repeated direction reversals within a rep.
        if len(angles) >= 5:
            deltas = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
            reversals = sum(
                1 for i in range(len(deltas) - 1)
                if deltas[i] * deltas[i + 1] < 0 and abs(deltas[i + 1]) > 1.5
            )
            # One reversal is the natural concentric/eccentric turnaround.
            if reversals > 3:
                errors.append(MovementError(
                    "UnstableMovement", joint,
                    cls.severity_for(float(reversals), 4.0), float(reversals)))

        return errors


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class RepPhase(Enum):
    IDLE = auto()
    CONCENTRIC = auto()
    ECCENTRIC = auto()
    REP_COUNTED = auto()


class ExerciseTracker:
    """Per-exercise state machine, metrics, and rep-level telemetry capture."""

    def __init__(self, config: ExerciseConfig):
        self.config = config
        self.phase = RepPhase.IDLE

        self.valid_reps = 0
        self.missed_reps = 0
        self.records: List[RepetitionRecord] = []

        self.total_samples = 0
        self.in_corridor_samples = 0
        self.max_rom_angle: Optional[float] = None
        self.time_spent = 0.0
        self._last_tick: Optional[float] = None

        self.current_angle = config.start_angle
        self.current_progress = 0.0
        self.in_tolerance = True
        self.feedback_text = "Move to the starting position"

        self._reset_rep_buffers()

    def _reset_rep_buffers(self) -> None:
        self._rep_angles: List[float] = []
        self._rep_readings: List[JointAngleReading] = []
        self._rep_start_ts: Optional[str] = None
        self._rep_peak_progress = 0.0
        self._reached_target = False
        self._left_start_zone = False

    @staticmethod
    def _within(value: float, target: float, margin: float) -> bool:
        return abs(value - target) <= margin

    def _begin_rep(self) -> None:
        self._reset_rep_buffers()
        self._rep_start_ts = utc_now_iso()
        self.phase = RepPhase.CONCENTRIC

    def _close_rep(self, is_correct: bool) -> None:
        if self._rep_start_ts is None:
            self._reset_rep_buffers()
            return

        errors = ErrorDetector.evaluate(
            self.config, self._rep_angles, self._rep_peak_progress)

        # Downsample readings to keep the payload compact but representative.
        readings = self._rep_readings
        if len(readings) > 30:
            step = len(readings) / 30.0
            readings = [readings[int(i * step)] for i in range(30)]

        self.records.append(RepetitionRecord(
            repNumber=len(self.records) + 1,
            startTime=self._rep_start_ts,
            endTime=utc_now_iso(),
            isCorrect=is_correct,
            jointAngleReadings=readings,
            movementErrors=errors,
        ))

        if is_correct:
            self.valid_reps += 1
        else:
            self.missed_reps += 1
        self._reset_rep_buffers()

    def update(self, angle: float) -> str:
        now = time.time()
        if self._last_tick is not None:
            self.time_spent += now - self._last_tick
        self._last_tick = now

        cfg = self.config
        self.current_angle = angle
        self.total_samples += 1

        # Max ROM in the intended direction.
        if self.max_rom_angle is None:
            self.max_rom_angle = angle
        elif cfg.direction > 0:
            self.max_rom_angle = max(self.max_rom_angle, angle)
        else:
            self.max_rom_angle = min(self.max_rom_angle, angle)

        low, high = cfg.corridor
        self.in_tolerance = low <= angle <= high
        if self.in_tolerance:
            self.in_corridor_samples += 1

        progress = 0.0
        if cfg.total_rom > 0:
            progress = max(0.0, min(100.0,
                           abs(angle - cfg.start_angle) / cfg.total_rom * 100.0))
        self.current_progress = progress

        at_start = self._within(angle, cfg.start_angle, cfg.error_margin)
        at_target = self._within(angle, cfg.target_angle, cfg.error_margin)

        if self.phase in (RepPhase.CONCENTRIC, RepPhase.ECCENTRIC):
            self._rep_angles.append(angle)
            self._rep_readings.append(
                JointAngleReading(cfg.target_joint, angle, utc_now_iso()))
            self._rep_peak_progress = max(self._rep_peak_progress, progress)

        if self.phase in (RepPhase.IDLE, RepPhase.REP_COUNTED):
            if at_start:
                self._begin_rep()
                self.feedback_text = "Start the movement"
            else:
                self.feedback_text = "Return to the starting position"

        elif self.phase == RepPhase.CONCENTRIC:
            if not at_start:
                self._left_start_zone = True

            if at_target:
                self._reached_target = True
                self.phase = RepPhase.ECCENTRIC
                self.feedback_text = "Target reached - return slowly"
            elif at_start and self._left_start_zone:
                self._close_rep(is_correct=False)
                self.phase = RepPhase.IDLE
                self.feedback_text = "Incomplete rep - extend further"
            elif not self.in_tolerance:
                self.feedback_text = "Correct your posture"
            elif abs(cfg.target_angle - angle) <= cfg.error_margin * 2:
                self.feedback_text = "Almost there"
            else:
                self.feedback_text = "Keep moving toward the target"

        elif self.phase == RepPhase.ECCENTRIC:
            if at_target:
                self._reached_target = True
            if at_start:
                self._close_rep(is_correct=self._reached_target)
                self.phase = RepPhase.REP_COUNTED
                self.feedback_text = "Good rep"
            elif not self.in_tolerance:
                self.feedback_text = "Control the return"
            else:
                self.feedback_text = "Return to the starting position"

        return self.feedback_text

    # A rep must show this much progress to be treated as a genuine attempt.
    MIN_ATTEMPT_PROGRESS = 15.0

    def finalise(self) -> None:
        """Close any rep still in flight. Reps that never left the start zone
        are discarded rather than counted as failures, so simply standing in
        the start position does not inflate the invalid count."""
        if self.phase not in (RepPhase.CONCENTRIC, RepPhase.ECCENTRIC):
            return

        if self._left_start_zone and self._rep_peak_progress >= self.MIN_ATTEMPT_PROGRESS:
            self._close_rep(is_correct=self._reached_target)
        else:
            self._reset_rep_buffers()
        self.phase = RepPhase.IDLE

    @property
    def form_accuracy(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return round(self.in_corridor_samples / self.total_samples * 100.0, 1)

    @property
    def rom_achieved(self) -> float:
        if self.max_rom_angle is None:
            return 0.0
        return round(abs(self.max_rom_angle - self.config.start_angle), 1)


# ---------------------------------------------------------------------------
# Backend client
# ---------------------------------------------------------------------------

class BackendClient:
    def __init__(self, base_url: str = API_BASE_URL, timeout: int = API_TIMEOUT_SECONDS):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete_session(self, session_id: str, payload: dict) -> Optional[dict]:
        if requests is None:
            print("`requests` not installed - skipping upload.")
            return None

        url = f"{self.base_url}/api/v1/Session/{session_id}/complete"
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            print(f"Backend sync failed ({exc}). Payload saved locally.")
            return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class TargetedRenderer:
    """Draws only the keypoints relevant to the active exercise."""

    @staticmethod
    def _to_pixels(landmark, width: int, height: int) -> Tuple[int, int]:
        return int(landmark.x * width), int(landmark.y * height)

    @classmethod
    def draw_active_chain(cls, frame, landmarks, config: ExerciseConfig,
                          in_tolerance: bool) -> None:
        height, width = frame.shape[:2]
        idx_a, idx_b, idx_c = config.mediapipe_keypoints
        color = COLOR_CORRECT if in_tolerance else COLOR_WARNING

        try:
            pa = cls._to_pixels(landmarks[idx_a], width, height)
            pb = cls._to_pixels(landmarks[idx_b], width, height)
            pc = cls._to_pixels(landmarks[idx_c], width, height)
        except IndexError:
            return

        # Segments forming the tracked angle.
        cv2.line(frame, pa, pb, color, 4, cv2.LINE_AA)
        cv2.line(frame, pb, pc, color, 4, cv2.LINE_AA)

        # Outer keypoints.
        for point in (pa, pc):
            cv2.circle(frame, point, 8, color, -1, cv2.LINE_AA)
            cv2.circle(frame, point, 8, (25, 25, 25), 1, cv2.LINE_AA)

        # Vertex joint emphasised.
        cv2.circle(frame, pb, 13, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pb, 13, (25, 25, 25), 2, cv2.LINE_AA)
        cv2.circle(frame, pb, 20, color, 2, cv2.LINE_AA)

        label = LANDMARK_LABELS.get(idx_b, config.target_joint)
        cv2.putText(frame, label, (pb[0] + 24, pb[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    @staticmethod
    def draw_hud(frame, tracker: ExerciseTracker, index: int, total: int,
                 angle: Optional[float], feedback: str) -> None:
        height, width = frame.shape[:2]
        cfg = tracker.config
        accent = COLOR_CORRECT if tracker.in_tolerance else COLOR_WARNING

        panel = frame.copy()
        cv2.rectangle(panel, (0, 0), (width, 118), (18, 18, 18), -1)
        cv2.rectangle(panel, (0, height - 96), (width, height), (18, 18, 18), -1)
        cv2.addWeighted(panel, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, cfg.exercise_name, (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"[{index + 1}/{total}]  Joint: {cfg.target_joint}",
                    (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (185, 185, 185), 1, cv2.LINE_AA)

        shown = f"{angle:.1f}" if angle is not None else "--"
        cv2.putText(frame, f"Angle {shown}  ->  Target {cfg.target_angle:.0f}"
                           f" (+/-{cfg.error_margin:.0f})",
                    (16, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.62, accent, 2, cv2.LINE_AA)

        cv2.putText(frame, f"Valid {tracker.valid_reps}", (width - 210, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, COLOR_CORRECT, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Missed {tracker.missed_reps}", (width - 210, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 245), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{tracker.phase.name}  Form {tracker.form_accuracy:.0f}%",
                    (width - 210, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (215, 215, 215), 1, cv2.LINE_AA)

        # ROM progress bar.
        bar_x, bar_y = 16, height - 74
        bar_w, bar_h = width - 32, 16
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (58, 58, 58), -1)
        filled = int(bar_w * tracker.current_progress / 100.0)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                      accent, -1)
        cv2.putText(frame, f"ROM {tracker.current_progress:.0f}%",
                    (bar_x + 6, bar_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (215, 215, 215), 1, cv2.LINE_AA)

        cv2.putText(frame, feedback, (16, height - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, accent, 2, cv2.LINE_AA)
        cv2.putText(frame, "N next | P prev | Q quit", (width - 250, height - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (150, 150, 150), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class KinovaCVEngine:
    def __init__(self, config_path: str, camera_index: int = 0,
                 session_id: Optional[str] = None,
                 payload_path: str = "session_payload.json"):
        self.exercises = ConfigLoader.load(config_path)
        self.trackers = {e.exercise_id: ExerciseTracker(e) for e in self.exercises}
        self.smoothers = {e.exercise_id: AngleSmoother() for e in self.exercises}

        self.index = 0
        self.camera_index = camera_index
        self.payload_path = payload_path
        self.session_id = session_id or f"kinova-{datetime.now():%Y%m%d-%H%M%S}"

        self.client = BackendClient()
        self.cap: Optional[cv2.VideoCapture] = None
        self.session_start = time.time()

        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks_python.BaseOptions(
                model_asset_path=ensure_pose_model()),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    @property
    def exercise(self) -> ExerciseConfig:
        return self.exercises[self.index]

    @property
    def tracker(self) -> ExerciseTracker:
        return self.trackers[self.exercise.exercise_id]

    def _switch(self, step: int) -> None:
        self.tracker.finalise()
        self.index = (self.index + step) % len(self.exercises)
        self.smoothers[self.exercise.exercise_id].reset()

    # ----------------------------- main loop -----------------------------

    def run(self) -> None:
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.camera_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print(f"Session {self.session_id} started. N next | P prev | Q quit")
        clock_origin = time.time()
        last_ts = -1

        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("Camera read failed.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                ts = int((time.time() - clock_origin) * 1000)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts

                result = self.landmarker.detect_for_video(image, ts)

                angle: Optional[float] = None
                feedback = self.tracker.feedback_text
                cfg = self.exercise

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    if PoseMath.visible(landmarks, cfg.mediapipe_keypoints):
                        a, b, c = (landmarks[i] for i in cfg.mediapipe_keypoints)
                        raw = PoseMath.angle_3d(a, b, c)
                        angle = self.smoothers[cfg.exercise_id].update(raw)
                        feedback = self.tracker.update(angle)
                        TargetedRenderer.draw_active_chain(
                            frame, landmarks, cfg, self.tracker.in_tolerance)
                    else:
                        feedback = "Target joint not visible"
                else:
                    feedback = "No person detected"

                TargetedRenderer.draw_hud(
                    frame, self.tracker, self.index, len(self.exercises),
                    angle, feedback)
                cv2.imshow("Kinova - Smart Mirror", frame)

                key = cv2.waitKey(5) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if key in (ord("n"), ord("N")):
                    self._switch(1)
                elif key in (ord("p"), ord("P")):
                    self._switch(-1)

        finally:
            self.shutdown()

    # ------------------------------ teardown ------------------------------

    def build_payload(self) -> dict:
        """Flatten every tracker's records into the API telemetry contract."""
        reps: List[dict] = []
        for cfg in self.exercises:
            for record in self.trackers[cfg.exercise_id].records:
                entry = record.to_dict()
                entry["repNumber"] = len(reps) + 1
                reps.append(entry)
        return {"reps": reps}

    def shutdown(self) -> None:
        for tracker in self.trackers.values():
            tracker.finalise()

        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()

        payload = self.build_payload()
        with open(self.payload_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        total = len(payload["reps"])
        valid = sum(1 for r in payload["reps"] if r["isCorrect"])
        print(f"\nSession {self.session_id}: {total} reps "
              f"({valid} valid, {total - valid} invalid)")
        print(f"Payload written to {os.path.abspath(self.payload_path)}")

        if total == 0:
            print("No repetitions recorded - skipping upload.")
            return

        scores = self.client.complete_session(self.session_id, payload)
        if scores:
            print("Backend scores:")
            for key, value in scores.items():
                print(f"  {key}: {value}")


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "exercises_config.json"
    camera_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    KinovaCVEngine(config_path, camera_index).run()


if __name__ == "__main__":
    main()

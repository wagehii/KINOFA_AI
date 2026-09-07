"""
Kinova AI Physiotherapy Engine
------------------------------
Dynamic, camera-only exercise evaluation pipeline built on MediaPipe Pose + OpenCV.

IMPORTANT CALIBRATION NOTE:
The angle thresholds in EXERCISE_LIBRARY below are domain-standard physiotherapy
ROM reference values for the exercise types described in the UI-PRMD dataset
(Vakanski et al., 2018), NOT values statistically mined from the actual UI-PRMD
CSV/Vicon files. Replace `ideal_peak_angle`, `rest_angle`, and `tolerance_deg`
per exercise with numbers computed from real per-frame joint-angle statistics
(mean/std at peak ROM, split by is_correct) once that offline analysis is done.
The state machine, rep-counting, and UX logic are fully self-contained and will
work unchanged once you swap in calibrated numbers.

Run: python kinova_engine.py
Controls: N = next exercise | P = previous exercise | Q = quit + write session_report.json
"""

from __future__ import annotations

import cv2
import json
import math
import time
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from typing import Optional


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class ExerciseConfig:
    name: str
    # Landmark triplet (a, b, c) -> angle is measured at b, between vectors b->a and b->c
    joint_triplet: tuple  # mp_pose.PoseLandmark names
    # Additional limb connections (pairs of landmark names) to color-code on screen
    highlight_connections: list
    rest_angle: float          # angle at the "idle" position (deg)
    ideal_peak_angle: float    # angle at the target end-range-of-motion (deg)
    tolerance_deg: float       # +/- margin counted as "correct" near the peak
    min_rom_to_start_rep: float  # angle delta from rest required to leave IDLE
    direction: str             # "decrease" if peak < rest, "increase" if peak > rest
    feedback_ok: str
    feedback_bad: str
    min_phase_seconds: float = 0.25   # debounce: min time a phase must hold


EXERCISE_LIBRARY = [
    ExerciseConfig(
        name="Deep Squat",
        joint_triplet=("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
        highlight_connections=[
            ("LEFT_HIP", "LEFT_KNEE"), ("LEFT_KNEE", "LEFT_ANKLE"),
            ("RIGHT_HIP", "RIGHT_KNEE"), ("RIGHT_KNEE", "RIGHT_ANKLE"),
        ],
        rest_angle=170.0,
        ideal_peak_angle=90.0,
        tolerance_deg=15.0,
        min_rom_to_start_rep=20.0,
        direction="decrease",
        feedback_ok="Good depth! Keep your back straight.",
        feedback_bad="Go lower and keep knees over toes.",
    ),
    ExerciseConfig(
        name="Sit To Stand",
        joint_triplet=("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"),
        highlight_connections=[
            ("LEFT_HIP", "LEFT_KNEE"), ("LEFT_SHOULDER", "LEFT_HIP"),
            ("RIGHT_HIP", "RIGHT_KNEE"), ("RIGHT_SHOULDER", "RIGHT_HIP"),
        ],
        rest_angle=95.0,
        ideal_peak_angle=170.0,
        tolerance_deg=12.0,
        min_rom_to_start_rep=20.0,
        direction="increase",
        feedback_ok="Full extension, nice!",
        feedback_bad="Stand up fully, extend your hips.",
    ),
    ExerciseConfig(
        name="Standing Shoulder Abduction",
        joint_triplet=("LEFT_HIP", "LEFT_SHOULDER", "LEFT_ELBOW"),
        highlight_connections=[
            ("LEFT_SHOULDER", "LEFT_ELBOW"), ("LEFT_ELBOW", "LEFT_WRIST"),
            ("RIGHT_SHOULDER", "RIGHT_ELBOW"), ("RIGHT_ELBOW", "RIGHT_WRIST"),
        ],
        rest_angle=15.0,
        ideal_peak_angle=160.0,
        tolerance_deg=15.0,
        min_rom_to_start_rep=25.0,
        direction="increase",
        feedback_ok="Full abduction reached, controlled!",
        feedback_bad="Raise your arm higher, keep it straight.",
    ),
]

COLOR_GOOD = (0, 200, 0)
COLOR_BAD = (0, 100, 255)
COLOR_NEUTRAL = (200, 200, 200)
COLOR_TEXT = (255, 255, 255)


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

def calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b (degrees), given three (x, y) points."""
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def landmark_px(landmarks, name: str, w: int, h: int, mp_pose) -> np.ndarray:
    idx = mp_pose.PoseLandmark[name].value
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])


def landmark_visible(landmarks, name: str, mp_pose, min_vis: float = 0.5) -> bool:
    idx = mp_pose.PoseLandmark[name].value
    return landmarks[idx].visibility >= min_vis


# --------------------------------------------------------------------------- #
# Rep state machine
# --------------------------------------------------------------------------- #

class Phase(Enum):
    IDLE = auto()
    CONCENTRIC = auto()   # moving toward target ROM
    ECCENTRIC = auto()    # returning toward rest
    COUNTED = auto()


class RepStateMachine:
    """
    Robust rep counter with hysteresis + time debounce to reject noise/false
    positives from jittery pose estimation.
    """

    def __init__(self, config: ExerciseConfig, smoothing_window: int = 5):
        self.cfg = config
        self.phase = Phase.IDLE
        self.angle_buffer: deque = deque(maxlen=smoothing_window)
        self.phase_entered_at = time.time()
        self.peak_angle: Optional[float] = None
        self.valid_reps = 0
        self.missed_reps = 0
        self.max_rom_seen = None
        self.last_rep_was_correct = False
        self.feedback = "Get ready"

    def _smoothed_angle(self, raw_angle: float) -> float:
        self.angle_buffer.append(raw_angle)
        return float(np.mean(self.angle_buffer))

    def _delta_from_rest(self, angle: float) -> float:
        return abs(angle - self.cfg.rest_angle)

    def _reached_target(self, angle: float) -> bool:
        return abs(angle - self.cfg.ideal_peak_angle) <= self.cfg.tolerance_deg

    def _moving_toward_peak(self, angle: float) -> bool:
        if self.cfg.direction == "decrease":
            return angle < self.cfg.rest_angle - self.cfg.min_rom_to_start_rep
        return angle > self.cfg.rest_angle + self.cfg.min_rom_to_start_rep

    def _back_near_rest(self, angle: float) -> bool:
        return self._delta_from_rest(angle) <= self.cfg.min_rom_to_start_rep * 0.5

    def _phase_hold_ok(self) -> bool:
        return (time.time() - self.phase_entered_at) >= self.cfg.min_phase_seconds

    def _set_phase(self, new_phase: Phase):
        if new_phase != self.phase:
            self.phase = new_phase
            self.phase_entered_at = time.time()

    def update(self, raw_angle: float) -> dict:
        angle = self._smoothed_angle(raw_angle)

        if self.max_rom_seen is None:
            self.max_rom_seen = angle
        else:
            if self.cfg.direction == "decrease":
                self.max_rom_seen = min(self.max_rom_seen, angle)
            else:
                self.max_rom_seen = max(self.max_rom_seen, angle)

        on_target = self._reached_target(angle)

        if self.phase == Phase.IDLE:
            self.feedback = "Start the movement"
            if self._moving_toward_peak(angle) and self._phase_hold_ok():
                self._set_phase(Phase.CONCENTRIC)

        elif self.phase == Phase.CONCENTRIC:
            self.feedback = self.cfg.feedback_ok if on_target else self.cfg.feedback_bad
            self.peak_angle = angle if self.peak_angle is None else (
                min(self.peak_angle, angle) if self.cfg.direction == "decrease"
                else max(self.peak_angle, angle)
            )
            still_progressing = self._moving_toward_peak(angle)
            if not still_progressing and self._phase_hold_ok():
                # direction reversed -> user is on the way back
                self._set_phase(Phase.ECCENTRIC)

        elif self.phase == Phase.ECCENTRIC:
            self.feedback = "Controlled return..."
            if self._back_near_rest(angle) and self._phase_hold_ok():
                self._set_phase(Phase.COUNTED)

        elif self.phase == Phase.COUNTED:
            was_correct = self.peak_angle is not None and self._reached_target(self.peak_angle)
            if was_correct:
                self.valid_reps += 1
                self.last_rep_was_correct = True
                self.feedback = "Rep counted - Good form!"
            else:
                self.missed_reps += 1
                self.last_rep_was_correct = False
                self.feedback = "Rep counted - insufficient ROM"
            self.peak_angle = None
            self._set_phase(Phase.IDLE)

        return {
            "phase": self.phase.name,
            "angle": angle,
            "on_target": on_target,
            "feedback": self.feedback,
        }

    def accuracy_pct(self) -> float:
        total = self.valid_reps + self.missed_reps
        return round(100.0 * self.valid_reps / total, 1) if total else 0.0


# --------------------------------------------------------------------------- #
# Session reporting
# --------------------------------------------------------------------------- #

class SessionRecorder:
    def __init__(self):
        self.start_time = time.time()
        self.exercise_stats = {}

    def record(self, exercise_name: str, sm: RepStateMachine):
        self.exercise_stats[exercise_name] = {
            "valid_reps": sm.valid_reps,
            "missed_reps": sm.missed_reps,
            "accuracy_pct": sm.accuracy_pct(),
            "max_rom_deg": None if sm.max_rom_seen is None else round(float(sm.max_rom_seen), 1),
            "baseline_ideal_peak_deg": sm.cfg.ideal_peak_angle,
            "baseline_source": "domain_reference_not_dataset_calibrated",
        }

    def _insight_for(self, name: str, stats: dict) -> str:
        if stats["valid_reps"] + stats["missed_reps"] == 0:
            return f"No reps completed for {name}."
        if stats["accuracy_pct"] >= 80:
            return f"{name}: strong form, ROM consistently near target ({stats['max_rom_deg']} deg)."
        if stats["accuracy_pct"] >= 50:
            return f"{name}: moderate form, ROM undershoot vs baseline of {stats['baseline_ideal_peak_deg']} deg."
        return f"{name}: form needs attention, frequent undershoot of target ROM."

    def to_dict(self) -> dict:
        duration_s = round(time.time() - self.start_time, 1)
        insights = [self._insight_for(name, s) for name, s in self.exercise_stats.items()]
        return {
            "session_duration_seconds": duration_s,
            "exercises": self.exercise_stats,
            "ai_insights": insights,
            "note": "Baselines are domain-reference defaults; recalibrate against "
                     "actual UI-PRMD per-frame joint statistics before clinical use.",
        }

    def write(self, path: str = "session_report.json"):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# --------------------------------------------------------------------------- #
# Drawing / HUD
# --------------------------------------------------------------------------- #

def draw_angle_arc(frame, vertex: np.ndarray, a: np.ndarray, c: np.ndarray,
                    angle_deg: float, color, radius: int = 60):
    v_ang_a = math.degrees(math.atan2(a[1] - vertex[1], a[0] - vertex[0]))
    v_ang_c = math.degrees(math.atan2(c[1] - vertex[1], c[0] - vertex[0]))
    start, end = v_ang_a, v_ang_c
    # normalize sweep to the shorter, visually-correct arc
    if abs(end - start) > 180:
        if end > start:
            start += 360
        else:
            end += 360
    cv2.ellipse(frame, tuple(vertex.astype(int)), (radius, radius), 0,
                min(start, end), max(start, end), color, 3)


def draw_highlight_connections(frame, landmarks, connections, w, h, mp_pose, color):
    for name_a, name_b in connections:
        if not (landmark_visible(landmarks, name_a, mp_pose) and
                landmark_visible(landmarks, name_b, mp_pose)):
            continue
        pa = landmark_px(landmarks, name_a, w, h, mp_pose).astype(int)
        pb = landmark_px(landmarks, name_b, w, h, mp_pose).astype(int)
        cv2.line(frame, tuple(pa), tuple(pb), color, 6)
        cv2.circle(frame, tuple(pa), 7, color, -1)
        cv2.circle(frame, tuple(pb), 7, color, -1)


def draw_hud(frame, exercise_name, sm: RepStateMachine, feedback: str, fps: float):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, exercise_name, (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, COLOR_TEXT, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Valid: {sm.valid_reps}   Missed: {sm.missed_reps}   "
                        f"Acc: {sm.accuracy_pct()}%",
                (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_TEXT, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Phase: {sm.phase.name}   FPS: {fps:.0f}",
                (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 220, 255), 2, cv2.LINE_AA)

    fb_color = COLOR_GOOD if ("good" in feedback.lower() or "!" in feedback) else COLOR_TEXT
    cv2.putText(frame, feedback, (30, h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.95, fb_color, 2, cv2.LINE_AA)
    cv2.putText(frame, "[N] Next  [P] Prev  [Q] Quit + Report",
                (30, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Main application
# --------------------------------------------------------------------------- #

class KinovaEngine:
    def __init__(self, camera_index: int = 0):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.cap = cv2.VideoCapture(camera_index)
        self.window_name = "Kinova AI Physiotherapy"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN,
                               cv2.WINDOW_FULLSCREEN)

        self.exercise_idx = 0
        self.state_machine = RepStateMachine(EXERCISE_LIBRARY[self.exercise_idx])
        self.recorder = SessionRecorder()
        self._prev_time = time.time()

    def _switch_exercise(self, delta: int):
        self.recorder.record(self.state_machine.cfg.name, self.state_machine)
        self.exercise_idx = (self.exercise_idx + delta) % len(EXERCISE_LIBRARY)
        self.state_machine = RepStateMachine(EXERCISE_LIBRARY[self.exercise_idx])

    def _fps(self) -> float:
        now = time.time()
        dt = now - self._prev_time
        self._prev_time = now
        return 1.0 / dt if dt > 0 else 0.0

    def run(self):
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb)

                cfg = self.state_machine.cfg
                feedback = "No person detected"

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    name_a, name_b, name_c = cfg.joint_triplet
                    if all(landmark_visible(lm, n, self.mp_pose) for n in (name_a, name_b, name_c)):
                        pa = landmark_px(lm, name_a, w, h, self.mp_pose)
                        pb = landmark_px(lm, name_b, w, h, self.mp_pose)
                        pc = landmark_px(lm, name_c, w, h, self.mp_pose)
                        raw_angle = calc_angle(pa, pb, pc)
                        status = self.state_machine.update(raw_angle)
                        feedback = status["feedback"]
                        color = COLOR_GOOD if status["on_target"] else (
                            COLOR_BAD if status["phase"] in ("CONCENTRIC", "ECCENTRIC") else COLOR_NEUTRAL
                        )
                        draw_highlight_connections(frame, lm, cfg.highlight_connections,
                                                    w, h, self.mp_pose, color)
                        draw_angle_arc(frame, pb, pa, pc, status["angle"], color)
                    else:
                        feedback = "Move into full frame view"

                fps = self._fps()
                draw_hud(frame, cfg.name, self.state_machine, feedback, fps)
                cv2.imshow(self.window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('n'):
                    self._switch_exercise(1)
                elif key == ord('p'):
                    self._switch_exercise(-1)
        finally:
            self.recorder.record(self.state_machine.cfg.name, self.state_machine)
            self.recorder.write("session_report.json")
            self.cap.release()
            cv2.destroyAllWindows()
            self.pose.close()


if __name__ == "__main__":
    KinovaEngine(camera_index=0).run()

"""
Kinova Physiotherapy Platform - Telemetry Ingestion API
=======================================================
Receives per-rep telemetry from the CV engine and computes session scores.

Run:
    pip install fastapi uvicorn
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Docs:
    http://localhost:8000/docs
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import List, Dict

from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("kinova.api")


# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


# Penalty weight applied to the accuracy score per error severity.
SEVERITY_WEIGHT: Dict[str, float] = {
    Severity.LOW: 1.0,
    Severity.MEDIUM: 2.5,
    Severity.HIGH: 5.0,
}

# Score composition weights for the overall score.
WEIGHT_ACCURACY = 0.40
WEIGHT_ROM = 0.35
WEIGHT_STABILITY = 0.25


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class JointAngleReading(BaseModel):
    jointName: str = Field(..., min_length=1, max_length=64)
    angle: float = Field(..., ge=0.0, le=180.0)
    timestamp: datetime

    model_config = {"extra": "forbid"}


class MovementError(BaseModel):
    errorType: str = Field(..., min_length=1, max_length=64)
    bodyPart: str = Field(..., min_length=1, max_length=64)
    severity: Severity
    deviationValue: float = Field(..., ge=0.0)

    model_config = {"extra": "forbid"}


class Repetition(BaseModel):
    repNumber: int = Field(..., ge=1)
    startTime: datetime
    endTime: datetime
    isCorrect: bool
    jointAngleReadings: List[JointAngleReading] = Field(default_factory=list)
    movementErrors: List[MovementError] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    @field_validator("endTime")
    @classmethod
    def _end_after_start(cls, v: datetime, info):
        start = info.data.get("startTime")
        if start is not None and v < start:
            raise ValueError("endTime must not precede startTime")
        return v


class SessionTelemetry(BaseModel):
    reps: List[Repetition] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------

class SessionScoreResponse(BaseModel):
    accuracyScore: float
    rangeOfMotionScore: float
    stabilityScore: float
    overallScore: float
    validRepetitions: int
    invalidRepetitions: int


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------

class SessionScorer:
    """Derives session-level scores from raw per-rep telemetry."""

    @staticmethod
    def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
        return max(low, min(high, value))

    @classmethod
    def accuracy_score(cls, reps: List[Repetition]) -> float:
        """Share of valid reps, penalised by severity-weighted movement errors."""
        if not reps:
            return 0.0

        valid_ratio = sum(1 for r in reps if r.isCorrect) / len(reps)
        base = valid_ratio * 100.0

        penalty = 0.0
        for rep in reps:
            for err in rep.movementErrors:
                penalty += SEVERITY_WEIGHT[err.severity]
        penalty_per_rep = penalty / len(reps)

        return cls._clamp(base - penalty_per_rep)

    @classmethod
    def range_of_motion_score(cls, reps: List[Repetition]) -> float:
        """Mean achieved angular sweep per rep, normalised against the best
        sweep observed in the session."""
        sweeps: List[float] = []
        for rep in reps:
            angles = [r.angle for r in rep.jointAngleReadings]
            if len(angles) >= 2:
                sweeps.append(max(angles) - min(angles))

        if not sweeps:
            return 0.0

        best = max(sweeps)
        if best <= 0.0:
            return 0.0

        mean_ratio = sum(s / best for s in sweeps) / len(sweeps)
        return cls._clamp(round(mean_ratio * 100.0, 1))

    @classmethod
    def stability_score(cls, reps: List[Repetition]) -> float:
        """Measures movement smoothness via the dispersion of frame-to-frame
        angular steps. Controlled motion produces evenly sized steps; jerky or
        trembling motion produces erratic ones. Sampling-rate independent."""
        scores: List[float] = []
        for rep in reps:
            angles = [r.angle for r in rep.jointAngleReadings]
            if len(angles) < 3:
                continue

            velocities = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
            mean_speed = sum(abs(v) for v in velocities) / len(velocities)
            if mean_speed <= 1e-6:
                scores.append(100.0)
                continue

            # Acceleration magnitude (jerk proxy), normalised by mean speed so
            # the metric is independent of sampling rate and rep tempo.
            accels = [abs(velocities[i + 1] - velocities[i]) for i in range(len(velocities) - 1)]
            if not accels:
                scores.append(100.0)
                continue

            normalised_jerk = (sum(accels) / len(accels)) / mean_speed
            # Smooth motion keeps direction and pace: jerk ratio near 0.
            # A single direction reversal per rep is expected and tolerated.
            scores.append(cls._clamp(100.0 - (normalised_jerk * 60.0)))

        if not scores:
            return 100.0 if reps else 0.0

        return cls._clamp(round(sum(scores) / len(scores), 1))

    @classmethod
    def score(cls, telemetry: SessionTelemetry) -> SessionScoreResponse:
        reps = telemetry.reps
        valid = sum(1 for r in reps if r.isCorrect)
        invalid = len(reps) - valid

        accuracy = round(cls.accuracy_score(reps), 1)
        rom = round(cls.range_of_motion_score(reps), 1)
        stability = round(cls.stability_score(reps), 1)

        overall = round(
            accuracy * WEIGHT_ACCURACY
            + rom * WEIGHT_ROM
            + stability * WEIGHT_STABILITY,
            1,
        )

        return SessionScoreResponse(
            accuracyScore=accuracy,
            rangeOfMotionScore=rom,
            stabilityScore=stability,
            overallScore=overall,
            validRepetitions=valid,
            invalidRepetitions=invalid,
        )


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Kinova Physiotherapy API",
    version="1.0.0",
    description="Ingests CV engine telemetry and returns session scoring.",
)

# In-memory store; replace with the persistent datastore in production.
_SESSION_RESULTS: Dict[str, SessionScoreResponse] = {}


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok"}


@app.post(
    "/api/v1/Session/{id}/complete",
    response_model=SessionScoreResponse,
    tags=["session"],
)
def complete_session(
    telemetry: SessionTelemetry,
    id: str = Path(..., min_length=1, max_length=128),
) -> SessionScoreResponse:
    """Receive end-of-session telemetry and return computed scores."""
    if not telemetry.reps:
        raise HTTPException(status_code=422, detail="Session contains no repetitions.")

    result = SessionScorer.score(telemetry)
    _SESSION_RESULTS[id] = result

    logger.info(
        "Session %s scored: overall=%.1f valid=%d invalid=%d",
        id, result.overallScore, result.validRepetitions, result.invalidRepetitions,
    )
    return result


@app.get(
    "/api/v1/Session/{id}/result",
    response_model=SessionScoreResponse,
    tags=["session"],
)
def get_session_result(id: str = Path(..., min_length=1, max_length=128)) -> SessionScoreResponse:
    result = _SESSION_RESULTS.get(id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No result for session '{id}'.")
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)

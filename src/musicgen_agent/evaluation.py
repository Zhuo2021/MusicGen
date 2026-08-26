from __future__ import annotations

from dataclasses import dataclass

from musicgen_agent.theory import extract_features


@dataclass(frozen=True)
class EvaluationReport:
    note_count: int
    pitch_range: int
    average_interval: float
    density: float
    key: str
    mode: str
    repetition_rate: float
    leap_rate: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "note_count": self.note_count,
            "pitch_range": self.pitch_range,
            "average_interval": round(self.average_interval, 3),
            "density": round(self.density, 3),
            "key": self.key,
            "mode": self.mode,
            "repetition_rate": round(self.repetition_rate, 3),
            "leap_rate": round(self.leap_rate, 3),
        }


def evaluate(pitches: tuple[int, ...], durations: tuple[float, ...]) -> EvaluationReport:
    features = extract_features(pitches, durations)
    intervals = [abs(b - a) for a, b in zip(pitches, pitches[1:])]
    repetition_rate = sum(1 for interval in intervals if interval == 0) / max(len(intervals), 1)
    leap_rate = sum(1 for interval in intervals if interval >= 7) / max(len(intervals), 1)
    return EvaluationReport(
        note_count=len(pitches),
        pitch_range=features.pitch_range,
        average_interval=features.average_interval,
        density=features.density,
        key=features.key,
        mode=features.mode,
        repetition_rate=repetition_rate,
        leap_rate=leap_rate,
    )

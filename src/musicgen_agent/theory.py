from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import sqrt
from statistics import mean

PITCH_CLASS_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MAJOR_PROFILE = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
MINOR_PROFILE = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)


@dataclass(frozen=True)
class PhraseFeatures:
    contour: tuple[int, ...]
    interval_histogram: tuple[float, ...]
    pitch_class_histogram: tuple[float, ...]
    average_interval: float
    pitch_range: int
    density: float
    key: str
    mode: str


def note_name(pitch: int) -> str:
    octave = pitch // 12 - 1
    return f"{PITCH_CLASS_NAMES[pitch % 12]}{octave}"


def parse_note_token(token: str) -> int:
    token = token.strip()
    if "." in token:
        token = token.split(".")[0]
    if token.isdigit() or (token.startswith("-") and token[1:].isdigit()):
        return int(token)
    if len(token) >= 3 and token[-2] == "-":
        token = token[:-2] + "b" + token[-1]
    names = {
        "C": 0,
        "C#": 1,
        "Db": 1,
        "D": 2,
        "D#": 3,
        "Eb": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "Gb": 6,
        "G": 7,
        "G#": 8,
        "Ab": 8,
        "A": 9,
        "A#": 10,
        "Bb": 10,
        "B": 11,
    }
    name = token[:-1]
    octave = int(token[-1])
    return (octave + 1) * 12 + names[name]


def normalize(values: list[float]) -> tuple[float, ...]:
    total = sum(values)
    if total == 0:
        return tuple(0.0 for _ in values)
    return tuple(v / total for v in values)


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def detect_key(pitches: tuple[int, ...]) -> tuple[str, str]:
    hist = [0.0] * 12
    for pitch in pitches:
        hist[pitch % 12] += 1
    profile = normalize(hist)
    best_score = -1.0
    best = ("C", "major")
    for root in range(12):
        major = tuple(MAJOR_PROFILE[(i - root) % 12] for i in range(12))
        minor = tuple(MINOR_PROFILE[(i - root) % 12] for i in range(12))
        major_score = cosine(profile, normalize(list(major)))
        minor_score = cosine(profile, normalize(list(minor)))
        if major_score > best_score:
            best_score = major_score
            best = (PITCH_CLASS_NAMES[root], "major")
        if minor_score > best_score:
            best_score = minor_score
            best = (PITCH_CLASS_NAMES[root], "minor")
    return best


def extract_features(pitches: tuple[int, ...], durations: tuple[float, ...] | None = None) -> PhraseFeatures:
    if not pitches:
        return PhraseFeatures((), (0.0,) * 12, (0.0,) * 12, 0.0, 0, 0.0, "C", "major")
    intervals = [b - a for a, b in zip(pitches, pitches[1:])]
    contour = tuple(1 if i > 0 else -1 if i < 0 else 0 for i in intervals)
    interval_counts = [0.0] * 12
    for interval in intervals:
        interval_counts[abs(interval) % 12] += 1
    pitch_class_counts = [0.0] * 12
    for pitch in pitches:
        pitch_class_counts[pitch % 12] += 1
    key, mode = detect_key(pitches)
    duration_values = durations or tuple(1.0 for _ in pitches)
    return PhraseFeatures(
        contour=contour,
        interval_histogram=normalize(interval_counts),
        pitch_class_histogram=normalize(pitch_class_counts),
        average_interval=mean(abs(i) for i in intervals) if intervals else 0.0,
        pitch_range=max(pitches) - min(pitches),
        density=len(pitches) / max(sum(duration_values), 0.001),
        key=key,
        mode=mode,
    )


def contour_similarity(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    length = min(len(a), len(b))
    if length == 0:
        return 0.0
    return sum(1 for i in range(length) if a[i] == b[i]) / length


def most_common_step(pitches: tuple[int, ...]) -> int:
    intervals = [b - a for a, b in zip(pitches, pitches[1:]) if b != a]
    if not intervals:
        return 2
    return Counter(intervals).most_common(1)[0][0]

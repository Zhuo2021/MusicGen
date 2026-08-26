from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class NoteEvent:
    pitch: int
    start: float
    duration: float
    velocity: int = 84

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass(frozen=True)
class Phrase:
    notes: tuple[NoteEvent, ...]
    source: str
    start_index: int
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def pitches(self) -> tuple[int, ...]:
        return tuple(note.pitch for note in self.notes)

    @property
    def durations(self) -> tuple[float, ...]:
        return tuple(note.duration for note in self.notes)


@dataclass(frozen=True)
class RetrievalHit:
    phrase: Phrase
    score: float
    reason: str


@dataclass(frozen=True)
class GenerationRequest:
    seed: tuple[int, ...]
    target_notes: int = 24
    emotion: str | None = None
    temperature: float = 0.45
    output: Path | None = None

    @classmethod
    def from_notes(
        cls,
        notes: Iterable[int],
        target_notes: int = 24,
        emotion: str | None = None,
        temperature: float = 0.45,
        output: Path | None = None,
    ) -> "GenerationRequest":
        return cls(
            seed=tuple(notes),
            target_notes=target_notes,
            emotion=emotion,
            temperature=temperature,
            output=output,
        )

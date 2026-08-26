from __future__ import annotations

from dataclasses import dataclass
from random import Random

from musicgen_agent.evaluation import EvaluationReport, evaluate
from musicgen_agent.midi_io import write_midi
from musicgen_agent.models import GenerationRequest, NoteEvent, RetrievalHit
from musicgen_agent.retrieval import PhraseRetriever
from musicgen_agent.theory import most_common_step, note_name


@dataclass(frozen=True)
class GenerationResult:
    seed: tuple[int, ...]
    continuation: tuple[int, ...]
    notes: tuple[NoteEvent, ...]
    retrieval_hits: tuple[RetrievalHit, ...]
    report: EvaluationReport

    @property
    def full_pitches(self) -> tuple[int, ...]:
        return self.seed + self.continuation

    def note_names(self) -> list[str]:
        return [note_name(pitch) for pitch in self.full_pitches]


class MusicGenerationAgent:
    """A deterministic, interview-friendly RAG agent for symbolic melody continuation."""

    def __init__(self, retriever: PhraseRetriever, seed: int = 7) -> None:
        self.retriever = retriever
        self.random = Random(seed)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if len(request.seed) < 2:
            raise ValueError("At least two seed notes are required.")
        hits = tuple(self.retriever.search(request.seed, request.emotion, k=8))
        continuation = self._compose_continuation(request, hits)
        events = tuple(self._render_events(request.seed + continuation, hits))
        report = evaluate(tuple(note.pitch for note in events), tuple(note.duration for note in events))
        if request.output:
            write_midi(request.output, list(events))
        return GenerationResult(request.seed, continuation, events, hits, report)

    def _compose_continuation(self, request: GenerationRequest, hits: tuple[RetrievalHit, ...]) -> tuple[int, ...]:
        seed = request.seed
        last_pitch = seed[-1]
        anchor_intervals = [b - a for a, b in zip(seed, seed[1:])]
        fallback_step = most_common_step(seed)
        candidate_intervals: list[int] = []

        for hit in hits:
            phrase = hit.phrase.pitches
            intervals = [b - a for a, b in zip(phrase, phrase[1:])]
            if not intervals:
                continue
            candidate_intervals.extend(intervals[: request.target_notes])

        if not candidate_intervals:
            candidate_intervals = anchor_intervals or [fallback_step]

        generated: list[int] = []
        previous = last_pitch
        for index in range(request.target_notes):
            interval = candidate_intervals[index % len(candidate_intervals)]
            if index % 7 == 6 and anchor_intervals:
                interval = -anchor_intervals[index % len(anchor_intervals)]
            interval = self._shape_interval(interval, request.temperature)
            next_pitch = self._keep_playable(previous + interval, center=sum(seed) / len(seed))
            generated.append(next_pitch)
            previous = next_pitch
        return tuple(generated)

    def _shape_interval(self, interval: int, temperature: float) -> int:
        if abs(interval) <= 7:
            return interval
        direction = 1 if interval > 0 else -1
        softened = 7 + int((abs(interval) - 7) * min(max(temperature, 0.0), 1.0))
        return direction * min(12, softened)

    def _keep_playable(self, pitch: int, center: float) -> int:
        while pitch < 48:
            pitch += 12
        while pitch > 84:
            pitch -= 12
        if abs(pitch - center) > 18:
            pitch += -12 if pitch > center else 12
        return max(45, min(88, pitch))

    def _render_events(self, pitches: tuple[int, ...], hits: tuple[RetrievalHit, ...]) -> list[NoteEvent]:
        durations = self._borrow_rhythm(hits)
        notes: list[NoteEvent] = []
        cursor = 0.0
        for index, pitch in enumerate(pitches):
            duration = durations[index % len(durations)]
            notes.append(NoteEvent(pitch=pitch, start=round(cursor, 4), duration=duration))
            cursor += duration
        return notes

    def _borrow_rhythm(self, hits: tuple[RetrievalHit, ...]) -> tuple[float, ...]:
        for hit in hits:
            durations = tuple(d for d in hit.phrase.durations if 0.1 <= d <= 4.0)
            if durations:
                return durations
        return (0.5, 0.5, 1.0, 0.5, 0.5, 1.0)

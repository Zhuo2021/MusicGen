from __future__ import annotations

from pathlib import Path

from musicgen_agent.midi_io import read_midi
from musicgen_agent.models import NoteEvent, Phrase


def load_midi_corpus(
    root: Path,
    phrase_length: int = 16,
    hop: int = 8,
    limit: int | None = None,
    max_phrases_per_file: int = 8,
) -> list[Phrase]:
    phrases: list[Phrase] = []
    files = sorted(root.rglob("*.mid"))
    for midi_path in files:
        try:
            notes = _monophonic(read_midi(midi_path))
        except Exception:
            continue
        if len(notes) < phrase_length:
            continue
        file_count = 0
        for start in range(0, len(notes) - phrase_length + 1, hop):
            phrase_notes = tuple(_rebase(notes[start : start + phrase_length]))
            phrases.append(
                Phrase(
                    notes=phrase_notes,
                    source=str(midi_path.relative_to(root)),
                    start_index=start,
                    tags=_tags_from_path(midi_path),
                )
            )
            file_count += 1
            if limit and len(phrases) >= limit:
                return phrases
            if max_phrases_per_file and file_count >= max_phrases_per_file:
                break
    return phrases


def _monophonic(notes: list[NoteEvent]) -> list[NoteEvent]:
    by_start: dict[float, NoteEvent] = {}
    for note in notes:
        current = by_start.get(note.start)
        if current is None or note.pitch > current.pitch:
            by_start[note.start] = note
    return [by_start[key] for key in sorted(by_start)]


def _rebase(notes: list[NoteEvent]) -> list[NoteEvent]:
    if not notes:
        return []
    base = notes[0].start
    return [
        NoteEvent(
            pitch=note.pitch,
            start=round(note.start - base, 4),
            duration=round(note.duration, 4),
            velocity=note.velocity,
        )
        for note in notes
    ]


def _tags_from_path(path: Path) -> tuple[str, ...]:
    parts = {part.lower() for part in path.parts}
    tags: list[str] = []
    for tag in ("q1", "q2", "q3", "q4", "happy", "sad", "relaxed", "angry"):
        if tag in parts or path.name.lower().startswith(tag):
            tags.append(tag)
    return tuple(tags)

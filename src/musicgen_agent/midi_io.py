from __future__ import annotations

from pathlib import Path
from struct import pack, unpack

from musicgen_agent.models import NoteEvent


def _read_varlen(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    while True:
        byte = data[offset]
        offset += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, offset


def _write_varlen(value: int) -> bytes:
    buffer = value & 0x7F
    value >>= 7
    while value:
        buffer <<= 8
        buffer |= (value & 0x7F) | 0x80
        value >>= 7
    out = bytearray()
    while True:
        out.append(buffer & 0xFF)
        if buffer & 0x80:
            buffer >>= 8
        else:
            break
    return bytes(out)


def read_midi(path: Path) -> list[NoteEvent]:
    data = path.read_bytes()
    if data[:4] != b"MThd":
        raise ValueError(f"{path} is not a Standard MIDI file")
    header_length = unpack(">I", data[4:8])[0]
    _, _, ticks_per_beat = unpack(">HHH", data[8:14])
    offset = 8 + header_length
    notes: list[NoteEvent] = []

    while offset < len(data):
        if data[offset : offset + 4] != b"MTrk":
            break
        track_length = unpack(">I", data[offset + 4 : offset + 8])[0]
        track = data[offset + 8 : offset + 8 + track_length]
        offset += 8 + track_length
        tick = 0
        i = 0
        running_status = None
        active: dict[tuple[int, int], tuple[int, int]] = {}

        while i < len(track):
            delta, i = _read_varlen(track, i)
            tick += delta
            status = track[i]
            if status < 0x80:
                if running_status is None:
                    break
                status = running_status
            else:
                i += 1
                running_status = status

            event_type = status & 0xF0
            channel = status & 0x0F
            if status == 0xFF:
                meta_type = track[i]
                i += 1
                length, i = _read_varlen(track, i)
                i += length
                if meta_type == 0x2F:
                    break
            elif status in (0xF0, 0xF7):
                length, i = _read_varlen(track, i)
                i += length
            elif event_type in (0x80, 0x90):
                pitch = track[i]
                velocity = track[i + 1]
                i += 2
                key = (channel, pitch)
                if event_type == 0x90 and velocity > 0:
                    active[key] = (tick, velocity)
                elif key in active:
                    start_tick, start_velocity = active.pop(key)
                    duration = max(tick - start_tick, 1) / ticks_per_beat
                    notes.append(
                        NoteEvent(
                            pitch=pitch,
                            start=start_tick / ticks_per_beat,
                            duration=duration,
                            velocity=start_velocity,
                        )
                    )
            elif event_type in (0xA0, 0xB0, 0xE0):
                i += 2
            elif event_type in (0xC0, 0xD0):
                i += 1
            else:
                break

    return sorted(notes, key=lambda note: (note.start, note.pitch))


def write_midi(path: Path, notes: list[NoteEvent], tempo_bpm: int = 96, ticks_per_beat: int = 480) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    events: list[tuple[int, bytes]] = []
    tempo = int(60_000_000 / tempo_bpm)
    events.append((0, b"\xff\x51\x03" + tempo.to_bytes(3, "big")))
    events.append((0, b"\xc0\x00"))
    for note in notes:
        start = int(round(note.start * ticks_per_beat))
        end = int(round(note.end * ticks_per_beat))
        velocity = max(1, min(127, note.velocity))
        pitch = max(0, min(127, note.pitch))
        events.append((start, bytes([0x90, pitch, velocity])))
        events.append((max(end, start + 1), bytes([0x80, pitch, 0])))
    events.sort(key=lambda item: (item[0], item[1][0] == 0x80))

    track = bytearray()
    last_tick = 0
    for tick, payload in events:
        track.extend(_write_varlen(tick - last_tick))
        track.extend(payload)
        last_tick = tick
    track.extend(b"\x00\xff\x2f\x00")

    header = b"MThd" + pack(">IHHH", 6, 0, 1, ticks_per_beat)
    chunk = b"MTrk" + pack(">I", len(track)) + bytes(track)
    path.write_bytes(header + chunk)

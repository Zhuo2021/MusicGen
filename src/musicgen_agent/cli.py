from __future__ import annotations

import argparse
import json
from pathlib import Path

from musicgen_agent.agent import MusicGenerationAgent
from musicgen_agent.corpus import load_midi_corpus
from musicgen_agent.models import GenerationRequest, NoteEvent, Phrase
from musicgen_agent.retrieval import PhraseRetriever
from musicgen_agent.theory import parse_note_token


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a MIDI continuation with a RAG music agent.")
    parser.add_argument("--seed", default="G4,Eb4,D4,G3", help="Comma-separated note names or MIDI numbers.")
    parser.add_argument("--corpus", default="lstm生成音乐/midi_songs", help="Folder containing MIDI files.")
    parser.add_argument("--output", default="outputs/generated.mid", help="Where to write the generated MIDI.")
    parser.add_argument("--target-notes", type=int, default=24, help="How many continuation notes to generate.")
    parser.add_argument("--emotion", default=None, help="Optional retrieval tag, e.g. q1/q2/q3/q4.")
    parser.add_argument("--limit", type=int, default=300, help="Max indexed phrases for fast demos.")
    parser.add_argument("--max-phrases-per-file", type=int, default=8, help="Keep retrieval corpus diverse.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed = tuple(parse_note_token(token) for token in args.seed.split(",") if token.strip())
    corpus_root = Path(args.corpus)
    phrases = load_midi_corpus(corpus_root, limit=args.limit, max_phrases_per_file=args.max_phrases_per_file)
    if not phrases:
        phrases = _fallback_phrases()

    request = GenerationRequest.from_notes(
        seed,
        target_notes=args.target_notes,
        emotion=args.emotion,
        output=Path(args.output),
    )
    agent = MusicGenerationAgent(PhraseRetriever(phrases))
    result = agent.generate(request)

    payload = {
        "seed": list(result.seed),
        "continuation": list(result.continuation),
        "note_names": result.note_names(),
        "output": str(request.output),
        "top_retrieval": [
            {"source": hit.phrase.source, "score": hit.score, "reason": hit.reason}
            for hit in result.retrieval_hits[:5]
        ],
        "evaluation": result.report.as_dict(),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_human(payload)


def _print_human(payload: dict) -> None:
    print("MusicGen Agent")
    print(f"Output MIDI: {payload['output']}")
    print("Generated notes:")
    print(", ".join(payload["note_names"]))
    print("\nTop retrieval hits:")
    for hit in payload["top_retrieval"]:
        print(f"- {hit['source']}  score={hit['score']}  ({hit['reason']})")
    print("\nEvaluation:")
    for key, value in payload["evaluation"].items():
        print(f"- {key}: {value}")


def _fallback_phrases() -> list[Phrase]:
    base = [60, 62, 64, 67, 69, 67, 64, 62, 60, 55, 60, 64, 67, 72, 71, 67]
    notes = tuple(NoteEvent(pitch=pitch, start=i * 0.5, duration=0.5) for i, pitch in enumerate(base))
    return [Phrase(notes=notes, source="built_in/c_major_arc", start_index=0, tags=("q1",))]


if __name__ == "__main__":
    main()

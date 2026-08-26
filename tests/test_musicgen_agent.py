import unittest
from pathlib import Path

from musicgen_agent.agent import MusicGenerationAgent
from musicgen_agent.corpus import load_midi_corpus
from musicgen_agent.midi_io import read_midi, write_midi
from musicgen_agent.models import GenerationRequest, NoteEvent, Phrase
from musicgen_agent.retrieval import PhraseRetriever


class MusicGenAgentTest(unittest.TestCase):
    def test_midi_round_trip(self) -> None:
        target = Path(self._tmpdir.name) / "roundtrip.mid"
        notes = [
            NoteEvent(60, 0.0, 0.5),
            NoteEvent(64, 0.5, 0.5),
            NoteEvent(67, 1.0, 1.0),
        ]
        write_midi(target, notes)

        parsed = read_midi(target)

        self.assertEqual([note.pitch for note in parsed], [60, 64, 67])
        self.assertEqual(parsed[-1].duration, 1.0)

    def test_agent_generates_requested_length(self) -> None:
        phrase_notes = tuple(NoteEvent(60 + i % 5, i * 0.5, 0.5) for i in range(16))
        phrase = Phrase(phrase_notes, "fixture.mid", 0)
        agent = MusicGenerationAgent(PhraseRetriever([phrase]))

        result = agent.generate(GenerationRequest.from_notes([60, 62, 64], target_notes=8))

        self.assertEqual(len(result.continuation), 8)
        self.assertEqual(len(result.notes), 11)
        self.assertEqual(result.retrieval_hits[0].phrase.source, "fixture.mid")

    def test_load_midi_corpus_extracts_phrases(self) -> None:
        midi_path = Path(self._tmpdir.name) / "song.mid"
        notes = [NoteEvent(60 + i % 8, i * 0.5, 0.5) for i in range(24)]
        write_midi(midi_path, notes)

        phrases = load_midi_corpus(Path(self._tmpdir.name), phrase_length=8, hop=4)

        self.assertGreaterEqual(len(phrases), 4)
        self.assertEqual(phrases[0].source, "song.mid")

    def setUp(self) -> None:
        import tempfile

        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()


if __name__ == "__main__":
    unittest.main()

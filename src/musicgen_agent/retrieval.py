from __future__ import annotations

from musicgen_agent.models import Phrase, RetrievalHit
from musicgen_agent.theory import contour_similarity, cosine, extract_features


class PhraseRetriever:
    def __init__(self, phrases: list[Phrase]) -> None:
        self._items = [(phrase, extract_features(phrase.pitches, phrase.durations)) for phrase in phrases]

    def search(self, seed: tuple[int, ...], emotion: str | None = None, k: int = 8) -> list[RetrievalHit]:
        seed_features = extract_features(seed)
        hits: list[RetrievalHit] = []
        for phrase, features in self._items:
            interval_score = cosine(seed_features.interval_histogram, features.interval_histogram)
            pc_score = cosine(seed_features.pitch_class_histogram, features.pitch_class_histogram)
            contour_score = contour_similarity(seed_features.contour, features.contour)
            range_penalty = min(abs(seed_features.pitch_range - features.pitch_range) / 24.0, 1.0)
            tag_bonus = 0.05 if emotion and emotion.lower() in phrase.tags else 0.0
            score = 0.44 * interval_score + 0.28 * contour_score + 0.18 * pc_score + 0.10 * (1 - range_penalty)
            score += tag_bonus
            hits.append(
                RetrievalHit(
                    phrase=phrase,
                    score=round(score, 4),
                    reason=f"interval={interval_score:.2f}, contour={contour_score:.2f}, key={features.key} {features.mode}",
                )
            )
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:k]

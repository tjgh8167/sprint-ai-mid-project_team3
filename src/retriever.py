import math
import re
from collections import Counter
from dataclasses import dataclass


@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict
    score: float


_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


def _cosine_similarity(query_counts: Counter, doc_counts: Counter) -> float:
    common = set(query_counts) & set(doc_counts)
    numerator = sum(query_counts[token] * doc_counts[token] for token in common)
    query_norm = math.sqrt(sum(value * value for value in query_counts.values()))
    doc_norm = math.sqrt(sum(value * value for value in doc_counts.values()))

    if query_norm == 0 or doc_norm == 0:
        return 0.0
    return numerator / (query_norm * doc_norm)


def _metadata_matches(metadata: dict, filters: dict | None) -> bool:
    if not filters:
        return True

    for key, expected in filters.items():
        actual = str(metadata.get(key, "")).lower()
        if str(expected).lower() not in actual:
            return False
    return True


class SimpleRetriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self._chunk_vectors = [Counter(tokenize(chunk["text"])) for chunk in chunks]

    def search(self, query: str, top_k: int = 3, filters: dict | None = None) -> list[SearchResult]:
        query_vector = Counter(tokenize(query))
        results = []

        for chunk, chunk_vector in zip(self.chunks, self._chunk_vectors):
            metadata = chunk.get("metadata", {})
            if not _metadata_matches(metadata, filters):
                continue

            score = _cosine_similarity(query_vector, chunk_vector)
            if score <= 0:
                continue

            results.append(
                SearchResult(
                    chunk_id=chunk["chunk_id"],
                    doc_id=chunk["doc_id"],
                    text=chunk["text"],
                    metadata=metadata,
                    score=round(score, 4),
                )
            )

        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]

from src.retriever import SearchResult


class LocalChromaRetriever:
    """Retrieval 2가 구현할 HuggingFace 임베딩 + Chroma 검색기 계약입니다."""

    def __init__(self, chunks: list[dict], config: dict):
        self.chunks = chunks
        self.config = config

    def search(self, query: str, top_k: int = 3, filters: dict | None = None) -> list[SearchResult]:
        raise NotImplementedError(
            "Retrieval 2 담당자가 HuggingFace 임베딩 생성, Chroma 저장·조회, "
            "SearchResult 변환을 구현해야 합니다."
        )

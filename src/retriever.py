import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

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
        #불필요한 org_name 매핑 로직 제거, 바로 key(agency) 값으로 타겟팅 처리
        actual = str(metadata.get(key, "")).lower()
        if str(expected).lower() not in actual:
            return False
    return True


class SimpleRetriever:
    """메모리 기반 단순 토큰 카운트 유사도 검색기 (기존 유지)"""
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


class LocalChromaRetriever:
    """Retrieval 2: HuggingFace 임베딩 + Chroma 검색기 구현체 (Read-only 최적화 버전)"""
    
    def __init__(self, config: Dict[str, Any]):
        """이미 단 한 번 빌드되어 저장된 Vector DB를 로드만 수행하여 초기화 속도 극대화"""
        self.config = config
        
        model_name = self.config.get("embedding", "dragonkue/BGE-m3-ko")
        cache_dir = self.config.get("cache_path")
        device = self.config.get("device", "cpu")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=cache_dir,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )

        # 기본 경로를 로컬 개발용이 아닌 지정하신 공유 벡터 DB 경로로 변경
        persist_directory = self.config.get("persist_directory", "/data/processed/vector_db/local")
        collection_name = self.config.get("collection_name", "bidmate_localgit")

        # 실시간 적재(add_documents) 구문을 원천 차단하여 검색 지연 최소화 및 무결성 확보
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

    def search(self, query: str, top_k: Optional[int] = None, filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """SimpleRetriever와 인터페이스 규격을 일치시킨 공통 메타데이터 필터 고속 Chroma 검색"""
        k = top_k or self.config.get("top_k", 4)
        
        # Chroma가 이해할 수 있는 형태의 메타데이터 필터 딕셔너리 동적 빌드
        search_filter = None
        if filters:
            filter_conditions = []
            for key, val in filters.items():
                if val:
                    # Chroma 내 저장된 메타데이터 규격에 맞춰 agency 필터링 연동
                    filter_conditions.append({key: val})
            
            if len(filter_conditions) == 1:
                search_filter = filter_conditions[0]
            elif len(filter_conditions) > 1:
                search_filter = {"$and": filter_conditions}

        try:
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query, k=k, filter=search_filter
            )
        except Exception:
            # 예외 발생 시 필터 없이 서치하도록 안정성 마진 확보
            docs_with_scores = self.vector_store.similarity_search_with_score(query=query, k=k)

        results = []
        for doc, score in docs_with_scores:
            results.append(
                SearchResult(
                    chunk_id=doc.metadata.get("chunk_id", "N/A"),
                    doc_id=doc.metadata.get("doc_id", "N/A"),
                    text=doc.page_content,
                    metadata=doc.metadata,
                    score=float(score)
                )
            )
        return results
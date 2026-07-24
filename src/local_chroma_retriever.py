import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# [수정 반영] 프로젝트 루트 경로 절대경로 정의
PROJECT_ROOT = Path("/home/taehoon/sprint-ai-mid-project_team3")


# [OpenAI 버전과 인터페이스 규격을 일치시킨 SearchResult dataclass]
@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any]
    score: float


class LocalChromaRetriever:
    """[fetch_k, top_k, 부분 일치 필터링이 적용된 Local Retriever]"""

    def __init__(self, config: Dict[str, Any], chunks: Optional[List[Dict[str, Any]]] = None, **kwargs):
        # 1. Config 로드 및 profile 설정 유연 처리
        retrieval_cfg = config.get("retrieval", config)
        profiles = retrieval_cfg.get("profiles", {})
        local_cfg = profiles.get("local", retrieval_cfg)

        # 외부에서 주입된 shared client 수신
        self.client = config.get("client") or kwargs.get("client")

        # 2. 주요 설정값 매핑 (기본 DB 경로: /data/processed/vector_db/local)
        self.embedding_model_name = local_cfg.get("embedding_model", "dragonkue/BGE-m3-ko")
        self.persist_directory = local_cfg.get("persist_directory", "/data/processed/vector_db/local")
        self.collection_name = local_cfg.get("collection_name", "bidmate_local")
        self.cache_dir = local_cfg.get("cache_path", "model_cache")
        self.device = local_cfg.get("device", "cpu")

        # top_k 및 fetch_k 연동
        self.default_top_k = retrieval_cfg.get("top_k", 5)
        self.fetch_k = local_cfg.get("fetch_k", 100)  # 기본 100개 후보군

        # [수정 반영] 상대 경로 지정 시 PROJECT_ROOT 기준으로 절대 경로 변환
        if not os.path.isabs(self.persist_directory):
            self.persist_directory = str(PROJECT_ROOT / self.persist_directory)

        # 3. Embeddings & Chroma 연결
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            cache_folder=self.cache_dir,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )

        chroma_kwargs = {
            "collection_name": self.collection_name,
            "embedding_function": self.embeddings,
            "persist_directory": self.persist_directory,
            "collection_metadata": {"hnsw:space": "cosine"},
        }
        if self.client is not None:
            chroma_kwargs["client"] = self.client

        self.vectorstore = Chroma(**chroma_kwargs)

    def normalize_text(self, text: str) -> str:
        """[OpenAI와 동일한 부분 일치 필터링용 텍스트 정규화]"""
        if not text:
            return ""
        text = unicodedata.normalize("NFC", str(text))
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        [후보군(fetch_k) 사전 검색 후, 텍스트 정규화 기반 메타데이터 필터링 적용]
        """
        search_k = top_k if top_k is not None else self.default_top_k
        active_filters = filters or {}

        # 후보군 수(fetch_k)는 최소 search_k 이상을 보장
        fetch_k_count = max(search_k, self.fetch_k)

        try:
            # 1. fetch_k 개수만큼 Vector DB에서 1차 후보군 검색
            docs_and_scores = self.vectorstore.similarity_search_with_score(
                query=query,
                k=fetch_k_count,
                filter=None,
            )
        except Exception as e:
            print(f"[Error] 검색 처리 중 오류 발생: {e}")
            raise e

        if not docs_and_scores:
            return []

        results = []
        # 2. 인메모리 부분 일치 필터링 및 유사도 점수 산출
        for doc, score in docs_and_scores:
            is_match = True

            if active_filters:
                for key, value in active_filters.items():
                    if value is None or str(value).strip() == "":
                        continue

                    clean_filter_val = self.normalize_text(str(value))
                    clean_metadata_val = self.normalize_text(str(doc.metadata.get(key, "")))

                    # 부분 일치 조건 검사
                    if clean_filter_val not in clean_metadata_val:
                        is_match = False
                        break

            if is_match:
                # [수정 반영] Cosine Distance -> 유사도 점수(Similarity Score) 변환 공식 보완
                # 거리(score)가 크더라도 항상 거리가 멀어질수록 점수가 낮아지도록 (1 / (1 + distance)) 방식 적용
                distance = float(score)
                similarity_score = 1.0 / (1.0 + distance)

                results.append(
                    SearchResult(
                        chunk_id=str(doc.metadata.get("chunk_id", "")),
                        doc_id=str(doc.metadata.get("doc_id", "")),
                        text=doc.page_content,
                        metadata=doc.metadata,
                        score=similarity_score,
                    )
                )

                # 원하는 top_k 수를 채우면 바로 반환
                if len(results) >= search_k:
                    break

        if not results and active_filters:
            print(f"안내: '{active_filters}' 필터 조건에 일치하는 문서가 후보군({fetch_k_count}개) 내에 없습니다.")

        return results


# Simple Test Code
if __name__ == "__main__":
    # [수정 반영] configs/config.yaml -> configs/default.yaml -> config.yaml 순차 로드
    config_file = PROJECT_ROOT / "configs" / "config.yaml"
    if not config_file.exists():
        config_file = PROJECT_ROOT / "configs" / "default.yaml"
    if not config_file.exists():
        config_file = PROJECT_ROOT / "config.yaml"

    if config_file.exists():
        print(f"[Info] 로드된 Config 파일: {config_file}")
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        retriever = LocalChromaRetriever(config=cfg)
        test_query = "입찰"
        test_filters = {"title": "공고"}

        print(f"\n[검색 테스트] Query: '{test_query}', Filter: {test_filters}")
        results = retriever.search(query=test_query, top_k=5, filters=test_filters)

        for idx, res in enumerate(results, 1):
            print(f"{idx}. [Score: {res.score:.4f}] {res.metadata.get('title')} - {res.text[:40]}...")
    else:
        print(f"[Warning] 설정 파일을 찾을 수 없어 테스트를 스킵합니다.")
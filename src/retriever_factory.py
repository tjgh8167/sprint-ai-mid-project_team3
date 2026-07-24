요청하신 사용하지 않는 조건문 블록(hasattr 검사 및 add_documents / index_chunks 실행 로직)을 깔끔하게 제거하였습니다.

💡 주요 수정 내역
불필요한 메소드 체크 로직 제거 ([수정 반영] 표시):
chunks 인수를 받아 인스턴스 생성 후 동적으로 문서를 추가하려던 비활성 코드를 제거했습니다.
이제 LocalChromaRetriever 생성 시 merged_local_config만 깔끔하게 전달한 뒤 바로 인스턴스를 반환합니다.

📝 전체 수정 코드 (retriever_factory.py)
Python
import os
from typing import List, Dict, Any, Optional
import chromadb

# 각 Retriever 구현체 임포트
from src.local_chroma_retriever import LocalChromaRetriever
from src.openai_chroma_retriever import OpenAIChromaRetriever
from src.retriever import SimpleRetriever


# -------------------------------------------------------------------------
# ChromaDB PersistentClient 전역 싱글톤 캐시
# - 동일 DB 저장 경로(persist_directory)에 대해 Client를 1개만 유지
# -------------------------------------------------------------------------
_CHROMA_CLIENT_CACHE: Dict[str, chromadb.PersistentClient] = {}

def get_shared_chroma_client(persist_directory: str) -> chromadb.PersistentClient:
    """DB 저장 경로별로 단 하나의 PersistentClient만 생성 및 공유하는 헬퍼 함수"""
    abs_path = os.path.abspath(persist_directory)
    if abs_path not in _CHROMA_CLIENT_CACHE:
        _CHROMA_CLIENT_CACHE[abs_path] = chromadb.PersistentClient(path=abs_path)
    return _CHROMA_CLIENT_CACHE[abs_path]


def create_retriever(
    chunks: List[Dict[str, Any]], 
    retrieval_config: Dict[str, Any], 
    profile: Optional[str] = None
):
    """
    설정 프로필(--profile local / openai / baseline)에 따라 
    알맞은 Retriever 인스턴스를 동적으로 생성하는 팩토리 함수
    """
    selected_profile = profile or retrieval_config.get("active_profile", "baseline")
    profiles = retrieval_config.get("profiles", {})

    # 1. Baseline 키워드/TF-IDF 검색기
    if selected_profile == "baseline":
        return SimpleRetriever(chunks)
        
    # 2. OpenAI 기반 Chroma VectorDB 검색기
    if selected_profile == "openai":
        openai_config = profiles.get("openai", {})
        merged_openai_config = {**retrieval_config, **openai_config}
        
        persist_dir = merged_openai_config.get("persist_directory", "./chroma_db")
        merged_openai_config["client"] = get_shared_chroma_client(persist_dir)
        
        return OpenAIChromaRetriever(
            config=merged_openai_config
        )
        
    # 3. HF 로컬 임베딩 + Chroma Retriever 인스턴스 생성
    if selected_profile == "local":
        local_config = profiles.get("local", {})
        merged_local_config = {**retrieval_config, **local_config}
        
        # OpenAI 규격과 동일하게 client를 config 딕셔너리에 주입
        persist_dir = merged_local_config.get("persist_directory", "/data/processed/vector_db/local")
        shared_client = get_shared_chroma_client(persist_dir)
        merged_local_config["client"] = shared_client
        
        # [수정 반영] 미사용 동적 메소드 검사(add_documents/index_chunks) 로직을 제거하고 단일 인스턴스 반환
        return LocalChromaRetriever(
            config=merged_local_config
        )

    # 4. 예외 처리
    raise ValueError(
        f"지원하지 않는 Retrieval 프로필입니다: '{selected_profile}'. "
        f"('local', 'openai', 'baseline' 중 하나를 선택해 주세요.)"
    )
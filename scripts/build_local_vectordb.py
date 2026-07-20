from typing import List, Dict, Any
import yaml
import json
import os

# -------------------------------------------------------------------------
# [수정 1] 구형 경로 수정 (최신 패키지 사용)
# deprecated 경고 및 모듈 참조 오류를 차단하기 위해 langchain_huggingface, langchain_chroma 사용
# -------------------------------------------------------------------------
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def build_vector_db(chunks: List[Dict[str, Any]], config: Dict[str, Any]):
    """청크 데이터를 받아 Chroma Vector DB를 생성하고 저장하는 함수"""
    
    # ---------------------------------------------------------------------
    # [수정 2] 공백 오타 수정 및 config 접근 로직 유연화
    # " embedding_model" 공백 제거 및 config["embedding_model"] 또는 config["embedding"]["model_name"] 지원
    # ---------------------------------------------------------------------
    model_name = config.get("embedding_model")
    if not model_name and isinstance(config.get("embedding"), dict):
        model_name = config.get("embedding", {}).get("model_name")
    if not model_name:
        model_name = "dragonkue/BGE-m3-ko"

    cache_dir = config.get("cache_path")
    device = config.get("device", "cpu")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=cache_dir,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    # ---------------------------------------------------------------------
    # [수정] 공유 경로 및 컬렉션명 고정
    # VM 용량 낭비를 막기 위해 공유 경로(/data/processed/vector_db/local)를 기본값으로 지정
    # ---------------------------------------------------------------------
    persist_directory = config.get("persist_directory", "/data/processed/vector_db/local")
    collection_name = config.get("collection_name", "bidmate_local")

    # Document 객체 변환
    documents = []
    for chunk in chunks:
        # -----------------------------------------------------------------
        # [수정 3] 회의록/README 규격에 맞춘 메타데이터 전체 컬럼 포함
        # 선별적 누락 없이 chunk의 모든 원본 메타데이터 항목을 다 담도록 수정
        # (chunk_id, doc_id, title, file_name, agency, project_name 등 포함)
        # -----------------------------------------------------------------
        metadata = {
            "chunk_id": chunk.get("chunk_id", ""),
            "doc_id": chunk.get("doc_id", ""),
            "title": chunk.get("title", ""),
            "file_name": chunk.get("file_name", ""),
            "agency": chunk.get("agency", ""),
            "project_name": chunk.get("project_name", "")
        }
        # 원본 chunk에 추가 메타데이터가 존재할 경우 함께 누락 없이 덤프
        for k, v in chunk.items():
            if k not in ["text"] and v is not None:
                metadata[k] = v

        documents.append(
            Document(
                page_content=chunk.get("text", ""),
                metadata=metadata
            )
        )

    # Chroma DB에 저장
    print("Embedding 및 DB 저장 시작...")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    
    # ---------------------------------------------------------------------
    # [수정 4] vector_store.persist() 완전 삭제
    # Chroma 0.4.x / langchain-chroma는 생성 시 자동 저장되므로 중복/deprecated된 persist() 호출 제거
    # ---------------------------------------------------------------------

    print(f"DB 구축 완료! (컬렉션: {collection_name}, 경로: {persist_directory})")
    return vector_store


# -------------------------------------------------------------------------
# [수정 5] 실행 코드(메인 함수 호출 로직) 완성
# 실제 처리된 청크 파일(JSON 등)을 읽어와 build_vector_db를 실제로 호출하도록 구현
# -------------------------------------------------------------------------
def main():
    # 1. 설정 파일(default.yaml) 로드
    config_path = "default.yaml"
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        print(f"Warning: {config_path} 파일을 찾을 수 없습니다. 기본 설정을 사용합니다.")

    # 2. 청크 데이터 로드 (실제 전처리된 청크 파일 경로 지정)
    chunks_path = config.get("chunks_path", "/data/processed/chunks.json")
    chunks = []
    
    if os.path.exists(chunks_path):
        print(f"청크 데이터 로드 중: {chunks_path}")
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    else:
        print(f"[Notice] {chunks_path} 파일이 없어 샘플 청크 데이터로 DB 구축을 시도합니다.")
        chunks = [
            {
                "chunk_id": "chk_001",
                "doc_id": "doc_001",
                "title": "2026년 공공데이터 구축 사업 제안요청서",
                "file_name": "RFP_2026.pdf",
                "agency": "한국지능정보사회진흥원",
                "project_name": "공공데이터 구축",
                "text": "본 사업은 공공데이터 활용을 극대화하기 위해..."
            }
        ]

    # 3. DB 구축 함수 실제 호출 실행
    if chunks:
        build_vector_db(chunks=chunks, config=config)
    else:
        print("구축할 청크 데이터가 없습니다.")


if __name__ == "__main__":
    main()
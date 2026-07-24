import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import yaml
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# [수정 반영] 프로젝트 루트 디렉토리 절대경로 정의
PROJECT_ROOT = Path("/home/taehoon/sprint-ai-mid-project_team3")


def load_config(config_path: str = None) -> Dict[str, Any]:
    """[수정 반영] 프로젝트 루트 기준 yaml 설정 파일 로드"""
    if config_path:
        target_path = Path(config_path)
    else:
        # 1차 탐색: /home/taehoon/sprint-ai-mid-project_team3/configs/config.yaml
        target_path = PROJECT_ROOT / "configs" / "config.yaml"

    # 2차/3차 Fallback 경로 순차 확인
    if not target_path.exists():
        alt_default = PROJECT_ROOT / "configs" / "default.yaml"
        alt_root = PROJECT_ROOT / "config.yaml"

        if alt_default.exists():
            target_path = alt_default
        elif alt_root.exists():
            target_path = alt_root

    if not target_path.exists():
        print(f"[Warning] 설정 파일을 찾을 수 없습니다: {target_path}")
        return {}

    print(f"[Info] Config 로드 완료: {target_path}")
    with open(target_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def check_and_reset_db_if_config_changed(
    persist_directory: str,
    collection_name: str,
    embeddings: Any,
    current_params: Dict[str, Any],
) -> None:
    """
    [파라미터 변경 감지 시 기존 DB 삭제 및 재 생성 메커니즘]
    임베딩 모델, chunk_size, chunk_overlap 변경 시 기존 Chroma DB 컬렉션을 초기화합니다.
    """
    os.makedirs(persist_directory, exist_ok=True)
    state_file = Path(persist_directory) / "param_state.json"

    old_params = {}
    if state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                old_params = json.load(f)
        except Exception:
            old_params = {}

    is_model_changed = old_params.get("embedding_model") and old_params.get("embedding_model") != current_params.get("embedding_model")
    is_size_changed = old_params.get("chunk_size") and old_params.get("chunk_size") != current_params.get("chunk_size")
    is_overlap_changed = old_params.get("chunk_overlap") and old_params.get("chunk_overlap") != current_params.get("chunk_overlap")

    if is_model_changed or is_size_changed or is_overlap_changed:
        print("[Warning] Vector DB 관련 설정 변경이 감지되었습니다!")
        if is_model_changed:
            print(f" - 임베딩 모델 변경: {old_params.get('embedding_model')} -> {current_params.get('embedding_model')}")
        if is_size_changed:
            print(f" - 청크 크기 변경: {old_params.get('chunk_size')} -> {current_params.get('chunk_size')}")
        if is_overlap_changed:
            print(f" - 청크 중첩 변경: {old_params.get('chunk_overlap')} -> {current_params.get('chunk_overlap')}")
        print("충돌 방지를 위해 기존 DB 컬렉션을 초기화하고 새로 생성합니다.")

        temp_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )
        temp_store.delete_collection()

    # 현재 파라미터 상태 기록
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(current_params, f, indent=4, ensure_ascii=False)


def build_local_vectordb(chunks: List[Dict[str, Any]], config: Dict[str, Any]) -> Chroma:
    """
    [수정 반영] 증분 업데이트 및 /data/processed/vector_db/local 경로 기반 DB 구축 함수
    """
    retrieval_cfg = config.get("retrieval", {})
    local_cfg = retrieval_cfg.get("profiles", {}).get("local", {})
    chunking_cfg = config.get("chunking", {})

    # 1. 설정값 추출 (기본 경로: /data/processed/vector_db/local)
    embedding_model_name = local_cfg.get("embedding_model", "dragonkue/BGE-m3-ko")
    persist_dir = local_cfg.get("persist_directory", "/data/processed/vector_db/local")
    collection_name = local_cfg.get("collection_name", "bidmate_local")
    cache_path = local_cfg.get("cache_path", "model_cache")
    device = local_cfg.get("device", "cpu")
    
    # Config에서 batch_size 연동 (기본값: 1000)
    batch_size = local_cfg.get("batch_size", 1000)

    # 상대 경로로 전달되었을 경우 PROJECT_ROOT 기준으로 절대 경로 변환
    if not os.path.isabs(persist_dir):
        persist_dir = str(PROJECT_ROOT / persist_dir)

    # 2. Embedding 모델 초기화
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        cache_folder=cache_path,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 3. 설정 변경 확인 및 DB 초기화 검사
    current_params = {
        "embedding_model": embedding_model_name,
        "chunk_size": chunking_cfg.get("chunk_size", 800),
        "chunk_overlap": chunking_cfg.get("chunk_overlap", 120),
    }
    check_and_reset_db_if_config_changed(persist_dir, collection_name, embeddings, current_params)

    # 4. Vector DB 로드
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )

    # 5. 증분 동기화 및 batch_size 처리 로직
    before_count = vectorstore._collection.count()
    print(f"적재 전 Local Chroma DB 청크 수: {before_count}개")

    incoming_docs = {}
    for idx, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {}).copy() if isinstance(chunk.get("metadata"), dict) else {}
        chunk_id = str(chunk.get("chunk_id") or meta.get("chunk_id") or f"chunk_{idx}")
        doc_id = str(chunk.get("doc_id") or meta.get("doc_id", ""))

        meta["title"] = meta.get("title", chunk.get("title", ""))
        meta["file_name"] = meta.get("file_name", chunk.get("file_name", ""))
        meta["chunk_id"] = chunk_id
        meta["doc_id"] = doc_id

        if doc_id not in incoming_docs:
            incoming_docs[doc_id] = []

        incoming_docs[doc_id].append({
            "id": chunk_id,
            "text": chunk.get("text", ""),
            "metadata": meta
        })

    # 삭제된 문서 완전 제거
    if before_count > 0:
        all_db_data = vectorstore.get(include=["metadatas"])
        all_db_doc_ids = {m["doc_id"] for m in all_db_data.get("metadatas", []) if m and "doc_id" in m}
        completely_deleted_doc_ids = all_db_doc_ids - set(incoming_docs.keys())

        for d_id in completely_deleted_doc_ids:
            vectorstore.delete(where={"doc_id": d_id})

    to_upsert_texts, to_upsert_metadatas, to_upsert_ids, to_delete_ids = [], [], [], []
    stat_unchanged = stat_modified = stat_deleted = stat_added = 0

    for doc_id, new_chunks in incoming_docs.items():
        existing = vectorstore.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
        existing_ids = existing.get("ids", [])
        existing_docs = existing.get("documents", [])
        existing_metadatas = existing.get("metadatas", [])

        existing_map = {
            eid: {"text": edoc, "metadata": emeta}
            for eid, edoc, emeta in zip(existing_ids, existing_docs, existing_metadatas)
        }
        seen_new_ids = set()

        for nc in new_chunks:
            nid, ntext, nmeta = nc["id"], nc["text"], nc["metadata"]
            seen_new_ids.add(nid)

            if nid in existing_map:
                if existing_map[nid]["text"] != ntext or existing_map[nid]["metadata"] != nmeta:
                    stat_modified += 1
                    to_upsert_texts.append(ntext)
                    to_upsert_metadatas.append(nmeta)
                    to_upsert_ids.append(nid)
                else:
                    stat_unchanged += 1
            else:
                stat_added += 1
                to_upsert_texts.append(ntext)
                to_upsert_metadatas.append(nmeta)
                to_upsert_ids.append(nid)

        for eid in existing_map:
            if eid not in seen_new_ids:
                stat_deleted += 1
                to_delete_ids.append(eid)

    if to_delete_ids:
        vectorstore.delete(ids=to_delete_ids)

    # batch_size 단위 저장 적용
    total_upsert = len(to_upsert_ids)
    if total_upsert > 0:
        print(f"추가/수정 대상 {total_upsert}개 청크를 batch_size({batch_size}) 단위로 저장합니다.")
        for i in range(0, total_upsert, batch_size):
            vectorstore.add_texts(
                texts=to_upsert_texts[i : i + batch_size],
                metadatas=to_upsert_metadatas[i : i + batch_size],
                ids=to_upsert_ids[i : i + batch_size],
            )
    else:
        print("변경 사항이 없어 Vector DB 저장을 스킵합니다.")

    after_count = vectorstore._collection.count()
    print(f"== Local DB 구축 완료 (최종 청크 수: {after_count}개 / 유지:{stat_unchanged}, 수정:{stat_modified}, 삭제:{stat_deleted}, 추가:{stat_added}) ==")
    return vectorstore


if __name__ == "__main__":
    cfg = load_config()
    
    # [수정 반영] 요청하신 기본 청크 파일 경로 매핑: /data/processed/chunks_800_120.jsonl
    chunks_file = cfg.get("paths", {}).get("chunks", "/data/processed/chunks_800_120.jsonl")
    
    if os.path.exists(chunks_file):
        print(f"청크 데이터 로드 중: {chunks_file}")
        chunks_data = []
        with open(chunks_file, "r", encoding="utf-8") as f:
            if chunks_file.endswith(".jsonl"):
                for line in f:
                    if line.strip():
                        chunks_data.append(json.loads(line))
            else:
                chunks_data = json.load(f)
        build_local_vectordb(chunks_data, cfg)
    else:
        print(f"[Warning] 청크 파일 경로가 존재하지 않습니다: {chunks_file}")
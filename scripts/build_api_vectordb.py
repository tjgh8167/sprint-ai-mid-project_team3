import yaml
import json
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def build_db():
    load_dotenv() # 환경변수(.env)에서 OPENAI_API_KEY 로직 추가

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 환경변수(.env)에 설정되지 않았습니다.") 
        return

    # 설정 및 데이터 로드
    yaml_path = "./config/default.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # yaml에 있는 청크 파일 경로 확인
    chunk_path = config["paths"]["chunks"]

    if not os.path.exists(chunk_path):
        print(f"청크 파일을 찾을 수 없습니다: {chunk_path}")
        return

    chunks = []
    with open(chunk_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    
    total_input_chunks = len(chunks)

    # 청크가 잘 로드 됐는지 출력으로 명시(확인 용도)
    print(f"JSONL 청크 파일 로드 완료: {total_input_chunks}개의 청크 확인")

    # OpenAI yaml 설정 확인
    openai_config = config.get("retrieval", {}).get("profiles", {}).get("openai", {})
    embedding_model = openai_config.get("embedding_model", "text-embedding-3-small")
    persist_directory = openai_config.get("persist_directory", "vector_db/openai")
    collection_name = openai_config.get("collection_name", "bidmate_openai")

    # 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings(model=embedding_model)

    # Vector DB 초기화
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    # 로깅: 적재 전 DB 청크 수 확인
    try:
        before_count = vectorstore._collection.count()
        print(f"적재 전 Chroma DB 청크 수: {before_count}개")
    except Exception as e:
        print(f"[경고] 기존 DB 상태를 확인할 수 없습니다 (최초 생성일 수 있음): {e}")
        before_count = 0

    # 입력 받은 청크 데이터를 문서로 묶기 (doc_id)
    incoming_docs = {}
    for idx, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {}).copy()
        chunk_id = chunk.get("chunk_id", f"chunk_{idx}")
        doc_id = chunk.get("doc_id", meta.get("doc_id", ""))
        
        meta["chunk_id"] = chunk_id
        meta["doc_id"] = doc_id
        
        if doc_id not in incoming_docs:
            incoming_docs[doc_id] = []
            
        incoming_docs[doc_id].append({
            "id": chunk_id,
            "text": chunk["text"],
            "metadata": meta
        })

    # 통계를 위한 카운터
    total_existing_chunks_processed = 0 # 기존 전체 청크 수
    stat_unchanged = 0                  # 변경 되지 않은 청크 (유지)
    stat_modified = 0                   # 변경 된 청크
    stat_deleted = 0                    # 삭제된 청크
    stat_added = 0                      # 추가된 청크

    # 실제 DB 작업에 쓰일 리스트 모음
    to_upsert_texts = []
    to_upsert_metadatas = []
    to_upsert_ids = []
    to_delete_ids = []

    # 기존 DB 데이터와 새로운 청크 파일 비교 분석
    for doc_id, new_chunks in incoming_docs.items():
        
        try:                                     # 현재 DB에서 새롭게 받은 문서(doc_id)에 해당하는 기존 청크들을 직접 조회
            existing = vectorstore.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
            existing_ids = existing.get("ids", [])
            existing_docs = existing.get("documents", [])
            existing_metadatas = existing.get("metadatas", [])
        except Exception:
            existing_ids, existing_docs, existing_metadatas = [], [], []

        # 기존 DB에 있던 청크 데이터 수 += (유지, 수정, 삭제 청크 수)
        total_existing_chunks_processed += len(existing_ids)

        # 빠른 비교를 위해 기존 데이터를 딕셔너리로 맵핑
        existing_map = {}
        for eid, edoc, emeta in zip(existing_ids, existing_docs, existing_metadatas):
            existing_map[eid] = {"text": edoc, "metadata": emeta}

        # 새 파일에 존재하는 청크 ID들을 기록 (삭제된 청크 구별용)
        seen_new_ids = set()

        for nc in new_chunks:
            nid = nc["id"]
            ntext = nc["text"]
            nmeta = nc["metadata"]
            seen_new_ids.add(nid)

            if nid in existing_map:
                # 기존에 있던 청크인 경우: 본문이나 메타데이터가 바뀌었는지 검사
                if existing_map[nid]["text"] != ntext or existing_map[nid]["metadata"] != nmeta:
                    stat_modified += 1
                    to_upsert_texts.append(ntext)
                    to_upsert_metadatas.append(nmeta)
                    to_upsert_ids.append(nid)
                else:
                    stat_unchanged += 1
            else:
                # 완전히 처음 보는 새로운 청크인 경우: 추가 대상으로 등록
                stat_added += 1
                to_upsert_texts.append(ntext)
                to_upsert_metadatas.append(nmeta)
                to_upsert_ids.append(nid)

        # 기존 DB에는 있었으나, 새 파일에서는 없어진 청크인 경우: 삭제 대상으로 등록
        for eid in existing_map:
            if eid not in seen_new_ids:
                stat_deleted += 1
                to_delete_ids.append(eid)

    # 정보가 사라진 청크 데이터 삭제
    if to_delete_ids:
        print(f"더 이상 존재하지 않는 구버전 청크 {len(to_delete_ids)}개 삭제 중...")
        vectorstore.delete(ids=to_delete_ids)

    # 추가 및 정보가 변경된 청크 저장
    total_upsert_count = len(to_upsert_ids)
    if total_upsert_count > 0:
        batch_size = openai_config.get("batch_size", 1000)
        print(f"추가(변경)된 청크 {total_upsert_count}개 저장 중")
        
        for i in range(0, total_upsert_count, batch_size):
            batch_texts = to_upsert_texts[i : i + batch_size]
            batch_metadatas = to_upsert_metadatas[i : i + batch_size]
            batch_ids = to_upsert_ids[i : i + batch_size]

            current_batch_num = (i // batch_size) + 1
            total_batches = (total_upsert_count + batch_size - 1) // batch_size
            
            print(f"[{current_batch_num}/{total_batches}] {i} ~ {min(i + batch_size, total_upsert_count)}번째 대상 처리 중")
            vectorstore.add_texts(
                texts=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
    else:
        print("업데이트할 내용이 없습니다. VectorDB 내 모든 데이터가 이미 최신 상태입니다.")

    total_details = stat_unchanged + stat_modified + stat_deleted + stat_added
    after_count = vectorstore._collection.count()

    print("== VectorDB 최신화 완료 ==")
    print(f" 기존 청크 수: {total_existing_chunks_processed}개")
    print(f" 최신화 후 최종 청크 수: {after_count}개")
    print("-"*40)
    print(f" == 세부 변경 내역 총 {total_details}개 ==")
    print(f" - 유지 : {stat_unchanged}개")
    print(f" - 수정 : {stat_modified}개")
    print(f" - 삭제 : {stat_deleted}개")
    print(f" - 추가 : {stat_added}개")
    print("-"*40)

if __name__ == "__main__":
    build_db()
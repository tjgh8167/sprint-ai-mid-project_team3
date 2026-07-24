import yaml
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent.parent

def build_db():
    load_dotenv() # 환경변수(.env)에서 OPENAI_API_KEY 로직 추가

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 환경변수(.env)에 설정되지 않았습니다.") 
        return

    # 설정 및 데이터 로드
    yaml_path = BASE_DIR / "config/default.yaml" # 절대 경로로 지정
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # yaml에 있는 청크 파일 경로 확인 (상대 경로라면 BASE_DIR 기준으로 절대 경로 변환)
    raw_chunk_path = config["paths"]["chunks"]
    chunk_path = Path(raw_chunk_path) if os.path.isabs(raw_chunk_path) else BASE_DIR / raw_chunk_path

    if not chunk_path.exists():
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

    chunking_config = config.get("chunking", {})
    chunk_size = chunking_config.get("chunk_size", 800)
    chunk_overlap = chunking_config.get("chunk_overlap", 120)

    # 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings(model=embedding_model)

    # Vector DB 초기화
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    # 옵션 변경 감지 로직
    os.makedirs(persist_directory, exist_ok=True)               # persist_directory에 새로운 파일 제작
    state_file = Path(persist_directory) / "param_state.json"   # 옵션 값 저장을 위한 param_state.json 생성
    old_state = {}

    # 이미 param_state를 생성한 기록이 있다면 old_state로 가져와서 읽기
    if state_file.exists():
        with open(state_file, "r", encoding="utf-8") as f:
            old_state = json.load(f)

    # 옵션이 바뀌지 않았다고 우선 설정
    is_option_changed = False

    is_model_changed = old_state.get("embedding_model") and old_state.get("embedding_model") != embedding_model
    is_size_changed = old_state.get("chunk_size") and old_state.get("chunk_size") != chunk_size
    is_overlap_changed = old_state.get("chunk_overlap") and old_state.get("chunk_overlap") != chunk_overlap

    # 3가지 중 하나라도 바뀌었다면 is_option_changed가 True가 됨
    is_option_changed = is_model_changed or is_size_changed or is_overlap_changed

    if is_option_changed:
        print("Vector DB 환경 설정 변경이 감지되었습니다!")
        # 변경된 파라미터만 출력
        if is_model_changed: print(f" - 임베딩 모델: {old_state.get('embedding_model')} -> {embedding_model}")
        if is_size_changed: print(f" - 청크 크기: {old_state.get('chunk_size')} -> {chunk_size}")
        if is_overlap_changed: print(f" - 청크 중첩: {old_state.get('chunk_overlap')} -> {chunk_overlap}")
        print("충돌 방지를 위해 기존 DB를 초기화하고 처음부터 재구축합니다.")

        # 기존 Vector DB 삭제
        vectorstore.delete_collection()

        # 재 구축
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )

    # 로깅: 적재 전 DB 청크 수 확인
    try:
        before_count = vectorstore._collection.count()
        print(f"적재 전 Chroma DB 청크 수: {before_count}개")
    except Exception as exc:
        raise RuntimeError(f"기존 Vector DB 조회에 실패했습니다.") from exc

    # 입력 받은 청크 데이터를 문서로 묶기 (doc_id)
    incoming_chunks_map = {}
    incoming_doc_ids = set() # 삭제 로그용 doc_id 모음

    for idx, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {}).copy()
        
        raw_chunk_id = meta.get("chunk_id") or chunk.get("chunk_id")
        doc_id = meta.get("doc_id") or chunk.get("doc_id") or ""
        
        unique_id = str(raw_chunk_id if raw_chunk_id else f"{doc_id}_chunk_{idx}" if doc_id else f"chunk_{idx}")
        
        meta["chunk_id"] = unique_id
        meta["doc_id"] = doc_id
        
        if doc_id:
            incoming_doc_ids.add(doc_id)

        incoming_chunks_map[unique_id] = {
            "id": unique_id,
            "text": chunk.get("text", ""),
            "metadata": meta
        }

    incoming_ids = set(incoming_chunks_map.keys())

    # 통계를 위한 카운터
    total_existing_chunks_processed = before_count  # 기존 전체 청크 수
    stat_unchanged = 0                              # 변경 되지 않은 청크 (유지)
    stat_modified = 0                               # 변경 된 청크
    stat_deleted = 0                                # 삭제된 청크
    stat_added = 0                                  # 추가된 청크

    # 변경사항이 있는 데이터를 담을 리스트 
    to_upsert_texts = []
    to_upsert_metadatas = []
    to_upsert_ids = []
    to_delete_ids = []

    if not is_option_changed and before_count > 0:
        print("기존 Vector DB 데이터와 비교 분석 중...")
        
        # for문 안에서 DB를 매번 부르지 않고, 전체 데이터를 한 번만 가져옴
        existing_data = vectorstore.get(include=["documents", "metadatas"])
        existing_ids_list = existing_data.get("ids", [])
        existing_docs = existing_data.get("documents", [])
        existing_metas = existing_data.get("metadatas", [])
        
        existing_map = {}
        db_doc_to_chunks = {} # 삭제 로그용: doc_id별 청크 목록 맵핑

        for eid, edoc, emeta in zip(existing_ids_list, existing_docs, existing_metas):
            existing_map[eid] = {"text": edoc, "metadata": emeta}
            
            # doc_id별 청크 추적
            edoc_id = emeta.get("doc_id", "") if emeta else ""
            if edoc_id:
                if edoc_id not in db_doc_to_chunks:
                    db_doc_to_chunks[edoc_id] = []
                db_doc_to_chunks[edoc_id].append(eid)
            
        existing_ids = set(existing_map.keys())

        to_delete_ids = list(existing_ids - incoming_ids)
        ids_to_add = incoming_ids - existing_ids
        ids_in_both = incoming_ids & existing_ids

        # 삭제 로그
        completely_deleted_doc_ids = set(db_doc_to_chunks.keys()) - incoming_doc_ids
        if completely_deleted_doc_ids:
            for d_id in completely_deleted_doc_ids:
                deleted_count = len(db_doc_to_chunks[d_id])
                print(f"더 이상 존재하지 않는 문서 삭제 중 (doc_id: {d_id}, 청크 {deleted_count}개)")

        # 텍스트, 메타데이터 변경 감지
        ids_to_modify = set()
        for cid in ids_in_both:
            old_text = existing_map[cid]["text"]
            new_text = incoming_chunks_map[cid]["text"]
            old_meta = existing_map[cid]["metadata"]
            new_meta = incoming_chunks_map[cid]["metadata"]

            if old_text != new_text or old_meta != new_meta:
                ids_to_modify.add(cid)

        ids_unchanged = ids_in_both - ids_to_modify

        # 통계 최신화
        stat_deleted = len(to_delete_ids)
        stat_added = len(ids_to_add)
        stat_modified = len(ids_to_modify)
        stat_unchanged = len(ids_unchanged)

        # upsert 리스트 만들기 (추가된 것 + 수정된 것)
        target_upsert_ids = ids_to_add | ids_to_modify
        for tid in target_upsert_ids:
            to_upsert_ids.append(tid)
            to_upsert_texts.append(incoming_chunks_map[tid]["text"])
            to_upsert_metadatas.append(incoming_chunks_map[tid]["metadata"])

    else:
        # 옵션이 바뀌어 초기화됐거나, DB가 비어있는 경우 모두 추가
        stat_added = len(incoming_ids)
        for tid, data in incoming_chunks_map.items():
            to_upsert_ids.append(tid)
            to_upsert_texts.append(data["text"])
            to_upsert_metadatas.append(data["metadata"])

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

    with open(state_file, "w", encoding="utf-8") as f:      # 업데이트 된 내용을 param_state.json파일에 최신화
        json.dump({                                         # 들어갈 내용
            "embedding_model": embedding_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }, f, indent=4)                                     # 들여쓰기 (하드코딩 사유: 들여쓰기라 굳이..?)

    after_count = vectorstore._collection.count()

    print("== VectorDB 최신화 완료 ==")
    print(f" 기존 청크 수: {total_existing_chunks_processed}개")
    print(f" 최신화 후 최종 청크 수: {after_count}개")
    print("-"*40)
    print(f" == 세부 처리 결과 ==")
    print(f" - 유지 : {stat_unchanged}개")
    print(f" - 수정 : {stat_modified}개")
    print(f" - 삭제 : {stat_deleted}개")
    print(f" - 추가 : {stat_added}개")
    print("-"*40)

if __name__ == "__main__":
    build_db()
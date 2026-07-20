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
    yaml_path = "../config/default.yaml"
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

    # 텍스트 데이터 파싱
    texts = []      # 실제 텍스트
    metadatas = []  # 각 텍스트에 대한 데이터
    ids = []        # 청크 별 ID (chunk_id)

    for idx, chunk in enumerate(chunks):                            # 문서 로드 / 청킹하면서 생긴 metadata와 chunk_id를 기반으로 texts, metadatas, ids 생성
        texts.append(chunk["text"])                                 # 본문을 texts에 추가

        meta = chunk.get("metadata", {}).copy()                     # 텍스트의 metadata를 가져온다 copy = 직접 수정 방지
        meta["chunk_id"] = chunk.get("chunk_id", f"chunk_{idx}")    # metadata의 chunk_id를 가져오는데 f"chunk_{idx}"는 chunk_id가 없을 경우 새로 만든다~
        meta["doc_id"] = chunk.get("doc_id", "")                    # metadata의 doc_id를 가져오는데 없으면 빈 문자열로 설정 (사실 존재하니, chunk["doc_id"]로 가져와도 됨)
        
        metadatas.append(meta)                                      # metadatas에 위에서 만든 meta를 추가
        ids.append(meta["chunk_id"])                                # 청크별 고유 식별 id (Vector DB에서 각 청크를 구분하기 위해 필요)

    batch_size = openai_config.get("batch_size", 1000) # 한 번에 데이터베이스로 보낼 청크 단위 / 지정을 안 하면 오래 걸리기 때문에 배치사이즈 설정 필요
        
    for i in range(0, total_input_chunks, batch_size):
        # 전체 데이터에서 batch_size만큼 슬라이싱
        batch_texts = texts[i : i + batch_size]
        batch_metadatas = metadatas[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]

        current_batch_num = (i // batch_size) + 1
        total_batches = (total_input_chunks + batch_size - 1) // batch_size
        
        # 몇 번째 배치가 들어가는지 모니터링 로그 출력
        print(f"[{current_batch_num}/{total_batches}] {i} ~ {min(i + batch_size, total_input_chunks)}번째 청크 저장 중")
        
        # 실제로 Vector DB에 배치 적재
        vectorstore.add_texts(
        texts=batch_texts,
        metadatas=batch_metadatas,
        ids=batch_ids
    )

    # 벡터 DB 적재 후 청크 수 after_count 로 명시
    after_count = vectorstore._collection.count()
    print(f"최종 Chroma DB에 저장 된 청크 수: {after_count}개") # 시각적 확인

    if after_count >= total_input_chunks:
        print(f"모든 청크가 성공적으로 {persist_directory}에 저장 되었습니다.")
    else:
        print("입력된 청크 수와 DB의 청크 수가 다릅니다. (중복된 ID가 병합되었을 수 있습니다.)")
        print(f"입력된 청크 수: {total_input_chunks}, DB 청크 수: {after_count}")

if __name__ == "__main__":
    build_db()
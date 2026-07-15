import yaml
import json
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def build_db():

    # 1. 설정 및 데이터 로드
    with open("config/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    chunks = []
    with open(config["paths"]["chunks"], "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    # 2. OpenAI yaml 설정 확인
    openai_config = config.get("retrieval", {}).get("profiles", {}).get("openai", {})
    embedding_model = openai_config.get("embedding_model", "text-embedding-3-small")
    persist_directory = openai_config.get("persist_directory", "vector_db/openai")
    collection_name = openai_config.get("collection_name", "bidmate_openai")

    # 3. 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings(model=embedding_model)

    # 4. 텍스트 데이터 파싱
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


    # 5. DB 생성 및 저장 

    Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        persist_directory=persist_directory,
        collection_name=collection_name
    )

if __name__ == "__main__":
    build_db()
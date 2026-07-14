import os
from src.retriever import SearchResult
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

class OpenAIChromaRetriever:

    def __init__(self, chunks: list[dict], config: dict):
        self.chunks = chunks
        self.config = config

        # retrieval 설정이 config에 있는지 확인하고, 없으면 그냥 self.config 가져오고 있으면 retriever 가져오기
        if "retrieval" in self.config:
            retrieval_config = self.config["retrieval"]
        else:
            retrieval_config = self.config

        # yaml에서 설정 가져오기
        openai_config = retrieval_config.get("profiles", {}).get("openai", {})
        
        embedding_model = openai_config.get("embedding_model", "text-embedding-3-small") # 뒤에 text-embedding-3-small 은 안전장치 (하드코딩 아님)
        persist_directory = openai_config.get("persist_directory", "vector_db/openai") # 마찬가지
        collection_name = openai_config.get("collection_name", "bidmate_openai") # 마찬가지
        
        # OpenAI 임베딩 생성기 초기화
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        # top_k 값 저장 (yaml에서 설정한 값이 없으면 기본값 3으로 설정)
        self.default_top_k = retrieval_config.get("top_k", 3)

        # Vector DB를 위한 데이터 파싱 (sample chunks를 기반으로 texts, metadatas, ids 생성 / 추후 문서 청킹 후 필요 시 수정)
        texts = []      # 실제 텍스트
        metadatas = []  # 각 텍스트에 대한 데이터
        ids = []        # 청크 별 ID (chunk_id)

        for idx, chunk in enumerate(self.chunks):                       # 문서 로드 / 청킹하면서 생긴 metadata와 chunk_id를 기반으로 texts, metadatas, ids 생성
            texts.append(chunk["text"])                                 # 본문을 texts에 추가
            meta = chunk.get("metadata", {}).copy()                     # 텍스트의 metadata를 가져온다 copy = 직접 수정 방지
            meta["chunk_id"] = chunk.get("chunk_id", f"chunk_{idx}")    # metadata의 chunk_id를 가져오는데 f"chunk_{idx}"는 chunk_id가 없을 경우 새로 만든다~
            meta["doc_id"] = chunk.get("doc_id", "")                    # metadata의 doc_id를 가져오는데 없으면 빈 문자열로 설정 (사실 존재하니, chunk["doc_id"]로 가져와도 됨)
            metadatas.append(meta)                                      # metadatas에 위에서 만든 meta를 추가
            ids.append(meta["chunk_id"])                                # 청크별 고유 식별 id (Vector DB에서 각 청크를 구분하기 위해 필요)

        # Vector DB 생성
        self.vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            ids=ids,
            persist_directory=persist_directory,
            collection_name=collection_name
        )

    def search(self, query: str, top_k: int | None = None, filters: dict | None = None) -> list[SearchResult]:

        # CLI에서 호출할때는 yaml값 그대로, 코드에서 바꾸면 인자값으로 (yaml 바꾸면 바뀜)
        search_k = top_k if top_k is not None else self.default_top_k

        # Chroma에서 유사도 검색 수행(질문벡터와 거리점수(유사도) 계산)
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            query=query,
            k=search_k,
            filter=filters
        )

        results = []
        for doc, score in docs_and_scores:
            results.append(
                SearchResult(
                    chunk_id=doc.metadata.get("chunk_id", ""),
                    doc_id=doc.metadata.get("doc_id", ""),
                    text=doc.page_content,
                    metadata=doc.metadata,
                    score=float(score)  # 거리를 뜻하는 수치(float) 반영
                )
            )
        
        return results
    
        # raise NotImplementedError(
        #     "Retrieval 1 담당자가 OpenAI 임베딩 생성, Chroma 저장·조회, "
        #     "SearchResult 변환을 구현해야 합니다."
        # )

import os
from src.retriever import SearchResult
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

class OpenAIChromaRetriever:

    def __init__(self, config: dict):
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

        # Vector DB 생성
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
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

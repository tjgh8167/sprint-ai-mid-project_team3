import os
import re
import unicodedata
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
    # 정규화 함수 추가: 텍스트를 정규화하여 필터링 시 일관성을 유지
    def normalize_text(self, text: str) -> str:
        if not text: return ""
        text = unicodedata.normalize('NFC', str(text)) # 유니코드 정규화
        text = text.lower()                            # 소문자 통일
        text = re.sub(r'\s+', ' ', text)               # 공백 문자(스페이스, 탭 등)를 단일 공백으로 변환
        return text.strip()

    def search(self, query: str, top_k: int | None = None, filters: dict | None = None) -> list[SearchResult]:

        # CLI에서 호출할때는 yaml값 그대로, 코드에서 바꾸면 인자값으로 (yaml top_k바꾸면 바뀜)
        search_k = top_k if top_k is not None else self.default_top_k

        # yaml 설정에서 기본 필터 가져오기 및 병합
        openai_config = self.config.get("retrieval", {}).get("profiles", {}).get("openai", {})
        base_filters = openai_config.get("default_filters", {})
        
        # filters 인자가 None이면 base_filters 사용, 아니면 filters 사용
        active_filters = filters if filters is not None else base_filters

        # 부분 일치 설정 (ChromaDB는 기본적으로 정확히 일치하는 필터만 지원하므로, 부분 일치를 위해서는 검색 결과를 더 많이 가져와서 필터링 후 top_k만큼 반환)
        fetch_k = max(search_k * 10, 50) # fetch_k는 top_k보다 충분히 큰 값으로 설정 (최소 50개 이상)

        # Chroma에서 유사도 검색 수행(질문벡터와 거리점수(유사도) 계산)
        try: 
            docs_and_scores = self.vectorstore.similarity_search_with_score(
                query=query,
                k=fetch_k,
                filter= None
            )

        except Exception as e:                                                # ChromaDB에서 필터 조건이 잘못되었거나, 검색 중 오류가 발생하면 예외 처리, 강제 종료 방지
            print(f"검색 중 오류 발생 (필터 조건 확인): {e}")                        # 오류 원인 및 로그 출력
            return []

        if not docs_and_scores:                                               # docs_and_scores가 비어있으면 (즉, 검색 결과가 없으면) 안내 메시지 출력
            return []

        # 부분 일치 필터링 
        results = []
        for doc, score in docs_and_scores:
            is_match = True

            if active_filters:
                for key, value in active_filters.items():
                    if not value:
                        continue
                    
                    clean_filter_val = self.normalize_text(str(value)) # 필터 값 정규화
                    clean_metadata_val = self.normalize_text(str(doc.metadata.get(key, ""))) # 문서 메타데이터 값 정규화
                    
                    if clean_filter_val not in clean_metadata_val: # 부분 일치 여부 확인
                        is_match = False
                        break   
            if is_match:
                results.append(
                    SearchResult(
                    chunk_id=doc.metadata.get("chunk_id", ""),
                    doc_id=doc.metadata.get("doc_id", ""),
                    text=doc.page_content,
                    metadata=doc.metadata,
                    score=float(score)
                    )
                )

                if len(results) >= search_k: # 부분 일치 필터링 후 top_k만큼 결과 반환
                    break

        # 필터 미일치 시 안내 동작
        if not results:                                               # docs_and_scores가 비어있으면 (즉, 검색 결과가 없으면) 안내 메시지 출력
            print(f"안내: '{active_filters}' 조건에 일치하는 문서를 찾을 수 없습니다.")  # active_filters 조건에 일치하는 문서가 없음을 안내
            return []
                
        
        return results
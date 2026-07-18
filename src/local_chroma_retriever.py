from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

@dataclass
class SearchResult:
    """rag_engine.py 규격에 맞춘 검색 결과 데이터 클래스"""
    text: str
    metadata: Dict[str, Any]
    score: float

class LocalChromaRetriever:
    def __init__(self, chunks: List[Dict[str, Any]], config: Dict[str, Any]):
        """
        Local Chroma Retriever 초기화
        """
        self.chunks = chunks
        self.config = config
        
        # 1. HuggingFace 로컬 임베딩 로드
        model_name = self.config.get("embedding_model", "dragonkue/BGE-m3-ko")
        cache_dir = self.config.get("cache_dir")
        device = self.config.get("device", "cpu")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=cache_dir,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # 2. Chroma Vector DB 연결
        persist_directory = self.config.get("persist_directory", "vector_db/local")
        collection_name = self.config.get("collection_name", "bidmate_localgit")
        
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

    def search(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        org_name: Optional[str] = None,
        project_name: Optional[str] = None,
        doc_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        메타데이터 필터 및 유사도 점수(Score)를 포함한 검색 수행
        """
        k = top_k or self.config.get("top_k", 4)
        
        # Chroma 메타데이터 필터 딕셔너리 동적 빌드
        filter_conditions = []
        if org_name:
            filter_conditions.append({"org_name": org_name})
        if project_name:
            filter_conditions.append({"project_name": project_name})
        if doc_id:
            filter_conditions.append({"doc_id": doc_id})
            
        search_filter = None
        if len(filter_conditions) == 1:
            search_filter = filter_conditions[0]
        elif len(filter_conditions) > 1:
            search_filter = {"$and": filter_conditions}

        # 유사도 점수를 함께 반환하는 메서드 호출
        # (Document, score) 튜플 리스트를 반환합니다.
        docs_with_scores = self.vector_store.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            filter=search_filter
        )
        
        # rag_engine 규격(SearchResult)에 맞게 변환
        results = []
        for doc, score in docs_with_scores:
            results.append(
                SearchResult(
                    text=doc.page_content,    # Document의 본문을 text로 매핑
                    metadata=doc.metadata,    # 메타데이터 유지
                    score=float(score)        # 유사도 점수 변환
                )
            )
        return results

    def add_documents(self, docs: List[Document]):
        """테스트 데이터 적재용 메서드"""
        self.vector_store.add_documents(docs)
from typing import Dict, Any, List, Optional
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 프로젝트 통합 공통 계약 규격(SearchResult) 임포트 (chunk_id, doc_id 포함)
from src.retriever import SearchResult

class LocalChromaRetriever:
    """Retrieval 2: HuggingFace 임베딩 + Chroma 검색기 구현체 (Read-only 최적화 버전)"""

    def __init__(self, chunks: List[Dict[str, Any]], config: Dict[str, Any]):
        """
        Local Chroma Retriever 초기화 (질문 시점에는 저장된 Vector DB 로드만 수행)
        """
        self.chunks = chunks  # 질문 시점에는 일반적으로 빈 리스트([])가 유입되거나 무시됩니다.
        self.config = config
        
        # default.yaml 구조 정렬 및 맵핑 일치
        model_name = self.config.get("embedding", "dragonkue/BGE-m3-ko")
        cache_dir = self.config.get("cache_path")
        device = self.config.get("device", "cpu")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=cache_dir,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # VM/GCS 공용 데이터 경로 및 컬렉션 연동 (Git 커밋 금지 경로)
        persist_directory = self.config.get("persist_directory", "vector_db/local")
        collection_name = self.config.get("collection_name", "bidmate_localgit")
        
        # 실시간 적재(add_documents) 구문을 완전히 제거하여 검색 지연 최소화 및 동기화 무결성 확보
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

    def search(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        agency: Optional[str] = None,       #  org_name이 아닌 'agency' 필터 키로 통일
        project_name: Optional[str] = None,
        doc_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        공통 메타데이터 필터 조건을 충족하는 고속 Chroma Vector 유사도 검색 수행
        """
        k = top_k or self.config.get("top_k", 4)
        
        # Chroma 메타데이터 필터 딕셔너리 동적 빌드
        filter_conditions = []
        if agency:
            # 공통 전처리 규격에 매칭되도록 쿼리 타깃 키를 agency로 고정
            filter_conditions.append({"agency": agency})
        if project_name:
            filter_conditions.append({"project_name": project_name})
        if doc_id:
            filter_conditions.append({"doc_id": doc_id})
            
        search_filter = None
        if len(filter_conditions) == 1:
            search_filter = filter_conditions[0]  # 순수 단일 Dict 추출로 Chroma 구문 오류 차단
        elif len(filter_conditions) > 1:
            search_filter = {"$and": filter_conditions}

        # 무자비한 Threshold 필터링으로 0건을 반환할 여지가 있는_relevance_scores 대신 
        # 점수 손실이 전혀 없는 순수 거리 기반 연산 수식 호출
        try:
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=search_filter
            )
        except Exception:
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
        
        # 출처 표시 및 RAG 평가를 위한 5대 필수 반환 규격 조합
        results = []
        for doc, score in docs_with_scores:
            current_chunk_id = doc.metadata.get("chunk_id", "N/A")
            current_doc_id = doc.metadata.get("doc_id", "N/A")
            
            results.append(
                SearchResult(
                    chunk_id=current_chunk_id,
                    doc_id=current_doc_id,
                    text=doc.page_content,
                    metadata=doc.metadata,
                    score=float(score)
                )
            )
        return results

    def add_documents(self, docs: List[Document]):
        """별도 DB 빌드 스크립트(scripts/build_local_vectordb.py)에서 인덱싱 단행 시 호출할 통로 제공"""
        self.vector_store.add_documents(docs)
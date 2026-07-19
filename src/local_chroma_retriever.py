ffrom typing import Dict, Any, List, Optional
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 프로젝트 통합 계약 규격(SearchResult) 임포트 (chunk_id, doc_id 포함)
from src.retriever import SearchResult

class LocalChromaRetriever:
    """Retrieval 2: HuggingFace 임베딩 + Chroma 검색기 구현체"""

    def __init__(self, chunks: List[Dict[str, Any]], config: Dict[str, Any]):
        """
        Local Chroma Retriever 초기화 및 데이터 자동 적재
        """
        self.chunks = chunks
        self.config = config
        
        # default.yaml 구조 정렬
        # 외부 팩토리에서 profiles["local"] 딕셔너리를 config 인자로 넘겨주므로, 
        # yaml 스펙 규격명('embedding', 'cache_path', 'device')에 정확히 일치시킵니다.
        model_name = self.config.get("embedding", "dragonkue/BGE-m3-ko")
        cache_dir = self.config.get("cache_path")
        device = self.config.get("device", "cpu")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=cache_dir,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Chroma Vector DB 물리 경로 및 컬렉션 연동
        persist_directory = self.config.get("persist_directory", "vector_db/local")
        collection_name = self.config.get("collection_name", "bidmate_localgit")
        
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
    
        # 인스턴스가 호출되는 시점에 생성자 내부에서 add_documents를 강제 실행하여 적재 누락을 원천 차단합니다.
        if self.chunks:
            # 테스트/빌드 간 데이터 중복 누적 및 오염 방지를 위한 컬렉션 비우기(Reset)
            try:
                self.vector_store.delete_collection()
                self.vector_store = Chroma(
                    collection_name=collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=persist_directory
                )
            except Exception:
                pass
                
            documents_to_add = []
            for idx, c in enumerate(self.chunks):
                if isinstance(c, Document):
                    # 🎯 추적성 보완: chunk_id가 누락되어 있다면 인덱스 기반 자동 부여
                    if "chunk_id" not in c.metadata:
                        c.metadata["chunk_id"] = f"CHK_{idx:03d}"
                    documents_to_add.append(c)
                elif isinstance(c, dict):
                    text_content = c.get("text") or c.get("page_content", "")
                    metadata = {k: v for k, v in c.items() if k not in ["text", "page_content"]}
                    
                    if "chunk_id" not in metadata:
                        metadata["chunk_id"] = f"CHK_{idx:03d}"
                        
                    documents_to_add.append(Document(page_content=text_content, metadata=metadata))
            
            # 생성자 내부에서 add_documents() 메서드를 직접 트리거하여 데이터 영속 적재 완료
            self.add_documents(documents_to_add)

    def search(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        agency: Optional[str] = None,     
        project_name: Optional[str] = None,
        doc_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        메타데이터 필터 및 유사도 점수(Score)를 포함한 검색 수행
        """
        k = top_k or self.config.get("top_k", 4)
        
        # Chroma 메타데이터 필터 딕셔너리 동적 빌드
        filter_conditions = []
        if agency:
            #DB 적재 조건 및 쿼리 필터 키 명칭을 agency로 일치
            filter_conditions.append({"agency": agency})
        if project_name:
            filter_conditions.append({"project_name": project_name})
        if doc_id:
            filter_conditions.append({"doc_id": doc_id})
            
        search_filter = None
        if len(filter_conditions) == 1:
            search_filter = filter_conditions[0] # 순수 딕셔너리로 언랩핑하여 가공 (자료형 버그 박멸)
        elif len(filter_conditions) > 1:
            search_filter = {"$and": filter_conditions}

        # 무자비한 Threshold 필터링으로 0건을 반환할 여지가 있는 relevance_scores 대신 
        # 안정적인 거리 연산 스코어링 수식 호출 방어선 구축
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
        
        # 소스 추적을 위한 5대 변수(chunk_id, doc_id, text, metadata, score) 매핑 바인딩
        results = []
        for doc, score in docs_with_scores:
            current_chunk_id = doc.metadata.get("chunk_id", "N/A")
            
            # 만약 데이터 생성 적재 시점에 agency로 들어갔다면 doc_id 추적 처리를 유연하게 격리 보장
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
        """청크 데이터를 포지셔닝하여 실제 Chroma DB 본체에 영구 적재하는 메서드"""
        self.vector_store.add_documents(docs)
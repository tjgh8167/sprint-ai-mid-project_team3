from typing import Dict, Any, List, Optional
from langchain_core.documents import Document
# 모든 환경에서 호환되도록 표준 community 패키지 경로로 변경
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

class LocalChromaRetriever:
    def __init__(self, chunks: List[Dict[str, Any]], config: Dict[str, Any]):
        """
        Local Chroma Retriever 초기화
        
        Args:
            chunks: 데이터 적재를 위한 청크 리스트 (retriever_factory에서 전달)
            config: profiles["local"]에 해당하는 설정 딕셔너리
        """
        self.chunks = chunks
        self.config = config  # 전달받은 config가 곧 local 설정입니다.
        
        # 1. HuggingFace 로컬 임베딩 로드 (default.yaml 구조와 일치)
        model_name = self.config.get("embedding_model", "dragonkue/BGE-m3-ko")
        cache_dir = self.config.get("cache_dir")
        device = self.config.get("device", "cpu")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=cache_dir,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}  # 코사인 유사도 최적화
        )
        
        # 2. Chroma Vector DB 연결
        chroma_config = self.config.get("chroma", {})
        
        self.vector_store = Chroma(
            collection_name=chroma_config.get("collection_name", "default_collection"),
            embedding_function=self.embeddings,
            persist_directory=chroma_config.get("persist_directory")
        )

    def retrieve(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        org_name: Optional[str] = None,
        project_name: Optional[str] = None,
        doc_id: Optional[str] = None
    ) -> List[Document]:
        """
        메타데이터 필터가 적용된 유사도 검색 수행
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

        # similarity_search 수행
        docs = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=search_filter
        )
        return docs

    def add_documents(self, docs: List[Document]):
        """테스트 데이터 적재용 메서드"""
        self.vector_store.add_documents(docs)
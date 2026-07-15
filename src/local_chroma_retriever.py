from typing import List, Dict, Any, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

class LocalChromaRetriever:
    def __init__(self, config: Dict[str, Any]):
        """
        Local Chroma Retriever 초기화
        """
        self.config = config
        
        # 1. HuggingFace 로컬 임베딩 로드
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config["embedding"]["dragonkue/BGE-m3-ko"],
            cache_folder=config["embedding"].get("cache_dir"),
            model_kwargs={"device": config["embedding"].get("device", "cpu")},
            encode_kwargs={"normalize_embeddings": True}  # 코사인 유사도 최적화
        )
        
        # 2. Chroma Vector DB 연결
        self.vector_store = Chroma(
            collection_name=config["chroma"]["collection_name"],
            embedding_function=self.embeddings,
            persist_directory=config["chroma"]["persist_directory"]
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
        k = top_k or self.config["retriever"]["top_k"]
        
        # Chroma 메타데이터 필터 딕셔너리 동적 빌드
        # ($and 연산자를 활용하여 다중 조건 충족)
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



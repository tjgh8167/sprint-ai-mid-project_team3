import hashlib
from typing import Dict, Any, List, Optional

# -------------------------------------------------------------------------
# [수정] 구형 패키지 경로 완전히 제거 및 최신 패키지 통일
# langchain_community 사용하지 않고 langchain_huggingface, langchain_chroma 사용
# -------------------------------------------------------------------------
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 프로젝트 통합 공통 계약 규격(SearchResult) 임포트
from src.retriever import SearchResult


# -------------------------------------------------------------------------
# [수정] 단 하나의 LocalChromaRetriever 클래스만 유효하게 남김
# - HF 로컬 임베딩 기반 Chroma Similarity 검색
# - agency, project_name, doc_id 메타데이터 동적 필터링
# - 해시 기반 중복 인덱싱 방지 및 실시간 업데이트 Sync
# -------------------------------------------------------------------------
class LocalChromaRetriever:
    """Retrieval 2: HuggingFace 임베딩 + Chroma 검색기 구현체 (전체 메타데이터 추출 + 실시간 갱신 버전)"""

    def __init__(
        self, 
        chunks: List[Dict[str, Any]], 
        config: Dict[str, Any], 
        client: Optional[Any] = None
    ):
        """
        Local Chroma Retriever 초기화
        
        :param chunks: 초기 인덱싱할 청크 데이터 리스트
        :param config: 모델 및 DB 관련 설정 딕셔너리
        :param client: 전역 캐시 처리된 chromadb.PersistentClient (선택)
        """
        self.chunks = chunks
        self.config = config
        
        # 설정 파일 규격 처리 (embedding_model 또는 config 내 embedding.model_name 참조)
        embedding_cfg = self.config.get("embedding", {})
        model_name = embedding_cfg.get("model_name") if isinstance(embedding_cfg, dict) else None
        if not model_name:
            model_name = self.config.get("embedding_model", "dragonkue/BGE-m3-ko")

        cache_dir = self.config.get("cache_path")
        device = self.config.get("device", "cpu")
        
        # HuggingFace 로컬 임베딩 초기화
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=cache_dir,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # -------------------------------------------------------------------------
        # [수정] collection_name 오타 수정 (bidmate_localgit -> bidmate_local)
        # 및 persist_directory 팀 규격 경로(/data/processed/vector_db/local)로 고정
        # -------------------------------------------------------------------------
        persist_directory = self.config.get("persist_directory", "/data/processed/vector_db/local")
        collection_name = self.config.get("collection_name", "bidmate_local")
        
        # Chroma VectorDB 초기화 (클라이언트 캐시 주입 지원)
        if client is not None:
            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                client=client
            )
        else:
            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )

        # -------------------------------------------------------------------------
        # [추가] 초기 전달받은 chunks가 있을 경우 Document 변환 후 자동 적재/갱신
        # -------------------------------------------------------------------------
        if self.chunks:
            initial_docs = [
                Document(
                    page_content=c.get("text", ""),
                    metadata={
                        "chunk_id": c.get("chunk_id"),
                        "doc_id": c.get("doc_id"),
                        "agency": c.get("agency"),           # 기관명 메타데이터 필터용
                        "project_name": c.get("project_name") # 사업명 메타데이터 필터용
                    }
                )
                for c in self.chunks if c.get("text")
            ]
            if initial_docs:
                self.add_documents(initial_docs)

    # -------------------------------------------------------------------------
    # [수정] api_main.py 및 다중 인자 호환성을 위한 filters, agency, project_name, doc_id 파라미터 추가
    # -------------------------------------------------------------------------
    def search(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        agency: Optional[str] = None,
        project_name: Optional[str] = None,
        doc_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        공통 메타데이터 필터 조건(agency, project_name, doc_id)을 충족하는 고속 Chroma Vector 유사도 검색 수행
        """
        k = top_k or self.config.get("top_k", 4)
        
        # filters 딕셔너리 및 개별 인자 병합 처리
        filter_dict = filters or {}
        target_agency = agency or filter_dict.get("agency")
        target_project_name = project_name or filter_dict.get("project_name")
        target_doc_id = doc_id or filter_dict.get("doc_id")

        # Chroma 메타데이터 필터 딕셔너리 동적 빌드
        filter_conditions = []
        if target_agency:
            filter_conditions.append({"agency": target_agency})
        if target_project_name:
            filter_conditions.append({"project_name": target_project_name})
        if target_doc_id:
            filter_conditions.append({"doc_id": target_doc_id})
            
        search_filter = None
        if len(filter_conditions) == 1:
            search_filter = filter_conditions[0]
        elif len(filter_conditions) > 1:
            search_filter = {"$and": filter_conditions}

        # -------------------------------------------------------------------------
        # [수정] 오류 발생 시 필터 없이 전체 검색으로 Fallback하는 구문 제거
        # 검색 예외 발생 시 무조건 빈 결과([])를 반환하여 잘못된 정보 생성을 방지합니다.
        # -------------------------------------------------------------------------
        try:
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=search_filter
            )
        except Exception as e:
            print(f"[ERROR] Chroma DB 검색 실행 중 오류 발생 (필터 조건: {search_filter}): {e}")
            docs_with_scores = []
        
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

    def _calculate_hash(self, text: str) -> str:
        """청크 텍스트의 고유 MD5 해시값 계산 헬퍼 함수"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def add_documents(self, docs: List[Document]):
        """
        중복 및 데이터 최신 갱신 여부를 판별하여 선별적 인덱싱 및 Sync 수행
        """
        if not docs:
            return

        incoming_chunk_ids = []
        doc_map = {}
        
        for doc in docs:
            chunk_id = doc.metadata.get("chunk_id")
            new_hash = self._calculate_hash(doc.page_content)
            
            doc.metadata["content_hash"] = new_hash
            
            if chunk_id:
                incoming_chunk_ids.append(chunk_id)
                doc_map[chunk_id] = doc
            else:
                self.vector_store.add_documents(documents=[doc])

        existing_hash_map = {}
        if incoming_chunk_ids:
            existing_data = self.vector_store.get(
                where={"chunk_id": {"$in": incoming_chunk_ids}},
                include=["metadatas"]
            )
            
            if existing_data and "ids" in existing_data:
                for idx, existing_id in enumerate(existing_data["ids"]):
                    metadatas_list = existing_data.get("metadatas", [])
                    if metadatas_list and idx < len(metadatas_list) and metadatas_list[idx]:
                        existing_hash_map[existing_id] = metadatas_list[idx].get("content_hash")

        docs_to_insert = []
        ids_to_insert = []
        ids_to_delete = []

        for chunk_id, doc in doc_map.items():
            new_hash = doc.metadata["content_hash"]
            
            if chunk_id in existing_hash_map:
                old_hash = existing_hash_map[chunk_id]
                
                # 내용이 변경된 경우 기존 ID 삭제 후 재적재 대상 등록
                if old_hash != new_hash:
                    ids_to_delete.append(chunk_id)
                    docs_to_insert.append(doc)
                    ids_to_insert.append(chunk_id)
                else:
                    continue  # 동일한 해시값이 존재하면 추가 작업을 스킵
            else:
                docs_to_insert.append(doc)
                ids_to_insert.append(chunk_id)

        # 내용 변경 건 삭제
        if ids_to_delete:
            self.vector_store.delete(ids=ids_to_delete)
            
        # 신규 및 수정 문서 신규 적재
        if docs_to_insert:
            self.vector_store.add_documents(documents=docs_to_insert, ids=ids_to_insert)

        print(f"[VectorDB Update Sync] Target Chunk Total: {len(docs)}건 -> Deleted(Updated): {len(ids_to_delete)}건, Newly Saved: {len(docs_to_insert)}건 완료.")
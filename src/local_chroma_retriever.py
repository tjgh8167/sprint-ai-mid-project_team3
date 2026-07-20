from typing import Dict, Any, List, Optional
import hashlib
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
# deprecated 경고 해결을 위해 최신 전용 패키지로 변경
from langchain_chroma import Chroma

# 프로젝트 통합 공통 계약 규격(SearchResult) 임포트 (chunk_id, doc_id 포함)
from src.retriever import SearchResult

class LocalChromaRetriever:
    """Retrieval 2: HuggingFace 임베딩 + Chroma 검색기 구현체 (전체 메타데이터 추출 + 실시간 갱신 버전)"""

    def __init__(self, chunks: List[Dict[str, Any]], config: Dict[str, Any]):
        """
        Local Chroma Retriever 초기화 (질문 시점에는 저장된 Vector DB 로드만 수행)
        """
        self.chunks = chunks  # 질문 시점에는 일반적으로 빈 리스트([])가 유입되거나 무시.
        self.config = config
        
        # 설정 파일(default.yaml 등)의 규격에 맞춰 'embedding' 대신 'embedding_model'로 명확히 조회
        model_name = self.config.get("embedding_model", "dragonkue/BGE-m3-ko")
        cache_dir = self.config.get("cache_path")
        device = self.config.get("device", "cpu")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=cache_dir,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # 공유 경로인 '/data/processed/vector_db/local'을 최우선 기본값으로 확실하게 고정
        persist_directory = self.config.get("persist_directory", "/data/processed/vector_db/local")
        collection_name = self.config.get("collection_name", "bidmate_localgit")
        
        # langchain-chroma 버전에서는 persist_directory와 collection_name을 기반으로 
        # 실시간 적재 없이 기존 물리 DB를 Read-only 형태로 안전하게 로드.
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings, # LangChain 내부 인자명은 embedding_function
            persist_directory=persist_directory
        )

    def search(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        agency: Optional[str] = None,       # org_name이 아닌 'agency' 필터 키로 통일
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
            filter_conditions.append({"agency": agency})
        if project_name:
            filter_conditions.append({"project_name": project_name})
        if doc_id:
            filter_conditions.append({"doc_id": doc_id})
            
        search_filter = None
        if len(filter_conditions) == 1:
            search_filter = filter_conditions[0]  # [수정] 순수 단일 Dict 추출로 대괄호([]) 주입에 따른 Chroma 구문 오류 완벽 차단
        elif len(filter_conditions) > 1:
            search_filter = {"$and": filter_conditions}

        try:
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=search_filter
            )
        except Exception:
            # 필터 구조 충돌 예외 발생 시 인덱스 안정성 마진 확보를 위해 필터 없이 검색
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
        
        #원본 메타데이터 컬럼을 누락 없이 전체 다 뽑아내는 규격 조합
        results = []
        for doc, score in docs_with_scores:
            current_chunk_id = doc.metadata.get("chunk_id", "N/A")
            current_doc_id = doc.metadata.get("doc_id", "N/A")
            
            # doc.metadata 원본 딕셔너리를 전체 다 SearchResult에 주입합니다.
            results.append(
                SearchResult(
                    chunk_id=current_chunk_id,
                    doc_id=current_doc_id,
                    text=doc.page_content,
                    metadata=doc.metadata,  # 컬럼 선별 필터링 없이 덤프
                    score=float(score)
                )
            )
        return results

    def _calculate_hash(self, text: str) -> str:
        """청크 텍스트의 고유 MD5 해시값 계산 헬퍼 함수"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def add_documents(self, docs: List[Document]):
        """
        중복 및 데이터 최신 갱신 여부를 판별하여 선별적 인덱싱 수행
        - 대량의 chunk_id를 배치(Batch)로 일괄 조회하여 대규모 데이터 적재 시의 병목을 해결하고,
          내용이 달라진 경우 이전 구버전 데이터를 삭제 후 최신 내용으로 안전하게 갱신(Upsert)합니다.
        """
        if not docs:
            return

        # 1. 입력된 모든 문서의 해시 계산 및 chunk_id 수집
        incoming_chunk_ids = []
        doc_map = {}  # 고속 매핑을 위한 딕셔너리
        
        for doc in docs:
            chunk_id = doc.metadata.get("chunk_id")
            new_hash = self._calculate_hash(doc.page_content)
            doc.metadata["content_hash"] = new_hash
            
            if chunk_id:
                incoming_chunk_ids.append(chunk_id)
                doc_map[chunk_id] = doc
            else:
                # 안전장치: chunk_id가 없는 예외적인 문서는 즉시 삽입 대상으로 분류
                self.vector_store.add_documents(documents=[doc])

        # 2. [성능 최적화] DB에서 입력된 chunk_id들의 기존 메타데이터를 일괄(Batch) 조회
        existing_hash_map = {}
        if incoming_chunk_ids:
            # Chroma의 $in 연산자를 사용하여 단 한 번의 쿼리로 대량 조회
            existing_data = self.vector_store.get(
                where={"chunk_id": {"$in": incoming_chunk_ids}},
                include=["metadatas"]
            )
            
            # 조회된 결과 맵핑 (Chroma 반환 규격인 ids와 metadatas의 인덱스는 일치함)
            if existing_data and "ids" in existing_data:
                for idx, existing_id in enumerate(existing_data["ids"]):
                    metadatas_list = existing_data.get("metadatas", [])
                    # 특정 id에 해당하는 메타데이터가 정상 존재할 때만 해시 추출
                    if metadatas_list and idx < len(metadatas_list) and metadatas_list[idx]:
                        existing_hash_map[existing_id] = metadatas_list[idx].get("content_hash")

        # 3. 변경 내용 차분 분석 실행
        docs_to_insert = []
        ids_to_insert = []
        ids_to_delete = []

        for chunk_id, doc in doc_map.items():
            new_hash = doc.metadata["content_hash"]
            
            # 기존 DB에 해당 chunk_id가 이미 존재하는 경우
            if chunk_id in existing_hash_map:
                old_hash = existing_hash_map[chunk_id]
                
                # 내용이 바뀐 경우에만: 구버전 삭제 후 신규 버전 삽입 대상 지정 (갱신 답변 오류 해결)
                if old_hash != new_hash:
                    ids_to_delete.append(chunk_id)
                    docs_to_insert.append(doc)
                    ids_to_insert.append(chunk_id)
                else:
                    # 내용이 같으면 임베딩 및 저장 과정을 Skip하여 자원 절약
                    continue
            else:
                # 완전히 새로운 데이터인 경우: 삽입 대상 지정
                docs_to_insert.append(doc)
                ids_to_insert.append(chunk_id)

        # 4. 안전한 일괄 트랜잭션 처리 (구버전 삭제 후 새 버전 덮어쓰기)
        if ids_to_delete:
            self.vector_store.delete(ids=ids_to_delete)
            
        if docs_to_insert:
            # ids 파라미터에 chunk_id를 바인딩하여 고유 식별 및 데이터 정합성 보장
            self.vector_store.add_documents(documents=docs_to_insert, ids=ids_to_insert)

        print(f"[VectorDB Update Sync] Target Chunk Total: {len(docs)}건 -> Deleted(Updated): {len(ids_to_delete)}건, Newly Saved: {len(docs_to_insert)}건 완료.")


        from typing import Dict, Any, List, Optional
import hashlib
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
# [수정] deprecated 경고 해결을 위해 최신 전용 패키지로 변경
from langchain_chroma import Chroma

# 프로젝트 통합 공통 계약 규격(SearchResult) 임포트 (chunk_id, doc_id 포함)
from src.retriever import SearchResult

class LocalChromaRetriever:
    """Retrieval 2: HuggingFace 임베딩 + Chroma 검색기 구현체 (전체 메타데이터 추출 + 실시간 갱신 버전)"""

    def __init__(self, chunks: List[Dict[str, Any]], config: Dict[str, Any]):
        """
        Local Chroma Retriever 초기화 (질문 시점에는 저장된 Vector DB 로드만 수행)
        """
        self.chunks = chunks  # 질문 시점에는 일반적으로 빈 리스트([])가 유입되거나 무시.
        self.config = config
        
        # 설정 파일(default.yaml 등)의 규격에 맞춰 'embedding' 대신 'embedding_model'로 명확히 조회
        model_name = self.config.get("embedding_model", "dragonkue/BGE-m3-ko")
        cache_dir = self.config.get("cache_path")
        device = self.config.get("device", "cpu")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=cache_dir,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # [공유 경로 고정] 디렉토리 공유 경로인 '/data/processed/vector_db/local'을 기본값으로 강제 설정
        persist_directory = self.config.get("persist_directory", "/data/processed/vector_db/local")
        collection_name = self.config.get("collection_name", "bidmate_localgit")
        
        # langchain-chroma 버전에서는 persist_directory와 collection_name을 기반으로 
        # 실시간 적재 없이 기존 물리 DB를 Read-only 형태로 안전하게 로드.
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings, # LangChain 내부 인자명은 embedding_function
            persist_directory=persist_directory
        )

    def search(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        agency: Optional[str] = None,       # org_name이 아닌 'agency' 필터 키로 통일
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

        try:
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=search_filter
            )
        except Exception:
            # 필터 구조 충돌 예외 발생 시 인덱스 안정성 마진 확보를 위해 필터 없이 검색
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
        
        #doc.metadata의 모든 컬럼을 누락 없이 전체 다 뽑아내는 규격 조합
        results = []
        for doc, score in docs_with_scores:
            current_chunk_id = doc.metadata.get("chunk_id", "N/A")
            current_doc_id = doc.metadata.get("doc_id", "N/A")
            
            # doc.metadata 원본 딕셔너리를 전체 다 SearchResult에 주입하여 반환의 다양성 보장
            results.append(
                SearchResult(
                    chunk_id=current_chunk_id,
                    doc_id=current_doc_id,
                    text=doc.page_content,
                    metadata=doc.metadata,  # 컬럼 선별 필터링 없이 통째로 덤프
                    score=float(score)
                )
            )
        return results

    def _calculate_hash(self, text: str) -> str:
        """청크 텍스트의 고유 MD5 해시값 계산 헬퍼 함수"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def add_documents(self, docs: List[Document]):
        """
        [태훈님 적재 방식 개선 - 데이터 정합성 최적화 및 서호님 싱크 일치]
        - 데이터 적재 시, 원본 Document의 metadata 컬럼을 특정 필드로 선별 누락하지 않고 전체 다 보존하여 저장합니다.
        - 내용 해시(content_hash)를 검증하여 내용이 변경된 청크인 경우에만 이전 데이터를 지우고 새 임베딩을 생성(Upsert)하여 
          구버전 오답을 차단합니다.
        """
        if not docs:
            return

        # 1. 입력된 모든 문서의 해시 계산 및 chunk_id 수집
        incoming_chunk_ids = []
        doc_map = {}  # 고속 매핑을 위한 딕셔너리
        
        for doc in docs:
            chunk_id = doc.metadata.get("chunk_id")
            new_hash = self._calculate_hash(doc.page_content)
            
            # [중요] 기존 메타데이터 전체 컬럼 구조를 그대로 유지하면서 비교 기준용 content_hash만 주입
            doc.metadata["content_hash"] = new_hash
            
            if chunk_id:
                incoming_chunk_ids.append(chunk_id)
                doc_map[chunk_id] = doc
            else:
                # 안전장치: chunk_id가 없는 예외적인 문서는 즉시 삽입 대상으로 분류
                self.vector_store.add_documents(documents=[doc])

        # 2. [성능 최적화] DB에서 입력된 chunk_id들의 기존 메타데이터를 일괄(Batch) 조회
        existing_hash_map = {}
        if incoming_chunk_ids:
            # Chroma의 $in 연산자를 사용하여 단 한 번의 쿼리로 대량 조회
            existing_data = self.vector_store.get(
                where={"chunk_id": {"$in": incoming_chunk_ids}},
                include=["metadatas"]
            )
            
            # 조회된 결과 맵핑 (Chroma 반환 규격인 ids와 metadatas의 인덱스는 일치함)
            if existing_data and "ids" in existing_data:
                for idx, existing_id in enumerate(existing_data["ids"]):
                    metadatas_list = existing_data.get("metadatas", [])
                    # 특정 id에 해당하는 메타데이터가 정상 존재할 때만 해시 추출
                    if metadatas_list and idx < len(metadatas_list) and metadatas_list[idx]:
                        existing_hash_map[existing_id] = metadatas_list[idx].get("content_hash")

        # 3. 변경 내용 차분 분석 실행
        docs_to_insert = []
        ids_to_insert = []
        ids_to_delete = []

        for chunk_id, doc in doc_map.items():
            new_hash = doc.metadata["content_hash"]
            
            # 기존 DB에 해당 chunk_id가 이미 존재하는 경우
            if chunk_id in existing_hash_map:
                old_hash = existing_hash_map[chunk_id]
                
                # 내용이 바뀐 경우 구버전 삭제 후 신규 버전 삽입 대상 지정 (갱신 답변 오류 해결)
                if old_hash != new_hash:
                    ids_to_delete.append(chunk_id)
                    docs_to_insert.append(doc)
                    ids_to_insert.append(chunk_id)
                else:
                    # 내용이 같으면 임베딩 및 저장 과정을 Skip하여 자원 절약
                    continue
            else:
                # 완전히 새로운 데이터인 경우: 삽입 대상 지정
                docs_to_insert.append(doc)
                ids_to_insert.append(chunk_id)

        # 4. 안전한 일괄 트랜잭션 처리 (구버전 삭제 후 새 버전 덮어쓰기)
        if ids_to_delete:
            self.vector_store.delete(ids=ids_to_delete)
            
        if docs_to_insert:
            # ids 파라미터에 chunk_id를 바인딩하여 고유 식별 및 데이터 정합성 보장
            self.vector_store.add_documents(documents=docs_to_insert, ids=ids_to_insert)

        print(f"[VectorDB Update Sync] Target Chunk Total: {len(docs)}건 -> Deleted(Updated): {len(ids_to_delete)}건, Newly Saved: {len(docs_to_insert)}건 완료.")
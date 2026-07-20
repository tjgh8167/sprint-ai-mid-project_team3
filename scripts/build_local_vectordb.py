from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def build_vector_db(chunks: List[Dict[str, Any]], config: Dict[str, Any]):
    """청크 데이터를 받아 Chroma Vector DB를 생성하고 저장하는 함수"""
    
    model_name = config.get("embedding", "dragonkue/BGE-m3-ko")
    cache_dir = config.get("cache_path")
    device = config.get("device", "cpu")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=cache_dir,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    persist_directory = config.get("persist_directory", "vector_db/local")
    collection_name = config.get("collection_name", "bidmate_localgit")

    # Document 객체 변환 (metadata에 chunk_id, doc_id 포함)
    documents = []
    for chunk in chunks:
        documents.append(
            Document(
                page_content=chunk.get("text", ""),
                metadata={
                    "chunk_id": chunk.get("chunk_id"),
                    "doc_id": chunk.get("doc_id"),
                    "agency": chunk.get("agency"),
                    "project_name": chunk.get("project_name")
                }
            )
        )

    # Chroma DB에 저장
    print("Embedding 및 DB 저장 시작...")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    # 명시적 저장 (persist)
    vector_store.persist() 
    print("DB 구축 완료!")
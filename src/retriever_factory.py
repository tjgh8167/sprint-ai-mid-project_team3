from typing import List, Dict, Any, Optional
from src.retriever import SimpleRetriever


def create_retriever(
    chunks: List[Dict[str, Any]], 
    retrieval_config: Dict[str, Any], 
    profile: Optional[str] = None
):
    """
    설정 프로필에 따라 알맞은 Retriever 인스턴스를 동적으로 생성하고 데이터를 주입하는 팩토리 함수
    """
    selected_profile = profile or retrieval_config.get("active_profile", "baseline")
    profiles = retrieval_config.get("profiles", {})

    # 1. Baseline (SimpleRetriever)
    if selected_profile == "baseline":
        return SimpleRetriever(chunks)
        
    # 2. OpenAI Profile
    if selected_profile == "openai":
        from src.openai_chroma_retriever import OpenAIChromaRetriever
        openai_cfg = profiles.get("openai", {})
        merged_openai_config = {**retrieval_config, **openai_cfg}
        return OpenAIChromaRetriever(chunks=chunks, config=merged_openai_config)
        
    # 3. Local HF Profile (--profile local)
    if selected_profile == "local":
        from src.local_chroma_retriever import LocalChromaRetriever
        local_cfg = profiles.get("local", {})
        # [수정] 상위 retrieval_config의 공통 설정(top_k, persist_directory 등)과 프로필 개별 설정 병합
        merged_local_config = {**retrieval_config, **local_cfg}
        
        # chunks 데이터와 병합된 최적 설정 딕셔너리를 함께 주입
        return LocalChromaRetriever(chunks=chunks, config=merged_local_config)

    raise ValueError(f"지원하지 않는 Retrieval 프로필입니다: {selected_profile}")
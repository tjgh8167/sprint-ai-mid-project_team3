from src.local_chroma_retriever import LocalChromaRetriever
from src.openai_chroma_retriever import OpenAIChromaRetriever
from src.retriever import SimpleRetriever

def create_retriever(chunks: list[dict], retrieval_config: dict, profile: str | None = None):
    """
    설정 프로필에 따라 알맞은 Retriever 인스턴스를 동적으로 생성하고 데이터를 주입하는 팩토리 함수
    """
    selected_profile = profile or retrieval_config.get("active_profile", "baseline")
    profiles = retrieval_config.get("profiles", {})

    if selected_profile == "baseline":
        return SimpleRetriever(chunks)
        
    if selected_profile == "openai":
        return OpenAIChromaRetriever(config=retrieval_config)
        
    if selected_profile == "local":
        # 표준 계약 스펙에 일치하도록 'chunks'와 'profiles["local"]' 인자 2개를 정확하게 순서대로 전달하여 반환합니다.
        return LocalChromaRetriever(chunks, profiles["local"])

    raise ValueError(f"지원하지 않는 Retrieval 프로필입니다: {selected_profile}")
from dataclasses import asdict

from src.retriever import SearchResult


SYSTEM_RULE = "문서에 있는 내용만 근거로 답변하고, 문서에 없으면 모른다고 답합니다."


def build_context(results: list[SearchResult]) -> str:
    context_blocks = []
    for idx, result in enumerate(results, start=1):
        metadata = result.metadata
        title = metadata.get("title", "제목 없음")
        agency = metadata.get("agency", "기관 없음")
        source = metadata.get("file_name", "출처 없음")
        context_blocks.append(
            f"[{idx}] title={title} agency={agency} source={source} score={result.score}\n{result.text}"
        )
    return "\n\n".join(context_blocks)


def generate_answer(question: str, results: list[SearchResult]) -> dict:
    if not results:
        return {
            "answer": "관련 문서 내용을 찾지 못했습니다. 원본 문서나 검색 조건을 다시 확인해 주세요.",
            "sources": [],
        }

    context = build_context(results)
    answer = (
        f"질문: {question}\n\n"
        f"답변: 검색된 RFP 문서 기준으로 보면, 핵심 근거는 다음과 같습니다.\n"
        f"{context}\n\n"
        f"주의: 이 답변은 현재 검색된 청크만 기반으로 한 기본 생성 결과입니다."
    )

    return {
        "answer": answer,
        "sources": [asdict(result) for result in results],
        "system_rule": SYSTEM_RULE,
    }

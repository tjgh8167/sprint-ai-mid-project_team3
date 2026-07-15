# src/rag_engine.py
import os
from dataclasses import asdict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.retriever import SearchResult

load_dotenv()

SYSTEM_RULE = (
    "당신은 공공입찰 및 제안요청서(RFP) 분석을 돕는 최고 수준의 전문 AI 어시스턴트입니다.\n"
    "반드시 아래에 제공된 맥락(Context) 문서 내용에만 기반하여 질문에 답변하십시오.\n\n"
    "[작성 원칙]\n"
    "1. 제공된 맥락 내에서만 사실에 기반하여 간결하고 명확하게 답변할 것.\n"
    "2. 주어진 맥락으로 답변이 어려운 경우, 절대 추측하거나 지어내지 말고 '제공된 문서에서 근거를 찾을 수 없어 확인할 수 없습니다'라고 단호하게 답할 것.\n"
    "3. 답변 하단에는 반드시 참고한 문서의 출처(예: 파일명 또는 사업명)를 명시할 것.\n"
    "4. 사용자가 읽기 편하도록 적절히 글머리 기호(-, *)를 사용하여 한국어로 작성할 것.\n\n"
    "[출력 어조 규칙]\n"
    "- 모든 답변은 신뢰감을 주는 **정중한 해요체/하십시오체('~합니다', '~안내해 드립니다')** 또는 **깔끔한 명사형/개조식(~함, ~기재)** 중 하나로 일관되게 작성할 것.\n"
    "- 반말, 혼잣말, 혹은 문장이 중간에 끊기는 현상이 절대 없도록 문장 끝맺음을 완벽하게 마무리할 것."
)

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

def generate_answer(question: str, results: list[SearchResult], config: dict = None) -> dict:
    if not results:
        return {
            "answer": "관련 문서 내용을 찾지 못했습니다. 원본 문서나 검색 조건을 다시 확인해 주세요.",
            "sources": [],
        }

    # 1. 설정값 로드
    gen_config = config.get("generation", {}) if config else {}
    model_name = gen_config.get("model", "gpt-5-nano")
    temperature = gen_config.get("temperature", 0.1)
    top_p = gen_config.get("top_p", 0.95)
    max_tokens = gen_config.get("max_tokens", 1000)

    # 2. LLM 및 체인 구성
    # gpt-5/o-시리즈(reasoning 모델)는 temperature/top_p가 1로 고정되어 있어
    # 커스텀 값을 보내면 400 Unsupported parameter 에러가 발생한다.
    # 해당 모델일 땐 두 파라미터를 빼고, 그 외 모델(gpt-4o 등)일 땐 YAML 값을 그대로 적용한다.
    is_reasoning_model = model_name.startswith(("gpt-5", "o1", "o3", "o4"))

    llm_kwargs = {"model": model_name, "max_tokens": max_tokens}
    if not is_reasoning_model:
        llm_kwargs["temperature"] = temperature
        llm_kwargs["top_p"] = top_p

    llm = ChatOpenAI(**llm_kwargs)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_RULE + "\n\nContext:\n{context}"),
        ("human", "{question}")
    ])
    
    # 여기서 'chain' 변수가 정의됩니다.
    chain = prompt | llm | StrOutputParser()

    # 3. 컨텍스트 빌드 후 실행
    context = build_context(results)
    answer = chain.invoke({"context": context, "question": question})

    return {
        "answer": answer,
        "sources": [asdict(result) for result in results],
        "system_rule": SYSTEM_RULE,
    }
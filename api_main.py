import argparse
import json
from pathlib import Path

import yaml

from src.parser_chunker import demo_chunks, load_chunks_jsonl
from src.rag_engine import generate_answer
from src.retriever_factory import create_retriever


def load_config(path: str = "config/default.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def load_chunks(chunks_path: str) -> list[dict]:
    path = Path(chunks_path)
    if path.exists():
        return load_chunks_jsonl(path)
    return demo_chunks()


def run(
    question: str,
    config_path: str = "config/default.yaml",
    profile: str | None = None,
    filters: dict | None = None,  # 필터 로직 추가
) -> dict:
    config = load_config(config_path)
    chunks = load_chunks(config["paths"]["chunks"])
    retriever = create_retriever(chunks, config["retrieval"], profile)

    selected_profile = profile or config["retrieval"]["active_profile"]
    try:
        results = retriever.search(question, top_k=config["retrieval"]["top_k"], filters=filters)
    except NotImplementedError as exc:
        return {
            "answer": f"'{selected_profile}' 프로필은 아직 준비되지 않았습니다.\n사유: {exc}",
            "sources": [],
        }

    return generate_answer(question, results, config)


def main(default_profile: str = "baseline") -> None:
    parser = argparse.ArgumentParser(description="BidMate 최소 End-to-End RAG 실행")
    parser.add_argument("question", nargs="?", default="AI 학습지원 플랫폼의 주요 기능 요구사항을 알려줘")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument(
        "--profile",
        choices=["baseline", "openai", "local"],
        default=default_profile,
    )
    # 필터 로직 추가
    parser.add_argument("--filters", help="검색 필터를 JSON 형식으로 입력 (예: '{\"agency\": \"가상디지털진흥원\"}')")

    args = parser.parse_args()

    # JSON 문자열을 딕셔너리로 변환, 필터 로직
    filter_dict = json.loads(args.filters) if args.filters else None

    response = run(args.question, args.config, args.profile, filter_dict)
    print(response["answer"])

    if response["sources"]:
        print(f"\n[참고 문서 {len(response['sources'])}건]")
        for src in response["sources"]:
            file_name = src["metadata"].get("file_name", "출처 없음")
            print(f"- {file_name} (chunk_id: {src['chunk_id']}, score: {src['score']})")


if __name__ == "__main__":
    main()
import argparse
from dataclasses import asdict
from pathlib import Path

import yaml

from src.parser_chunker import build_chunks, save_chunks_jsonl


DEFAULT_METADATA = {
    "agency": "가상디지털진흥원",
    "title": "2026년 공공 AI 학습지원 플랫폼 구축 사업",
    "document_type": "가상 RFP",
}


def load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="샘플 RFP 파일을 chunks.jsonl로 변환")
    parser.add_argument("--input", default="samples/raw/sample_rfp.txt")
    parser.add_argument("--output", default="samples/processed/sample_chunks.jsonl")
    parser.add_argument("--doc-id", default="sample_rfp")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    chunks = build_chunks(
        file_path=args.input,
        doc_id=args.doc_id,
        metadata=DEFAULT_METADATA,
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"],
    )
    save_chunks_jsonl(chunks, args.output)
    print(f"saved {len(chunks)} chunks to {args.output}")
    if chunks:
        print(asdict(chunks[0])["text"][:300])


if __name__ == "__main__":
    main()

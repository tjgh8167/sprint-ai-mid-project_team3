import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


SOURCE_COLUMNS = [
    "공고 번호",
    "공고 차수",
    "사업명",
    "사업 금액",
    "발주 기관",
    "공개 일자",
    "입찰 참여 시작일",
    "입찰 참여 마감일",
    "사업 요약",
    "파일형식",
    "파일명",
]

OUTPUT_COLUMNS = [
    "doc_id",
    "file_name",
    "project_name",
    "title",
    "agency",
    "document_type",
    "file_type",
    "page",
    "source_path",
    "source_exists",
    "notice_number",
    "notice_round",
    "budget",
    "published_at",
    "bid_start_at",
    "bid_end_at",
    "summary",
]

REQUIRED_FIELDS = [
    "doc_id",
    "file_name",
    "project_name",
    "title",
    "agency",
    "document_type",
    "source_path",
]


# YAML 설정 파일을 읽습니다.
def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8-sig") as file:
        return yaml.safe_load(file)


# 원본 CSV의 필수 열이 모두 있는지 확인합니다.
def validate_source_columns(source: pd.DataFrame) -> None:
    missing_columns = [column for column in SOURCE_COLUMNS if column not in source.columns]
    if missing_columns:
        raise ValueError(f"원본 CSV에 없는 필수 열: {', '.join(missing_columns)}")


# 결측치를 빈 문자열로 바꾸고 앞뒤 공백을 제거합니다.
def clean_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


# 원본 data_list.csv를 Retriever 공통 metadata.csv 형식으로 변환합니다.
def build_metadata(source_path: str | Path, raw_documents_path: str | Path) -> pd.DataFrame:
    source = pd.read_csv(source_path, dtype=str, encoding="utf-8-sig")
    validate_source_columns(source)

    raw_documents_path = Path(raw_documents_path)
    rows = []

    for index, row in source.iterrows():
        file_name = clean_value(row["파일명"])
        document_type = clean_value(row["파일형식"]).lower().lstrip(".")
        if not document_type:
            document_type = Path(file_name).suffix.lower().lstrip(".")

        project_name = clean_value(row["사업명"])
        source_path_value = raw_documents_path / file_name

        rows.append(
            {
                "doc_id": f"doc_{index + 1:03d}",
                "file_name": file_name,
                "project_name": project_name,
                "title": project_name,
                "agency": clean_value(row["발주 기관"]),
                "document_type": document_type,
                "file_type": document_type,
                "page": "",
                "source_path": str(source_path_value),
                "source_exists": source_path_value.is_file(),
                "notice_number": clean_value(row["공고 번호"]),
                "notice_round": clean_value(row["공고 차수"]),
                "budget": clean_value(row["사업 금액"]),
                "published_at": clean_value(row["공개 일자"]),
                "bid_start_at": clean_value(row["입찰 참여 시작일"]),
                "bid_end_at": clean_value(row["입찰 참여 마감일"]),
                "summary": clean_value(row["사업 요약"]),
            }
        )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


# 표준 metadata.csv의 행 수, 필수값, 파일 매칭 상태를 검증합니다.
def validate_metadata(metadata: pd.DataFrame) -> dict[str, object]:
    missing_columns = [column for column in REQUIRED_FIELDS if column not in metadata.columns]
    if missing_columns:
        raise ValueError(f"표준 metadata.csv에 없는 필수 열: {', '.join(missing_columns)}")

    empty_required = {
        column: int(metadata[column].fillna("").astype(str).str.strip().eq("").sum())
        for column in REQUIRED_FIELDS
    }
    duplicate_doc_ids = int(metadata["doc_id"].duplicated().sum())
    missing_sources = int((~metadata["source_exists"]).sum())

    return {
        "row_count": len(metadata),
        "empty_required": empty_required,
        "duplicate_doc_ids": duplicate_doc_ids,
        "missing_sources": missing_sources,
    }


# 표준 metadata.csv를 UTF-8 BOM 형식으로 저장합니다.
def save_metadata(metadata: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_path, index=False, encoding="utf-8-sig")


# 기존 청크의 metadata를 표준 metadata.csv 기준으로 보강합니다.
def synchronize_chunk_metadata(chunks_path: str | Path, metadata: pd.DataFrame) -> dict[str, int]:
    chunks_path = Path(chunks_path)
    metadata_by_doc_id = {
        row["doc_id"]: row
        for row in metadata.to_dict(orient="records")
    }
    output_path = chunks_path.with_suffix(chunks_path.suffix + ".tmp")

    chunk_count = 0
    missing_doc_ids = 0

    with (
        chunks_path.open("r", encoding="utf-8") as source,
        output_path.open("w", encoding="utf-8") as target,
    ):
        for line in source:
            if not line.strip():
                continue

            chunk = json.loads(line)
            doc_id = chunk["doc_id"]
            document_metadata = metadata_by_doc_id.get(doc_id)
            if document_metadata is None:
                missing_doc_ids += 1
                continue

            chunk["metadata"].update(document_metadata)
            target.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            chunk_count += 1

    if missing_doc_ids:
        output_path.unlink(missing_ok=True)
        raise ValueError(f"metadata.csv에 없는 doc_id를 가진 청크: {missing_doc_ids}건")

    output_path.replace(chunks_path)
    return {"chunk_count": chunk_count, "missing_doc_ids": missing_doc_ids}


# 설정 경로를 사용해 표준 metadata.csv를 생성하고 청크 metadata를 동기화합니다.
def main() -> None:
    parser = argparse.ArgumentParser(description="원본 RFP metadata를 표준 metadata.csv로 변환합니다.")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]

    metadata = build_metadata(
        source_path=paths["metadata"],
        raw_documents_path=paths["raw_documents"],
    )
    validation = validate_metadata(metadata)
    save_metadata(metadata, paths["normalized_metadata"])
    chunk_result = synchronize_chunk_metadata(paths["chunks"], metadata)

    print(f"표준 메타데이터: {validation['row_count']}건")
    print(f"중복 doc_id: {validation['duplicate_doc_ids']}건")
    print(f"원본 파일 미매칭: {validation['missing_sources']}건")
    print("필수값 누락:", validation["empty_required"])
    print(f"metadata 저장 경로: {paths['normalized_metadata']}")
    print(f"metadata 동기화 청크: {chunk_result['chunk_count']}건")


if __name__ == "__main__":
    main()

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict


def read_document(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8-sig")

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    if suffix == ".hwp":
        raise NotImplementedError("HWP parsing은 Data Engineer가 olefile 또는 hwp 변환기로 구현합니다.")

    raise ValueError(f"지원하지 않는 파일 형식입니다: {path.suffix}")


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks = []
    start = 0
    while start < len(cleaned):
        end = start + chunk_size
        chunks.append(cleaned[start:end])
        if end >= len(cleaned):
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks


def build_chunks(
    file_path: str | Path,
    doc_id: str,
    metadata: dict | None = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    path = Path(file_path)
    text = read_document(path)
    base_metadata = {
        "file_name": path.name,
        "source_path": str(path),
        **(metadata or {}),
    }

    return [
        Chunk(
            chunk_id=f"{doc_id}_chunk_{idx:04d}",
            doc_id=doc_id,
            text=chunk,
            metadata=base_metadata,
        )
        for idx, chunk in enumerate(chunk_text(text, chunk_size, chunk_overlap), start=1)
    ]


def save_chunks_jsonl(chunks: list[Chunk], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def load_chunks_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def demo_chunks() -> list[dict]:
    text = (
        "가상 RFP 샘플 문서입니다. 발주기관은 가상디지털진흥원이고, "
        "사업명은 공공 AI 학습지원 플랫폼 구축 사업입니다. 주요 요구사항은 교육과정 추천, "
        "학습 이력 관리, 관리자 통계 화면 제공입니다. 제출 방식은 나라장터 온라인 제출이며, "
        "제출 마감일과 예산은 실제 원본 문서 메타데이터를 기준으로 확인해야 합니다."
    )
    chunk = Chunk(
        chunk_id="demo_doc_chunk_0001",
        doc_id="demo_doc",
        text=text,
        metadata={
            "title": "공공 AI 학습지원 플랫폼 구축 사업",
            "agency": "가상디지털진흥원",
            "file_name": "sample_rfp.txt",
        },
    )
    return [asdict(chunk)]

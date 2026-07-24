import argparse
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path

# 라이브러리는 개인 가상환경에 설치하고, 대용량 Qwen 모델만 팀 공용 캐시를 사용한다.
os.environ.setdefault("HF_HOME", "/data/model_cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fitz
import pandas as pd
import torch
import yaml
from paddleocr import PaddleOCR
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from src.ocr_extractor import extract_hwp_images, extract_pdf_images

def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def image_size(image_bytes: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(image_bytes)) as image:
        return image.size


def document_images(path: Path, min_pdf_text_length: int) -> list[dict]:
    if path.suffix.lower() == ".hwp":
        return extract_hwp_images(path)

    images = extract_pdf_images(path)
    document = fitz.open(path)
    try:
        for page_number, page in enumerate(document, start=1):
            if len(page.get_text("text").strip()) >= min_pdf_text_length:
                continue
            rendered = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            images.append(
                {
                    "page_number": page_number,
                    "image_number": None,
                    "image_bytes": rendered.tobytes("png"),
                    "image_extension": "png",
                    "source_type": "rendered_pdf_page",
                }
            )
    finally:
        document.close()
    return images


def image_type(vlm_text: str) -> str:
    match = re.search(
        r"^유형:\s*(diagram|form|table|logo|photo|other)\b",
        vlm_text,
        re.MULTILINE,
    )
    return match.group(1) if match else "other"


class MultimodalExtractor:
    def __init__(self, model_name: str):
        self.ocr = PaddleOCR(
            lang="korean",
            ocr_version="PP-OCRv5",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
        )
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def extract(self, image_path: str) -> tuple[str, str]:
        ocr_result = self.ocr.predict(image_path)[0]
        ocr_text = "\n".join(ocr_result["rec_texts"]).strip()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {
                        "type": "text",
                        "text": (
                            "이미지를 분석해 주세요. 반드시 첫 줄을 "
                            "'유형: diagram|form|table|logo|photo|other' 형식으로 쓰고, "
                            "둘째 줄부터 이미지에 실제로 보이는 내용만 한국어로 정리하세요. "
                            "diagram이면 모든 박스의 문구와 화살표 연결 관계를 빠짐없이 적고, "
                            "form 또는 table이면 문서 제목과 핵심 항목을 적으세요. "
                            "읽을 수 없는 내용은 추측하지 마세요."
                        ),
                    },
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
        )
        output_ids = [
            row[len(input_ids):]
            for input_ids, row in zip(inputs.input_ids, generated_ids)
        ]
        vlm_text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return ocr_text, vlm_text


def report_row(
    doc_id: str,
    file_name: str,
    image: dict,
    image_hash: str,
    status: str,
    reason: str,
    width: int | None = None,
    height: int | None = None,
) -> dict:
    return {
        "doc_id": doc_id,
        "file_name": file_name,
        "source_type": image.get("source_type", "embedded_image"),
        "page_number": image.get("page_number"),
        "image_number": image.get("image_number"),
        "image_sha256": image_hash,
        "width": width,
        "height": height,
        "status": status,
        "reason": reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    paths = config["paths"]
    options = config["multimodal"]
    metadata = pd.read_csv(paths["metadata"], encoding="utf-8")
    raw_documents = Path(paths["raw_documents"])
    if args.limit is not None:
        metadata = metadata.head(args.limit)

    extractor = MultimodalExtractor(options["model"])
    records: list[dict] = []
    report: list[dict] = []

    for index, row in metadata.iterrows():
        doc_id = f"doc_{index + 1:03d}"
        file_name = str(row["파일명"]).strip()
        file_path = raw_documents / file_name
        if not file_path.is_file():
            report.append(
                {
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "status": "failed",
                    "reason": "source_file_not_found",
                }
            )
            continue

        seen_hashes: set[str] = set()
        try:
            images = document_images(file_path, options["min_pdf_text_length"])
        except Exception as error:
            report.append(
                {
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "status": "failed",
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            continue

        for image in images:
            payload = image["image_bytes"]
            image_hash = sha256(payload)
            if image_hash in seen_hashes:
                report.append(
                    report_row(
                        doc_id, file_name, image, image_hash,
                        "duplicate", "document_image_duplicate"
                    )
                )
                continue
            seen_hashes.add(image_hash)

            try:
                width, height = image_size(payload)
                if width < options["min_width"] or height < options["min_height"]:
                    report.append(
                        report_row(
                            doc_id, file_name, image, image_hash,
                            "excluded", "image_too_small", width, height
                        )
                    )
                    continue

                extension = image.get("image_extension", "png")
                with tempfile.NamedTemporaryFile(suffix=f".{extension}") as temporary:
                    temporary.write(payload)
                    temporary.flush()
                    ocr_text, vlm_text = extractor.extract(temporary.name)

                records.append(
                    {
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "file_type": file_path.suffix.lower().lstrip("."),
                        "source_type": image.get("source_type", "embedded_image"),
                        "page_number": image.get("page_number"),
                        "image_number": image.get("image_number"),
                        "image_sha256": image_hash,
                        "width": width,
                        "height": height,
                        "ocr_engine": "PaddleOCR PP-OCRv5",
                        "ocr_text": ocr_text,
                        "vlm_model": options["model"],
                        "image_type": image_type(vlm_text),
                        "vlm_text": vlm_text,
                    }
                )
                report.append(
                    report_row(
                        doc_id, file_name, image, image_hash,
                        "applied", "", width, height
                    )
                )
            except Exception as error:
                report.append(
                    report_row(
                        doc_id, file_name, image, image_hash,
                        "failed", f"{type(error).__name__}: {error}"
                    )
                )

    output = Path(paths["multimodal_documents"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    pd.DataFrame(report).to_csv(paths["multimodal_report"], index=False, encoding="utf-8")

    print(f"멀티모달 처리 완료: {len(records)}건")
    print(f"결과 파일: {output}")
    print(f"처리 리포트: {paths['multimodal_report']}")


if __name__ == "__main__":
    main()

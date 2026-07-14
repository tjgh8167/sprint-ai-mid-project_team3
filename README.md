# 입찰메이트 (BidMate) - 사내 RAG 입찰지원 시스템

## 1. 프로젝트 개요

공공·기업 RFP 문서를 기반으로 질문과 관련된 내용을 검색하고 요약하는 RAG 시스템입니다.
가상 RFP로 최소 End-to-End 흐름을 먼저 실행한 뒤 실제 PDF/HWP 데이터와 모델을 연결합니다.

- **진행 기간**: 2026년 7월 10일 ~ 2026년 8월 3일
- **목표**: 주요 요구사항, 발주기관, 예산, 제출 방식 등을 빠르게 검색하고 근거와 함께 답변합니다.

```text
RFP 문서 → 파싱·청킹 → chunks.jsonl → 임베딩·Chroma 검색 → LLM 답변 → 평가
```

## 2. 회의에서 확정한 구현 방향

- Retrieval 1과 Retrieval 2는 같은 `chunks.jsonl`을 입력으로 사용하고 처음부터 병렬 개발합니다.
- Retrieval 1은 OpenAI API 임베딩, Retrieval 2는 HuggingFace 로컬 임베딩을 사용합니다.
- 두 Retrieval 모두 Vector DB는 Chroma로 통일합니다.
- 각 파이프라인은 기본 유사도 검색으로 베이스라인을 먼저 완성합니다.
- 이후 MMR, Hybrid Search, Re-ranking을 추가 실험하고 성능을 비교합니다.
- Generation은 두 Retrieval이 반환하는 공통 결과 형식 하나만 사용합니다.
- Generation 모델과 프롬프트는 Generation 담당자가 비교 후 선정합니다.
- FAISS 등 다른 Vector DB 비교는 현재 기본 범위에서 제외합니다.

## 3. 팀원 역할

| 담당자 | 역할 | 주요 업무 | 주요 수정 파일 |
| :--- | :--- | :--- | :--- |
| 유재열 | PM + Data Engineer | 일정·이슈·PR 관리, PDF/HWP 파싱, 메타데이터, 청킹, 공통 입출력 계약 및 통합 | `src/parser_chunker.py`, `scripts/build_chunks.py`, `config/default.yaml` |
| 정서호 | Retrieval 1 | OpenAI 임베딩 모델 선정, OpenAI용 Chroma DB 구축, 기본 검색 및 고도화 실험 | `src/openai_chroma_retriever.py` |
| 이태훈 | Retrieval 2 | HuggingFace 로컬 임베딩 모델 선정, 로컬용 Chroma DB 구축, 기본 검색 및 고도화 실험 | `src/local_chroma_retriever.py` |
| 김효섭 | Generation | LLM 선정, 프롬프트, 근거 기반 답변, 출처와 대화 흐름 구현 | `src/rag_engine.py`, `api_main.py` |

`src/retriever.py`와 `src/retriever_factory.py`는 두 Retrieval이 함께 사용하는 계약 파일입니다. 변경이 필요하면 먼저 팀에 공유합니다.

## 4. 파트별 입력·출력 계약

| 파트 | 입력 | 출력 |
| :--- | :--- | :--- |
| Data | PDF/HWP/TXT, `data_list.csv` | `chunks.jsonl` |
| Retrieval 1 | 질문, `chunks.jsonl` | OpenAI 임베딩·Chroma 검색 결과 |
| Retrieval 2 | 질문, `chunks.jsonl` | 로컬 임베딩·Chroma 검색 결과 |
| Generation | 질문, 공통 검색 결과 | 최종 답변과 출처 |
| Evaluation | 평가 질문·정답, 실행 결과 | 검색·답변 품질, 속도, 비용 비교 |

`chunks.jsonl` 한 줄 형식:

```json
{
  "chunk_id": "doc001_chunk_0001",
  "doc_id": "doc001",
  "text": "문서 내용...",
  "metadata": {
    "title": "사업명",
    "agency": "발주기관",
    "file_name": "원본파일.pdf",
    "page": 3
  }
}
```

두 Retrieval이 반드시 맞춰야 하는 검색 결과 형식:

```json
{
  "chunk_id": "doc001_chunk_0001",
  "doc_id": "doc001",
  "text": "관련 청크 내용...",
  "metadata": {
    "title": "사업명",
    "agency": "발주기관",
    "file_name": "원본파일.pdf"
  },
  "score": 0.87
}
```

## 5. 병렬 개발 방식

```text
유재열: PDF/HWP → chunks.jsonl ─┬─→ 정서호: OpenAI 임베딩 + Chroma ─┐
                                └─→ 이태훈: 로컬 임베딩 + Chroma ───┤
가상 RFP + baseline 결과 ───────────→ 김효섭: Generation ────────────┤
                                                                  ↓
                                                    공통 평가 및 최종 조합 선정
```

- Data 담당은 공통 `chunks.jsonl` 형식을 보장합니다.
- Retrieval 담당자는 별도 Chroma 저장 경로를 사용하므로 DB가 충돌하지 않습니다.
- Generation 담당자는 실제 Retrieval 완성 전에도 baseline 검색 결과로 개발할 수 있습니다.
- 통합 시 `retrieval.active_profile`만 바꿔 같은 Generation과 연결합니다.

## 6. 프로젝트 구조

```text
sprint-ai-mid-project_team3/
├── README.md
├── .gitignore
├── config/
│   └── default.yaml
├── data/
│   ├── raw/                           # 실제 원본, Git 업로드 금지
│   └── processed/                     # 실제 chunks.jsonl, Git 업로드 금지
├── samples/
│   ├── raw/sample_rfp.txt             # 병렬 개발용 가상 RFP
│   └── processed/sample_chunks.jsonl
├── scripts/
│   └── build_chunks.py
├── src/
│   ├── parser_chunker.py              # Data 담당
│   ├── retriever.py                   # 공통 결과 형식과 baseline
│   ├── retriever_factory.py           # Retrieval 프로필 선택
│   ├── openai_chroma_retriever.py     # Retrieval 1 담당
│   ├── local_chroma_retriever.py      # Retrieval 2 담당
│   └── rag_engine.py                  # Generation 담당
├── notebook/
├── api_main.py                        # 공통 실행 진입점
├── gcp_main.py                        # 로컬 Retrieval 실행 진입점
└── evaluate.py                        # 프로필별 평가
```

Vector DB는 Git에 올리지 않고 VM에 분리해 저장합니다.

```text
vector_db/
├── openai/
└── local/
```

## 7. 실행 방법

가상환경을 활성화한 뒤 가상 RFP를 청킹합니다.

```bash
python -m scripts.build_chunks
```

현재 동작하는 최소 End-to-End baseline:

```bash
python api_main.py "사업 예산과 수행 기간은 어떻게 돼?" --profile baseline
python evaluate.py --profile baseline
```

각 Retrieval 구현 완료 후:

```bash
python api_main.py "사업 예산과 수행 기간은 어떻게 돼?" --profile openai
python api_main.py "사업 예산과 수행 기간은 어떻게 돼?" --profile local
python evaluate.py --profile openai
python evaluate.py --profile local
```

`sample_rfp.txt`는 실제 기관이나 사업과 관련 없는 가상 문서입니다.

## 8. 실험 순서

1. 가상 RFP로 baseline End-to-End 실행을 확인합니다.
2. Data 담당이 실제 PDF/HWP를 공통 `chunks.jsonl`로 변환합니다.
3. Retrieval 1과 Retrieval 2가 각자 기본 유사도 검색을 병렬 구현합니다.
4. Generation 담당이 공통 검색 결과로 근거 기반 답변을 생성합니다.
5. 같은 평가 질문으로 OpenAI와 로컬 Retrieval의 검색 품질·속도·비용을 비교합니다.
6. MMR, Hybrid Search, Re-ranking을 추가 실험합니다.
7. 검색 품질, 답변 품질, 응답 속도, 비용을 종합해 최종 조합을 선정합니다.

## 9. 주요 설정

공통 설정은 `config/default.yaml`에서 관리합니다.

- `chunking.chunk_size`, `chunking.chunk_overlap`: 청크 크기와 중첩
- `retrieval.active_profile`: `baseline`, `openai`, `local`
- `retrieval.top_k`: 반환할 청크 수
- `retrieval.search_method`: `similarity`, 이후 `mmr`, `hybrid`, `rerank`
- `retrieval.profiles.*.embedding_model`: 각 담당자가 선정한 임베딩 모델
- `generation.provider`, `generation.model`: Generation 담당자가 선정한 LLM

모델명과 실험값은 코드에 직접 적지 않고 설정 파일과 실험 기록에 남깁니다.

## 10. 협업 규칙

### 브랜치

브랜치 이름은 `이슈번호-역할-작업내용` 형식으로 작성합니다.

```text
3-data-pdf-hwp-parser
16-retrieval1-openai-chroma
17-retrieval2-local-chroma
8-generation-rag-prompt
```

### PR

- PR 제목에는 역할, 이슈 번호, 작업 내용을 적습니다.
- 완료되는 이슈는 `Closes #이슈번호`, 중간 작업은 `Related to #이슈번호`로 연결합니다.
- 화면 작업은 스크린샷, 코드·검색·모델 작업은 실행 결과나 로그 또는 표를 첨부합니다.
- 설정 변경은 변경된 설정값과 적용 결과를 설명합니다.

### 리뷰와 머지

- PR은 최소 2명 이상의 리뷰와 Approve를 받은 뒤 머지합니다.
- 본인이 올린 PR을 본인이 바로 머지하지 않습니다.
- 리뷰어는 코드, 실행 결과, 이슈 완료 기준 충족 여부를 확인합니다.
- 수정 요청을 반영한 뒤 다시 리뷰를 요청합니다.

### 저장 금지 항목

원본 RFP, 대용량 추출 데이터, Vector DB, `.env`, API 키, 모델 파일, 가상환경, GCP·SSH 키는 Git에 올리지 않습니다.

## 11. 작업 관리 및 팀 문서

| 목록 | 링크 |
| :--- | :--- |
| GitHub Issues | [Issues](https://github.com/tjgh8167/sprint-ai-mid-project_team3/issues) |
| GitHub Project | 팀장 Project URL 확정 후 기입 |
| 협업일지 | [협업일지 링크](https://docs.google.com/spreadsheets/d/1LoEBOuxMkzjaf2hdq9GiLNaeNl8UUU9WKPQUzh2PhEM/edit?usp=drive_link) |
| 보고서 작성 | [보고서 링크](https://canva.link/q0494iur016u7uq) |

1
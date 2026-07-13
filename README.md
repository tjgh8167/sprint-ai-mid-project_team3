# 입찰메이트 (BidMate) - 사내 RAG 입찰지원 시스템

## 1. 프로젝트 개요

- **진행 기간**: 2026년 7월 10일 ~ 2026년 8월 3일
- **목표**: 공공입찰 RFP 문서를 기반으로 주요 요구사항, 발주기관, 예산, 제출 방식 등을 빠르게 검색하고 답변하는 RAG 시스템을 구축합니다.
- **주요 과제**
  - HWP / PDF 다중 포맷 제안서 파싱 및 의미 단위 청킹 전략 설계
  - 사용자의 불명확한 입력에 대응하는 메타데이터 필터링 기법 고도화
  - 시나리오 A(GCP 오픈소스 LLM)와 시나리오 B(OpenAI API) 투 트랙 구현 및 비교 실험
  - 대화 맥락(History) 유지 기능 및 RAG 성능 평가 지표 수립

## 2. 팀원 소개 및 역할

| 이름 | 역할 | 주요 업무 |
| :--- | :--- | :--- |
| **유재열** | **PM + Data Engineer** | 프로젝트 범위, 일정, 회의, GitHub 관리<br>원본 RFP 및 `data_list.csv` 구조 확인<br>HWP/PDF 다중 포맷 파싱<br>`metadata.csv`, `chunks.jsonl` 스키마 설계<br>청킹 전략 설계 및 구현<br>평가 질문셋/지표 설계와 실험 결과 정리 |
| **정서호** | **Retrieval 1** | 임베딩 모델 및 Vector DB 선택<br>임베딩 생성 및 Vector DB 구축<br>기본 top-k 검색 구현<br>메타데이터 필터링 구현 |
| **이태훈** | **Retrieval 2** | Retrieval 심화 실험 우선순위 정리<br>MMR, Hybrid Search, Multi-Query, Re-Ranking 실험<br>HuggingFace 로컬 모델 기반 RAG 실험<br>검색 고도화 결과 비교 |
| **김효섭** | **Generation** | 답변 생성 모델 선정<br>프롬프트 엔지니어링 및 RAG Chain 구성<br>데모 UI/CLI 구현<br>대화 맥락 유지와 비용/응답 속도 최적화 |

## 3. 프로젝트 구조

현재 구조는 역할별 병렬 구현이 가능하도록 `src/` 안에서 데이터 처리, 검색, 답변 생성을 분리합니다. 공통 설정값은 `config/default.yaml`에서 관리하여 모델명, chunk 크기, top_k 같은 값이 코드에 하드코딩되지 않도록 합니다.

```text
sprint-ai-mid-project_team3/
├── .gitignore
├── README.md                         |  # 프로젝트 소개, 역할, 작업 규칙
├── config/
│   └── default.yaml                  |  # 모델, 경로, chunk, retrieval, generation 기본 설정
│
├── data/                             |  # (로컬/VM) RFP 원본 및 중간 산출물, GitHub 업로드 제외
├── vector_db/                        |  # (로컬/VM) FAISS/Chroma 저장소, GitHub 업로드 제외
│
├── src/                              |  # 공통 모듈 소스코드
│   ├── __init__.py
│   ├── parser_chunker.py             |  # PM+Data: 문서 로드, 메타데이터 매핑, 청킹
│   ├── retriever.py                  |  # Retrieval 1/2: 임베딩, Vector DB, 검색, 필터링, 검색 고도화
│   └── rag_engine.py                 |  # Generation: 프롬프트, 답변 생성, RAG Chain
│
├── notebook/                         |  # 역할별 실험 노트북
│   ├── 01_parser_chunker_test.ipynb  |  # PM+Data 실험
│   ├── 02_retriever_test.ipynb       |  # Retrieval 1/2 실험
│   ├── 03_rag_engine_test.ipynb      |  # Generation 실험
│   └── 04_evaluate_test.ipynb        |  # PM+Data 평가 실험
│
├── gcp_main.py                       |  # 시나리오 A: GCP/오픈소스 모델 기반 실행
├── api_main.py                       |  # 시나리오 B: LLM API 기반 실행
└── evaluate.py                       |  # 성능 평가 실행
```

## 4. 역할별 병렬 작업 방식

| 역할 | 담당 | 주 작업 파일 | 선행 의존성 |
| :--- | :--- | :--- | :--- |
| PM + Data Engineer | 유재열 | `src/parser_chunker.py`, `notebook/01_parser_chunker_test.ipynb`, `evaluate.py`, `notebook/04_evaluate_test.ipynb` | 원본 RFP, `data_list.csv` |
| Retrieval 1 | 정서호 | `src/retriever.py`, `scripts/build_embeddings.py`, `scripts/build_vector_db.py`, `notebook/02_retriever_test.ipynb` | PM+Data의 `metadata.csv`, `chunks.jsonl` |
| Retrieval 2 | 이태훈 | `src/retriever.py`, `notebook/02_retriever_test.ipynb`, `gcp_main.py` | Retrieval 1의 기본 검색 결과 |
| Generation | 김효섭 | `src/rag_engine.py`, `notebook/03_rag_engine_test.ipynb`, `api_main.py`, `app/streamlit_app.py` | Retrieval 검색 결과 |

초반에는 PM+Data가 `metadata.csv`, `chunks.jsonl` 샘플을 먼저 제공하고, Retrieval 1/2와 Generation은 샘플 데이터 기준으로 병렬 개발합니다. 이후 전체 데이터가 준비되면 같은 인터페이스로 확장합니다.

## 5. 작업 관리

| 항목 | 링크 |
| :--- | :--- |
| GitHub Issues | [Issues](https://github.com/tjgh8167/sprint-ai-mid-project_team3/issues) |
| GitHub Project | 팀장 Project URL 확정 후 기입 |
| 협업일지 | [협업일지 링크](https://docs.google.com/spreadsheets/d/1LoEBOuxMkzjaf2hdq9GiLNaeNl8UUU9WKPQUzh2PhEM/edit?usp=drive_link) |
| 보고서 작성<br>(추후 완성본 링크 기입) | [보고서 링크](https://canva.link/q0494iur016u7uq) |

## 6. 이슈 및 PR 규칙

작업은 GitHub Issues의 번호와 연결합니다. 각자 맡은 이슈 번호를 기준으로 브랜치를 만들고, 작업 완료 후 PR을 올립니다.

### 6-1. 브랜치 이름 규칙

브랜치 이름은 앞에 이슈 번호를 붙여서 만듭니다.

```text
이슈번호-역할-작업내용
```

브랜치 이름 예시:

```text
3-data-pdf-hwp-parser
4-data-create-chunks-jsonl
8-generation-rag-prompt
16-retrieval1-build-embeddings
23-retrieval2-mmr-search
```

브랜치명은 영어 소문자와 하이픈(`-`) 위주로 작성합니다. 같은 이슈를 여러 번 나눠 작업하는 경우에는 뒤에 세부 작업명을 붙입니다.

### 6-2. PR 제목 규칙

PR 제목에는 역할, 이슈 번호, 작업 내용을 함께 적습니다.

```text
[Data] #3 PDF/HWP 텍스트 추출 파이프라인 구현
[Retrieval 1] #16 임베딩 생성 스크립트 구현
[Retrieval 2] #23 심화: MMR 검색 실험
[Generation] #8 답변 생성 프롬프트 구현
```

### 6-3. PR 본문 규칙

PR 본문에는 완료되는 이슈 번호를 넣습니다.

```text
Closes #3
```

아직 이슈를 닫지 않는 중간 작업이면 아래처럼 씁니다.

```text
Related to #3
```

PR 본문에는 반드시 결과 확인 내용을 첨부합니다.

- 화면이 있는 작업: 실행 화면 스크린샷 첨부
- 코드/모델/검색 작업: 실행 결과, 로그, 표, 노트북 결과 중 하나 첨부
- 설정 변경 작업: 변경 전후 또는 적용된 설정값 설명 첨부

### 6-4. PR 리뷰 및 머지 규칙

PR은 최소 2명 이상의 리뷰와 Approve를 받은 뒤 머지합니다.

- 본인이 올린 PR은 본인이 바로 머지하지 않습니다.
- 리뷰어는 코드, 실행 결과, 이슈 완료 기준 충족 여부를 확인합니다.
- 리뷰 반영이 필요한 경우 수정 후 다시 리뷰를 요청합니다.
- 2명 이상 Approve가 완료되면 머지할 수 있습니다.
## 7. 기본 구현 범위

1. 원본 RFP 및 `data_list.csv` 구조 확인
2. PDF/HWP 텍스트 추출
3. `metadata.csv`, `chunks.jsonl` 생성
4. OpenAI API 기반 임베딩 생성
5. Chroma 또는 FAISS 기반 Vector DB 구축
6. 기본 similarity search 구현
7. 검색 결과 기반 답변 생성
8. 평가 질문셋 및 결과표 작성
9. 간단한 데모 UI 또는 CLI 구현

## 8. 심화 구현 후보

- 의미 단위 청킹
- MMR 검색
- Hybrid Search
- Multi-Query Retrieval
- Re-Ranking
- OCR 및 표 추출 개선
- HuggingFace 로컬 모델 기반 RAG
- 대화 맥락 유지
- 비용 및 응답 속도 최적화

## 9. 설정 관리 규칙

모델명, chunk 크기, overlap, top_k, temperature 같은 실험 설정은 `config/default.yaml`에 기록하고 코드에서는 설정 파일을 읽어 사용합니다.

하드코딩하지 않을 값:

- `chunk_size`, `chunk_overlap`
- embedding model
- Vector DB 종류와 저장 경로
- `top_k`, search type
- LLM model, `temperature`, `top_p`, `max_tokens`
- 평가 질문셋 경로와 결과 저장 경로

## 10. 저장 규칙

GitHub에는 코드와 문서만 저장합니다.

GitHub에 올리지 않는 항목:

- RFP 원본 파일
- 추출된 대용량 텍스트 파일
- Vector DB 파일
- `.env` 및 API 키
- HuggingFace 모델 파일
- GCP/SSH 키

대용량 데이터는 VM 또는 로컬의 `data/`, `vector_db/`, `models/` 등에 보관합니다.
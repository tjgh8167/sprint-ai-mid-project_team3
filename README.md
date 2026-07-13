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
| **이름** | **Project Manager + Retrieval 2** | 회의 주도 및 프로젝트 매니징<br>평가 질문셋/지표 설계<br>실험 결과표 및 발표자료 정리<br>MMR, Hybrid Search, Multi-Query, Re-Ranking 등 Retrieval 심화 실험 조율 |
| **이름** | **Data Engineer** | 원본 RFP 및 `data_list.csv` 구조 확인<br>HWP/PDF 다중 포맷 파싱<br>`metadata.csv`, `chunks.jsonl` 스키마 설계<br>청킹 전략 설계 및 구현 |
| **이름** | **Retrieval 1** | 임베딩 모델 및 Vector DB 선택<br>임베딩 생성 및 Vector DB 구축<br>기본 top-k 검색 구현<br>메타데이터 필터링 구현 |
| **이름** | **Generation** | 답변 생성 모델 선정<br>프롬프트 엔지니어링 및 RAG Chain 구성<br>데모 UI/CLI 구현<br>대화 맥락 유지와 비용/응답 속도 최적화 |

## 3. 프로젝트 구조

```text
sprint-ai-mid-project_team3-/
├── .gitignore
├── README.md                         |  # 프로젝트 소개, 보고서 링크, 협업일지 링크 배치
│
├── data/                             |  # (로컬) 데이터 원본
│
├── src/                              │  # 공통 모듈 소스코드
│   ├── __init__.py
│   ├── parser_chunker.py             │  # 문서로드, 데이터 매핑 및 청킹 구현
│   ├── retriever.py                  │  # 임베딩, Vector DB 연결 및 관리, Retriever 구현
│   └── rag_engine.py                 │  # 프롬프트 엔지니어링 및 RAG Chain 구성
│
├── notebooks/                        │  # 실험 노트북
│   ├── 01_parser_chunker_test.ipynb
│   ├── 02_retriever_test.ipynb
│   ├── 03_rag_engine_test.ipynb
│   └── 04_evaluate_test.ipynb
│
├── gcp_main.py                       │  # 오픈소스 모델 기반 실행 코드
├── api_main.py                       │  # OpenAI API 기반 실행 코드
└── evaluate.py                       │  # 성능 평가
```

## 4. 작업 관리

| 항목 | 링크 |
| :--- | :--- |
| GitHub Issues | [Issues](https://github.com/tjgh8167/sprint-ai-mid-project_team3/issues) |
| GitHub Project | 팀장 Project URL 확정 후 기입 |
| 협업일지 | [협업일지 링크](https://docs.google.com/spreadsheets/d/1LoEBOuxMkzjaf2hdq9GiLNaeNl8UUU9WKPQUzh2PhEM/edit?usp=drive_link) |
| 보고서 작성<br>(추후 완성본 링크 기입) | [보고서 링크](https://canva.link/q0494iur016u7uq) |

## 5. 이슈 및 PR 규칙

작업은 GitHub Issues의 번호와 연결합니다.

PR 제목 예시:

```text
[Data] #3 PDF/HWP 텍스트 추출 파이프라인 구현
[Retrieval 1] #16 임베딩 생성 스크립트 구현
[PM][Retrieval 2] #23 심화: MMR 검색 실험
[Generation] #8 답변 생성 프롬프트 구현
```

PR 본문에는 완료되는 이슈 번호를 넣습니다.

```text
Closes #3
```

아직 이슈를 닫지 않는 중간 작업이면 아래처럼 씁니다.

```text
Related to #3
```

## 6. 기본 구현 범위

1. 원본 RFP 및 `data_list.csv` 구조 확인
2. PDF/HWP 텍스트 추출
3. `metadata.csv`, `chunks.jsonl` 생성
4. OpenAI API 기반 임베딩 생성
5. Chroma 또는 FAISS 기반 Vector DB 구축
6. 기본 similarity search 구현
7. 검색 결과 기반 답변 생성
8. 평가 질문셋 및 결과표 작성
9. 간단한 데모 UI 또는 CLI 구현

## 7. 심화 구현 후보

- 의미 단위 청킹
- MMR 검색
- Hybrid Search
- Multi-Query Retrieval
- Re-Ranking
- OCR 및 표 추출 개선
- HuggingFace 로컬 모델 기반 RAG
- 대화 맥락 유지
- 비용 및 응답 속도 최적화

## 8. 저장 규칙

GitHub에는 코드와 문서만 저장합니다.

GitHub에 올리지 않는 항목:

- RFP 원본 파일
- 추출된 대용량 텍스트 파일
- Vector DB 파일
- `.env` 및 API 키
- HuggingFace 모델 파일
- GCP/SSH 키

대용량 데이터는 VM 또는 로컬의 `data/`, `vector_db/`, `models/` 등에 보관합니다.

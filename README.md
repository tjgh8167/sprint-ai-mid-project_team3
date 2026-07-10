# 입찰메이트 (BidMate) - 사내 RAG 입찰지원 시스템

## 1. 프로젝트 개요
- **진행 기간**: 2026년 7월 10일 ~ 2026년 8월 3일
* **주요 과제:**
  - HWP / PDF 다중 포맷 제안서 파싱 및 의미 단위 청킹 전략 설계
  - 사용자의 불명확한 입력에 대응하는 메타데이터 필터링 기법 고도화
  - 시나리오 A(GCP 오픈소스 LLM)와 시나리오 B(OpenAI API) 투트랙 구현 및 비교 실험
  - 대화 맥락(History) 유지 기능 및 RAG 성능 평가 지표 수립

## 2. 팀원 소개 및 역할

| 이름 | 역할 | 주요 업무 |
| :--- | :--- | :--- |
| **정서호** | **PM** | ** |
| **김효섭** | **** | ** |
| **유재열** | **** | ** |
| **이태훈** | **** | ** |


## 3. 프로젝트 구조
```text
sprint-ai-mid-project_team3-/
├── .gitignore
├── README.md                         |  # 프로젝트 소개, 보고서 다운로드 링크, 개인 협업일지 링크 배치
│
├── data/
│
├── src/                              │  # 공통 모듈 소스코드
│   ├── __init__.py  
│   ├── parser_chunker.py             │  # 문서로드, 데이터 매핑 및 청킹 구현
│   ├── retriever.py                  │  # 임베딩, Vector DB 연결 및 관리
│   └── rag_engine.py                 │  # 리트리버, 프롬프트, RAG Chain
│   
│
├── notebooks/                        │  # 실험 노트북
│   ├── 01_data_pipeline_test.ipynb
│   ├── 02_retriever_test.ipynb
│   └── 03_database_test.ipynb
│
│
├── gcp_main.py/                      │  # 오픈소스 모델 기반 실행 코드
│   
├── api_main.py/                      │  # OpenAI API 기반 실행 코드
│
└── evaluate.py/                      │  # 성능 평가

```

## 4. 팀 문서

| 목록| 링크 |
| :--- | :--- |
| **협업일지** | [협업일지 링크](https://docs.google.com/spreadsheets/d/1LoEBOuxMkzjaf2hdq9GiLNaeNl8UUU9WKPQUzh2PhEM/edit?usp=drive_link) |
| **보고서** | [보고서 링크](https://canva.link/q0494iur016u7uq) |

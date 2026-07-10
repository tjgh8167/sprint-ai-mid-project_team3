# 입찰메이트 (BidMate) - 사내 RAG 입찰지원 시스템

## 1. 프로젝트 개요
- **진행 기간**: 2026년 7월 10일 ~ 2026년 8월 3일
* **주요 과제:**
  - HWP / PDF 다중 포맷 제안서 파싱 및 의미 단위 청킹 전략 설계
  - 사용자의 불명확한 입력에 대응하는 메타데이터 필터링 기법 고도화
  - 시나리오 A(GCP 오픈소스 LLM)와 시나리오 B(OpenAI API) 투 트랙 구현 및 비교 실험
  - 대화 맥락(History) 유지 기능 및 RAG 성능 평가 지표 수립

## 2. 팀원 소개 및 역할

| 이름 | 역할 | 주요 업무 |
| :--- | :--- | :--- |
| **이름** | **Project Manager** | - 회의 주도 및 프로젝트 매니징<br>- 파트별 백업<br> - 전체 파이프라인 통합 성능 평가 총괄<br>- 발표자료 제작  |
| **이름** | **Data Engineer** | - 문서 로드 및 HWP/PDF 다중 포맷 파싱<br>- 청킹 전략 설계 및 구현 |
| **이름** | **Retriever** | - RAG 베이스 모델 별 임베딩 생성 및 Vector DB 구축<br>- 메타데이터 필터링 및 Retriever 고도화 |
| **이름** | **Generation** | - 프롬프트 엔지니어링 및 RAG Chain 구성 |


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
│
├── notebooks/                        │  # 실험 노트북
│   ├── 01_data_pipeline_test.ipynb
│   ├── 02_retriever_test.ipynb
│   └── 03_database_test.ipynb
│
│
├── gcp_main.py                       │  # 오픈소스 모델 기반 실행 코드
│    
├── api_main.py                       │  # OpenAI API 기반 실행 코드
│
└── evaluate.py                       │  # 성능 평가

```

## 4. 팀 문서

| 목록| 링크 |
| :--- | :--- |
| **협업일지** | [협업일지 링크](https://docs.google.com/spreadsheets/d/1LoEBOuxMkzjaf2hdq9GiLNaeNl8UUU9WKPQUzh2PhEM/edit?usp=drive_link) |
| **보고서 작성<br>(추후 완성본 링크 기입)** | [보고서 링크](https://canva.link/q0494iur016u7uq) |

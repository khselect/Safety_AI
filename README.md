Project_Root/
│
├── 📂 core/                  # 핵심 모듈 모음
│   ├── __init__.py          # (생성 필요) 패키지 인식용 빈 파일
│   ├── config.py            # 환경 설정 및 경로 상수
│   ├── data.py              # 문서 파싱(HWP, PDF) 및 전처리
│   ├── decision_ai.py       # 위험도 판별 AI 로직
│   ├── llm.py               # LLM 및 임베딩 모델 로드
│   └── rag.py               # 검색(Retrieval) 및 답변 생성 로직
│
├── 📂 shared/                # (자동 생성됨) Admin-Main 데이터 공유 폴더
│   └── risk_df.pkl          # 전처리된 상황보고 데이터
│
├── 📂 chroma_db/             # (자동 생성됨) 벡터 데이터베이스 저장소
│
├── admin.py                 # [관리자용] 문서 학습 및 데이터 관리
├── main.py                  # [사용자용] 규정 검색 및 위험 분석 대시보드
├── requirements.txt         # 의존성 라이브러리 목록
└── 상황보고.xlsx             # (샘플) 과거 사고 데이터
<img width="1088" height="846" alt="image" src="https://github.com/user-attachments/assets/2fa06695-f0e4-4b48-9973-5d9e27eea1b2" />
<img width="1083" height="831" alt="image" src="https://github.com/user-attachments/assets/94d531a1-2754-438c-a187-60724041ea1f" />
<img width="1068" height="818" alt="image" src="https://github.com/user-attachments/assets/b99280fc-8050-4ebc-92ad-dbf4bca50a50" />
<img width="1039" height="789" alt="image" src="https://github.com/user-attachments/assets/444d0c92-35b4-44e7-b6cd-838e51bd01d0" />
<img width="1040" height="815" alt="image" src="https://github.com/user-attachments/assets/9a1a37c9-af5b-4239-afcf-1477c213493b" />
<img width="1012" height="798" alt="image" src="https://github.com/user-attachments/assets/cef4c507-2b4a-4e4f-9c1f-c26609439ab1" />

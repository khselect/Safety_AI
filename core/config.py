# app/
# ├─ admin.py              # 🔐 관리자 콘솔 (학습 + 관리 + 분류)
# ├─ main.py               # 👤 사용자 챗봇 UI
# ├─ core/
# │  ├─ rag.py             # 📚 규정 RAG 공통 로직
# │  ├─ data.py            # 📊 엑셀 처리
# │  ├─ llm.py             # 🤖 LLM 생성
# │  └─ config.py          # ⚙️ 공통 설정
# └─ chroma_db/

import os

PERSIST_DIRECTORY = "./chroma_db"

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:11434")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st
import os
import shutil
import time
import pandas as pd

from langchain_chroma import Chroma
from core.config import PERSIST_DIRECTORY
from core.llm import get_embeddings
from core.data import process_file_to_docs

# ------------------------------------------------------------------
# Streamlit 기본 설정
# ------------------------------------------------------------------
st.set_page_config(page_title="🛠 관리자 콘솔", layout="wide")
st.title("🛠 규정 · 리스크 관리 관리자 콘솔")

# ------------------------------------------------------------------
# 🔹 ChromaDB 관리 함수들
# ------------------------------------------------------------------

def get_vectorstore():
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=get_embeddings()
    )


def get_learned_regulations():
    """
    현재 ChromaDB에 학습된 규정 메타데이터 반환
    """
    if not os.path.exists(PERSIST_DIRECTORY):
        return None

    vectorstore = get_vectorstore()
    data = vectorstore.get()

    if not data or not data.get("metadatas"):
        return None

    return pd.DataFrame(data["metadatas"])


def delete_regulation(source_name: str):
    """
    특정 규정 파일 전체 삭제
    """
    vectorstore = get_vectorstore()
    vectorstore._collection.delete(where={"source": source_name})


# ------------------------------------------------------------------
# 🔹 사이드바
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 관리자 메뉴")

    if st.button("🗑 전체 규정 초기화"):
        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
        st.success("모든 규정 데이터가 삭제되었습니다.")
        st.rerun()

# ------------------------------------------------------------------
# 📂 1. 규정 학습
# ------------------------------------------------------------------
st.subheader("📂 1. 사내 규정 학습")

uploaded_files = st.file_uploader(
    "규정 파일 업로드 (PDF, DOCX, HWP)",
    type=["pdf", "docx", "hwp"],
    accept_multiple_files=True
)

if st.button("🚀 규정 학습 시작"):
    if not uploaded_files:
        st.warning("학습할 규정 파일을 업로드하세요.")
    else:
        with st.spinner("규정 학습 중..."):
            all_docs = []
            for file in uploaded_files:
                try:
                    docs = process_file_to_docs(file, file.name)
                    all_docs.extend(docs)
                except Exception as e:
                    st.error(f"{file.name} 처리 실패: {e}")

            if all_docs:
                vectorstore = get_vectorstore()
                vectorstore.add_documents(all_docs)
                st.success(f"✅ 학습 완료 ({len(all_docs)} 청크)")
                time.sleep(1)
                st.rerun()

# ------------------------------------------------------------------
# 📑 2. 학습된 규정 관리
# ------------------------------------------------------------------
st.divider()
st.subheader("📑 2. 학습된 규정 관리")

df_meta = get_learned_regulations()

if df_meta is None:
    st.info("아직 학습된 규정이 없습니다.")
else:
    summary = (
        df_meta
        .groupby("source")
        .agg(
            chunks=("Article_Title", "count"),
            articles=("Article_Title", lambda x: len(set(x)))
        )
        .reset_index()
    )

    st.dataframe(summary, use_container_width=True)

    st.markdown("### 🗑 규정 삭제")

    target = st.selectbox(
        "삭제할 규정 선택",
        summary["source"].tolist()
    )

    if st.button("선택 규정 삭제"):
        delete_regulation(target)
        st.success(f"'{target}' 규정이 삭제되었습니다.")
        st.rerun()

# ------------------------------------------------------------------
# 📊 3. 상황보고 엑셀 업로드 (관리용)
# ------------------------------------------------------------------
st.divider()
st.subheader("📊 3. 상황보고 데이터 업로드")

excel = st.file_uploader(
    "상황보고 엑셀 업로드 (.xlsx)",
    type=["xlsx"]
)

if excel:
    try:
        df = pd.read_excel(excel, engine="openpyxl")
        st.success("엑셀 데이터 로드 완료")
        st.dataframe(df, use_container_width=True)
        st.session_state["risk_df"] = df
    except Exception as e:
        st.error(f"엑셀 로드 실패: {e}")

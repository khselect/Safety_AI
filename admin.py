import streamlit as st
import os
import shutil
import time
import pandas as pd
import re
import tempfile
import sys
import warnings

# 문서 변환 라이브러리
import pymupdf4llm
import mammoth
import markdownify
import olefile

# LangChain & Chroma 관련
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document

# core 모듈 (사용자 환경에 맞게 경로 확인)
try:
    from core.config import PERSIST_DIRECTORY
    from core.llm import get_embeddings
except ImportError:
    # core 모듈이 없을 경우를 대비한 하드코딩 (테스트용)
    PERSIST_DIRECTORY = "./chroma_db"
    from langchain_huggingface import HuggingFaceEmbeddings
    def get_embeddings():
        return HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

# 경고 무시
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ------------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------------
st.set_page_config(page_title="🛠 관리자 콘솔", layout="wide")
st.title("🛠 규정 · 리스크 관리 관리자 콘솔")

# ------------------------------------------------------------------
# 1. 핵심 함수 정의 (텍스트 정제 및 문서 처리)
# ------------------------------------------------------------------

def clean_markdown_text(text):
    """
    마크다운 텍스트에서 불필요한 기호, 빈 표, 과도한 공백을 제거합니다.
    """
    if not isinstance(text, str):
        return ""

    # 1. 무의미한 표 행 제거 (예: | | | | | )
    # 파이프(|), 공백(\s), 하이픈(-)으로만 구성된 줄을 삭제
    text = re.sub(r'^[|\s-]+$', '', text, flags=re.MULTILINE)
    
    # 2. 연속된 줄바꿈 및 공백 정리
    text = re.sub(r'\n{3,}', '\n\n', text)  # 3줄 이상 공백 -> 2줄로
    text = re.sub(r'[ \t]+', ' ', text)     # 연속된 스페이스/탭 -> 공백 1개
    
    # 3. 마크다운 이미지/링크 태그 제거 (텍스트 분석에 방해됨)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    
    # 4. 특수문자 노이즈 제거 (물결표 등)
    text = text.replace("~~", "")
    
    return text.strip()

def extract_hwp_text(hwp_path):
    """HWP 파일 텍스트 추출"""
    try:
        f = olefile.OleFileIO(hwp_path)
        encoded_text = f.openstream("PrvText").read()
        decoded_text = encoded_text.decode("utf-16le")
        return decoded_text
    except Exception as e:
        return f"[HWP 오류] 변환 실패: {e}"

def process_file_to_docs(file, source_name):
    """파일을 읽어 청크(Chunk) 단위의 Document 리스트로 변환"""
    file_ext = os.path.splitext(file.name)[1].lower()
    
    # 임시 파일 생성
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name

    try:
        # 1. 텍스트 추출
        md_text = ""
        if file_ext == ".pdf":
            md_text = pymupdf4llm.to_markdown(tmp_path)
        elif file_ext == ".docx":
            result = mammoth.convert_to_html(tmp_path)
            html = result.value
            md_text = markdownify.markdownify(html, heading_style="ATX", strip=['img'])
        elif file_ext in [".hwp", ".hwpx"]:
            raw_text = extract_hwp_text(tmp_path) 
            md_text = f"# {source_name} 본문\n\n{raw_text}"
        else:
            return []
        
        # 2. [중요] 텍스트 강력 정제
        md_text = clean_markdown_text(md_text)
        
        # 3. 헤더 처리 (제N조 -> # 제N조)
        md_text = re.sub(r'(^|\n)(제\s*\d+(?:의\d+)?\s*조)', r'\n# \2', md_text)
        
        # 4. 청크 분할 (Chunking)
        headers_to_split_on = [("#", "Article_Title")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        header_splits = markdown_splitter.split_text(md_text)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        
        final_docs = []
        for doc in header_splits:
            if len(doc.page_content.strip()) < 10:
                continue
                
            splits = text_splitter.split_text(doc.page_content)
            for split_content in splits:
                if re.match(r'^[|\s-]+$', split_content):
                    continue

                new_doc = Document(
                    page_content=split_content,
                    metadata={
                        "source": source_name,
                        "Article_Title": doc.metadata.get("Article_Title", "일반"),
                        "file_type": file_ext
                    }
                )
                final_docs.append(new_doc)
                
        return final_docs
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ------------------------------------------------------------------
# 2. 사이드바 UI (파일 업로드 및 관리)
# ------------------------------------------------------------------
with st.sidebar:
    st.header("📂 데이터 관리")
    
    # --- [섹션 1] 규정 파일 학습 ---
    st.subheader("1. 규정 파일 학습")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, HWP 파일 업로드", 
        type=["pdf", "docx", "hwp"], 
        accept_multiple_files=True
    )
    
    if st.button("🚀 DB 학습 시작", use_container_width=True):
        if uploaded_files:
            with st.spinner("문서 분석 및 벡터 DB 저장 중..."):
                all_docs = []
                for file in uploaded_files:
                    try:
                        docs = process_file_to_docs(file, file.name)
                        if docs:
                            all_docs.extend(docs)
                            st.toast(f"✅ {file.name} 처리 완료 ({len(docs)} chunks)")
                        else:
                            st.error(f"⚠️ {file.name}: 텍스트 추출 실패")
                    except Exception as e:
                        st.error(f"❌ {file.name} 오류: {e}")
                
                if all_docs:
                    vectorstore = Chroma(
                        persist_directory=PERSIST_DIRECTORY, 
                        embedding_function=get_embeddings()
                    )
                    vectorstore.add_documents(all_docs)
                    st.success(f"🎉 전체 학습 완료! (총 {len(all_docs)}개 데이터)")
                    time.sleep(1)
                    st.rerun()
        else:
            st.warning("파일을 먼저 선택해주세요.")

    st.divider()
    
    # --- [섹션 2] 시스템 초기화 ---
    st.subheader("2. 시스템 초기화")
    if st.button("🗑️ 규정 DB 전체 삭제", type="primary", use_container_width=True):
        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
            st.success("DB가 초기화되었습니다. 새로 학습시켜주세요.")
            time.sleep(1)
            st.rerun()
        else:
            st.info("삭제할 DB가 없습니다.")

    st.divider()

    # --- [섹션 3] 상황보고 엑셀 업로드 (복구된 부분) ---
    st.subheader("3. 상황보고 데이터 업로드")
    excel = st.file_uploader(
        "상황보고 엑셀 업로드 (.xls, .xlsx)",
        type=["xls", "xlsx"]
    )

    if excel is not None:
        # 파일 포인터 초기화
        excel.seek(0)
        try:
            filename = excel.name.lower()
            if filename.endswith(".xls"):
                # .xls 지원을 위해 xlrd 라이브러리 필요 (pip install xlrd)
                df = pd.read_excel(excel, engine="xlrd")
            elif filename.endswith(".xlsx"):
                df = pd.read_excel(excel, engine="openpyxl")
            else:
                st.error("지원하지 않는 엑셀 형식입니다.")
                st.stop() 

            st.success(f"엑셀 데이터 로드 완료 ({len(df)}행)")
            
            # shared 폴더에 피클 파일로 저장 (Main 앱과 공유)
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            SHARED_DIR = os.path.join(BASE_DIR, "shared")
            os.makedirs(SHARED_DIR, exist_ok=True)
            FILE_PATH = os.path.join(SHARED_DIR, "risk_df.pkl")

            df.to_pickle(FILE_PATH)
            st.success("✅ 상황보고 데이터가 공용 저장소에 저장되었습니다.")
            
        except Exception as e:
            st.error(f"엑셀 로드 실패: {e}")


# ------------------------------------------------------------------
# 3. 메인 화면 (상태 모니터링)
# ------------------------------------------------------------------
st.header("📊 현재 시스템 상태")

# [1] 규정 데이터 상태 (전체 너비 사용)
st.subheader("📚 규정 데이터 (Chroma DB)")

if os.path.exists(PERSIST_DIRECTORY):
    try:
        vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=get_embeddings())
        collection = vectorstore.get()
        doc_count = len(collection['ids']) if collection else 0
        
        st.metric("학습된 규정 청크 수", f"{doc_count} 개")
        
        if doc_count > 0:
            sources = list(set([m['source'] for m in collection['metadatas'] if m.get('source')]))
            st.markdown("**학습된 파일 목록:**")
            st.dataframe(pd.DataFrame(sources, columns=["파일명"]), use_container_width=True)
            
            # (선택사항) 데이터 샘플 확인이 필요하면 주석 해제
            # with st.expander("🔍 데이터 샘플 확인 (최근 5개)"):
            #     for i in range(min(5, doc_count)):
            #         st.info(f"**[{collection['metadatas'][i].get('source')}]**\n\n{collection['documents'][i][:200]}...")
                    
    except Exception as e:
        st.error(f"DB 로드 중 오류: {e}")
else:
    st.info("학습된 규정 데이터가 없습니다. 사이드바에서 파일을 업로드하세요.")

st.divider() # 섹션 구분선

# [2] 상황보고 데이터 상태 (전체 너비 사용)
st.subheader("📈 상황보고 데이터 (Excel)")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DIR = os.path.join(BASE_DIR, "shared")
FILE_PATH = os.path.join(SHARED_DIR, "risk_df.pkl")

if os.path.exists(FILE_PATH):
    try:
        saved_df = pd.read_pickle(FILE_PATH)
        st.metric("저장된 상황보고 건수", f"{len(saved_df)} 건")
        
        st.markdown("**데이터 미리보기 (상위 5건):**")
        st.dataframe(saved_df.head(), use_container_width=True)
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
else:
    st.info("업로드된 상황보고 엑셀 데이터가 없습니다. 사이드바에서 엑셀을 업로드하세요.")
import streamlit as st
import os
import shutil
import time
import datetime
import pandas as pd
import re
import tempfile
import sys
import warnings
import json

# 문서 변환 라이브러리
import pymupdf4llm
import mammoth
import markdownify
import olefile

# LangChain & Chroma 관련
from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever
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
# 기본경로 설정 (절대 경로로 고정하여 main.py와 불일치 방지)
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DIR = os.path.join(BASE_DIR, "shared")
if not os.path.exists(SHARED_DIR):
    os.makedirs(SHARED_DIR)
# [config 파일 경로 정의]
CONFIG_FILE = os.path.join(SHARED_DIR, "system_config.json")

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
    text = re.sub(r'^[|\s-]+$', '', text, flags=re.MULTILINE)
    
    # 2. 연속된 줄바꿈 및 공백 정리
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 3. 마크다운 이미지/링크 태그 제거
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    
    # 4. 특수문자 노이즈 제거
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
    """파일을 읽어 청크(Chunk) 단위의 Document 리스트로 변환 (완전 방어 모드)"""
    file_ext = os.path.splitext(file.name)[1].lower()
    
    # [수정 1] 변수를 함수 시작 지점에서 미리 선언하여 UnboundLocalError 방지
    md_text = "" 
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name

    try:
        # 1. 텍스트 추출 시도
        try:
            if file_ext == ".pdf":
                md_text = pymupdf4llm.to_markdown(tmp_path)
            elif file_ext == ".docx":
                result = mammoth.convert_to_html(tmp_path)
                # Mammoth 결과 검증
                html_content = result.value if (result and result.value) else ""
                md_text = markdownify.markdownify(html_content, heading_style="ATX", strip=['img'])
            elif file_ext in [".hwp", ".hwpx"]:
                raw_text = extract_hwp_text(tmp_path) 
                md_text = f"# {source_name} 본문\n\n{raw_text}"
            else:
                return []
        except Exception as e:
            st.error(f"파일 파싱 중 오류 발생 ({file.name}): {e}")
            md_text = "" # 오류 발생 시 빈 문자열로 초기화

        # [수정 2] 추출된 내용이 None인 경우에 대한 2중 방어
        if md_text is None:
            md_text = ""

        # 2. 텍스트 정제
        md_text = clean_markdown_text(md_text)
        
        if len(md_text.strip()) < 10:
            return []
        
        # 3. 헤더 처리 및 청킹 (기존 로직 유지)
        md_text = re.sub(r'(^|\n)(제\s*\d+(?:의\d+)?\s*조)', r'\n# \2', md_text)
        
        headers_to_split_on = [("#", "Article_Title")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        header_splits = markdown_splitter.split_text(md_text)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        final_docs = []
        for doc in header_splits:
            content = str(doc.page_content) if doc.page_content else ""
            if len(content.strip()) < 10:
                continue
                
            splits = text_splitter.split_text(content)
            for split_content in splits:
                # [수정 3] Document 생성 시 page_content가 절대 None이 되지 않도록 강제 변환
                new_doc = Document(
                    page_content=str(split_content), 
                    metadata={
                        "source": source_name,
                        "Article_Title": str(doc.metadata.get("Article_Title", "일반")),
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
    st.header("⚙️ 시스템 설정")
    
    # [개선1] Ollama 모델 선택 기능
    st.subheader("🤖 LLM 모델 선택")
    ollama_models = [
        "korean-gemma2:latest",
        "korean-llama3:latest",
        "my-korean-llama3:latest",
        "gemma2:latest",
        "llama3:latest",
        "gemma3:4b",
        "nomic-embed-text:latest" 
    ]
    # 1. 현재 저장된 설정 불러오기 (초기값 설정)
    current_index = 1 # 기본값 (korean-llama3)
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved_config = json.load(f)
                saved_model = saved_config.get("selected_model", "korean-llama3:latest")
                if saved_model in ollama_models:
                    current_index = ollama_models.index(saved_model)
        except:
            pass # 파일 읽기 실패 시 기본값 유지

    # 2. 선택박스 표시
    selected_model = st.selectbox(
        "사용할 모델을 선택하세요:",
        options=ollama_models,
        index=current_index
    )
    
    # 3. 변경 감지 및 파일 저장
    # 이전 선택과 다르면 파일에 씁니다.
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = ollama_models[current_index]

    if st.session_state.selected_model != selected_model:
        st.session_state.selected_model = selected_model
        
        # [핵심] JSON 파일로 저장하여 main.py와 공유
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"selected_model": selected_model}, f)
            
        st.toast(f"✅ 모델이 변경되었습니다: {selected_model}")
    st.divider()
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

    # --- [섹션 3] 상황보고 엑셀 업로드 ---
    st.subheader("3. 상황보고 데이터 업로드")
    uploaded_file = st.file_uploader(
        "파일을 드래그하여 업로드하세요.",
        type=["csv", "xls", "xlsx"],
        help="CSV 또는 엑셀 파일을 업로드하면 시스템에 즉시 반영됩니다."
    )

    if uploaded_file is not None:
        try:
            filename = uploaded_file.name.lower()
            df = None

            # 2. 파일 확장자에 따른 처리
            if filename.endswith(".csv"):
                # 한글 깨짐 방지를 위해 인코딩 순차 시도
                try:
                    df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
                except:
                    uploaded_file.seek(0)  # 파일 포인터 초기화
                    df = pd.read_csv(uploaded_file, encoding="cp949")
            
            elif filename.endswith(".xls"):
                df = pd.read_excel(uploaded_file, engine="xlrd")
                
            elif filename.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file, engine="openpyxl")

            # 3. 데이터 저장 로직
            if df is not None:
                # shared 폴더 내 risk_df.pkl로 저장 (메인 분석기 연동용)
                FILE_PATH = os.path.join(SHARED_DIR, "risk_df.pkl")
                df.to_pickle(FILE_PATH)
                
                st.sidebar.success(f"✅ 업로드 완료: {filename}")
                st.sidebar.info(f"데이터 개수: {len(df)}행")
                
                # 업로드 후 화면 갱신 (반영 확인용)
                if st.sidebar.button("데이터 즉시 갱신"):
                    st.rerun()
            else:
                st.sidebar.error("지원하지 않는 파일 형식입니다.")

        except Exception as e:
            st.sidebar.error(f"❌ 오류 발생: {e}")


# ------------------------------------------------------------------
# 3. 메인 화면 (상태 모니터링 및 피드백)
# ------------------------------------------------------------------
st.header("📊 현재 시스템 상태")

# [1] 규정 데이터 상태 (전체 너비 사용)
st.subheader("📚 규정 데이터 관리 (Chroma DB)")

if os.path.exists(PERSIST_DIRECTORY):
    try:
        vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=get_embeddings())
        collection = vectorstore.get() 
        
        total_docs = len(collection['ids']) if collection else 0
        
        if total_docs > 0:
            file_stats = {}
            for idx, meta in enumerate(collection['metadatas']):
                src = meta.get('source', '알수없음')
                doc_content = collection['documents'][idx]
                doc_id = collection['ids'][idx]
                
                if src not in file_stats:
                    file_stats[src] = {
                        "ids": [], 
                        "count": 0, 
                        "preview": doc_content[:50].replace("\n", " ") + "..."
                    }
                file_stats[src]["ids"].append(doc_id)
                file_stats[src]["count"] += 1

            df_data = []
            for src, info in file_stats.items():
                df_data.append({
                    "파일명": src,
                    "청크(Chunk) 수": info["count"],
                    "내용 미리보기 (Article)": info["preview"]
                })
            
            df_files = pd.DataFrame(df_data)

            c1, c2 = st.columns([1, 1])
            c1.metric("총 학습된 파일", f"{len(df_files)} 개")
            c2.metric("총 벡터 청크 수", f"{total_docs} 개")
            
            st.markdown("##### 📋 학습된 파일 목록 상세")
            st.dataframe(
                df_files, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "청크(Chunk) 수": st.column_config.NumberColumn(format="%d 개"),
                    "내용 미리보기 (Article)": st.column_config.TextColumn(width="large")
                }
            )

            st.markdown("##### 🗑️ 파일 삭제 관리")
            files_to_delete = st.multiselect(
                "삭제할 파일을 선택하세요 (복수 선택 가능):",
                options=df_files["파일명"].tolist()
            )
            
            if files_to_delete:
                st.warning(f"선택한 {len(files_to_delete)}개 파일을 DB에서 영구 삭제하시겠습니까?")
                if st.button("🗑️ 선택 항목 영구 삭제", type="primary"):
                    try:
                        total_deleted_ids = []
                        for file_name in files_to_delete:
                            ids = file_stats[file_name]["ids"]
                            total_deleted_ids.extend(ids)
                        
                        if total_deleted_ids:
                            vectorstore.delete(ids=total_deleted_ids)
                            st.success(f"✅ 총 {len(total_deleted_ids)}개의 청크가 삭제되었습니다.")
                            time.sleep(1.5)
                            st.rerun()
                    except Exception as e:
                        st.error(f"삭제 중 오류 발생: {e}")

        else:
            st.info("학습된 규정 데이터가 없습니다. (데이터 0건)")

    except Exception as e:
        st.error(f"DB 로드 중 오류 발생: {e}")
        st.caption("DB 파일이 손상되었거나 경로가 잘못되었을 수 있습니다.")
else:
    st.info("학습된 규정 데이터가 없습니다. 사이드바에서 파일을 업로드하세요.")

st.divider()

# [2] 상황보고 데이터 상태
st.subheader("📈 상황보고 데이터 (Excel)")

RISK_FILE_PATH = os.path.join(SHARED_DIR, "risk_df.pkl")

if os.path.exists(RISK_FILE_PATH):
    try:
        saved_df = pd.read_pickle(RISK_FILE_PATH) 
        st.metric("저장된 상황보고 건수", f"{len(saved_df)} 건")
        
        st.markdown("**데이터 미리보기 (상위 5건):**")
        st.dataframe(saved_df.head(), use_container_width=True)
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
else:
    st.info("업로드된 상황보고 엑셀 데이터가 없습니다. 사이드바에서 엑셀을 업로드하세요.")

st.divider()

# ==================================================================
# admin.py 수정 코드
# [3] 피드백 루프 (Human-in-the-Loop) 부분 전체 교체
# ==================================================================

st.divider()

st.subheader("🎓 사용자 피드백 관리 (Human-in-the-Loop)")
st.caption("사용자의 피드백을 검토하여 AI를 강화학습 시키고, 과거 학습 이력을 확인합니다.")

# 절대 경로 사용
feedback_file = os.path.join(SHARED_DIR, "feedback_log.csv")

if os.path.exists(feedback_file):
    try:
        # 데이터 로드
        df_fb = pd.read_csv(feedback_file)
        
        # 탭 분리: [검토 대기] vs [학습 완료 이력]
        tab_review, tab_history = st.tabs(["🔥 검토 및 학습 (Pending)", "📜 학습 완료 로그 (History)"])
        
        # ----------------------------------------------------------
        # TAB 1: 검토 및 학습 (기존 기능)
        # ----------------------------------------------------------
        with tab_review:
            # Pending 상태만 필터링
            pending_df = df_fb[df_fb['Status'] == 'Pending']
            
            if pending_df.empty:
                st.info("🎉 현재 대기 중인 피드백이 없습니다. (모두 처리됨)")
            else:
                st.write(f"총 **{len(pending_df)}건**의 새로운 피드백이 대기 중입니다.")
                
                for index, row in pending_df.iterrows():
                    with st.expander(f"[{row['Rating']}] {str(row['Question'])[:40]}...", expanded=True):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.info(f"🤖 **AI 기존 답변:**\n\n{row['AI_Answer']}")
                        with c2:
                            st.error(f"👤 **사용자 수정 제안:**\n\n{row['User_Correction']}")
                        
                        btn_col1, btn_col2 = st.columns([1, 5])
                        with btn_col1:
                            # 학습 버튼
                            if st.button(f"✅ DB 학습 반영", key=f"train_{index}"):
                                try:
                                    raw_q = row['Question'] if pd.notna(row['Question']) else ""
                                    raw_cor = row['User_Correction'] if pd.notna(row['User_Correction']) else ""
                                    
                                    # [데이터 통일] 검색 엔진이 'Article_Title'에서 정답을 찾기 쉽게 구성
                                    unified_title = f"철도안전법 {raw_q[:15]}" 
                                    
                                    enhanced_content = f"질문: {raw_q}\n정답: {raw_cor}\n설명: 현장 전문가 검증을 거친 철도안전법 시행규칙 준수 사항입니다."

                                    new_doc = Document(
                                        page_content=enhanced_content, 
                                        metadata={
                                            "source": "Expert_Knowledge", # 출처 통일
                                            "type": "feedback",
                                            # "reward_score": 5.0,           # 검색 가중치를 위해 높은 보상 점수 부여
                                            # "Article_Title": unified_title, # 기존 규정과 매칭되도록 제목 부여
                                            "Article_Title": f"[검증] {raw_q[:15]}", # 제목 필드 강화(v1.3)
                                            "reward_score": 10.0, # 검색 순위 최상단 보장 점수(v1.3)
                                            "timestamp": str(datetime.datetime.now())
                                        }
                                    )
                                    
                                    # DB 저장
                                    vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=get_embeddings())
                                    vectorstore.add_documents([new_doc])
                                    
                                    # CSV 업데이트 (Status: Applied)
                                    df_fb.at[index, 'Status'] = 'Applied'
                                    df_fb.to_csv(feedback_file, index=False, encoding='utf-8-sig')
                                    
                                    st.success(f"✅ '{unified_title}' 지식이 강화학습 정책에 반영되었습니다.")
                                    st.rerun()
                                    
                                except Exception as e:
                                    st.error(f"학습 처리 중 오류: {e}")
                        
                        with btn_col2:
                            # 기각(삭제) 버튼
                            if st.button("🗑️ 무시(삭제)", key=f"del_{index}"):
                                df_fb.at[index, 'Status'] = 'Ignored'
                                df_fb.to_csv(feedback_file, index=False)
                                st.rerun()

                            # RL 관점: 잘못된 답변 패턴 기록
                            negative_doc = Document(
                                page_content=f"[부정 피드백]\n질문:{row['Question']}\n잘못된 응답:{row['AI_Answer']}",
                                metadata={
                                    "type": "negative_feedback",
                                    "reward_score": -1.0,
                                    "confidence": 0.9,
                                    "source": "관리자 기각"
                                }
                            )
                            vectorstore.add_documents([negative_doc])
                            st.rerun()

        # ----------------------------------------------------------
        # TAB 2: 학습 완료 로그 (신규 기능)
        # ----------------------------------------------------------
        with tab_history:
            # Applied 상태만 필터링 (최신순 정렬)
            history_df = df_fb[df_fb['Status'] == 'Applied'].sort_values(by='Timestamp', ascending=False)
            
            if history_df.empty:
                st.info("아직 학습에 반영된 이력이 없습니다.")
            else:
                st.success(f"총 **{len(history_df)}건**의 지식이 AI에 추가 학습되었습니다.")
                
                # 가독성을 위해 데이터프레임 표시 (필요한 컬럼만)
                display_cols = ['Timestamp', 'Question', 'User_Correction', 'Rating']
                
                # 컬럼명 한글화 (보기 좋게)
                display_df = history_df[display_cols].copy()
                display_df.columns = ['처리일시(접수)', '질문 내용', '학습시킨 정답', '평가']
                
                st.dataframe(
                    display_df, 
                    use_container_width=True,
                    hide_index=True
                )
                
                # 상세 보기 (아코디언 형태)
                with st.expander("🔍 상세 이력 조회 (클릭)"):
                    for i, row in history_df.iterrows():
                        st.markdown(f"**[{row['Timestamp']}] {row['Question']}**")
                        st.text(f"👉 학습된 정답: {row['User_Correction']}")
                        st.divider()

    except Exception as e:
        st.error(f"피드백 데이터 처리 중 오류 발생: {e}")

else:
    st.info("아직 수집된 피드백 데이터가 없습니다. (shared/feedback_log.csv 파일 없음)")
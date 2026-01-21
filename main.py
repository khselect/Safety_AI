import streamlit as st
import os
import pandas as pd
import time
import re
import csv
from datetime import datetime
import altair as alt
import json

# LangChain & Core
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.schema import HumanMessage # [개선2] HumanMessage 추가
from langchain_core.prompts import PromptTemplate

# Core 모듈
try:
    from core.config import PERSIST_DIRECTORY
    from core.llm import get_llm, get_embeddings
    from core.decision_ai import decision_ai
except ImportError:
    PERSIST_DIRECTORY = "./chroma_db"
    from core.llm import get_llm, get_embeddings

# ------------------------------------------------------------------
# [개선] 경로 설정 (절대 경로로 고정)
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DIR = os.path.join(BASE_DIR, "shared")
if not os.path.exists(SHARED_DIR):
    os.makedirs(SHARED_DIR)
# [설정 파일 경로 정의]
CONFIG_FILE = os.path.join(SHARED_DIR, "system_config.json")

st.set_page_config(page_title="철도안전 AI 시스템", layout="wide")
st.title("🚄 철도안전 AI 통합 분석 시스템 (v1.0)")

# ------------------------------------------------------------------
# 함수 정의
# ------------------------------------------------------------------
def get_vectorstore():
    # 1. DB 폴더가 존재하는지 확인
    if not os.path.exists(PERSIST_DIRECTORY):
        return None
    
    try:
        # 2. ChromaDB 로드
        # 주의: collection_name="regulations" 부분을 삭제했습니다.
        # admin.py에서 저장한 기본 설정과 맞추기 위함입니다
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=get_embeddings()
        )
        return vectorstore
    except Exception as e:
        st.error(f"벡터 저장소 로드 중 오류 발생: {e}")
        return None

# [신규 함수] 설정된 모델명 가져오기
def get_selected_model():
    default_model = "korean-llama3:latest"
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
                return config.get("selected_model", default_model)
        except:
            return default_model
    return default_model

def save_feedback(user_q, ai_a, user_correction, rating):
    """사용자 피드백을 CSV에 저장 (관리자 학습용)"""
    feedback_file = os.path.join(SHARED_DIR, "feedback_log.csv")
    
    # 파일이 없으면 헤더 생성
    if not os.path.exists(feedback_file):
        with open(feedback_file, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Question", "AI_Answer", "User_Correction", "Rating", "Status"])

    # 데이터 추가
    with open(feedback_file, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user_q,
            ai_a,
            user_correction,
            rating,
            "Pending"  # 관리자가 아직 반영 안 함
        ])

# def query_regulation(query, vectorstore, llm):
#     """
#     질문에 대해 벡터 저장소에서 문서를 찾고, 
#     '사용자 피드백'을 최우선으로 정렬하여 답변을 생성합니다.
#     """
#     # 1. 검색 수행 (피드백 누락 방지를 위해 k값을 넉넉히 8로 설정)
#     # MMR(Maximal Marginal Relevance)을 사용하여 다양한 맥락의 문서를 가져옵니다.
#     retriever = vectorstore.as_retriever(
#         search_type="mmr",
#         search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.5}
#     )
#     docs = retriever.invoke(query)

#     if not docs:
#         return "관련된 규정 정보를 찾을 수 없습니다.", []

#     # 2. [핵심] 사용자 피드백 우선순위 정렬
#     # metadata['type'] == 'feedback' 이거나 source가 '사용자 피드백'인 것을 최상단으로 올림
#     def get_priority(doc):
#         m = doc.metadata
#         reward = float(m.get("reward_score", 0))
#         confidence = float(m.get("confidence", 0.5))

#         # 시간 감쇠 (오래된 피드백은 영향 감소)
#         try:
#             ts = datetime.fromisoformat(m.get("timestamp"))
#             age_days = (datetime.now() - ts).days
#             decay = max(0.5, 1 - age_days / 365)
#         except:
#             decay = 1.0
        
#         # 핵심 강화학습 근사식입니다
#         return reward * confidence * decay

#     sorted_docs = sorted(
#         [d for d in docs if d.metadata.get("reward_score", 0) >= 0],
#         key=get_priority,
#         reverse=True
#     )

#     # 3. 컨텍스트 구성 (출처를 명확히 하여 LLM이 판단하기 쉽게 함)
#     context_parts = []
#     for d in sorted_docs:
#         source_name = d.metadata.get('source', '알 수 없음')
#         # 파일 경로일 경우 파일명만 추출
#         source_name = os.path.basename(source_name)
        
#         # 피드백 데이터인 경우 강조 표시
#         if get_priority(d) >= 1:
#             context_parts.append(f"[‼️ 최신 교정 정보 - 출처: {source_name}]\n{d.page_content}")
#         else:
#             context_parts.append(f"[출처: {source_name}]\n{d.page_content}")
    
#     context_text = "\n\n".join(context_parts)

#     # 4. 프롬프트 정의 (피드백 최우선 원칙 강제)
#     prompt_template = """
#     ### [Role]
#     당신은 철도 안전 규정 및 실무 전문가입니다. 
#     제공된 [규정 및 피드백 문맥]만을 바탕으로 질문에 답변하십시오.

#     ### [Priority Rule - 매우 중요]
#     1. 문맥 중 **'[‼️ 최신 교정 정보]'**라고 표시된 내용은 사용자가 직접 교정한 정답입니다.
#     2. 일반 규정 파일의 내용과 '최신 교정 정보'의 내용이 서로 충돌하거나 다를 경우, **반드시 '최신 교정 정보'를 정답으로 채택**하십시오.
#     3. 만약 이전의 'Feedback' 파일과 현재의 '사용자 피드백' 내용이 다르다면, 가장 위에 배치된 내용을 신뢰하십시오.

#     ### [Guidelines]
#     - 반드시 한국어(Korean)로 답변하십시오.
#     - 답변 시 "최신 교정된 피드백에 따르면..." 또는 "현행 규정 제O조에 따르면..."과 같이 근거를 밝히십시오.

#     [규정 및 피드백 문맥]
#     {context}

#     질문: {question}

#     답변:
#     """
    
#     full_prompt = prompt_template.format(context=context_text, question=query)
    
#     try:
#         # LLM 호출
#         response = llm.invoke([HumanMessage(content=full_prompt)])
#         return response.content, sorted_docs
#     except Exception as e:
#         return f"답변 생성 중 오류가 발생했습니다: {e}", sorted_docs

# [수정] query_regulation 함수: 키워드 정밀도 및 전문가 정책 반영
def query_regulation(query, vectorstore, llm):
    # 1. 전문가 피드백(Applied) 데이터 최우선 확인 (Fast-Track)
    # 관리자가 admin.py에서 '청원휴가' 정답을 승인했다면 여기서 즉시 반환됩니다.
    feedback_file = os.path.join(SHARED_DIR, "feedback_log.csv")
    if os.path.exists(feedback_file):
        try:
            fb_df = pd.read_csv(feedback_file)
            # 질문의 앞 5글자가 포함된 'Applied' 상태의 정답 검색
            match = fb_df[(fb_df['Status'] == 'Applied') & (fb_df['Question'].str.contains(query[:5]))]
            if not match.empty:
                return f"✅ **[전문가 검증 답변]**\n\n{match.iloc[-1]['User_Correction']}", []
        except: pass

    # 2. 하이브리드 검색 엔진 구성 (BM25 + Vector)
    # [문제해결] '청원휴가'라는 정확한 단어를 찾기 위해 BM25 가중치를 0.8로 설정
    all_docs_data = vectorstore.get()
    all_docs = [
        Document(page_content=text, metadata=meta) 
        for text, meta in zip(all_docs_data['documents'], all_docs_data['metadatas'])
    ]
    
    # 키워드 검색기 (단어 일치 중심)
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 3
    
    # 벡터 검색기 (의미 유사도 중심)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 앙상블: 법령은 키워드가 중요하므로 BM25에 압도적 가중치 부여
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.8, 0.2] 
    )
    
    docs = ensemble_retriever.invoke(query)

    # 3. 제목 기반 재정렬 (Article_Title 우선순위 부여)
    # 질문 키워드가 조문 제목에 포함되어 있다면 검색 결과 1위로 올림
    query_keywords = query.split()
    docs = sorted(docs, key=lambda d: any(kw in d.metadata.get('Article_Title', '') for kw in query_keywords), reverse=True)

    # 4. 프롬프트 구성 (LLM의 환각 방지 지침 강화)
    context_text = "\n\n".join([f"### {doc.metadata.get('Article_Title', '조문 정보 없음')}\n{doc.page_content}" for doc in docs])
    
    template = """당신은 우리공사 규정 전문가입니다. 
    제공된 [규정 내용]의 '제목'과 '본문'을 대조하여 질문에 정확히 답하세요.

    [규정 내용]
    {context}

    [질문]
    {question}

    [답변 가이드라인]
    1. 질문이 '{question}'인 경우, 제목에 이 단어가 포함된 조문(예: 제26조 청원휴가)을 우선적으로 설명하세요.
    2. 조문 번호를 절대 다른 조문(제9조 등)과 혼동하지 마세요.
    3. 일반적인 노동법 상식을 섞지 말고, 오직 위에 제공된 텍스트로만 답변하세요.
    """
    
    prompt_text = template.format(context=context_text, question=query)
    response = llm.invoke([HumanMessage(content=prompt_text)])
    
    return response.content, docs

# ------------------------------------------------------------------
# 세션 상태 초기화
# ------------------------------------------------------------------
if "llm" not in st.session_state:
    st.session_state["llm"] = get_llm("korean-llama3")

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = get_vectorstore()

# 대화 기록 초기화 (messages 리스트에 sources 정보도 함께 저장)
if "messages" not in st.session_state:
    st.session_state["messages"] = []

llm = st.session_state["llm"]
vectorstore = st.session_state["vectorstore"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DIR = os.path.join(BASE_DIR, "shared")
FILE_PATH = os.path.join(SHARED_DIR, "risk_df.pkl")

# ------------------------------------------------------------------
# 사이드바
# ------------------------------------------------------------------
# with st.sidebar:
#     st.header("🔌 시스템 상태")
#     if vectorstore is not None:
#         try:
#             count = vectorstore._collection.count()
#             st.success(f"규정 DB 연결됨 (문서 청크: {count}개)")
#         except:
#             st.warning("규정 DB 연결 불안정")
#     else:
#         st.error("규정 DB를 찾을 수 없습니다.")
        
#     if st.button("대화 내용 초기화"):
#         st.session_state["messages"] = []
#         st.rerun()

# ======================================
# 탭 구성
# ======================================
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 규정 챗봇",
    "📈 위험 상황 대시보드",
    "🧠 통합 위험 분석",
    "🚨 위험 판단 & 조치 추천"
])

# ==================================================================
# TAB 1. 💬 규정 챗봇 (멀티턴 + 고급검색 + 피드백 루프)
# ==================================================================
with tab1:
    
    current_model = get_selected_model()
    st.markdown(f"#### 💬 철도안전 규정 전문 챗봇 (Model: :orange[{current_model}])")
    st.caption("💡 규정 검색부터 업무 질의까지, AI가 문맥을 이해하고 답변합니다.")
    
    # [1] 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # [2] 새로운 사용자 입력 처리
    if prompt := st.chat_input("규정에 대해 궁금한 점을 물어보세요..."):
        
        # 3-1. 사용자 메시지 즉시 표시 및 저장
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 3-2. 규정 DB 로드 확인
        vectorstore = get_vectorstore()
        
        if vectorstore is None:
            st.error("🚨 학습된 규정 DB가 없습니다. 관리자 페이지에서 문서를 먼저 학습시켜주세요.")
        else:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                status_placeholder = st.empty()
                
                with st.spinner("규정 정밀 분석 및 답변 생성 중..."):
                    try:
                        # --- [신규 추가] 1. 전문가 피드백 데이터 우선 확인 로직 ---
                        feedback_file = os.path.join(SHARED_DIR, "feedback_log.csv")
                        verified_answer = None
                        
                        if os.path.exists(feedback_file):
                            fb_df = pd.read_csv(feedback_file)
                            # Applied(학습 완료) 상태이면서 사용자의 질문과 유사한 데이터 검색
                            # (질문의 앞 7글자 정도가 포함되거나, 질문 전체가 포함되는 경우)
                            match = fb_df[
                                (fb_df['Status'] == 'Applied') & 
                                (fb_df['Question'].apply(lambda x: x[:7] in prompt or prompt[:7] in x))
                            ]
                            
                            if not match.empty:
                                # 가장 최근에 승인된 정답을 가져옴
                                verified_answer = match.iloc[-1]['User_Correction']

                        # --- 2. 답변 결정 (피드백 정답 vs RAG 생성) ---
                        if verified_answer:
                            # 전문가가 이미 교정한 정답이 있는 경우
                            response_text = f"✅ **[전문가 검증 답변]**\n\n{verified_answer}"
                            docs = [] # 피드백 답변이므로 별도 검색 결과는 비움 (또는 검색 병행 가능)
                            status_msg = "💡 관리자가 승인한 전문가 지식베이스에서 정답을 찾았습니다."
                        else:
                            # 검증된 정답이 없는 경우 기존 RAG 로직 실행
                            model_name = get_selected_model() 
                            llm_instance = get_llm(model_name)
                            response_text, docs = query_regulation(prompt, vectorstore, llm_instance)
                            status_msg = "🔍 피드백 우선 검색 알고리즘이 적용되었습니다. (AI 생성 답변)"
                            
                        # 소스 정보 구조화 (UI 출력용)
                        sources_for_ui = []
                        seen_titles = set()
                        for doc in docs:
                            src_file = os.path.basename(doc.metadata.get("source", "파일"))
                            title = doc.metadata.get("Article_Title", "본문")
                            
                            key = (src_file, title)
                            if key not in seen_titles:
                                sources_for_ui.append({
                                    "source": src_file, 
                                    "title": title, 
                                    "content": doc.page_content
                                })
                                seen_titles.add(key)

                        # 결과 출력
                        message_placeholder.markdown(response_text)
                        status_placeholder.caption("🔍 피드백 우선 검색 알고리즘이 적용되었습니다.")
                        
                        # 세션에 Assistant 메시지 저장
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response_text,
                            "sources": sources_for_ui,
                            "status": "🔍 피드백 우선 검색 결과입니다.",
                            "timestamp": time.time()
                        })
                        
                        # 화면 갱신 (피드백 버튼 및 대화 내역 반영)
                        st.rerun()

                    except Exception as e:
                        st.error(f"오류 발생: {e}")

    # [3] 대화 내역 출력 (if prompt 블록 외부)
    # 질문을 입력하지 않았을 때도 이전 대화가 보여야 하므로 인덴트를 밖으로 뺍니다.
    st.divider()
    
    if st.session_state.messages:
        # 메시지를 (질문, 답변) 쌍으로 그룹화
        conversations = []
        msgs = st.session_state.messages
        
        for i in range(0, len(msgs), 2):
            if i + 1 < len(msgs):
                conversations.append((msgs[i], msgs[i+1]))
            else:
                conversations.append((msgs[i], None))
        
        # 최신 대화가 위로 오도록 역순 출력
        for user_msg, ai_msg in reversed(conversations):
            with st.container():
                # A. 사용자 질문
                if user_msg:
                    with st.chat_message("user"):
                        st.write(user_msg["content"])
                
                # B. AI 답변
                if ai_msg:
                    with st.chat_message("assistant"):
                        st.write(ai_msg["content"])
                        
                        if ai_msg.get("status"):
                            st.caption(ai_msg["status"])
                        
                        if ai_msg.get("sources"):
                            with st.expander("📚 근거 규정 보기"):
                                for src in ai_msg["sources"]:
                                    st.markdown(f"**📄 {src['source']} - {src['title']}**")
                                    # 표 형식 깨짐 방지 및 길이 제한
                                    safe_content = src['content'].replace("|", " ").replace("\n", " ")[:250]
                                    st.caption(f"{safe_content}...")
                        
                        # 피드백 버튼 UI
                        ts = ai_msg.get("timestamp", int(time.time()))
                        fb_key = f"fb_{ts}"
                        
                        col_f1, col_f2 = st.columns([1, 4])
                        with col_f1:
                            if st.button("👍 좋아요", key=f"lk_{fb_key}"):
                                save_feedback(user_msg["content"], ai_msg["content"], "", "Good")
                                st.toast("평가 감사합니다!")
                        with col_f2:
                            with st.popover("👎 수정 제안"):
                                correction = st.text_area("올바른 내용:", key=f"tx_{fb_key}")
                                if st.button("전송", key=f"sd_{fb_key}"):
                                    if correction:
                                        save_feedback(user_msg["content"], ai_msg["content"], correction, "Bad")
                                        st.success("전송되었습니다. 관리자가 검토 후 DB에 반영합니다.")

            st.divider()
                                
# ======================================
# TAB 2. 📈 위험 상황 대시보드 (Professional Ver.)
# ======================================
with tab2:
    # 1. 헤더 & 컨트롤 패널
    col_header, col_filter = st.columns([3, 1])
    
    with col_header:
        st.subheader("📈 위험 상황 대시보드")
        st.caption("데이터 기반 사고 원인 및 추세 심층 분석")
        
    with col_filter:
        line_options = ["전체", "1호선", "2호선", "7호선"]
        selected_line = st.selectbox("🔍 호선 필터", line_options)

    st.markdown("---")

    if os.path.exists(FILE_PATH):
        try:
            df = pd.read_pickle(FILE_PATH)
            
            # -----------------------------------------------------------
            # [1] 데이터 전처리
            # -----------------------------------------------------------
            col_map = {
                "line": "호선",
                "date": "발생일자",
                "cause": "부원인",
                "place": "발생장소",
                "r_type": "귀책구분",
                "age": "연령대"
            }
            
            for key, actual_col in col_map.items():
                if actual_col not in df.columns:
                    df[actual_col] = "정보없음" if key != "date" else "2024-01-01"

            # 호선 정제 함수
            def clean_line_name(val):
                val_str = str(val)
                if "1호선" in val_str: return "1호선"
                if "2호선" in val_str: return "2호선"
                if "7호선" in val_str: return "7호선"
                return "기타"
            
            # 괄호 및 숫자 제거 함수 ([2]음주 -> 음주)
            def clean_label_text(val):
                val_str = str(val)
                # 대괄호와 그 안의 숫자/문자 제거 후 앞뒤 공백 제거
                return re.sub(r'\[.*?\]', '', val_str).strip()

            df["호선_정제"] = df[col_map["line"]].apply(clean_line_name)
            df[col_map["date"]] = pd.to_datetime(df[col_map["date"]], errors='coerce')
            df["월"] = df[col_map["date"]].dt.strftime('%Y-%m')

            # 분석용 컬럼들에 대해 라벨 클리닝 미리 적용 (가독성 향상)
            target_cols_clean = [col_map["cause"], col_map["place"], col_map["r_type"], col_map["age"]]
            for col in target_cols_clean:
                df[col] = df[col].apply(clean_label_text)

            # -----------------------------------------------------------
            # [2] 데이터 필터링
            # -----------------------------------------------------------
            target_lines = ["1호선", "2호선", "7호선"]
            
            if selected_line == "전체":
                filtered_df = df[df["호선_정제"].isin(target_lines)]
                if filtered_df.empty: filtered_df = df 
            else:
                filtered_df = df[df["호선_정제"] == selected_line]

            # -----------------------------------------------------------
            # [3] 대시보드 시각화
            # -----------------------------------------------------------
            if not filtered_df.empty:
                
                # [KPI Section] 핵심 요약
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                
                total_cnt = len(filtered_df)
                kpi1.metric("총 발생 건수", f"{total_cnt}건")
                
                top_cause = filtered_df[col_map["cause"]].mode()[0] if not filtered_df[col_map["cause"]].empty else "-"
                cause_cnt = filtered_df[col_map["cause"]].value_counts().iloc[0] if not filtered_df[col_map["cause"]].empty else 0
                kpi2.metric("최다 빈도 원인", top_cause, f"{cause_cnt}건")
                
                top_place = filtered_df[col_map["place"]].mode()[0] if not filtered_df[col_map["place"]].empty else "-"
                kpi3.metric("주요 발생 장소", top_place)

                top_resp = filtered_df[col_map["r_type"]].mode()[0] if not filtered_df[col_map["r_type"]].empty else "-"
                kpi4.metric("주요 귀책 사유", top_resp)

                st.markdown("###")

                # [Chart Row 1] 시계열 분석 (기존 유지)
                st.markdown("##### 📅 월별 사고 발생 추이 및 원인 구성")
                time_chart_data = filtered_df.groupby(["월", col_map["cause"]]).size().reset_index(name='건수')
                
                time_chart = alt.Chart(time_chart_data).mark_bar().encode(
                    x=alt.X('월', title='기간'),
                    y=alt.Y('건수', title='발생 건수'),
                    color=alt.Color(col_map["cause"], title='부원인'),
                    tooltip=['월', col_map["cause"], '건수']
                ).properties(height=300) # 높이 약간 조정
                
                st.altair_chart(time_chart, use_container_width=True)

                st.divider()

                # [Chart Row 2] 상세 통계 분석 (1x4 구조로 변경 + 코멘트)
                st.markdown("##### 📊 상세 통계 및 인사이트 분석")

                # --- [함수] 차트, 테이블, 코멘트 생성 ---
                def create_analysis_component(data, col_name, title):
                    # 1. 데이터 집계
                    counts = data[col_name].value_counts().reset_index()
                    counts.columns = ["항목", "건수"]
                    
                    # 2. 비율 계산
                    total = counts["건수"].sum()
                    if total > 0:
                        counts["비율"] = ((counts["건수"] / total) * 100).round(1) # 소수점 1자리
                    else:
                        counts["비율"] = 0
                    
                    # 3. Altair 도넛 차트
                    base = alt.Chart(counts).encode(theta=alt.Theta("건수", stack=True))
                    
                    pie = base.mark_arc(innerRadius=50, outerRadius=90).encode(
                        color=alt.Color("항목", legend=alt.Legend(orient="right", title=None)), 
                        order=alt.Order("건수", sort="descending"),
                        tooltip=["항목", "건수", alt.Tooltip("비율", format=".1f")]
                    )
                    
                    # [수정] .filter() -> .transform_filter() 로 변경
                    text = base.mark_text(radius=110).encode(
                        text=alt.Text("비율", format=".0f"), 
                        order=alt.Order("건수", sort="descending"),
                        color=alt.value("black")
                    ).transform_filter(
                        alt.datum.비율 > 4  # 비율이 4% 초과인 것만 텍스트 표시
                    )

                    chart = (pie + text).properties(height=250)

                    # 4. 테이블 데이터 정리
                    table_df = counts[["항목", "건수", "비율"]].copy()
                    table_df.columns = ["항목", "건수", "비율(%)"]
                    
                    # 5. 자동 분석 텍스트 생성
                    if not counts.empty:
                        top1 = counts.iloc[0]
                        insight_text = f"""
                        - **최다 빈도:** <span style='color:red'>**{top1['항목']}**</span> ({top1['건수']}건, {top1['비율']}%)
                        """
                        
                        if len(counts) > 1:
                            top2 = counts.iloc[1]
                            diff = top1['건수'] - top2['건수']
                            insight_text += f"""
                            - **2위 항목:** {top2['항목']} ({top2['비율']}%)
                            - **분석:** 1위 항목이 2위 대비 **{diff}건** 더 많이 발생했습니다.
                            """
                    else:
                        insight_text = "데이터가 충분하지 않습니다."

                    return chart, table_df, insight_text

                # --- [메인 로직] 1 Row per Metric (1x4 Stack) ---
                metrics = [
                    ("cause", "1. 부원인 분석"),
                    ("place", "2. 발생장소 분석"),
                    ("r_type", "3. 귀책구분 분석"),
                    ("age", "4. 연령대별 분석")
                ]

                for col_key, title in metrics:
                    st.markdown(f"**📌 {title}**")
                    
                    # 레이아웃 비율 [차트(2) : 테이블(1.5) : 코멘트(1.5)]
                    c1, c2, c3 = st.columns([2, 1.5, 1.5])
                    
                    chart, df_table, insight = create_analysis_component(filtered_df, col_map[col_key], title)
                    
                    with c1:
                        st.altair_chart(chart, use_container_width=True)
                    
                    with c2:
                        st.dataframe(
                            df_table, 
                            hide_index=True, 
                            use_container_width=True,
                            height=200
                        )
                    
                    with c3:
                        st.info("💡 **AI Insight**")
                        st.markdown(insight, unsafe_allow_html=True)
                    
                    st.divider() # 항목 간 구분선

                # [List Section] 데이터 리스트 (기존 유지)
                st.markdown(f"##### 📋 {selected_line} Raw Data (상위 100건)")
                st.dataframe(
                    filtered_df.head(100), 
                    use_container_width=True, 
                    height=250, 
                    hide_index=True
                )
            
            else:
                st.warning(f"선택하신 '{selected_line}'에 해당하는 데이터가 없습니다.")

        except Exception as e:
            st.error(f"대시보드 생성 중 오류 발생: {e}")
            # 디버깅용 traceback 출력 (필요시 주석 해제)
            # import traceback
            # st.text(traceback.format_exc())
    else:
        st.info("아직 상황보고 데이터가 없습니다. 관리자 페이지에서 데이터를 업로드해주세요.")

# ==================================================================
# TAB 3. 🧠 통합 위험 분석 (Risk Matrix) - [시각화 강화 & 자동 제안 Ver]
# ==================================================================
with tab3:
    st.subheader("🧠 통합 위험도 평가 (Risk Matrix)")
    st.caption("발생 빈도(데이터 기반)와 심각도(사용자 설정)를 분석하여 위험 우선순위를 도출합니다.")
    
    # 파일 경로 변수 확인 (전역 변수 FILE_PATH 사용 가정)
    target_file = FILE_PATH if os.path.exists(FILE_PATH) else None
    
    if target_file:
        try:
            df_risk = pd.read_pickle(target_file)
            
            # ----------------------------------------------------------
            # [1] 데이터 전처리
            # ----------------------------------------------------------
            col_cause = "주원인" if "주원인" in df_risk.columns else "cause"
            if col_cause not in df_risk.columns:
                df_risk[col_cause] = "정보없음"
            
            unique_causes = df_risk[col_cause].unique()
            
            # ----------------------------------------------------------
            # [2] 심각도 설정 (키워드 기반 자동 제안)
            # ----------------------------------------------------------
            def suggest_severity(cause_text):
                text = str(cause_text)
                if any(k in text for k in ['사망', '폭발', '화재', '붕괴', '충돌']): return 5 # 치명적
                if any(k in text for k in ['추락', '협착', '끼임', '감전', '절단']): return 4 # 중대
                if any(k in text for k in ['골절', '화상', '누출']): return 3 # 보통
                if any(k in text for k in ['전도', '넘어짐', '부딪힘', '미끄러짐']): return 2 # 경미
                return 1 # 무시 가능

            with st.expander("⚙️ [설정] 사고 유형별 심각도 조정 (AI 자동 제안 적용됨)", expanded=False):
                st.info("💡 사고 유형 키워드를 분석하여 심각도 초기값을 자동으로 제안했습니다. 실제 상황에 맞게 조정해주세요.")
                
                suggested_data = [{"사고유형": c, "심각도": suggest_severity(c)} for c in unique_causes]
                df_severity_base = pd.DataFrame(suggested_data)
                
                edited_df = st.data_editor(
                    df_severity_base,
                    column_config={
                        "심각도": st.column_config.NumberColumn(
                            "심각도 (1-5)", 
                            help="1(경미) ~ 5(치명적)", 
                            min_value=1, max_value=5, step=1,
                            format="%d점"
                        )
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
            # ----------------------------------------------------------
            # [3] 위험도 계산 로직
            # ----------------------------------------------------------
            # 1. 빈도 계산
            df_freq = df_risk[col_cause].value_counts().reset_index()
            df_freq.columns = ["사고유형", "발생건수"]
            
            # 2. 심각도 병합
            df_calc = pd.merge(df_freq, edited_df, on="사고유형", how="left")
            
            # 3. 빈도 등급 계산
            max_cnt = df_calc["발생건수"].max()
            df_calc["빈도등급"] = df_calc["발생건수"].apply(
                lambda x: int((x / max_cnt) * 4.99) + 1 if max_cnt > 0 else 1
            )
            
            # 4. 위험 점수 및 등급 판정
            df_calc["위험점수"] = df_calc["빈도등급"] * df_calc["심각도"]
            
            def get_grade(score):
                if score >= 15: return "High"
                elif score >= 8: return "Medium"
                return "Low"
            df_calc["위험등급"] = df_calc["위험점수"].apply(get_grade)
            
            # 5. 세션 저장 (Tab 4 연동)
            top_risks = df_calc.sort_values(["위험점수", "발생건수"], ascending=[False, False])
            st.session_state['priority_risks'] = top_risks

            st.divider()

            # ----------------------------------------------------------
            # [4] 시각화: 매트릭스 & 리스트
            # ----------------------------------------------------------
            c_left, c_right = st.columns([1.4, 1])
            
            with c_left:
                st.markdown("##### 📊 5x5 Risk Matrix")
                
                # --- [4-1] 매트릭스 데이터 준비 ---
                grid_data = []
                for s in range(1, 6):
                    for f in range(1, 6):
                        score = s * f
                        if score >= 15: color, label = "#FF7675", "High"
                        elif score >= 8: color, label = "#FDCB6E", "Med"
                        else: color, label = "#55EFC4", "Low"
                        grid_data.append({"심각도_X": s, "빈도_Y": f, "점수": score, "Color": color, "Label": label})
                df_grid_base = pd.DataFrame(grid_data)
                
                # 실제 데이터 집계
                df_agg = df_calc.groupby(['심각도', '빈도등급']).agg(
                    사고유형_리스트=('사고유형', lambda x: '<br>'.join(x[:10])),
                    대표사고유형=('사고유형', 'first'),
                    타입수=('사고유형', 'count'),
                    총발생건수=('발생건수', 'sum')
                ).reset_index()
                
                # 병합
                df_matrix_final = pd.merge(
                    df_grid_base, df_agg,
                    left_on=['심각도_X', '빈도_Y'], right_on=['심각도', '빈도등급'],
                    how='left'
                ).fillna({'타입수': 0, '총발생건수': 0, '사고유형_리스트': '-'})

                # 라벨 컬럼 생성
                def create_label(row):
                    if row['타입수'] > 1:
                        return f"{row['대표사고유형']} 외 {int(row['타입수'])-1}건"
                    elif row['타입수'] == 1:
                        return str(row['대표사고유형'])
                    else:
                        return ""

                df_matrix_final['셀_텍스트'] = df_matrix_final.apply(create_label, axis=1)

                # --- [4-2] Altair 차트 구성 ---
                base = alt.Chart(df_matrix_final).encode(
                    x=alt.X('심각도_X:O', title='심각도 (중대성) ➡️', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('빈도_Y:O', title='빈도 (가능성) ⬆️', sort="descending")
                )

                # Layer 1: 배경
                heatmap = base.mark_rect(stroke='white', strokeWidth=1).encode(
                    color=alt.Color('Color', scale=None, legend=None),
                    tooltip=[
                        alt.Tooltip('Label', title='위험등급'),
                        alt.Tooltip('점수', title='위험점수'),
                        alt.Tooltip('총발생건수', title='총 발생 건수'),
                        alt.Tooltip('타입수', title='포함된 사고유형 수'),
                        alt.Tooltip('사고유형_리스트', title='사고유형 목록')
                    ]
                )

                # Layer 2: 점수
                text_score = base.mark_text(align='right', baseline='top', dx=25, dy=-25, size=11, opacity=0.6).encode(
                    text=alt.Text('점수', format='d'),
                    color=alt.value('black')
                )
                
                # Layer 3: 내용
                text_content = base.transform_filter(
                    alt.datum.타입수 > 0 
                ).mark_text(baseline='middle', size=12, fontWeight='bold', dy=5).encode(
                    text=alt.Text('셀_텍스트:N'),
                    color=alt.value('black')
                )

                chart = alt.layer(heatmap, text_score, text_content).properties(
                    width='container', height=400
                ).configure_axis(labelFontSize=12, titleFontSize=14)
                
                st.altair_chart(chart, use_container_width=True)
                st.caption("💡 마우스를 올리면 상세 정보를 확인할 수 있습니다.")

            with c_right:
                st.markdown("##### 🚨 위험 우선순위 (Top Risks)")
                
                if not top_risks.empty:
                    worst = top_risks.iloc[0]
                    st.error(
                        f"**⚠️ 최우선 관리 대상**\n\n"
                        f"### {worst['사고유형']}\n"
                        f"- 위험점수: **{worst['위험점수']:.0f}점** ({worst['위험등급']})\n"
                        f"- 발생: {worst['발생건수']}건 / 심각도: {worst['심각도']}등급"
                    )
                
                st.divider()
                
                # ==========================================================
            # [추가 기능] ℹ️ 위험성 평가 기준 및 로직 설명 (Legend)
            # ==========================================================
            with st.expander("ℹ️ 위험성 평가 기준 및 산정 로직 (상세 보기)", expanded=True):
                st.caption("본 시스템은 철도안전관리체계 기술기준 및 ICAO SMS 매뉴얼을 준용한 위험도 평가 모델을 따릅니다.")
                
                l_col1, l_col2, l_col3 = st.columns(3)
                
                # 1. 심각도 (Severity) 정의
                with l_col1:
                    st.markdown("**1️⃣ 심각도(Severity) 산정 기준**")
                    st.markdown(
                        """
                        <div style='font-size:13px; background-color:#f9f9f9; padding:10px; border-radius:5px;'>
                        <b>키워드 기반 자동 매핑 (AI)</b><br>
                        <span style='color:#FF4B4B'>🔴 5점 (치명):</span> 사망, 폭발, 화재, 붕괴<br>
                        <span style='color:#FF8800'>🟠 4점 (중대):</span> 추락, 협착, 끼임, 감전<br>
                        <span style='color:#FFBB00'>🟡 3점 (보통):</span> 골절, 화상, 누출<br>
                        <span style='color:#00CC96'>🟢 2점 (경미):</span> 전도, 넘어짐, 부딪힘<br>
                        <span style='color:grey'>⚪ 1점 (무시):</span> 기타 경미한 사항
                        </div>
                        """, unsafe_allow_html=True
                    )

                # 2. 빈도 (Frequency) 로직
                with l_col2:
                    st.markdown("**2️⃣ 빈도(Frequency) 계산 로직**")
                    st.markdown(
                        """
                        <div style='font-size:13px; background-color:#f9f9f9; padding:10px; border-radius:5px;'>
                        <b>상대 평가 (Relative Grading)</b><br>
                        1.데이터 내 최다 발생 건수를 기준, 5등급 구간으로 자동 환산<br>
                        [예: 최다 100건일 때, 80건 이상은 5등급]<br>
                        2.경각심을 위한 보수적 평가, 등급은 소수점 올림(ceiling) 처리<br>
                        [예: 2.5 -> 3등급 (무조건 올림)]<br>
                        
                        </div>
                        """, unsafe_allow_html=True
                    )

                # 3. 위험도 (Risk) 판정
                with l_col3:
                    st.markdown("**3️⃣ 위험 등급(Risk Grade) 판정**")
                    st.markdown(
                        """
                        <div style='font-size:13px; background-color:#f9f9f9; padding:10px; border-radius:5px;'>
                        <b>Risk Score = 심각도 × 빈도</b><br>
                        <br>
                        <span style='background-color:#FFDDDD; padding:2px 5px; border-radius:3px;'>🔴 <b>High (15~25)</b></span>
                        즉시 개선 대책 수립 필요 (Tab 4 연동)<br>
                        <span style='background-color:#FFF8DD; padding:2px 5px; border-radius:3px;'>🟡 <b>Medium (8~14)</b></span>
                        지속적 모니터링 및 관리 필요<br>
                        <span style='background-color:#DDFFDD; padding:2px 5px; border-radius:3px;'>🟢 <b>Low (1~7)</b></span>
                        현 상태 유지 및 관찰
                        </div>
                        """, unsafe_allow_html=True
                    )

                st.markdown("**상위 위험 리스트**")
                display_df = top_risks[['사고유형', '위험등급', '위험점수', '발생건수']].head(5)
                st.dataframe(
                    display_df, hide_index=True, use_container_width=True,
                    column_config={
                        "위험등급": st.column_config.TextColumn("등급"),
                        "위험점수": st.column_config.ProgressColumn("위험 점수", format="%d점", min_value=0, max_value=25),
                    }
                )
                st.info("👉 **Tab 4**에서 AI 조치 매뉴얼을 확인하세요.")
                    
        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")

    else:
        st.warning("분석할 데이터 파일이 없습니다.")


# ==================================================================
# TAB 4. 🚨 위험 판단 & 조치 추천 (Action Plan) - [레이아웃 개선 Ver]
# ==================================================================
with tab4:
    st.subheader("🚨 위험 대응 솔루션 (Action Plan)")
    st.caption("위험 요인에 대한 구체적인 조치 방안을 전체 화면으로 생성합니다.")

    if 'priority_risks' in st.session_state and not st.session_state['priority_risks'].empty:
        priority_df = st.session_state['priority_risks']
        risk_list = priority_df['사고유형'].tolist()
        
        # 1. 상단 컨트롤 패널
        with st.container():
            c1, c2, c3 = st.columns([2, 2, 1])
            
            with c1:
                selected_risk = st.selectbox("📌 분석할 위험 요인 선택", risk_list)
            
            # 선택된 위험 요인 정보
            target_row = priority_df[priority_df['사고유형'] == selected_risk].iloc[0]
            
            with c2:
                st.metric(
                    label="위험도 정보", 
                    value=f"{target_row['위험등급']} ({target_row['위험점수']}점)",
                    delta=f"발생 {target_row['발생건수']}건",
                    delta_color="inverse"
                )
            
            with c3:
                st.write("") # 줄바꿈
                btn_generate = st.button("🧬 조치방안 생성", type="primary", use_container_width=True)

        st.divider()

        # 2. 결과 출력 화면
        if btn_generate:
            # (주의) get_vectorstore, get_selected_model, get_llm 함수가 main.py에 정의되어 있어야 함
            vectorstore = get_vectorstore() 
            if not vectorstore:
                st.error("🚨 벡터 DB가 로드되지 않았습니다. Tab 1에서 DB 상태를 확인해주세요.")
            else:
                with st.spinner(f"'{selected_risk}' 관련 규정 분석 및 조치안 작성 중..."):
                    try:
                        # RAG 검색
                        query = f"{selected_risk} 사고 예방 작업 절차 안전 수칙"
                        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                        docs = retriever.invoke(query)
                        context = "\n".join([d.page_content for d in docs])
                        
                        # 프롬프트 생성
                        prompt = f"""
                        당신은 철도 안전 전문가입니다. 
                        아래 [검색된 규정]을 근거로 '{selected_risk}' 위험에 대한 구체적인 행동 매뉴얼을 작성하세요.
                        
                        [검색된 규정]
                        {context}
                        
                        [작성 요령]
                        1. 제목을 크고 명확하게 작성하세요.
                        2. '작업 전', '작업 중', '비상 시' 단계별로 구체적으로 서술하세요.
                        3. 규정에 없는 내용은 일반 안전 수칙을 적용하되 명시하세요.
                        4. 불릿 포인트를 활용해 가독성을 높이세요.
                        """
                        
                        model_name = get_selected_model()
                        llm = get_llm(model_name)
                        
                        # LLM 호출
                        if hasattr(llm, 'invoke'):
                            resp = llm.invoke([HumanMessage(content=prompt)])
                            result_text = resp.content
                        else:
                            result_text = llm.predict(prompt)
                            
                        # 결과 출력
                        st.markdown(f"### 📋 [{selected_risk}] 안전 조치 가이드")
                        
                        with st.container(border=True):
                            st.markdown(result_text)
                        
                        with st.expander("📎 참고한 규정 원문 보기"):
                            st.text(context)
                            
                    except Exception as e:
                        st.error(f"생성 중 오류 발생: {e}")
    else:
        st.warning("⚠️ Tab 3(통합 위험 분석)를 먼저 실행하여 위험 데이터를 생성해주세요.")
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
from langchain_chroma import Chroma
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
        # admin.py에서 저장한 기본 설정과 맞추기 위함입니다.
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

def query_regulation(query, vectorstore, llm):
    """
    질문에 대해 벡터 저장소에서 문서를 찾고 LLM이 답변을 생성합니다.
    """
    # 검색 범위 (k=6)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    
    docs = retriever.invoke(query)
    if not docs:
        return "해당 질문과 관련된 규정 정보를 찾을 수 없습니다.", []

    prompt_template = """
    ### [Role]
    당신은 한국의 철도 안전 규정 전문가입니다. 
    아래 [규정 문맥]을 바탕으로 사용자의 질문에 답변하십시오.

    ### [Guidelines]
    1. **반드시 한국어(Korean)로만 답변하십시오.** (Do not use English).
    2. 답변은 [규정 문맥]에 있는 내용에만 기반해야 합니다.
    3. 규정에 없는 내용을 질문하면 "규정에 관련 내용이 없습니다"라고 답하세요.
    4. 조항 번호(예: 제3조)나 수치(예: 10m, 30%)는 정확히 인용하세요.
    5. 답변 톤은 전문적이고 명확하며 친절하게 작성하세요.

    [규정 문맥]:
    {context}

    질문: {question}
    
    답변:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    result = chain.invoke({"query": query})
    return result["result"], result["source_documents"]
                                
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

    # [2] 이전 대화 기록 출력 (채팅방 스타일)
    # for msg in st.session_state.messages:
    #     with st.chat_message(msg["role"]):
    #         st.markdown(msg["content"])
            
    #         if msg["role"] == "assistant":
    #             if msg.get("sources"):
    #                 with st.expander("📚 근거 규정 및 출처 확인"):
    #                     for src in msg["sources"]:
    #                         st.markdown(f"**📄 {src['source']}**")
    #                         safe_content = src['content'].replace("|", " ").replace("\n", " ")[:200]
    #                         st.caption(f"{safe_content}...")
    #             if msg.get("status"):
    #                 st.caption(msg["status"])

    # [3] 새로운 사용자 입력 처리
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
                        # --- [Core 1] 대화 히스토리 포맷팅 ---
                        history_text = ""
                        recent_msgs = st.session_state.messages[-6:-1] 
                        if recent_msgs:
                            history_text = "[이전 대화 내역]\n"
                            for m in recent_msgs:
                                role_label = "User" if m["role"] == "user" else "Assistant"
                                history_text += f"- {role_label}: {m['content']}\n"
                        
                        # --- [Core 2] 스마트 필터링 ---
                        search_kwargs = {"k": 6}
                        status_msg = "🔍 전체 규정 문서에서 검색했습니다."
                        
                        try:
                            all_data = vectorstore.get()
                            unique_sources = list(set([m['source'] for m in all_data['metadatas'] if m]))
                            
                            target_source = None
                            for src in unique_sources:
                                base_name = os.path.basename(src).split('.')[0]
                                if len(base_name) >= 2 and base_name in prompt:
                                    target_source = src
                                    break
                            
                            if target_source:
                                search_kwargs["filter"] = {"source": target_source}
                                status_msg = f"🎯 **'{os.path.basename(target_source)}'** 문서 내에서 집중 검색했습니다."
                        except:
                            pass 
                        
                        # --- [Core 3] MMR 검색 수행 ---
                        retriever = vectorstore.as_retriever(
                            search_type="mmr",
                            search_kwargs={**search_kwargs, "fetch_k": 20, "lambda_mult": 0.6}
                        )
                        docs = retriever.invoke(prompt)
                        context_text = "\n\n".join([d.page_content for d in docs])
                        
                        # 소스 정보 구조화
                        sources_for_ui = []
                        seen_titles = set()
                        for doc in docs:
                            src_file = os.path.basename(doc.metadata.get("source", "파일"))
                            raw_title = doc.metadata.get("Article_Title", "본문")
                            match = re.match(r"(제\s*\d+\s*조(?:의\d+)?(?:\([^)]*\))?)", raw_title)
                            clean_title = match.group(1) if match else raw_title[:30]
                            
                            key = (src_file, clean_title)
                            if key not in seen_titles:
                                sources_for_ui.append({"source": src_file, "title": clean_title, "content": doc.page_content})
                                seen_titles.add(key)

                        # --- [Core 4] LLM 답변 생성 (수정된 부분) ---
                        if not context_text:
                            response_text = "죄송합니다. 관련된 규정 내용을 찾을 수 없습니다."
                        else:
                            tmpl = f"""
                            [System Instruction]
                            당신은 철도안전 규정 전문가입니다. 
                            
                            **답변 작성 원칙**:
                            1. **근거 중심**: 상상하지 말고 반드시 [Context]에 있는 내용으로만 답하세요. 
                            2. **맥락 유지**: [History]를 참고하여 대명사('그것', '앞의 내용')가 무엇을 지칭하는지 파악하세요.
                            3. **표/수치 유지**: 등급표, 지급율 등은 마크다운 표(Table)로 깔끔하게 정리하세요.
                            4. **조항 명시**: 가능하다면 "제OO조에 따르면..." 형태로 출처를 밝히세요.
                            5. **계산**: 수식을 설명할 때도 한국어로 풀어서 설명하십시오. (예: "A multiplied by B" -> "A에 B를 곱하여")
                            
                            {history_text}

                            [Context]:
                            {context_text}

                            [Current Question]:
                            {prompt}

                            [Answer]:
                            """
                            
                            # LLM 호출 및 변수 할당 (NameError 해결)
                            model_name = get_selected_model() 
                            llm = get_llm(model_name)
                            
                            if hasattr(llm, 'invoke'):
                                resp = llm.invoke([HumanMessage(content=tmpl)])
                                response_text = resp.content
                            else:
                                response_text = llm.predict(tmpl)

                        # --- [UI Update] 결과 출력 ---
                        # response_text가 위에서 반드시 할당되므로 에러가 발생하지 않습니다.
                        message_placeholder.markdown(response_text)
                        status_placeholder.caption(status_msg)
                        
                        # 근거 문서 아코디언
                        if sources_for_ui:
                            with st.expander("📚 근거 규정 및 출처 확인"):
                                for src in sources_for_ui:
                                    st.markdown(f"**📄 {src['source']} - {src['title']}**")
                                    safe_content = src['content'].replace("|", " ").replace("\n", " ")[:200]
                                    st.caption(f"{safe_content}...")

                        # 세션에 Assistant 메시지 저장
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response_text,
                            "sources": sources_for_ui,
                            "status": status_msg,
                            "timestamp": time.time() # 피드백 ID용
                        })
                        # 메시지 저장 후 화면 갱신 (피드백 버튼 활성화를 위해)
                        st.rerun()

                    except Exception as e:
                        st.error(f"오류 발생: {e}")

    # 5. 대화 내역 출력 (역순 + 피드백 UI 포함)
    st.divider()
    
    # 메시지가 있을 때만 처리
    if st.session_state.messages:
        # (1) 메시지를 (질문, 답변) 쌍으로 그룹화
        # 가정: 리스트는 항상 [User, Assistant, User, Assistant...] 순서로 저장됨
        conversations = []
        msgs = st.session_state.messages
        
        # 2개씩 묶어서 리스트에 담음
        for i in range(0, len(msgs), 2):
            if i + 1 < len(msgs):
                # (User Msg, Assistant Msg) 튜플로 저장
                conversations.append((msgs[i], msgs[i+1]))
            else:
                # 짝이 안 맞는 마지막 메시지 (혹시 모를 예외 처리)
                conversations.append((msgs[i], None))
        
        # (2) 그룹 자체를 역순으로 순회 (최신 대화 세트가 먼저 나옴)
        for user_msg, ai_msg in reversed(conversations):
            with st.container():
                # A. 사용자 질문 출력
                if user_msg:
                    with st.chat_message("user"):
                        st.write(user_msg["content"])
                
                # B. AI 답변 출력
                if ai_msg:
                    with st.chat_message("assistant"):
                        st.write(ai_msg["content"])
                        
                        # 부가 정보 (상태, 출처)
                        if ai_msg.get("status"):
                            st.caption(ai_msg["status"])
                        
                        if ai_msg.get("sources"):
                            with st.expander("📚 근거 규정 보기"):
                                for src in ai_msg["sources"]:
                                    st.markdown(f"**📄 {src['source']}**")
                                    safe_content = src['content'].replace("|", " ").replace("\n", " ")[:200]
                                    st.caption(f"{safe_content}...")
                        
                        # 피드백 버튼 (답변 바로 아래 위치)
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
                                        st.success("전송되었습니다.")

            # 세트 간 구분선
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
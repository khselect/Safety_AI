import os
import re
import pandas as pd
import streamlit as st
import altair as alt  

# ------------------------------------------------------------------
# 라이브러리 및 설정
# ------------------------------------------------------------------
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# core 모듈 설정
from core.config import PERSIST_DIRECTORY
from core.llm import get_llm, get_embeddings
from core.decision_ai import decision_ai

# ------------------------------------------------------------------
# 기본 페이지 설정
# ------------------------------------------------------------------
st.set_page_config(
    page_title="사용자 콘솔",
    layout="wide"
)

st.title("🛡️ 철도안전 AI 통합 분석 시스템")

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

# # ======================================
# # TAB 1. 💬 규정 챗봇 (Smart Filtering & Enhanced Prompt 적용)
# # ======================================
# with tab1:
#     st.subheader("💬 규정 전문 챗봇")
#     st.caption("💡 팁: 규정 이름을 포함하면 정확도가 비약적으로 상승합니다.")
    
#     # [1] 상단 고정 입력창
#     with st.form(key="chat_form", clear_on_submit=True):
#         col1, col2 = st.columns([8, 1])
#         with col1:
#             user_input = st.text_input(
#                 "질문 입력", 
#                 placeholder="예: 보수규정에서 평가급 지급율은 어떻게 돼?", 
#                 label_visibility="collapsed"
#             )
#         with col2:
#             submit_btn = st.form_submit_button("질문하기", use_container_width=True)

#     # [2] 질문 처리 로직
#     if submit_btn and user_input:
#         if vectorstore is None:
#             st.error("⚠️ 규정 데이터베이스가 로드되지 않았습니다.")
#         else:
#             with st.spinner("규정 정밀 분석 및 답변 생성 중..."):
#                 # ------------------------------------------------------------------
#                 # [핵심 로직 1] 스마트 검색 필터링 (Smart Filtering)
#                 # ------------------------------------------------------------------
#                 # 1. DB에 있는 모든 파일명 가져오기
#                 try:
#                     all_data = vectorstore.get()
#                     unique_sources = list(set([m['source'] for m in all_data['metadatas']]))
#                 except:
#                     unique_sources = []

#                 # 2. 질문에 파일명이 포함되어 있는지 확인
#                 search_filter = None
#                 target_source_name = "전체 규정"
                
#                 for source in unique_sources:
#                     # 파일명 정제 (예: "인사규정.pdf" -> "인사규정")
#                     clean_name = os.path.splitext(os.path.basename(source))[0]
#                     # 질문에 키워드가 있고, 2글자 이상일 때 필터 적용
#                     if len(clean_name) >= 2 and clean_name in user_input:
#                         search_filter = {"source": source}
#                         target_source_name = clean_name
#                         break 
                
#                 # 3. 검색 수행 (필터 유무에 따라 전략 변경)
#                 if search_filter:
#                     # 특정 문서 지정 시: 해당 문서 집중 검색 (k=7)
#                     retriever = vectorstore.as_retriever(
#                         search_kwargs={"k": 7, "filter": search_filter}
#                     )
#                     status_msg = f"🎯 **'{target_source_name}'** 문서 내에서 집중 검색했습니다."
#                 else:
#                     # 지정 없을 시: 넓은 범위 검색 (k=10) -> 유사 문서가 많아도 정답 포함 확률 높임
#                     retriever = vectorstore.as_retriever(
#                         search_kwargs={"k": 10}
#                     )
#                     status_msg = "🔍 전체 규정 문서에서 검색했습니다."

#                 # 4. 문서 조회 및 중복 제거
#                 retrieved_docs = retriever.invoke(user_input)
                
#                 seen_content = set()
#                 final_docs = []
#                 for d in retrieved_docs:
#                     if d.page_content not in seen_content:
#                         final_docs.append(d)
#                         seen_content.add(d.page_content)
                
#                 # 컨텍스트 생성 (토큰 절약 위해 상위 6개 사용)
#                 context_text = "\n\n".join([d.page_content for d in final_docs[:6]])

#                 # ------------------------------------------------------------------
#                 # [핵심 로직 2] 프롬프트 엔지니어링 강화 (Enhanced Prompt)
#                 # ------------------------------------------------------------------
#                 if not context_text:
#                     response_text = "관련된 규정 내용을 찾을 수 없습니다."
#                 else:
#                     # 용어 혼동 방지 및 표 처리 지침 추가
#                     prompt_template = f"""
#                     [System Instruction]
#                     당신은 사내 규정 전문가입니다. 아래 [제공된 규정]을 바탕으로 질문에 답변하세요.
                    
#                     **중요 지침**:
#                     1. 답변은 반드시 한국어로만 작성하십시오.
#                     2. 질문에 언급된 규정 내 단어를 참조해(예: 보수, 인사, 안전) 최우선으로 인용하세요.
#                     3. 아래 [규정 문서]에 포함된 내용만 사용하여 답변하십시오.
#                     4. 불확실한 내용은 지어내지 말고 "규정에 명시되어 있지 않습니다"라고 답하세요.
#                     5. 가능한 경우 조문 번호와 함께 설명하십시오.

#                     [제공된 규정]:
#                     {context_text}

#                     [질문]:
#                     {user_input}

#                     [답변]:
#                     """
                    
#                     # LLM 호출 (main.py 상단에 llm 객체가 있다고 가정)
#                     try:
#                         response_text = llm.invoke(prompt_template).content
#                     except Exception as e:
#                         response_text = f"오류가 발생했습니다: {e}"

#                 # ------------------------------------------------------------------
#                 # [UI 처리] 메시지 저장 및 출처 정제 (기존 로직 유지+개선)
#                 # ------------------------------------------------------------------
                
#                 # 사용자 메시지 저장
#                 st.session_state.messages.append({"role": "user", "content": user_input})
                
#                 # 출처 정제 로직 (User Code 반영)
#                 formatted_sources = []
#                 seen_titles = set()
                
#                 for doc in final_docs: # 검색된 final_docs 사용
#                     source_file = os.path.basename(doc.metadata.get("source", "파일 정보 없음"))
#                     raw_title = doc.metadata.get("Article_Title", "조항 정보 없음")
                    
#                     # 제목 정제 (Regex)
#                     match = re.match(r"(제\s*\d+\s*조(?:의\d+)?(?:\([^)]*\))?)", raw_title)
#                     if match:
#                         clean_title = match.group(1)
#                     else:
#                         clean_title = raw_title[:30] + "..." if len(raw_title) > 30 else raw_title

#                     unique_key = (source_file, clean_title)
                    
#                     if unique_key not in seen_titles:
#                         formatted_sources.append({
#                             "source": source_file,
#                             "title": clean_title,
#                             "content": doc.page_content
#                         })
#                         seen_titles.add(unique_key)

#                 # AI 메시지 저장 (status_msg 포함)
#                 st.session_state.messages.append({
#                     "role": "assistant", 
#                     "content": response_text,
#                     "sources": formatted_sources,
#                     "status": status_msg # 검색 상태 정보 추가
#                 })

#     # [3] 대화 내용 출력 (역순)
#     st.divider()
    
#     # 메시지가 있으면 역순으로 출력
#     if "messages" in st.session_state and st.session_state.messages:
#         for msg in reversed(st.session_state.messages):
#             with st.chat_message(msg["role"]):
#                 st.write(msg["content"])
                
#                 # AI 답변인 경우 부가 정보 표시
#                 if msg["role"] == "assistant":
#                     # 검색 상태 (필터링 여부) 표시 - 토스트 메시지 느낌으로 작게
#                     if msg.get("status"):
#                         st.caption(msg["status"])
                    
#                     # 근거 규정 표시
#                     if msg.get("sources"):
#                         with st.expander("📚 관련 근거 규정 확인 (원문 보기)"):
#                             for i, src in enumerate(msg["sources"]):
#                                 st.markdown(f"**[{i+1}] {src['source']} - {src['title']}**")
#                                 st.info(f"{src['content'][:300]} ... (생략)")
# # ======================================
# # TAB 1. 💬 규정 챗봇 (유연한 키워드 매칭 & 표 인식 강화)
# # ======================================
# with tab1:
#     st.subheader("💬 규정 전문 챗봇")
#     st.caption("💡 팁: 규정 이름을 포함하면 정확도가 비약적으로 상승합니다.")
    
#     # [1] 상단 고정 입력창
#     with st.form(key="chat_form", clear_on_submit=True):
#         col1, col2 = st.columns([8, 1])
#         with col1:
#             user_input = st.text_input(
#                 "질문 입력", 
#                 placeholder="예: 위험도평가 절차는 어떻게 돼?", 
#                 label_visibility="collapsed"
#             )
#         with col2:
#             submit_btn = st.form_submit_button("질문하기", use_container_width=True)

#     # [2] 질문 처리 로직
#     if submit_btn and user_input:
#         if vectorstore is None:
#             st.error("⚠️ 규정 데이터베이스가 로드되지 않았습니다.")
#         else:
#             with st.spinner("규정 정밀 분석 및 답변 생성 중..."):
#                 # ------------------------------------------------------------------
#                 # [핵심 로직 1] 유연한 스마트 검색 필터링 (Flexible Smart Filtering)
#                 # ------------------------------------------------------------------
#                 try:
#                     all_data = vectorstore.get()
#                     unique_sources = list(set([m['source'] for m in all_data['metadatas']]))
#                 except:
#                     unique_sources = []

#                 search_filter = None
#                 target_source_name = "전체 규정"
                
#                 # 파일명 분석 및 유연한 매칭
#                 # 예: 파일명이 "02.보수 및 복리후생규정.pdf"일 때 -> "보수"라는 단어만 질문에 있어도 매칭 성공시킴
#                 for source in unique_sources:
#                     base_name = os.path.basename(source) # 예: 02_보수규정_v1.pdf
#                     clean_name = os.path.splitext(base_name)[0] # 예: 02_보수규정_v1
                    
#                     # 파일명을 특수문자 기준으로 쪼개서 키워드 추출 (예: ['02', '보수규정', 'v1'])
#                     # 더 세밀하게: 2글자 이상인 한글 단어를 추출하는 것이 좋음
#                     keywords = re.split(r'[_\s\.\-\(\)\[\]]+', clean_name)
                    
#                     match_found = False
#                     for kw in keywords:
#                         # 키워드가 2글자 이상이고, 사용자 질문에 포함되어 있다면 필터 적용
#                         # 예: kw="보수" -> user_input="보수규정에서..." (매칭 성공)
#                         if len(kw) >= 2 and kw in user_input:
#                             search_filter = {"source": source}
#                             target_source_name = base_name
#                             match_found = True
#                             break
                    
#                     if match_found:
#                         break # 하나라도 매칭되면 중단 (우선순위 로직이 필요하면 수정 가능)
                
#                 # 3. 검색 수행
#                 if search_filter:
#                     # 필터링 성공 시: 해당 문서 집중 검색
#                     retriever = vectorstore.as_retriever(
#                         search_kwargs={"k": 8, "filter": search_filter} # k를 조금 더 늘림
#                     )
#                     status_msg = f"🎯 **'{target_source_name}'** 문서 내에서 집중 검색했습니다."
#                 else:
#                     # 필터링 실패 시: 전체 검색 (k=10)
#                     retriever = vectorstore.as_retriever(
#                         search_kwargs={"k": 10}
#                     )
#                     status_msg = "🔍 전체 규정 문서에서 검색했습니다."

#                 # 4. 문서 조회 및 중복 제거
#                 retrieved_docs = retriever.invoke(user_input)
                
#                 seen_content = set()
#                 final_docs = []
#                 for d in retrieved_docs:
#                     # 내용 중복 제거 (정확히 같은 청크가 여러 번 잡힐 때)
#                     if d.page_content not in seen_content:
#                         final_docs.append(d)
#                         seen_content.add(d.page_content)
                
#                 context_text = "\n\n".join([d.page_content for d in final_docs[:6]])

#                 # ------------------------------------------------------------------
#                 # [핵심 로직 2] 프롬프트 엔지니어링 강화 (표/수치 강조)
#                 # ------------------------------------------------------------------
#                 if not context_text:
#                     response_text = "관련된 규정 내용을 찾을 수 없습니다."
#                 else:
#                     prompt_template = f"""
#                     [System Instruction]
#                     당신은 사내 규정 전문가입니다. 아래 [제공된 규정]을 바탕으로 질문에 답변하세요.
                    
#                     **절대 규칙**:
#                     1. 답변은 반드시 한국어로만 작성하십시오.
#                     2. 질문에 특정 규정(예: 보수규정)이 언급되었다면, 해당 규정의 내용만 사용하세요. (다른 규정 내용은 무시할 것)
#                     3. **'지급율(%)', '인원배분(%)', '등급(S,A,B...)'** 같은 수치 데이터는 텍스트로 풀지 말고, **반드시 마크다운 표(Table)**로 작성하세요.
#                     4. '평가(Assessment)'와 '평가급(Payment/Bonus)'을 엄격히 구분하세요. 질문이 돈(지급)에 관한 것이면 '보수/복리후생' 관련 표를 찾으세요.
#                     5. 문맥에 맞는 정답이 [제공된 규정]에 없으면 솔직하게 모른다고 답하세요.
#                     6. 질문에 언급된 규정 내 단어를 참조해(예: 보수, 인사, 안전) 최우선으로 인용하세요.
#                     7. 불확실한 내용은 지어내지 말고 "규정에 명시되어 있지 않습니다"라고 답하세요.
#                     8. 가능한 경우 조문 번호와 함께 설명하십시오.
#                     9. 다시 한번 확인하지만 답변은 무조건 한국어로만 작성해줘!

#                     [제공된 규정]:
#                     {context_text}

#                     [질문]:
#                     {user_input}

#                     [답변]:
#                     """
                    
#                     try:
#                         response_text = llm.invoke(prompt_template).content
#                     except Exception as e:
#                         response_text = f"오류가 발생했습니다: {e}"

#                 # ------------------------------------------------------------------
#                 # [UI 처리] 메시지 및 출처 정제
#                 # ------------------------------------------------------------------
#                 st.session_state.messages.append({"role": "user", "content": user_input})
                
#                 formatted_sources = []
#                 seen_titles = set()
                
#                 for doc in final_docs: 
#                     source_file = os.path.basename(doc.metadata.get("source", "파일 정보 없음"))
#                     raw_title = doc.metadata.get("Article_Title", "조항 정보 없음")
                    
#                     match = re.match(r"(제\s*\d+\s*조(?:의\d+)?(?:\([^)]*\))?)", raw_title)
#                     if match:
#                         clean_title = match.group(1)
#                     else:
#                         clean_title = raw_title[:30] + "..." if len(raw_title) > 30 else raw_title

#                     unique_key = (source_file, clean_title)
#                     if unique_key not in seen_titles:
#                         formatted_sources.append({
#                             "source": source_file,
#                             "title": clean_title,
#                             "content": doc.page_content
#                         })
#                         seen_titles.add(unique_key)

#                 st.session_state.messages.append({
#                     "role": "assistant", 
#                     "content": response_text,
#                     "sources": formatted_sources,
#                     "status": status_msg
#                 })

#     # [3] 대화 내용 출력 (스택 구조: 최신 대화가 상단, 내부는 질문->답변 순)
#     st.divider()
    
#     # 1. 메시지를 대화 쌍(질문-답변)으로 그룹화
#     conversations = []
#     current_group = []

#     # 전체 메시지를 순회하며 [User, Assistant] 단위로 묶음
#     for msg in st.session_state.messages:
#         if msg["role"] == "user":
#             # 새로운 질문이 시작되면, 이전 그룹(있다면)을 저장하고 초기화
#             if current_group:
#                 conversations.append(current_group)
#             current_group = [msg]
#         else:
#             # AI 답변(assistant)은 현재 질문 그룹에 포함
#             current_group.append(msg)

#     # 마지막 남은 그룹 저장
#     if current_group:
#         conversations.append(current_group)

#     # 2. 그룹 단위로 역순(최신순) 정렬하여 출력
#     # (그룹 자체는 최신순으로 나오지만, 그룹 내부의 for msg in group은 정순(질문->답변)으로 출력됨)
#     if conversations:
#         for group in reversed(conversations):
#             with st.container(): # 그룹별 컨테이너 (시각적 분리)
#                 for msg in group:
#                     with st.chat_message(msg["role"]):
#                         st.write(msg["content"])
                        
#                         # AI 답변일 경우 부가 정보(출처, 상태 등) 표시
#                         if msg["role"] == "assistant":
#                             if msg.get("status"):
#                                 st.caption(msg["status"])
                            
#                             if msg.get("sources"):
#                                 with st.expander("📚 관련 근거 규정 확인 (원문 보기)"):
#                                     for i, src in enumerate(msg["sources"]):
#                                         st.markdown(f"**[{i+1}] {src['source']} - {src['title']}**")
#                                         st.info(f"{src['content'][:300]} ... (생략)")
                
#                 # 대화 세트 간 구분선 (가독성 향상)
#                 st.divider()

# ======================================
# TAB 1. 💬 규정 챗봇 (개선된 버전)
# ======================================
with tab1:
    st.subheader("💬 규정 전문 챗봇")
    st.caption("💡 팁: '보수규정에서 평가급 지급율은?' 처럼 규정 이름을 포함하면 정확도가 올라갑니다.")
    
    # [1] 상단 고정 입력창
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([8, 1])
        with col1:
            user_input = st.text_input(
                "질문 입력", 
                placeholder="예: 위험도평가 절차는 어떻게 돼?", 
                label_visibility="collapsed"
            )
        with col2:
            submit_btn = st.form_submit_button("질문하기", use_container_width=True)

    # [2] 질문 처리 로직
    if submit_btn and user_input:
        if vectorstore is None:
            st.error("⚠️ 규정 데이터베이스가 로드되지 않았습니다.")
        else:
            with st.spinner("규정 정밀 분석 및 답변 생성 중..."):
                
                # --- 1. 스마트 필터링 로직 ---
                search_kwargs = {"k": 6} # 기본 검색 설정
                status_msg = "🔍 전체 규정 문서에서 검색했습니다."
                
                try:
                    # DB에 있는 소스 목록 가져오기 (가벼운 메타데이터 조회)
                    # 주의: 데이터가 많으면 이 부분이 느려질 수 있으므로, 실제 운영시엔 캐싱 권장
                    all_data = vectorstore.get() 
                    unique_sources = list(set([m['source'] for m in all_data['metadatas'] if m]))
                except:
                    unique_sources = []

                # 사용자 입력에서 키워드 추출 및 필터링
                target_source_name = None
                for source in unique_sources:
                    base_name = os.path.basename(source)
                    clean_name = os.path.splitext(base_name)[0]
                    
                    # 파일명을 토큰화 (예: '02_보수규정' -> ['02', '보수규정'])
                    keywords = re.split(r'[_\s\.\-\(\)\[\]]+', clean_name)
                    
                    for kw in keywords:
                        # 2글자 이상이고 질문에 포함된 경우 필터 적용
                        if len(kw) >= 2 and kw in user_input:
                            search_kwargs["filter"] = {"source": source}
                            target_source_name = base_name
                            break
                    if target_source_name: break

                if target_source_name:
                    status_msg = f"🎯 **'{target_source_name}'** 문서 내에서 집중 검색했습니다."
                
                # --- 2. 검색 수행 (MMR 방식 도입) ---
                # MMR: 유사도뿐만 아니라 문서 간 다양성도 고려하여 중복된 내용(같은 조항의 반복)을 줄임
                retriever = vectorstore.as_retriever(
                    search_type="mmr", 
                    search_kwargs={**search_kwargs, "fetch_k": 20, "lambda_mult": 0.7} 
                    # fetch_k: 후보군 20개, lambda_mult: 0.7 (1에 가까우면 유사도 중심, 0에 가까우면 다양성 중심)
                )
                
                retrieved_docs = retriever.invoke(user_input)
                
                # --- 3. 문서 정제 및 컨텍스트 생성 ---
                final_docs = []
                seen_content = set()
                
                for d in retrieved_docs:
                    # 내용 중복 제거
                    if d.page_content not in seen_content:
                        # 쓰레기 데이터(파이프 라인 등)가 혹시 남아있다면 제외
                        if "|||" in d.page_content or len(d.page_content.strip()) < 10:
                            continue
                        final_docs.append(d)
                        seen_content.add(d.page_content)
                
                context_text = "\n\n".join([d.page_content for d in final_docs])

                # --- 4. 프롬프트 및 LLM 호출 ---
                if not context_text:
                    response_text = "죄송합니다. 관련된 규정 내용을 찾을 수 없습니다. (검색된 문서 없음)"
                    final_docs = [] # 소스 없음
                else:
                    prompt_template = f"""
                    [System Instruction]
                    당신은 사내 규정 전문가입니다. 아래 [Context]를 바탕으로 질문에 답변하세요.
                    
                    **답변 작성 원칙**:
                    1. **근거 중심**: 상상하지 말고 반드시 [Context]에 있는 내용으로만 답하세요. 
                    2. **표/수치 유지**: 등급표, 지급율 등은 마크다운 표(Table)로 깔끔하게 정리하세요.
                    3. **조항 명시**: 가능하다면 "제OO조에 따르면..." 형태로 출처를 밝히세요.
                    4. **모르면 모른다고 하기**: 문맥에 답이 없으면 "규정에 해당 내용이 없습니다"라고 답하세요.
                    5. **언어**: 한국어로 정중하게 답변하세요.

                    [Context]:
                    {context_text}

                    [Question]:
                    {user_input}

                    [Answer]:
                    """
                    try:
                        response_text = llm.invoke(prompt_template).content
                    except Exception as e:
                        response_text = f"AI 응답 생성 중 오류 발생: {e}"

                # --- 5. UI 업데이트 (메시지 저장) ---
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # 소스 정제 (제목 깔끔하게)
                formatted_sources = []
                seen_titles = set()
                
                for doc in final_docs:
                    src_file = os.path.basename(doc.metadata.get("source", "파일"))
                    raw_title = doc.metadata.get("Article_Title", "본문")
                    
                    # 제목 정제 Regex
                    match = re.match(r"(제\s*\d+\s*조(?:의\d+)?(?:\([^)]*\))?)", raw_title)
                    clean_title = match.group(1) if match else raw_title[:30]
                    
                    key = (src_file, clean_title)
                    if key not in seen_titles:
                        formatted_sources.append({
                            "source": src_file,
                            "title": clean_title,
                            "content": doc.page_content
                        })
                        seen_titles.add(key)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": formatted_sources,
                    "status": status_msg
                })
                
    # [3] 대화 내용 출력 (그룹화 + 최신순 정렬)
    st.divider()
    
    conversations = []
    current_group = []

    # 메시지 그룹화
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            if current_group:
                conversations.append(current_group)
            current_group = [msg]
        else:
            current_group.append(msg)
    if current_group:
        conversations.append(current_group)

    # 출력 loop
    if conversations:
        for group in reversed(conversations):
            with st.container():
                for msg in group:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])
                        
                        if msg["role"] == "assistant":
                            if msg.get("status"):
                                st.caption(msg["status"])
                            if msg.get("sources"):
                                with st.expander("📚 관련 근거 규정 확인"):
                                    for i, src in enumerate(msg["sources"]):
                                        st.markdown(f"**[{i+1}] {src['source']} - {src['title']}**")
                                        # 본문 미리보기에서 파이프 문자 등이 보이면 필터링해서 보여줌
                                        display_content = src['content'].replace("|", " ").replace("\n", " ")[:200]
                                        st.caption(f"{display_content}...")
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
                            집중적인 관리가 필요합니다.
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

# ======================================
# TAB 3. 🛡️ 통합 위험 분석 (Integrated Risk Analysis) - Ver 1.2
# ======================================
with tab3:
    st.subheader("🛡️ 통합 위험도 평가 (Risk Assessment)")
    
    # -------------------------------------------------------
    # [Data Load & Preprocessing]
    # -------------------------------------------------------
    if os.path.exists(FILE_PATH):
        df_risk = pd.read_pickle(FILE_PATH)
        
        # 데이터 전처리
        col_map = {"cause": "부원인"}
        if "부원인" not in df_risk.columns:
            df_risk["부원인"] = df_risk["cause"] if "cause" in df_risk.columns else "정보없음"
        
        # 라벨 클리닝
        df_risk["부원인"] = df_risk["부원인"].apply(lambda x: re.sub(r'\[.*?\]', '', str(x)).strip())

        # -------------------------------------------------------
        # [Step 1] 심각도(Severity) 데이터 준비 (설정 UI는 하단/숨김 처리)
        # -------------------------------------------------------
        unique_causes = df_risk["부원인"].unique()

        # 기본 심각도 매핑 함수
        def get_default_severity(cause):
            cause = str(cause)
            if any(x in cause for x in ['사망', '탈선', '충돌', '화재', '폭발', '자살']): return 5
            if any(x in cause for x in ['추락', '감전', '끼임', '협착', '절단']): return 4
            if any(x in cause for x in ['골절', '베임', '화상', '누수']): return 3
            if any(x in cause for x in ['넘어짐', '전도', '부딪힘', '음주', '부주의']): return 2
            return 1

        # 데이터프레임 생성
        severity_data = []
        for c in unique_causes:
            severity_data.append({"사고유형": c, "심각도(1-5)": get_default_severity(c)})
        
        df_severity_default = pd.DataFrame(severity_data).sort_values(by="심각도(1-5)", ascending=False)

        # -------------------------------------------------------
        # [Step 1-1] 설정창: 시각적 간소화를 위해 Expander 사용
        # -------------------------------------------------------
        # 챗봇의 핵심(분석 결과)을 먼저 보여주기 위해 설정을 접어둠
        with st.expander("⚙️ [설정] 사고 유형별 심각도 기준 변경하기 (클릭)", expanded=False):
            st.caption("아래 표에서 사고 유형별 심각도(1~5)를 수정하면 매트릭스에 즉시 반영됩니다.")
            edited_severity_df = st.data_editor(
                df_severity_default,
                column_config={
                    "심각도(1-5)": st.column_config.NumberColumn(
                        "심각도", help="1:경미 ~ 5:치명", min_value=1, max_value=5, step=1
                    )
                },
                use_container_width=True,
                hide_index=True,
                key="severity_editor" # 키를 지정하여 상태 유지
            )

        # -------------------------------------------------------
        # [Step 2] 위험도(Risk) 계산 로직
        # -------------------------------------------------------
        # 1. 빈도(Frequency) 집계
        df_freq = df_risk["부원인"].value_counts().reset_index()
        df_freq.columns = ["사고유형", "발생건수"]

        # 2. 심각도(Severity) 병합 (편집된 데이터 사용)
        df_calc = pd.merge(df_freq, edited_severity_df, on="사고유형", how="left")

        # 3. 빈도 등급(Freq Level) 자동 산출 (1~5등급)
        max_count = df_calc["발생건수"].max()
        def get_freq_level(cnt, max_val):
            if max_val == 0: return 1
            ratio = cnt / max_val
            if ratio >= 0.8: return 5
            elif ratio >= 0.6: return 4
            elif ratio >= 0.4: return 3
            elif ratio >= 0.2: return 2
            else: return 1

        df_calc["빈도등급(1-5)"] = df_calc["발생건수"].apply(lambda x: get_freq_level(x, max_count))
        df_calc["위험점수"] = df_calc["빈도등급(1-5)"] * df_calc["심각도(1-5)"]

        # 4. Risk Zone 판정
        def get_risk_zone(score):
            if score >= 15: return "🔴 High"
            elif score >= 8: return "🟡 Medium"
            else: return "🟢 Low"

        df_calc["위험등급"] = df_calc["위험점수"].apply(get_risk_zone)

        # Tab 4 연동을 위한 세션 저장
        top_risks = df_calc.sort_values(by=["위험점수", "발생건수"], ascending=[False, False])
        st.session_state['priority_risks'] = top_risks

        st.divider()

        # -------------------------------------------------------
        # [Step 3] 5x5 Risk Matrix 시각화 (개선된 버전)
        # -------------------------------------------------------
        c_left, c_right = st.columns([1.6, 1])

        with c_left:
            st.markdown("##### 📊 위험성 평가 매트릭스 (Risk Matrix)")
            
            # 1. 그리드 데이터 생성
            grid_data = []
            for s in range(1, 6): 
                for f in range(1, 6):
                    score = s * f
                    if score >= 15: 
                        risk_grade, color, text_color = "High", "#FF4B4B", "white"
                    elif score >= 8: 
                        risk_grade, color, text_color = "Medium", "#FFAA00", "black"
                    else: 
                        risk_grade, color, text_color = "Low", "#00CC96", "black"
                    
                    grid_data.append({
                        "심각도_X": s, "빈도등급_Y": f,
                        "위험점수": score,
                        "라벨": f"{risk_grade}\n{score}", # 줄바꿈 적용
                        "배경색": color,
                        "글자색": text_color
                    })
            df_grid_base = pd.DataFrame(grid_data)

            # 2. 데이터 집계
            df_agg = df_calc.groupby(["심각도(1-5)", "빈도등급(1-5)"]).agg(
                총건수=("발생건수", "sum"),
                사고유형_리스트=("사고유형", lambda x: ", ".join(x.unique()))
            ).reset_index()
            
            # 3. 병합
            df_matrix_final = pd.merge(
                df_grid_base, df_agg, 
                left_on=["심각도_X", "빈도등급_Y"], 
                right_on=["심각도(1-5)", "빈도등급(1-5)"], 
                how="left"
            ).fillna({"총건수": 0, "사고유형_리스트": "-"})

            # 4. Altair 차트 (폰트 크기 및 배치 수정)
            base = alt.Chart(df_matrix_final).encode(
                x=alt.X("심각도_X:O", title="중대성 (Severity) ➡️", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("빈도등급_Y:O", title="가능성 (Frequency) ⬆️", sort="descending")
            ).properties(height=450) # 높이 확보

            # Layer 1: 배경 히트맵
            heatmap = base.mark_rect().encode(
                color=alt.Color("배경색:N", scale=None),
                tooltip=["심각도_X", "빈도등급_Y", "라벨", "총건수", "사고유형_리스트"]
            )

            # Layer 2: 위험 점수/등급 텍스트 (폰트 크게!)
            text_score = base.mark_text(baseline="middle").encode(
                text=alt.Text("라벨"),
                color=alt.Color("글자색:N", scale=None),
                size=alt.value(15),  # [수정] 폰트 크기 대폭 확대 (24px)
                opacity=alt.value(0.6) # 배경 글씨처럼 은은하게
            )

            # Layer 3: 실제 발생 건수 (강조)
            text_count = base.transform_filter(
                alt.datum.총건수 > 0
            ).mark_text(dy=25, fontWeight="bold").encode( # dy: 위치를 아래로
                text=alt.Text("총건수", format="d"), # 숫자만 표시
                color=alt.value("blue"),
                size=alt.value(16)   # [수정] 건수 폰트 크기 확대 (16px)
            )

            final_chart = (heatmap + text_score + text_count).configure_axis(
                labelFontSize=12,
                titleFontSize=14
            )
            
            st.altair_chart(final_chart, use_container_width=True)

        with c_right:
            st.markdown("##### 🚨 위험 우선순위 (Top Risks)")
            
            # 최우선 위험 항목 카드뷰
            if not top_risks.empty:
                worst = top_risks.iloc[0]
                st.error(f"**1위: {worst['사고유형']}**")
                
                col_kpi1, col_kpi2 = st.columns(2)
                col_kpi1.metric("위험 점수", f"{worst['위험점수']}점", help="심각도 x 빈도")
                col_kpi2.metric("발생 건수", f"{worst['발생건수']}건")
                
                st.progress(min(worst['위험점수']/25.0, 1.0), text="위험도 수준")

            st.write("") # 여백
            
            # 리스트 뷰
            st.dataframe(
                top_risks[["사고유형", "위험등급", "위험점수", "발생건수"]],
                hide_index=True,
                use_container_width=True,
                height=300
            )

        # AI Insight
        st.info("💡 **AI 분석:** 붉은색(High) 영역에 위치한 사고 유형은 '즉시 개선'이 필요한 항목입니다. 우측 리스트 상위 항목을 중심으로 안전 대책을 수립하세요.")

    else:
        st.warning("데이터 파일이 없습니다.")

# ======================================
# TAB 4. 🤖 위험 판단 및 조치 (AI Action Plan) - Ver 1.2 (Debug Mode)
# ======================================
from langchain.schema import HumanMessage, SystemMessage

with tab4:
    st.subheader("🤖 AI 위험 대응 솔루션")
    st.caption("위험성 평가 결과(Tab 3)를 기반으로 사규 데이터(Tab 1)를 검색하여 맞춤형 조치를 제안합니다.")

    # 1. Tab 3 분석 결과 불러오기
    if 'priority_risks' in st.session_state and not st.session_state['priority_risks'].empty:
        priority_df = st.session_state['priority_risks']
        risk_options = priority_df['사고유형'].tolist()

        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.markdown("##### 🔍 분석 대상 선택")
            selected_risk = st.selectbox("조치 방안을 확인할 위험 요인:", risk_options)
            
            # 선택된 항목 정보
            target_row = priority_df[priority_df['사고유형'] == selected_risk].iloc[0]
            st.info(f"""
            **{selected_risk}**
            - 등급: {target_row['위험등급']}
            - 점수: {target_row['위험점수']} (건수: {target_row['발생건수']})
            """)
            
            generate_btn = st.button("🧬 조치방안 생성 (Real-time)", type="primary", use_container_width=True)

        with c2:
            st.markdown(f"##### 📋 '{selected_risk}' 안전 조치 가이드")
            
            if generate_btn:
                # [필수 체크] 벡터 DB와 LLM 로드 여부 확인
                if 'vectorstore' not in st.session_state:
                    st.error("⚠️ 벡터 DB가 없습니다. Tab 1에서 문서를 먼저 임베딩해주세요.")
                # elif 'llm' not in st.session_state: # (혹시 llm을 세션에 넣으셨다면 체크)
                #     st.error("⚠️ LLM 모델이 로드되지 않았습니다.")
                else:
                    with st.spinner(f"규정 DB 검색 및 분석 중..."):
                        try:
                            # ---------------------------------------------------
                            # [1] 검색 쿼리 최적화 (단순 키워드 -> 문장형)
                            # ---------------------------------------------------
                            # 키워드가 너무 짧으면 검색이 잘 안되므로 꼬리말을 붙여줍니다.
                            search_query = f"{selected_risk} 사고 예방 작업 절차 안전 수칙 금지 사항"
                            
                            # ---------------------------------------------------
                            # [2] 벡터 DB 검색 (Retriever)
                            # ---------------------------------------------------
                            # k=4: 가장 유사한 문서 조각 4개를 가져옴
                            retriever = st.session_state['vectorstore'].as_retriever(search_kwargs={"k": 4})
                            docs = retriever.get_relevant_documents(search_query)
                            
                            # 검색된 문서 내용 합치기
                            context_text = "\n\n".join([doc.page_content for doc in docs])
                            
                            # ---------------------------------------------------
                            # [3] 디버깅용: 검색된 문서가 무엇인지 확인 (중요!)
                            # ---------------------------------------------------
                            with st.expander("🔍 [디버깅] 벡터 DB가 찾아낸 원문 내용 보기", expanded=False):
                                if not docs:
                                    st.warning("관련 문서를 찾지 못했습니다. 검색어가 매뉴얼에 없을 수 있습니다.")
                                for i, doc in enumerate(docs):
                                    st.text(f"[문서 {i+1}] Source: {doc.metadata.get('source', 'unknown')}")
                                    st.caption(doc.page_content[:300] + "...") # 앞부분만 미리보기

                            # ---------------------------------------------------
                            # [4] LLM 생성 요청 (실제 연동)
                            # ---------------------------------------------------
                            # 문맥이 너무 없으면 솔직하게 말하도록 프롬프트 조정
                            if not context_text:
                                st.warning("관련 규정을 찾을 수 없어 일반적인 안전 지침을 생성합니다.")
                                context_text = "관련 사규 없음. 일반적인 산업 안전 기준을 적용할 것."

                            prompt = f"""
                            당신은 철도/산업 안전 전문가입니다. 
                            아래의 [검색된 사규/매뉴얼]만을 근거로 하여, 
                            위험 요인 '{selected_risk}'에 대한 구체적인 대응 매뉴얼을 작성하세요.

                            [검색된 사규/매뉴얼]
                            {context_text}

                            [작성 원칙]
                            1. 검색된 내용에 '{selected_risk}'와 직접 관련된 내용이 없다면, "관련된 내부 규정을 찾을 수 없습니다."라고 솔직히 말하고 일반적인 안전 수칙을 제안하세요.
                            2. 뜬구름 잡는 소리 대신 '작업 전', '작업 중', '비상 시' 등 구체적인 행동 요령을 나열하세요.
                            3. 출처(문서명 등)를 알 수 있다면 함께 명시하세요.
                            """
                            
                            # 실제 LLM 호출 (main.py 상단에 선언된 llm 객체 사용)
                            # (st.write_stream을 쓰면 타자기 효과 가능 - LangChain 버전에 따라 다름)
                            
                            # 방법 A: invoke 사용 (LangChain 최신)
                            # response = llm.invoke(prompt)
                            # result_text = response.content
                            
                            # 방법 B: predict 사용 (LangChain 구버전)
                            # result_text = llm.predict(prompt)
                            
                            # [가정] main.py에 'llm' 객체가 있다고 가정하고 실행
                            # 만약 llm 객체 이름이 다르면(chat_model 등) 수정 필요
                            if 'llm' in globals():
                                response = llm.invoke([HumanMessage(content=prompt)])
                                result_text = response.content
                            elif 'chat_model' in globals(): # 이름이 chat_model일 경우
                                response = chat_model.invoke([HumanMessage(content=prompt)])
                                result_text = response.content
                            else:
                                result_text = "⚠️ 코드 에러: 'llm' 또는 'chat_model' 객체가 정의되지 않았습니다. main.py를 확인해주세요."

                            st.markdown(result_text)

                        except Exception as e:
                            st.error(f"생성 오류: {e}")
                            st.info("💡 팁: Tab 1에서 벡터 DB가 정상적으로 생성되었는지 확인해보세요.")
            else:
                st.info("👈 왼쪽에서 위험 요인을 선택하고 버튼을 눌러주세요.")

    else:
        st.warning("⚠️ Tab 3(위험 분석)를 먼저 실행하여 데이터를 생성해주세요.")
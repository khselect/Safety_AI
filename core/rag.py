from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain.schema import HumanMessage, SystemMessage
import pandas as pd
import re
import os

# ------------------------------------------------------------------
# [핵심 개선 1] 한국어 전용 SystemMessage
# - user 프롬프트의 지시보다 system 레벨 지시를 모델이 더 강하게 따름
# - ChatOllama는 SystemMessage를 별도 system 파라미터로 처리함
# ------------------------------------------------------------------
KOREAN_SYSTEM_MESSAGE = SystemMessage(content=(
    "You are a Korean railway safety regulation expert. "
    "You MUST answer ONLY in Korean (한국어). "
    "Never use English, Chinese, or any other language in your response. "
    "If you start writing in any language other than Korean, stop immediately and restart in Korean."
))

# ------------------------------------------------------------------
# [핵심 개선 2] 한국어 감지 + 자동 재번역 후처리
# - langdetect 없이도 동작하는 간단한 한국어 비율 체크
# ------------------------------------------------------------------
def is_korean_response(text: str) -> bool:
    """한글 문자 비율이 10% 미만이면 한국어 답변이 아닌 것으로 판단"""
    if not text or len(text.strip()) == 0:
        return False
    korean_chars = len(re.findall(r'[\uAC00-\uD7A3]', text))
    total_chars = len(re.sub(r'\s', '', text))  # 공백 제외
    if total_chars == 0:
        return False
    ratio = korean_chars / total_chars
    return ratio >= 0.1  # 한글 비율 10% 이상이면 한국어 답변으로 판단


def translate_to_korean(text: str, llm) -> str:
    """영어로 된 답변을 한국어로 재번역 요청"""
    translate_prompt = f"""다음 텍스트를 한국어로 번역해주세요. 번역문만 출력하고 다른 설명은 하지 마세요.

번역할 텍스트:
{text}

한국어 번역:"""
    try:
        response = llm.invoke([
            KOREAN_SYSTEM_MESSAGE,
            HumanMessage(content=translate_prompt)
        ])
        return response.content
    except Exception:
        return text  # 번역 실패 시 원문 반환


# ------------------------------------------------------------------
# [핵심 개선 3] 모델별 template prefix 분기
# ------------------------------------------------------------------
def get_template(model_name: str) -> str:
    """qwen3 계열만 /no_think 적용"""
    prefix = "/no_think\n" if "qwen3" in model_name.lower() else ""
    return f"""{prefix}
[규정 내용]
{{context}}

[질문]
{{question}}

[답변 가이드라인]
1. 반드시 제공된 [규정 내용]에 명시된 내용만 답변하십시오.
2. 질문 키워드가 포함된 조문 제목(예: 제26조 청원휴가)을 최우선으로 설명하세요.
3. 조문 번호를 다른 조문과 절대 혼동하지 마세요.
4. 답변은 "제NN조(조문명)에 따르면," 형식으로 근거를 명시하며 시작하세요.
5. 규정에 없는 내용은 "해당 규정에는 관련 내용이 명시되어 있지 않습니다"라고만 답하세요.
6. 일반적인 노동법 상식을 섞지 말고 제공된 규정 텍스트로만 답변하세요.
반드시 한국어로만 답변하세요."""


# ------------------------------------------------------------------
# 메인 RAG 함수
# ------------------------------------------------------------------
def query_regulation(query, vectorstore, llm, model_name="", shared_dir=""):
    SHARED_DIR = shared_dir  # main.py에서 주입받은 경로 사용

    # ① 전문가 피드백(Applied) Fast-Track
    feedback_file = os.path.join(SHARED_DIR, "feedback_log.csv")
    if os.path.exists(feedback_file):
        try:
            fb_df = pd.read_csv(feedback_file)
            match = fb_df[
                (fb_df['Status'] == 'Applied') &
                (fb_df['Question'].str.contains(query[:5], na=False))
            ]
            if not match.empty:
                return f"✅ **[전문가 검증 답변]**\n\n{match.iloc[-1]['User_Correction']}", []
        except Exception:
            pass

    # ② 하이브리드 검색 (BM25 0.8 + Vector 0.2)
    all_docs_data = vectorstore.get()
    all_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_docs_data['documents'], all_docs_data['metadatas'])
    ]

    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 3

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.8, 0.2]
    )

    docs = ensemble_retriever.invoke(query)

    # ③ 제목 기반 재정렬
    query_keywords = query.split()
    docs = sorted(
        docs,
        key=lambda d: any(kw in d.metadata.get('Article_Title', '') for kw in query_keywords),
        reverse=True
    )

    # ④ 컨텍스트 구성
    context_text = "\n\n".join([
        f"### {doc.metadata.get('Article_Title', '조문 정보 없음')}\n{doc.page_content}"
        for doc in docs
    ])

    # ⑤ 프롬프트 구성
    template = get_template(model_name)
    prompt_text = template.format(context=context_text, question=query)

    # ⑥ LLM 호출 — [개선 1] SystemMessage를 별도로 분리하여 전달
    response = llm.invoke([
        KOREAN_SYSTEM_MESSAGE,          # system 레벨 한국어 강제
        HumanMessage(content=prompt_text)
    ])
    answer = response.content

    # ⑦ [개선 2] 후처리: 한국어 감지 → 영어면 자동 재번역
    if not is_korean_response(answer):
        print(f"[경고] 비한국어 답변 감지 → 재번역 시도 (모델: {model_name})")
        answer = translate_to_korean(answer, llm)

    return answer, docs

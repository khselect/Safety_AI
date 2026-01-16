# core/rag.py

from langchain_chroma import Chroma
from langchain.schema import HumanMessage  # [필수] 에러 방지용 임포트
from langchain_community.embeddings import HuggingFaceEmbeddings # 패키지명 최신화 권장
from langchain.prompts import PromptTemplate

# ======================================
# Embedding model (전역 1회 생성)
# ======================================
def get_embeddings_rag():
    # CPU 환경에 최적화된 한국어 임베딩 모델
    return HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

# ======================================
# RAG QA 함수
# ======================================
def ask_regulation(question, llm, vectorstore, chat_history=[]):
    
    # 1. 유사 문서 검색
    docs = vectorstore.similarity_search(question, k=5)

    if not docs:
        return "해당 질문에 대한 규정을 찾을 수 없습니다.", []

    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    # 2. 대화 기록 포맷팅
    history_text = ""
    if chat_history:
        history_text = "[이전 대화 내역]\n"
        for role, msg in chat_history[-6:]: 
             history_text += f"- {role}: {msg}\n"

    # 3. 프롬프트 작성
    template = f"""
    당신은 철도안전 규정 전문가입니다. 이전 대화 내역과 제공된 규정을 바탕으로 답변하십시오.

    [답변 규칙]
    1. 반드시 한국어로 답변하십시오.
    2. 제공된 '규정 내용'을 최우선 근거로 삼으십시오.
    3. '이전 대화 내역'을 참고하여 문맥에 맞는 답변을 하십시오.
    4. 조항 번호가 있다면 반드시 언급하십시오.

    {history_text}

    [규정 내용]
    {context_text}

    [질문]
    {question}

    [답변]
    """
    
    # [개선2] LangChain 구버전/신버전 호환성을 위한 분기 처리
    try:
        if hasattr(llm, 'invoke'):
            # 최신 LangChain (invoke 사용)
            response = llm.invoke([HumanMessage(content=template)])
            return response.content, docs
        else:
            # 구버전 LangChain (predict 사용)
            return llm.predict(template), docs
    except Exception as e:
        return f"오류 발생: {e}", []
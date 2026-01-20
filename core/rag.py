from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
import pandas as pd
import os

def query_regulation(query, vectorstore, llm):
    # [설정] 피드백 파일 경로
    FEEDBACK_FILE = os.path.join(SHARED_DIR, "feedback_log.csv")

    # 1. 전문가 정책(Policy) 우선 필터링 - CSV 직접 탐색
    # 강화학습의 'Policy Update'를 가장 확실하게 반영하는 방법입니다.
    if os.path.exists(FEEDBACK_FILE):
        try:
            fb_df = pd.read_csv(FEEDBACK_FILE)
            # 'Applied' 상태이며 질문의 핵심 키워드가 포함된 경우 (가장 최근 데이터 우선)
            keywords = [k for k in query.split() if len(k) > 1]
            match = fb_df[
                (fb_df['Status'] == 'Applied') & 
                (fb_df['Question'].apply(lambda x: all(k in str(x) for k in keywords[:2])))
            ]
            if not match.empty:
                best_answer = match.iloc[-1]['User_Correction']
                return f"✅ **[전문가 검증 답변]**\n\n{best_answer}", []
        except:
            pass

    # 2. 하이브리드 검색 (BM25 + Vector) - [문제 1/3 해결]
    # 키워드 검색(BM25)은 '시행시기'와 '평가'를 명확히 구분합니다.
    all_docs_data = vectorstore.get()
    all_docs = [
        Document(page_content=text, metadata=meta) 
        for text, meta in zip(all_docs_data['documents'], all_docs_data['metadatas'])
    ]
    
    # BM25(키워드) 가중치 강화
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 3
    
    # 벡터(의미) 검색
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 앙상블 검색기 (키워드 0.7 : 벡터 0.3) -> '시행시기' 단어 적중률 향상
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.7, 0.3]
    )
    
    docs = ensemble_retriever.invoke(query)

    # 3. 프롬프트 엔지니어링 (강화학습 보상 반영)
    context_text = ""
    for doc in docs:
        # 전문가 피드백 데이터에 가중치(Reward) 부여 문구 추가
        prefix = "⭐[최우선 참고: 전문가 검증 지식]" if doc.metadata.get('type') == 'feedback' else ""
        context_text += f"\n{prefix}\n[{doc.metadata.get('Article_Title', '규정')}] {doc.page_content}\n"

    template = """당신은 철도안전법 전문가입니다. 
    제공된 [규정 내용]을 바탕으로 질문에 답하세요. 
    특히 질문의 핵심(예: '시행시기' 인지 '평가방법' 인지)을 조문 번호와 함께 정확히 구분하세요.
    별표나 시행규칙 내용이 있다면 이를 우선순위에 두어 답변하세요.

    [규정 내용]
    {context}

    [질문]
    {question}
    """
    
    prompt_text = template.format(context=context_text, question=query)
    response = llm.invoke([HumanMessage(content=prompt_text)])
    
    return response.content, docs
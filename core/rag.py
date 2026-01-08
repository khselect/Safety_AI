# core/rag.py

from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

# ======================================
# Embedding model (전역 1회 생성)
# ======================================
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sbert-nli"
)

# ======================================
# VectorStore Loader
# ======================================
def load_vectorstore():
    return Chroma(
        collection_name="regulations",
        persist_directory="vectorstore",
        embedding_function=embeddings
    )


# ======================================
# RAG QA
# ======================================
def ask_regulation(question, llm, vectorstore):

    docs = vectorstore.similarity_search(question, k=5)

    if not docs:
        return "해당 질문에 대한 규정을 찾을 수 없습니다.", []

    context_text = "\n\n".join([doc.page_content for doc in docs])

    template = """
    당신은 안전관리 규정 전문가입니다.

    [답변 규칙]
    1. 반드시 한국어로만 답변하십시오.
    2. 아래 제공된 규정 원문 내용만 사용하십시오.
    3. 제공된 규정 문서(context)에 없는 내용은 추론하거나 일반 지식을 사용하지 마십시오.
    4. 가능하면 조문 번호를 포함하여 설명하십시오.
    5. 규정에 명시되지 않은 경우 "해당 규정에 명시되어 있지 않습니다."라고 답하십시오.


    [규정 내용]
    {context}

    [질문]
    {question}

    [답변]
    """

    prompt_obj = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    formatted_prompt = prompt_obj.format(
        context=context_text,
        question=question
    )

    response = llm.invoke(formatted_prompt).content

    return response, docs

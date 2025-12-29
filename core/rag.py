from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from core.config import PERSIST_DIRECTORY
from core.llm import get_embeddings

def load_retrievers():
    if not PERSIST_DIRECTORY:
        return None, None

    vs = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=get_embeddings()
    )

    chroma = vs.as_retriever(search_kwargs={"k": 5})

    data = vs.get()
    bm25 = None
    if data.get("documents"):
        docs = [
            Document(page_content=t, metadata=m)
            for t, m in zip(data["documents"], data["metadatas"])
        ]
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = 5

    return chroma, bm25


def ask_regulation(query, llm):
    chroma, bm25 = load_retrievers()

    docs = []
    if chroma:
        docs += chroma.invoke(query)
    if bm25:
        docs += bm25.invoke(query)

    context = "\n\n".join([d.page_content for d in docs[:5]])

    prompt = f"""
    사내 규정에 근거하여 답변하세요.

    [규정]
    {context}

    [질문]
    {query}

    [답변]
    """

    return llm.invoke(prompt).content

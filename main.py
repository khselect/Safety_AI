import streamlit as st
from core.llm import get_llm
from core.rag import ask_regulation

st.set_page_config(page_title="사내 규정 챗봇", layout="centered")
st.title("💬 사내 규정 AI 챗봇")

model = "korean-llama3"
llm = get_llm(model)

query = st.chat_input("규정 관련 질문을 입력하세요")

if query:
    with st.chat_message("assistant"):
        answer = ask_regulation(query, llm)
        st.write(answer)

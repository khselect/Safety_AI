from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def get_llm(model_name: str, temperature=0):
    return ChatOllama(
        model=model_name,
        base_url="http://127.0.0.1:11434",
        temperature=temperature
    )

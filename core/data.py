import os
import tempfile
import shutil
import olefile # hwp 기초 분석용
import re
import mammoth  # docx 변환용
import markdownify # html to markdown 용
from typing import List
import docx2txt  # 또는 mammoth 사용 가능
import PyPDF2

# ChromaDB 관련 임포트
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def clean_markdown_text(text: str) -> str:
    text = text.replace("~~", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def convert_docx_to_markdown(file_path: str) -> str:
    """DOCX 파일을 텍스트(마크다운 형식)로 변환합니다."""
    try:
        text = docx2txt.process(file_path)
        return text
    except Exception as e:
        print(f"DOCX 변환 중 오류 발생: {e}")
        return ""

def extract_hwp_text(hwp_path: str) -> str:
    try:
        f = olefile.OleFileIO(hwp_path)
        encoded_text = f.openstream("PrvText").read()
        return encoded_text.decode("utf-16le")
    except Exception as e:
        return f"[HWP 오류] 변환 실패: {e}"

def convert_pdf_to_text(file_path: str) -> str:
    """PDF 파일을 텍스트로 변환합니다."""
    text = ""
    try:
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"PDF 변환 중 오류 발생: {e}")
        return ""

def convert_docx_to_markdown(docx_path: str) -> str:
    with open(docx_path, "rb") as docx_file:
        result = mammoth.convert_to_html(docx_file)
        html = result.value
    return markdownify.markdownify(html, heading_style="ATX")

def get_document_statistics(docs: List[Document]):
    """생성된 문서 조각(chunk)의 통계를 반환합니다 (관리자 페이지용)."""
    return {
        "chunk_count": len(docs),
        "total_characters": sum(len(doc.page_content) for doc in docs)
    }

def extract_hwp_text(hwp_path: str) -> str:
    try:
        f = olefile.OleFileIO(hwp_path)
        encoded_text = f.openstream("PrvText").read()
        return encoded_text.decode("utf-16le")
    except Exception as e:
        return f"[HWP 오류] 변환 실패: {e}"

def process_file_to_docs(file, source_name):
    file_ext = os.path.splitext(file.name)[1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name

    try:
        if file_ext == ".pdf":
            # pymupdf_layout 경고는 무시해도 됩니다 (단순 안내 메시지)
            md_text = pymupdf4llm.to_markdown(tmp_path)
        elif file_ext == ".docx":
            md_text = convert_docx_to_markdown(tmp_path)
        elif file_ext in [".hwp", ".hwpx"]:
            raw_text = extract_hwp_text(tmp_path)
            md_text = f"# {source_name} 본문\n\n{clean_markdown_text(raw_text)}"
        else:
            return []

        md_text = clean_markdown_text(md_text)
        
        # 헤더 보정
        md_text = re.sub(r'(^|\n)(제\s*\d+(?:의\d+)?\s*조)', r'\1# \2', md_text)
        md_text = re.sub(r'(^|\n)(\[별표\s*\d+.*?\])', r'\1# \2', md_text)
        md_text = re.sub(r'(^|\n)(\[별지\s*.*?\])', r'\1# \2', md_text)

        # 1단계 청킹
        headers_to_split_on = [("#", "Article_Title")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        header_splits = markdown_splitter.split_text(md_text)
        
        # 2단계 청킹
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        
        final_docs = []
        for doc in header_splits:
            splits = text_splitter.split_text(doc.page_content)
            for split_content in splits:
                new_doc = Document(
                    page_content=split_content,
                    metadata={
                        "source": source_name,
                        "Article_Title": doc.metadata.get("Article_Title", "일반"),
                        "file_type": file_ext
                    }
                )
                final_docs.append(new_doc)
        return final_docs
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
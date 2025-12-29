import os

PERSIST_DIRECTORY = "./chroma_db"

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

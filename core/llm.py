import os
from langchain_ollama import ChatOllama

# LLM Configuration:
# Default model: llama3.1:8b running locally via Ollama.
# Can be overridden via OLLAMA_MODEL and OLLAMA_BASE_URL environment variables.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
)

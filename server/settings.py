import os
from pathlib import Path

# Configuration
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://bge-m3-embedding:80")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
MAX_REPHRASES = int(os.getenv("MAX_REPHRASES", "3"))
EMBEDDING_DIMENSION = 1024

# RAG LLM Configuration
RAG_MODEL = os.getenv("RAG_MODEL", "")
RAG_URL = os.getenv("RAG_URL", "")
RAG_API_KEY = os.getenv("RAG_API_KEY", "")
RAG_MAX_TOKENS = int(os.getenv("RAG_MAX_TOKENS", "1024"))
RAG_TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "1.0"))
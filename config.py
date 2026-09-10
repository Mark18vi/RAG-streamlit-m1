import os

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 5

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

CHROMA_DB = "RAG_DB"
CHROMA_COLLECTION = "Sports_Document_Collection"

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

FILE_UPLOAD_DIR = "C:\\Users\\megha\\Videos\\AI engineering\\RAG streamlit m1\\documents"
import chromadb 
from chromadb.config import Settings
from config import CHUNK_SIZE, CHUNk_OVERLAP

from embeddings import EmbeddingService

from config import CHROMA_DB, CHROMA_COLLECTION

class VectorDB:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DB)
        self.collection = self.client.get_or_create_collection(name=CHROMA_COLLECTION)
        self.embedding_service = EmbeddingService()

    def add_documents(self, documents):
        for doc in documents:
            embedding = self.embedding_service.get_embedding(doc)
            self.collection.add(
                ids = [str(hash(doc))],
                documents=[doc],
                embeddings=[embedding]
            )

    def query(self, query_text, top_k=5):
        query_embedding = self.embedding_service.get_embedding(query_text)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results

    def list_collections(self):
        return self.client.list_collections()

    def delete_collection(self, collection_name):
        self.client.delete_collection(name=collection_name)

    
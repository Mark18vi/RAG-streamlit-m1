import chromadb 
from chromadb.config import Settings

from embeddings import EmbeddingService

from config import CHROMA_DB, CHROMA_COLLECTION, CHUNK_SIZE, CHUNK_OVERLAP
from graph_db import GraphDB
from llm import LLMservice
from logging_config import get_logger

logger = get_logger(__name__)

class VectorDB:
    def __init__(self):
        logger.info("Initializing ChromaDB: path=%s collection=%s", CHROMA_DB, CHROMA_COLLECTION)
        self.client = chromadb.PersistentClient(path=CHROMA_DB)
        self.collection = self.client.get_or_create_collection(name=CHROMA_COLLECTION)
        self.embedding_service = EmbeddingService()
        self.graph_db = GraphDB()
        logger.info("VectorDB initialized: existing_documents=%d", self.collection.count())

    def add_documents(self, documents, llm_service=None):
        logger.info("Document storage started: chunks=%d graph_enabled=%s", len(documents), bool(llm_service))
        vector_count = 0
        triplet_count = 0
        for index, doc in enumerate(documents, start=1):
            logger.info("Processing chunk %d/%d", index, len(documents))
            embedding = self.embedding_service.get_embedding(doc)
            self.collection.add(
                ids = [str(hash(doc))],
                documents=[doc],
                embeddings=[embedding]
            )
            vector_count += 1
            logger.info("Chunk stored in ChromaDB: chunk=%d", index)

            # If LLM service is provided, extract SPO triplets and add to Neo4j
            if llm_service:
                triplets = llm_service.extract_spo(doc)
                if triplets:
                    self.graph_db.add_triplets(triplets)
                    triplet_count += len(triplets)
                    logger.info("Chunk graph extraction stored: chunk=%d triplets=%d", index, len(triplets))
                else:
                    logger.warning("No valid SPO triplets stored for chunk=%d", index)
        logger.info("Document storage completed: vector_chunks=%d graph_triplets=%d", vector_count, triplet_count)

    def query(self, query_text, top_k=5):
        logger.info("Hybrid query started: top_k=%d query_characters=%d", top_k, len(query_text))
        query_embedding = self.embedding_service.get_embedding(query_text)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances"]
        )
        
        # Hybrid Enrichment:
        # Use LLM to extract entities from the query, then fetch their neighborhood from Neo4j
        llm_service = LLMservice()
        entities = llm_service.extract_entities(query_text)
        logger.info("Query entities extracted: count=%d", len(entities))
        
        graph_context = []
        graph_facts = []
        for entity in entities:
            neighbors = self.graph_db.query_graph(entity)
            for neighbor in neighbors:
                fact = f"{neighbor['start']} {neighbor['predicate']} {neighbor['end']}"
                graph_context.append(fact)
                graph_facts.append({
                    "entity": entity,
                    "subject": neighbor["start"],
                    "predicate": neighbor["predicate"],
                    "object": neighbor["end"],
                })
            
        # Combine vector results and graph context
        vector_docs = results['documents'][0] if results['documents'] else []
        vector_distances = results.get("distances", [[]])[0]
        combined_context = "\n".join(vector_docs)
        if graph_context:
            combined_context += "\n\nGraph Context:\n" + "\n".join(graph_context)
        logger.info(
            "Hybrid query completed: vector_results=%d graph_facts=%d context_characters=%d",
            len(vector_docs),
            len(graph_facts),
            len(combined_context),
        )
        
        return {
            "context": combined_context,
            "trace": {
                "vector": [
                    {
                        "rank": index + 1,
                        "distance": distance,
                        "document": document,
                    }
                    for index, (document, distance) in enumerate(
                        zip(vector_docs, vector_distances)
                    )
                ],
                "graph": graph_facts,
                "entities": entities,
            },
        }

    def list_collections(self):
        return self.client.list_collections()

    def delete_collection(self, collection_name):
        self.client.delete_collection(name=collection_name)

    
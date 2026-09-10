from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from embeddings import EmbeddingService

class GraphDB:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        self.embedding_service = EmbeddingService()

    def close(self):
        self.driver.close()

    def add_triplets(self, triplets):
        """
        Adds SPO triplets to Neo4j.
        triplets: List of (subject, predicate, object)
        """
        with self.driver.session() as session:
            for s, p, o in triplets:
                # Generate embeddings for subject and object for hybrid search
                s_embedding = self.embedding_service.get_embedding(s)
                o_embedding = self.embedding_service.get_embedding(o)
                
                session.execute_write(self._create_triplet, s, p, o, s_embedding, o_embedding)

    @staticmethod
    def _create_triplet(tx, s, p, o, s_emb, o_emb):
        # Use MERGE to avoid duplicate nodes
        query = """
        MERGE (sub:Entity {name: $s})
        SET sub.embedding = $s_emb
        MERGE (obj:Entity {name: $o})
        SET obj.embedding = $o_emb
        MERGE (sub)-[r:RELATION {predicate: $p}]->(obj)
        """
        tx.run(query, s=s, p=p, o=o, s_emb=s_emb, o_emb=o_emb)

    def query_graph(self, entity_name, depth=1):
        """
        Retrieves the neighborhood of a given entity.
        """
        with self.driver.session() as session:
            query = """
            MATCH (e:Entity {name: $name})-[r*1..$depth]-(neighbor)
            RETURN e.name as start, type(r[0]) as rel, neighbor.name as end
            """
            # This is a simplified query; since we use a custom RELATION type with a predicate property:
            query = """
            MATCH (e:Entity {name: $name})-[r:RELATION]-(neighbor)
            RETURN e.name as start, r.predicate as predicate, neighbor.name as end
            """
            result = session.run(query, name=entity_name)
            return [record.data() for record in result]

    def hybrid_search(self, query_text, vector_results):
        """
        Combines vector results with graph traversal.
        vector_results: documents retrieved from ChromaDB
        """
        # Extract potential entities from the query (simple approach: use LLM or just the vector results)
        # For this implementation, we'll look for entities mentioned in the top vector results
        context_fragments = []
        
        # To make it a true hybrid search, we can identify entities in the query 
        # and expand them using the graph.
        # Since we don't have a named entity recognizer, we'll use the vector results as seeds.
        
        seeds = []
        for doc in vector_results:
            # Very basic extraction: we'd usually use LLM to find entities in the doc
            # For now, we'll assume we can find some matches in the graph
            pass
        
        # A better hybrid approach:
        # 1. Use vector search to find relevant nodes/documents.
        # 2. Traverse the graph from those nodes to find related facts.
        
        return context_fragments


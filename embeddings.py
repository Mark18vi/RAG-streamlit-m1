from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL
from logging_config import get_logger

logger = get_logger(__name__)

class EmbeddingService:
    def __init__(self):
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded: %s", EMBEDDING_MODEL)

    def get_embedding(self, text: str):
        embedding = self.model.encode(text)
        logger.debug("Embedding generated: input_characters=%d", len(text))
        return embedding


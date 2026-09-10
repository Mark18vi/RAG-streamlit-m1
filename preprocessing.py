import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from logging_config import get_logger

logger = get_logger(__name__)


def clean_text(text):
    """
    Cleans the input text by removing unwanted characters and formatting.

    Args:
        text (str): The input text to be cleaned.
    """
    logger.info("Text cleaning started: characters=%d", len(text))
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with a single space
    text = re.sub(r'[^\w\s.,!?]', '', text)  # Remove unwanted characters (keeping only alphanumeric, whitespace, and basic punctuation)
    text = text.strip()  # Remove leading and trailing whitespace
    logger.info("Text cleaning completed: characters=%d", len(text))
    return text

def split_text(text, chunk_size=500, chunk_overlap=50):
    """
    Splits the input text into smaller chunks.

    Args:
        text (str): The input text to be split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of overlapping characters between chunks.
    """
    logger.info(
        "Text chunking started: characters=%d chunk_size=%d overlap=%d",
        len(text),
        chunk_size,
        chunk_overlap,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    logger.info("Text chunking completed: chunks=%d", len(chunks))
    return chunks
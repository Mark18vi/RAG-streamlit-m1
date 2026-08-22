import re

from langchain_text_splitters import RecursiveCharacterTextSplitter



def clean_text(text):
    """
    Cleans the input text by removing unwanted characters and formatting.

    Args:
        text (str): The input text to be cleaned.
    """
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with a single space
    text = re.sub(r'[^\w\s.,!?]', '', text)  # Remove unwanted characters (keeping only alphanumeric, whitespace, and basic punctuation)
    text = text.strip()  # Remove leading and trailing whitespace
    return text

def split_text(text, chunk_size=500, chunk_overlap=50):
    """
    Splits the input text into smaller chunks.

    Args:
        text (str): The input text to be split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of overlapping characters between chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(text)
from ingestion import extract_pdf_text
from preprocessing import clean_text, split_text
from embeddings import EmbeddingService
from vector_db import VectorDB
from llm import LLMservice

from config import FILE_UPLOAD_DIR

files_path = FILE_UPLOAD_DIR

pdf_path = files_path + "\\fifa_rag.pdf"  # Replace with the actual PDF file name

def start_ingestion(pdf_path):
    """
    Starts the ingestion process for a given PDF file.

    Args:
        pdf_path (str): The path to the PDF file.
    """
    # Step 1: Extract text from the PDF
    raw_text = extract_pdf_text(pdf_path)

    print("--> Raw text extracted from PDF")

    # Step 2: Clean the extracted text
    cleaned_text = clean_text(raw_text)
    print("--> Text cleaned")

    # Step 3: Split the cleaned text into chunks
    text_chunks = split_text(cleaned_text)
    print(f"--> Text split into {len(text_chunks)} chunks")

    # Step 4: Initialize the vector database
    vector_db = VectorDB()
    print("--> Vector database initialized")

    # Step 5: Add the text chunks to the vector database
    vector_db.add_documents(text_chunks)
    print("--> Text chunks added to the vector database")

    print(f"Ingestion completed for {pdf_path}. Total chunks added: {len(text_chunks)}")
    return 200


# start_ingestion(pdf_path)

def call_llm_with_query(query):
    """
    Calls the LLM with a given query and retrieves relevant documents from the vector database.

    Args:
        query (str): The query to be processed by the LLM.
    """
    vector_db = VectorDB()
    results = vector_db.query(query)
    print("--> Query results retrieved from the vector database", len(results))
    llm_service = LLMservice()

    response = llm_service.ask_question(query, results)

    print("\n\n ** Thinking --> ", response)

    return response

call_llm_with_query("where Fifa 2026 has been held?")
import os

from ingestion import extract_pdf_text
from preprocessing import clean_text, split_text
from embeddings import EmbeddingService
from vector_db import VectorDB
from llm import LLMservice
import json
import requests

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

# call_llm_with_query("where Fifa 2026 has been held?")

# --- Live Research Assistant Implementation ---
TAVILY_KEY = os.getenv("TAVILY_KEY")

def mock_google_search(query):
    """Perfrom real time web search using TAVILY API."""
    print(f" [Tool] Searching Google for: {query}...")

    url = "https://api.tavily.com/search"

    payload = {
        "api_key": TAVILY_KEY,
        "query": query,
        "search_depth": "basic",
        "max_results": 5 
    }

    response = requests.post(url, json=payload)

    results = response.raise_for_status()

    data = response.json()

    if not data.get("results"):
        return "No results found for the query."

    results = []

    for item in data["results"]:
        results.append({
            f'''
            Title: {item.get("title")} \n,
            url: {item.get("url")} \n,
            Content: {item.get("content")}
            '''
        })

        
    return "\n\n".join([str(result) for result in results])

def mock_calculator(expression):
    """Simulates a calculator tool."""
    print(f" [Tool] Calculating: {expression}...")
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error calculating: {str(e)}"

def call_research_assistant(query):
    """
    Implements a ReAct loop using LLM Function Calling.
    """
    llm_service = LLMservice()
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "google_search",
                "description": "Search the web for live, current information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Perform mathematical calculations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "The math expression to evaluate (e.g., '2 + 2')"}
                    },
                    "required": ["expression"]
                }
            }
        }
    ]

    messages = [
        {"role": "system", "content": "You are a Live Research Assistant. Use the provided tools to find accurate, current information. Always reason before acting (ReAct pattern)."},
        {"role": "user", "content": query}
    ]

    max_iterations = 5
    for i in range(max_iterations):
        response_message = llm_service.call_with_tools(messages, tools)
        
        if response_message.tool_calls:
            messages.append(response_message)
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name == "google_search":
                    result = mock_google_search(function_args.get("query"))
                elif function_name == "calculator":
                    result = mock_calculator(function_args.get("expression"))
                else:
                    result = "Tool not found."
                
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": result
                })
        else:
            return response_message.content

    return "Reached maximum iterations without a final answer."

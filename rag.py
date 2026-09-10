import os

from ingestion import extract_pdf_text
from preprocessing import clean_text, split_text
from embeddings import EmbeddingService
from vector_db import VectorDB
from llm import LLMservice
import json
import requests
from logging_config import get_logger

from config import FILE_UPLOAD_DIR

files_path = FILE_UPLOAD_DIR
logger = get_logger(__name__)

pdf_path = files_path + "\\fifa_rag.pdf"  # Replace with the actual PDF file name

def start_ingestion(pdf_path):
    """
    Starts the ingestion process for a given PDF file.

    Args:
        pdf_path (str): The path to the PDF file.
    """
    logger.info("Ingestion flow started: file=%s", pdf_path)
    # Step 1: Extract text from the PDF
    raw_text = extract_pdf_text(pdf_path)

    # Step 2: Clean the extracted text
    cleaned_text = clean_text(raw_text)

    # Step 3: Split the cleaned text into chunks
    text_chunks = split_text(cleaned_text)

    # Step 4: Initialize the vector database
    vector_db = VectorDB()

    # Step 5: Initialize LLM for SPO extraction
    llm_service = LLMservice()

    logger.info("Starting storage flow: ChromaDB and Neo4j")
    vector_db.add_documents(text_chunks, llm_service=llm_service)
    logger.info("Ingestion flow completed: file=%s chunks=%d", pdf_path, len(text_chunks))
    return 200


# start_ingestion(pdf_path)

def call_llm_with_query(query):
    """
    Calls the LLM with a given query and retrieves relevant documents from the vector database.

    Args:
        query (str): The query to be processed by the LLM.
    """
    logger.info("Document question flow started: query_characters=%d", len(query))
    vector_db = VectorDB()
    results = vector_db.query(query)
    retrieval_trace = results["trace"]
    logger.info(
        "Hybrid retrieval completed: vector_chunks=%d neo4j_facts=%d",
        len(retrieval_trace["vector"]),
        len(retrieval_trace["graph"]),
    )
    llm_service = LLMservice()

    response = llm_service.ask_question(query, results["context"])

    logger.info("Document question flow completed")
    return {"answer": response, "retrieval": retrieval_trace}

# call_llm_with_query("where Fifa 2026 has been held?")

# --- Live Research Assistant Implementation ---
TAVILY_KEY = os.getenv("TAVILY_KEY")

def mock_google_search(query):
    """Perfrom real time web search using TAVILY API."""
    logger.info("Research tool started: web_search query_characters=%d", len(query))

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

        
    output = "\n\n".join([str(result) for result in results])
    logger.info("Research tool completed: web_search results=%d", len(results))
    return output

def mock_calculator(expression):
    """Simulates a calculator tool."""
    logger.info("Research tool started: calculator")
    try:
        result = str(eval(expression))
        logger.info("Research tool completed: calculator")
        return result
    except Exception as e:
        logger.exception("Research tool failed: calculator")
        return f"Error calculating: {str(e)}"

def call_research_assistant(query):
    """
    Implements a ReAct loop using LLM Function Calling.
    """
    logger.info("Live research flow started: query_characters=%d", len(query))
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
        logger.info("Live research iteration started: iteration=%d", i + 1)
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
                    logger.warning("Unknown tool requested: %s", function_name)
                    result = "Tool not found."
                
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": result
                })
        else:
            logger.info("Live research flow completed: iterations=%d", i + 1)
            return response_message.content

    logger.warning("Live research flow reached iteration limit: limit=%d", max_iterations)
    return "Reached maximum iterations without a final answer."

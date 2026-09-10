import streamlit as st
from rag import start_ingestion, call_llm_with_query, call_research_assistant
from logging_config import get_logger

logger = get_logger(__name__)

# st.title("RAG Information App")

st.set_page_config(page_title="RAG Information App", page_icon=":guardsman:", layout="wide")

st.title("✨Chat with your data")

if "messages" not in st.session_state:
    st.session_state.messages = []
with st.sidebar:
    st.title("RAG Information App")
    st.markdown(
        """
        This app allows you to upload a PDF file and ask questions about its content.
        The app uses a vector database to store the content of the PDF and a language model to answer your questions.
        """
    )
    
    st.divider()
    st.header("Assistant Mode")
    mode = st.radio("Select Mode", ["Document RAG", "Live Research Assistant"], index=0)
    
    st.divider()
    st.header("Upload your PDF file")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:

        if st.button("Start Ingestion"):
            logger.info("UI action: ingestion requested file=%s", uploaded_file.name)
            with st.spinner("Ingesting the PDF..."):
                from rag import start_ingestion

                # Save the uploaded file to a temporary location
                temp_file_path = f"temp_{uploaded_file.name}"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                logger.info("Uploaded file saved: path=%s bytes=%d", temp_file_path, uploaded_file.size)
                # Start the ingestion process
                response = start_ingestion(temp_file_path)

                if response == 200:
                    logger.info("UI action completed: ingestion succeeded")
                    st.success("Ingestion completed successfully!")
                else:
                    logger.error("UI action failed: ingestion response=%s", response)
                    st.error("Ingestion failed. Please try again.")

st.divider()

# chat interface
st.header(f"Ask questions ({mode})")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("retrieval"):
            retrieval = message["retrieval"]
            vector_count = len(retrieval.get("vector", []))
            graph_count = len(retrieval.get("graph", []))
            with st.expander(
                f"Retrieval details · {vector_count} vector chunks · {graph_count} Neo4j facts"
            ):
                st.caption(
                    "Hybrid retrieval combines semantic vector matches with SPO relationships from Neo4j."
                )
                st.markdown(f"**Entities used for graph expansion:** {', '.join(retrieval.get('entities', [])) or 'None'}")
                st.markdown("**Vector search results**")
                for item in retrieval.get("vector", []):
                    st.markdown(
                        f"**#{item['rank']}** · distance `{item['distance']:.4f}`"
                    )
                    st.caption(item["document"])
                st.markdown("**Neo4j SPO results**")
                if graph_count:
                    for fact in retrieval["graph"]:
                        st.code(
                            f"{fact['subject']} --[{fact['predicate']}]--> {fact['object']}",
                            language="text",
                        )
                else:
                    st.caption("No matching Neo4j relationships were found.")

placeholder_text = "Ask a question about the documents" if mode == "Document RAG" else "Ask a live research question (e.g. 'Where is Fifa 2026 held?')"
question = st.chat_input(placeholder_text)

if question:
    logger.info("UI action: question submitted mode=%s query_characters=%d", mode, len(question))
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Thinking..."):
        if mode == "Document RAG":
            from rag import call_llm_with_query
            result = call_llm_with_query(question)
            response = result["answer"]
            retrieval = result["retrieval"]
        else:
            from rag import call_research_assistant
            response = call_research_assistant(question)
            retrieval = None
        logger.info("UI action completed: response generated mode=%s", mode)

        st.session_state.messages.append(
            {"role": "assistant", "content": response, "retrieval": retrieval}
        )
        with st.chat_message("assistant"):
            st.markdown(response)
            if retrieval:
                vector_count = len(retrieval.get("vector", []))
                graph_count = len(retrieval.get("graph", []))
                with st.expander(
                    f"Retrieval details · {vector_count} vector chunks · {graph_count} Neo4j facts"
                ):
                    st.caption(
                        "Hybrid retrieval combines semantic vector matches with SPO relationships from Neo4j."
                    )
                    st.markdown(
                        f"**Entities used for graph expansion:** "
                        f"{', '.join(retrieval.get('entities', [])) or 'None'}"
                    )
                    st.markdown("**Vector search results**")
                    for item in retrieval.get("vector", []):
                        st.markdown(
                            f"**#{item['rank']}** · distance `{item['distance']:.4f}`"
                        )
                        st.caption(item["document"])
                    st.markdown("**Neo4j SPO results**")
                    if graph_count:
                        for fact in retrieval["graph"]:
                            st.code(
                                f"{fact['subject']} --[{fact['predicate']}]--> {fact['object']}",
                                language="text",
                            )
                    else:
                        st.caption("No matching Neo4j relationships were found.")
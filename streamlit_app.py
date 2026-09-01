import streamlit as st
from rag import start_ingestion, call_llm_with_query

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
    st.header("Upload your PDF file")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:

        if st.button("Start Ingestion"):
            with st.spinner("Ingesting the PDF..."):
                from rag import start_ingestion

                # Save the uploaded file to a temporary location
                temp_file_path = f"temp_{uploaded_file.name}"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                print(f"Temporary file saved at: {temp_file_path}")
                # Start the ingestion process
                response = start_ingestion(temp_file_path)

                if response == 200:
                    st.success("Ingestion completed successfully!")
                else:
                    st.error("Ingestion failed. Please try again.")

st.divider()

# chat interface
st.header("Ask questions about the documents")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about the documents")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Thinking..."):
        from rag import call_llm_with_query

        response = call_llm_with_query(question)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
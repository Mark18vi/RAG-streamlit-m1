import streamlit as st
from rag import start_ingestion, call_llm_with_query, call_research_assistant
from travel_planner import (
    apply_travel_answer,
    approve_trip,
    build_travel_graph,
    is_waiting_for_approval,
    next_travel_question,
    start_trip,
)

# st.title("RAG Information App")


def _render_travel_results(data):
    """Render live travel results as readable cards inside the chat response."""
    recommendation_index = data.get("recommendation_index", 0)
    flights = data.get("flight_options", [])
    hotels = data.get("hotel_options", [])
    locations = data.get("location_options", [])

    st.markdown("### Live trip recommendations")
    if flights:
        flight = flights[min(recommendation_index, len(flights) - 1)]
        st.markdown("#### Recommended flight")
        st.markdown(f"**{flight.get('title', 'Flight option')}**")
        st.caption(flight.get("description", "Live flight information"))
        if flight.get("rating") != "Rating not reported":
            st.write(f"Rating: {flight['rating']}")

    if hotels:
        hotel = hotels[min(recommendation_index, len(hotels) - 1)]
        st.markdown("#### Recommended hotel")
        hotel_columns = st.columns([1, 2])
        with hotel_columns[0]:
            if hotel.get("image"):
                st.image(hotel["image"], use_container_width=True)
        with hotel_columns[1]:
            st.markdown(f"**{hotel.get('title', 'Hotel option')}**")
            st.write(f"Rating: {hotel.get('rating', 'Rating not reported')}")
            st.caption(hotel.get("description", "Live hotel information"))

    itinerary = data.get("itinerary", [])
    if itinerary:
        st.markdown("#### Country itinerary")
        location_columns = st.columns(min(3, len(itinerary)))
        for index, item in enumerate(itinerary):
            with location_columns[index % len(location_columns)]:
                location = locations[index] if index < len(locations) else {}
                if location.get("image"):
                    st.image(location["image"], use_container_width=True)
                st.markdown(f"**Day {item['day']}: {item['location']}**")
                st.write(f"Rating: {item['rating']}")
                st.caption(item["date"])


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
    mode = st.radio(
        "Select Mode",
        ["Document RAG", "Live Research Assistant", "Travel Planner"],
        index=0,
    )

if mode == "Travel Planner":
    st.header("Travel Planner")
    st.caption(
        "Tell me about your trip one detail at a time. I will search flights, "
        "cities, attractions, and hotels in the country you choose."
    )

    if "travel_graph" not in st.session_state:
        st.session_state.travel_graph = build_travel_graph()
    if "travel_memory" not in st.session_state:
        st.session_state.travel_memory = {}
    if "travel_chat_messages" not in st.session_state:
        st.session_state.travel_chat_messages = [
            {"role": "assistant", "content": "Hi! I can help plan your trip. " + next_travel_question({})}
        ]
    if "travel_details" not in st.session_state:
        st.session_state.travel_details = {}

    for message in st.session_state.travel_chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    result = st.session_state.get("travel_result")
    if result and is_waiting_for_approval(result):
        st.info("Human approval required before the itinerary is finalized.")
        interrupt_value = result["__interrupt__"][0].value
        with st.chat_message("assistant"):
            _render_travel_results(interrupt_value)
            st.caption("These recommendations are live search results. Review them before approving.")
        col1, col2 = st.columns(2)
        if col1.button("Approve itinerary", type="primary"):
            st.session_state.travel_result = approve_trip(
                st.session_state.travel_graph,
                st.session_state.travel_thread_id,
                True,
            )
            st.rerun()
        if col2.button("Reject and revise"):
            st.session_state.travel_result = approve_trip(
                st.session_state.travel_graph,
                st.session_state.travel_thread_id,
                False,
            )
            st.rerun()
    elif result and result.get("summary"):
        with st.chat_message("assistant"):
            st.markdown(result["summary"])
            _render_travel_results(result)
        st.caption(
            "Short-term memory: this graph run is checkpointed by thread. "
            "Long-term memory: your hotel-area preference is kept in this session."
        )
    else:
        question = st.chat_input("Answer the travel planner")
        if question:
            st.session_state.travel_chat_messages.append({"role": "user", "content": question})
            details, error = apply_travel_answer(st.session_state.travel_details, question)
            st.session_state.travel_details = details
            if error:
                reply = error
            else:
                next_question = next_travel_question(details)
                if next_question:
                    reply = next_question
                else:
                    st.session_state.travel_thread_id = (
                        f"travel-{st.session_state.get('travel_run', 0) + 1}"
                    )
                    st.session_state.travel_run = st.session_state.get("travel_run", 0) + 1
                    st.session_state.travel_memory["hotel_area"] = details["hotel_area"]
                    st.session_state.travel_result = start_trip(
                        st.session_state.travel_graph,
                        st.session_state.travel_thread_id,
                        {
                            "origin": details["origin"],
                            "destination": details["destination"],
                            "departure": details["departure"],
                            "return": details["return"],
                            "travelers": details["travelers"],
                            "cabin": details["cabin"],
                        },
                        {"hotel_area": details["hotel_area"]},
                    )
                    reply = "I have prepared flight and hotel options. Please review them below."
            st.session_state.travel_chat_messages.append({"role": "assistant", "content": reply})
            st.rerun()
    st.stop()
    
    st.divider()
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
st.header(f"Ask questions ({mode})")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

placeholder_text = "Ask a question about the documents" if mode == "Document RAG" else "Ask a live research question (e.g. 'Where is Fifa 2026 held?')"
question = st.chat_input(placeholder_text)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Thinking..."):
        if mode == "Document RAG":
            from rag import call_llm_with_query
            response = call_llm_with_query(question)
        else:
            from rag import call_research_assistant
            response = call_research_assistant(question)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
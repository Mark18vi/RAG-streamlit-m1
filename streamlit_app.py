import streamlit as st
from rag import start_ingestion, call_llm_with_query, call_research_assistant
from travel_planner import (
    apply_travel_answer,
    approve_trip,
    build_travel_graph,
    get_memory_snapshot,
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
    selected_flight = data.get("selected_flight")
    selected_hotel = data.get("selected_hotel")
    if selected_flight:
        flight = selected_flight
    elif flights:
        flight = flights[min(recommendation_index, len(flights) - 1)]
    else:
        flight = None
    if selected_hotel:
        hotel = selected_hotel
    elif hotels:
        hotel = hotels[min(recommendation_index, len(hotels) - 1)]
    else:
        hotel = None

    st.markdown(
        f"### Recommendation {recommendation_index + 1}: "
        f"{data.get('request', {}).get('destination', 'your destination')}"
    )
    if flight:
        st.markdown("#### ✈️ Recommended flight")
        st.markdown(f"**{flight.get('title', 'Flight option')}**")
        st.caption(flight.get("description", "Live flight information"))
        st.write(
            "Why this option: matches your requested route, travel date, and cabin preference."
        )

    if hotel:
        st.markdown("#### 🏨 Recommended hotel")
        hotel_columns = st.columns([1, 2])
        with hotel_columns[0]:
            if hotel.get("image"):
                st.image(hotel["image"], use_container_width=True)
        with hotel_columns[1]:
            st.markdown(f"**{hotel.get('title', 'Hotel option')}**")
            st.write(f"Rating: {hotel.get('rating', 'Rating not reported')}")
            st.caption(hotel.get("description", "Live hotel information"))
            st.write(hotel.get("why_visit", "Selected from current live hotel research."))

    itinerary = data.get("itinerary", [])
    if itinerary:
        st.markdown("#### 🗺️ Day-by-day country itinerary")
        map_points = []
        for item in itinerary:
            if item.get("latitude") and item.get("longitude"):
                map_points.append(
                    {
                        "latitude": item["latitude"],
                        "longitude": item["longitude"],
                        "day": item["day"],
                        "location": item["location"],
                    }
                )
        if map_points:
            st.map(map_points, latitude="latitude", longitude="longitude", zoom=5)
            st.caption("Route overview using OpenStreetMap location data.")
        for item in itinerary:
            with st.container(border=True):
                columns = st.columns([1, 2])
                with columns[0]:
                    if item.get("image"):
                        st.image(item["image"], use_container_width=True)
                with columns[1]:
                    st.markdown(f"**Day {item['day']} · {item['date']}**")
                    st.subheader(item["location"])
                    st.write(f"Rating: {item['rating']}")
                    st.caption(item.get("description", "Live location information"))
                    st.write(item.get("why_visit", "Selected from current travel research."))


def _render_memory_tooltip():
    """Render graph memory as a compact bottom-of-chat popover."""
    snapshot = get_memory_snapshot(
        st.session_state.travel_graph,
        st.session_state.get("travel_thread_id"),
        st.session_state.travel_details,
        st.session_state.travel_memory,
        st.session_state.get("travel_result"),
    )
    nodes = [
        ("intake", "Chat intake"),
        ("plan_trip", "Plan trip"),
        ("search_flights", "Live flights"),
        ("summarize_hotels", "Hotels"),
        ("search_locations", "Locations"),
        ("build_itinerary", "Itinerary"),
        ("request_approval", "Approval"),
        ("finalize", "Final plan"),
    ]
    active = snapshot["active_node"]
    graph_lines = ["digraph {", "rankdir=LR;", 'node [shape=box style="rounded,filled" fontname="Arial"];']
    for index, (node_id, label) in enumerate(nodes):
        color = "#b7e4c7" if node_id == active else "#e8eef7"
        graph_lines.append(f'"{node_id}" [label="{label}" fillcolor="{color}"];')
        if index:
            graph_lines.append(f'"{nodes[index - 1][0]}" -> "{node_id}";')
    graph_lines.append("}")

    with st.popover("🧠 Agent memory", use_container_width=False):
        st.caption("LangGraph checkpoint memory and session preference memory")
        st.graphviz_chart("\n".join(graph_lines), use_container_width=True)
        st.markdown(f"**Active node:** `{active}`")
        st.markdown(f"**Thread checkpoint:** `{snapshot['thread_id']}`")
        st.markdown("**Short-term memory**")
        st.json(snapshot["short_term"])
        st.markdown("**Long-term memory**")
        st.json(snapshot["long_term"] or {"status": "No saved preferences yet"})


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
            if is_waiting_for_approval(st.session_state.travel_result):
                next_index = st.session_state.travel_result["__interrupt__"][0].value.get(
                    "recommendation_index", 1
                )
                st.session_state.travel_chat_messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"That option was rejected. Here is recommendation "
                            f"{next_index + 1} for comparison."
                        ),
                    }
                )
            st.rerun()
    elif result and result.get("summary"):
        with st.chat_message("assistant"):
            st.markdown(result["summary"])
            _render_travel_results(result)
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
    st.divider()
    _render_memory_tooltip()
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
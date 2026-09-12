"""LangGraph travel planning workflow used by the Streamlit Travel Planner."""

from __future__ import annotations

from datetime import date, datetime
import os
import re
from typing import Any, TypedDict

import requests
from dotenv import load_dotenv

load_dotenv()


class TravelState(TypedDict, total=False):
    request: dict[str, Any]
    preferences: dict[str, Any]
    flight_options: list[dict[str, Any]]
    hotel_options: list[dict[str, Any]]
    location_options: list[dict[str, Any]]
    itinerary: list[dict[str, Any]]
    live_sources: list[dict[str, str]]
    live_search_error: str
    recommendation_index: int
    approval_request: dict[str, Any]
    approved: bool
    summary: str


TRAVEL_QUESTIONS = (
    ("origin", "Which airport or city are you departing from?"),
    ("destination", "Which country would you like to visit?"),
    ("departure", "What is your departure date? Use YYYY-MM-DD."),
    ("return", "What is your return date? Use YYYY-MM-DD."),
    ("travelers", "How many travelers are going?"),
    ("cabin", "Which cabin do you prefer: Economy, Premium economy, or Business?"),
    ("hotel_area", "Which hotel area do you prefer? You can say 'central' or 'skip'."),
)


def next_travel_question(details: dict[str, Any]) -> str | None:
    """Return the next context-aware question for the chat intake."""
    for key, question in TRAVEL_QUESTIONS:
        if not details.get(key):
            return question
    return None


def _parse_date(value: str) -> str | None:
    for format_string in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(value.strip(), format_string).date().isoformat()
        except ValueError:
            continue
    return None


def apply_travel_answer(details: dict[str, Any], answer: str) -> tuple[dict[str, Any], str | None]:
    """Validate one chat answer and return updated details plus an error message."""
    key = next((field for field, _ in TRAVEL_QUESTIONS if not details.get(field)), None)
    if key is None:
        return details, None

    value = answer.strip()
    if key in {"origin", "destination"}:
        if len(value) < 2:
            return details, "Please provide an airport code or city name."
        details[key] = value
    elif key in {"departure", "return"}:
        parsed = _parse_date(value)
        if not parsed:
            return details, "I couldn't read that date. Please use YYYY-MM-DD."
        if key == "return" and details.get("departure"):
            if date.fromisoformat(parsed) < date.fromisoformat(details["departure"]):
                return details, "The return date cannot be before the departure date."
        details[key] = parsed
    elif key == "travelers":
        try:
            count = int(value)
        except ValueError:
            return details, "Please enter the number of travelers as a whole number."
        if not 1 <= count <= 12:
            return details, "Please choose between 1 and 12 travelers."
        details[key] = count
    elif key == "cabin":
        cabins = {"economy": "Economy", "premium economy": "Premium economy", "business": "Business"}
        cabin = cabins.get(value.lower())
        if not cabin:
            return details, "Please choose Economy, Premium economy, or Business."
        details[key] = cabin
    elif key == "hotel_area":
        details[key] = "central" if value.lower() in {"", "skip"} else value

    return details, None


def _plan_trip(state: TravelState) -> dict[str, Any]:
    request = state["request"]
    return {
        "request": {
            **request,
            "travelers": max(1, int(request.get("travelers", 1))),
            "cabin": request.get("cabin", "Economy"),
        }
    }


def _search_flights(state: TravelState) -> dict[str, Any]:
    request = state["request"]
    origin = request["origin"].upper()
    destination = request["destination"]
    departure = request["departure"]
    flights, error = _live_search_or_empty(
        f"live flights {origin} to {destination} on {departure} {request['cabin']}",
        "flight",
    )
    return {"flight_options": flights, "live_search_error": error or ""}


def _tavily_search(query: str, result_type: str) -> list[dict[str, Any]]:
    """Return current web results, preserving URLs for user verification."""
    api_key = os.getenv("TAVILY_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_KEY is not configured; live travel search is unavailable.")

    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": 8,
            "include_answer": False,
            "include_images": True,
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results", [])
    images = payload.get("images", [])
    return [
        {
            "type": result_type,
            "title": item.get("title", "Untitled result"),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
            "rating": _extract_rating(f"{item.get('title', '')} {item.get('content', '')}"),
            "image": images[index] if index < len(images) else "",
            "description": _short_description(item.get("content", "")),
        }
        for index, item in enumerate(results)
    ]


def _geocode_location(name: str) -> tuple[float | None, float | None]:
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": name, "format": "jsonv2", "limit": 1},
        headers={"User-Agent": "travel-planner-streamlit/1.0"},
        timeout=10,
    )
    response.raise_for_status()
    matches = response.json()
    if not matches:
        return None, None
    return float(matches[0]["lat"]), float(matches[0]["lon"])


def _live_search_or_empty(query: str, result_type: str) -> tuple[list[dict[str, Any]], str | None]:
    try:
        return _tavily_search(query, result_type), None
    except (requests.RequestException, RuntimeError) as error:
        return [], str(error)


def _extract_rating(text: str) -> str:
    match = re.search(r"\b([1-5](?:\.\d)?)\s*(?:/ ?5|stars?|out of 5)\b", text, re.IGNORECASE)
    return f"{match.group(1)}/5" if match else "Rating not reported"


def _short_description(text: str) -> str:
    compact = " ".join(text.split())
    return compact[:220] + ("..." if len(compact) > 220 else "")


def _summarize_hotels(state: TravelState) -> dict[str, Any]:
    request = state["request"]
    preferences = state.get("preferences", {})
    preferred_area = preferences.get("hotel_area", "central")
    hotels, error = _live_search_or_empty(
        f"best hotels in {request['destination']} near {preferred_area} current reviews ratings price",
        "hotel",
    )
    for hotel in hotels:
        hotel["why_visit"] = (
            f"Fits your preference for {preferred_area} and is selected from current "
            "hotel reviews and availability discussions."
        )
    return {"hotel_options": hotels, "live_search_error": error or state.get("live_search_error", "")}


def _build_itinerary(state: TravelState) -> dict[str, Any]:
    request = state["request"]
    departure = date.fromisoformat(request["departure"])
    return_date = date.fromisoformat(request["return"])
    trip_days = max(1, (return_date - departure).days)
    locations = state.get("location_options", [])
    visit_count = min(trip_days, len(locations))
    itinerary = [
        {
            "day": day,
            "date": (departure.fromordinal(departure.toordinal() + day - 1)).isoformat(),
            "location": locations[day - 1]["title"],
            "rating": locations[day - 1]["rating"],
            "url": locations[day - 1]["url"],
            "image": locations[day - 1].get("image", ""),
            "description": locations[day - 1].get("description", ""),
            "why_visit": locations[day - 1].get("why_visit", ""),
            "latitude": locations[day - 1].get("latitude"),
            "longitude": locations[day - 1].get("longitude"),
        }
        for day in range(1, visit_count + 1)
    ]
    return {"itinerary": itinerary}


def _search_locations(state: TravelState) -> dict[str, Any]:
    destination = state["request"]["destination"]
    locations, error = _live_search_or_empty(
        f"best cities and places to visit in {destination} ratings current travel guide",
        "location",
    )
    try:
        country_latitude, country_longitude = _geocode_location(destination)
    except requests.RequestException:
        country_latitude, country_longitude = None, None
    for location in locations:
        location["why_visit"] = (
            f"Recommended for a {state['request']['return']} return date and your "
            f"{state.get('preferences', {}).get('hotel_area', 'central')} base preference."
        )
        try:
            latitude, longitude = _geocode_location(
                f"{location['title']} {destination}"
            )
        except requests.RequestException:
            latitude, longitude = None, None
        location["latitude"] = latitude if latitude is not None else country_latitude
        location["longitude"] = longitude if longitude is not None else country_longitude
    return {
        "location_options": locations,
        "live_search_error": error or state.get("live_search_error", ""),
    }


def _request_approval(state: TravelState) -> dict[str, Any]:
    from langgraph.types import interrupt

    index = min(
        state.get("recommendation_index", 0),
        max(0, len(state.get("flight_options", [])) - 1),
        max(0, len(state.get("hotel_options", [])) - 1),
    )
    approval = interrupt(
        {
            "message": "Review the proposed trip before generating the final itinerary.",
            "request": state["request"],
            "flight_options": state["flight_options"],
            "hotel_options": state["hotel_options"],
            "location_options": state.get("location_options", []),
            "itinerary": state.get("itinerary", []),
            "recommendation_index": index,
            "selected_flight": state.get("flight_options", [{}])[index],
            "selected_hotel": state.get("hotel_options", [{}])[index],
        }
    )
    index = state.get("recommendation_index", 0)
    return {
        "approved": bool(approval),
        "recommendation_index": index if approval else index + 1,
    }


def _approval_route(state: TravelState) -> str:
    if state.get("approved", False):
        return "finalize"
    option_count = min(
        len(state.get("flight_options", [])),
        len(state.get("hotel_options", [])),
    )
    if state.get("recommendation_index", 0) < option_count:
        return "request_approval"
    return "finalize"


def _finalize(state: TravelState) -> dict[str, Any]:
    if not state.get("approved", False):
        return {"summary": "No more live flight and hotel recommendations were available for approval."}

    request = state["request"]
    index = state.get("recommendation_index", 0)
    flight = state.get("flight_options", [{}])[index]
    hotel = state.get("hotel_options", [{}])[index]
    itinerary = state.get("itinerary", [])
    live_error = state.get("live_search_error")
    itinerary_lines = "\n".join(
        f"- **Day {item['day']} ({item['date']}):** {item['location']} "
        f"- rating: {item['rating']}"
        for item in itinerary
    ) or "No live attraction results were returned."
    return {
        "summary": (
            f"### Trip plan: {request['origin'].upper()} to {request['destination'].title()}\n\n"
            f"**Dates:** {request['departure']} to {request['return']} ({len(itinerary)} "
            f"recommended locations across {(date.fromisoformat(request['return']) - date.fromisoformat(request['departure'])).days} days)  \n"
            f"**Flight recommendation:** {flight.get('title', 'Live flight option')}  \n"
            f"**Hotel recommendation:** {hotel.get('title', 'Live hotel option')} "
            f"- rating: {hotel.get('rating', 'Rating not reported')}\n\n"
            f"### Day-by-day itinerary\n{itinerary_lines}\n\n"
            f"{'**Live search warning:** ' + live_error if live_error else ''}\n\n"
            "Ratings and availability come from live search snippets and must be confirmed "
            "with the linked provider before booking."
        )
    }


def build_travel_graph() -> Any:
    """Build the multi-step workflow and its in-memory short-term checkpoint."""
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph

    workflow = StateGraph(TravelState)
    workflow.add_node("plan_trip", _plan_trip)
    workflow.add_node("search_flights", _search_flights)
    workflow.add_node("summarize_hotels", _summarize_hotels)
    workflow.add_node("search_locations", _search_locations)
    workflow.add_node("build_itinerary", _build_itinerary)
    workflow.add_node("request_approval", _request_approval)
    workflow.add_node("finalize", _finalize)
    workflow.add_edge(START, "plan_trip")
    workflow.add_edge("plan_trip", "search_flights")
    workflow.add_edge("search_flights", "summarize_hotels")
    workflow.add_edge("summarize_hotels", "search_locations")
    workflow.add_edge("search_locations", "build_itinerary")
    workflow.add_edge("build_itinerary", "request_approval")
    workflow.add_conditional_edges(
        "request_approval",
        _approval_route,
        {"request_approval": "request_approval", "finalize": "finalize"},
    )
    workflow.add_edge("finalize", END)

    return workflow.compile(checkpointer=MemorySaver())


def start_trip(graph: Any, thread_id: str, request: dict[str, Any], preferences: dict[str, Any]) -> dict[str, Any]:
    """Run the workflow until it needs human approval or finishes."""
    return graph.invoke(
        {"request": request, "preferences": preferences},
        {"configurable": {"thread_id": thread_id}},
    )


def approve_trip(graph: Any, thread_id: str, approved: bool) -> dict[str, Any]:
    """Resume a paused workflow after the user approves or rejects the draft."""
    from langgraph.types import Command

    return graph.invoke(
        Command(resume=approved),
        {"configurable": {"thread_id": thread_id}},
    )


def is_waiting_for_approval(result: dict[str, Any]) -> bool:
    return bool(result.get("__interrupt__"))


def get_memory_snapshot(
    graph: Any,
    thread_id: str | None,
    details: dict[str, Any],
    long_term_memory: dict[str, Any],
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Expose a safe, UI-friendly view of graph and session memory."""
    checkpoint_values: dict[str, Any] = {}
    if thread_id:
        checkpoint = graph.get_state({"configurable": {"thread_id": thread_id}})
        checkpoint_values = checkpoint.values or {}

    if result and is_waiting_for_approval(result):
        active_node = "request_approval"
    elif result and result.get("summary"):
        active_node = "finalize"
    elif details:
        active_node = "intake"
    else:
        active_node = "idle"

    return {
        "active_node": active_node,
        "thread_id": thread_id or "not started",
        "short_term": {
            "fields_collected": sorted(details.keys()),
            "checkpoint_keys": sorted(checkpoint_values.keys()),
            "recommendation_index": checkpoint_values.get(
                "recommendation_index",
                result.get("recommendation_index", 0) if result else 0,
            ),
        },
        "long_term": dict(long_term_memory),
    }

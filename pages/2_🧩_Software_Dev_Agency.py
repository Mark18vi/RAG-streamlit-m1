from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.software_dev_agency.agency import AgencyRun, AgentEvent, SoftwareDevAgency


st.set_page_config(page_title="Software Dev Agency", page_icon="🧩", layout="wide")

st.markdown(
    """
    <style>
    .agent-card { border: 1px solid #e4e4e7; border-radius: 12px; padding: 16px;
        min-height: 130px; background: #fff; }
    .agent-card h3 { margin: 0 0 8px; }
    .agent-status { color: #71717a; font-size: 0.85rem; }
    .flow-arrow { text-align: center; font-size: 2rem; color: #2563eb; padding-top: 36px; }
    .event-row { border-left: 3px solid #2563eb; padding: 6px 12px; margin: 6px 0;
        background: #f8fafc; border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_agent_cards(run: AgencyRun | None) -> None:
    states = {"PM": "Waiting", "Coder": "Waiting", "QA": "Waiting"}
    if run:
        for event in run.events:
            if event.agent in states:
                states[event.agent] = event.event_type.replace("_", " ").title()
    columns = st.columns([3, 1, 3, 1, 3])
    for index, (agent, icon, purpose) in enumerate(
        [
            ("PM", "🧭", "Clarifies goals and acceptance criteria"),
            ("Coder", "💻", "Produces an implementation and tests"),
            ("QA", "🔍", "Checks the acceptance gate and requests revisions"),
        ]
    ):
        with columns[index * 2]:
            st.markdown(
                f'<div class="agent-card"><h3>{icon} {agent} Agent</h3>'
                f'<div class="agent-status"><b>{states[agent]}</b><br>{purpose}</div></div>',
                unsafe_allow_html=True,
            )
        if index < 2:
            with columns[index * 2 + 1]:
                st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)


def render_events(events: list[AgentEvent]) -> None:
    if not events:
        st.info("Agent events will appear here as the run progresses.")
        return
    for event in reversed(events):
        st.markdown(
            f'<div class="event-row"><b>{event.agent}</b> · {event.event_type.replace("_", " ").title()}'
            f'<br><small>{event.message} · {event.timestamp}</small></div>',
            unsafe_allow_html=True,
        )


st.title("🧩 Software Dev Agency")
st.caption("Watch a PM, Coder, and QA Agent turn a request into a reviewed deliverable.")

if "agency_run" not in st.session_state:
    st.session_state.agency_run = None

with st.sidebar:
    st.header("Run controls")
    mode = st.radio("Agent mode", ["Demo", "OpenAI"], help="Demo is deterministic and needs no API key.")
    revisions = st.slider("Maximum QA revisions", 0, 3, 2)
    st.divider()
    st.caption("Generated artifacts are saved under `projects/software_dev_agency/runs/`.")

request = st.text_area(
    "What should the agency build?",
    value="Build a small utility that validates a user input and returns a clear processed result.",
    height=110,
)

run_clicked = st.button("🚀 Start agency run", type="primary", use_container_width=True)
if run_clicked:
    try:
        with st.spinner("Agents are collaborating..."):
            st.session_state.agency_run = SoftwareDevAgency(
                mode=mode,
                on_event=lambda event: None,
            ).run(request, max_revisions=revisions)
    except (RuntimeError, ValueError) as error:
        st.error(str(error))

run = st.session_state.agency_run
render_agent_cards(run)

if run:
    st.divider()
    status_label = "✅ QA passed" if run.qa_passed else "⚠️ Needs review"
    st.subheader(f"{status_label} · Run `{run.run_id}`")
    st.caption(
        f"Mode: {run.mode} · Revisions: {run.revision_count} · "
        f"Artifacts: `{run.artifact_dir}`"
    )
    tabs = st.tabs(["Workflow timeline", "PM brief", "Coder output", "QA report"])
    with tabs[0]:
        render_events(run.events)
    with tabs[1]:
        st.markdown(run.specification)
    with tabs[2]:
        st.code(run.implementation, language="python")
    with tabs[3]:
        st.markdown(run.test_report)
else:
    st.divider()
    st.subheader("Workflow timeline")
    render_events([])

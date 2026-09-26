from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.rag_evaluation.evaluate_rag import load_dataset, run_evaluation

st.set_page_config(
    page_title="RAG Evaluation Lab",
    page_icon="📚",
    layout="wide",
)

st.title("📚 RAG Evaluation Lab")
st.caption("Compare answer precision, recall, and evidence faithfulness.")

dataset = load_dataset()
st.metric("Golden examples", len(dataset))

with st.expander("Evaluation dataset", expanded=True):
    st.dataframe(
        [
            {
                "ID": row["id"],
                "Question": row["question"],
                "Reference answer": row["reference_answer"],
            }
            for row in dataset
        ],
        use_container_width=True,
        hide_index=True,
    )

limit = st.number_input(
    "Examples to evaluate",
    min_value=1,
    max_value=len(dataset),
    value=min(1, len(dataset)),
    step=1,
)

if st.button("Run evaluation", type="primary"):
    with st.spinner("Running the RAG evaluation..."):
        results = run_evaluation(limit=int(limit))
    st.success(f"Evaluated {len(results)} example(s).")
    st.dataframe(
        [
            {
                "ID": row["id"],
                "Precision": row["precision"],
                "Recall": row["recall"],
                "Faithfulness": row["faithfulness"],
                "Judge reason": row["faithfulness_reason"],
            }
            for row in results
        ],
        use_container_width=True,
        hide_index=True,
    )

"""Streamlit UI — chat with your data via the LangGraph agent."""

import streamlit as st
import pandas as pd
from pathlib import Path

from src.Graph.workflow import build_graph
from langchain_core.messages import HumanMessage

# ─── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Data Science Workflow",
    page_icon="🤖",
    layout="wide",
)

# ─── Session state defaults ───────────────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent = build_graph()
if "file_path" not in st.session_state:
    st.session_state.file_path = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ─── Helper ───────────────────────────────────────────────────
def _run_agent(query: str) -> None:
    """Invoke the graph, render the response, and update chat history."""
    st.session_state.chat_history.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Agents are working..."):
            result = st.session_state.agent.invoke(
                {
                    "messages": [HumanMessage(content=query)],
                    "file_path": st.session_state.file_path,
                    "task": query,
                    "llm_calls": 0,
                    # profiler
                    "dataset_profile": "",
                    "profiler_report": "",
                    # eda
                    "react_messages": [],
                    "react_iterations": 0,
                    "eda_report": "",
                    # planner
                    "plan": "",
                    "generated_plots": [],
                    "final_report": "",
                }
            )
            response = result["messages"][-1].content
            st.markdown(response)
            st.caption(f"🔄 LLM Calls: {result.get('llm_calls', 0)}")

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.rerun()


# ─── Title ────────────────────────────────────────────────────
st.title("🤖 Multi-Agent Data Science Workflow")
st.markdown("Upload your CSV and let AI agents analyze your data!")

# ─── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Data Upload")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = upload_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.file_path = str(file_path)
        st.success(f"✅ File uploaded: {uploaded_file.name}")

        with st.expander("📊 Data Preview"):
            df = pd.read_csv(file_path, nrows=5)
            st.dataframe(df)
            st.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns (showing first 5 rows)")

    st.divider()
    st.header("⚡ Quick Actions")

    _QUICK_ACTIONS = {
        "📈 EDA Summary": "Give me a summary of the columns and check for missing values.",
        "🧹 Data Quality Check": "Check data quality: missing values, duplicates, and outliers.",
        "🔍 Column Analysis": "Analyze all columns: data types, unique values, and distributions.",
    }

    for label, action_query in _QUICK_ACTIONS.items():
        if st.button(label):
            if st.session_state.file_path:
                st.session_state.quick_query = action_query
            else:
                st.warning("Please upload a file first!")

    st.divider()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# ─── Chat interface ───────────────────────────────────────────
st.header("💬 Chat with Your Data")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Quick-action trigger
if st.session_state.get("quick_query"):
    query = st.session_state.quick_query
    st.session_state.quick_query = None
    if st.session_state.file_path:
        _run_agent(query)

# Free-form chat input
if prompt := st.chat_input("Ask me anything about your data..."):
    if not st.session_state.file_path:
        st.warning("⚠️ Please upload a CSV file first!")
    else:
        _run_agent(prompt)

# ─── Footer ───────────────────────────────────────────────────
st.divider()
st.caption("🚀 Powered by LangGraph & LangChain")
"""Shared state definitions for the main orchestrator graph.

The orchestrator composes three ReAct sub-agents:
  profiler_agent → eda_agent → planner_agent

Each sub-agent has its own internal state; the fields below are the
*shared* surface that flows between them.
"""

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    """State carried through the main orchestrator graph.

    Fields
    ------
    messages : list[AnyMessage]
        Conversation history (managed by LangGraph's message reducer).
    file_path : str
        Path to the CSV file being analysed.
    task : str
        Free-form user objective forwarded to every agent.
    llm_calls : int
        Running total of LLM invocations (observability).

    Profiler outputs
    ----------------
    dataset_profile : str
        Raw JSON profile from dataset_profile_tool.
    profiler_report : str
        Structured Markdown report from the profiler agent.

    EDA outputs
    -----------
    eda_report : str
        Comprehensive EDA Markdown report.

    Planner + Plotting outputs
    --------------------------
    plan : str
        Strategic analysis plan (Markdown).
    generated_plots : list
        List of plot metadata dicts ({chart_type, path, message}).
    final_report : str
        Combined summary with plot references.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    file_path: str
    task: str
    llm_calls: int
    # profiler
    dataset_profile: str
    profiler_report: str
    # eda
    react_messages: list
    react_iterations: int
    eda_report: str
    # planner + plotting
    plan: str
    generated_plots: list
    final_report: str
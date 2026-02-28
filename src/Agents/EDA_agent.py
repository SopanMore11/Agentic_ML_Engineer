"""
EDA Agent — A ReAct-style agent that performs Exploratory Data Analysis.

It reasons step-by-step, uses tools to inspect the data, and produces
a comprehensive EDA report with statistical summaries and visualization
recommendations.

Flow:
    [reason] ──▶ has tool calls? ──▶ [act] ──▶ loop back to [reason]
        │                                        
        └── no tool calls ──▶ [finish] ──▶ END
"""

from langgraph.graph import StateGraph, END
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
    AIMessage,
    AnyMessage,
)
from typing import TypedDict, Annotated
import json

from langgraph.graph.message import add_messages
from src.services.llm_service import get_chat_model, bind_tools_to_model
from src.tools.file_tools import dataset_profile_tool


# ─────────────────────────── State ───────────────────────────
class EDAAgentState(TypedDict):
    """State schema carried across every node of the EDA agent."""

    messages: Annotated[list[AnyMessage], add_messages]  # outer conversation
    file_path: str
    task: str                 # what the user wants analysed
    react_messages: list      # internal reasoning chain
    react_iterations: int
    llm_calls: int
    eda_report: str           # final structured report


# ─────────────────────────── Config ──────────────────────────
MAX_ITERATIONS = 12

EDA_TOOLS = [dataset_profile_tool]
EDA_TOOLS_BY_NAME = {t.name: t for t in EDA_TOOLS}

llm = get_chat_model()
llm_with_tools = bind_tools_to_model(EDA_TOOLS)

SYSTEM_PROMPT = """\
You are a ReAct EDA Agent — an expert Data Analyst who performs
Exploratory Data Analysis by thinking step-by-step.

## Your Loop
1. **Thought** — Reason about what you know so far and what you still need.
2. **Action** — Call a tool to gather concrete data (never guess).
3. **Observation** — Read the tool result and update your understanding.
4. **Repeat** until you have enough evidence.
5. **Final Answer** — Write a comprehensive EDA report (see template below).

## EDA Report Template (use this for your final answer)

### 1. Dataset Overview
- Dimensions (rows × columns)
- Memory usage
- Column types breakdown

### 2. Statistical Summary
- Key descriptive statistics for numeric features
- Frequency tables for top categorical features
- Target variable distribution

### 3. Data Quality Assessment
| Issue | Affected Columns | Severity | Recommended Fix |
|-------|------------------|----------|-----------------|
| ... | ... | ... | ... |

### 4. Feature Insights
- Top correlated feature pairs
- High-cardinality categoricals
- Potential data leakage flags
- Zero / near-zero variance features

### 5. Univariate Highlights
- Skewed distributions worth transforming
- Outlier-prone columns (IQR method)

### 6. Bivariate Hypotheses
| # | Hypothesis | Features Involved | Suggested Chart |
|---|-----------|-------------------|-----------------|
| 1 | ... | ... | ... |

### 7. Recommended Next Steps
- Preprocessing actions (encoding, scaling, imputation)
- Modelling suggestions based on data structure

## Rules
- ALWAYS call tools before making claims about the data.
- Be precise with numbers — cite values from tool outputs.
- When you have enough information, respond WITHOUT tool calls to finish.
"""


# ─────────────────────────── Nodes ───────────────────────────
def reason(state: EDAAgentState) -> dict:
    """Think + optionally request a tool call."""
    task = state.get("task", "Perform a full EDA on the dataset.")
    react_msgs = state.get("react_messages") or []

    all_msgs = (
        [SystemMessage(content=SYSTEM_PROMPT)]
        + [HumanMessage(content=f"TASK: {task}\nFILE: {state.get('file_path', 'N/A')}")]
        + react_msgs
    )

    response = llm_with_tools.invoke(all_msgs)

    return {
        "react_messages": react_msgs + [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
        "react_iterations": state.get("react_iterations", 0) + 1,
    }


def act(state: EDAAgentState) -> dict:
    """Execute every tool the LLM requested."""
    last_msg = state["react_messages"][-1]
    tool_results: list[ToolMessage] = []

    for tool_call in last_msg.tool_calls:
        tool = EDA_TOOLS_BY_NAME[tool_call["name"]]
        args = dict(tool_call["args"])

        # Inject the real file path
        if "file_path" in args and state.get("file_path"):
            args["file_path"] = state["file_path"]

        observation = tool.invoke(args)

        # Normalise to string
        if isinstance(observation, dict):
            obs_str = json.dumps(observation, indent=2, default=str)
        elif hasattr(observation, "to_string"):
            obs_str = observation.to_string()
        else:
            obs_str = str(observation)

        tool_results.append(
            ToolMessage(content=obs_str, tool_call_id=tool_call["id"])
        )

    return {
        "react_messages": (state.get("react_messages") or []) + tool_results,
    }


def finish(state: EDAAgentState) -> dict:
    """Extract the final EDA report from the last AI message."""
    react_msgs = state.get("react_messages") or []

    eda_report = "EDA report could not be generated."
    for msg in reversed(react_msgs):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            eda_report = msg.content
            break

    # Push the report into the outer message list so the UI can read it
    return {
        "eda_report": eda_report,
        "messages": [AIMessage(content=eda_report)],
    }


# ─────────────────────────── Router ──────────────────────────
def should_continue(state: EDAAgentState) -> str:
    """Decide whether to call tools or wrap up."""
    react_msgs = state.get("react_messages") or []
    if not react_msgs:
        return "finish"

    last_msg = react_msgs[-1]
    iterations = state.get("react_iterations", 0)

    # Safety cap
    if iterations >= MAX_ITERATIONS:
        return "finish"

    # If the LLM wants to call tools, keep looping
    if getattr(last_msg, "tool_calls", None):
        return "act"

    return "finish"


# ─────────────────────────── Graph ───────────────────────────
def build_eda_agent():
    """Compile and return the EDA ReAct agent graph."""
    builder = StateGraph(EDAAgentState)

    builder.add_node("reason", reason)
    builder.add_node("act", act)
    builder.add_node("finish", finish)

    builder.set_entry_point("reason")

    builder.add_conditional_edges("reason", should_continue, {
        "act": "act",
        "finish": "finish",
    })
    builder.add_edge("act", "reason")   # Observe → Think again
    builder.add_edge("finish", END)

    return builder.compile()


# Singleton for easy import
eda_agent = build_eda_agent()

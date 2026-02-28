"""
Profiler Agent — A ReAct-style agent that builds a comprehensive
dataset profile with quality alerts and preprocessing directives.

It reasons step-by-step:
  1. Inspect columns and types.
  2. Run the full dataset_profile_tool.
  3. Identify target variable, quality issues, and technical directives.
  4. Emit a structured profile report for downstream agents.

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
from src.tools.plot_tools import list_columns


# ─────────────────────────── State ───────────────────────────
class ProfilerAgentState(TypedDict):
    """State schema for the Dataset Profiler agent."""

    messages: Annotated[list[AnyMessage], add_messages]  # outer conversation
    file_path: str
    task: str                   # user objective
    react_messages: list        # internal reasoning chain
    react_iterations: int
    llm_calls: int
    dataset_profile: str        # raw JSON profile from the tool
    profiler_report: str        # final structured Markdown report


# ─────────────────────────── Config ──────────────────────────
MAX_ITERATIONS = 10

PROFILER_TOOLS = [dataset_profile_tool, list_columns]
PROFILER_TOOLS_BY_NAME = {t.name: t for t in PROFILER_TOOLS}

llm = get_chat_model()
llm_with_tools = bind_tools_to_model(PROFILER_TOOLS)

SYSTEM_PROMPT = """\
You are a **Dataset Profiler Agent** — an expert Data Engineer who
creates a thorough, structured profile of any CSV dataset.

## Your Workflow
1. **Peek** — Call `list_columns` to get column names, dtypes, null
   counts, and sample values quickly.
2. **Profile** — Call `dataset_profile_tool` for the full statistical
   summary (shape, nulls, uniques, numeric min/max/mean, sample rows).
3. **Analyse** — Reason about the results: identify the likely target
   variable, flag quality issues, and decide preprocessing directives.
4. **Report** — When you have enough evidence, produce the final
   profile report (see template) WITHOUT any tool calls.

## Available Tools
| Tool | Purpose |
|------|---------|
| `list_columns` | Fast column overview (names, dtypes, nulls, uniques, samples) |
| `dataset_profile_tool` | Full statistical profile of the dataset |

## Profile Report Template (final answer)

### 1 · DATASET SNAPSHOT
- **Dimensions**: Rows × Columns
- **Memory estimate**: ~X MB
- **Column types**: N numeric, M categorical, K datetime, …

### 2 · TARGET VARIABLE
- **Name**: …
- **Type**: Numeric / Categorical
- **Rationale**: Why this is the target
- **Distribution**: Value counts or mean / range summary

### 3 · FEATURE REGISTRY
| Column | Logical Type | Null % | Uniques | Technical Directive |
|--------|-------------|--------|---------|---------------------|
| … | Numeric / Categorical / ID / Date | …% | … | One-Hot / Scale / Drop / Parse dates |

### 4 · DATA QUALITY ALERTS
| Severity | Issue | Affected Columns | Recommended Fix |
|----------|-------|-------------------|-----------------|
| 🔴 Critical | … | … | … |
| 🟡 Warning  | … | … | … |
| 🟢 Info     | … | … | … |

### 5 · PREPROCESSING CONTRACT
- **Encoding**: Detailed instructions for categorical handling
- **Scaling**: Which features need normalisation and method
- **Imputation**: Strategy per column (median / mode / drop / flag)
- **Feature engineering**: Suggested derived features

### 6 · MODELLING GUIDANCE
- **Recommended algorithms**: (Tree-based / Linear / …) and why
- **Train-test split notes**: Stratification, time-based, etc.
- **Class balance**: Balanced / imbalanced → suggested remedy

## Rules
- ALWAYS call tools before making claims — never guess numbers.
- Cite exact values from tool outputs.
- When done, respond WITHOUT tool calls.
"""


# ─────────────────────────── Nodes ───────────────────────────
def reason(state: ProfilerAgentState) -> dict:
    """Think + optionally request a tool call."""
    task = state.get("task", "Profile this dataset thoroughly.")
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


def act(state: ProfilerAgentState) -> dict:
    """Execute every tool the LLM requested."""
    last_msg = state["react_messages"][-1]
    tool_results: list[ToolMessage] = []
    dataset_profile = state.get("dataset_profile", "")

    for tool_call in last_msg.tool_calls:
        tool = PROFILER_TOOLS_BY_NAME[tool_call["name"]]
        args = dict(tool_call["args"])

        # Inject the real file path
        if "file_path" in args and state.get("file_path"):
            args["file_path"] = state["file_path"]

        observation = tool.invoke(args)

        # Normalise to string
        if isinstance(observation, dict):
            obs_str = json.dumps(observation, indent=2, default=str)
        else:
            obs_str = str(observation)

        tool_results.append(
            ToolMessage(content=obs_str, tool_call_id=tool_call["id"])
        )

        # Capture the full profile when dataset_profile_tool runs
        if tool_call["name"] == "dataset_profile_tool":
            dataset_profile = obs_str

    return {
        "react_messages": (state.get("react_messages") or []) + tool_results,
        "dataset_profile": dataset_profile,
    }


def finish(state: ProfilerAgentState) -> dict:
    """Extract the final profiler report from the last AI message."""
    react_msgs = state.get("react_messages") or []

    profiler_report = "Profiler report could not be generated."
    for msg in reversed(react_msgs):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            profiler_report = msg.content
            break

    return {
        "profiler_report": profiler_report,
        "messages": [AIMessage(content=profiler_report)],
    }


# ─────────────────────────── Router ──────────────────────────
def should_continue(state: ProfilerAgentState) -> str:
    """Decide whether to keep acting or wrap up."""
    react_msgs = state.get("react_messages") or []
    if not react_msgs:
        return "finish"

    last_msg = react_msgs[-1]
    iterations = state.get("react_iterations", 0)

    if iterations >= MAX_ITERATIONS:
        return "finish"

    if getattr(last_msg, "tool_calls", None):
        return "act"

    return "finish"


# ─────────────────────────── Graph ───────────────────────────
def build_profiler_agent():
    """Compile and return the Dataset Profiler ReAct agent graph."""
    builder = StateGraph(ProfilerAgentState)

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
profiler_agent = build_profiler_agent()

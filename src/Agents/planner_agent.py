"""
Planner + Plotting Agent — A ReAct-style agent that builds an analysis
strategy, then generates concrete visualisations.

It reasons step-by-step:
  1. Inspect the dataset (profile + column listing).
  2. Formulate business hypotheses and choose chart types.
  3. Generate charts one-by-one via the ``generate_plot`` tool.
  4. Summarise findings with plot paths.

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
from src.tools.plot_tools import generate_plot, list_columns


# ─────────────────────────── State ───────────────────────────
class PlannerAgentState(TypedDict):
    """State schema for the Planner + Plotting agent."""

    messages: Annotated[list[AnyMessage], add_messages]  # outer conversation
    file_path: str
    task: str                  # user objective
    eda_report: str            # optional upstream EDA report for context
    react_messages: list       # internal reasoning chain
    react_iterations: int
    llm_calls: int
    plan: str                  # strategic analysis plan (Markdown)
    generated_plots: list      # list of {chart_type, path, message}
    final_report: str          # combined summary + plot references


# ─────────────────────────── Config ──────────────────────────
MAX_ITERATIONS = 20  # plotting may need several rounds

PLANNER_TOOLS = [dataset_profile_tool, list_columns, generate_plot]
PLANNER_TOOLS_BY_NAME = {t.name: t for t in PLANNER_TOOLS}

llm = get_chat_model()
llm_with_tools = bind_tools_to_model(PLANNER_TOOLS)

SYSTEM_PROMPT = """\
You are a **Planner + Plotting Agent** — a Senior Data Scientist who
creates a strategic analysis plan and then generates publication-quality
visualisations to support each hypothesis.

## Your Workflow
1. **Inspect** — Use `list_columns` and/or `dataset_profile_tool` to
   understand the dataset's columns, types, and distributions.
2. **Plan** — Formulate 5–7 business hypotheses and map each one to a
   specific chart type and the relevant columns.
3. **Plot** — Call `generate_plot` for every planned chart.  After each
   chart, note whether it succeeded and what insight it reveals.
4. **Summarise** — When all plots are done, write a final report that:
   - Lists every hypothesis
   - References the saved plot path for each
   - Highlights the most important findings

## Available Tools
| Tool | Purpose |
|------|---------|
| `dataset_profile_tool` | Full statistical profile of the CSV |
| `list_columns` | Quick column names, dtypes, nulls, uniques |
| `generate_plot` | Render & save a chart (histogram, scatter, box, violin, bar, line, heatmap, countplot, pairplot) |

## Chart Selection Guide
- Numeric distribution → **histogram**
- Numeric vs Numeric → **scatter**
- Categorical vs Numeric → **box** or **violin**
- Categorical counts → **countplot**
- Time series → **line**
- Correlations overview → **heatmap**
- Multi-feature overview → **pairplot**

## Output Format (Final Answer — no tool calls)

### Strategic Analysis Plan
| # | Hypothesis | X Column | Y Column | Chart Type |
|---|-----------|----------|----------|------------|
| 1 | … | … | … | … |

### Generated Visualisations
| # | Chart | Insight | Path |
|---|-------|---------|------|
| 1 | … | … | `outputs/plots/…` |

### Key Findings
1. …
2. …

### Recommended Next Steps
- …

## Rules
- ALWAYS inspect the data before planning charts.
- Call `generate_plot` once per chart — do NOT batch.
- Use the exact column names returned by tools.
- After all plots are generated, respond WITHOUT tool calls to finish.
"""


# ─────────────────────────── Nodes ───────────────────────────
def reason(state: PlannerAgentState) -> dict:
    """Think, plan, or request a tool call."""
    task = state.get("task", "Create an analysis plan and generate visualisations.")
    react_msgs = state.get("react_messages") or []
    plots_so_far = len(state.get("generated_plots") or [])

    context_parts = [
        f"TASK: {task}",
        f"FILE: {state.get('file_path', 'N/A')}",
        f"PLOTS GENERATED SO FAR: {plots_so_far}",
    ]
    if plots_so_far == 0:
        context_parts.append(
            "⚠️  You have NOT generated any plots yet. "
            "Start by calling `list_columns`, then call `generate_plot` for each hypothesis."
        )
    if state.get("eda_report"):
        context_parts.append(f"EDA REPORT (from upstream agent):\n{state['eda_report']}")

    all_msgs = (
        [SystemMessage(content=SYSTEM_PROMPT)]
        + [HumanMessage(content="\n\n".join(context_parts))]
        + react_msgs
    )

    response = llm_with_tools.invoke(all_msgs)

    return {
        "react_messages": react_msgs + [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
        "react_iterations": state.get("react_iterations", 0) + 1,
    }


def act(state: PlannerAgentState) -> dict:
    """Execute every tool the LLM requested and track plot outputs."""
    last_msg = state["react_messages"][-1]
    tool_results: list[ToolMessage] = []
    new_plots: list[dict] = list(state.get("generated_plots") or [])

    for tool_call in last_msg.tool_calls:
        tool = PLANNER_TOOLS_BY_NAME[tool_call["name"]]
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

        # Track successful plots
        if tool_call["name"] == "generate_plot":
            try:
                plot_info = json.loads(obs_str)
                if plot_info.get("status") == "success":
                    new_plots.append(plot_info)
            except json.JSONDecodeError:
                pass

    return {
        "react_messages": (state.get("react_messages") or []) + tool_results,
        "generated_plots": new_plots,
    }


def finish(state: PlannerAgentState) -> dict:
    """Extract the final report from the last AI message."""
    react_msgs = state.get("react_messages") or []

    final_report = "Planner report could not be generated."
    for msg in reversed(react_msgs):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            final_report = msg.content
            break

    # Append plot manifest to the report
    plots = state.get("generated_plots") or []
    if plots:
        plot_summary = "\n\n---\n**📊 Plot Manifest**\n"
        for i, p in enumerate(plots, 1):
            plot_summary += f"{i}. `{p.get('chart_type', '?')}` → `{p.get('path', '?')}`\n"
        final_report += plot_summary

    return {
        "final_report": final_report,
        "messages": [AIMessage(content=final_report)],
    }


# ─────────────────────────── Router ──────────────────────────
def should_continue(state: PlannerAgentState) -> str:
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
def build_planner_agent():
    """Compile and return the Planner + Plotting ReAct agent graph."""
    builder = StateGraph(PlannerAgentState)

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
planner_agent = build_planner_agent()

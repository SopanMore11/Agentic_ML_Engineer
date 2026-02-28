"""LangGraph workflow: build, stream, and visualise the orchestrator graph.

The orchestrator composes three ReAct sub-agents as nodes:
    profiler_agent → eda_agent → planner_agent
"""

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

from src.Graph.nodes import profiler_agent, eda_agent, planner_agent
from src.Graph.state import AgentState
from src.utils.logger import logger


# ─── Graph construction ───────────────────────────────────────
def build_graph():
    """Compile and return the main orchestrator StateGraph.

    Pipeline:
        profiler_agent → eda_agent → planner_agent → END
    """
    builder = StateGraph(AgentState)

    builder.add_node("profiler", profiler_agent)
    builder.add_node("eda", eda_agent)
    builder.add_node("planner", planner_agent)

    builder.add_edge(START, "profiler")
    builder.add_edge("profiler", "eda")
    builder.add_edge("eda", "planner")
    builder.add_edge("planner", END)

    return builder.compile()


# ─── Default inputs ───────────────────────────────────────────
def _default_inputs(query: str, file_path: str) -> dict:
    """Build the initial state dict for a workflow run."""
    return {
        "messages": [HumanMessage(content=query)],
        "file_path": file_path,
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


# ─── Streaming helper ─────────────────────────────────────────
def run_workflow_with_streaming(query: str, file_path: str) -> dict:
    """Run the workflow with streaming updates and return the final state."""
    agent = build_graph()
    save_graph_image(agent)
    inputs = _default_inputs(query, file_path)

    logger.info("=" * 80)
    logger.info(f"Starting workflow for: {file_path}")
    logger.info("=" * 80)

    for event in agent.stream(inputs, stream_mode="updates"):
        for node_name, node_output in event.items():
            logger.info(f"Node completed: {node_name}")

            if node_output.get("profiler_report"):
                print("\n" + "=" * 30 + " PROFILER REPORT " + "=" * 30)
                print(node_output["profiler_report"][:500] + "…")
                print("=" * 77 + "\n")

            if node_output.get("eda_report"):
                print("\n" + "=" * 30 + " EDA REPORT " + "=" * 30)
                print(node_output["eda_report"][:500] + "…")
                print("=" * 77 + "\n")

            if node_output.get("final_report"):
                print("\n" + "=" * 30 + " PLANNER REPORT " + "=" * 30)
                print(node_output["final_report"][:500] + "…")
                print("=" * 77 + "\n")

    final_result = agent.invoke(inputs)
    logger.info("WORKFLOW COMPLETE")
    return final_result


# ─── Visualisation ────────────────────────────────────────────
def save_graph_image(agent, filename: str = "eda_workflow.jpg") -> None:
    """Save the compiled graph as a Mermaid PNG image."""
    try:
        image_data = agent.get_graph().draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(image_data)
        logger.info(f"Graph image saved as {filename}")
    except Exception as e:
        logger.error(f"Could not save graph image: {e}")


# ─── CLI entry-point ──────────────────────────────────────────
if __name__ == "__main__":
    result = run_workflow_with_streaming(
        query="Give me a summary of the columns and entire dataset.",
        file_path="data/data.csv",
    )
    print("\n" + "=" * 80)
    print("FINAL ANSWER:")
    print("=" * 80)
    print(result["messages"][-1].content)
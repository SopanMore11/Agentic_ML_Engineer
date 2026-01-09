from src.Graph.nodes import tool_node, should_continue, llm_call, planning_node, designer_node
from langgraph.graph import StateGraph, START, END
from src.Graph.state import AgentState
from langchain_core.messages import HumanMessage
from src.utils.logger import logger
from dotenv import load_dotenv
import os

load_dotenv()
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGSMITH_TRACING'] = os.getenv('LANGSMITH_TRACING')
os.environ['LANGSMITH_PROJECT'] = os.getenv('LANGSMITH_PROJECT')
os.environ['LANGSMITH_ENDPOINT'] = os.getenv('LANGSMITH_ENDPOINT')

def build_graph():
    builder = StateGraph(AgentState)
    
    builder.add_node("profiler", llm_call)
    builder.add_node("tool_node", tool_node)
    builder.add_node("planner", planning_node)  # The strategy brain
    builder.add_node("designer", designer_node) # The JSON spec writer
    
    builder.add_edge(START, "profiler")
    
    # Explicit mapping prevents the KeyError and handles transitions
    builder.add_conditional_edges(
        "profiler",
        should_continue,
        {
            "tool_node": "tool_node",
            "planner": "planner",  # Transition to planner when profiling is done
        }
    )
    
    builder.add_edge("tool_node", "profiler")
    builder.add_edge("planner", "designer")
    builder.add_edge("designer", END)
    
    return builder.compile()




def run_workflow_with_streaming(query: str, file_path: str):
    """Run workflow with streaming updates and display specific state outputs"""
    
    agent = build_graph()
    save_graph_image(agent, "eda_workflow.jpg")
    
    inputs = {
        "messages": [HumanMessage(content=query)],
        "file_path": file_path,
        "llm_calls": 0,
        "dataset_profile": "", # Initialize for clarity
        "strategy": "",
        "plot_plan": {}
    }
    
    logger.info("=" * 80)
    logger.info(f"Starting workflow for: {file_path}")
    logger.info("=" * 80)
    
    # Stream events to observe state changes
    final_state = inputs
    for event in agent.stream(inputs, stream_mode="updates"):
        for node_name, node_output in event.items():
            logger.info(f"\nNode Completed: {node_name}")
            
            # Print the Profile if the profiler/tool_node just updated it
            if "dataset_profile" in node_output and node_output["dataset_profile"]:
                print("\n" + "=" * 30 + " DATASET PROFILE " + "=" * 30)
                print(node_output["dataset_profile"])
                print("=" * 77 + "\n")

            # Print the Plot Plan if the designer node just finished
            if "plot_plan" in node_output and node_output["plot_plan"]:
                import json
                print("\n" + "=" * 30 + " FINAL PLOT JSON " + "=" * 30)
                print(json.dumps(node_output["plot_plan"], indent=2))
                print("=" * 77 + "\n")
        
        # Keep track of the full state
        final_state.update(event)
    
    # Final result invocation to get the absolute last state
    final_result = agent.invoke(inputs)
    
    logger.info("WORKFLOW COMPLETE")
    return final_result

def save_graph_image(agent, filename="workflow_graph.jpg"):
    """Saves the LangGraph workflow visualization as an image"""
    try:
        # Generate the PNG/JPG data from the mermaid graph
        image_data = agent.get_graph().draw_mermaid_png()
        
        with open(filename, "wb") as f:
            f.write(image_data)
        
        logger.info(f"Graph image saved successfully as {filename}")
    except Exception as e:
        logger.error(f"Could not save graph image: {e}")

if __name__ == "__main__":
    result = run_workflow_with_streaming(
        query="Give me a summary of the columns and entire dataset.",
        file_path="data/data.csv"
    )
    
    print("\n" + "=" * 80)
    print("FINAL ANSWER:")
    print("=" * 80)
    print(result['messages'][-1].content)
    print(result)
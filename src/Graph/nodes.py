from src.Agents.agent_manager import agent_manager
from src.Graph.state import AgentState
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from langgraph.graph import END
from typing import Literal

def llm_call(state: AgentState):
    """Node that calls the appropriate agent"""
    # For now, we always use EDA agent
    # Later you can add logic to select agent based on state
    agent = agent_manager.get_agent("eda")
    return agent.execute_profiler(state)

def tool_node(state: AgentState):
    """Node that executes tools"""
    agent = agent_manager.get_agent("eda")
    result = []
    
    for tool_call in state["messages"][-1].tool_calls:
        tool = agent.tools_by_name[tool_call["name"]]
        
        args = tool_call["args"]
        if "file_path" in args and state.get("file_path"):
            args["file_path"] = state["file_path"]
        
        observation = tool.invoke(args)
        observation_str = agent._format_observation(observation)
        
        result.append(
            ToolMessage(content=observation_str, tool_call_id=tool_call["id"])
        )
    
    # Extract dataset_profile from the tool results for downstream nodes
    # Assuming the first/only tool is dataset_profile_tool
    dataset_profile_content = result[0].content if result else ""
    
    return {
        "messages": result,
        "dataset_profile": dataset_profile_content
    }


def planning_node(state: AgentState):
    """Business logic node: Identifies domain and hypotheses."""
    profile = state.get("dataset_profile", "")
    PLANNER_PROMPT = """You are a Senior Data Scientist. 
        Analyze the Dataset Profile provided and create a Strategic EDA Plan.
        
        1. Identify the Business Domain (e.g., Finance, Marketing, Operations).
        2. Define the 'Target Variable' and why it matters to a stakeholder.
        3. List 5-7 Business Hypotheses to test (e.g., 'Do higher interest rates lead to more defaults?').
        4. Outline a visualization strategy: Which segments should we compare? Which trends matter?
        
        Output your plan in clear Markdown. Focus on 'Why', not 'How'."""
    
    # This prompt is generic because it asks the LLM to identify the domain
    response = agent_manager.llm.invoke([
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=f"Dataset Profile:\n{profile}")
    ])
    
    return {
        "strategy": response.content,
        "llm_calls": state.get("llm_calls", 0) + 1
    }

def designer_node(state: AgentState):
    """Technical logic node: Converts strategy to JSON."""
    agent = agent_manager.get_agent("eda")
    return agent.execute_designer(state)

# src/Graph/nodes.py

# def should_continue(state: AgentState):
#     last_message = state["messages"][-1]
    
#     # If the LLM still wants to call tools (profiling), stay in the loop
#     if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
#         return "tool_node"
    
#     # Instead of END, move to the next logical stage of your pipeline

#     return "planner"

def should_continue(state: AgentState):
    last_message = state["messages"][-1]

    # If the LLM still wants to call tools, stay in the tool loop
    if getattr(last_message, "tool_calls", None):
        return "tool_node"

    # Profiling complete, move to planning stage
    return "planner"

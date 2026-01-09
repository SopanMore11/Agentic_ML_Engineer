from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from src.tools.file_tools import load_dataset, dataset_profile_tool

class EDAAgent:
    """Agent responsible for Exploratory Data Analysis"""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.tools = [dataset_profile_tool]
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.model_with_tools = llm.bind_tools(self.tools)
        
    def get_profile_prompt(self, file_path: str) -> str:
        """Generate system prompt for EDA agent"""
        return f"""You are an expert Data Analyst specializing in Exploratory Data Analysis (EDA).

            Your task is to analyze the dataset at: {file_path}

            Your responsibilities:
            1. DATASET_SNAPSHOT
                - **Dimensions**: [Rows x Columns]
                - **Target_Variable**: [Name | Type | Logic for choosing it as target]
                - **Target_Distribution**: [Value counts or mean/range summary]
            2. FEATURE_REGISTRY
                | Column Name | Logical Type | Data Stats (Null% / Uniques) | Technical Directive |
                | :--- | :--- | :--- | :--- |
                | [Name] | [Numeric/Categorical/ID] | [e.g., 0% Null / 5 Unique] | [e.g., "One-Hot Encode", "MinMax Scale", "Drop"] |
            3. DATA_QUALITY_ALERTS
                - **Critical Issues**: [List any blockers like high nullity, zero variance, or extreme class imbalance]
                - **Preprocessing Requirements**: [Explicit steps for the next agent to follow]
            4. AGENT_INSTRUCTIONS (TECHNICAL CONTRACT)
                - **Recommended_Encoding**: [Detailed instruction for handling categories]
                - **Recommended_Scaling**: [Which features need normalization and why]
                - **Modeling_Approach**: [Suggested algorithms (e.g., Tree-based, Linear) based on the data structure]
            5. Provide actionable insights

            Available tools:
            - dataset_profile_tool: Get detailed statistical profile of the dataset

            Always use tools to gather data before making conclusions.
            Provide clear, structured summaries that are easy to understand.
        """

    def get_designer_prompt(self, strategy: str, profile: str) -> str:
        """Generic prompt to convert strategy into a PlotPlan JSON"""
        return f"""You are a Visualization Expert. Create a PlotPlan JSON.
        
            STRATEGY: {strategy}
            DATASET PROFILE: {profile}

            GENERIC MAPPING RULES:
            - CATEGORICAL vs TARGET: Use 'box' or 'violin'.
            - NUMERIC vs TARGET: Use 'scatter' (set sample_rows if N > 50,000).
            - TEMPORAL: Use 'time_series' with 'date_freq=W' or 'M'.
            - CLASS IMBALANCE: Use 'stacked_bar' with 'normalize=percent'.

            Return ONLY raw JSON following the PlotPlan schema.
        """

    def execute_designer(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Specific logic to convert Strategy + Profile into PlotPlan JSON"""
        strategy = state.get("strategy", "")
        profile = state.get("dataset_profile", "")
        
        # Generic Designer Prompt: Focuses on mapping rules, not domain logic
        designer_prompt = f"""You are a Visualization Expert.
        Convert the following Strategy into a PlotPlan JSON.
        
        STRATEGY: {strategy}
        PROFILE: {profile}

        MAPPING RULES:
        - If relationship is Numeric vs Numeric -> scatter.
        - If relationship is Categorical vs Numeric -> box/violin.
        - If time is involved -> time_series (freq=W/M).
        """
        
        response = self.llm.invoke([SystemMessage(content=designer_prompt)])
        # In a real scenario, you'd parse JSON and validate with Pydantic here
        return {"plot_plan": response.content, "llm_calls": state.get("llm_calls", 0) + 1}

    def execute_profiler(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute EDA agent logic"""
        
        # Create system message with context
        system_msg = SystemMessage(content=self.get_profile_prompt(state['file_path']))
        
        # Invoke LLM with tools
        response = self.model_with_tools.invoke([system_msg] + state["messages"])
        
        return {
            "messages": [response],  # Add AI message to conversation so tool_calls are visible
            "llm_calls": state.get('llm_calls', 0) + 1
        }
    

    
    def _format_observation(self, observation: Any) -> str:
        """Format tool observation for LLM consumption"""
        import json
        
        if hasattr(observation, 'to_string'):  # DataFrame
            return f"Dataset loaded. Shape: {observation.shape}. Use profiling tools for details."
        elif isinstance(observation, dict):
            return json.dumps(observation, indent=2)
        else:
            return str(observation)
    
    def should_continue(self, state: Dict[str, Any]) -> bool:
        """Determine if agent should continue processing"""
        last_message = state["messages"][-1]
        return hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0
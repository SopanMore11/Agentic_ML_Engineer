from abc import ABC, abstractmethod
from typing import Dict, Any, List
from langchain_core.language_models import BaseChatModel

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.tools = []
        self.tools_by_name = {}
        self.model_with_tools = None
    
    @abstractmethod
    def get_profile_prompt(self, **kwargs) -> str:
        """Get agent-specific system prompt"""
        pass
    
    @abstractmethod
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent logic"""
        pass
    
    def _format_observation(self, observation: Any) -> str:
        """Common observation formatting"""
        import json
        
        if hasattr(observation, 'to_string'):
            return f"Data loaded. Shape: {observation.shape}"
        elif isinstance(observation, dict):
            return json.dumps(observation, indent=2)
        return str(observation)
from typing import Dict, Type
from src.Agents.base_agent import BaseAgent
from src.Agents.EDA_agent import EDAAgent
from src.services.llm_service import get_chat_model

class AgentManager:
    """Manages all agents in the workflow"""
    
    def __init__(self):
        self.llm = get_chat_model()
        self._agents: Dict[str, BaseAgent] = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all available agents"""
        self._agents = {
            "eda": EDAAgent(self.llm),
            # Add more agents here as you build them
            # "cleaning": CleaningAgent(self.llm),
            # "preprocessing": PreprocessingAgent(self.llm),
        }
    
    def get_agent(self, agent_name: str) -> BaseAgent:
        """Get agent by name"""
        if agent_name not in self._agents:
            raise ValueError(f"Agent '{agent_name}' not found")
        return self._agents[agent_name]
    
    def list_agents(self):
        """List all available agents"""
        return list(self._agents.keys())

# Singleton instance
agent_manager = AgentManager()
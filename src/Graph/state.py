from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator

# 1. State Definition
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    file_path: str
    llm_calls: int
    dataset_profile: str
    strategy: str
    plot_plan: dict
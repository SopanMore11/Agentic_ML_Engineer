from dotenv import load_dotenv
import os
from typing import Iterable, Optional

from langchain.chat_models import init_chat_model
from langchain_core.tools import BaseTool


def _configure_env() -> None:
    """
    Load environment variables from .env and configure process env for Azure OpenAI.

    Expected variables:
    - AZURE_OPENAI_API_KEY
    - OPENAI_API_VERSION
    - AZURE_OPENAI_ENDPOINT
    """
    load_dotenv()
    os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY", "")
    os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION", "")
    os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT", "")


def get_chat_model(temperature: float = 0.0):
    """
    Initialize and return the Azure OpenAI chat model used across the app.

    Parameters
    ----------
    temperature : float, default 0.0
        Sampling temperature for the model.

    Returns
    -------
    langchain_openai.chat_models.azure.AzureChatOpenAI
        Configured chat model instance.
    """
    _configure_env()
    return init_chat_model(
        "gpt-5-chat",
        model_provider="azure_openai",
        temperature=temperature,
    )


def bind_tools_to_model(
    tools: Optional[Iterable[BaseTool]] = None,
    *,
    temperature: float = 0.0,
):
    """
    Convenience helper to create a chat model and bind provided tools.

    Parameters
    ----------
    tools : Optional[Iterable[BaseTool]]
        Iterable of LangChain tools to bind. If None or empty, returns the model as-is.
    temperature : float, default 0.0
        Sampling temperature passed to the model initializer.

    Returns
    -------
    langchain_openai.chat_models.azure.AzureChatOpenAI
        Chat model instance, optionally bound with tools.
    """
    model = get_chat_model(temperature=temperature)
    if tools:
        return model.bind_tools(list(tools))
    return model
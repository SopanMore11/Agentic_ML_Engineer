## Agentic ML Engineer

An agentic, LangGraph-powered Exploratory Data Analysis (EDA) workflow with a Streamlit UI. Upload a CSV, chat with your data, and let specialized agents profile the dataset, plan analysis, and generate a plot specification JSON for downstream visualization.

## What’s implemented so far

- Multi-node LangGraph workflow compiled from `src/Graph/workflow.py` and `src/Graph/nodes.py`:
	- profiler: LLM call that prepares system context and requests tools
	- tool_node: executes `dataset_profile_tool` and feeds observations back to the LLM
	- planner: transforms the dataset profile into a strategy/hypotheses summary
	- designer: converts the strategy and profile into a PlotPlan JSON
- EDA Agent (`src/Agents/EDA_agent.py`):
	- Binds tools to LLM and formats tool observations
	- Generates profile prompts and designer prompts
	- Produces `plot_plan` (JSON-like content) and tracks `llm_calls`
- Streamlit App (`src/streamlit_app.py`):
	- CSV upload and file management
	- Chat interface with quick actions (EDA Summary, Data Quality Check, Column Analysis)
	- Invokes the compiled graph with `HumanMessage`
	- Displays assistant responses and LLM call count
- Tools (`src/tools/file_tools.py`): dataset loading and profiling exposed as tools for the agent
- State model (`src/Graph/state.py`): TypedDict state tracked across nodes: messages, file_path, llm_calls, dataset_profile, strategy, plot_plan
- Logging and workflow streaming in `src/Graph/workflow.py` with `eda_workflow.jpg` saved from graph visualization
- Notebooks for experimentation (`notebooks/test_agents.ipynb`, `notebooks/langgraph_agents.ipynb`) and generated EDA plots in `notebooks/eda_plots` and `notebooks/eda_plots_loan`
- Data folder added to `.gitignore` to keep large/sensitive files out of version control

## Repository structure

- `src/Agents/` — Agent implementations and manager
- `src/Graph/` — LangGraph nodes, state, and workflow orchestration
- `src/services/` — LLM model service (`get_chat_model`) and environment wiring
- `src/tools/` — File and dataset profiling tools
- `src/streamlit_app.py` — Streamlit UI to chat with your data
- `data/` — Local datasets (ignored by Git)
- `notebooks/` — Experiments and sample outputs
- `logs/` — Workflow execution logs
- `eda_workflow.jpg` — Graph visualization of the pipeline
- `pyproject.toml` — Dependencies pinned for reproducibility

## Architecture snapshot

Data flow:
1. User uploads/selects CSV in Streamlit and sends a question.
2. Graph starts at `profiler` (LLM call) with a system prompt tailored to the file.
3. If LLM requests a tool, `tool_node` runs `dataset_profile_tool` (and can inject `file_path`).
4. `planner` summarizes domain, hypotheses, and analysis strategy from the profile.
5. `designer` emits a `plot_plan` JSON suitable for generating EDA visualizations.
6. Streamlit displays the agent’s response and tracks `llm_calls`.

Key files:
- `src/Graph/workflow.py`: `build_graph()`, `run_workflow_with_streaming()`, `save_graph_image()`
- `src/Graph/nodes.py`: `llm_call`, `tool_node`, `planning_node`, `designer_node`, `should_continue`
- `src/Agents/EDA_agent.py`: EDA prompts, tool binding, observation formatting, designer/profiler execution

## Setup

Prerequisites:
- Python 3.13+
- API keys for your chosen LLM provider (e.g., OpenAI/Azure OpenAI), supplied via environment variables

Install dependencies:

```powershell
uv sync
```

Environment variables (examples):
- `OPENAI_API_KEY` or Azure-specific envs
- LangSmith (optional, for tracing): `LANGSMITH_API_KEY`, `LANGSMITH_TRACING`, `LANGSMITH_PROJECT`, `LANGSMITH_ENDPOINT`

## Run

Streamlit app:

```powershell
uv run streamlit run src/streamlit_app.py
```

CLI smoke test (workflow streaming):

```powershell
uv run python -m src.Graph.workflow
```

## Usage in the app

1. Upload a CSV in the sidebar.
2. Use quick-action buttons for common EDA tasks or type a question.
3. The agent profiles the dataset using tools, plans analysis, and surfaces insights in chat.
4. Plot plan JSON can be used downstream to render charts.

## Current progress (as of Jan 14, 2026)

- End-to-end EDA agent workflow compiled and visualized.
- Tool execution loop wired with automatic `file_path` injection.
- Dataset profiling integrated; profile content passed between nodes.
- Strategy and designer stages produce structured outputs (`strategy`, `plot_plan`).
- Streamlit UI functional with chat history and quick queries.
- Logs captured per run under `logs/` and graph image generated at repo root.
- `.gitignore` includes `data/` (and typical Python artifacts) to avoid committing datasets.

## Data & privacy

- `data/` is ignored by Git to prevent leaking large or sensitive files.
- Uploaded files (via Streamlit) are kept local in `data/uploads/`.
- Be mindful of including PII; profiling only summarizes structure and stats.

## Notebooks

- `notebooks/test_agents.ipynb` shows the evolution of the tool loop and state handling.
- `notebooks/langgraph_agents.ipynb` explores plotting and plan generation.
- Generated plots reside under `notebooks/eda_plots` and `notebooks/eda_plots_loan`.

## Roadmap

- Robust PlotPlan schema with Pydantic validation and chart renderer.
- More tools: missing value imputer, outlier detector, correlation, feature importance.
- Persisted session state and export of EDA report (Markdown/PDF).
- Expanded domain strategies (time series, classification/regression guidance).
- Cloud-friendly storage and optional remote tracing.

## Troubleshooting

- Ensure environment variables are set for your LLM provider.
- If datasets fail to load, check file encoding and separators in `src/tools/file_tools.py`.
- For large files, consider sampling or chunked profiling.

## License

This repository is for educational and experimental use. Add your license of choice if you plan to distribute.
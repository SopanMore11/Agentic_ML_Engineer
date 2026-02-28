"""Entry point — run the multi-agent workflow from the command line."""

from src.Graph.workflow import run_workflow_with_streaming


def main():
    result = run_workflow_with_streaming(
        query="Give me a full analysis of this dataset.",
        file_path="data/data.csv",
    )

    print("\n" + "=" * 80)
    print("FINAL ANSWER:")
    print("=" * 80)
    print(result["messages"][-1].content)

    print(f"\n🔄 Total LLM Calls: {result.get('llm_calls', 0)}")
    print(f"📊 Plots generated: {len(result.get('generated_plots', []))}")
    for p in result.get("generated_plots", []):
        print(f"   → {p.get('path')}")


if __name__ == "__main__":
    main()

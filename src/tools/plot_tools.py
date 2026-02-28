"""LangChain tools for generating and saving matplotlib / seaborn plots."""

from langchain.tools import tool
import pandas as pd
import matplotlib
matplotlib.use("Agg")                        # headless backend for servers
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


# ─── Defaults ─────────────────────────────────────────────────
PLOT_DIR = Path("outputs/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


@tool
def generate_plot(
    file_path: str,
    chart_type: str,
    x_column: str,
    y_column: str = "",
    hue_column: str = "",
    title: str = "",
    figsize_w: float = 10.0,
    figsize_h: float = 6.0,
) -> str:
    """Generate a single chart from a CSV and save it as a PNG.

    Parameters
    ----------
    file_path : str
        Path to the CSV dataset.
    chart_type : str
        One of: histogram, scatter, box, violin, bar, line, heatmap, countplot, pairplot.
    x_column : str
        Column to use on the x-axis (or the single column for histogram / countplot).
    y_column : str
        Column for the y-axis (leave empty for univariate charts).
    hue_column : str
        Optional column for colour grouping.
    title : str
        Chart title (auto-generated if empty).
    figsize_w : float
        Figure width in inches (default 10).
    figsize_h : float
        Figure height in inches (default 6).

    Returns
    -------
    str
        JSON with keys: ``status``, ``chart_type``, ``path``, ``message``.
    """
    df = pd.read_csv(file_path)

    # Validate columns exist
    for col_name, col_val in [("x_column", x_column), ("y_column", y_column), ("hue_column", hue_column)]:
        if col_val and col_val not in df.columns:
            return json.dumps({
                "status": "error",
                "message": f"Column '{col_val}' not found. Available: {list(df.columns)}",
            })

    hue = hue_column if hue_column else None
    chart_type = chart_type.strip().lower()

    fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))

    try:
        if chart_type == "histogram":
            sns.histplot(data=df, x=x_column, hue=hue, kde=True, ax=ax)

        elif chart_type == "scatter":
            sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue, ax=ax, alpha=0.6)

        elif chart_type == "box":
            sns.boxplot(data=df, x=x_column, y=y_column, hue=hue, ax=ax)

        elif chart_type == "violin":
            sns.violinplot(data=df, x=x_column, y=y_column, hue=hue, ax=ax)

        elif chart_type == "bar":
            sns.barplot(data=df, x=x_column, y=y_column, hue=hue, ax=ax)

        elif chart_type == "line":
            sns.lineplot(data=df, x=x_column, y=y_column, hue=hue, ax=ax)

        elif chart_type == "countplot":
            sns.countplot(data=df, x=x_column, hue=hue, ax=ax)
            plt.xticks(rotation=45, ha="right")

        elif chart_type == "heatmap":
            plt.close(fig)
            numeric_df = df.select_dtypes(include="number")
            fig, ax = plt.subplots(figsize=(max(figsize_w, len(numeric_df.columns)),
                                            max(figsize_h, len(numeric_df.columns) * 0.6)))
            sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)

        elif chart_type == "pairplot":
            plt.close(fig)
            cols = [c for c in [x_column, y_column, hue_column] if c and c in df.columns]
            pair_df = df[cols].dropna() if cols else df.select_dtypes(include="number")
            g = sns.pairplot(pair_df, hue=hue, corner=True)
            safe_title = (title or f"pairplot_{'_'.join(cols)}").replace(" ", "_")[:60]
            out_path = PLOT_DIR / f"{safe_title}.png"
            g.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close("all")
            return json.dumps({
                "status": "success",
                "chart_type": "pairplot",
                "path": str(out_path),
                "message": f"Pairplot saved to {out_path}",
            })

        else:
            plt.close(fig)
            return json.dumps({
                "status": "error",
                "message": f"Unknown chart_type '{chart_type}'. Use: histogram, scatter, box, violin, bar, line, heatmap, countplot, pairplot.",
            })

        # Title & layout
        chart_title = title or f"{chart_type.title()}: {x_column}" + (f" vs {y_column}" if y_column else "")
        ax.set_title(chart_title, fontsize=14, fontweight="bold")
        fig.tight_layout()

        safe_name = chart_title.replace(" ", "_").replace(":", "")[:60]
        out_path = PLOT_DIR / f"{safe_name}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return json.dumps({
            "status": "success",
            "chart_type": chart_type,
            "path": str(out_path),
            "message": f"Chart saved to {out_path}",
        })

    except Exception as e:
        plt.close("all")
        return json.dumps({"status": "error", "message": str(e)})


@tool
def list_columns(file_path: str) -> str:
    """Return column names, dtypes, and null counts for quick inspection.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    str
        JSON list of column info dicts.
    """
    df = pd.read_csv(file_path, nrows=500)   # fast peek
    info = []
    for col in df.columns:
        info.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "nulls": int(df[col].isna().sum()),
            "nunique": int(df[col].nunique()),
            "sample_values": [str(v) for v in df[col].dropna().head(3).tolist()],
        })
    return json.dumps(info, indent=2)

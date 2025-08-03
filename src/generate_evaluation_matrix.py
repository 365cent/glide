
import pandas as pd
from pathlib import Path
import argparse

def generate_evaluation_matrix(log_type: str, output_dir: Path = Path("results")):
    """Generates a sortable evaluation matrix in Markdown format."""
    results_csv = output_dir / f"evaluation_results_{log_type}.csv"
    
    if not results_csv.exists():
        print(f"Error: Results CSV not found at {results_csv}")
        return
    
    df = pd.read_csv(results_csv)
    
    # Select and reorder columns for the matrix
    # Prioritize key overall metrics and then per-class F1 for rare events
    display_columns = [
        "embedding_type",
        "micro_f1",
        "macro_f1",
        "hamming_loss",
        "pseudo_label_micro_f1",
        "micro_precision",
        "micro_recall",
        "jaccard_micro",
    ]

    # Add rare event F1 scores if they exist
    rare_event_f1_cols = [col for col in df.columns if col.startswith("rare_event_") and col.endswith("_f1")]
    display_columns.extend(sorted(rare_event_f1_cols))

    # Filter out columns that don't exist in the DataFrame
    display_columns = [col for col in display_columns if col in df.columns]

    # Sort the DataFrame for reproducibility and easy comparison
    df_sorted = df[display_columns].sort_values(by=["micro_f1", "macro_f1"], ascending=False)
    
    markdown_output = df_sorted.to_markdown(index=False)
    
    matrix_path = output_dir / f"evaluation_matrix_{log_type}.md"
    with open(matrix_path, "w") as f:
        f.write(f"# Evaluation Matrix for {log_type} Anomaly Detection\n\n")
        f.write(markdown_output)
    
    print(f"Evaluation matrix saved to {matrix_path}")
    print("\n--- Sortable Evaluation Matrix ---")
    print(markdown_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation matrix.")
    parser.add_argument("--log_type", type=str, required=True,
                        help="Log type for which to generate the evaluation matrix.")
    args = parser.parse_args()
    generate_evaluation_matrix(args.log_type)



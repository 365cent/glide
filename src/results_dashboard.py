
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

def create_dashboard(log_type: str, output_dir: Path = Path("results")):
    """Creates a comprehensive dashboard from evaluation results."""
    results_csv = output_dir / f"evaluation_results_{log_type}.csv"
    
    if not results_csv.exists():
        print(f"Error: Results CSV not found at {results_csv}")
        return
    
    df = pd.read_csv(results_csv)
    
    print(f"Generating dashboard for {log_type}...")
    
    # Set up plotting style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(18, 12))
    
    # Plot 1: Micro F1 Score across Embedding Types
    plt.subplot(2, 2, 1)
    sns.barplot(x="embedding_type", y="micro_f1", data=df)
    plt.title("Micro F1 Score by Embedding Type")
    plt.ylabel("Micro F1 Score")
    plt.xlabel("Embedding Type")
    
    # Plot 2: Macro F1 Score across Embedding Types
    plt.subplot(2, 2, 2)
    sns.barplot(x="embedding_type", y="macro_f1", data=df)
    plt.title("Macro F1 Score by Embedding Type")
    plt.ylabel("Macro F1 Score")
    plt.xlabel("Embedding Type")

    # Plot 3: Hamming Loss across Embedding Types
    plt.subplot(2, 2, 3)
    sns.barplot(x="embedding_type", y="hamming_loss", data=df)
    plt.title("Hamming Loss by Embedding Type")
    plt.ylabel("Hamming Loss")
    plt.xlabel("Embedding Type")

    # Plot 4: Pseudo-label Micro F1 Score across Embedding Types
    plt.subplot(2, 2, 4)
    sns.barplot(x="embedding_type", y="pseudo_label_micro_f1", data=df)
    plt.title("Pseudo-label Micro F1 Score by Embedding Type")
    plt.ylabel("Pseudo-label Micro F1 Score")
    plt.xlabel("Embedding Type")
    
    plt.tight_layout()
    dashboard_path = output_dir / f"dashboard_{log_type}.png"
    plt.savefig(dashboard_path)
    print(f"Dashboard saved to {dashboard_path}")

    # Also output key metrics to a text file for easy review
    with open(output_dir / f"summary_metrics_{log_type}.txt", "w") as f:
        f.write(f"Summary Metrics for {log_type} Anomaly Detection\n")
        f.write("=====================================================\n\n")
        f.write(df[["embedding_type", "micro_f1", "macro_f1", "hamming_loss", "pseudo_label_micro_f1"]].to_string(index=False))
        f.write("\n\n")
        f.write("Per-class F1 Scores (Top 5 for brevity):\n")
        for _, row in df.iterrows():
            f.write(f"\nEmbedding Type: {row['embedding_type']}\n")
            per_class_f1s = {k: v for k, v in row.items() if k.startswith("per_class_") and k.endswith("_f1")}
            sorted_f1s = sorted(per_class_f1s.items(), key=lambda item: item[1], reverse=True)
            for i, (metric, value) in enumerate(sorted_f1s):
                if i >= 5: break
                f.write(f"  {metric}: {value:.4f}\n")

    print(f"Summary metrics saved to {output_dir / f'summary_metrics_{log_type}.txt'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation dashboard.")
    parser.add_argument("--log_type", type=str, required=True,
                        help="Log type for which to generate the dashboard.")
    args = parser.parse_args()
    create_dashboard(args.log_type)



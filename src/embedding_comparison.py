
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from halo import Halo
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EmbeddingComparison")

# Configuration
EMBEDDINGS_DIR = Path("embeddings")
RESULTS_DIR = Path("results")

class EmbeddingComparator:
    def __init__(self, embeddings_dir=EMBEDDINGS_DIR, results_dir=RESULTS_DIR):
        self.embeddings_dir = embeddings_dir
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_embeddings(self, embedding_type, log_type=None):
        """Loads log embeddings and labels for a given embedding type and optional log type."""
        spinner = Halo(text=f"Loading {embedding_type} embeddings for {log_type if log_type else 'all'} log types", spinner='dots')
        spinner.start()
        
        try:
            if log_type:
                log_embedding_path = self.embeddings_dir / embedding_type / f"log_{log_type}.pkl"
                label_embedding_path = self.embeddings_dir / embedding_type / f"label_{log_type}.pkl"
            else:
                # Load all log types for the given embedding type
                all_log_embeddings = []
                all_label_data = []
                
                for sub_dir in (self.embeddings_dir / embedding_type).iterdir():
                    if sub_dir.is_dir(): # This should be the log_type directory
                        current_log_type = sub_dir.name
                        log_embedding_path = sub_dir / f"log_{current_log_type}.pkl"
                        label_embedding_path = sub_dir / f"label_{current_log_type}.pkl"
                        
                        if log_embedding_path.exists() and label_embedding_path.exists():
                            with open(log_embedding_path, 'rb') as f:
                                all_log_embeddings.append(pickle.load(f))
                            with open(label_embedding_path, 'rb') as f:
                                all_label_data.append(pickle.load(f))
                        else:
                            logger.warning(f"Missing embedding files for {embedding_type}/{current_log_type}")
                
                if not all_log_embeddings:
                    spinner.fail(f"No {embedding_type} embeddings found.")
                    return None, None
                
                # Combine dataframes
                log_embeddings_df = pd.concat(all_log_embeddings, ignore_index=True)
                
                # Combine label data, assuming consistent structure
                # This part might need more sophisticated merging if label structures differ significantly
                combined_label_vectors = np.vstack([ld['vectors'] for ld in all_label_data])
                combined_classes = list(set(cls for ld in all_label_data for cls in ld['classes']))
                combined_description = all_label_data[0]['description'] if all_label_data else ""
                
                label_data = {
                    'vectors': combined_label_vectors,
                    'classes': combined_classes,
                    'description': combined_description
                }
                
                spinner.succeed(f"Loaded {embedding_type} embeddings successfully.")
                return log_embeddings_df, label_data

            if not log_embedding_path.exists() or not label_embedding_path.exists():
                spinner.fail(f"Embedding files not found for {embedding_type} and log type {log_type}.")
                return None, None

            with open(log_embedding_path, 'rb') as f:
                log_embeddings_df = pickle.load(f)
            with open(label_embedding_path, 'rb') as f:
                label_data = pickle.load(f)
            
            spinner.succeed(f"Loaded {embedding_type} embeddings for {log_type} successfully.")
            return log_embeddings_df, label_data

        except Exception as e:
            spinner.fail(f"Error loading {embedding_type} embeddings: {e}")
            return None, None

    def visualize_embeddings(self, df, embedding_name, output_file=None):
        """Create t-SNE visualization with balanced class sampling for performance and minority visibility."""
        MAX_TOTAL_POINTS = 50000   # Hard cap on total points sent to t-SNE
        MAX_POINTS_PER_CLASS = 1500  # Limit for any single class to avoid domination

        spinner = Halo(text="Preparing visualization data", spinner='dots')
        spinner.start()

        df['viz_label'] = df.apply(lambda row: 'Normal' if not row['label_json'] else 'Attack', axis=1)

        sampled_df = pd.DataFrame()
        unique_viz_labels = df['viz_label'].unique()

        for label in unique_viz_labels:
            class_df = df[df['viz_label'] == label]
            if len(class_df) > MAX_POINTS_PER_CLASS:
                sampled_df = pd.concat([sampled_df, class_df.sample(MAX_POINTS_PER_CLASS, random_state=42)])
            else:
                sampled_df = pd.concat([sampled_df, class_df])

        if len(sampled_df) > MAX_TOTAL_POINTS:
            sampled_df = sampled_df.sample(MAX_TOTAL_POINTS, random_state=42)

        spinner.succeed(f"Sampled {len(sampled_df)} points for visualization")

        if sampled_df.empty:
            print("No data to visualize after sampling.")
            return

        spinner.text = "Running t-SNE dimensionality reduction"
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, learning_rate=200, metric='cosine')
            embeddings_list = [np.array(e, dtype=np.float32) for e in sampled_df['log_embedding'].tolist()]
            X_tsne = tsne.fit_transform(np.array(embeddings_list))
            spinner.succeed("t-SNE complete")
        except ValueError as e:
            spinner.fail(f"t-SNE failed: {e}. This might happen if there's not enough variance in the data or too few samples.")
            print("Skipping t-SNE visualization.")
            return

        spinner.text = "Generating plot"
        plt.figure(figsize=(12, 10))
        sns.scatterplot(
            x=X_tsne[:, 0],
            y=X_tsne[:, 1],
            hue=sampled_df['viz_label'],
            palette=sns.color_palette("hsv", len(sampled_df['viz_label'].unique())),
            legend='full',
            alpha=0.7
        )
        plt.title(f't-SNE Visualization of {embedding_name} Embeddings')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        
        if output_file:
            plt.savefig(output_file)
            spinner.succeed(f"Saved t-SNE plot to {output_file}")
        else:
            plt.show()
            spinner.succeed("Displayed t-SNE plot")

        plt.close()

    def compare_embeddings(self, embedding_types: List[str], log_type: Optional[str] = None):
        """Compares embeddings from different types and visualizes them."""
        logger.info(f"Starting comparison for embeddings: {embedding_types}")
        
        all_dfs = []
        for emb_type in embedding_types:
            df, _ = self.load_embeddings(emb_type, log_type)
            if df is not None:
                df['embedding_type'] = emb_type
                all_dfs.append(df)
        
        if not all_dfs:
            logger.error("No embeddings loaded for comparison.")
            return
            
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Now visualize the combined embeddings, coloring by embedding_type and anomaly status
        self.visualize_embeddings(combined_df, "Combined", self.results_dir / "combined_embeddings_tsne.png")
        
        logger.info("Embedding comparison complete.")

def main():
    parser = argparse.ArgumentParser(description="Compare and visualize different log embeddings.")
    parser.add_argument("--embedding_types", nargs='+', required=True,
                        help="List of embedding types to compare (e.g., fasttext word2vec logbert).")
    parser.add_argument("--log_type", type=str, default=None,
                        help="Optional: Compare embeddings for a specific log type (e.g., 'vpn').")
    
    args = parser.parse_args()

    comparator = EmbeddingComparator()
    comparator.compare_embeddings(args.embedding_types, args.log_type)

if __name__ == '__main__':
    main()



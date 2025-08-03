
import subprocess
import argparse
import os
from pathlib import Path

def run_command(command: str, cwd: str = "."):
    """Helper function to run shell commands and print output."""
    print(f"Executing: {command} in {cwd}")
    process = subprocess.Popen(command, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {command}")

def main():
    parser = argparse.ArgumentParser(description="Run the end-to-end anomaly detection pipeline.")
    parser.add_argument("--log_type", type=str, required=True,
                        help="Log type to process and evaluate (e.g., 'vpn', 'wp-error').")
    parser.add_argument("--embedding_types", nargs="+", default=["fasttext", "word2vec", "logbert"],
                        help="List of embedding types to use for training and evaluation.")
    parser.add_argument("--optimize_thresholds", action='store_true', help="Optimize per-class thresholds during evaluation.")

    args = parser.parse_args()

    base_dir = Path(__file__).parent
    src_dir = base_dir / "src"
    models_dir = base_dir / "models"
    embeddings_dir = base_dir / "embeddings"
    results_dir = base_dir / "results"

    # Create necessary directories
    models_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Phase 1: Data Preprocessing and Compaction ---")
    # This step would typically involve reading raw logs and generating the compacted format.
    # For this example, we'll assume a dummy log file for demonstration.
    # In a real scenario, you'd have a script to generate `compacted_logs.pkl` and `ground_truth_labels.pkl`
    # based on the `preprocessing.py` logic.
    # For now, let's create dummy files if they don't exist.
    dummy_log_path = base_dir / "logs" / f"raw_{args.log_type}.log"
    if not dummy_log_path.exists():
        print(f"Creating dummy raw log file: {dummy_log_path}")
        dummy_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dummy_log_path, "w") as f:
            f.write("2024-01-01 10:00:00 INFO: User logged in successfully.\n")
            f.write("2024-01-01 10:01:00 ERROR: Failed to connect to database.\n")
            f.write("2024-01-01 10:02:00 INFO: Data processed.\n")
            f.write("2024-01-01 10:03:00 WARNING: Disk space low.\n")
            f.write("2024-01-01 10:04:00 ERROR: Unauthorized access attempt.\n")

    print(f"Running preprocessing for {args.log_type}...")
    # This command would run your preprocessing script to generate the necessary data files.
    # Assuming preprocessing.py generates `compacted_logs.pkl` and `ground_truth_labels.pkl`
    # in the `embeddings` directory for each log type.
    # For now, we'll simulate this by creating dummy files.
    compacted_log_path = embeddings_dir / f"compacted_{args.log_type}.pkl"
    ground_truth_path = embeddings_dir / f"ground_truth_{args.log_type}.pkl"

    if not compacted_log_path.exists() or not ground_truth_path.exists():
        print(f"Creating dummy compacted log and ground truth files for {args.log_type}...")
        import numpy as np
        import pickle
        # Dummy data: 5 samples, 2 columns (log_message, log_type)
        dummy_compacted_logs = [("User logged in successfully.", "INFO"),
                                ("Failed to connect to database.", "ERROR"),
                                ("Data processed.", "INFO"),
                                ("Disk space low.", "WARNING"),
                                ("Unauthorized access attempt.", "ERROR")]
        # Dummy ground truth labels (binary for simplicity, 2 classes: normal, anomaly)
        # Assuming INFO/WARNING are normal (0), ERROR is anomaly (1)
        # This needs to be aligned with the actual attack types and their labels.
        # For multi-label, this would be a matrix.
        dummy_ground_truth = np.array([[0, 0], [1, 0], [0, 0], [0, 0], [1, 0]]) # Example: [is_anomaly, is_db_error]
        dummy_attack_types = ["is_anomaly", "is_db_error"]

        with open(compacted_log_path, "wb") as f:
            pickle.dump(dummy_compacted_logs, f)
        with open(ground_truth_path, "wb") as f:
            pickle.dump({"labels": dummy_ground_truth, "attack_types": dummy_attack_types}, f)
        print("Dummy compacted log and ground truth files created.")

    print("\n--- Phase 2: Embedding Generation ---")
    for emb_type in args.embedding_types:
        print(f"Generating {emb_type} embeddings for {args.log_type}...")
        if emb_type == "fasttext":
            run_command(f"python {src_dir}/fasttext_embedding_mock.py --log_type {args.log_type}")
        elif emb_type == "word2vec":
            run_command(f"python {src_dir}/word2vec_embedding.py --log_type {args.log_type}")
        elif emb_type == "logbert":
            run_command(f"python {src_dir}/logbert_embeddings.py --log_type {args.log_type}")
        else:
            print(f"Warning: Unknown embedding type {emb_type}. Skipping.")

    print("\n--- Phase 3: Transformer Training ---")
    for emb_type in args.embedding_types:
        print(f"Training transformer with {emb_type} embeddings for {args.log_type}...")
        # The transformer.py script should save its predictions and true labels for evaluation
        run_command(f"python {src_dir}/transformer_simple.py --log_type {args.log_type} --embedding_type {emb_type}")

    print("\n--- Phase 4: Model Evaluation ---")
    eval_cmd = f"python {src_dir}/evaluate_models_simple.py --log_type {args.log_type} --embedding_types {' '.join(args.embedding_types)}"
    if args.optimize_thresholds:
        eval_cmd += " --optimize_thresholds"
    run_command(eval_cmd)

    # print("\n--- Phase 5: Generate Evaluation Matrix and Dashboard ---")
    # run_command(f"python {src_dir}/generate_evaluation_matrix.py --log_type {args.log_type}")
    # run_command(f"python {src_dir}/results_dashboard.py --log_type {args.log_type}")

    print("\n--- Pipeline Finished ---")
    print(f"Results are in the {results_dir} directory.")

if __name__ == "__main__":
    main()



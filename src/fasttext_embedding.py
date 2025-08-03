#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FastText Embedding for Log Analysis

This script loads TFRecord files, processes log entries, generates FastText embeddings,
and creates binary multi-label vectors.

Output files per log type (3 files for clarity):
- log_{type}.pkl: Raw log text embeddings (300D FastText vectors, float32)
- label_{type}.pkl: Binary label vectors with metadata
  * 'vectors': Binary arrays where [0 1 0] means only the second class is present
  * 'classes': List of attack types corresponding to each column
  * 'description': Explanation of the binary vector format
- attack_types_{type}.txt: Human-readable attack type mapping and examples

Performance optimizations:
- Batch processing (500 samples per batch)
- Memory-efficient data types (int8 for labels, float32 for embeddings)
- Vectorized operations where possible
- Optimized pickle protocol for faster I/O
- Barnes-Hut t-SNE for faster visualization
- Reduced progress update frequency

Example label structure:
{
  'vectors': array([[0, 1, 0], [1, 0, 1], ...], dtype=int8),  # Binary vectors
  'classes': ['attack_type_1', 'attack_type_2', 'attack_type_3'],
  'description': 'Binary multi-label vectors where [0 1 0] means only the second class is present'
}
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from gensim.models import FastText
from gensim.utils import simple_preprocess
from pathlib import Path
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import json
import multiprocessing
import argparse
from functools import partial
from halo import Halo

# Import configuration
from src.config import EMBEDDINGS_DIR as OUTPUT_DIR, PROCESSED_DIR, VECTOR_SIZE

def parse_example(example):
    """Parse a TensorFlow Example protocol buffer."""
    feature_description = {
        'log': tf.io.FixedLenFeature([], tf.string),
        'labels': tf.io.FixedLenFeature([], tf.string),
        'log_type': tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(example, feature_description)

def load_tfrecord_files(directory=PROCESSED_DIR, log_type_filter=None):
    """Load TFRecord files from directory into a DataFrame with optimized processing."""
    # Get list of all tfrecord files
    if log_type_filter:
        log_type_dir_path = directory / log_type_filter
        if not log_type_dir_path.exists():
            raise FileNotFoundError(f"No directory found for '{log_type_filter}'")
        tfrecord_files = list(log_type_dir_path.glob("*.tfrecord"))
        if not tfrecord_files:
            raise FileNotFoundError(f"No TFRecord files found for log type '{log_type_filter}'")
    else:
        tfrecord_files = []
        for log_dir_path in directory.iterdir():
            if log_dir_path.is_dir():
                tfrecord_files.extend(log_dir_path.glob("*.tfrecord"))
        if not tfrecord_files:
            raise FileNotFoundError(f"No TFRecord files found in {directory}")
    
    print(f"Loading {len(tfrecord_files)} TFRecord files...")
    
    # Process files in batches
    all_logs = []
    all_labels_json = []
    all_log_types = []
    
    spinner = Halo(text='Loading files', spinner='dots')
    spinner.start()
    
    for file_idx, file_path in enumerate(tfrecord_files):
        try:
            spinner.text = f"Loading file {file_idx+1}/{len(tfrecord_files)}: {file_path.name}"
            log_type = file_path.parent.name
            
            # Use TensorFlow's optimized API for batch processing
            dataset = tf.data.TFRecordDataset(str(file_path), compression_type="GZIP", num_parallel_reads=4)
            dataset = dataset.batch(1000)  # Process in batches
            
            for batch in dataset:
                parsed_batch = tf.io.parse_example(batch, {
                    'log': tf.io.FixedLenFeature([], tf.string),
                    'labels': tf.io.FixedLenFeature([], tf.string),
                    'log_type': tf.io.FixedLenFeature([], tf.string)
                })
                
                logs = [log.decode('utf-8') for log in parsed_batch['log'].numpy()]
                labels = [label.decode('utf-8') for label in parsed_batch['labels'].numpy()]
                log_types_batch = [log_type.decode('utf-8') for log_type in parsed_batch['log_type'].numpy()]
                
                all_logs.extend(logs)
                all_labels_json.extend(labels)
                all_log_types.extend(log_types_batch)
                
        except Exception as e:
            spinner.text = f"Error processing file {file_path}: {e}"
            spinner.fail()
            spinner = Halo(text='Loading files', spinner='dots')
            spinner.start()
    
    spinner.succeed(f"Loaded {len(all_logs)} log entries")
    
    return pd.DataFrame({
        'log': all_logs, 
        'label_json': all_labels_json,
        'log_type': all_log_types
    })

def normalize_label(label):
    """Normalize attack labels to ensure consistency."""
    if not label:
        return label
    return label.replace("-", "_").lower().strip()

def get_labels_from_json(label_json_str):
    """Extract labels from JSON string."""
    try:
        labels = json.loads(label_json_str)
        if not isinstance(labels, list):
            labels = [labels]
        return {normalize_label(label) for label in labels if label}
    except json.JSONDecodeError:
        return set()

def collect_unique_labels_from_data(df):
    """Extract all unique attack labels from the dataset efficiently."""
    all_unique_labels = set()
    
    spinner = Halo(text='Collecting unique labels', spinner='dots')
    spinner.start()
    
    for label_json_str in df['label_json']:
        all_unique_labels.update(get_labels_from_json(label_json_str))
    
    # Remove empty labels
    all_unique_labels.discard('')
    all_unique_labels.discard(None)
    
    spinner.succeed(f"Found {len(all_unique_labels)} unique attack types")
    return sorted(list(all_unique_labels))

def load_pretrained_fasttext():
    """Load pre-trained FastText model."""
    spinner = Halo(text='Loading pre-trained FastText model', spinner='dots')
    spinner.start()
    
    try:
        # Try to load from local cache first
        model = FastText.load_facebook_model('cc.en.300.bin')
        spinner.succeed("Loaded pre-trained FastText model")
        return model
    except Exception as e:
        spinner.fail(f"Failed to load pre-trained FastText model: {e}")
        print("Consider downloading the model manually or check internet connection")
        return None

def preprocess_text(text):
    """Preprocess text for embedding."""
    return simple_preprocess(text)

def embed_text(model, tokens):
    """Generate embedding for tokenized text using pre-trained FastText."""
    if not tokens:
        return np.zeros(model.vector_size, dtype=np.float32)
    
    # Vectorized approach for better performance
    valid_tokens = [token for token in tokens if token in model.wv]
    
    if valid_tokens:
        # Get all embeddings at once and compute mean
        embeddings_matrix = np.array([model.wv[token] for token in valid_tokens], dtype=np.float32)
        return np.mean(embeddings_matrix, axis=0)
    else:
        return np.zeros(model.vector_size, dtype=np.float32)

def create_binary_label_vector(label_json_str, all_attack_types):
    """Create binary vector representation for multi-label classification."""
    labels = get_labels_from_json(label_json_str)
    
    # Binary vector: [0 1 0] means only second class is present
    binary_vector = np.zeros(len(all_attack_types), dtype=np.int8)  # Use int8 for memory efficiency
    
    # Vectorized approach for better performance
    if labels:
        attack_indices = [i for i, attack in enumerate(all_attack_types) if attack in labels]
        binary_vector[attack_indices] = 1
    
    return binary_vector

def display_data_distribution(df, log_type_name="all combined"):
    """Calculate and display data distribution statistics."""
    print(f"\n{'='*20} Data Distribution for '{log_type_name}' {'='*20}")
    
    total_logs = len(df)
    print(f"Total log entries: {total_logs}")
    
    # Extract all unique labels
    all_labels_count = {}
    normal_count = 0
    attack_count = 0
    
    spinner = Halo(text='Analyzing data distribution', spinner='dots')
    spinner.start()
    
    for label_json_str in df['label_json']:
        labels = get_labels_from_json(label_json_str)
        if labels:
            for label in labels:
                all_labels_count[label] = all_labels_count.get(label, 0) + 1
            attack_count += 1
        else:
            normal_count += 1
    
    spinner.succeed("Data distribution analysis complete")
    
    # Display attack distribution
    if all_labels_count:
        print("\nAttack type distribution:")
        for attack, count in sorted(all_labels_count.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_logs) * 100
            print(f"  {attack}: {count} occurrences ({percentage:.2f}%)")
    
    # Display normal vs attack statistics
    attack_percentage = (attack_count / total_logs) * 100 if total_logs > 0 else 0
    normal_percentage = (normal_count / total_logs) * 100 if total_logs > 0 else 0
    
    print(f"\nLogs with attacks: {attack_count} ({attack_percentage:.2f}%)")
    print(f"Normal logs: {normal_count} ({normal_percentage:.2f}%)")
    
    # Display log type distribution if processing combined dataset
    if log_type_name == "all combined":
        print("\nLog type distribution:")
        type_counts = df['log_type'].value_counts()
        for log_type, count in type_counts.items():
            percentage = (count / total_logs) * 100
            print(f"  {log_type}: {count} entries ({percentage:.2f}%)")
    
    print(f"{'='*70}\n")
    return attack_count, normal_count

def process_embeddings_batch(model, tokens_batch):
    """Process a batch of tokens for embedding generation."""
    return [embed_text(model, tokens) for tokens in tokens_batch]

def process_labels_batch(labels_batch, attack_types):
    """Process a batch of labels for binary vector generation."""
    return [create_binary_label_vector(label_json, attack_types) for label_json in labels_batch]

def process_embeddings(df, model, use_global_attack_list=False):
    """Process logs and create embeddings with binary label vectors - optimized version."""
    spinner = Halo(text='Processing log embeddings', spinner='dots')
    spinner.start()
    
    total_count = len(df)
    batch_size = 500  # Optimized batch size for better performance
    
    # Tokenize logs in batches
    spinner.text = "Tokenizing logs in batches"
    tokenized_logs = []
    
    # Use vectorized string operations where possible
    for i in range(0, total_count, batch_size):
        end_idx = min(i + batch_size, total_count)
        batch_logs = df['log'].iloc[i:end_idx]
        
        if i % 2000 == 0:  # Update progress less frequently
            spinner.text = f"Tokenizing logs: {end_idx}/{total_count} ({end_idx/total_count*100:.1f}%)"
        
        # Process batch
        batch_tokens = [preprocess_text(log) for log in batch_logs]
        tokenized_logs.extend(batch_tokens)
    
    df['tokens'] = tokenized_logs
    
    # Create log embeddings in batches
    spinner.text = "Creating log embeddings in batches"
    log_embeddings = []
    
    for i in range(0, total_count, batch_size):
        end_idx = min(i + batch_size, total_count)
        tokens_batch = tokenized_logs[i:end_idx]
        
        if i % 2000 == 0:
            spinner.text = f"Creating log embeddings: {end_idx}/{total_count} ({end_idx/total_count*100:.1f}%)"
        
        # Process batch with multiprocessing could be added here if needed
        batch_embeddings = process_embeddings_batch(model, tokens_batch)
        log_embeddings.extend(batch_embeddings)
    
    # Convert to numpy array for better memory efficiency
    log_embeddings_array = np.array(log_embeddings, dtype=np.float32)
    df['log_embedding'] = list(log_embeddings_array)  # Convert back to list for pandas
    
    # Process labels in batches
    spinner.text = "Processing binary label vectors in batches"
    
    if use_global_attack_list:
        # Use all labels across all log types
        attack_types = collect_unique_labels_from_data(df)
        
        binary_labels = []
        for i in range(0, total_count, batch_size):
            end_idx = min(i + batch_size, total_count)
            labels_batch = df['label_json'].iloc[i:end_idx]
            
            if i % 2000 == 0:
                spinner.text = f"Processing labels: {end_idx}/{total_count} ({end_idx/total_count*100:.1f}%)"
            
            batch_binary = process_labels_batch(labels_batch, attack_types)
            binary_labels.extend(batch_binary)
        
        df['binary_labels'] = binary_labels
        df.attrs['attack_types'] = attack_types
    else:
        # Process by log type
        log_type_to_attacks = {}
        for log_type, group_df in df.groupby('log_type'):
            spinner.text = f"Processing log type: {log_type}"
            log_type_to_attacks[log_type] = collect_unique_labels_from_data(group_df)
        
        # Process all labels in batches
        binary_labels = []
        for i in range(0, total_count, batch_size):
            end_idx = min(i + batch_size, total_count)
            
            if i % 2000 == 0:
                spinner.text = f"Processing binary labels: {end_idx}/{total_count} ({end_idx/total_count*100:.1f}%)"
            
            batch_binary = []
            for j in range(i, end_idx):
                row = df.iloc[j]
                log_type = row['log_type']
                if log_type in log_type_to_attacks:
                    binary_vector = create_binary_label_vector(row['label_json'], log_type_to_attacks[log_type])
                else:
                    binary_vector = np.array([], dtype=np.int8)
                batch_binary.append(binary_vector)
            
            binary_labels.extend(batch_binary)
        
        df['binary_labels'] = binary_labels
        df.attrs['log_type_to_attacks'] = log_type_to_attacks
    
    spinner.succeed("Embedding processing complete")
    return df

def visualize_embeddings(df, output_file=None):
    """Create t-SNE visualization with balanced class sampling for performance and minority visibility."""
    # Parameters for sampling – tweak here if necessary
    MAX_TOTAL_POINTS = 50000   # Hard cap on total points sent to t-SNE
    MAX_POINTS_PER_CLASS = 1500  # Limit for any single class to avoid domination

    spinner = Halo(text="Preparing visualization data", spinner='dots')
    spinner.start()

    # Build visualization labels for every row (normal vs attacks etc.)
    df['viz_label'] = df.apply(lambda row: 'Normal' if not row['label_json'] else 'Attack', axis=1)

    # Sample data for t-SNE to manage computational load
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

    # Perform t-SNE dimensionality reduction
    spinner.text = "Running t-SNE dimensionality reduction"
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, learning_rate=200, metric='cosine')
        # Ensure all embeddings are numpy arrays before passing to t-SNE
        embeddings_list = [np.array(e, dtype=np.float32) for e in sampled_df['log_embedding'].tolist()]
        X_tsne = tsne.fit_transform(np.array(embeddings_list))
        spinner.succeed("t-SNE complete")
    except ValueError as e:
        spinner.fail(f"t-SNE failed: {e}. This might happen if there's not enough variance in the data or too few samples.")
        print("Skipping t-SNE visualization.")
        return

    # Plotting
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
    plt.title('t-SNE Visualization of Log Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    if output_file:
        plt.savefig(output_file)
        spinner.succeed(f"Saved t-SNE plot to {output_file}")
    else:
        plt.show()
        spinner.succeed("Displayed t-SNE plot")

    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate FastText embeddings for log data.")
    parser.add_argument("--log_type", type=str, required=True,
                        help="Specific log type to process (e.g., 'vpn', 'wp-error').")
    parser.add_argument("--use_global_attack_list", action='store_true',
                        help="Use a global list of attack types across all log types.")
    
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Starting FastText embedding process for log type: {args.log_type}")

    # 1. Load data
    try:
        df = load_tfrecord_files(log_type_filter=args.log_type)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Display initial data distribution
    display_data_distribution(df, args.log_type)

    # 2. Load pre-trained FastText model
    fasttext_model = load_pretrained_fasttext()
    if fasttext_model is None:
        print("Failed to load FastText model. Exiting.")
        return

    # 3. Process embeddings and binary labels
    df = process_embeddings(df, fasttext_model, use_global_attack_list=args.use_global_attack_list)

    # 4. Save embeddings and labels
    log_embeddings_path = OUTPUT_DIR / args.log_type / f"log_{args.log_type}.pkl"
    label_data_path = OUTPUT_DIR / args.log_type / f"label_{args.log_type}.pkl"
    attack_types_path = OUTPUT_DIR / args.log_type / f"attack_types_{args.log_type}.txt"

    (OUTPUT_DIR / args.log_type).mkdir(parents=True, exist_ok=True)

    with open(log_embeddings_path, 'wb') as f:
        pickle.dump(np.array(df['log_embedding'].tolist()), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Log embeddings saved to {log_embeddings_path}")

    # Prepare label data for saving
    if args.use_global_attack_list:
        all_attack_types = df.attrs['attack_types']
        label_vectors = np.array(df['binary_labels'].tolist())
    else:
        # For per-log-type processing, we need to handle the varying dimensions of binary_labels
        # This means we need to save attack_types per log_type, which is already handled by df.attrs
        # We need to ensure that all binary_labels in the DataFrame for this log_type have the same dimension
        # This implies that `process_embeddings` should have created consistent binary_labels for this log_type
        # which it does by using `log_type_to_attacks[log_type]`
        
        # Get the attack types for the current log_type
        current_log_type_attacks = df.attrs['log_type_to_attacks'][args.log_type]
        label_vectors = np.array(df['binary_labels'].tolist())
        all_attack_types = current_log_type_attacks

    label_data = {
        'vectors': label_vectors,
        'classes': all_attack_types,
        'description': 'Binary multi-label vectors where [0 1 0] means only the second class is present'
    }
    with open(label_data_path, 'wb') as f:
        pickle.dump(label_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Label data saved to {label_data_path}")

    with open(attack_types_path, 'w') as f:
        f.write("Attack Types:\n")
        for i, attack_type in enumerate(all_attack_types):
            f.write(f"  {i}: {attack_type}\n")
    print(f"Attack types mapping saved to {attack_types_path}")

    # 5. Visualize embeddings
    visualization_output_path = OUTPUT_DIR / args.log_type / f"tsne_visualization_{args.log_type}.png"
    visualize_embeddings(df, output_file=visualization_output_path)

    print(f"FastText embedding process for {args.log_type} completed.")

if __name__ == '__main__':
    main()
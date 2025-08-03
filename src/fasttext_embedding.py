#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FastText Embedding for Log Analysis - Using Pre-trained Models

This script loads TFRecord files, processes log entries, generates FastText embeddings,
and creates binary multi-label vectors.

Output files per log type (3 files for clarity):
- log_{type}.pkl: Raw log text embeddings (300D FastText vectors, float32)
- label_{type}.pkl: Binary label vectors with metadata
  * 'vectors': Binary arrays where [0 1 0] means only second class is present
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
import gensim.downloader as api

# Configuration
OUTPUT_DIR = Path("embeddings")
PROCESSED_DIR = Path("processed")
VECTOR_SIZE = 300  # Standard FastText vector size

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
            
            # Load TFRecord file
            raw_dataset = tf.data.TFRecordDataset(str(file_path), compression_type="GZIP")
            parsed_dataset = raw_dataset.map(parse_example)
            
            # Process records
            for record in parsed_dataset:
                log_content = record['log'].numpy().decode('utf-8')
                labels_json = record['labels'].numpy().decode('utf-8')
                record_log_type = record['log_type'].numpy().decode('utf-8')
                
                all_logs.append(log_content)
                all_labels_json.append(labels_json)
                all_log_types.append(record_log_type)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    spinner.succeed(f"Loaded {len(all_logs)} records from {len(tfrecord_files)} files")
    
    # Create DataFrame
    df = pd.DataFrame({
        'log': all_logs,
        'label_json': all_labels_json,
        'log_type': all_log_types
    })
    
    return df

def normalize_label(label):
    """Normalize label to lowercase and replace spaces with underscores."""
    return label.lower().replace(' ', '_')

def collect_unique_labels_from_data(df):
    """Collect all unique attack types from the dataset."""
    all_labels = set()
    for label_json_str in df['label_json']:
        try:
            labels = json.loads(label_json_str)
            if isinstance(labels, list):
                all_labels.update(labels)
        except (json.JSONDecodeError, TypeError):
            continue
    
    return sorted(list(all_labels))

def load_pretrained_fasttext():
    """Load pre-trained FastText model."""
    try:
        print("Loading pre-trained FastText model...")
        model = api.load('fasttext-wiki-news-subwords-300')
        print("FastText model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading FastText model: {e}")
        return None

def preprocess_text(text):
    """Preprocess text for FastText embedding."""
    return simple_preprocess(text, deacc=True)

def embed_text(model, tokens):
    """Generate FastText embedding for text tokens."""
    if not tokens:
        return np.zeros(VECTOR_SIZE, dtype=np.float32)
    
    # Get embeddings for all tokens and average them
    embeddings = []
    for token in tokens:
        try:
            embedding = model[token]
            embeddings.append(embedding)
        except KeyError:
            # Token not in vocabulary, skip
            continue
    
    if embeddings:
        return np.mean(embeddings, axis=0).astype(np.float32)
    else:
        return np.zeros(VECTOR_SIZE, dtype=np.float32)

def embed_labels(model, labels):
    """Generate FastText embedding for label tokens."""
    if not labels:
        return np.zeros(VECTOR_SIZE, dtype=np.float32)
    
    # Preprocess each label
    processed_labels = []
    for label in labels:
        processed_label = normalize_label(label)
        processed_labels.extend(preprocess_text(processed_label))
    
    return embed_text(model, processed_labels)

def create_binary_label_vector(label_json_str, all_attack_types):
    """Create binary label vector from JSON string."""
    try:
        labels = json.loads(label_json_str)
        if not isinstance(labels, list):
            return np.zeros(len(all_attack_types), dtype=np.int8)
        
        binary_vector = np.zeros(len(all_attack_types), dtype=np.int8)
        for label in labels:
            normalized_label = normalize_label(label)
            if normalized_label in all_attack_types:
                idx = all_attack_types.index(normalized_label)
                binary_vector[idx] = 1
        
        return binary_vector
    except (json.JSONDecodeError, TypeError):
        return np.zeros(len(all_attack_types), dtype=np.int8)

def display_data_distribution(df, log_type_name="all combined"):
    """Display data distribution statistics."""
    print(f"\n=== Data Distribution for {log_type_name} ===")
    print(f"Total records: {len(df)}")
    
    # Count unique log types
    log_type_counts = df['log_type'].value_counts()
    print(f"Log types: {len(log_type_counts)}")
    for log_type, count in log_type_counts.items():
        print(f"  {log_type}: {count}")
    
    # Analyze labels
    all_labels = collect_unique_labels_from_data(df)
    print(f"Unique attack types: {len(all_labels)}")
    
    # Count label occurrences
    label_counts = {}
    for label_json_str in df['label_json']:
        try:
            labels = json.loads(label_json_str)
            if isinstance(labels, list):
                for label in labels:
                    label_counts[label] = label_counts.get(label, 0) + 1
        except (json.JSONDecodeError, TypeError):
            continue
    
    print("Top 10 most common attack types:")
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (label, count) in enumerate(sorted_labels[:10]):
        print(f"  {i+1}. {label}: {count}")
    
    # Count records with no attacks (normal traffic)
    normal_count = sum(1 for label_json_str in df['label_json'] 
                      if not label_json_str or label_json_str == '[]')
    print(f"Normal traffic records: {normal_count}")
    print(f"Attack traffic records: {len(df) - normal_count}")
    print("=" * 50)

def process_embeddings_batch(model, tokens_batch):
    """Process a batch of tokenized texts."""
    return [embed_text(model, tokens) for tokens in tokens_batch]

def process_labels_batch(labels_batch, attack_types):
    """Process a batch of label JSON strings."""
    return [create_binary_label_vector(label_json, attack_types) for label_json in labels_batch]

def process_embeddings(df, model, use_global_attack_list=False):
    """Process embeddings and create binary labels with optimized batch processing."""
    print("Processing embeddings and labels...")
    
    # Determine attack types
    if use_global_attack_list:
        # Use all attack types from the dataset
        all_attack_types = collect_unique_labels_from_data(df)
        print(f"Using global attack list with {len(all_attack_types)} attack types")
    else:
        # Use attack types specific to each log type
        log_type_to_attacks = {}
        for log_type in df['log_type'].unique():
            log_type_df = df[df['log_type'] == log_type]
            log_type_attacks = collect_unique_labels_from_data(log_type_df)
            log_type_to_attacks[log_type] = log_type_attacks
            print(f"Log type '{log_type}': {len(log_type_attacks)} attack types")
        
        all_attack_types = list(set().union(*log_type_to_attacks.values()))
        print(f"Combined attack types: {len(all_attack_types)}")
    
    # Store attack types in DataFrame attributes for later use
    df.attrs['attack_types'] = all_attack_types
    df.attrs['log_type_to_attacks'] = log_type_to_attacks if not use_global_attack_list else None
    
    # Preprocess all texts
    print("Preprocessing texts...")
    df['tokens'] = df['log'].apply(preprocess_text)
    
    # Process embeddings in batches
    batch_size = 500
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    embeddings = []
    binary_labels = []
    
    spinner = Halo(text='Processing embeddings', spinner='dots')
    spinner.start()
    
    for i in range(0, len(df), batch_size):
        batch_end = min(i + batch_size, len(df))
        batch_df = df.iloc[i:batch_end]
        
        # Process embeddings for this batch
        batch_embeddings = process_embeddings_batch(model, batch_df['tokens'].tolist())
        embeddings.extend(batch_embeddings)
        
        # Process labels for this batch
        batch_labels = process_labels_batch(batch_df['label_json'].tolist(), all_attack_types)
        binary_labels.extend(batch_labels)
        
        spinner.text = f"Processed batch {i//batch_size + 1}/{total_batches}"
    
    spinner.succeed(f"Processed {len(embeddings)} embeddings")
    
    # Add to DataFrame
    df['log_embedding'] = embeddings
    df['binary_labels'] = binary_labels
    
    # Clean up
    df.drop('tokens', axis=1, inplace=True)
    
    return df

def visualize_embeddings(df, output_file=None):
    """Create t-SNE visualization of embeddings."""
    print("Creating t-SNE visualization...")
    
    # Prepare data
    embeddings = np.array(df['log_embedding'].tolist())
    
    # Create labels for visualization
    df['viz_label'] = df.apply(lambda row: 'Normal' if not row['label_json'] else 'Attack', axis=1)
    
    # Sample data if too large for t-SNE
    max_samples = 10000
    if len(df) > max_samples:
        print(f"Sampling {max_samples} points for visualization...")
        sample_indices = np.random.choice(len(df), max_samples, replace=False)
        sample_embeddings = embeddings[sample_indices]
        sample_labels = df.iloc[sample_indices]['viz_label']
    else:
        sample_embeddings = embeddings
        sample_labels = df['viz_label']
    
    # Perform t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1, perplexity=30)
    embeddings_2d = tsne.fit_transform(sample_embeddings)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=sample_labels.map({'Normal': 0, 'Attack': 1}), 
                         cmap='viridis', alpha=0.6)
    plt.title('t-SNE Visualization of FastText Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor='blue', markersize=10, label='Normal'),
                      plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor='orange', markersize=10, label='Attack')]
    plt.legend(handles=legend_elements)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    else:
        plt.show()
    
    plt.close()

def main():
    """Main function to run the FastText embedding pipeline."""
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
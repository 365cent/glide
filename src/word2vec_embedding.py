#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Word2Vec Embedding for Log Analysis - Using Pre-trained Models

This script is adapted from fasttext_embedding.py to use pre-trained Word2Vec models.
It loads TFRecord files, processes log entries, generates Word2Vec embeddings,
and creates binary multi-label vectors.

Output files per log type (3 files for clarity):
- log_{type}.pkl: Raw log text embeddings (300D Word2Vec vectors, float32)
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
from gensim.models import KeyedVectors
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

# Import configuration
from config import EMBEDDINGS_DIR, PROCESSED_DIR, VECTOR_SIZE

# Create word2vec subdirectory
OUTPUT_DIR = EMBEDDINGS_DIR / "word2vec"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

def load_pretrained_word2vec():
    """Load pre-trained Word2Vec model."""
    spinner = Halo(text='Loading pre-trained Word2Vec model', spinner='dots')
    spinner.start()
    
    try:
        # Try to load from local cache first
        model = api.load("word2vec-google-news-300")
        spinner.succeed("Loaded pre-trained Word2Vec model (word2vec-google-news-300)")
        return model
    except Exception as e:
        spinner.fail(f"Failed to load pre-trained Word2Vec model: {e}")
        print("Consider installing word2vec manually or check internet connection")
        return None

def preprocess_text(text):
    """Preprocess text for embedding."""
    return simple_preprocess(text)

def embed_text(model, tokens):
    """Generate embedding for tokenized text using pre-trained Word2Vec."""
    if not tokens:
        return np.zeros(model.vector_size, dtype=np.float32)
    
    # Vectorized approach for better performance
    valid_tokens = [token for token in tokens if token in model.key_to_index]
    
    if valid_tokens:
        # Get all embeddings at once and compute mean
        embeddings_matrix = np.array([model[token] for token in valid_tokens], dtype=np.float32)
        return np.mean(embeddings_matrix, axis=0)
    else:
        return np.zeros(model.vector_size, dtype=np.float32)

def embed_labels(model, labels):
    """Generate embeddings for labels."""
    if not labels:
        return np.zeros(model.vector_size)
    
    label_embeddings = []
    for label in labels:
        if label:
            tokens = preprocess_text(label)
            embedding = embed_text(model, tokens)
            label_embeddings.append(embedding)
    
    if label_embeddings:
        return np.mean(label_embeddings, axis=0)
    else:
        return np.zeros(model.vector_size)

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

def save_embeddings_and_labels(df, model, log_type, use_global_attack_list=False):
    """Save embeddings, labels, and attack type mappings."""
    print(f"\nSaving embeddings and labels for log type: {log_type}")
    
    # Extract embeddings as numpy array
    embeddings = np.vstack(df['log_embedding'].tolist()).astype(np.float32)
    
    # Save log embeddings
    log_output_file = OUTPUT_DIR / f"log_{log_type}.pkl"
    with open(log_output_file, 'wb') as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved log embeddings: {log_output_file} ({embeddings.shape})")
    
    # Prepare label data
    binary_vectors = np.array([vec for vec in df['binary_labels']], dtype=np.int8)
    
    # Get attack types based on mode
    if use_global_attack_list:
        attack_types = df.attrs.get('attack_types', [])
    else:
        log_type_to_attacks = df.attrs.get('log_type_to_attacks', {})
        attack_types = log_type_to_attacks.get(log_type, [])
    
    label_data = {
        'vectors': binary_vectors,
        'classes': attack_types,
        'description': 'Binary multi-label vectors where [0 1 0] means only the second class is present'
    }
    
    # Save label data
    label_output_file = OUTPUT_DIR / f"label_{log_type}.pkl"
    with open(label_output_file, 'wb') as f:
        pickle.dump(label_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved label data: {label_output_file} (vectors: {binary_vectors.shape}, classes: {len(attack_types)})")
    
    # Save human-readable attack type mapping
    attack_types_file = OUTPUT_DIR / f"attack_types_{log_type}.txt"
    with open(attack_types_file, 'w') as f:
        f.write(f"Attack Types for Log Type: {log_type}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total attack types: {len(attack_types)}\n")
        f.write(f"Vector dimension: {len(attack_types)}\n\n")
        
        if attack_types:
            f.write("Attack Type Index Mapping:\n")
            f.write("-" * 30 + "\n")
            for i, attack_type in enumerate(attack_types):
                f.write(f"{i:3d}: {attack_type}\n")
            
            f.write(f"\nBinary Vector Format:\n")
            f.write("-" * 20 + "\n")
            f.write("Each log entry has a binary vector where:\n")
            f.write("- 1 indicates the presence of that attack type\n")
            f.write("- 0 indicates the absence of that attack type\n")
            f.write("- Multiple 1s indicate multiple attack types in one log\n\n")
            
            f.write("Examples:\n")
            f.write("-" * 10 + "\n")
            if len(attack_types) >= 3:
                example_vector = [0] * len(attack_types)
                example_vector[1] = 1
                f.write(f"{example_vector} = Only '{attack_types[1]}' attack\n")
                
                example_vector = [0] * len(attack_types)
                example_vector[0] = 1
                example_vector[2] = 1
                f.write(f"{example_vector} = Both '{attack_types[0]}' and '{attack_types[2]}' attacks\n")
                
                example_vector = [0] * len(attack_types)
                f.write(f"{example_vector} = Normal log (no attacks)\n")
        else:
            f.write("No attack types found for this log type.\n")
    
    print(f"Saved attack types mapping: {attack_types_file}")

def visualize_embeddings(df, output_file=None):
    """Create t-SNE visualization of all embeddings."""
    spinner = Halo(text="Preparing visualization data", spinner='dots')
    spinner.start()

    # Generate visualization labels
    viz_labels = []
    for label_json_str in df['label_json']:
        labels = get_labels_from_json(label_json_str)
        if not labels:
            viz_labels.append("normal")
        else:
            viz_labels.append(", ".join(sorted(labels)))

    # Prepare embeddings for t-SNE
    embeddings = np.vstack(df['log_embedding'].tolist()).astype(np.float32)
    
    spinner.text = f"Running t-SNE on {len(embeddings)} data points"
    
    # Calculate perplexity
    perplexity = min(50, max(5, len(embeddings)//3))
    
    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=500,
        learning_rate='auto',
        init='pca',
        method='barnes_hut',
        random_state=42
    )
    
    reduced = tsne.fit_transform(embeddings)
    
    # Create plot DataFrame
    df_plot = pd.DataFrame({
        'x': reduced[:, 0],
        'y': reduced[:, 1],
        'label': viz_labels,
        'log_type': df['log_type'].tolist()
    })
    
    spinner.succeed("t-SNE dimensionality reduction complete")

    # Count visualization labels
    label_counts = df_plot['label'].value_counts()
    print(f"\nVisualization showing ALL {len(label_counts)} unique label combinations:")
    for label, count in label_counts.head(10).items():
        percentage = (count / len(df_plot)) * 100
        print(f"  {label}: {count} ({percentage:.2f}%)")
    
    if len(label_counts) > 10:
        print(f"  ... and {len(label_counts) - 10} more label combinations")
    
    # Create color palette - excluding green shades for non-normal logs
    unique_labels = sorted(df_plot['label'].unique())
    
    # Create color palette excluding green colors
    available_colors = sns.color_palette("Set1", n_colors=9) + sns.color_palette("Set2", n_colors=8) + sns.color_palette("Dark2", n_colors=8)
    # Filter out green-like colors (approximate RGB ranges for green)
    non_green_colors = []
    for color in available_colors:
        r, g, b = color
        # Exclude colors where green component is dominant
        if not (g > 0.6 and g > r and g > b):
            non_green_colors.append(color)
    
    # Extend with more colors if needed
    if len(unique_labels) > len(non_green_colors):
        # Add more non-green colors from other palettes
        extra_colors = sns.color_palette("tab20", n_colors=20)
        for color in extra_colors:
            r, g, b = color
            if not (g > 0.6 and g > r and g > b):
                non_green_colors.append(color)
    
    # Create color mapping
    color_map = {}
    color_idx = 0
    
    for label in unique_labels:
        if label == "normal":
            color_map[label] = "green"
        else:
            if color_idx < len(non_green_colors):
                # Make colors softer by reducing intensity
                original_color = non_green_colors[color_idx]
                r, g, b = original_color
                soft_color = (r * 0.7 + 0.3, g * 0.7 + 0.3, b * 0.7 + 0.3)
                color_map[label] = soft_color
            else:
                # Fallback to matplotlib default colors if we run out
                color_map[label] = f"C{color_idx % 10}"
            color_idx += 1
    
    # Create plot
    spinner = Halo(text="Creating visualization plot", spinner='dots')
    spinner.start()
    
    plt.figure(figsize=(16, 10))
    
    for label in unique_labels:
        mask = df_plot['label'] == label
        subset = df_plot[mask]
        
        plt.scatter(
            subset['x'], 
            subset['y'], 
            c=[color_map[label]], 
            label=label,
            alpha=0.6,
            s=20,
            edgecolors='none'
        )
    
    plt.title(f't-SNE Visualization: Word2Vec Log Embeddings (All {len(unique_labels)} Classes)', fontsize=14)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    
    # Add legend
    if len(unique_labels) <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout(rect=[0,0,0.85,1])
    else:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6, ncol=2)
        plt.tight_layout(rect=[0,0,0.8,1])
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        spinner.succeed(f"Saved visualization to {output_file}")
    else:
        plt.show()
        spinner.succeed("Displayed visualization")
    
    plt.close()
    
    # Clear memory
    del embeddings, reduced, df_plot

def get_labels_from_json(label_json_str):
    """Extract labels from JSON string."""
    try:
        labels = json.loads(label_json_str)
        if not isinstance(labels, list):
            labels = [labels]
        return {normalize_label(label) for label in labels if label}
    except json.JSONDecodeError:
        return set()

def main():
    parser = argparse.ArgumentParser(description="Generate Word2Vec embeddings for log data.")
    parser.add_argument("--log_type", type=str, default=None,
                        help="Optional: Process only a specific log type (e.g., 'vpn').")
    parser.add_argument("--global_attack_list", action="store_true", 
                        help="Use a global list of attack types across all log types.")
    
    args = parser.parse_args()
    
    try:
        # Load Word2Vec model
        print("Step 1: Loading pre-trained Word2Vec model...")
        model = load_pretrained_word2vec()
        if model is None:
            print("❌ Failed to load Word2Vec model. Exiting.")
            return
        
        print(f"✅ Word2Vec model loaded: {model.vector_size}D vectors, {len(model.key_to_index)} vocabulary size")
        
        # Load data
        print("\nStep 2: Loading TFRecord data...")
        df = load_tfrecord_files(log_type_filter=args.log_type)
        
        if df.empty:
            print("❌ No data loaded. Exiting.")
            return
        
        print(f"✅ Loaded {len(df)} log entries")
        
        # Display data distribution
        if args.log_type:
            display_data_distribution(df, args.log_type)
        else:
            display_data_distribution(df, "all combined")
        
        # Process embeddings
        print("\nStep 3: Processing embeddings...")
        df = process_embeddings(df, model, use_global_attack_list=args.global_attack_list)
        
        print(f"✅ Generated embeddings for {len(df)} log entries")
        
        # Save results
        print("\nStep 4: Saving embeddings and labels...")
        
        if args.log_type:
            # Process single log type
            save_embeddings_and_labels(df, model, args.log_type, args.global_attack_list)
        else:
            # Process all log types
            for log_type, group_df in df.groupby('log_type'):
                save_embeddings_and_labels(group_df, model, log_type, args.global_attack_list)
        
        # Create visualization
        print("\nStep 5: Creating visualization...")
        if args.log_type:
            viz_file = OUTPUT_DIR / f"visualization_{args.log_type}.png"
            visualize_embeddings(df, output_file=viz_file)
        else:
            viz_file = OUTPUT_DIR / "visualization_all.png"
            visualize_embeddings(df, output_file=viz_file)
        
        print("\n🎉 Word2Vec embedding generation complete!")
        print(f"📁 Output directory: {OUTPUT_DIR}")
        
        # Print summary
        log_types = df['log_type'].unique()
        for log_type in log_types:
            log_file = OUTPUT_DIR / f"log_{log_type}.pkl"
            label_file = OUTPUT_DIR / f"label_{log_type}.pkl"
            attack_file = OUTPUT_DIR / f"attack_types_{log_type}.txt"
            
            if log_file.exists() and label_file.exists():
                print(f"  📊 {log_type}: {log_file.name}, {label_file.name}, {attack_file.name}")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        raise

if __name__ == '__main__':
    main()

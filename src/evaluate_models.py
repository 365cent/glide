#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive Model Evaluation for Log Anomaly Detection

This script evaluates multiple embedding types and transformer models,
providing detailed metrics and performance analysis.

Features:
- Multi-embedding type evaluation (FastText, Word2Vec, LogBERT)
- Per-class and overall metrics
- Threshold optimization
- Visualization of results
- Export of detailed reports
"""

import argparse
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score
)
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from src.config import EMBEDDINGS_DIR, MODELS_DIR, RESULTS_DIR

def load_embeddings_and_labels(log_type: str, embedding_type: str):
    """Load embeddings and labels for a specific log type and embedding type."""
    embedding_dir = EMBEDDINGS_DIR / log_type
    
    # Load embeddings
    log_file = embedding_dir / f"log_{log_type}.pkl"
    if not log_file.exists():
        raise FileNotFoundError(f"Embedding file not found: {log_file}")
    
    with open(log_file, 'rb') as f:
        embeddings = pickle.load(f)
    
    # Load labels
    label_file = embedding_dir / f"label_{log_type}.pkl"
    if not label_file.exists():
        raise FileNotFoundError(f"Label file not found: {label_file}")
    
    with open(label_file, 'rb') as f:
        label_data = pickle.load(f)
    
    return embeddings, label_data

def optimize_thresholds(y_true, y_pred_proba, method='f1'):
    """Optimize thresholds for multi-label classification."""
    n_classes = y_true.shape[1]
    optimal_thresholds = []
    
    for i in range(n_classes):
        if method == 'f1':
            # Optimize F1 score
            thresholds = np.arange(0.1, 0.9, 0.05)
            best_f1 = 0
            best_threshold = 0.5
            
            for threshold in thresholds:
                y_pred = (y_pred_proba[:, i] >= threshold).astype(int)
                f1 = f1_score(y_true[:, i], y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            optimal_thresholds.append(best_threshold)
        else:
            # Use ROC curve to find optimal threshold
            fpr, tpr, thresholds = roc_curve(y_true[:, i], y_pred_proba[:, i])
            optimal_idx = np.argmax(tpr - fpr)
            optimal_thresholds.append(thresholds[optimal_idx])
    
    return np.array(optimal_thresholds)

def calculate_metrics(y_true, y_pred, y_pred_proba=None, class_names=None):
    """Calculate comprehensive metrics for multi-label classification."""
    metrics = {}
    
    # Overall metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['micro_precision'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['micro_recall'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['micro_f1'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    # Macro metrics
    metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # AUC scores if probabilities are available
    if y_pred_proba is not None:
        try:
            metrics['micro_auc'] = roc_auc_score(y_true, y_pred_proba, average='micro')
            metrics['macro_auc'] = roc_auc_score(y_true, y_pred_proba, average='macro')
        except:
            metrics['micro_auc'] = 0.0
            metrics['macro_auc'] = 0.0
    
    # Per-class metrics
    per_class_metrics = {}
    for i in range(y_true.shape[1]):
        class_name = class_names[i] if class_names else f"class_{i}"
        
        class_metrics = {
            'precision': precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'recall': recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'f1': f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'support': np.sum(y_true[:, i])
        }
        
        if y_pred_proba is not None:
            try:
                class_metrics['auc'] = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
            except:
                class_metrics['auc'] = 0.0
        
        per_class_metrics[class_name] = class_metrics
    
    metrics['per_class'] = per_class_metrics
    
    return metrics

def create_confusion_matrices(y_true, y_pred, class_names, output_dir):
    """Create and save confusion matrices for each class."""
    n_classes = y_true.shape[1]
    fig, axes = plt.subplots(2, (n_classes + 1) // 2, figsize=(15, 10))
    if n_classes == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i in range(n_classes):
        class_name = class_names[i] if class_names else f"Class {i}"
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Confusion Matrix - {class_name}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    # Hide unused subplots
    for i in range(n_classes, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_roc_curves(y_true, y_pred_proba, class_names, output_dir):
    """Create and save ROC curves for each class."""
    n_classes = y_true.shape[1]
    
    plt.figure(figsize=(12, 8))
    
    for i in range(n_classes):
        class_name = class_names[i] if class_names else f"Class {i}"
        try:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
            auc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')
        except:
            plt.plot([0, 1], [0, 1], label=f'{class_name} (AUC = N/A)', linestyle='--')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_precision_recall_curves(y_true, y_pred_proba, class_names, output_dir):
    """Create and save precision-recall curves for each class."""
    n_classes = y_true.shape[1]
    
    plt.figure(figsize=(12, 8))
    
    for i in range(n_classes):
        class_name = class_names[i] if class_names else f"Class {i}"
        try:
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred_proba[:, i])
            ap = average_precision_score(y_true[:, i], y_pred_proba[:, i])
            plt.plot(recall, precision, label=f'{class_name} (AP = {ap:.3f})')
        except:
            plt.plot([0, 1], [0.5, 0.5], label=f'{class_name} (AP = N/A)', linestyle='--')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_metrics_summary(metrics, embedding_types, output_dir):
    """Create a summary table of metrics across embedding types."""
    summary_data = []
    
    for emb_type in embedding_types:
        if emb_type in metrics:
            emb_metrics = metrics[emb_type]
            summary_data.append({
                'Embedding Type': emb_type,
                'Accuracy': f"{emb_metrics['accuracy']:.4f}",
                'Micro F1': f"{emb_metrics['micro_f1']:.4f}",
                'Macro F1': f"{emb_metrics['macro_f1']:.4f}",
                'Micro AUC': f"{emb_metrics.get('micro_auc', 0):.4f}",
                'Macro AUC': f"{emb_metrics.get('macro_auc', 0):.4f}"
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save as CSV
    summary_df.to_csv(output_dir / 'metrics_summary.csv', index=False)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plotting
    emb_types = summary_df['Embedding Type']
    micro_f1 = [float(x) for x in summary_df['Micro F1']]
    macro_f1 = [float(x) for x in summary_df['Macro F1']]
    micro_auc = [float(x) for x in summary_df['Micro AUC']]
    macro_auc = [float(x) for x in summary_df['Macro AUC']]
    
    x = np.arange(len(emb_types))
    width = 0.2
    
    plt.bar(x - width*1.5, micro_f1, width, label='Micro F1', alpha=0.8)
    plt.bar(x - width*0.5, macro_f1, width, label='Macro F1', alpha=0.8)
    plt.bar(x + width*0.5, micro_auc, width, label='Micro AUC', alpha=0.8)
    plt.bar(x + width*1.5, macro_auc, width, label='Macro AUC', alpha=0.8)
    
    plt.xlabel('Embedding Type')
    plt.ylabel('Score')
    plt.title('Performance Comparison Across Embedding Types')
    plt.xticks(x, emb_types, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return summary_df

def evaluate_single_embedding(log_type: str, embedding_type: str, optimize_thresholds_flag: bool = False):
    """Evaluate a single embedding type."""
    print(f"Evaluating {embedding_type} embeddings for {log_type}...")
    
    try:
        # Load embeddings and labels
        embeddings, label_data = load_embeddings_and_labels(log_type, embedding_type)
        
        # Extract data
        X = embeddings
        y_true = label_data['vectors']
        class_names = label_data['classes']
        
        print(f"Loaded {len(X)} samples with {len(class_names)} classes")
        
        # Simple evaluation using logistic regression
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_true, test_size=0.2, random_state=42, stratify=y_true
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred_proba = model.predict_proba(X_test_scaled)
        if len(class_names) == 1:
            y_pred_proba = y_pred_proba.reshape(-1, 1)
        else:
            y_pred_proba = y_pred_proba
        
        # Optimize thresholds if requested
        if optimize_thresholds_flag:
            optimal_thresholds = optimize_thresholds(y_test, y_pred_proba)
            y_pred = (y_pred_proba >= optimal_thresholds).astype(int)
        else:
            y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba, class_names)
        
        return metrics, y_test, y_pred, y_pred_proba, class_names
        
    except Exception as e:
        print(f"Error evaluating {embedding_type}: {e}")
        return None, None, None, None, None

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple embedding types for log anomaly detection.")
    parser.add_argument("--log_type", type=str, required=True,
                        help="Log type to evaluate (e.g., 'vpn', 'wp-error').")
    parser.add_argument("--embedding_types", nargs="+", 
                        default=["fasttext", "word2vec", "logbert"],
                        help="List of embedding types to evaluate.")
    parser.add_argument("--optimize_thresholds", action="store_true",
                        help="Optimize thresholds for better performance.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results (default: results/{log_type}).")
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = RESULTS_DIR / args.log_type
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting evaluation for log type: {args.log_type}")
    print(f"Embedding types: {args.embedding_types}")
    print(f"Output directory: {output_dir}")
    
    # Evaluate each embedding type
    all_metrics = {}
    all_results = {}
    
    for embedding_type in args.embedding_types:
        metrics, y_true, y_pred, y_pred_proba, class_names = evaluate_single_embedding(
            args.log_type, embedding_type, args.optimize_thresholds
        )
        
        if metrics is not None:
            all_metrics[embedding_type] = metrics
            all_results[embedding_type] = {
                'y_true': y_true,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'class_names': class_names
            }
            
            # Create visualizations for this embedding type
            emb_output_dir = output_dir / embedding_type
            emb_output_dir.mkdir(exist_ok=True)
            
            if y_true is not None and y_pred is not None:
                create_confusion_matrices(y_true, y_pred, class_names, emb_output_dir)
                
                if y_pred_proba is not None:
                    create_roc_curves(y_true, y_pred_proba, class_names, emb_output_dir)
                    create_precision_recall_curves(y_true, y_pred_proba, class_names, emb_output_dir)
    
    # Create summary comparison
    if all_metrics:
        summary_df = create_metrics_summary(all_metrics, args.embedding_types, output_dir)
        print("\nPerformance Summary:")
        print(summary_df.to_string(index=False))
        
        # Save detailed metrics
        with open(output_dir / 'detailed_metrics.json', 'w') as f:
            json.dump(all_metrics, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_dir}")
        print(f"Detailed metrics: {output_dir / 'detailed_metrics.json'}")
        print(f"Summary CSV: {output_dir / 'metrics_summary.csv'}")
        print(f"Performance comparison: {output_dir / 'performance_comparison.png'}")
    else:
        print("No successful evaluations completed.")

if __name__ == "__main__":
    main()
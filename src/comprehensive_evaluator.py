#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Comprehensive Evaluation Framework for Anomaly Detection Pipeline

This module provides a streamlined evaluation framework that assesses the performance
of the enhanced anomaly detection pipeline including SMOTE oversampling, hierarchical
classification, and unsupervised transformer models.

Key Features:
- Hierarchical classification evaluation
- SMOTE oversampling quality assessment
- Unsupervised learning performance metrics
- Cross-embedding comparison
- Statistical significance testing
- Concise visualization and reporting

Evaluation Dimensions:
1. Detection Performance: Precision, Recall, F1, AUC-ROC
2. Hierarchical Quality: Level-wise accuracy, path consistency
3. Oversampling Quality: Distribution balance, synthetic quality
4. Unsupervised Metrics: Silhouette score, cluster quality
5. Computational Efficiency: Training time, inference speed

Author: Anomaly Detection Pipeline
Version: 2.0.0
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import warnings
from tqdm import tqdm
import time

# Scientific computing
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, silhouette_score, adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom modules
from smote_oversampling import UnsupervisedSMOTEOversampler
from hierarchical_classifier import UnsupervisedHierarchicalClassifier, ClassificationResult
from transformer import EnhancedTransformerTrainer, TransformerConfig
from config import EMBEDDINGS_DIR, MODELS_DIR, PROCESSED_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ComprehensiveEvaluator")

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    # Detection metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0
    auc_pr: float = 0.0
    
    # Hierarchical metrics
    hierarchical_accuracy: Dict[int, float] = None
    path_consistency: float = 0.0
    level_coverage: Dict[int, float] = None
    
    # Clustering metrics
    silhouette_score: float = 0.0
    adjusted_rand_index: float = 0.0
    normalized_mutual_info: float = 0.0
    
    # SMOTE metrics
    balance_improvement: float = 0.0
    synthetic_quality: float = 0.0
    diversity_score: float = 0.0
    
    # Computational metrics
    training_time: float = 0.0
    inference_time: float = 0.0
    memory_usage: float = 0.0
    
    def __post_init__(self):
        if self.hierarchical_accuracy is None:
            self.hierarchical_accuracy = {}
        if self.level_coverage is None:
            self.level_coverage = {}

@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    # Cross-validation
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    
    # Statistical testing
    significance_level: float = 0.05
    bootstrap_samples: int = 1000
    
    # Visualization
    save_plots: bool = True
    plot_format: str = 'png'
    plot_dpi: int = 300
    
    # Hierarchical evaluation
    max_hierarchy_depth: int = 4
    min_samples_per_level: int = 10
    
    # SMOTE evaluation
    evaluate_synthetic_quality: bool = True
    synthetic_sample_ratio: float = 0.1
    
    # Performance benchmarking
    benchmark_iterations: int = 10
    memory_profiling: bool = True

class ComprehensiveEvaluator:
    """
    Simplified comprehensive evaluator for the enhanced anomaly detection pipeline.
    """
    
    def __init__(self, 
                 config: Optional[EvaluationConfig] = None,
                 output_dir: Path = MODELS_DIR / "evaluation"):
        self.config = config if config else EvaluationConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.smote_oversampler = UnsupervisedSMOTEOversampler()
        self.hierarchical_classifier = UnsupervisedHierarchicalClassifier()
    
    def evaluate_detection_performance(self, 
                                     y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     y_scores: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate detection performance metrics."""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # AUC metrics if scores provided
        if y_scores is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_scores, average='weighted')
                metrics['auc_pr'] = average_precision_score(y_true, y_scores, average='weighted')
            except ValueError:
                metrics['auc_roc'] = 0.0
                metrics['auc_pr'] = 0.0
        
        return metrics
    
    def evaluate_hierarchical_performance(self, 
                                        hierarchical_results: List[ClassificationResult],
                                        true_hierarchy: Optional[Dict] = None) -> Dict[str, Any]:
        """Evaluate hierarchical classification performance."""
        metrics = {}
        
        if not hierarchical_results:
            return metrics
        
        # Level-wise accuracy
        level_accuracies = defaultdict(list)
        path_consistencies = []
        
        for result in hierarchical_results:
            # Level accuracy
            for level, prediction in result.level_predictions.items():
                level_accuracies[level].append(1.0 if prediction else 0.0)
            
            # Path consistency
            if len(result.path) > 1:
                path_consistencies.append(self._calculate_path_consistency([result]))
        
        # Average level accuracies
        for level in level_accuracies:
            metrics[f'level_{level}_accuracy'] = np.mean(level_accuracies[level])
        
        # Overall path consistency
        if path_consistencies:
            metrics['path_consistency'] = np.mean(path_consistencies)
        
        # Coverage metrics
        total_levels = max(len(result.level_predictions) for result in hierarchical_results)
        metrics['level_coverage'] = total_levels / self.config.max_hierarchy_depth
        
        return metrics
    
    def evaluate_smote_quality(self, 
                             original_data: np.ndarray,
                             original_labels: np.ndarray,
                             smote_data: np.ndarray,
                             smote_labels: np.ndarray) -> Dict[str, float]:
        """Evaluate SMOTE oversampling quality."""
        metrics = {}
        
        # Balance improvement
        original_dist = np.bincount(original_labels.astype(int))
        smote_dist = np.bincount(smote_labels.astype(int))
        
        original_balance = np.std(original_dist) / np.mean(original_dist) if np.mean(original_dist) > 0 else 0
        smote_balance = np.std(smote_dist) / np.mean(smote_dist) if np.mean(smote_dist) > 0 else 0
        
        metrics['balance_improvement'] = (original_balance - smote_balance) / (original_balance + 1e-8)
        
        # Synthetic quality (distance to original samples)
        if len(original_data) > 0 and len(smote_data) > 0:
            from sklearn.neighbors import NearestNeighbors
            
            # Find nearest neighbors
            nbrs = NearestNeighbors(n_neighbors=1).fit(original_data)
            distances, _ = nbrs.kneighbors(smote_data)
            
            metrics['synthetic_quality'] = 1.0 / (1.0 + np.mean(distances))
        else:
            metrics['synthetic_quality'] = 0.0
        
        # Diversity score
        if len(smote_data) > 1:
            pca = PCA(n_components=min(2, smote_data.shape[1]))
            smote_pca = pca.fit_transform(smote_data)
            metrics['diversity_score'] = silhouette_score(smote_pca, smote_labels)
        else:
            metrics['diversity_score'] = 0.0
        
        return metrics
    
    def evaluate_clustering_quality(self, 
                                  embeddings: np.ndarray,
                                  cluster_labels: np.ndarray,
                                  true_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate clustering quality metrics."""
        metrics = {}
        
        # Silhouette score
        if len(np.unique(cluster_labels)) > 1:
            metrics['silhouette_score'] = silhouette_score(embeddings, cluster_labels)
        else:
            metrics['silhouette_score'] = 0.0
        
        # Adjusted Rand Index if true labels available
        if true_labels is not None and len(np.unique(true_labels)) > 1:
            metrics['adjusted_rand_index'] = adjusted_rand_score(true_labels, cluster_labels)
            metrics['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, cluster_labels)
        else:
            metrics['adjusted_rand_index'] = 0.0
            metrics['normalized_mutual_info'] = 0.0
        
        return metrics
    
    def evaluate_computational_performance(self, 
                                         model_trainer: Any,
                                         test_data: np.ndarray) -> Dict[str, float]:
        """Evaluate computational performance metrics."""
        metrics = {}
        
        # Inference time
        start_time = time.time()
        predictions = model_trainer.predict_anomalies(test_data)
        inference_time = time.time() - start_time
        
        metrics['inference_time'] = inference_time
        metrics['samples_per_second'] = len(test_data) / inference_time
        
        # Memory usage (approximate)
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        metrics['memory_usage_mb'] = memory_info.rss / 1024 / 1024
        
        return metrics
    
    def perform_cross_validation(self, 
                                embeddings: np.ndarray,
                                labels: np.ndarray,
                                model_class: Any,
                                model_params: Dict) -> Dict[str, Any]:
        """Perform cross-validation evaluation."""
        cv_results = {}
        
        # Stratified k-fold cross-validation
        skf = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
        
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'auc_roc': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(embeddings, labels)):
            X_train, X_val = embeddings[train_idx], embeddings[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            # Train model
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            y_scores = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            fold_metrics = self.evaluate_detection_performance(y_val, y_pred, y_scores)
            
            for metric, value in fold_metrics.items():
                if metric in cv_scores:
                    cv_scores[metric].append(value)
        
        # Average scores
        for metric, scores in cv_scores.items():
            cv_results[f'cv_{metric}_mean'] = np.mean(scores)
            cv_results[f'cv_{metric}_std'] = np.std(scores)
        
        return cv_results
    
    def compare_embedding_methods(self, 
                                embedding_results: Dict[str, Dict],
                                log_type: str) -> Dict[str, Any]:
        """Compare different embedding methods."""
        comparison = {}
        
        # Extract metrics for comparison
        methods = list(embedding_results.keys())
        metrics = ['f1_score', 'auc_roc', 'silhouette_score', 'training_time']
        
        for metric in metrics:
            values = [embedding_results[method].get(metric, 0.0) for method in methods]
            comparison[f'{metric}_by_method'] = dict(zip(methods, values))
        
        # Statistical significance testing
        if len(methods) > 1:
            for metric in ['f1_score', 'auc_roc']:
                values = [embedding_results[method].get(metric, 0.0) for method in methods]
                if len(set(values)) > 1:  # Only test if there's variation
                    # Simple t-test between best and worst
                    best_idx = np.argmax(values)
                    worst_idx = np.argmin(values)
                    t_stat, p_value = stats.ttest_ind([values[best_idx]], [values[worst_idx]])
                    comparison[f'{metric}_significance'] = {
                        'best_method': methods[best_idx],
                        'worst_method': methods[worst_idx],
                        'p_value': p_value,
                        'significant': p_value < self.config.significance_level
                    }
        
        return comparison
    
    def generate_evaluation_report(self, 
                                 results: Dict[str, Any],
                                 log_type: str,
                                 embedding_type: str) -> str:
        """Generate a concise evaluation report."""
        report = []
        report.append("=" * 60)
        report.append(f"EVALUATION REPORT")
        report.append(f"Log Type: {log_type}")
        report.append(f"Embedding Type: {embedding_type}")
        report.append("=" * 60)
        
        # Detection Performance
        if 'detection_metrics' in results:
            metrics = results['detection_metrics']
            report.append("\nDETECTION PERFORMANCE:")
            report.append(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
            report.append(f"  Precision: {metrics.get('precision', 0):.4f}")
            report.append(f"  Recall:    {metrics.get('recall', 0):.4f}")
            report.append(f"  F1-Score:  {metrics.get('f1_score', 0):.4f}")
            report.append(f"  AUC-ROC:   {metrics.get('auc_roc', 0):.4f}")
        
        # Hierarchical Performance
        if 'hierarchical_metrics' in results:
            metrics = results['hierarchical_metrics']
            report.append("\nHIERARCHICAL PERFORMANCE:")
            for level, acc in metrics.items():
                if level.startswith('level_') and level.endswith('_accuracy'):
                    report.append(f"  Level {level.split('_')[1]} Accuracy: {acc:.4f}")
            report.append(f"  Path Consistency: {metrics.get('path_consistency', 0):.4f}")
        
        # Clustering Quality
        if 'clustering_metrics' in results:
            metrics = results['clustering_metrics']
            report.append("\nCLUSTERING QUALITY:")
            report.append(f"  Silhouette Score: {metrics.get('silhouette_score', 0):.4f}")
            report.append(f"  Adjusted Rand Index: {metrics.get('adjusted_rand_index', 0):.4f}")
        
        # SMOTE Quality
        if 'smote_metrics' in results:
            metrics = results['smote_metrics']
            report.append("\nSMOTE OVERSAMPLING QUALITY:")
            report.append(f"  Balance Improvement: {metrics.get('balance_improvement', 0):.4f}")
            report.append(f"  Synthetic Quality: {metrics.get('synthetic_quality', 0):.4f}")
            report.append(f"  Diversity Score: {metrics.get('diversity_score', 0):.4f}")
        
        # Computational Performance
        if 'computational_metrics' in results:
            metrics = results['computational_metrics']
            report.append("\nCOMPUTATIONAL PERFORMANCE:")
            report.append(f"  Inference Time: {metrics.get('inference_time', 0):.4f}s")
            report.append(f"  Samples/Second: {metrics.get('samples_per_second', 0):.1f}")
            report.append(f"  Memory Usage: {metrics.get('memory_usage_mb', 0):.1f}MB")
        
        # Cross-validation Results
        if 'cv_results' in results:
            cv = results['cv_results']
            report.append("\nCROSS-VALIDATION RESULTS:")
            report.append(f"  F1-Score: {cv.get('cv_f1_score_mean', 0):.4f} ± {cv.get('cv_f1_score_std', 0):.4f}")
            report.append(f"  AUC-ROC: {cv.get('cv_auc_roc_mean', 0):.4f} ± {cv.get('cv_auc_roc_std', 0):.4f}")
        
        # Method Comparison
        if 'method_comparison' in results:
            comparison = results['method_comparison']
            report.append("\nMETHOD COMPARISON:")
            for metric, methods in comparison.items():
                if metric.endswith('_by_method'):
                    metric_name = metric.replace('_by_method', '').replace('_', ' ').title()
                    report.append(f"  {metric_name}:")
                    for method, value in methods.items():
                        report.append(f"    {method}: {value:.4f}")
        
        report.append("\n" + "=" * 60)
        return "\n".join(report)
    
    def create_visualizations(self, 
                            results: Dict[str, Any],
                            log_type: str,
                            embedding_type: str):
        """Create concise visualizations."""
        # Create performance summary plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Evaluation Results: {log_type} - {embedding_type}', fontsize=14)
        
        # Detection metrics
        if 'detection_metrics' in results:
            metrics = results['detection_metrics']
            detection_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            values = [metrics.get(m, 0) for m in detection_metrics]
            
            axes[0, 0].bar(detection_metrics, values, color='skyblue')
            axes[0, 0].set_title('Detection Performance')
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Hierarchical metrics
        if 'hierarchical_metrics' in results:
            metrics = results['hierarchical_metrics']
            level_accuracies = {k: v for k, v in metrics.items() if k.startswith('level_') and k.endswith('_accuracy')}
            if level_accuracies:
                levels = [int(k.split('_')[1]) for k in level_accuracies.keys()]
                accuracies = list(level_accuracies.values())
                
                axes[0, 1].plot(levels, accuracies, 'o-', color='green')
                axes[0, 1].set_title('Hierarchical Accuracy by Level')
                axes[0, 1].set_xlabel('Hierarchy Level')
                axes[0, 1].set_ylabel('Accuracy')
                axes[0, 1].set_ylim(0, 1)
        
        # Clustering metrics
        if 'clustering_metrics' in results:
            metrics = results['clustering_metrics']
            clustering_metrics = ['silhouette_score', 'adjusted_rand_index', 'normalized_mutual_info']
            values = [metrics.get(m, 0) for m in clustering_metrics]
            
            axes[1, 0].bar(clustering_metrics, values, color='orange')
            axes[1, 0].set_title('Clustering Quality')
            axes[1, 0].set_ylim(-1, 1)
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # SMOTE metrics
        if 'smote_metrics' in results:
            metrics = results['smote_metrics']
            smote_metrics = ['balance_improvement', 'synthetic_quality', 'diversity_score']
            values = [metrics.get(m, 0) for m in smote_metrics]
            
            axes[1, 1].bar(smote_metrics, values, color='red')
            axes[1, 1].set_title('SMOTE Quality')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"evaluation_{log_type}_{embedding_type}.png"
        plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {plot_path}")
    
    def _calculate_path_consistency(self, results: List[ClassificationResult]) -> float:
        """Calculate path consistency for hierarchical results."""
        if len(results) < 2:
            return 1.0
        
        # Calculate average path similarity
        similarities = []
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                similarity = self._path_similarity(results[i].path, results[j].path)
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 1.0
    
    def _path_similarity(self, path1: List[str], path2: List[str]) -> float:
        """Calculate similarity between two hierarchical paths."""
        if not path1 or not path2:
            return 0.0
        
        # Jaccard similarity
        set1, set2 = set(path1), set(path2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_pipeline(self, 
                         log_type: str,
                         embedding_type: str,
                         embeddings: np.ndarray,
                         labels: np.ndarray) -> Dict[str, Any]:
        """Evaluate the complete pipeline for a specific configuration."""
        results = {}
        
        logger.info(f"Evaluating pipeline for {log_type} with {embedding_type} embeddings")
        
        # 1. Detection Performance
        try:
            # Simple binary classification evaluation
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            y_scores = clf.predict_proba(X_test)[:, 1]
            
            results['detection_metrics'] = self.evaluate_detection_performance(y_test, y_pred, y_scores)
            
        except Exception as e:
            logger.warning(f"Detection evaluation failed: {e}")
            results['detection_metrics'] = {}
        
        # 2. Clustering Quality
        try:
            kmeans = KMeans(n_clusters=min(5, len(np.unique(labels))), random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            results['clustering_metrics'] = self.evaluate_clustering_quality(
                embeddings, cluster_labels, labels
            )
            
        except Exception as e:
            logger.warning(f"Clustering evaluation failed: {e}")
            results['clustering_metrics'] = {}
        
        # 3. SMOTE Quality
        try:
            # Generate pseudo-labels for SMOTE
            pseudo_labels = self.smote_oversampler.generate_pseudo_labels(embeddings)
            
            # Apply SMOTE
            smote_embeddings, smote_labels = self.smote_oversampler.apply_smote_variant(
                embeddings, pseudo_labels, variant='smote'
            )
            
            results['smote_metrics'] = self.evaluate_smote_quality(
                embeddings, pseudo_labels, smote_embeddings, smote_labels
            )
            
        except Exception as e:
            logger.warning(f"SMOTE evaluation failed: {e}")
            results['smote_metrics'] = {}
        
        # 4. Hierarchical Classification
        try:
            hierarchical_results = self.hierarchical_classifier.predict_hierarchical(
                embeddings, confidence_threshold=0.5
            )
            
            results['hierarchical_metrics'] = self.evaluate_hierarchical_performance(
                hierarchical_results
            )
            
        except Exception as e:
            logger.warning(f"Hierarchical evaluation failed: {e}")
            results['hierarchical_metrics'] = {}
        
        # 5. Cross-validation
        try:
            cv_results = self.perform_cross_validation(
                embeddings, labels, RandomForestClassifier, 
                {'n_estimators': 100, 'random_state': 42}
            )
            results['cv_results'] = cv_results
            
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            results['cv_results'] = {}
        
        return results

def main():
    """Main function for command-line evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive evaluation of anomaly detection pipeline")
    parser.add_argument("--log_type", type=str, required=True,
                        help="Log type to evaluate")
    parser.add_argument("--embedding_type", type=str, required=True,
                        choices=['fasttext', 'word2vec', 'logbert'],
                        help="Embedding type to evaluate")
    parser.add_argument("--output_dir", type=str, default=str(MODELS_DIR / "evaluation"),
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(output_dir=args.output_dir)
    
    # Load embeddings and labels
    try:
        from transformer import load_embeddings_and_labels
        embeddings, label_data = load_embeddings_and_labels(args.log_type, args.embedding_type)
        
        # Convert labels to binary (anomaly vs normal)
        labels = np.any(label_data['vectors'], axis=1).astype(int)
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Evaluate pipeline
    results = evaluator.evaluate_pipeline(
        args.log_type, args.embedding_type, embeddings, labels
    )
    
    # Generate report
    report = evaluator.generate_evaluation_report(results, args.log_type, args.embedding_type)
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save report
    report_path = output_path / f"evaluation_report_{args.log_type}_{args.embedding_type}.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Save detailed results
    results_path = output_path / f"evaluation_results_{args.log_type}_{args.embedding_type}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualizations
    evaluator.create_visualizations(results, args.log_type, args.embedding_type)
    
    # Print report
    print(report)
    
    logger.info(f"Evaluation completed. Results saved to {output_path}")

if __name__ == "__main__":
    main()


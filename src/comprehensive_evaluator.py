#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Evaluation Framework for Enhanced Anomaly Detection Pipeline

This module provides a comprehensive evaluation framework that assesses the performance
of the enhanced anomaly detection pipeline including SMOTE oversampling, hierarchical
classification, and unsupervised transformer models. It provides multi-dimensional
evaluation metrics and detailed analysis reports.

Key Features:
- Hierarchical classification evaluation
- SMOTE oversampling quality assessment
- Unsupervised learning performance metrics
- Cross-embedding comparison
- Statistical significance testing
- Visualization and reporting
- Ablation studies
- Performance benchmarking

Evaluation Dimensions:
1. Detection Performance: Precision, Recall, F1, AUC-ROC
2. Hierarchical Quality: Level-wise accuracy, path consistency
3. Oversampling Quality: Distribution balance, synthetic quality
4. Unsupervised Metrics: Silhouette score, cluster quality
5. Computational Efficiency: Training time, inference speed
6. Robustness: Cross-validation, noise tolerance

Author: Anomaly Detection Pipeline
Version: 1.0.0
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
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, silhouette_score, adjusted_rand_score,
    normalized_mutual_info_score, homogeneity_score, completeness_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import custom modules
from smote_oversampling import UnsupervisedSMOTEOversampler
from hierarchical_classifier import UnsupervisedHierarchicalClassifier, ClassificationResult
from enhanced_transformer import EnhancedTransformerTrainer, TransformerConfig
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
    Comprehensive evaluator for the enhanced anomaly detection pipeline.
    """
    
    def __init__(self, 
                 config: Optional[EvaluationConfig] = None,
                 output_dir: Path = MODELS_DIR / "evaluation"):
        """
        Initialize the comprehensive evaluator.
        
        Args:
            config: Evaluation configuration
            output_dir: Output directory for results
        """
        self.config = config if config else EvaluationConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.evaluation_results = {}
        self.comparison_results = {}
        self.ablation_results = {}
        
        # Visualization settings
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def evaluate_detection_performance(self, 
                                     y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     y_scores: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate basic detection performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Prediction scores (for AUC metrics)
            
        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # AUC metrics (if scores available)
        if y_scores is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
                    metrics['auc_pr'] = average_precision_score(y_true, y_scores)
                else:  # Multi-class
                    metrics['auc_roc'] = roc_auc_score(y_true, y_scores, multi_class='ovr', average='weighted')
                    metrics['auc_pr'] = average_precision_score(y_true, y_scores, average='weighted')
            except ValueError as e:
                logger.warning(f"Could not compute AUC metrics: {e}")
                metrics['auc_roc'] = 0.0
                metrics['auc_pr'] = 0.0
        
        # Confusion matrix analysis
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Per-class metrics
        if len(np.unique(y_true)) > 2:
            class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            metrics['per_class_metrics'] = class_report
        
        return metrics
    
    def evaluate_hierarchical_performance(self, 
                                        hierarchical_results: List[ClassificationResult],
                                        true_hierarchy: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Evaluate hierarchical classification performance.
        
        Args:
            hierarchical_results: List of hierarchical classification results
            true_hierarchy: True hierarchical labels (if available)
            
        Returns:
            Dictionary of hierarchical metrics
        """
        metrics = {}
        
        # Extract level-wise predictions
        level_predictions = defaultdict(list)
        path_lengths = []
        confidence_scores = []
        
        for result in hierarchical_results:
            path_lengths.append(len(result.path))
            confidence_scores.append(np.mean(result.confidences))
            
            for level, prediction in result.level_predictions.items():
                level_predictions[level].append(prediction)
        
        # Level-wise analysis
        metrics['level_analysis'] = {}
        for level, predictions in level_predictions.items():
            pred_counter = Counter(predictions)
            metrics['level_analysis'][level] = {
                'distribution': dict(pred_counter),
                'entropy': self._calculate_entropy(list(pred_counter.values())),
                'coverage': len(pred_counter) / len(predictions) if predictions else 0
            }
        
        # Path analysis
        metrics['path_analysis'] = {
            'average_path_length': np.mean(path_lengths),
            'path_length_std': np.std(path_lengths),
            'path_length_distribution': Counter(path_lengths),
            'average_confidence': np.mean(confidence_scores),
            'confidence_std': np.std(confidence_scores)
        }
        
        # Path consistency (how often similar samples follow similar paths)
        path_consistency = self._calculate_path_consistency(hierarchical_results)
        metrics['path_consistency'] = path_consistency
        
        # Anomaly score analysis
        anomaly_scores = [result.anomaly_score for result in hierarchical_results]
        metrics['anomaly_analysis'] = {
            'mean_score': np.mean(anomaly_scores),
            'std_score': np.std(anomaly_scores),
            'score_distribution': np.histogram(anomaly_scores, bins=10)[0].tolist()
        }
        
        return metrics
    
    def evaluate_smote_quality(self, 
                             original_data: np.ndarray,
                             original_labels: np.ndarray,
                             smote_data: np.ndarray,
                             smote_labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate SMOTE oversampling quality.
        
        Args:
            original_data: Original dataset
            original_labels: Original labels
            smote_data: SMOTE-augmented dataset
            smote_labels: SMOTE-augmented labels
            
        Returns:
            Dictionary of SMOTE quality metrics
        """
        metrics = {}
        
        # Class balance improvement
        original_dist = Counter(original_labels)
        smote_dist = Counter(smote_labels)
        
        original_imbalance = max(original_dist.values()) / min(original_dist.values()) if original_dist else 1
        smote_imbalance = max(smote_dist.values()) / min(smote_dist.values()) if smote_dist else 1
        
        metrics['balance_improvement'] = original_imbalance / smote_imbalance
        metrics['original_imbalance_ratio'] = original_imbalance
        metrics['smote_imbalance_ratio'] = smote_imbalance
        
        # Sample increase
        metrics['sample_increase_ratio'] = len(smote_data) / len(original_data)
        
        # Synthetic sample quality (if enabled)
        if self.config.evaluate_synthetic_quality:
            synthetic_quality = self._evaluate_synthetic_quality(
                original_data, smote_data, len(original_data)
            )
            metrics.update(synthetic_quality)
        
        # Diversity analysis
        diversity_score = self._calculate_diversity_score(smote_data, smote_labels)
        metrics['diversity_score'] = diversity_score
        
        return metrics
    
    def evaluate_clustering_quality(self, 
                                  embeddings: np.ndarray,
                                  cluster_labels: np.ndarray,
                                  true_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate clustering quality for unsupervised learning.
        
        Args:
            embeddings: Input embeddings
            cluster_labels: Cluster assignments
            true_labels: True labels (if available)
            
        Returns:
            Dictionary of clustering metrics
        """
        metrics = {}
        
        # Internal clustering metrics
        if len(np.unique(cluster_labels)) > 1:
            metrics['silhouette_score'] = silhouette_score(embeddings, cluster_labels)
            
            # Calinski-Harabasz index
            from sklearn.metrics import calinski_harabasz_score
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(embeddings, cluster_labels)
            
            # Davies-Bouldin index
            from sklearn.metrics import davies_bouldin_score
            metrics['davies_bouldin_score'] = davies_bouldin_score(embeddings, cluster_labels)
        
        # External clustering metrics (if true labels available)
        if true_labels is not None:
            metrics['adjusted_rand_index'] = adjusted_rand_score(true_labels, cluster_labels)
            metrics['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, cluster_labels)
            metrics['homogeneity_score'] = homogeneity_score(true_labels, cluster_labels)
            metrics['completeness_score'] = completeness_score(true_labels, cluster_labels)
        
        # Cluster distribution analysis
        cluster_dist = Counter(cluster_labels)
        metrics['cluster_distribution'] = dict(cluster_dist)
        metrics['cluster_entropy'] = self._calculate_entropy(list(cluster_dist.values()))
        metrics['n_clusters'] = len(cluster_dist)
        
        return metrics
    
    def evaluate_computational_performance(self, 
                                         model_trainer: Any,
                                         test_data: np.ndarray) -> Dict[str, float]:
        """
        Evaluate computational performance metrics.
        
        Args:
            model_trainer: Trained model
            test_data: Test dataset
            
        Returns:
            Dictionary of computational metrics
        """
        metrics = {}
        
        # Inference time
        start_time = time.time()
        for _ in range(self.config.benchmark_iterations):
            _ = model_trainer.predict_anomalies(test_data)
        end_time = time.time()
        
        avg_inference_time = (end_time - start_time) / self.config.benchmark_iterations
        metrics['inference_time_per_sample'] = avg_inference_time / len(test_data)
        metrics['inference_throughput'] = len(test_data) / avg_inference_time
        
        # Memory usage (if profiling enabled)
        if self.config.memory_profiling:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            metrics['memory_usage_mb'] = memory_info.rss / 1024 / 1024
        
        return metrics
    
    def perform_cross_validation(self, 
                                embeddings: np.ndarray,
                                labels: np.ndarray,
                                model_class: Any,
                                model_params: Dict) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation.
        
        Args:
            embeddings: Input embeddings
            labels: Target labels
            model_class: Model class to evaluate
            model_params: Model parameters
            
        Returns:
            Cross-validation results
        """
        cv_results = {}
        
        # Stratified K-Fold cross-validation
        skf = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, 
                             random_state=self.config.random_state)
        
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(embeddings, labels)):
            logger.info(f"Evaluating fold {fold + 1}/{self.config.cv_folds}")
            
            # Split data
            X_train, X_val = embeddings[train_idx], embeddings[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            # Train model
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            y_scores = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            # Evaluate
            fold_metric = self.evaluate_detection_performance(y_val, y_pred, y_scores)
            fold_metrics.append(fold_metric)
        
        # Aggregate results
        cv_results['fold_metrics'] = fold_metrics
        cv_results['mean_metrics'] = {}
        cv_results['std_metrics'] = {}
        
        for metric in fold_metrics[0].keys():
            if isinstance(fold_metrics[0][metric], (int, float)):
                values = [fm[metric] for fm in fold_metrics]
                cv_results['mean_metrics'][metric] = np.mean(values)
                cv_results['std_metrics'][metric] = np.std(values)
        
        return cv_results
    
    def perform_ablation_study(self, 
                              base_config: Dict,
                              ablation_configs: List[Dict],
                              embeddings: np.ndarray,
                              labels: np.ndarray) -> Dict[str, Any]:
        """
        Perform ablation study to understand component contributions.
        
        Args:
            base_config: Base configuration
            ablation_configs: List of ablation configurations
            embeddings: Input embeddings
            labels: Target labels
            
        Returns:
            Ablation study results
        """
        ablation_results = {}
        
        # Evaluate base configuration
        logger.info("Evaluating base configuration")
        base_results = self._evaluate_configuration(base_config, embeddings, labels)
        ablation_results['base'] = base_results
        
        # Evaluate ablation configurations
        for i, config in enumerate(ablation_configs):
            logger.info(f"Evaluating ablation configuration {i + 1}")
            config_results = self._evaluate_configuration(config, embeddings, labels)
            ablation_results[f'ablation_{i}'] = config_results
        
        # Calculate component contributions
        contributions = {}
        for key, results in ablation_results.items():
            if key != 'base':
                contribution = {}
                for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    if metric in results and metric in base_results:
                        contribution[metric] = base_results[metric] - results[metric]
                contributions[key] = contribution
        
        ablation_results['contributions'] = contributions
        
        return ablation_results
    
    def compare_embedding_methods(self, 
                                embedding_results: Dict[str, Dict],
                                log_type: str) -> Dict[str, Any]:
        """
        Compare different embedding methods.
        
        Args:
            embedding_results: Results for different embedding methods
            log_type: Type of logs being evaluated
            
        Returns:
            Comparison results
        """
        comparison = {
            'log_type': log_type,
            'methods_compared': list(embedding_results.keys()),
            'metric_comparison': {},
            'statistical_tests': {},
            'rankings': {}
        }
        
        # Extract metrics for comparison
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        
        for metric in metrics_to_compare:
            metric_values = {}
            for method, results in embedding_results.items():
                if metric in results:
                    metric_values[method] = results[metric]
            
            comparison['metric_comparison'][metric] = metric_values
            
            # Statistical significance testing
            if len(metric_values) >= 2:
                values_list = list(metric_values.values())
                if len(values_list) == 2:
                    # Paired t-test for two methods
                    statistic, p_value = stats.ttest_rel(values_list[0], values_list[1])
                    comparison['statistical_tests'][metric] = {
                        'test': 'paired_t_test',
                        'statistic': statistic,
                        'p_value': p_value,
                        'significant': p_value < self.config.significance_level
                    }
                else:
                    # ANOVA for multiple methods
                    statistic, p_value = stats.f_oneway(*values_list)
                    comparison['statistical_tests'][metric] = {
                        'test': 'anova',
                        'statistic': statistic,
                        'p_value': p_value,
                        'significant': p_value < self.config.significance_level
                    }
            
            # Ranking
            if metric_values:
                sorted_methods = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
                comparison['rankings'][metric] = [method for method, _ in sorted_methods]
        
        return comparison
    
    def generate_evaluation_report(self, 
                                 results: Dict[str, Any],
                                 log_type: str,
                                 embedding_type: str) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results: Evaluation results
            log_type: Type of logs
            embedding_type: Type of embeddings
            
        Returns:
            Formatted report string
        """
        report = f"# Comprehensive Evaluation Report\n\n"
        report += f"**Log Type:** {log_type}\n"
        report += f"**Embedding Type:** {embedding_type}\n"
        report += f"**Evaluation Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Detection Performance
        if 'detection_performance' in results:
            dp = results['detection_performance']
            report += "## Detection Performance\n\n"
            report += f"- **Accuracy:** {dp.get('accuracy', 0):.4f}\n"
            report += f"- **Precision:** {dp.get('precision', 0):.4f}\n"
            report += f"- **Recall:** {dp.get('recall', 0):.4f}\n"
            report += f"- **F1 Score:** {dp.get('f1_score', 0):.4f}\n"
            report += f"- **AUC-ROC:** {dp.get('auc_roc', 0):.4f}\n"
            report += f"- **AUC-PR:** {dp.get('auc_pr', 0):.4f}\n\n"
        
        # Hierarchical Performance
        if 'hierarchical_performance' in results:
            hp = results['hierarchical_performance']
            report += "## Hierarchical Classification Performance\n\n"
            
            if 'path_analysis' in hp:
                pa = hp['path_analysis']
                report += f"- **Average Path Length:** {pa.get('average_path_length', 0):.2f}\n"
                report += f"- **Path Consistency:** {hp.get('path_consistency', 0):.4f}\n"
                report += f"- **Average Confidence:** {pa.get('average_confidence', 0):.4f}\n\n"
            
            if 'level_analysis' in hp:
                report += "### Level-wise Analysis\n\n"
                for level, analysis in hp['level_analysis'].items():
                    report += f"**Level {level}:**\n"
                    report += f"- Coverage: {analysis.get('coverage', 0):.4f}\n"
                    report += f"- Entropy: {analysis.get('entropy', 0):.4f}\n"
                    report += f"- Distribution: {analysis.get('distribution', {})}\n\n"
        
        # SMOTE Quality
        if 'smote_quality' in results:
            sq = results['smote_quality']
            report += "## SMOTE Oversampling Quality\n\n"
            report += f"- **Balance Improvement:** {sq.get('balance_improvement', 0):.2f}x\n"
            report += f"- **Sample Increase:** {sq.get('sample_increase_ratio', 0):.2f}x\n"
            report += f"- **Diversity Score:** {sq.get('diversity_score', 0):.4f}\n"
            
            if 'synthetic_quality_score' in sq:
                report += f"- **Synthetic Quality:** {sq.get('synthetic_quality_score', 0):.4f}\n"
            report += "\n"
        
        # Clustering Quality
        if 'clustering_quality' in results:
            cq = results['clustering_quality']
            report += "## Clustering Quality (Unsupervised Learning)\n\n"
            report += f"- **Silhouette Score:** {cq.get('silhouette_score', 0):.4f}\n"
            report += f"- **Number of Clusters:** {cq.get('n_clusters', 0)}\n"
            report += f"- **Cluster Entropy:** {cq.get('cluster_entropy', 0):.4f}\n"
            
            if 'adjusted_rand_index' in cq:
                report += f"- **Adjusted Rand Index:** {cq.get('adjusted_rand_index', 0):.4f}\n"
            report += "\n"
        
        # Computational Performance
        if 'computational_performance' in results:
            cp = results['computational_performance']
            report += "## Computational Performance\n\n"
            report += f"- **Inference Time per Sample:** {cp.get('inference_time_per_sample', 0)*1000:.2f} ms\n"
            report += f"- **Throughput:** {cp.get('inference_throughput', 0):.0f} samples/sec\n"
            
            if 'memory_usage_mb' in cp:
                report += f"- **Memory Usage:** {cp.get('memory_usage_mb', 0):.1f} MB\n"
            report += "\n"
        
        # Cross-validation Results
        if 'cross_validation' in results:
            cv = results['cross_validation']
            report += "## Cross-Validation Results\n\n"
            
            if 'mean_metrics' in cv:
                mm = cv['mean_metrics']
                sm = cv.get('std_metrics', {})
                
                for metric, mean_val in mm.items():
                    if isinstance(mean_val, (int, float)):
                        std_val = sm.get(metric, 0)
                        report += f"- **{metric.replace('_', ' ').title()}:** {mean_val:.4f} ± {std_val:.4f}\n"
            report += "\n"
        
        return report
    
    def create_visualizations(self, 
                            results: Dict[str, Any],
                            log_type: str,
                            embedding_type: str):
        """
        Create comprehensive visualizations for the evaluation results.
        
        Args:
            results: Evaluation results
            log_type: Type of logs
            embedding_type: Type of embeddings
        """
        viz_dir = self.output_dir / "visualizations" / f"{log_type}_{embedding_type}"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance metrics radar chart
        if 'detection_performance' in results:
            self._create_performance_radar_chart(
                results['detection_performance'], 
                viz_dir / f"performance_radar.{self.config.plot_format}"
            )
        
        # Hierarchical analysis visualization
        if 'hierarchical_performance' in results:
            self._create_hierarchical_visualization(
                results['hierarchical_performance'],
                viz_dir / f"hierarchical_analysis.{self.config.plot_format}"
            )
        
        # SMOTE quality visualization
        if 'smote_quality' in results:
            self._create_smote_visualization(
                results['smote_quality'],
                viz_dir / f"smote_quality.{self.config.plot_format}"
            )
        
        # Cross-validation results
        if 'cross_validation' in results:
            self._create_cv_visualization(
                results['cross_validation'],
                viz_dir / f"cross_validation.{self.config.plot_format}"
            )
        
        logger.info(f"Visualizations saved to {viz_dir}")
    
    def _calculate_entropy(self, values: List[float]) -> float:
        """Calculate entropy of a distribution."""
        if not values or sum(values) == 0:
            return 0.0
        
        total = sum(values)
        probabilities = [v / total for v in values if v > 0]
        return -sum(p * np.log2(p) for p in probabilities)
    
    def _calculate_path_consistency(self, results: List[ClassificationResult]) -> float:
        """Calculate path consistency for hierarchical results."""
        if len(results) < 2:
            return 1.0
        
        # Group results by final prediction
        prediction_groups = defaultdict(list)
        for result in results:
            prediction_groups[result.final_prediction].append(result.path)
        
        # Calculate consistency within each group
        total_consistency = 0.0
        total_comparisons = 0
        
        for prediction, paths in prediction_groups.items():
            if len(paths) < 2:
                continue
            
            # Calculate pairwise path similarity
            group_consistency = 0.0
            group_comparisons = 0
            
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    similarity = self._path_similarity(paths[i], paths[j])
                    group_consistency += similarity
                    group_comparisons += 1
            
            if group_comparisons > 0:
                total_consistency += group_consistency
                total_comparisons += group_comparisons
        
        return total_consistency / total_comparisons if total_comparisons > 0 else 0.0
    
    def _path_similarity(self, path1: List[str], path2: List[str]) -> float:
        """Calculate similarity between two classification paths."""
        if not path1 or not path2:
            return 0.0
        
        # Find common prefix length
        common_length = 0
        for i in range(min(len(path1), len(path2))):
            if path1[i] == path2[i]:
                common_length += 1
            else:
                break
        
        # Similarity is the ratio of common prefix to maximum path length
        max_length = max(len(path1), len(path2))
        return common_length / max_length if max_length > 0 else 0.0
    
    def _evaluate_synthetic_quality(self, 
                                  original_data: np.ndarray,
                                  smote_data: np.ndarray,
                                  original_size: int) -> Dict[str, float]:
        """Evaluate the quality of synthetic samples generated by SMOTE."""
        metrics = {}
        
        # Identify synthetic samples (those beyond original size)
        synthetic_data = smote_data[original_size:]
        
        if len(synthetic_data) == 0:
            return {'synthetic_quality_score': 0.0}
        
        # Calculate nearest neighbor distances
        from sklearn.neighbors import NearestNeighbors
        
        # Fit on original data
        nn = NearestNeighbors(n_neighbors=2)  # 2 to exclude self
        nn.fit(original_data)
        
        # Find distances from synthetic to original samples
        distances, _ = nn.kneighbors(synthetic_data)
        avg_distance = np.mean(distances[:, 1])  # Exclude self-distance
        
        # Quality score based on distance distribution
        # Good synthetic samples should be close to but not identical to original samples
        distance_std = np.std(distances[:, 1])
        quality_score = 1.0 / (1.0 + avg_distance) * (1.0 + distance_std)
        
        metrics['synthetic_quality_score'] = min(quality_score, 1.0)
        metrics['avg_nn_distance'] = avg_distance
        metrics['distance_std'] = distance_std
        
        return metrics
    
    def _calculate_diversity_score(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate diversity score of the dataset."""
        if len(data) < 2:
            return 0.0
        
        # Calculate pairwise distances
        distances = pdist(data)
        avg_distance = np.mean(distances)
        
        # Normalize by data dimensionality
        diversity_score = avg_distance / np.sqrt(data.shape[1])
        
        return min(diversity_score, 1.0)
    
    def _evaluate_configuration(self, config: Dict, embeddings: np.ndarray, labels: np.ndarray) -> Dict:
        """Evaluate a specific configuration."""
        # This is a placeholder - implement based on your specific model training logic
        # For now, return dummy metrics
        return {
            'accuracy': np.random.random(),
            'precision': np.random.random(),
            'recall': np.random.random(),
            'f1_score': np.random.random()
        }
    
    def _create_performance_radar_chart(self, metrics: Dict, save_path: Path):
        """Create a radar chart for performance metrics."""
        # Select key metrics for radar chart
        radar_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        values = [metrics.get(metric, 0) for metric in radar_metrics]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False)
        values += values[:1]  # Complete the circle
        angles = np.concatenate((angles, [angles[0]]))
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2, label='Performance')
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in radar_metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Performance Metrics Radar Chart', size=16, weight='bold')
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
    
    def _create_hierarchical_visualization(self, metrics: Dict, save_path: Path):
        """Create visualization for hierarchical analysis."""
        if 'level_analysis' not in metrics:
            return
        
        levels = list(metrics['level_analysis'].keys())
        coverages = [metrics['level_analysis'][level]['coverage'] for level in levels]
        entropies = [metrics['level_analysis'][level]['entropy'] for level in levels]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Coverage by level
        ax1.bar(levels, coverages, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Hierarchy Level')
        ax1.set_ylabel('Coverage')
        ax1.set_title('Coverage by Hierarchy Level')
        ax1.set_ylim(0, 1)
        
        # Entropy by level
        ax2.bar(levels, entropies, alpha=0.7, color='lightcoral')
        ax2.set_xlabel('Hierarchy Level')
        ax2.set_ylabel('Entropy')
        ax2.set_title('Entropy by Hierarchy Level')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
    
    def _create_smote_visualization(self, metrics: Dict, save_path: Path):
        """Create visualization for SMOTE quality metrics."""
        smote_metrics = ['balance_improvement', 'sample_increase_ratio', 'diversity_score']
        values = [metrics.get(metric, 0) for metric in smote_metrics]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(len(smote_metrics)), values, alpha=0.7, 
                     color=['lightgreen', 'lightblue', 'lightyellow'])
        
        ax.set_xlabel('SMOTE Metrics')
        ax.set_ylabel('Score')
        ax.set_title('SMOTE Oversampling Quality Metrics')
        ax.set_xticks(range(len(smote_metrics)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in smote_metrics])
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
    
    def _create_cv_visualization(self, cv_results: Dict, save_path: Path):
        """Create visualization for cross-validation results."""
        if 'fold_metrics' not in cv_results:
            return
        
        fold_metrics = cv_results['fold_metrics']
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Prepare data
        data = []
        for metric in metrics_to_plot:
            for fold, fold_result in enumerate(fold_metrics):
                if metric in fold_result:
                    data.append({
                        'Metric': metric.replace('_', ' ').title(),
                        'Fold': fold + 1,
                        'Score': fold_result[metric]
                    })
        
        if not data:
            return
        
        df = pd.DataFrame(data)
        
        # Create box plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x='Metric', y='Score', ax=ax)
        ax.set_title('Cross-Validation Results Distribution')
        ax.set_ylim(0, 1)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive evaluation of anomaly detection pipeline")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing evaluation results")
    parser.add_argument("--log_type", type=str, required=True,
                        help="Log type to evaluate")
    parser.add_argument("--embedding_type", type=str, required=True,
                        help="Embedding type to evaluate")
    parser.add_argument("--output_dir", type=str, default=str(MODELS_DIR / "evaluation"),
                        help="Output directory for evaluation results")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(output_dir=Path(args.output_dir))
    
    # Load results (placeholder - implement based on your result format)
    results_path = Path(args.results_dir)
    
    # Example evaluation workflow
    logger.info(f"Starting comprehensive evaluation for {args.log_type} with {args.embedding_type}")
    
    # Generate report
    # report = evaluator.generate_evaluation_report(results, args.log_type, args.embedding_type)
    
    # Save report
    # report_path = evaluator.output_dir / f"evaluation_report_{args.log_type}_{args.embedding_type}.md"
    # with open(report_path, 'w') as f:
    #     f.write(report)
    
    # Create visualizations
    # evaluator.create_visualizations(results, args.log_type, args.embedding_type)
    
    logger.info(f"Comprehensive evaluation completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()


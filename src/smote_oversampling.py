#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMOTE-based Oversampling for Imbalanced Log Data

This module implements advanced SMOTE (Synthetic Minority Oversampling Technique) 
for handling imbalanced log data in an unsupervised manner. It provides multiple
SMOTE variants optimized for different scenarios and maintains compatibility with
the existing embedding pipeline.

Key Features:
- Multiple SMOTE variants (SMOTE, BorderlineSMOTE, ADASYN, SVMSMOTE)
- Unsupervised approach using clustering for pseudo-labeling
- Hierarchical oversampling for multi-level attack taxonomies
- Adaptive sampling ratios based on class distribution
- Memory-efficient processing for large datasets
- Comprehensive evaluation metrics for oversampling quality

Author: Anomaly Detection Pipeline
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from imblearn.over_sampling import (
    SMOTE, BorderlineSMOTE, ADASYN, SVMSMOTE, 
    RandomOverSampler, KMeansSMOTE
)
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from halo import Halo
import json

# Import configuration
from config import EMBEDDINGS_DIR, PROCESSED_DIR, MODELS_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SMOTEOversampler")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class UnsupervisedSMOTEOversampler:
    """
    Advanced SMOTE oversampling for unsupervised anomaly detection.
    
    This class implements multiple SMOTE variants and uses clustering-based
    pseudo-labeling to create balanced datasets for training unsupervised
    anomaly detection models.
    """
    
    def __init__(self, 
                 embedding_dir: Path = EMBEDDINGS_DIR,
                 output_dir: Path = MODELS_DIR / "smote_balanced",
                 random_state: int = 42):
        """
        Initialize the SMOTE oversampler.
        
        Args:
            embedding_dir: Directory containing embedding files
            output_dir: Directory to save balanced datasets
            random_state: Random state for reproducibility
        """
        self.embedding_dir = Path(embedding_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        # SMOTE variants configuration
        self.smote_variants = {
            'smote': SMOTE(random_state=random_state),
            'borderline': BorderlineSMOTE(random_state=random_state),
            'adasyn': ADASYN(random_state=random_state),
            'svm_smote': SVMSMOTE(random_state=random_state),
            'kmeans_smote': KMeansSMOTE(random_state=random_state),
            'smote_enn': SMOTEENN(random_state=random_state),
            'smote_tomek': SMOTETomek(random_state=random_state)
        }
        
        # Clustering algorithms for pseudo-labeling
        self.clustering_algorithms = {
            'kmeans': KMeans(random_state=random_state),
            'dbscan': DBSCAN()
        }
        
        # Statistics tracking
        self.oversampling_stats = {}
        
    def load_embeddings_and_labels(self, 
                                 log_type: str, 
                                 embedding_type: str = 'logbert') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load embeddings and labels for a specific log type and embedding method.
        
        Args:
            log_type: Type of logs (e.g., 'vpn', 'wp-access')
            embedding_type: Type of embeddings ('fasttext', 'word2vec', 'logbert')
            
        Returns:
            Tuple of (embeddings, labels, attack_types)
        """
        # Construct paths
        embedding_path = self.embedding_dir / embedding_type / log_type / f"log_{log_type}.pkl"
        label_path = self.embedding_dir / embedding_type / log_type / f"label_{log_type}.pkl"
        
        if not embedding_path.exists() or not label_path.exists():
            raise FileNotFoundError(f"Embedding or label files not found for {log_type} with {embedding_type}")
        
        # Load embeddings
        with open(embedding_path, 'rb') as f:
            embeddings = pickle.load(f)
        
        # Load labels
        with open(label_path, 'rb') as f:
            label_data = pickle.load(f)
        
        labels = label_data['vectors']
        attack_types = label_data['classes']
        
        logger.info(f"Loaded {len(embeddings)} embeddings and {len(labels)} labels for {log_type}")
        logger.info(f"Embedding dimension: {embeddings.shape[1]}")
        logger.info(f"Number of attack types: {len(attack_types)}")
        
        return embeddings, labels, attack_types
    
    def analyze_class_imbalance(self, 
                              labels: np.ndarray, 
                              attack_types: List[str]) -> Dict[str, Dict]:
        """
        Analyze class imbalance in the dataset.
        
        Args:
            labels: Binary label matrix
            attack_types: List of attack type names
            
        Returns:
            Dictionary containing imbalance analysis
        """
        analysis = {}
        
        # Overall statistics
        total_samples = len(labels)
        normal_samples = np.sum(np.sum(labels, axis=1) == 0)
        attack_samples = total_samples - normal_samples
        
        analysis['overall'] = {
            'total_samples': total_samples,
            'normal_samples': normal_samples,
            'attack_samples': attack_samples,
            'imbalance_ratio': normal_samples / max(attack_samples, 1)
        }
        
        # Per-class statistics
        analysis['per_class'] = {}
        for i, attack_type in enumerate(attack_types):
            positive_samples = np.sum(labels[:, i])
            negative_samples = total_samples - positive_samples
            
            analysis['per_class'][attack_type] = {
                'positive_samples': int(positive_samples),
                'negative_samples': int(negative_samples),
                'imbalance_ratio': negative_samples / max(positive_samples, 1),
                'prevalence': positive_samples / total_samples
            }
        
        return analysis
    
    def generate_pseudo_labels(self, 
                             embeddings: np.ndarray, 
                             method: str = 'kmeans',
                             n_clusters: Optional[int] = None) -> np.ndarray:
        """
        Generate pseudo-labels using clustering for unsupervised learning.
        
        Args:
            embeddings: Input embeddings
            method: Clustering method ('kmeans' or 'dbscan')
            n_clusters: Number of clusters (for kmeans)
            
        Returns:
            Pseudo-labels array
        """
        logger.info(f"Generating pseudo-labels using {method}")
        
        # Normalize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        if method == 'kmeans':
            if n_clusters is None:
                # Determine optimal number of clusters using elbow method
                n_clusters = self._find_optimal_clusters(embeddings_scaled)
            
            clusterer = KMeans(n_clusters=n_clusters, random_state=self.random_state)
            pseudo_labels = clusterer.fit_predict(embeddings_scaled)
            
        elif method == 'dbscan':
            # Use DBSCAN for density-based clustering
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            pseudo_labels = clusterer.fit_predict(embeddings_scaled)
            
            # Handle noise points (label -1) by assigning them to nearest cluster
            if -1 in pseudo_labels:
                noise_mask = pseudo_labels == -1
                if np.sum(~noise_mask) > 0:  # If there are non-noise points
                    # Find nearest neighbors for noise points
                    nn = NearestNeighbors(n_neighbors=1)
                    nn.fit(embeddings_scaled[~noise_mask])
                    _, indices = nn.kneighbors(embeddings_scaled[noise_mask])
                    pseudo_labels[noise_mask] = pseudo_labels[~noise_mask][indices.flatten()]
        
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        logger.info(f"Generated {len(np.unique(pseudo_labels))} pseudo-label clusters")
        return pseudo_labels
    
    def _find_optimal_clusters(self, embeddings: np.ndarray, max_clusters: int = 20) -> int:
        """
        Find optimal number of clusters using elbow method and silhouette score.
        
        Args:
            embeddings: Input embeddings
            max_clusters: Maximum number of clusters to test
            
        Returns:
            Optimal number of clusters
        """
        max_clusters = min(max_clusters, len(embeddings) // 10)  # Ensure reasonable cluster size
        
        if max_clusters < 2:
            return 2
        
        silhouette_scores = []
        inertias = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            labels = kmeans.fit_predict(embeddings)
            
            silhouette_scores.append(silhouette_score(embeddings, labels))
            inertias.append(kmeans.inertia_)
        
        # Find elbow point
        optimal_k = 2 + np.argmax(silhouette_scores)
        
        logger.info(f"Optimal number of clusters: {optimal_k}")
        return optimal_k
    
    def apply_smote_variant(self, 
                          embeddings: np.ndarray, 
                          labels: np.ndarray, 
                          variant: str = 'smote',
                          sampling_strategy: Union[str, Dict] = 'auto') -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply a specific SMOTE variant to the data.
        
        Args:
            embeddings: Input embeddings
            labels: Target labels
            variant: SMOTE variant to use
            sampling_strategy: Sampling strategy
            
        Returns:
            Tuple of (resampled_embeddings, resampled_labels)
        """
        if variant not in self.smote_variants:
            raise ValueError(f"Unknown SMOTE variant: {variant}")
        
        smote = self.smote_variants[variant]
        
        # Configure sampling strategy
        if isinstance(sampling_strategy, str) and sampling_strategy == 'auto':
            # Automatically determine sampling strategy based on class distribution
            class_counts = Counter(labels)
            majority_count = max(class_counts.values())
            sampling_strategy = {cls: majority_count for cls in class_counts.keys() 
                               if class_counts[cls] < majority_count * 0.5}
        
        try:
            # Apply SMOTE
            embeddings_resampled, labels_resampled = smote.fit_resample(embeddings, labels)
            
            logger.info(f"Applied {variant}: {len(embeddings)} -> {len(embeddings_resampled)} samples")
            return embeddings_resampled, labels_resampled
            
        except Exception as e:
            logger.warning(f"Failed to apply {variant}: {e}")
            logger.info("Falling back to random oversampling")
            
            # Fallback to random oversampling
            ros = RandomOverSampler(random_state=self.random_state, sampling_strategy=sampling_strategy)
            return ros.fit_resample(embeddings, labels)
    
    def hierarchical_oversampling(self, 
                                embeddings: np.ndarray, 
                                labels: np.ndarray, 
                                attack_types: List[str],
                                hierarchy: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply hierarchical oversampling based on attack taxonomy.
        
        Args:
            embeddings: Input embeddings
            labels: Multi-label binary matrix
            attack_types: List of attack type names
            hierarchy: Attack hierarchy dictionary
            
        Returns:
            Tuple of (resampled_embeddings, resampled_labels)
        """
        if hierarchy is None:
            # Create a simple hierarchy based on attack type patterns
            hierarchy = self._create_default_hierarchy(attack_types)
        
        logger.info("Applying hierarchical oversampling")
        
        # Start with original data
        current_embeddings = embeddings.copy()
        current_labels = labels.copy()
        
        # Apply oversampling at each hierarchy level
        for level, attack_groups in hierarchy.items():
            logger.info(f"Processing hierarchy level: {level}")
            
            for group_name, attack_list in attack_groups.items():
                # Find indices of attacks in this group
                attack_indices = [i for i, attack in enumerate(attack_types) if attack in attack_list]
                
                if not attack_indices:
                    continue
                
                # Create binary labels for this group
                group_labels = np.any(current_labels[:, attack_indices], axis=1).astype(int)
                
                # Apply SMOTE to this group
                try:
                    group_embeddings_resampled, group_labels_resampled = self.apply_smote_variant(
                        current_embeddings, group_labels, variant='smote'
                    )
                    
                    # Update current data
                    current_embeddings = group_embeddings_resampled
                    
                    # Expand labels to match new samples
                    new_samples = len(group_labels_resampled) - len(current_labels)
                    if new_samples > 0:
                        # Add new label rows (copy from similar samples)
                        additional_labels = np.zeros((new_samples, current_labels.shape[1]), dtype=current_labels.dtype)
                        current_labels = np.vstack([current_labels, additional_labels])
                    
                except Exception as e:
                    logger.warning(f"Failed hierarchical oversampling for group {group_name}: {e}")
        
        return current_embeddings, current_labels
    
    def _create_default_hierarchy(self, attack_types: List[str]) -> Dict:
        """
        Create a default attack hierarchy based on common patterns.
        
        Args:
            attack_types: List of attack type names
            
        Returns:
            Hierarchy dictionary
        """
        hierarchy = {
            'level_1': {
                'web_attacks': [attack for attack in attack_types if any(web_term in attack.lower() 
                               for web_term in ['xss', 'sql', 'injection', 'csrf', 'lfi', 'rfi'])],
                'network_attacks': [attack for attack in attack_types if any(net_term in attack.lower() 
                                   for net_term in ['dos', 'ddos', 'scan', 'brute', 'flood'])],
                'system_attacks': [attack for attack in attack_types if any(sys_term in attack.lower() 
                                  for sys_term in ['privilege', 'escalation', 'backdoor', 'trojan'])],
                'data_attacks': [attack for attack in attack_types if any(data_term in attack.lower() 
                                for data_term in ['exfiltration', 'leak', 'theft', 'breach'])]
            }
        }
        
        # Add remaining attacks to 'other' category
        categorized_attacks = set()
        for group in hierarchy['level_1'].values():
            categorized_attacks.update(group)
        
        remaining_attacks = [attack for attack in attack_types if attack not in categorized_attacks]
        if remaining_attacks:
            hierarchy['level_1']['other'] = remaining_attacks
        
        return hierarchy
    
    def evaluate_oversampling_quality(self, 
                                    original_embeddings: np.ndarray,
                                    original_labels: np.ndarray,
                                    resampled_embeddings: np.ndarray,
                                    resampled_labels: np.ndarray) -> Dict:
        """
        Evaluate the quality of oversampling.
        
        Args:
            original_embeddings: Original embeddings
            original_labels: Original labels
            resampled_embeddings: Resampled embeddings
            resampled_labels: Resampled labels
            
        Returns:
            Quality metrics dictionary
        """
        metrics = {}
        
        # Basic statistics
        metrics['sample_increase'] = len(resampled_embeddings) / len(original_embeddings)
        
        # Class distribution comparison
        original_dist = Counter(original_labels) if original_labels.ndim == 1 else Counter(np.argmax(original_labels, axis=1))
        resampled_dist = Counter(resampled_labels) if resampled_labels.ndim == 1 else Counter(np.argmax(resampled_labels, axis=1))
        
        metrics['original_distribution'] = dict(original_dist)
        metrics['resampled_distribution'] = dict(resampled_dist)
        
        # Imbalance ratio improvement
        original_ratios = [max(original_dist.values()) / count for count in original_dist.values()]
        resampled_ratios = [max(resampled_dist.values()) / count for count in resampled_dist.values()]
        
        metrics['imbalance_improvement'] = {
            'original_max_ratio': max(original_ratios),
            'resampled_max_ratio': max(resampled_ratios),
            'improvement_factor': max(original_ratios) / max(resampled_ratios)
        }
        
        return metrics
    
    def process_log_type(self, 
                        log_type: str, 
                        embedding_types: List[str] = ['fasttext', 'word2vec', 'logbert'],
                        smote_variants: List[str] = ['smote', 'borderline', 'adasyn'],
                        use_hierarchical: bool = True,
                        use_pseudo_labels: bool = True) -> Dict:
        """
        Process a specific log type with multiple SMOTE variants and embedding types.
        
        Args:
            log_type: Type of logs to process
            embedding_types: List of embedding types to process
            smote_variants: List of SMOTE variants to apply
            use_hierarchical: Whether to use hierarchical oversampling
            use_pseudo_labels: Whether to generate pseudo-labels
            
        Returns:
            Processing results dictionary
        """
        results = {}
        
        for embedding_type in embedding_types:
            logger.info(f"Processing {log_type} with {embedding_type} embeddings")
            
            try:
                # Load data
                embeddings, labels, attack_types = self.load_embeddings_and_labels(log_type, embedding_type)
                
                # Analyze class imbalance
                imbalance_analysis = self.analyze_class_imbalance(labels, attack_types)
                
                # Generate pseudo-labels if requested
                if use_pseudo_labels:
                    pseudo_labels = self.generate_pseudo_labels(embeddings)
                    # Convert to binary format for SMOTE
                    pseudo_labels_binary = np.eye(len(np.unique(pseudo_labels)))[pseudo_labels]
                else:
                    pseudo_labels_binary = labels
                
                embedding_results = {
                    'original_stats': imbalance_analysis,
                    'variants': {}
                }
                
                # Apply each SMOTE variant
                for variant in smote_variants:
                    logger.info(f"Applying {variant} to {embedding_type} embeddings")
                    
                    try:
                        if use_hierarchical and labels.shape[1] > 1:
                            # Use hierarchical oversampling for multi-label data
                            resampled_embeddings, resampled_labels = self.hierarchical_oversampling(
                                embeddings, labels, attack_types
                            )
                        else:
                            # Convert multi-label to single-label for SMOTE
                            if labels.shape[1] > 1:
                                single_labels = np.argmax(labels, axis=1)
                            else:
                                single_labels = labels.flatten()
                            
                            resampled_embeddings, resampled_labels = self.apply_smote_variant(
                                embeddings, single_labels, variant
                            )
                        
                        # Evaluate oversampling quality
                        quality_metrics = self.evaluate_oversampling_quality(
                            embeddings, labels if labels.shape[1] == 1 else np.argmax(labels, axis=1),
                            resampled_embeddings, resampled_labels
                        )
                        
                        # Save resampled data
                        output_path = self.output_dir / f"{log_type}_{embedding_type}_{variant}"
                        output_path.mkdir(parents=True, exist_ok=True)
                        
                        with open(output_path / "embeddings.pkl", 'wb') as f:
                            pickle.dump(resampled_embeddings, f)
                        
                        with open(output_path / "labels.pkl", 'wb') as f:
                            pickle.dump(resampled_labels, f)
                        
                        with open(output_path / "metadata.json", 'w') as f:
                            json.dump({
                                'log_type': log_type,
                                'embedding_type': embedding_type,
                                'smote_variant': variant,
                                'attack_types': attack_types,
                                'quality_metrics': quality_metrics,
                                'original_shape': embeddings.shape,
                                'resampled_shape': resampled_embeddings.shape
                            }, f, indent=2)
                        
                        embedding_results['variants'][variant] = {
                            'output_path': str(output_path),
                            'quality_metrics': quality_metrics,
                            'original_samples': len(embeddings),
                            'resampled_samples': len(resampled_embeddings)
                        }
                        
                        logger.info(f"Saved {variant} results to {output_path}")
                        
                    except Exception as e:
                        logger.error(f"Failed to process {variant} for {embedding_type}: {e}")
                        embedding_results['variants'][variant] = {'error': str(e)}
                
                results[embedding_type] = embedding_results
                
            except Exception as e:
                logger.error(f"Failed to process {embedding_type} embeddings: {e}")
                results[embedding_type] = {'error': str(e)}
        
        return results
    
    def generate_oversampling_report(self, results: Dict, log_type: str) -> str:
        """
        Generate a comprehensive oversampling report.
        
        Args:
            results: Processing results dictionary
            log_type: Log type name
            
        Returns:
            Report string
        """
        report = f"# SMOTE Oversampling Report for {log_type.upper()}\n\n"
        
        for embedding_type, embedding_results in results.items():
            if 'error' in embedding_results:
                report += f"## {embedding_type.upper()} Embeddings\n"
                report += f"**Error:** {embedding_results['error']}\n\n"
                continue
            
            report += f"## {embedding_type.upper()} Embeddings\n\n"
            
            # Original statistics
            original_stats = embedding_results['original_stats']
            report += f"### Original Data Statistics\n"
            report += f"- Total samples: {original_stats['overall']['total_samples']}\n"
            report += f"- Normal samples: {original_stats['overall']['normal_samples']}\n"
            report += f"- Attack samples: {original_stats['overall']['attack_samples']}\n"
            report += f"- Imbalance ratio: {original_stats['overall']['imbalance_ratio']:.2f}\n\n"
            
            # Per-class statistics
            report += f"### Per-Class Distribution\n"
            for attack_type, stats in original_stats['per_class'].items():
                report += f"- **{attack_type}**: {stats['positive_samples']} samples "
                report += f"(prevalence: {stats['prevalence']:.3f}, ratio: {stats['imbalance_ratio']:.2f})\n"
            report += "\n"
            
            # SMOTE variant results
            report += f"### SMOTE Variant Results\n"
            for variant, variant_results in embedding_results['variants'].items():
                if 'error' in variant_results:
                    report += f"- **{variant.upper()}**: Error - {variant_results['error']}\n"
                    continue
                
                quality = variant_results['quality_metrics']
                report += f"- **{variant.upper()}**:\n"
                report += f"  - Sample increase: {quality['sample_increase']:.2f}x\n"
                report += f"  - Imbalance improvement: {quality['imbalance_improvement']['improvement_factor']:.2f}x\n"
                report += f"  - Output path: {variant_results['output_path']}\n"
            
            report += "\n"
        
        return report


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply SMOTE oversampling to log embeddings")
    parser.add_argument("--log_type", type=str, required=True,
                        help="Log type to process (e.g., 'vpn', 'wp-access')")
    parser.add_argument("--embedding_types", nargs="+", 
                        default=['fasttext', 'word2vec', 'logbert'],
                        help="Embedding types to process")
    parser.add_argument("--smote_variants", nargs="+",
                        default=['smote', 'borderline', 'adasyn'],
                        help="SMOTE variants to apply")
    parser.add_argument("--use_hierarchical", action='store_true',
                        help="Use hierarchical oversampling")
    parser.add_argument("--use_pseudo_labels", action='store_true',
                        help="Generate pseudo-labels using clustering")
    
    args = parser.parse_args()
    
    # Initialize oversampler
    oversampler = UnsupervisedSMOTEOversampler()
    
    # Process log type
    results = oversampler.process_log_type(
        log_type=args.log_type,
        embedding_types=args.embedding_types,
        smote_variants=args.smote_variants,
        use_hierarchical=args.use_hierarchical,
        use_pseudo_labels=args.use_pseudo_labels
    )
    
    # Generate report
    report = oversampler.generate_oversampling_report(results, args.log_type)
    
    # Save report
    report_path = oversampler.output_dir / f"smote_report_{args.log_type}.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"SMOTE oversampling completed for {args.log_type}")
    print(f"Report saved to: {report_path}")
    print("\nResults summary:")
    for embedding_type, embedding_results in results.items():
        if 'error' not in embedding_results:
            successful_variants = [v for v in embedding_results['variants'] 
                                 if 'error' not in embedding_results['variants'][v]]
            print(f"  {embedding_type}: {len(successful_variants)} variants processed")


if __name__ == "__main__":
    main()


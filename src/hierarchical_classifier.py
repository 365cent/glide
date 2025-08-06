#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hierarchical Classification Architecture for Unsupervised Anomaly Detection

This module implements a hierarchical classification system that organizes
anomaly detection in multiple levels, from coarse-grained to fine-grained
classification. The system maintains unsupervised learning principles while
providing structured, interpretable results.

Key Features:
- Multi-level hierarchical taxonomy for attack types
- Unsupervised learning at each hierarchy level
- Adaptive threshold optimization per level
- Cascading decision making with confidence propagation
- Interpretable classification paths
- Support for dynamic hierarchy expansion

Architecture:
Level 1: Anomaly vs Normal (binary classification)
Level 2: Attack categories (web, network, system, data)
Level 3: Specific attack types (SQL injection, XSS, DDoS, etc.)
Level 4: Attack variants and subtypes

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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import torch.nn.functional as F

# Import configuration
from config import EMBEDDINGS_DIR, MODELS_DIR, PROCESSED_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HierarchicalClassifier")

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class HierarchyNode:
    """Represents a node in the hierarchical classification tree."""
    name: str
    level: int
    parent: Optional[str] = None
    children: List[str] = None
    classifier: Optional[Any] = None
    threshold: float = 0.5
    confidence: float = 0.0
    samples: int = 0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

@dataclass
class ClassificationResult:
    """Represents the result of hierarchical classification."""
    sample_id: int
    path: List[str]
    confidences: List[float]
    final_prediction: str
    anomaly_score: float
    level_predictions: Dict[int, str]

class AttackTaxonomy:
    """
    Defines the hierarchical taxonomy for attack types.
    """
    
    def __init__(self):
        self.taxonomy = self._build_default_taxonomy()
        self.level_mappings = self._create_level_mappings()
    
    def _build_default_taxonomy(self) -> Dict:
        """Build the default attack taxonomy."""
        return {
            "level_0": {
                "root": {
                    "name": "root",
                    "children": ["normal", "anomaly"]
                }
            },
            "level_1": {
                "normal": {
                    "name": "normal",
                    "parent": "root",
                    "children": []
                },
                "anomaly": {
                    "name": "anomaly", 
                    "parent": "root",
                    "children": ["web_attacks", "network_attacks", "system_attacks", "data_attacks"]
                }
            },
            "level_2": {
                "web_attacks": {
                    "name": "web_attacks",
                    "parent": "anomaly",
                    "children": ["injection_attacks", "xss_attacks", "csrf_attacks", "file_inclusion"]
                },
                "network_attacks": {
                    "name": "network_attacks", 
                    "parent": "anomaly",
                    "children": ["dos_attacks", "scan_attacks", "brute_force", "flooding"]
                },
                "system_attacks": {
                    "name": "system_attacks",
                    "parent": "anomaly", 
                    "children": ["privilege_escalation", "backdoor", "trojan", "rootkit"]
                },
                "data_attacks": {
                    "name": "data_attacks",
                    "parent": "anomaly",
                    "children": ["exfiltration", "theft", "breach", "leak"]
                }
            },
            "level_3": {
                # Web attack subtypes
                "injection_attacks": {
                    "name": "injection_attacks",
                    "parent": "web_attacks",
                    "children": ["sql_injection", "command_injection", "ldap_injection", "xpath_injection"]
                },
                "xss_attacks": {
                    "name": "xss_attacks",
                    "parent": "web_attacks", 
                    "children": ["stored_xss", "reflected_xss", "dom_xss"]
                },
                "csrf_attacks": {
                    "name": "csrf_attacks",
                    "parent": "web_attacks",
                    "children": ["csrf_get", "csrf_post", "csrf_ajax"]
                },
                "file_inclusion": {
                    "name": "file_inclusion",
                    "parent": "web_attacks",
                    "children": ["lfi", "rfi", "path_traversal"]
                },
                # Network attack subtypes
                "dos_attacks": {
                    "name": "dos_attacks",
                    "parent": "network_attacks",
                    "children": ["tcp_flood", "udp_flood", "icmp_flood", "syn_flood"]
                },
                "scan_attacks": {
                    "name": "scan_attacks", 
                    "parent": "network_attacks",
                    "children": ["port_scan", "network_scan", "vulnerability_scan"]
                },
                "brute_force": {
                    "name": "brute_force",
                    "parent": "network_attacks",
                    "children": ["password_brute", "ssh_brute", "ftp_brute"]
                },
                "flooding": {
                    "name": "flooding",
                    "parent": "network_attacks", 
                    "children": ["bandwidth_flood", "connection_flood", "request_flood"]
                }
            }
        }
    
    def _create_level_mappings(self) -> Dict[int, List[str]]:
        """Create mappings from levels to node names."""
        mappings = {}
        for level_key, nodes in self.taxonomy.items():
            level_num = int(level_key.split('_')[1])
            mappings[level_num] = list(nodes.keys())
        return mappings
    
    def get_node_path(self, node_name: str) -> List[str]:
        """Get the path from root to a specific node."""
        path = [node_name]
        current = node_name
        
        # Find the node in taxonomy
        for level_nodes in self.taxonomy.values():
            if current in level_nodes and 'parent' in level_nodes[current]:
                parent = level_nodes[current]['parent']
                if parent:
                    path.insert(0, parent)
                    current = parent
                else:
                    break
        
        return path
    
    def get_children(self, node_name: str) -> List[str]:
        """Get children of a specific node."""
        for level_nodes in self.taxonomy.values():
            if node_name in level_nodes:
                return level_nodes[node_name].get('children', [])
        return []
    
    def get_level(self, node_name: str) -> int:
        """Get the level of a specific node."""
        for level_key, nodes in self.taxonomy.items():
            if node_name in nodes:
                return int(level_key.split('_')[1])
        return -1
    
    def map_attack_types_to_taxonomy(self, attack_types: List[str]) -> Dict[str, str]:
        """Map attack types to taxonomy nodes."""
        mapping = {}
        
        for attack_type in attack_types:
            attack_lower = attack_type.lower().replace('_', ' ').replace('-', ' ')
            
            # Try to find best match in taxonomy
            best_match = None
            best_score = 0
            
            for level_nodes in self.taxonomy.values():
                for node_name in level_nodes:
                    node_lower = node_name.lower().replace('_', ' ')
                    
                    # Simple keyword matching
                    common_words = set(attack_lower.split()) & set(node_lower.split())
                    score = len(common_words) / max(len(attack_lower.split()), 1)
                    
                    if score > best_score and score > 0.3:  # Minimum similarity threshold
                        best_score = score
                        best_match = node_name
            
            mapping[attack_type] = best_match if best_match else "anomaly"
        
        return mapping

class UnsupervisedHierarchicalClassifier:
    """
    Unsupervised hierarchical classifier for anomaly detection.
    """
    
    def __init__(self, 
                 taxonomy: Optional[AttackTaxonomy] = None,
                 embedding_dim: int = 768,
                 device: str = 'auto'):
        """
        Initialize the hierarchical classifier.
        
        Args:
            taxonomy: Attack taxonomy structure
            embedding_dim: Dimension of input embeddings
            device: Computing device ('cpu', 'cuda', 'mps', or 'auto')
        """
        self.taxonomy = taxonomy if taxonomy else AttackTaxonomy()
        self.embedding_dim = embedding_dim
        self.device = self._get_device(device)
        
        # Hierarchy structure
        self.nodes = {}
        self.level_classifiers = {}
        self.thresholds = {}
        
        # Training data storage
        self.training_data = {}
        self.cluster_centers = {}
        
        # Performance tracking
        self.training_stats = {}
        
        self._initialize_hierarchy()
    
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate computing device."""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    def _initialize_hierarchy(self):
        """Initialize the hierarchy structure."""
        for level_key, level_nodes in self.taxonomy.taxonomy.items():
            level_num = int(level_key.split('_')[1])
            
            for node_name, node_info in level_nodes.items():
                self.nodes[node_name] = HierarchyNode(
                    name=node_name,
                    level=level_num,
                    parent=node_info.get('parent'),
                    children=node_info.get('children', [])
                )
    
    def _create_level_classifier(self, level: int, input_dim: int) -> nn.Module:
        """Create a neural network classifier for a specific level."""
        
        class LevelClassifier(nn.Module):
            def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim // 2, hidden_dim // 4),
                    nn.ReLU()
                )
                
                self.classifier = nn.Linear(hidden_dim // 4, num_classes)
                self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
                
            def forward(self, x):
                features = self.encoder(x)
                logits = self.classifier(features)
                return features, logits
        
        # Determine number of classes for this level
        level_nodes = self.taxonomy.level_mappings.get(level + 1, [])  # Next level nodes
        num_classes = len(level_nodes) if level_nodes else 2
        
        return LevelClassifier(input_dim, num_classes).to(self.device)
    
    def _cluster_embeddings(self, 
                          embeddings: np.ndarray, 
                          level: int,
                          method: str = 'kmeans') -> Tuple[np.ndarray, Dict]:
        """
        Cluster embeddings for unsupervised learning at a specific level.
        
        Args:
            embeddings: Input embeddings
            level: Hierarchy level
            method: Clustering method
            
        Returns:
            Tuple of (cluster_labels, cluster_info)
        """
        # Determine number of clusters based on taxonomy
        level_nodes = self.taxonomy.level_mappings.get(level + 1, [])
        n_clusters = len(level_nodes) if level_nodes else min(8, max(2, len(embeddings) // 100))
        
        # Normalize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(embeddings_scaled)
            cluster_centers = clusterer.cluster_centers_
            
        elif method == 'agglomerative':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(embeddings_scaled)
            
            # Calculate cluster centers manually
            cluster_centers = []
            for i in range(n_clusters):
                mask = cluster_labels == i
                if np.sum(mask) > 0:
                    center = np.mean(embeddings_scaled[mask], axis=0)
                    cluster_centers.append(center)
            cluster_centers = np.array(cluster_centers)
        
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Calculate clustering quality metrics
        silhouette = silhouette_score(embeddings_scaled, cluster_labels) if len(np.unique(cluster_labels)) > 1 else 0
        
        cluster_info = {
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'cluster_sizes': Counter(cluster_labels),
            'cluster_centers': cluster_centers,
            'scaler': scaler
        }
        
        return cluster_labels, cluster_info
    
    def train_level_classifier(self, 
                             embeddings: np.ndarray,
                             level: int,
                             epochs: int = 50,
                             batch_size: int = 32,
                             learning_rate: float = 0.001) -> Dict:
        """
        Train a classifier for a specific hierarchy level.
        
        Args:
            embeddings: Training embeddings
            level: Hierarchy level
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Training statistics
        """
        logger.info(f"Training classifier for level {level}")
        
        # Cluster embeddings to create pseudo-labels
        cluster_labels, cluster_info = self._cluster_embeddings(embeddings, level)
        
        # Create and train classifier
        classifier = self._create_level_classifier(level, embeddings.shape[1])
        optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Prepare data
        embeddings_tensor = torch.FloatTensor(embeddings).to(self.device)
        labels_tensor = torch.LongTensor(cluster_labels).to(self.device)
        
        dataset = TensorDataset(embeddings_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        classifier.train()
        training_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_embeddings, batch_labels in dataloader:
                optimizer.zero_grad()
                
                features, logits = classifier(batch_embeddings)
                loss = criterion(logits, batch_labels)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            training_losses.append(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Level {level}, Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        # Store classifier and related information
        self.level_classifiers[level] = classifier
        self.cluster_centers[level] = cluster_info
        
        # Calculate optimal thresholds using isolation forest
        classifier.eval()
        with torch.no_grad():
            features, logits = classifier(embeddings_tensor)
            probabilities = torch.softmax(logits, dim=1)
            
            # Train isolation forest on features for anomaly detection
            isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            isolation_forest.fit(features.cpu().numpy())
            
            # Calculate threshold based on anomaly scores
            anomaly_scores = isolation_forest.decision_function(features.cpu().numpy())
            threshold = np.percentile(anomaly_scores, 10)  # 10th percentile as threshold
            
            self.thresholds[level] = threshold
        
        training_stats = {
            'level': level,
            'n_samples': len(embeddings),
            'n_clusters': cluster_info['n_clusters'],
            'silhouette_score': cluster_info['silhouette_score'],
            'final_loss': training_losses[-1],
            'threshold': threshold,
            'cluster_sizes': dict(cluster_info['cluster_sizes'])
        }
        
        self.training_stats[level] = training_stats
        return training_stats
    
    def predict_hierarchical(self, 
                           embeddings: np.ndarray,
                           confidence_threshold: float = 0.5) -> List[ClassificationResult]:
        """
        Perform hierarchical prediction on embeddings.
        
        Args:
            embeddings: Input embeddings
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List of classification results
        """
        results = []
        
        for i, embedding in enumerate(embeddings):
            result = self._predict_single_sample(embedding, i, confidence_threshold)
            results.append(result)
        
        return results
    
    def _predict_single_sample(self, 
                             embedding: np.ndarray,
                             sample_id: int,
                             confidence_threshold: float) -> ClassificationResult:
        """
        Predict a single sample through the hierarchy.
        
        Args:
            embedding: Single embedding vector
            sample_id: Sample identifier
            confidence_threshold: Confidence threshold
            
        Returns:
            Classification result
        """
        path = ["root"]
        confidences = [1.0]
        level_predictions = {0: "root"}
        current_node = "root"
        
        # Traverse hierarchy levels
        for level in sorted(self.level_classifiers.keys()):
            if current_node not in self.nodes or not self.nodes[current_node].children:
                break
            
            classifier = self.level_classifiers[level]
            classifier.eval()
            
            # Prepare input
            embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features, logits = classifier(embedding_tensor)
                probabilities = torch.softmax(logits, dim=1)
                
                # Get prediction and confidence
                max_prob, predicted_class = torch.max(probabilities, 1)
                confidence = max_prob.item()
                
                # Map predicted class to node name
                children = self.nodes[current_node].children
                if predicted_class.item() < len(children):
                    predicted_node = children[predicted_class.item()]
                else:
                    predicted_node = children[0] if children else current_node
                
                # Check confidence threshold
                if confidence >= confidence_threshold:
                    path.append(predicted_node)
                    confidences.append(confidence)
                    level_predictions[level + 1] = predicted_node
                    current_node = predicted_node
                else:
                    # Stop traversal if confidence is too low
                    break
        
        # Calculate anomaly score
        anomaly_score = self._calculate_anomaly_score(embedding, path)
        
        # Determine final prediction
        final_prediction = path[-1] if len(path) > 1 else "unknown"
        
        return ClassificationResult(
            sample_id=sample_id,
            path=path,
            confidences=confidences,
            final_prediction=final_prediction,
            anomaly_score=anomaly_score,
            level_predictions=level_predictions
        )
    
    def _calculate_anomaly_score(self, embedding: np.ndarray, path: List[str]) -> float:
        """
        Calculate anomaly score based on the classification path.
        
        Args:
            embedding: Input embedding
            path: Classification path
            
        Returns:
            Anomaly score (0-1, higher means more anomalous)
        """
        if len(path) <= 1:
            return 0.5  # Neutral score for unknown
        
        # Base score on path depth and confidence
        max_depth = max(self.taxonomy.level_mappings.keys()) if self.taxonomy.level_mappings else 3
        depth_score = len(path) / max_depth
        
        # Check if path indicates anomaly
        anomaly_score = 0.0
        if "anomaly" in path:
            anomaly_score = 0.7  # High base score for anomalous paths
            
            # Increase score based on specific attack types
            if len(path) > 2:  # Has specific attack category
                anomaly_score += 0.2
            if len(path) > 3:  # Has specific attack type
                anomaly_score += 0.1
        
        return min(anomaly_score, 1.0)
    
    def train_hierarchical_model(self, 
                                embeddings: np.ndarray,
                                log_type: str,
                                embedding_type: str) -> Dict:
        """
        Train the complete hierarchical model.
        
        Args:
            embeddings: Training embeddings
            log_type: Type of logs
            embedding_type: Type of embeddings
            
        Returns:
            Training summary
        """
        logger.info(f"Training hierarchical model for {log_type} with {embedding_type} embeddings")
        
        training_summary = {
            'log_type': log_type,
            'embedding_type': embedding_type,
            'n_samples': len(embeddings),
            'embedding_dim': embeddings.shape[1],
            'levels_trained': [],
            'level_stats': {}
        }
        
        # Train classifiers for each level
        for level in range(3):  # Train levels 0, 1, 2
            if level in self.taxonomy.level_mappings:
                stats = self.train_level_classifier(embeddings, level)
                training_summary['levels_trained'].append(level)
                training_summary['level_stats'][level] = stats
        
        # Save model
        model_path = MODELS_DIR / "hierarchical" / f"{log_type}_{embedding_type}"
        model_path.mkdir(parents=True, exist_ok=True)
        
        self.save_model(model_path)
        training_summary['model_path'] = str(model_path)
        
        logger.info(f"Hierarchical model training completed for {log_type}")
        return training_summary
    
    def save_model(self, path: Path):
        """Save the trained hierarchical model."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save classifiers
        for level, classifier in self.level_classifiers.items():
            torch.save(classifier.state_dict(), path / f"classifier_level_{level}.pth")
        
        # Save other components
        with open(path / "taxonomy.json", 'w') as f:
            json.dump(self.taxonomy.taxonomy, f, indent=2)
        
        with open(path / "thresholds.json", 'w') as f:
            json.dump(self.thresholds, f, indent=2)
        
        with open(path / "cluster_centers.pkl", 'wb') as f:
            pickle.dump(self.cluster_centers, f)
        
        with open(path / "training_stats.json", 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path):
        """Load a trained hierarchical model."""
        path = Path(path)
        
        # Load taxonomy
        with open(path / "taxonomy.json", 'r') as f:
            taxonomy_data = json.load(f)
        
        self.taxonomy.taxonomy = taxonomy_data
        self.taxonomy.level_mappings = self.taxonomy._create_level_mappings()
        self._initialize_hierarchy()
        
        # Load thresholds
        with open(path / "thresholds.json", 'r') as f:
            self.thresholds = json.load(f)
        
        # Load cluster centers
        with open(path / "cluster_centers.pkl", 'rb') as f:
            self.cluster_centers = pickle.load(f)
        
        # Load training stats
        with open(path / "training_stats.json", 'r') as f:
            self.training_stats = json.load(f)
        
        # Load classifiers
        for level_file in path.glob("classifier_level_*.pth"):
            level = int(level_file.stem.split('_')[-1])
            classifier = self._create_level_classifier(level, self.embedding_dim)
            classifier.load_state_dict(torch.load(level_file, map_location=self.device))
            self.level_classifiers[level] = classifier
        
        logger.info(f"Model loaded from {path}")
    
    def evaluate_hierarchical_performance(self, 
                                        embeddings: np.ndarray,
                                        true_labels: Optional[np.ndarray] = None) -> Dict:
        """
        Evaluate hierarchical classification performance.
        
        Args:
            embeddings: Test embeddings
            true_labels: True labels (if available)
            
        Returns:
            Performance metrics
        """
        predictions = self.predict_hierarchical(embeddings)
        
        metrics = {
            'n_samples': len(embeddings),
            'prediction_distribution': Counter([p.final_prediction for p in predictions]),
            'average_confidence': np.mean([np.mean(p.confidences) for p in predictions]),
            'average_path_length': np.mean([len(p.path) for p in predictions]),
            'anomaly_score_distribution': {
                'mean': np.mean([p.anomaly_score for p in predictions]),
                'std': np.std([p.anomaly_score for p in predictions]),
                'min': np.min([p.anomaly_score for p in predictions]),
                'max': np.max([p.anomaly_score for p in predictions])
            }
        }
        
        # Level-wise analysis
        level_analysis = {}
        for level in self.level_classifiers.keys():
            level_predictions = [p.level_predictions.get(level, 'unknown') for p in predictions]
            level_analysis[level] = {
                'distribution': Counter(level_predictions),
                'coverage': sum(1 for p in level_predictions if p != 'unknown') / len(level_predictions)
            }
        
        metrics['level_analysis'] = level_analysis
        
        return metrics


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train hierarchical classifier")
    parser.add_argument("--log_type", type=str, required=True,
                        help="Log type to process")
    parser.add_argument("--embedding_type", type=str, default="logbert",
                        help="Embedding type to use")
    parser.add_argument("--embedding_path", type=str, required=True,
                        help="Path to embedding file")
    parser.add_argument("--output_dir", type=str, default=str(MODELS_DIR / "hierarchical"),
                        help="Output directory for trained model")
    
    args = parser.parse_args()
    
    # Load embeddings
    with open(args.embedding_path, 'rb') as f:
        embeddings = pickle.load(f)
    
    # Initialize and train classifier
    classifier = UnsupervisedHierarchicalClassifier(embedding_dim=embeddings.shape[1])
    
    # Train model
    training_summary = classifier.train_hierarchical_model(
        embeddings, args.log_type, args.embedding_type
    )
    
    # Evaluate model
    performance = classifier.evaluate_hierarchical_performance(embeddings)
    
    # Save results
    output_path = Path(args.output_dir) / f"{args.log_type}_{args.embedding_type}"
    with open(output_path / "training_summary.json", 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    with open(output_path / "performance_metrics.json", 'w') as f:
        json.dump(performance, f, indent=2, default=str)
    
    print(f"Hierarchical classifier training completed for {args.log_type}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()


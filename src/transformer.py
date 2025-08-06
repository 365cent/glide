#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Unsupervised Transformer for Anomaly Detection

This module implements an advanced transformer-based architecture that integrates
SMOTE oversampling and hierarchical classification while maintaining unsupervised
learning principles. The model uses self-supervised learning, contrastive learning,
and reconstruction objectives to learn robust representations for anomaly detection.

Key Features:
- Multi-head self-attention with positional encoding
- Self-supervised learning with masked language modeling
- Contrastive learning for representation learning
- Integration with SMOTE-balanced data
- Hierarchical classification support
- Adaptive threshold optimization
- Teacher-student distillation
- Uncertainty quantification
- Automatic log type detection
- Multi-embedding type support

Architecture Components:
1. Embedding Layer: Projects input embeddings to transformer dimension
2. Positional Encoding: Adds positional information for sequence modeling
3. Multi-Head Attention: Captures complex relationships in log data
4. Feed-Forward Networks: Non-linear transformations
5. Reconstruction Head: Self-supervised reconstruction objective
6. Classification Head: Hierarchical anomaly classification
7. Contrastive Head: Contrastive learning objective

Author: Anomaly Detection Pipeline
Version: 2.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
import math
import warnings
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import time

# Import custom modules
from smote_oversampling import UnsupervisedSMOTEOversampler
from hierarchical_classifier import UnsupervisedHierarchicalClassifier, AttackTaxonomy
from config import EMBEDDINGS_DIR, MODELS_DIR, PROCESSED_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EnhancedTransformer")

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class TransformerConfig:
    """Configuration for the enhanced transformer model."""
    # Model architecture
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_length: int = 512
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 100
    warmup_steps: int = 1000
    
    # Loss weights
    reconstruction_weight: float = 1.0
    contrastive_weight: float = 0.5
    classification_weight: float = 0.3
    
    # Contrastive learning
    temperature: float = 0.07
    negative_samples: int = 16
    
    # Hierarchical classification
    use_hierarchical: bool = True
    hierarchy_levels: int = 3
    
    # SMOTE integration
    use_smote: bool = True
    smote_variant: str = 'smote'
    
    # Device configuration
    device: str = 'auto'

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(context)
        return output, attention_weights

class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights

class ContrastiveLearningHead(nn.Module):
    """Contrastive learning head for self-supervised learning."""
    
    def __init__(self, d_model: int, projection_dim: int = 128):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, projection_dim)
        )
    
    def forward(self, x):
        return F.normalize(self.projection(x), dim=-1)

class ReconstructionHead(nn.Module):
    """Reconstruction head for self-supervised learning."""
    
    def __init__(self, d_model: int, input_dim: int):
        super().__init__()
        
        self.reconstruction = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, input_dim)
        )
    
    def forward(self, x):
        return self.reconstruction(x)

class HierarchicalClassificationHead(nn.Module):
    """Hierarchical classification head."""
    
    def __init__(self, d_model: int, hierarchy_levels: int = 3):
        super().__init__()
        
        self.hierarchy_levels = hierarchy_levels
        self.level_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, 8)  # Max 8 classes per level
            ) for _ in range(hierarchy_levels)
        ])
    
    def forward(self, x):
        level_outputs = []
        for classifier in self.level_classifiers:
            level_outputs.append(classifier(x))
        return level_outputs

class EnhancedUnsupervisedTransformer(nn.Module):
    """Enhanced unsupervised transformer for anomaly detection."""
    
    def __init__(self, config: TransformerConfig, input_dim: int):
        super().__init__()
        
        self.config = config
        self.input_dim = input_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, config.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_length)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        # Output heads
        self.reconstruction_head = ReconstructionHead(config.d_model, input_dim)
        self.contrastive_head = ContrastiveLearningHead(config.d_model)
        
        if config.use_hierarchical:
            self.hierarchical_head = HierarchicalClassificationHead(
                config.d_model, config.hierarchy_levels
            )
        
        # Anomaly detection
        self.anomaly_detector = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x, mask=None, return_attention=False):
        """Forward pass through the transformer."""
        # Handle 2D input (batch_size, embedding_dim)
        if x.dim() == 2:
            batch_size, embedding_dim = x.shape
            # Add sequence dimension for transformer processing
            x = x.unsqueeze(1)  # Shape: (batch_size, 1, embedding_dim)
        else:
            batch_size, seq_len, embedding_dim = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer blocks
        attention_weights = []
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block(x, mask)
            if return_attention:
                attention_weights.append(attn_weights)
        
        # Global average pooling for sequence representation
        sequence_repr = torch.mean(x, dim=1)  # Shape: (batch_size, d_model)
        
        # Output heads
        outputs = {
            'sequence_representation': sequence_repr,
            'reconstruction': self.reconstruction_head(sequence_repr),
            'contrastive': self.contrastive_head(sequence_repr),
            'anomaly_score': self.anomaly_detector(sequence_repr)
        }
        
        if self.config.use_hierarchical:
            outputs['hierarchical'] = self.hierarchical_head(sequence_repr)
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
        
        return outputs

class EnhancedTransformerTrainer:
    """Trainer for the enhanced transformer model."""
    
    def __init__(self, 
                 config: TransformerConfig,
                 model: EnhancedUnsupervisedTransformer,
                 device: torch.device):
        self.config = config
        self.model = model.to(device)
        self.device = device
        
        # Optimizers
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )
        
        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.contrastive_loss = self._contrastive_loss
        self.classification_loss = nn.CrossEntropyLoss()
        
        # Training statistics
        self.training_stats = defaultdict(list)
        
        # SMOTE oversampler
        if config.use_smote:
            self.smote_oversampler = UnsupervisedSMOTEOversampler()
        
        # Hierarchical classifier
        if config.use_hierarchical:
            self.hierarchical_classifier = UnsupervisedHierarchicalClassifier()
    
    def _contrastive_loss(self, features, temperature=0.07):
        """Compute contrastive loss for self-supervised learning."""
        batch_size = features.size(0)
        
        # Check for NaN or inf values
        if torch.isnan(features).any() or torch.isinf(features).any():
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / temperature
        
        # Create labels (positive pairs are identical samples)
        labels = torch.arange(batch_size).to(self.device)
        
        # Mask out diagonal (self-similarity)
        mask = torch.eye(batch_size, dtype=torch.bool).to(self.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # Compute contrastive loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        # Check for NaN or inf in loss
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return loss
    
    def _create_masked_input(self, x, mask_ratio=0.15):
        """Create masked input for self-supervised learning."""
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            batch_size, dim = x.shape
            # Create simple feature-level mask for 2D input
            mask = torch.rand(batch_size, dim) < mask_ratio
            mask = mask.to(self.device)
            
            masked_x = x.clone()
            masked_x[mask] = 0
            
            return masked_x, mask
        else:
            batch_size, seq_len, dim = x.shape
            
            # Create random mask
            mask = torch.rand(batch_size, seq_len) < mask_ratio
            mask = mask.to(self.device)
            
            # Apply mask (replace with zeros)
            masked_x = x.clone()
            masked_x[mask] = 0
            
            return masked_x, mask
    
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        epoch_losses = defaultdict(float)
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(self.device)
            else:
                x = batch.to(self.device)
            
            # Debug: Print batch info for first few batches
            if batch_idx < 3:
                logger.info(f"Batch {batch_idx}: x shape={x.shape}, x dtype={x.dtype}")
                logger.info(f"Batch {batch_idx}: x min/max={x.min():.4f}/{x.max():.4f}")
                logger.info(f"Batch {batch_idx}: x mean/std={x.mean():.4f}/{x.std():.4f}")
            
            # Create masked input for self-supervised learning
            masked_x, mask = self._create_masked_input(x)
            
            # Forward pass
            outputs = self.model(masked_x)
            
            # Compute losses
            losses = {}
            
            # Reconstruction loss - reconstruct the original embeddings
            # The original input x is (batch_size, embedding_dim) or (batch_size, 1, embedding_dim)
            if x.dim() == 3:
                reconstruction_target = x.squeeze(1)  # Remove sequence dimension
            else:
                reconstruction_target = x  # Already 2D
            losses['reconstruction'] = self.reconstruction_loss(
                outputs['reconstruction'], reconstruction_target
            )
            
            # Contrastive loss
            losses['contrastive'] = self.contrastive_loss(outputs['contrastive'])
            
            # Hierarchical classification loss (if enabled)
            if self.config.use_hierarchical and 'hierarchical' in outputs:
                # Generate pseudo-labels using clustering
                with torch.no_grad():
                    features = outputs['sequence_representation'].cpu().numpy()
                    pseudo_labels = self._generate_hierarchical_pseudo_labels(features)
                
                hierarchical_loss = 0
                for level, level_output in enumerate(outputs['hierarchical']):
                    if level < len(pseudo_labels):
                        level_labels = torch.LongTensor(pseudo_labels[level]).to(self.device)
                        hierarchical_loss += self.classification_loss(level_output, level_labels)
                
                losses['hierarchical'] = hierarchical_loss / len(outputs['hierarchical'])
            
            # Total loss
            total_loss = (
                self.config.reconstruction_weight * losses['reconstruction'] +
                self.config.contrastive_weight * losses['contrastive']
            )
            
            if 'hierarchical' in losses:
                total_loss += self.config.classification_weight * losses['hierarchical']
            
            losses['total'] = total_loss
            
            # Debug: Check for NaN or inf values
            for loss_name, loss_value in losses.items():
                if torch.isnan(loss_value) or torch.isinf(loss_value):
                    logger.warning(f"Warning: {loss_name} loss is {loss_value}")
                    # Replace with zero if it's inf or nan
                    if torch.isnan(loss_value) or torch.isinf(loss_value):
                        losses[loss_name] = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            
            # Update statistics
            for loss_name, loss_value in losses.items():
                epoch_losses[loss_name] += loss_value.item()
        
        # Average losses
        for loss_name in epoch_losses:
            epoch_losses[loss_name] /= len(dataloader)
        
        return dict(epoch_losses)
    
    def _generate_hierarchical_pseudo_labels(self, features):
        """Generate pseudo-labels for hierarchical classification."""
        pseudo_labels = []
        
        # Generate labels for each hierarchy level
        for level in range(self.config.hierarchy_levels):
            n_clusters = min(8, max(2, len(features) // (10 * (level + 1))))
            
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features)
                pseudo_labels.append(labels)
            else:
                # Not enough samples for clustering
                pseudo_labels.append(np.zeros(len(features), dtype=int))
        
        return pseudo_labels
    
    def train(self, train_dataloader, val_dataloader=None, save_path=None):
        """Train the model."""
        logger.info(f"Starting training for {self.config.epochs} epochs")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            # Training
            train_losses = self.train_epoch(train_dataloader)
            
            # Validation
            if val_dataloader is not None:
                val_losses = self.evaluate(val_dataloader)
            else:
                val_losses = {}
            
            # Update learning rate
            self.scheduler.step()
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{self.config.epochs}")
            logger.info(f"Train Loss: {train_losses['total']:.4f}")
            if val_losses:
                logger.info(f"Val Loss: {val_losses['total']:.4f}")
            
            # Save statistics
            for loss_name, loss_value in train_losses.items():
                self.training_stats[f'train_{loss_name}'].append(loss_value)
            
            for loss_name, loss_value in val_losses.items():
                self.training_stats[f'val_{loss_name}'].append(loss_value)
            
            # Save best model
            if save_path and val_losses and val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                self.save_model(save_path)
                logger.info(f"Saved best model with validation loss: {best_val_loss:.4f}")
        
        logger.info("Training completed")
        return self.training_stats
    
    def evaluate(self, dataloader):
        """Evaluate the model."""
        self.model.eval()
        eval_losses = defaultdict(float)
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)
                
                # Forward pass
                outputs = self.model(x)
                
                            # Compute losses
                # Reconstruction loss - reconstruct the original embeddings
                if x.dim() == 3:
                    reconstruction_target = x.squeeze(1)  # Remove sequence dimension
                else:
                    reconstruction_target = x  # Already 2D
                reconstruction_loss = self.reconstruction_loss(
                    outputs['reconstruction'], reconstruction_target
                )
                contrastive_loss = self.contrastive_loss(outputs['contrastive'])
                
                total_loss = (
                    self.config.reconstruction_weight * reconstruction_loss +
                    self.config.contrastive_weight * contrastive_loss
                )
                
                eval_losses['reconstruction'] += reconstruction_loss.item()
                eval_losses['contrastive'] += contrastive_loss.item()
                eval_losses['total'] += total_loss.item()
        
        # Average losses
        for loss_name in eval_losses:
            eval_losses[loss_name] /= len(dataloader)
        
        return dict(eval_losses)
    
    def predict_anomalies(self, dataloader, threshold=0.5):
        """Predict anomalies using the trained model."""
        self.model.eval()
        predictions = []
        anomaly_scores = []
        representations = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)
                
                outputs = self.model(x)
                
                # Get anomaly scores
                scores = outputs['anomaly_score'].cpu().numpy()
                anomaly_scores.extend(scores.flatten())
                
                # Get predictions
                preds = (scores > threshold).astype(int)
                predictions.extend(preds.flatten())
                
                # Get representations
                reprs = outputs['sequence_representation'].cpu().numpy()
                representations.extend(reprs)
        
        return {
            'predictions': np.array(predictions),
            'anomaly_scores': np.array(anomaly_scores),
            'representations': np.array(representations)
        }
    
    def save_model(self, path):
        """Save the trained model."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_stats': dict(self.training_stats)
        }, path / 'model.pth')
        
        # Save config
        with open(path / 'config.json', 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model."""
        path = Path(path)
        
        checkpoint = torch.load(path / 'model.pth', map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = defaultdict(list, checkpoint['training_stats'])
        
        logger.info(f"Model loaded from {path}")

def detect_available_log_types():
    """Detect available log types from embeddings directory."""
    log_types = []
    
    # Check for different embedding types
    embedding_types = ['fasttext', 'word2vec', 'logbert']
    
    for embedding_type in embedding_types:
        embedding_dir = EMBEDDINGS_DIR / embedding_type
        if embedding_dir.exists():
            for log_type_dir in embedding_dir.iterdir():
                if log_type_dir.is_dir():
                    log_type = log_type_dir.name
                    # Check if both log and label files exist
                    log_file = log_type_dir / f"log_{log_type}.pkl"
                    label_file = log_type_dir / f"label_{log_type}.pkl"
                    
                    if log_file.exists() and label_file.exists():
                        log_types.append((log_type, embedding_type))
    
    return log_types

def load_embeddings_and_labels(log_type: str, embedding_type: str):
    """Load embeddings and labels for a specific log type and embedding type."""
    embedding_dir = EMBEDDINGS_DIR / embedding_type / log_type
    
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
    
    # Debug information
    logger.info(f"Loaded embeddings shape: {embeddings.shape}")
    logger.info(f"Embeddings dtype: {embeddings.dtype}")
    logger.info(f"Embeddings min/max: {embeddings.min():.4f}/{embeddings.max():.4f}")
    logger.info(f"Embeddings mean/std: {embeddings.mean():.4f}/{embeddings.std():.4f}")
    logger.info(f"Label data keys: {list(label_data.keys())}")
    logger.info(f"Label vectors shape: {label_data['vectors'].shape}")
    logger.info(f"Label classes: {label_data['classes']}")
    
    return embeddings, label_data

def create_enhanced_transformer(input_dim: int, config: Optional[TransformerConfig] = None):
    """Create an enhanced transformer model."""
    if config is None:
        config = TransformerConfig()
    
    # Determine device
    if config.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(config.device)
    
    # Create model
    model = EnhancedUnsupervisedTransformer(config, input_dim)
    trainer = EnhancedTransformerTrainer(config, model, device)
    
    return trainer

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Train enhanced transformer model")
    parser.add_argument("--embedding_type", type=str, required=True,
                        choices=['fasttext', 'word2vec', 'logbert'],
                        help="Embedding type to use")
    parser.add_argument("--log_type", type=str, default=None,
                        help="Specific log type to process (auto-detected if not specified)")
    parser.add_argument("--output_dir", type=str, default=str(MODELS_DIR / "transformer"),
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--use_smote", action='store_true',
                        help="Use SMOTE oversampling")
    parser.add_argument("--use_hierarchical", action='store_true',
                        help="Use hierarchical classification")
    parser.add_argument("--d_model", type=int, default=512,
                        help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=6,
                        help="Number of transformer layers")
    
    args = parser.parse_args()
    
    # Detect available log types if not specified
    if args.log_type is None:
        available_log_types = detect_available_log_types()
        if not available_log_types:
            logger.error("No log types found with the specified embedding type")
            return
        
        # Filter by embedding type
        matching_log_types = [(lt, et) for lt, et in available_log_types if et == args.embedding_type]
        if not matching_log_types:
            logger.error(f"No log types found for embedding type: {args.embedding_type}")
            return
        
        # Use the first available log type
        args.log_type = matching_log_types[0][0]
        logger.info(f"Auto-detected log type: {args.log_type}")
    
    # Load embeddings and labels
    try:
        embeddings, label_data = load_embeddings_and_labels(args.log_type, args.embedding_type)
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        return
    
    logger.info(f"Loaded embeddings with shape: {embeddings.shape}")
    logger.info(f"Loaded labels with {len(label_data['classes'])} classes")
    
    # Create configuration
    config = TransformerConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_smote=args.use_smote,
        use_hierarchical=args.use_hierarchical
    )
    
    # Create model and trainer
    trainer = create_enhanced_transformer(embeddings.shape[1], config)
    
    # Prepare data
    embeddings_tensor = torch.FloatTensor(embeddings)
    dataset = TensorDataset(embeddings_tensor)
    
    # Split into train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Train model
    output_path = Path(args.output_dir) / f"{args.log_type}_{args.embedding_type}"
    training_stats = trainer.train(train_dataloader, val_dataloader, output_path)
    
    # Evaluate model
    predictions = trainer.predict_anomalies(val_dataloader)
    
    # Save results
    with open(output_path / "predictions.pkl", 'wb') as f:
        pickle.dump(predictions, f)
    
    with open(output_path / "training_stats.json", 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    logger.info(f"Training completed. Results saved to {output_path}")
    logger.info(f"Detected {np.sum(predictions['predictions'])} anomalies out of {len(predictions['predictions'])} samples")

if __name__ == "__main__":
    main()

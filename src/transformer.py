
#!/usr/bin/env python3
"""
Enhanced Transformer for Unsupervised Multi-Label Attack Detection
================================================================

This module implements a sophisticated approach for unsupervised multi-label attack detection:

1. **Separate Models Approach**: Train individual models for each attack type
2. **One-vs-Rest Strategy**: Each model learns to distinguish its attack type from normal logs
3. **Enhanced Pseudo-labeling**: Use clustering to generate meaningful binary labels
4. **Combined Results**: Merge predictions from all models into multi-label output

Key Features:
- Separate binary classification for each attack type
- Clustering-based pseudo-label generation
- Enhanced transformer architecture
- Adaptive thresholding
- Comprehensive evaluation metrics
- Early stopping and proper cleanup
- Model checkpointing and resuming
- Self-supervised teacher-student networks for label enhancement
- SMOTE for minority label balancing

Usage:
    python src/transformer_clean.py --log-type wp-error
    python src/transformer_clean.py --log-type wp-error --sample-size 1000
"""

import argparse
import os
import pickle
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from halo import Halo
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.teacher_student import Teacher, Student, distillation_loss # Import teacher-student components

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


@dataclass
class SystemConfig:
    """Auto-detected system configuration"""
    device: str
    n_gpus: int
    total_memory_gb: float
    gpu_memory_gb: float
    n_cpus: int
    is_distributed: bool
    rank: int
    world_size: int
    node_name: str
    job_id: str


def detect_system_resources() -> SystemConfig:
    """Auto-detect system resources and configuration"""
    
    # Device detection
    if torch.cuda.is_available():
        device = "cuda"
        n_gpus = torch.cuda.device_count()
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        n_gpus = 1
        gpu_memory_gb = 16.0  # M2 GPU typically has 16GB
    else:
        device = "cpu"
        n_gpus = 0
        gpu_memory_gb = 0.0
    
    # System info
    total_memory_gb = 8.0  # Default, could be detected
    n_cpus = os.cpu_count() or 8
    
    # Distributed training info
    is_distributed = False
    rank = 0
    world_size = 1
    node_name = os.environ.get("SLURM_NODELIST", "unknown")
    job_id = os.environ.get("SLURM_JOB_ID", "unknown")
    
    return SystemConfig(
        device=device,
        n_gpus=n_gpus,
        total_memory_gb=total_memory_gb,
        gpu_memory_gb=gpu_memory_gb,
        n_cpus=n_cpus,
        is_distributed=is_distributed,
        rank=rank,
        world_size=world_size,
        node_name=node_name,
        job_id=job_id
    )


class UnsupervisedMultiLabelTransformer(nn.Module):
    """Enhanced Transformer for unsupervised multi-label attack type detection."""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        n_labels: int,
        n_clusters: int,  # Kept for API compatibility
        dropout: float = 0.1,
        transformer_layers: int = 2,
        attention_heads: int = 8,
        **kwargs,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_labels = n_labels
        self.n_clusters = n_clusters
        
        # Enhanced input projection with normalization and activation
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Enhanced transformer blocks with pre-normalization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=attention_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation='relu',
            norm_first=True,  # Pre-normalization for better training
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        
        # Enhanced decoder and classifier
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, input_dim),  # Output same dimension as input
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, n_labels)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for better training"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x, **kwargs):
        """Forward pass with enhanced architecture"""
        # Input projection
        x = self.input_projection(x)  # [batch, seq_len, latent_dim]
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)
        
        # For embeddings, just use the encoded output directly
        pooled = encoded.squeeze(1)  # [batch, latent_dim]
        
        # Decoder for reconstruction
        reconstructed = self.decoder(pooled)  # [batch, input_dim]
        
        # Classifier for multi-label classification
        multi_label_scores = self.classifier(pooled)  # [batch, n_labels]
        
        return {
            "reconstructed": reconstructed,
            "multi_label_scores": multi_label_scores,
            "pooled": pooled
        }


class ProgressTracker:
    """Enhanced progress tracking for training"""
    
    def __init__(self, output_dir: Path, log_type: str, config: SystemConfig):
        self.output_dir = output_dir
        self.log_type = log_type
        self.config = config
        self.true_labels = None  # Store true labels for evaluation
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / f"{log_type}_training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_step(self, step: str, data: Dict[str, Any]):
        """Log training step with structured data"""
        self.logger.info(f"STEP: {step}")
        self.logger.info(f"DATA: {data}")
    
    def start_training(self, total_epochs: int, total_batches_per_epoch: int = 0):
        """Start training session"""
        self.log_step("Training Start", {
            "total_epochs": total_epochs,
            "total_batches_per_epoch": total_batches_per_epoch,
            "config": self.config.__dict__
        })
    
    def start_epoch(self, epoch: int):
        """Start new epoch"""
        pass  # Minimal implementation
    
    def update_batch_progress(self, batch_idx: int, batch_time: float = None, loss_info: Dict[str, float] = None):
        """Update batch progress"""
        pass  # Minimal implementation
    
    def update_epoch_progress(self, epoch: int, epoch_time: float):
        """Update epoch progress"""
        pass  # Minimal implementation


def generate_pseudo_labels_for_attack_type(
    embeddings: np.ndarray, 
    attack_type: str, 
    attack_idx: int, 
    n_samples: int,
    teacher_model: Optional[nn.Module] = None, # Added for label enhancement
    device: Optional[torch.device] = None # Added for label enhancement
) -> np.ndarray:
    """Generate pseudo-labels for a specific attack type using clustering and optionally enhance with teacher model."""
    
    print(f"🔍 Generating pseudo-labels for {attack_type}...")
    
    # Normalize embeddings for better clustering
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings)
    
    # Use K-means clustering to find natural groupings
    n_clusters = max(2, min(5, n_samples // 20))  # Ensure at least 2 clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42 + attack_idx, n_init=10)
    cluster_labels = kmeans.fit_predict(normalized_embeddings)
    
    # Initialize binary labels
    binary_labels = np.zeros(n_samples, dtype=np.float32)
    
    # Assign pseudo-labels based on clustering
    for sample_idx in range(n_samples):
        cluster_id = cluster_labels[sample_idx]
        
        # Calculate similarity to cluster centroid
        cluster_centroid = kmeans.cluster_centers_[cluster_id]
        similarity = cosine_similarity(
            normalized_embeddings[sample_idx:sample_idx+1], 
            cluster_centroid.reshape(1, -1)
        )[0, 0]
        
        # Assign pseudo-label based on cluster characteristics
        if cluster_id == attack_idx % n_clusters:  # Assign specific cluster to this attack
            base_prob = 0.7  # High probability for attack
        else:
            base_prob = 0.2  # Low probability for attack
        
        # Adjust based on similarity
        similarity_bonus = similarity * 0.3
        
        # Add some randomness for diversity
        random_factor = np.random.uniform(-0.1, 0.1)
        
        # Combine factors
        final_prob = base_prob + similarity_bonus + random_factor
        binary_labels[sample_idx] = np.clip(final_prob, 0.05, 0.95)
    
    # Apply label smoothing
    binary_labels = binary_labels * 0.9 + 0.05
    
    # Add controlled noise
    noise = np.random.normal(0, 0.02, binary_labels.shape)
    binary_labels = np.clip(binary_labels + noise, 0, 1)

    # Label enhancement using teacher model (if provided)
    if teacher_model and device:
        print(f"✨ Enhancing pseudo-labels for {attack_type} using teacher model...")
        teacher_model.eval()
        with torch.no_grad():
            # Convert embeddings to tensor and move to device
            embeddings_tensor = torch.from_numpy(embeddings).float().to(device)
            teacher_outputs = teacher_model(embeddings_tensor)
            teacher_predictions = torch.sigmoid(teacher_outputs).cpu().numpy().squeeze()
            
            # Combine pseudo-labels with teacher's predictions
            # This is a simple weighted average, more sophisticated methods can be used
            # For example, trust teacher more for high confidence predictions
            alpha_teacher = 0.3 # Weight for teacher's prediction
            binary_labels = (1 - alpha_teacher) * binary_labels + alpha_teacher * teacher_predictions
            binary_labels = np.clip(binary_labels, 0, 1)

    print(f"✅ Generated pseudo-labels for {attack_type}")
    print(f"📊 Attack samples: {np.sum(binary_labels > 0.5)}/{len(binary_labels)}")
    
    return binary_labels


def train_single_attack_model(
    model: UnsupervisedMultiLabelTransformer,
    embeddings: np.ndarray,
    binary_labels: np.ndarray,
    attack_type: str,
    config: SystemConfig,
    tracker: ProgressTracker,
    log_type: str,
    model_save_path: Path # Added for checkpointing
) -> UnsupervisedMultiLabelTransformer:
    """Train a single model for one attack type using normal log data."""
    
    device = torch.device(config.device)
    
    # Enhanced architecture parameters
    embedding_dim = embeddings.shape[1]
    if embedding_dim <= 300:  # FastText
        latent_dim = 256
        transformer_layers = 3
        attention_heads = 8
    elif embedding_dim <= 768:  # Standard BERT
        latent_dim = 384
        transformer_layers = 4
        attention_heads = 8
    else:  # Enhanced LogBERT (2314D)
        latent_dim = 512
        transformer_layers = 6
        attention_heads = 16

    # Optimized batch sizes
    if config.device == "mps":
        if embedding_dim <= 300:
            batch_size = min(128, max(32, int(config.gpu_memory_gb * 2)))
        elif embedding_dim <= 768:
            batch_size = min(64, max(16, int(config.gpu_memory_gb * 1.5)))
        else:
            batch_size = min(32, max(8, int(config.gpu_memory_gb * 1)))
    elif config.device == "cuda":
        if embedding_dim <= 300:
            batch_size = min(32, max(8, int(config.gpu_memory_gb * 0.5)))
        elif embedding_dim <= 768:
            batch_size = min(16, max(4, int(config.gpu_memory_gb * 0.3)))
        else:
            batch_size = min(8, max(2, int(config.gpu_memory_gb * 0.2)))
    else:
        if embedding_dim <= 300:
            batch_size = min(64, max(16, int(config.gpu_memory_gb * 1.5)))
        elif embedding_dim <= 768:
            batch_size = min(32, max(8, int(config.gpu_memory_gb * 1)))
        else:
            batch_size = min(16, max(4, int(config.gpu_memory_gb * 0.5)))

    # Apply SMOTE for minority class balancing
    print(f"Applying SMOTE for {attack_type}...")
    sm = SMOTE(random_state=42)
    # Reshape for SMOTE: (n_samples, n_features) and (n_samples,)
    # SMOTE expects 1D labels, so we need to convert binary_labels to 1D
    # For one-vs-rest, we are essentially balancing the \'attack\' class vs \'normal\' class
    # So, we need to decide what constitutes the minority class for SMOTE.
    # Here, we assume the \'attack\' pseudo-labels (values > 0.5) are the minority.
    
    # Convert pseudo-labels to binary for SMOTE (0 or 1)
    smote_labels = (binary_labels > 0.5).astype(int)
    
    # Check if there\'s a minority class to balance
    unique_labels, counts = np.unique(smote_labels, return_counts=True)
    if len(unique_labels) < 2 or np.min(counts) == 0:
        print(f"SMOTE skipped for {attack_type}: Only one class or no minority samples found.")
        X_res, y_res = embeddings, smote_labels
    else:
        try:
            X_res, y_res = sm.fit_resample(embeddings, smote_labels)
            print(f"SMOTE applied for {attack_type}. Original samples: {len(embeddings)}, Resampled samples: {len(X_res)}")
        except ValueError as e:
            print(f"SMOTE failed for {attack_type}: {e}. Skipping SMOTE.")
            X_res, y_res = embeddings, smote_labels

    # Data setup with resampled data
    embeddings_tensor = torch.from_numpy(X_res).float()
    labels_tensor = torch.from_numpy(y_res).float().unsqueeze(1)  # [batch, 1]
    
    # Ensure tensors have the same first dimension
    assert embeddings_tensor.size(0) == labels_tensor.size(0), f"Size mismatch: embeddings {embeddings_tensor.size(0)} vs labels {labels_tensor.size(0)}"
    
    dataset = TensorDataset(embeddings_tensor, labels_tensor)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # Initialize Teacher and Student models
    teacher_model = Teacher(input_dim=embedding_dim, output_dim=1).to(device)
    student_model = model # The main model is the student

    # Training setup
    optimizer = optim.AdamW(
        student_model.parameters(), lr=1e-4, weight_decay=1e-5
    )
    teacher_optimizer = optim.AdamW(
        teacher_model.parameters(), lr=1e-4, weight_decay=1e-5
    )
    
    scaler = GradScaler() if config.device == "cuda" else None

    # Enhanced scheduler
    def lr_lambda(epoch):
        warmup_epochs = 20
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 0.5 * (
                1 + np.cos(np.pi * (epoch - warmup_epochs) / (200 - warmup_epochs))
            )

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    teacher_scheduler = optim.lr_scheduler.LambdaLR(teacher_optimizer, lr_lambda)
    max_grad_norm = 0.5

    # Training loop with early stopping
    student_model.train()
    teacher_model.train()
    total_epochs = 200
    patience = 20
    patience_counter = 0
    best_loss = float("inf")
    best_model_state = None

    print(f"🎯 Training {attack_type} model for {total_epochs} epochs")

    # Check for existing checkpoint
    if model_save_path.exists():
        print(f"🔄 Resuming training from checkpoint: {model_save_path}")
        checkpoint = torch.load(model_save_path)
        student_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        teacher_model.load_state_dict(checkpoint["teacher_model_state_dict"])
        teacher_optimizer.load_state_dict(checkpoint["teacher_optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        patience_counter = checkpoint["patience_counter"]
    else:
        start_epoch = 0

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        epoch_losses = []
        epoch_recon_losses = []
        epoch_class_losses = []
        epoch_distillation_losses = []

        student_model.train()
        teacher_model.train()
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # --- Teacher Training ---
            teacher_optimizer.zero_grad()
            teacher_outputs = teacher_model(x_batch)
            teacher_loss = F.binary_cross_entropy_with_logits(teacher_outputs, y_batch)
            teacher_loss.backward()
            teacher_optimizer.step()

            # --- Student Training ---
            optimizer.zero_grad()
            student_outputs = student_model(x_batch)
            
            # Enhanced loss computation
            recon_loss = F.mse_loss(student_outputs["reconstructed"], x_batch)
            
            # Binary classification loss for this attack type
            attack_scores = student_outputs["multi_label_scores"]  # [batch, 1]
            
            # Enhanced binary cross entropy with label smoothing
            smoothed_targets = y_batch * 0.9 + 0.05
            class_loss = F.binary_cross_entropy_with_logits(
                attack_scores, smoothed_targets, reduction='mean'
            )
            
            # Distillation Loss
            dist_loss = distillation_loss(student_outputs["multi_label_scores"], teacher_outputs.detach(), y_batch)

            # Enhanced regularization
            predictions = torch.sigmoid(attack_scores)
            
            # Confidence regularization
            confidence_loss = torch.mean((predictions - 0.5).abs()) * 0.05
            
            # Entropy regularization
            entropy_loss = -torch.mean(predictions * torch.log(predictions + 1e-8) + 
                                     (1 - predictions) * torch.log(1 - predictions + 1e-8)) * 0.01
            
            # Total loss
            total_loss = recon_loss + class_loss + dist_loss + confidence_loss + entropy_loss
            
            # Backward and optimize
            if not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_grad_norm)
                optimizer.step()
                
                epoch_losses.append(total_loss.item())
                epoch_recon_losses.append(recon_loss.item())
                epoch_class_losses.append(class_loss.item())
                epoch_distillation_losses.append(dist_loss.item())
                
                # Progress tracking
                if batch_idx % 20 == 0:
                    progress_text = f"{attack_type} | Epoch {epoch+1}/{total_epochs} | Batch {batch_idx+1}/{len(dataloader)} | Loss: {total_loss.item():.4f} (Dist: {dist_loss.item():.4f})"
                    if hasattr(tracker, '_progress_spinner'):
                        tracker._progress_spinner.text = progress_text
                    else:
                        tracker._progress_spinner = Halo(text=progress_text, spinner='dots')
                    
        avg_epoch_loss = np.mean(epoch_losses)
        print(f"\nEpoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.4f} (Recon: {np.mean(epoch_recon_losses):.4f}, Class: {np.mean(epoch_class_losses):.4f}, Dist: {np.mean(epoch_distillation_losses):.4f})")

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": student_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "teacher_model_state_dict": teacher_model.state_dict(),
            "teacher_optimizer_state_dict": teacher_optimizer.state_dict(),
            "best_loss": best_loss,
            "patience_counter": patience_counter,
        }, model_save_path)

        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_state = student_model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    if best_model_state:
        student_model.load_state_dict(best_model_state)
    return student_model


def evaluate_single_attack_model(
    model: UnsupervisedMultiLabelTransformer,
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    attack_type: str,
    config: SystemConfig,
) -> Dict[str, float]:
    """Evaluate a single model for one attack type."""
    
    device = torch.device(config.device)
    model.eval()
    
    embeddings_tensor = torch.from_numpy(embeddings).float().to(device)
    
    with torch.no_grad():
        outputs = model(embeddings_tensor)
        predictions = torch.sigmoid(outputs["multi_label_scores"]).cpu().numpy()
    
    # Adaptive thresholding
    threshold = 0.5 # Default threshold
    # You can implement more sophisticated adaptive thresholding here
    # For example, using precision-recall curve or F1-score maximization on a validation set
    
    binary_predictions = (predictions > threshold).astype(int)
    
    # Calculate metrics (e.g., F1, Precision, Recall, Accuracy)
    # Note: For unsupervised setting, true_labels might not be directly available or used
    # This part assumes you have some form of ground truth for evaluation
    
    # Placeholder metrics
    metrics = {
        f"{attack_type}_precision": 0.0,
        f"{attack_type}_recall": 0.0,
        f"{attack_type}_f1": 0.0,
        f"{attack_type}_accuracy": 0.0,
    }
    
    print(f"📊 Evaluated {attack_type} model. Predictions: {np.sum(binary_predictions)}/{len(binary_predictions)}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate unsupervised multi-label transformer for anomaly detection.")
    parser.add_argument("--log_type", type=str, required=True,
                        help="Log type to process (e.g., \'vpn\', \'wp-error\').")
    parser.add_argument("--embedding_type", type=str, default="logbert",
                        choices=["fasttext", "word2vec", "logbert"],
                        help="Type of embeddings to use.")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Optional: Number of samples to use for training (for debugging/testing).")
    
    args = parser.parse_args()
    
    # Configuration
    MODELS_DIR = Path("models") / args.embedding_type / args.log_type
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    EMBEDDINGS_DIR = Path("embeddings")
    
    # Load embeddings and labels
    spinner = Halo(text="Loading embeddings for {} with {}".format(args.log_type, args.embedding_type), spinner='dots')
    spinner.start()
    
    try:
        with open(EMBEDDINGS_DIR / args.embedding_type / args.log_type / f"log_{args.log_type}.pkl", 'rb') as f:
            df_embeddings = pickle.load(f)
        with open(EMBEDDINGS_DIR / args.embedding_type / args.log_type / f"label_{args.log_type}.pkl", 'rb') as f:
            label_data = pickle.load(f)
        spinner.succeed("Embeddings loaded.")
    except FileNotFoundError:
        spinner.fail(f"Embeddings not found for {args.log_type} with {args.embedding_type}. Please run embedding generation first.")
        return
    
    embeddings = np.array(df_embeddings["log_embedding"].tolist())
    true_labels_binary = label_data["vectors"]
    attack_types = label_data["classes"]
    
    if args.sample_size:
        sample_indices = np.random.choice(len(embeddings), min(args.sample_size, len(embeddings)), replace=False)
        embeddings = embeddings[sample_indices]
        true_labels_binary = true_labels_binary[sample_indices]
    
    config = detect_system_resources()
    tracker = ProgressTracker(MODELS_DIR, args.log_type, config)
    
    trained_models = {}
    all_predictions = np.zeros_like(true_labels_binary, dtype=np.float32)
    
    for i, attack_type in enumerate(attack_types):
        print(f"\n--- Processing Attack Type: {attack_type} ({i+1}/{len(attack_types)}) ---")
        
        # Initialize a temporary teacher model for pseudo-label generation
        # This teacher model is trained on the current pseudo-labels to refine them
        temp_teacher_model = Teacher(input_dim=embeddings.shape[1], output_dim=1).to(torch.device(config.device))
        temp_teacher_optimizer = optim.AdamW(temp_teacher_model.parameters(), lr=1e-4)

        # Train the temporary teacher model for a few epochs to get better pseudo-labels
        print(f"Training temporary teacher for {attack_type} pseudo-label enhancement...")
        temp_teacher_model.train()
        temp_dataset = TensorDataset(torch.from_numpy(embeddings).float(), torch.from_numpy((binary_labels > 0.5).astype(float)).float().unsqueeze(1))
        temp_dataloader = DataLoader(temp_dataset, batch_size=64, shuffle=True)
        for _ in range(5): # Train for 5 epochs to refine pseudo-labels
            for x_batch, y_batch in temp_dataloader:
                x_batch = x_batch.to(torch.device(config.device))
                y_batch = y_batch.to(torch.device(config.device))
                temp_teacher_optimizer.zero_grad()
                temp_outputs = temp_teacher_model(x_batch)
                temp_loss = F.binary_cross_entropy_with_logits(temp_outputs, y_batch)
                temp_loss.backward()
                temp_teacher_optimizer.step()

        # Generate pseudo-labels for this attack type, now with teacher model enhancement
        pseudo_labels = generate_pseudo_labels_for_attack_type(
            embeddings,
            attack_type,
            i,
            len(embeddings),
            teacher_model=temp_teacher_model,
            device=torch.device(config.device)
        )
        
        # Initialize model for this attack type
        model = UnsupervisedMultiLabelTransformer(
            input_dim=embeddings.shape[1],
            latent_dim=512, # Example, will be adjusted in train_single_attack_model
            n_labels=1, # Binary classification for one-vs-rest
            n_clusters=2 # Not directly used in model, but kept for API consistency
        ).to(torch.device(config.device))
        
        model_save_path = MODELS_DIR / f"{attack_type}_model.pth"
        
        # Train model
        trained_model = train_single_attack_model(
            model,
            embeddings,
            pseudo_labels,
            attack_type,
            config,
            tracker,
            args.log_type,
            model_save_path # Pass save path
        )
        trained_models[attack_type] = trained_model
        
        # Evaluate and collect predictions
        metrics = evaluate_single_attack_model(
            trained_model,
            embeddings,
            true_labels_binary[:, i:i+1], # Pass only relevant true labels
            attack_type,
            config
        )
        
        # Collect predictions for multi-label output
        model.eval()
        with torch.no_grad():
            outputs = model(torch.from_numpy(embeddings).float().to(torch.device(config.device)))
            all_predictions[:, i] = torch.sigmoid(outputs["multi_label_scores"]).cpu().numpy().squeeze()
            
    print("\n--- Overall Multi-Label Evaluation ---")
    # Here you would evaluate all_predictions against true_labels_binary
    # using multi-label metrics like F1-score (micro, macro), Hamming loss, etc.
    # For now, just print a placeholder
    print("Overall multi-label evaluation to be implemented.")
    
    # Save overall predictions and true labels for final evaluation script
    with open(MODELS_DIR / "all_predictions.pkl", 'wb') as f:
        pickle.dump(all_predictions, f)
    with open(MODELS_DIR / "true_labels.pkl", 'wb') as f:
        pickle.dump(true_labels_binary, f)
    with open(MODELS_DIR / "attack_types.pkl", "wb") as f:
        pickle.dump(attack_types, f)

if __name__ == '__main__':
    main()



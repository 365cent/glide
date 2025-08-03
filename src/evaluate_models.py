#!/usr/bin/env python3
"""
Transformer Model Evaluation Pipeline
====================================

Clean, focused evaluation of trained multi-label transformer models using the direct approach:

1. Load trained UnsupervisedMultiLabelTransformer model (single model for all classes)
2. Load LogBERT embeddings and true labels  
3. Run forward pass through model to get multi-label predictions
4. Compute standard supervised metrics (F1, Hamming loss, etc.)
5. Optional per-class threshold optimization

Usage:
    python src/evaluate_models.py --log-type wp-error
    python src/evaluate_models.py --log-type wp-error --optimize-thresholds
"""

import argparse
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import normalize
from sklearn.metrics import (
    precision_recall_fscore_support, f1_score, accuracy_score, 
    hamming_loss, jaccard_score, precision_score, recall_score,
    classification_report
)
from scipy import stats

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from transformer module
import sys
sys.path.append(".")
from src.transformer import UnsupervisedMultiLabelTransformer, SystemConfig, detect_system_resources


class TransformerEvaluator:
    """Clean multi-label transformer model evaluator using direct supervised approach"""
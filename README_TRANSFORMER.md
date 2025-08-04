# Enhanced Transformer for Anomaly Detection

This updated transformer system provides automatic log type detection and works with multiple embedding types (FastText, Word2Vec, LogBERT) as downstream components.

## Key Features

- **Automatic Log Type Detection**: Automatically detects available log types from embeddings directory
- **Multi-Embedding Support**: Works with FastText, Word2Vec, and LogBERT embeddings
- **SMOTE Integration**: Optional SMOTE oversampling for imbalanced data
- **Hierarchical Classification**: Optional hierarchical classification support
- **Comprehensive Evaluation**: Built-in evaluation framework with concise reports
- **Self-Supervised Learning**: Uses reconstruction and contrastive learning objectives

## Quick Start

### 1. Basic Usage (Auto-detect log type)

```bash
# Train with FastText embeddings (auto-detects log type)
python src/run_pipeline.py --embedding_type fasttext

# Train with Word2Vec embeddings
python src/run_pipeline.py --embedding_type word2vec

# Train with LogBERT embeddings
python src/run_pipeline.py --embedding_type logbert
```

### 2. Specify Log Type

```bash
# Train with specific log type
python src/run_pipeline.py --embedding_type fasttext --log_type vpn

# Train with SMOTE oversampling
python src/run_pipeline.py --embedding_type word2vec --log_type wp-error --use_smote

# Train with hierarchical classification
python src/run_pipeline.py --embedding_type logbert --log_type vpn --use_hierarchical
```

### 3. Full Pipeline with Evaluation

```bash
# Run complete pipeline with evaluation
python src/run_pipeline.py --embedding_type fasttext --use_smote --use_hierarchical --evaluate
```

## Command Line Options

### run_pipeline.py

- `--embedding_type`: Embedding type to use (`fasttext`, `word2vec`, `logbert`)
- `--log_type`: Specific log type (auto-detected if not specified)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--use_smote`: Enable SMOTE oversampling
- `--use_hierarchical`: Enable hierarchical classification
- `--evaluate`: Run comprehensive evaluation after training
- `--output_dir`: Output directory for results

### transformer.py (Direct Usage)

```bash
# Basic training
python src/transformer.py --embedding_type fasttext

# Advanced configuration
python src/transformer.py --embedding_type logbert --log_type vpn --epochs 100 --batch_size 64 --use_smote --use_hierarchical
```

## Directory Structure

The system expects the following directory structure:

```
embeddings/
├── fasttext/
│   ├── vpn/
│   │   ├── log_vpn.pkl
│   │   └── label_vpn.pkl
│   └── wp-error/
│       ├── log_wp-error.pkl
│       └── label_wp-error.pkl
├── word2vec/
│   └── ...
└── logbert/
    └── ...
```

## Output Files

### Training Results

- `model.pth`: Trained model weights
- `config.json`: Model configuration
- `predictions.pkl`: Anomaly predictions
- `training_stats.json`: Training statistics

### Evaluation Results (if --evaluate is used)

- `evaluation_report_{log_type}_{embedding_type}.txt`: Human-readable evaluation report
- `evaluation_results_{log_type}_{embedding_type}.json`: Detailed evaluation metrics
- `evaluation_{log_type}_{embedding_type}.png`: Visualization plots

## Example Evaluation Report

```
============================================================
EVALUATION REPORT
Log Type: vpn
Embedding Type: fasttext
============================================================

DETECTION PERFORMANCE:
  Accuracy:  0.9234
  Precision: 0.9156
  Recall:    0.9289
  F1-Score:  0.9222
  AUC-ROC:   0.9456

HIERARCHICAL PERFORMANCE:
  Level 1 Accuracy: 0.9345
  Level 2 Accuracy: 0.9123
  Path Consistency: 0.8765

CLUSTERING QUALITY:
  Silhouette Score: 0.7234
  Adjusted Rand Index: 0.6543

SMOTE OVERSAMPLING QUALITY:
  Balance Improvement: 0.2345
  Synthetic Quality: 0.8123
  Diversity Score: 0.7456

COMPUTATIONAL PERFORMANCE:
  Inference Time: 0.0234s
  Samples/Second: 42.8
  Memory Usage: 156.7MB

CROSS-VALIDATION RESULTS:
  F1-Score: 0.9234 ± 0.0234
  AUC-ROC: 0.9456 ± 0.0156
============================================================
```

## Advanced Usage

### Custom Configuration

```python
from transformer import TransformerConfig, create_enhanced_transformer

# Create custom configuration
config = TransformerConfig(
    d_model=512,
    n_layers=6,
    epochs=100,
    batch_size=32,
    use_smote=True,
    use_hierarchical=True,
    reconstruction_weight=1.0,
    contrastive_weight=0.5,
    classification_weight=0.3
)

# Create trainer
trainer = create_enhanced_transformer(input_dim=300, config=config)
```

### Programmatic Usage

```python
from transformer import load_embeddings_and_labels, create_enhanced_transformer
from comprehensive_evaluator import ComprehensiveEvaluator

# Load data
embeddings, label_data = load_embeddings_and_labels('vpn', 'fasttext')

# Create and train model
trainer = create_enhanced_transformer(embeddings.shape[1])
# ... training code ...

# Evaluate
evaluator = ComprehensiveEvaluator()
results = evaluator.evaluate_pipeline('vpn', 'fasttext', embeddings, labels)
report = evaluator.generate_evaluation_report(results, 'vpn', 'fasttext')
print(report)
```

## Troubleshooting

### Common Issues

1. **No log types found**: Ensure embeddings are generated first using the embedding scripts
2. **Memory issues**: Reduce batch size or use smaller model dimensions
3. **CUDA out of memory**: Use CPU device or reduce model size
4. **Import errors**: Ensure all dependencies are installed

### Performance Tips

- Use `--batch_size 16` for large datasets
- Enable `--use_smote` for imbalanced data
- Use `--use_hierarchical` for structured classification
- Run with `--evaluate` for comprehensive analysis

## Dependencies

- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- tqdm
- halo (for progress indicators)

## Architecture

The enhanced transformer includes:

1. **Input Projection**: Maps embeddings to transformer dimension
2. **Positional Encoding**: Adds sequence position information
3. **Multi-Head Attention**: Captures complex relationships
4. **Feed-Forward Networks**: Non-linear transformations
5. **Reconstruction Head**: Self-supervised learning objective
6. **Contrastive Head**: Contrastive learning objective
7. **Hierarchical Head**: Multi-level classification (optional)
8. **Anomaly Detector**: Final anomaly scoring

The system automatically adapts to different embedding dimensions and provides comprehensive evaluation metrics for anomaly detection performance.
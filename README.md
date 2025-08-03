# Log Anomaly Detection Pipeline

A comprehensive, efficient pipeline for log anomaly detection using multiple embedding types and transformer models.

## 🚀 Features

- **Multi-Embedding Support**: FastText, Word2Vec, and LogBERT embeddings
- **Transformer Models**: Unsupervised multi-label transformer training
- **Comprehensive Evaluation**: Detailed metrics and visualizations
- **Resume Capability**: Automatic checkpointing and resume functionality
- **Performance Optimization**: Auto-optimization based on dataset size
- **Memory Efficient**: Optimized for large-scale log processing
- **GPU Support**: Automatic GPU detection and utilization

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for acceleration)
- 8GB+ RAM (16GB+ recommended for large datasets)

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd log-anomaly-detection
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "import torch, tensorflow, transformers; print('✅ All dependencies installed successfully')"
```

## 📁 Project Structure

```
log-anomaly-detection/
├── src/                          # Source code
│   ├── config.py                 # Configuration and paths
│   ├── preprocessing.py          # Data preprocessing
│   ├── fasttext_embedding.py    # FastText embeddings
│   ├── word2vec_embedding.py    # Word2Vec embeddings
│   ├── logbert_embeddings.py    # LogBERT embeddings
│   ├── transformer.py           # Transformer model training
│   ├── evaluate_models.py       # Model evaluation
│   └── ...
├── logs/                         # Input log files
├── labels/                       # Label files
├── processed/                    # TFRecord files
├── embeddings/                   # Generated embeddings
├── models/                       # Trained models
├── results/                      # Evaluation results
├── checkpoints/                  # Checkpoint files
├── run_pipeline.py              # Main pipeline script
└── requirements.txt              # Dependencies
```

## 🎯 Quick Start

### 1. Prepare Your Data

Place your log files in the `logs/` directory and corresponding label files in the `labels/` directory:

```
logs/
├── vpn/
│   ├── user1_vpn.log
│   └── user2_vpn.log
labels/
├── vpn/
│   ├── user1_vpn.json
│   └── user2_vpn.json
```

### 2. Run the Complete Pipeline

```bash
# Run complete pipeline for VPN logs
python run_pipeline.py --log_type vpn

# Run with specific embedding types
python run_pipeline.py --log_type wp-error --embedding_types fasttext logbert

# Run with threshold optimization
python run_pipeline.py --log_type auth --optimize_thresholds

# Force restart (ignore existing outputs)
python run_pipeline.py --log_type vpn --force_restart
```

### 3. Check Data Availability

```bash
# Check if data is ready for processing
python run_pipeline.py --log_type vpn --check_only
```

## 🔧 Pipeline Stages

The pipeline consists of 5 main stages:

### Stage 1: Data Preprocessing
- Converts raw log files to TFRecord format
- Handles various log formats and encodings
- Creates compact, efficient data representation

### Stage 2: Embedding Generation
- **FastText**: 300D vectors using pre-trained models
- **Word2Vec**: 300D vectors using Google News vectors
- **LogBERT**: 2314D enhanced BERT vectors (CLS + Mean + Max + Attention)

### Stage 3: Model Training
- Trains transformer models for each embedding type
- Uses unsupervised multi-label approach
- Automatic hyperparameter optimization

### Stage 4: Evaluation
- Comprehensive metrics calculation
- Per-class and overall performance analysis
- ROC curves and confusion matrices

### Stage 5: Results Analysis
- Performance comparison across embedding types
- Visualization and reporting
- Export of detailed metrics

## 📊 Output Files

For each log type, the pipeline generates:

```
results/{log_type}/
├── detailed_metrics.json         # Detailed evaluation metrics
├── metrics_summary.csv          # Performance comparison table
├── performance_comparison.png   # Visualization of results
├── {embedding_type}/           # Per-embedding results
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   └── precision_recall_curves.png
└── pipeline_report.json        # Complete pipeline report
```

## ⚡ Performance Optimization

The pipeline automatically optimizes performance based on:

- **Dataset Size**: Adjusts batch sizes and workers
- **Available Memory**: Scales processing accordingly
- **GPU Availability**: Uses CUDA/MPS when available
- **System Resources**: Monitors and adapts to constraints

### Performance Configurations

| Dataset Size | Batch Size | Workers | Clear Frequency |
|-------------|------------|---------|-----------------|
| < 10K       | 16         | 4       | 100            |
| < 100K      | 12         | 3       | 50             |
| < 500K      | 8          | 2       | 25             |
| > 500K      | 4          | 1       | 10             |

## 🔄 Resume Capability

The pipeline supports automatic resumption:

- **Checkpointing**: Saves progress every 5%
- **Incremental Processing**: Skips completed stages
- **Error Recovery**: Continues from last successful point
- **Data Validation**: Ensures consistency across restarts

## 🎛️ Advanced Usage

### Custom Configuration

Modify `src/config.py` for custom settings:

```python
# Performance thresholds
SMALL_DATASET_THRESHOLD = 10000
MEDIUM_DATASET_THRESHOLD = 100000
LARGE_DATASET_THRESHOLD = 500000

# Performance configurations
PERF_CONFIG = {
    'small': {'batch_size': 16, 'workers': 4, 'clear_freq': 100},
    'medium': {'batch_size': 12, 'workers': 3, 'clear_freq': 50},
    # ...
}
```

### Individual Stage Execution

Run specific stages independently:

```bash
# Preprocessing only
python src/preprocessing.py

# Embedding generation only
python src/fasttext_embedding.py --log_type vpn
python src/word2vec_embedding.py --log_type vpn
python src/logbert_embeddings.py --log_type vpn

# Model training only
python src/transformer.py --log_type vpn --embedding_type fasttext

# Evaluation only
python src/evaluate_models.py --log_type vpn --embedding_types fasttext word2vec logbert
```

### GPU Acceleration

The pipeline automatically detects and uses available GPUs:

- **CUDA**: NVIDIA GPUs
- **MPS**: Apple M1/M2 GPUs
- **CPU**: Fallback for systems without GPU

## 📈 Evaluation Metrics

The pipeline provides comprehensive evaluation:

### Overall Metrics
- Accuracy, Precision, Recall, F1-Score
- Micro and Macro averages
- AUC-ROC and AUC-PR

### Per-Class Metrics
- Individual class performance
- Confusion matrices
- ROC curves per class

### Visualization
- Performance comparison charts
- t-SNE embeddings visualization
- Precision-Recall curves

## 🐛 Troubleshooting

### Common Issues

1. **Memory Errors**:
   ```bash
   # Reduce batch size in config.py
   PERF_CONFIG['large']['batch_size'] = 4
   ```

2. **GPU Issues**:
   ```bash
   # Force CPU usage
   export CUDA_VISIBLE_DEVICES=""
   ```

3. **Data Format Issues**:
   ```bash
   # Validate data format
   python src/data_validation.py --log_type vpn
   ```

### Log Files

- **Pipeline Log**: `pipeline.log`
- **Individual Stage Logs**: Check console output
- **Error Details**: See specific stage logs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- FastText team for pre-trained models
- Hugging Face for transformers library
- PyTorch and TensorFlow communities
- Scikit-learn for evaluation metrics

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Review the logs for error details
3. Open an issue on GitHub
4. Contact the development team

---

**Happy Log Anomaly Detection! 🎉**

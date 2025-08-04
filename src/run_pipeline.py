#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Runner for Enhanced Anomaly Detection

This script demonstrates how to use the updated transformer with automatic
log type detection and different embedding types.

Usage:
    python run_pipeline.py --embedding_type fasttext
    python run_pipeline.py --embedding_type word2vec --log_type vpn
    python run_pipeline.py --embedding_type logbert --use_smote --use_hierarchical
"""

import argparse
import logging
from pathlib import Path
import sys

# Import custom modules
from transformer import detect_available_log_types, load_embeddings_and_labels, create_enhanced_transformer, TransformerConfig
from comprehensive_evaluator import ComprehensiveEvaluator
from config import EMBEDDINGS_DIR, MODELS_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PipelineRunner")

def main():
    """Main function for running the complete pipeline."""
    parser = argparse.ArgumentParser(description="Run enhanced anomaly detection pipeline")
    parser.add_argument("--embedding_type", type=str, required=True,
                        choices=['fasttext', 'word2vec', 'logbert'],
                        help="Embedding type to use")
    parser.add_argument("--log_type", type=str, default=None,
                        help="Specific log type to process (auto-detected if not specified)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--use_smote", action='store_true',
                        help="Use SMOTE oversampling")
    parser.add_argument("--use_hierarchical", action='store_true',
                        help="Use hierarchical classification")
    parser.add_argument("--evaluate", action='store_true',
                        help="Run comprehensive evaluation after training")
    parser.add_argument("--output_dir", type=str, default=str(MODELS_DIR / "enhanced_transformer"),
                        help="Output directory")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("ENHANCED ANOMALY DETECTION PIPELINE")
    logger.info("=" * 60)
    
    # Step 1: Detect available log types
    logger.info("Step 1: Detecting available log types...")
    available_log_types = detect_available_log_types()
    
    if not available_log_types:
        logger.error("No log types found in embeddings directory")
        return
    
    # Filter by embedding type
    matching_log_types = [(lt, et) for lt, et in available_log_types if et == args.embedding_type]
    if not matching_log_types:
        logger.error(f"No log types found for embedding type: {args.embedding_type}")
        logger.info(f"Available combinations: {available_log_types}")
        return
    
    # Select log type
    if args.log_type is None:
        # Use the first available log type
        args.log_type = matching_log_types[0][0]
        logger.info(f"Auto-detected log type: {args.log_type}")
    else:
        # Verify the specified log type exists
        if not any(lt == args.log_type for lt, et in matching_log_types):
            logger.error(f"Log type '{args.log_type}' not found for embedding type '{args.embedding_type}'")
            logger.info(f"Available log types for {args.embedding_type}: {[lt for lt, et in matching_log_types]}")
            return
    
    logger.info(f"Selected configuration: {args.log_type} with {args.embedding_type} embeddings")
    
    # Step 2: Load embeddings and labels
    logger.info("Step 2: Loading embeddings and labels...")
    try:
        embeddings, label_data = load_embeddings_and_labels(args.log_type, args.embedding_type)
        logger.info(f"Loaded embeddings with shape: {embeddings.shape}")
        logger.info(f"Loaded labels with {len(label_data['classes'])} classes")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Step 3: Create and configure transformer
    logger.info("Step 3: Creating enhanced transformer...")
    config = TransformerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_smote=args.use_smote,
        use_hierarchical=args.use_hierarchical
    )
    
    trainer = create_enhanced_transformer(embeddings.shape[1], config)
    
    # Step 4: Train the model
    logger.info("Step 4: Training enhanced transformer...")
    from torch.utils.data import DataLoader, TensorDataset
    import torch
    
    # Prepare data
    embeddings_tensor = torch.FloatTensor(embeddings)
    dataset = torch.utils.data.TensorDataset(embeddings_tensor)
    
    # Split into train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Train model
    output_path = Path(args.output_dir) / f"{args.log_type}_{args.embedding_type}"
    training_stats = trainer.train(train_dataloader, val_dataloader, output_path)
    
    # Step 5: Evaluate model
    logger.info("Step 5: Evaluating model...")
    predictions = trainer.predict_anomalies(val_dataloader)
    
    # Save results
    import pickle
    import json
    
    with open(output_path / "predictions.pkl", 'wb') as f:
        pickle.dump(predictions, f)
    
    with open(output_path / "training_stats.json", 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    logger.info(f"Training completed. Results saved to {output_path}")
    logger.info(f"Detected {sum(predictions['predictions'])} anomalies out of {len(predictions['predictions'])} samples")
    
    # Step 6: Comprehensive evaluation (optional)
    if args.evaluate:
        logger.info("Step 6: Running comprehensive evaluation...")
        evaluator = ComprehensiveEvaluator()
        
        # Convert labels to binary (anomaly vs normal)
        import numpy as np
        labels = np.any(label_data['vectors'], axis=1).astype(int)
        
        # Evaluate pipeline
        results = evaluator.evaluate_pipeline(
            args.log_type, args.embedding_type, embeddings, labels
        )
        
        # Generate and save report
        report = evaluator.generate_evaluation_report(results, args.log_type, args.embedding_type)
        
        evaluation_path = Path(args.output_dir) / "evaluation"
        evaluation_path.mkdir(parents=True, exist_ok=True)
        
        # Save report
        report_path = evaluation_path / f"evaluation_report_{args.log_type}_{args.embedding_type}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Save detailed results
        results_path = evaluation_path / f"evaluation_results_{args.log_type}_{args.embedding_type}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create visualizations
        evaluator.create_visualizations(results, args.log_type, args.embedding_type)
        
        # Print report
        print("\n" + "=" * 60)
        print("COMPREHENSIVE EVALUATION REPORT")
        print("=" * 60)
        print(report)
        
        logger.info(f"Evaluation completed. Results saved to {evaluation_path}")
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
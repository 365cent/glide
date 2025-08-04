#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the enhanced anomaly detection pipeline.

This script tests the integration of all components:
- Transformer with automatic log type detection
- SMOTE oversampling
- Hierarchical classification
- Comprehensive evaluation
"""

import numpy as np
import pickle
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestPipeline")

def create_test_data():
    """Create test embeddings and labels for testing."""
    logger.info("Creating test data...")
    
    # Create test embeddings (1000 samples, 300 dimensions)
    embeddings = np.random.randn(1000, 300).astype(np.float32)
    
    # Create test labels (binary: 0 = normal, 1 = anomaly)
    # Make it imbalanced: 80% normal, 20% anomaly
    labels = np.zeros(1000, dtype=np.int8)
    labels[800:] = 1  # Last 200 are anomalies
    
    # Create label data structure
    label_data = {
        'vectors': np.column_stack([labels, 1 - labels]),  # One-hot encoding
        'classes': ['normal', 'anomaly'],
        'description': 'Binary classification: normal vs anomaly'
    }
    
    return embeddings, label_data

def save_test_data(embeddings, label_data, log_type="test", embedding_type="fasttext"):
    """Save test data to the expected directory structure."""
    logger.info(f"Saving test data for {log_type} with {embedding_type} embeddings...")
    
    # Create directory structure
    from config import EMBEDDINGS_DIR
    output_dir = EMBEDDINGS_DIR / embedding_type / log_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings
    embeddings_path = output_dir / f"log_{log_type}.pkl"
    with open(embeddings_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
    # Save labels
    labels_path = output_dir / f"label_{log_type}.pkl"
    with open(labels_path, 'wb') as f:
        pickle.dump(label_data, f)
    
    logger.info(f"Test data saved to {output_dir}")
    return output_dir

def test_transformer_imports():
    """Test that all transformer components can be imported."""
    logger.info("Testing transformer imports...")
    
    try:
        from transformer import (
            TransformerConfig, 
            EnhancedUnsupervisedTransformer,
            EnhancedTransformerTrainer,
            detect_available_log_types,
            load_embeddings_and_labels,
            create_enhanced_transformer
        )
        logger.info("✓ Transformer imports successful")
        return True
    except Exception as e:
        logger.error(f"✗ Transformer imports failed: {e}")
        return False

def test_smote_imports():
    """Test that SMOTE components can be imported."""
    logger.info("Testing SMOTE imports...")
    
    try:
        from smote_oversampling import UnsupervisedSMOTEOversampler
        logger.info("✓ SMOTE imports successful")
        return True
    except Exception as e:
        logger.error(f"✗ SMOTE imports failed: {e}")
        return False

def test_hierarchical_imports():
    """Test that hierarchical classifier components can be imported."""
    logger.info("Testing hierarchical classifier imports...")
    
    try:
        from hierarchical_classifier import (
            UnsupervisedHierarchicalClassifier, 
            AttackTaxonomy,
            ClassificationResult
        )
        logger.info("✓ Hierarchical classifier imports successful")
        return True
    except Exception as e:
        logger.error(f"✗ Hierarchical classifier imports failed: {e}")
        return False

def test_evaluator_imports():
    """Test that evaluator components can be imported."""
    logger.info("Testing evaluator imports...")
    
    try:
        from comprehensive_evaluator import ComprehensiveEvaluator, EvaluationConfig
        logger.info("✓ Evaluator imports successful")
        return True
    except Exception as e:
        logger.error(f"✗ Evaluator imports failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality."""
    logger.info("Testing data loading...")
    
    try:
        from transformer import load_embeddings_and_labels
        
        # Create and save test data
        embeddings, label_data = create_test_data()
        save_test_data(embeddings, label_data)
        
        # Test loading
        loaded_embeddings, loaded_label_data = load_embeddings_and_labels("test", "fasttext")
        
        # Verify data integrity
        assert loaded_embeddings.shape == embeddings.shape, "Embedding shapes don't match"
        assert loaded_label_data['classes'] == label_data['classes'], "Label classes don't match"
        
        logger.info("✓ Data loading successful")
        return True
    except Exception as e:
        logger.error(f"✗ Data loading failed: {e}")
        return False

def test_transformer_creation():
    """Test transformer model creation."""
    logger.info("Testing transformer creation...")
    
    try:
        from transformer import TransformerConfig, create_enhanced_transformer
        
        # Create configuration
        config = TransformerConfig(
            d_model=256,  # Smaller for testing
            n_layers=2,   # Fewer layers for testing
            epochs=2,     # Few epochs for testing
            batch_size=16,
            use_smote=False,  # Disable for testing
            use_hierarchical=False  # Disable for testing
        )
        
        # Create trainer
        trainer = create_enhanced_transformer(input_dim=300, config=config)
        
        logger.info("✓ Transformer creation successful")
        return True
    except Exception as e:
        logger.error(f"✗ Transformer creation failed: {e}")
        return False

def test_smote_functionality():
    """Test SMOTE oversampling functionality."""
    logger.info("Testing SMOTE functionality...")
    
    try:
        from smote_oversampling import UnsupervisedSMOTEOversampler
        
        # Create test data
        embeddings = np.random.randn(100, 50).astype(np.float32)
        labels = np.random.randint(0, 3, 100)
        
        # Create SMOTE oversampler
        smote = UnsupervisedSMOTEOversampler()
        
        # Generate pseudo-labels
        pseudo_labels = smote.generate_pseudo_labels(embeddings)
        
        # Apply SMOTE
        smote_embeddings, smote_labels = smote.apply_smote_variant(
            embeddings, pseudo_labels, variant='smote'
        )
        
        logger.info(f"✓ SMOTE functionality successful (original: {len(embeddings)}, augmented: {len(smote_embeddings)})")
        return True
    except Exception as e:
        logger.error(f"✗ SMOTE functionality failed: {e}")
        return False

def test_hierarchical_functionality():
    """Test hierarchical classifier functionality."""
    logger.info("Testing hierarchical classifier functionality...")
    
    try:
        from hierarchical_classifier import UnsupervisedHierarchicalClassifier
        
        # Create test data
        embeddings = np.random.randn(100, 50).astype(np.float32)
        
        # Create hierarchical classifier
        classifier = UnsupervisedHierarchicalClassifier()
        
        # Test prediction
        results = classifier.predict_hierarchical(embeddings, confidence_threshold=0.5)
        
        logger.info(f"✓ Hierarchical classifier functionality successful (processed {len(results)} samples)")
        return True
    except Exception as e:
        logger.error(f"✗ Hierarchical classifier functionality failed: {e}")
        return False

def test_evaluator_functionality():
    """Test evaluator functionality."""
    logger.info("Testing evaluator functionality...")
    
    try:
        from comprehensive_evaluator import ComprehensiveEvaluator
        
        # Create test data
        embeddings = np.random.randn(100, 50).astype(np.float32)
        labels = np.random.randint(0, 2, 100)
        
        # Create evaluator
        evaluator = ComprehensiveEvaluator()
        
        # Test evaluation
        results = evaluator.evaluate_pipeline("test", "fasttext", embeddings, labels)
        
        # Test report generation
        report = evaluator.generate_evaluation_report(results, "test", "fasttext")
        
        logger.info("✓ Evaluator functionality successful")
        return True
    except Exception as e:
        logger.error(f"✗ Evaluator functionality failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("TESTING ENHANCED ANOMALY DETECTION PIPELINE")
    logger.info("=" * 60)
    
    tests = [
        ("Transformer Imports", test_transformer_imports),
        ("SMOTE Imports", test_smote_imports),
        ("Hierarchical Classifier Imports", test_hierarchical_imports),
        ("Evaluator Imports", test_evaluator_imports),
        ("Data Loading", test_data_loading),
        ("Transformer Creation", test_transformer_creation),
        ("SMOTE Functionality", test_smote_functionality),
        ("Hierarchical Classifier Functionality", test_hierarchical_functionality),
        ("Evaluator Functionality", test_evaluator_functionality),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\nRunning test: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
        if success:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! The pipeline is ready to use.")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
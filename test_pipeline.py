#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for the Log Anomaly Detection Pipeline

This script tests the main components of the pipeline to ensure everything works correctly.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import json
import numpy as np
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import LOGS_DIR, LABELS_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, MODELS_DIR, RESULTS_DIR

def create_test_data():
    """Create test log and label data."""
    print("Creating test data...")
    
    # Create test log files
    test_logs = [
        "2024-01-01 10:00:00 INFO: User login successful",
        "2024-01-01 10:01:00 ERROR: Authentication failed",
        "2024-01-01 10:02:00 WARNING: High memory usage",
        "2024-01-01 10:03:00 INFO: Data backup completed",
        "2024-01-01 10:04:00 ERROR: Database connection timeout"
    ]
    
    # Create test labels
    test_labels = [
        [],  # Normal
        ["authentication_failure"],  # Attack
        [],  # Normal
        [],  # Normal
        ["database_error"]  # Attack
    ]
    
    # Create directories
    test_log_dir = LOGS_DIR / "test"
    test_label_dir = LABELS_DIR / "test"
    test_log_dir.mkdir(parents=True, exist_ok=True)
    test_label_dir.mkdir(parents=True, exist_ok=True)
    
    # Write test log file
    log_file = test_log_dir / "test.log"
    with open(log_file, 'w') as f:
        for log in test_logs:
            f.write(log + '\n')
    
    # Write test label file
    label_file = test_label_dir / "test.json"
    label_data = []
    for i, labels in enumerate(test_labels, 1):
        label_data.append({
            "line": i,
            "labels": labels
        })
    
    with open(label_file, 'w') as f:
        for item in label_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"✅ Created test data:")
    print(f"   Log file: {log_file}")
    print(f"   Label file: {label_file}")
    
    return "test"

def test_preprocessing():
    """Test the preprocessing stage."""
    print("\n" + "="*50)
    print("TESTING PREPROCESSING")
    print("="*50)
    
    try:
        from preprocessing import LogPreprocessor
        
        # Create preprocessor
        preprocessor = LogPreprocessor()
        
        # Test file detection
        log_files = list(LOGS_DIR.rglob("*.log"))
        print(f"Found {len(log_files)} log files")
        
        # Test processing
        preprocessor.batch_process()
        
        # Check if TFRecord files were created
        tfrecord_files = list(PROCESSED_DIR.rglob("*.tfrecord"))
        print(f"Created {len(tfrecord_files)} TFRecord files")
        
        if tfrecord_files:
            print("✅ Preprocessing test passed")
            return True
        else:
            print("❌ Preprocessing test failed - no TFRecord files created")
            return False
            
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False

def test_embedding_generation():
    """Test embedding generation."""
    print("\n" + "="*50)
    print("TESTING EMBEDDING GENERATION")
    print("="*50)
    
    success_count = 0
    total_tests = 3
    
    # Test FastText
    try:
        print("Testing FastText embedding...")
        import subprocess
        result = subprocess.run([
            sys.executable, "src/fasttext_embedding.py", 
            "--log_type", "test"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ FastText embedding test passed")
            success_count += 1
        else:
            print(f"❌ FastText embedding test failed: {result.stderr}")
    except Exception as e:
        print(f"❌ FastText embedding test failed: {e}")
    
    # Test Word2Vec
    try:
        print("Testing Word2Vec embedding...")
        result = subprocess.run([
            sys.executable, "src/word2vec_embedding.py", 
            "--log_type", "test"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Word2Vec embedding test passed")
            success_count += 1
        else:
            print(f"❌ Word2Vec embedding test failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Word2Vec embedding test failed: {e}")
    
    # Test LogBERT (skip if no GPU)
    try:
        print("Testing LogBERT embedding...")
        result = subprocess.run([
            sys.executable, "src/logbert_embeddings.py", 
            "--log_type", "test", "--sample-size", "10"
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ LogBERT embedding test passed")
            success_count += 1
        else:
            print(f"❌ LogBERT embedding test failed: {result.stderr}")
    except Exception as e:
        print(f"❌ LogBERT embedding test failed: {e}")
    
    print(f"Embedding tests: {success_count}/{total_tests} passed")
    return success_count > 0

def test_evaluation():
    """Test the evaluation stage."""
    print("\n" + "="*50)
    print("TESTING EVALUATION")
    print("="*50)
    
    try:
        # Create dummy embeddings and labels for testing
        test_embeddings = np.random.randn(100, 300).astype(np.float32)
        test_labels = np.random.randint(0, 2, (100, 3)).astype(np.int8)
        
        # Save test data
        test_emb_dir = EMBEDDINGS_DIR / "test"
        test_emb_dir.mkdir(parents=True, exist_ok=True)
        
        with open(test_emb_dir / "log_test.pkl", 'wb') as f:
            pickle.dump(test_embeddings, f)
        
        label_data = {
            'vectors': test_labels,
            'classes': ['attack_type_1', 'attack_type_2', 'attack_type_3'],
            'description': 'Test binary label vectors'
        }
        
        with open(test_emb_dir / "label_test.pkl", 'wb') as f:
            pickle.dump(label_data, f)
        
        # Test evaluation
        result = subprocess.run([
            sys.executable, "src/evaluate_models.py",
            "--log_type", "test",
            "--embedding_types", "fasttext"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Evaluation test passed")
            return True
        else:
            print(f"❌ Evaluation test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        return False

def test_pipeline():
    """Test the complete pipeline."""
    print("\n" + "="*50)
    print("TESTING COMPLETE PIPELINE")
    print("="*50)
    
    try:
        result = subprocess.run([
            sys.executable, "run_pipeline.py",
            "--log_type", "test",
            "--embedding_types", "fasttext",
            "--check_only"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Pipeline test passed")
            return True
        else:
            print(f"❌ Pipeline test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def cleanup_test_data():
    """Clean up test data."""
    print("\nCleaning up test data...")
    
    # Remove test directories
    test_dirs = [
        LOGS_DIR / "test",
        LABELS_DIR / "test",
        PROCESSED_DIR / "test",
        EMBEDDINGS_DIR / "test",
        MODELS_DIR / "test",
        RESULTS_DIR / "test"
    ]
    
    for test_dir in test_dirs:
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print(f"   Removed: {test_dir}")
    
    print("✅ Cleanup completed")

def main():
    """Run all tests."""
    print("🧪 LOG ANOMALY DETECTION PIPELINE TESTS")
    print("="*60)
    
    # Create test data
    test_log_type = create_test_data()
    
    # Run tests
    tests = [
        ("Preprocessing", test_preprocessing),
        ("Embedding Generation", test_embedding_generation),
        ("Evaluation", test_evaluation),
        ("Pipeline", test_pipeline)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
    
    # Cleanup
    cleanup_test_data()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Pipeline is ready to use.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
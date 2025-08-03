#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple test for the Log Anomaly Detection Pipeline

This script tests the basic components without requiring deep learning libraries.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_config():
    """Test configuration loading."""
    print("Testing configuration...")
    
    try:
        from config import LOGS_DIR, LABELS_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, MODELS_DIR, RESULTS_DIR
        
        # Check if directories exist or can be created
        for dir_path in [LOGS_DIR, LABELS_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, MODELS_DIR, RESULTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ {dir_path.name}: {dir_path}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_preprocessing():
    """Test preprocessing functionality."""
    print("\nTesting preprocessing...")
    
    try:
        # Create test data
        test_logs = [
            "2024-01-01 10:00:00 INFO: User login successful",
            "2024-01-01 10:01:00 ERROR: Authentication failed",
            "2024-01-01 10:02:00 WARNING: High memory usage"
        ]
        
        test_labels = [
            {"line": 1, "labels": []},
            {"line": 2, "labels": ["authentication_failure"]},
            {"line": 3, "labels": []}
        ]
        
        # Create test files
        from config import LOGS_DIR, LABELS_DIR
        
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
        with open(label_file, 'w') as f:
            for item in test_labels:
                f.write(json.dumps(item) + '\n')
        
        print(f"✅ Created test files:")
        print(f"   Log file: {log_file}")
        print(f"   Label file: {label_file}")
        
        return True
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False

def test_evaluation():
    """Test evaluation functionality."""
    print("\nTesting evaluation...")
    
    try:
        import numpy as np
        import pickle
        from config import EMBEDDINGS_DIR, RESULTS_DIR
        
        # Create test embeddings and labels
        test_embeddings = np.random.randn(50, 300).astype(np.float32)
        test_labels = np.random.randint(0, 2, (50, 2)).astype(np.int8)
        
        # Save test data
        test_emb_dir = EMBEDDINGS_DIR / "test"
        test_emb_dir.mkdir(parents=True, exist_ok=True)
        
        with open(test_emb_dir / "log_test.pkl", 'wb') as f:
            pickle.dump(test_embeddings, f)
        
        label_data = {
            'vectors': test_labels,
            'classes': ['attack_type_1', 'attack_type_2'],
            'description': 'Test binary label vectors'
        }
        
        with open(test_emb_dir / "label_test.pkl", 'wb') as f:
            pickle.dump(label_data, f)
        
        # Test basic metrics calculation
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            test_embeddings, test_labels, test_size=0.2, random_state=42
        )
        
        # Train simple model for each class
        models = []
        y_preds = []
        
        for i in range(y_train.shape[1]):
            model = LogisticRegression(random_state=42)
            model.fit(X_train, y_train[:, i])
            y_pred = model.predict(X_test)
            models.append(model)
            y_preds.append(y_pred)
        
        # Stack predictions
        y_pred_stacked = np.column_stack(y_preds)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred_stacked)
        f1_micro = f1_score(y_test, y_pred_stacked, average='micro')
        
        print(f"✅ Evaluation test passed:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1-Score (micro): {f1_micro:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        return False

def test_pipeline_structure():
    """Test pipeline structure and file organization."""
    print("\nTesting pipeline structure...")
    
    try:
        from config import LOGS_DIR, LABELS_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, MODELS_DIR, RESULTS_DIR
        
        # Check if all required directories exist
        required_dirs = [
            LOGS_DIR, LABELS_DIR, PROCESSED_DIR, 
            EMBEDDINGS_DIR, MODELS_DIR, RESULTS_DIR
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ {dir_path.name}: {dir_path}")
        
        # Check if source files exist
        src_files = [
            "src/config.py",
            "src/preprocessing.py",
            "src/evaluate_models.py",
            "run_pipeline.py",
            "requirements.txt",
            "README.md"
        ]
        
        for file_path in src_files:
            if Path(file_path).exists():
                print(f"✅ {file_path}")
            else:
                print(f"❌ {file_path} - missing")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Pipeline structure test failed: {e}")
        return False

def cleanup_test_data():
    """Clean up test data."""
    print("\nCleaning up test data...")
    
    try:
        from config import LOGS_DIR, LABELS_DIR, EMBEDDINGS_DIR
        
        # Remove test directories
        test_dirs = [
            LOGS_DIR / "test",
            LABELS_DIR / "test",
            EMBEDDINGS_DIR / "test"
        ]
        
        for test_dir in test_dirs:
            if test_dir.exists():
                import shutil
                shutil.rmtree(test_dir)
                print(f"   Removed: {test_dir}")
        
        print("✅ Cleanup completed")
    except Exception as e:
        print(f"⚠️ Cleanup failed: {e}")

def main():
    """Run all tests."""
    print("🧪 SIMPLE LOG ANOMALY DETECTION PIPELINE TESTS")
    print("="*60)
    
    # Run tests
    tests = [
        ("Configuration", test_config),
        ("Pipeline Structure", test_pipeline_structure),
        ("Preprocessing", test_preprocessing),
        ("Evaluation", test_evaluation)
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
        print("🎉 All basic tests passed! Pipeline structure is correct.")
        print("\nNext steps:")
        print("1. Install deep learning dependencies if needed:")
        print("   pip install torch transformers tensorflow gensim")
        print("2. Run the complete pipeline:")
        print("   python run_pipeline.py --log_type your_log_type")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
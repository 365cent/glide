#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive Log Anomaly Detection Pipeline

This script orchestrates the entire pipeline from data preprocessing to model evaluation,
providing an efficient end-to-end solution for log anomaly detection.

Pipeline Stages:
1. Data Preprocessing - Convert raw logs to TFRecord format
2. Embedding Generation - Create FastText, Word2Vec, and LogBERT embeddings
3. Model Training - Train transformer models for each embedding type
4. Evaluation - Comprehensive evaluation with metrics and visualizations
5. Results Analysis - Generate comparison reports and dashboards

Features:
- Multi-embedding type support (FastText, Word2Vec, LogBERT)
- Automatic performance optimization based on dataset size
- Resume capability for interrupted processing
- Comprehensive evaluation metrics
- Visualization and reporting
- Memory-efficient processing for large datasets
"""

import subprocess
import argparse
import os
import sys
import time
from pathlib import Path
import json
import logging
from typing import List, Dict, Optional
import multiprocessing as mp

# Import configuration
from src.config import (
    LOGS_DIR, LABELS_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, 
    MODELS_DIR, RESULTS_DIR, CHECKPOINT_DIR
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PipelineExecutor:
    """Orchestrates the complete anomaly detection pipeline."""
    
    def __init__(self, log_type: str, embedding_types: List[str], 
                 optimize_thresholds: bool = False, force_restart: bool = False):
        self.log_type = log_type
        self.embedding_types = embedding_types
        self.optimize_thresholds = optimize_thresholds
        self.force_restart = force_restart
        self.base_dir = Path(__file__).parent
        self.src_dir = self.base_dir / "src"
        
        # Ensure all directories exist
        for dir_path in [LOGS_DIR, LABELS_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, 
                        MODELS_DIR, RESULTS_DIR, CHECKPOINT_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def run_command(self, command: str, stage: str, cwd: str = ".") -> bool:
        """Execute a command with proper logging and error handling."""
        logger.info(f"Starting {stage}...")
        logger.info(f"Command: {command}")
        
        start_time = time.time()
        try:
            process = subprocess.Popen(
                command, shell=True, cwd=cwd, 
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                text=True, bufsize=1, universal_newlines=True
            )
            
            # Stream output in real-time
            for line in process.stdout:
                print(line, end='')
            
            process.wait()
            elapsed_time = time.time() - start_time
            
            if process.returncode == 0:
                logger.info(f"✅ {stage} completed successfully in {elapsed_time:.2f}s")
                return True
            else:
                logger.error(f"❌ {stage} failed with exit code {process.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"❌ {stage} failed with exception: {e}")
            return False
    
    def check_data_availability(self) -> bool:
        """Check if required data is available for processing."""
        logger.info("Checking data availability...")
        
        # Check for log files (including .log.1, .log.2, etc.)
        log_files = []
        for file in LOGS_DIR.rglob("*"):
            if file.is_file() and ".log" in file.name:
                log_files.append(file)
        
        if not log_files:
            logger.warning("No log files found in logs directory")
            return False
        
        # Check for label files (they are .log files, not .json files)
        label_files = []
        for file in LABELS_DIR.rglob("*"):
            if file.is_file() and ".log" in file.name:
                label_files.append(file)
        
        if not label_files:
            logger.warning("No label files found in labels directory")
            return False
        
        logger.info(f"Found {len(log_files)} log files and {len(label_files)} label files")
        return True
    
    def stage_preprocessing(self) -> bool:
        """Stage 1: Data preprocessing and TFRecord generation."""
        logger.info("=" * 60)
        logger.info("STAGE 1: DATA PREPROCESSING")
        logger.info("=" * 60)
        
        # Check if preprocessing is already done
        processed_files = list(PROCESSED_DIR.rglob("*.tfrecord"))
        if processed_files and not self.force_restart:
            logger.info(f"Found {len(processed_files)} existing TFRecord files. Skipping preprocessing.")
            return True
        
        command = f"python3 {self.src_dir}/preprocessing.py"
        return self.run_command(command, "Data Preprocessing", str(self.base_dir))
    
    def stage_embedding_generation(self) -> bool:
        """Stage 2: Generate embeddings for all specified types."""
        logger.info("=" * 60)
        logger.info("STAGE 2: EMBEDDING GENERATION")
        logger.info("=" * 60)
        
        success_count = 0
        
        for embedding_type in self.embedding_types:
            logger.info(f"Generating {embedding_type} embeddings...")
            
            # Check if embeddings already exist
            embedding_dir = EMBEDDINGS_DIR / self.log_type
            embedding_files = [
                embedding_dir / f"log_{self.log_type}.pkl",
                embedding_dir / f"label_{self.log_type}.pkl",
                embedding_dir / f"attack_types_{self.log_type}.txt"
            ]
            
            if all(f.exists() for f in embedding_files) and not self.force_restart:
                logger.info(f"{embedding_type} embeddings already exist. Skipping.")
                success_count += 1
                continue
            
            # Generate embeddings
            if embedding_type == "fasttext":
                command = f"python3 {self.src_dir}/fasttext_embedding.py --log_type {self.log_type}"
            elif embedding_type == "word2vec":
                command = f"python3 {self.src_dir}/word2vec_embedding.py --log_type {self.log_type}"
            elif embedding_type == "logbert":
                command = f"python3 {self.src_dir}/logbert_embeddings.py --log_type {self.log_type}"
            else:
                logger.warning(f"Unknown embedding type: {embedding_type}")
                continue
            
            if self.run_command(command, f"{embedding_type} Embedding Generation", str(self.base_dir)):
                success_count += 1
        
        return success_count == len(self.embedding_types)
    
    def stage_model_training(self) -> bool:
        """Stage 3: Train transformer models for each embedding type."""
        logger.info("=" * 60)
        logger.info("STAGE 3: MODEL TRAINING")
        logger.info("=" * 60)
        
        success_count = 0
        
        for embedding_type in self.embedding_types:
            logger.info(f"Training transformer model for {embedding_type}...")
            
            # Check if model already exists
            model_dir = MODELS_DIR / self.log_type / embedding_type
            model_files = list(model_dir.glob("*.pt")) if model_dir.exists() else []
            
            if model_files and not self.force_restart:
                logger.info(f"{embedding_type} model already exists. Skipping training.")
                success_count += 1
                continue
            
            # Train model
            command = f"python3 {self.src_dir}/transformer.py --log_type {self.log_type} --embedding_type {embedding_type}"
            
            if self.run_command(command, f"{embedding_type} Model Training", str(self.base_dir)):
                success_count += 1
        
        return success_count == len(self.embedding_types)
    
    def stage_evaluation(self) -> bool:
        """Stage 4: Comprehensive model evaluation."""
        logger.info("=" * 60)
        logger.info("STAGE 4: MODEL EVALUATION")
        logger.info("=" * 60)
        
        # Build evaluation command
        eval_command = f"python3 {self.src_dir}/evaluate_models.py --log_type {self.log_type} --embedding_types {' '.join(self.embedding_types)}"
        
        if self.optimize_thresholds:
            eval_command += " --optimize_thresholds"
        
        return self.run_command(eval_command, "Model Evaluation", str(self.base_dir))
    
    def stage_results_analysis(self) -> bool:
        """Stage 5: Generate comprehensive results analysis."""
        logger.info("=" * 60)
        logger.info("STAGE 5: RESULTS ANALYSIS")
        logger.info("=" * 60)
        
        # Generate evaluation matrix
        matrix_command = f"python3 {self.src_dir}/generate_evaluation_matrix.py --log_type {self.log_type}"
        matrix_success = self.run_command(matrix_command, "Evaluation Matrix Generation", str(self.base_dir))
        
        # Generate results dashboard
        dashboard_command = f"python3 {self.src_dir}/results_dashboard.py --log_type {self.log_type}"
        dashboard_success = self.run_command(dashboard_command, "Results Dashboard Generation", str(self.base_dir))
        
        return matrix_success and dashboard_success
    
    def generate_pipeline_report(self) -> Dict:
        """Generate a comprehensive pipeline report."""
        report = {
            "pipeline_info": {
                "log_type": self.log_type,
                "embedding_types": self.embedding_types,
                "optimize_thresholds": self.optimize_thresholds,
                "force_restart": self.force_restart,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "data_statistics": {},
            "embedding_statistics": {},
            "model_statistics": {},
            "evaluation_results": {},
            "performance_metrics": {}
        }
        
        # Collect data statistics
        try:
            log_files = []
            for file in LOGS_DIR.rglob("*"):
                if file.is_file() and ".log" in file.name:
                    log_files.append(file)
            
            label_files = []
            for file in LABELS_DIR.rglob("*"):
                if file.is_file() and ".log" in file.name:
                    label_files.append(file)
            
            processed_files = list(PROCESSED_DIR.rglob("*.tfrecord"))
            
            report["data_statistics"] = {
                "log_files_count": len(log_files),
                "label_files_count": len(label_files),
                "processed_files_count": len(processed_files)
            }
        except Exception as e:
            logger.warning(f"Could not collect data statistics: {e}")
        
        # Collect embedding statistics
        for embedding_type in self.embedding_types:
            try:
                embedding_dir = EMBEDDINGS_DIR / self.log_type
                embedding_files = list(embedding_dir.glob(f"*{embedding_type}*.pkl"))
                
                report["embedding_statistics"][embedding_type] = {
                    "files_count": len(embedding_files),
                    "files": [f.name for f in embedding_files]
                }
            except Exception as e:
                logger.warning(f"Could not collect {embedding_type} statistics: {e}")
        
        # Collect model statistics
        for embedding_type in self.embedding_types:
            try:
                model_dir = MODELS_DIR / self.log_type / embedding_type
                model_files = list(model_dir.glob("*.pt")) if model_dir.exists() else []
                
                report["model_statistics"][embedding_type] = {
                    "models_count": len(model_files),
                    "models": [f.name for f in model_files]
                }
            except Exception as e:
                logger.warning(f"Could not collect {embedding_type} model statistics: {e}")
        
        # Save report
        report_file = RESULTS_DIR / self.log_type / "pipeline_report.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Pipeline report saved to: {report_file}")
        return report
    
    def run_pipeline(self) -> bool:
        """Execute the complete pipeline."""
        logger.info("🚀 Starting Comprehensive Log Anomaly Detection Pipeline")
        logger.info(f"Log Type: {self.log_type}")
        logger.info(f"Embedding Types: {', '.join(self.embedding_types)}")
        logger.info(f"Optimize Thresholds: {self.optimize_thresholds}")
        logger.info(f"Force Restart: {self.force_restart}")
        
        start_time = time.time()
        
        # Check data availability
        if not self.check_data_availability():
            logger.error("❌ Insufficient data for pipeline execution")
            return False
        
        # Execute pipeline stages
        stages = [
            ("Preprocessing", self.stage_preprocessing),
            ("Embedding Generation", self.stage_embedding_generation),
            ("Model Training", self.stage_model_training),
            ("Evaluation", self.stage_evaluation),
            ("Results Analysis", self.stage_results_analysis)
        ]
        
        successful_stages = 0
        total_stages = len(stages)
        
        for stage_name, stage_func in stages:
            logger.info(f"\n{'='*20} {stage_name.upper()} {'='*20}")
            
            if stage_func():
                successful_stages += 1
                logger.info(f"✅ {stage_name} completed successfully")
            else:
                logger.error(f"❌ {stage_name} failed")
                # Continue with other stages even if one fails
        
        # Generate final report
        self.generate_pipeline_report()
        
        # Pipeline completion summary
        elapsed_time = time.time() - start_time
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETION SUMMARY")
        logger.info("="*60)
        logger.info(f"Successful stages: {successful_stages}/{total_stages}")
        logger.info(f"Total execution time: {elapsed_time:.2f}s")
        
        if successful_stages == total_stages:
            logger.info("🎉 Pipeline completed successfully!")
            logger.info(f"Results available in: {RESULTS_DIR / self.log_type}")
        else:
            logger.warning(f"⚠️ Pipeline completed with {total_stages - successful_stages} failed stages")
        
        return successful_stages == total_stages

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Log Anomaly Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline for VPN logs
  python run_pipeline.py --log_type vpn

  # Run with specific embedding types
  python run_pipeline.py --log_type wp-error --embedding_types fasttext logbert

  # Run with threshold optimization
  python run_pipeline.py --log_type auth --optimize_thresholds

  # Force restart (ignore existing outputs)
  python run_pipeline.py --log_type vpn --force_restart
        """
    )
    
    parser.add_argument("--log_type", type=str, required=True,
                        help="Log type to process (e.g., 'vpn', 'wp-error', 'auth').")
    parser.add_argument("--embedding_types", nargs="+", 
                        default=["fasttext", "word2vec", "logbert"],
                        help="List of embedding types to use (default: all).")
    parser.add_argument("--optimize_thresholds", action="store_true",
                        help="Optimize per-class thresholds during evaluation.")
    parser.add_argument("--force_restart", action="store_true",
                        help="Force restart processing (ignore existing outputs).")
    parser.add_argument("--check_only", action="store_true",
                        help="Only check data availability without running pipeline.")
    
    args = parser.parse_args()
    
    # Validate embedding types
    valid_embedding_types = ["fasttext", "word2vec", "logbert"]
    for emb_type in args.embedding_types:
        if emb_type not in valid_embedding_types:
            logger.error(f"Invalid embedding type: {emb_type}")
            logger.error(f"Valid types: {', '.join(valid_embedding_types)}")
            return False
    
    # Create pipeline executor
    executor = PipelineExecutor(
        log_type=args.log_type,
        embedding_types=args.embedding_types,
        optimize_thresholds=args.optimize_thresholds,
        force_restart=args.force_restart
    )
    
    if args.check_only:
        # Only check data availability
        if executor.check_data_availability():
            logger.info("✅ Data availability check passed")
            return True
        else:
            logger.error("❌ Data availability check failed")
            return False
    else:
        # Run complete pipeline
        return executor.run_pipeline()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
import time
import logging
from pathlib import Path
from preprocessing import LogPreprocessor
from preprocessing_optimized import OptimizedLogPreprocessor
from src.config import LOGS_DIR, LABELS_DIR, PROCESSED_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PerformanceComparison")

def benchmark_preprocessor(preprocessor_class, name, logs_dir, labels_dir, output_dir):
    """Benchmark a preprocessor implementation."""
    logger.info(f"Starting benchmark for {name}")
    
    start_time = time.time()
    
    # Initialize preprocessor
    if preprocessor_class == OptimizedLogPreprocessor:
        preprocessor = preprocessor_class(logs_dir, labels_dir, output_dir, max_workers=4)
    else:
        preprocessor = preprocessor_class(logs_dir, labels_dir, output_dir)
    
    init_time = time.time()
    
    # Run batch processing
    preprocessor.batch_process()
    
    end_time = time.time()
    
    init_duration = init_time - start_time
    processing_duration = end_time - init_time
    total_duration = end_time - start_time
    
    logger.info(f"{name} Results:")
    logger.info(f"  Initialization time: {init_duration:.4f} seconds")
    logger.info(f"  Processing time: {processing_duration:.4f} seconds")
    logger.info(f"  Total time: {total_duration:.4f} seconds")
    
    return {
        'name': name,
        'init_time': init_duration,
        'processing_time': processing_duration,
        'total_time': total_duration
    }

def compare_path_matching_performance():
    """Compare path matching performance between implementations."""
    logger.info("Comparing path matching performance...")
    
    # Test files for path matching
    test_files = [
        Path("logs/user1/openvpn.log"),
        Path("logs/user2/access.log"),
        Path("logs/user3/error.log"),
        Path("logs/user4/intranet_server/logs/error.log"),
        Path("logs/user5/auth.log"),
        Path("logs/user6/audit.log"),
        Path("logs/user7/dnsmasq.log"),
        Path("logs/user8/internal_share/logs/audit/audit.log"),
        Path("logs/user9/system.cpu.log"),
        Path("logs/user10/dummy_log.log"),
        Path("logs/user11/unknown.log"),
    ]
    
    # Initialize preprocessors
    original_preprocessor = LogPreprocessor()
    optimized_preprocessor = OptimizedLogPreprocessor()
    
    # Benchmark path matching
    original_times = []
    optimized_times = []
    
    for test_file in test_files:
        # Test original implementation
        start_time = time.time()
        for _ in range(1000):  # Run 1000 times for accurate measurement
            original_preprocessor.determine_log_type(test_file)
        original_times.append(time.time() - start_time)
        
        # Test optimized implementation
        start_time = time.time()
        for _ in range(1000):  # Run 1000 times for accurate measurement
            optimized_preprocessor.determine_log_type(test_file)
        optimized_times.append(time.time() - start_time)
    
    logger.info("Path Matching Performance Results:")
    logger.info(f"  Original implementation average: {sum(original_times)/len(original_times):.6f} seconds per 1000 calls")
    logger.info(f"  Optimized implementation average: {sum(optimized_times)/len(optimized_times):.6f} seconds per 1000 calls")
    
    speedup = sum(original_times) / sum(optimized_times)
    logger.info(f"  Speedup: {speedup:.2f}x faster")

def main():
    """Run performance comparison between original and optimized implementations."""
    logger.info("Starting performance comparison...")
    
    # Create test directories
    test_logs_dir = Path("test_logs")
    test_labels_dir = Path("test_labels")
    test_output_dir = Path("test_output")
    
    # Clean up previous test results
    if test_output_dir.exists():
        import shutil
        shutil.rmtree(test_output_dir)
    
    # Run benchmarks
    results = []
    
    # Benchmark original implementation
    original_result = benchmark_preprocessor(
        LogPreprocessor, 
        "Original Implementation", 
        test_logs_dir, 
        test_labels_dir, 
        test_output_dir / "original"
    )
    results.append(original_result)
    
    # Benchmark optimized implementation
    optimized_result = benchmark_preprocessor(
        OptimizedLogPreprocessor, 
        "Optimized Implementation", 
        test_logs_dir, 
        test_labels_dir, 
        test_output_dir / "optimized"
    )
    results.append(optimized_result)
    
    # Compare results
    logger.info("\n" + "="*50)
    logger.info("PERFORMANCE COMPARISON SUMMARY")
    logger.info("="*50)
    
    for result in results:
        logger.info(f"\n{result['name']}:")
        logger.info(f"  Initialization: {result['init_time']:.4f}s")
        logger.info(f"  Processing: {result['processing_time']:.4f}s")
        logger.info(f"  Total: {result['total_time']:.4f}s")
    
    # Calculate speedup
    original_total = results[0]['total_time']
    optimized_total = results[1]['total_time']
    speedup = original_total / optimized_total
    
    logger.info(f"\nOverall Speedup: {speedup:.2f}x faster")
    
    # Compare path matching performance
    compare_path_matching_performance()
    
    logger.info("\nPerformance comparison complete!")

if __name__ == '__main__':
    main()
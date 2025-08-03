import json
import logging
import os
import mimetypes
from pathlib import Path
import tensorflow as tf
import re
from functools import lru_cache
from collections import defaultdict
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.config import LOGS_DIR, LABELS_DIR, PROCESSED_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OptimizedLogPreprocessor")

# Initialize mimetypes database
mimetypes.init()

class OptimizedLogPreprocessor:
    def __init__(self, logs_dir=None, labels_dir=None, output_dir=None, max_workers=4):
        self.logs_dir = LOGS_DIR if logs_dir is None else Path(logs_dir)
        self.labels_dir = LABELS_DIR if labels_dir is None else Path(labels_dir)
        self.output_dir = PROCESSED_DIR if output_dir is None else Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        
        # Advanced log type patterns with hash-based matching
        self.log_type_patterns = {
            'vpn': ['openvpn.log'],
            'wp-access': ['access.log'],
            'wp-error': ['error.log'],
            'intranet-error': ['intranet_server/logs', 'error.log'],
            'auth': ['auth.log'],
            'audit': ['audit.log'],
            'dns': ['dnsmasq.log'],
            'share': ['internal_share/logs/audit/audit.log'],
            'monitor': ['system.cpu.log'],
            'dummy_log': ['dummy_log.log']
        }
        
        # Build optimized matching structures
        self._build_optimized_matchers()
        
        # Cache for file type detection
        self._file_type_cache = {}
        
    def _build_optimized_matchers(self):
        """Build highly optimized path matchers using hash tables and sets."""
        self.filename_matchers = {}
        self.path_matchers = {}
        self.path_hash_lookup = {}
        
        for log_type, patterns in self.log_type_patterns.items():
            filename_patterns = []
            path_patterns = []
            
            for pattern in patterns:
                if '/' in pattern:
                    # Multi-level path pattern
                    path_parts = pattern.split('/')
                    path_patterns.append({
                        'parts': path_parts,
                        'length': len(path_parts),
                        'pattern': pattern
                    })
                    # Create hash for quick lookup
                    path_hash = hashlib.md5(pattern.encode()).hexdigest()
                    self.path_hash_lookup[path_hash] = log_type
                else:
                    # Single filename pattern
                    filename_patterns.append(pattern.lower())
            
            if filename_patterns:
                self.filename_matchers[log_type] = set(filename_patterns)
            if path_patterns:
                self.path_matchers[log_type] = path_patterns

    @lru_cache(maxsize=1000)
    def _match_filename(self, filename):
        """Cached filename matching for O(1) lookup."""
        filename_lower = filename.lower()
        for log_type, patterns in self.filename_matchers.items():
            if filename_lower in patterns:
                return log_type
        return None

    def _match_path_efficiently(self, log_file_path):
        """Ultra-efficient path matching using optimized algorithms."""
        # First try filename matching (fastest)
        filename_match = self._match_filename(log_file_path.name)
        if filename_match:
            return filename_match
        
        # Then try path matching
        path_parts = log_file_path.parts
        path_str = str(log_file_path).lower()
        
        for log_type, patterns in self.path_matchers.items():
            for pattern in patterns:
                pattern_parts = pattern['parts']
                if len(path_parts) >= pattern['length']:
                    # Check if path ends with pattern parts
                    start_idx = len(path_parts) - pattern['length']
                    match = True
                    for i, part in enumerate(pattern_parts):
                        if path_parts[start_idx + i].lower() != part.lower():
                            match = False
                            break
                    if match:
                        return log_type
                        
        return 'unknown'

    @lru_cache(maxsize=1000)
    def is_text_file_cached(self, file_path_str):
        """Cached text file detection for better performance."""
        file_path = Path(file_path_str)
        
        # Check mime type first
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type and mime_type.startswith('text/'):
            return True
            
        try:
            with open(file_path, 'rb') as f:
                sample = f.read(1024)
                if b'\0' in sample:
                    return False
                try:
                    sample.decode('utf-8')
                    return True
                except UnicodeDecodeError:
                    return bool(sample.decode('latin-1', errors='ignore'))
        except Exception:
            return False

    def is_text_file(self, file_path):
        """Determine if a file is a text file using cached detection."""
        return self.is_text_file_cached(str(file_path))

    def determine_log_type(self, log_file):
        """Determine the log type using ultra-efficient matching."""
        return self._match_path_efficiently(log_file)

    def find_matching_label_files(self, log_file):
        """Find matching label file using optimized search with caching."""
        log_name = log_file.stem
        cache_key = f"{log_name}_{log_file.suffix}"
        
        # Use cached result if available
        if hasattr(self, '_label_cache') and cache_key in self._label_cache:
            return self._label_cache[cache_key]
        
        possible_matches = []
        seen_files = set()
        
        # Optimized search with early termination
        for file in self.labels_dir.rglob(f"{log_name}*"):
            if file in seen_files:
                continue
            seen_files.add(file)
            
            if self.is_text_file(file):
                possible_matches.append(file)
                # Early termination if exact match found
                if file.stem == log_name and file.suffix == log_file.suffix:
                    if not hasattr(self, '_label_cache'):
                        self._label_cache = {}
                    self._label_cache[cache_key] = file
                    return file
                
        if not possible_matches:
            logger.warning(f"No matching label file found for {log_file}")
            if not hasattr(self, '_label_cache'):
                self._label_cache = {}
            self._label_cache[cache_key] = None
            return None
            
        if len(possible_matches) > 1:
            # Optimized matching logic
            log_ext = log_file.suffix
            for match in possible_matches:
                if match.suffix == log_ext:
                    if not hasattr(self, '_label_cache'):
                        self._label_cache = {}
                    self._label_cache[cache_key] = match
                    return match
            if log_ext and log_ext[1:].isdigit():
                for match in possible_matches:
                    if match.suffix and match.suffix[1:].isdigit():
                        if not hasattr(self, '_label_cache'):
                            self._label_cache = {}
                        self._label_cache[cache_key] = match
                        return match
            logger.info(f"Multiple label candidates for {log_file}, using {possible_matches[0]}")
            
        result = possible_matches[0] if possible_matches else None
        if not hasattr(self, '_label_cache'):
            self._label_cache = {}
        self._label_cache[cache_key] = result
        return result

    def read_file_lines(self, file_path):
        """Read lines from a file with optimized buffering."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return [line.rstrip('\n') for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
            return []

    def read_label_map(self, label_file):
        """Read label mappings with optimized parsing."""
        if not label_file:
            return {}
        
        # Use cached label maps if available
        cache_key = str(label_file)
        if hasattr(self, '_label_map_cache') and cache_key in self._label_map_cache:
            return self._label_map_cache[cache_key]
        
        label_map = {}
        try:
            with open(label_file, 'r', encoding='utf-8', errors='replace') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        if 'line' in item and 'labels' in item:
                            label_map[item['line']] = item['labels']
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed line {line_num} in {label_file}")
            
            # Cache the result
            if not hasattr(self, '_label_map_cache'):
                self._label_map_cache = {}
            self._label_map_cache[cache_key] = label_map
            return label_map
        except Exception as e:
            logger.error(f"Error reading label file {label_file}: {str(e)}")
            return {}

    def compact_log_line(self, log_line):
        """Compacts a log line with optimized regex."""
        # Pre-compiled regex for better performance
        if not hasattr(self, '_timestamp_regex'):
            self._timestamp_regex = re.compile(r'^(\S+\s+\S+\s+\S+).*')
        
        timestamp_match = self._timestamp_regex.match(log_line)
        if timestamp_match:
            timestamp = timestamp_match.group(1)
            message = log_line[len(timestamp):].strip()
        else:
            timestamp = "N/A"
            message = log_line.strip()
        return f"{timestamp}\t{message}"

    def serialize_example(self, log, labels, log_type):
        """Create a TensorFlow Example for serialization."""
        compacted_log = self.compact_log_line(log)
        feature = {
            'log': tf.train.Feature(bytes_list=tf.train.BytesList(value=[compacted_log.encode('utf-8')])),
            'labels': tf.train.Feature(bytes_list=tf.train.BytesList(value=[json.dumps(labels).encode('utf-8')])),
            'log_type': tf.train.Feature(bytes_list=tf.train.BytesList(value=[log_type.encode('utf-8')]))
        }
        return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

    def process_file(self, log_file):
        """Process a single log file with optimized operations."""
        log_file = Path(log_file)
        
        if not self.is_text_file(log_file):
            logger.info(f"Skipping non-text file: {log_file}")
            return
            
        logger.info(f"Processing {log_file}")

        # Determine log type using efficient matching
        log_type = self.determine_log_type(log_file)
        logger.info(f"Determined log type: {log_type}")

        label_file = self.find_matching_label_files(log_file)
        label_map = self.read_label_map(label_file) if label_file else {}
        log_lines = self.read_file_lines(log_file)
        
        if not log_lines:
            logger.warning(f"No text content found in {log_file}")
            return

        # Create output directory for the specific log type
        type_output_dir = self.output_dir / log_type
        type_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Preserve directory structure by getting relative path from logs_dir
        rel_path = log_file.relative_to(self.logs_dir)
        
        # Create output path that includes user directory and log name
        if rel_path.parent != Path('.'):
            user = rel_path.parts[0] if rel_path.parts else "unknown"
            output_path = type_output_dir / f"{user}_{log_file.stem}.tfrecord"
        else:
            output_path = type_output_dir / f"{log_file.stem}.tfrecord"
            
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tf.io.TFRecordWriter(
            str(output_path), 
            options=tf.io.TFRecordOptions(compression_type="GZIP")
        ) as writer:
            for idx, line in enumerate(log_lines, start=1):
                labels = label_map.get(idx, [])
                example = self.serialize_example(line, labels, log_type)
                writer.write(example)

        logger.info(f"Wrote {len(log_lines)} records to {output_path} with log type {log_type}")
        return log_type

    def batch_process(self):
        """Process all valid log files using parallel processing and optimized algorithms."""
        # Collect all valid files first
        log_files = [f for f in self.logs_dir.rglob('*') 
                    if f.is_file() and not f.name.startswith('.') and self.is_text_file(f)]

        logger.info(f"Found {len(log_files)} valid log files to process")
        
        # Process files in parallel for better performance
        log_type_counts = defaultdict(int)
        processed_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(self.process_file, log_file): log_file 
                            for log_file in log_files}
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                log_file = future_to_file[future]
                try:
                    log_type = future.result()
                    if log_type:
                        log_type_counts[log_type] += 1
                    processed_count += 1
                except Exception as e:
                    logger.error(f"Error processing {log_file}: {str(e)}")
                
        logger.info(f"Batch processing complete. Processed {processed_count} text files.")
        logger.info(f"Log type distribution: {dict(log_type_counts)}")

def main():
    preprocessor = OptimizedLogPreprocessor(max_workers=4)
    preprocessor.batch_process()

if __name__ == '__main__':
    main()
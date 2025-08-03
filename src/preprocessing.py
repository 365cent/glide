import json
import logging
import os
import mimetypes
from pathlib import Path
import tensorflow as tf
import re
from src.config import LOGS_DIR, LABELS_DIR, PROCESSED_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LogPreprocessor")

# Initialize mimetypes database
mimetypes.init()

class LogPreprocessor:
    def __init__(self, logs_dir=None, labels_dir=None, output_dir=None):
        self.logs_dir = LOGS_DIR if logs_dir is None else Path(logs_dir)
        self.labels_dir = LABELS_DIR if labels_dir is None else Path(labels_dir)
        self.output_dir = PROCESSED_DIR if output_dir is None else Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimized log type patterns using path-based matching
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
        
        # Pre-compile path components for efficient matching
        self._build_path_matchers()
        
    def _build_path_matchers(self):
        """Build efficient path matchers using sets and optimized lookups."""
        self.path_matchers = {}
        
        for log_type, patterns in self.log_type_patterns.items():
            matchers = []
            for pattern in patterns:
                if '/' in pattern:
                    # Multi-level path pattern
                    path_parts = pattern.split('/')
                    matchers.append({
                        'type': 'path',
                        'parts': path_parts,
                        'pattern': pattern
                    })
                else:
                    # Single filename pattern
                    matchers.append({
                        'type': 'filename',
                        'filename': pattern,
                        'pattern': pattern
                    })
            self.path_matchers[log_type] = matchers

    def _match_path_efficiently(self, log_file_path):
        """Efficiently match log file path against patterns using optimized lookups."""
        path_str = str(log_file_path).lower()
        path_parts = log_file_path.parts
        
        for log_type, matchers in self.path_matchers.items():
            for matcher in matchers:
                if matcher['type'] == 'filename':
                    # Direct filename match
                    if log_file_path.name.lower() == matcher['filename'].lower():
                        return log_type
                elif matcher['type'] == 'path':
                    # Multi-level path match
                    pattern_parts = matcher['parts']
                    if len(path_parts) >= len(pattern_parts):
                        # Check if path ends with pattern parts
                        for i in range(len(pattern_parts)):
                            if path_parts[-(len(pattern_parts) - i)].lower() != pattern_parts[i].lower():
                                break
                        else:
                            return log_type
                            
        return 'unknown'

    def is_text_file(self, file_path):
        """Determine if a file is a text file efficiently."""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type and mime_type.startswith('text/'):
            return True
            
        try:
            with open(file_path, 'rb') as f:
                sample = f.read(1024)  # Sample first 1KB
                if b'\0' in sample:
                    return False
                try:
                    sample.decode('utf-8')
                    return True
                except UnicodeDecodeError:
                    return bool(sample.decode('latin-1', errors='ignore'))
        except Exception:
            return False

    def determine_log_type(self, log_file):
        """Determine the log type using efficient path-based matching."""
        return self._match_path_efficiently(log_file)

    def find_matching_label_files(self, log_file):
        """Find matching label file for a log file using optimized search."""
        log_name = log_file.stem
        possible_matches = []
        
        # Use set for O(1) lookup performance
        seen_files = set()
        
        for file in self.labels_dir.rglob(f"{log_name}*"):
            if file in seen_files:
                continue
            seen_files.add(file)
            
            if self.is_text_file(file):
                possible_matches.append(file)
                
        if not possible_matches:
            logger.warning(f"No matching label file found for {log_file}")
            return None
            
        if len(possible_matches) > 1:
            # Optimize matching logic
            log_ext = log_file.suffix
            for match in possible_matches:
                if match.suffix == log_ext:
                    return match
            if log_ext and log_ext[1:].isdigit():
                for match in possible_matches:
                    if match.suffix and match.suffix[1:].isdigit():
                        return match
            logger.info(f"Multiple label candidates for {log_file}, using {possible_matches[0]}")
            
        return possible_matches[0]

    def read_file_lines(self, file_path):
        """Read lines from a file, handling encoding issues gracefully."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return [line.rstrip('\n') for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
            return []

    def read_label_map(self, label_file):
        """Read label mappings from a label file using optimized parsing."""
        if not label_file:
            return {}
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
            return label_map
        except Exception as e:
            logger.error(f"Error reading label file {label_file}: {str(e)}")
            return {}

    def compact_log_line(self, log_line):
        """Compacts a log line into a two-column format: [timestamp] [log message]."""
        # Optimized timestamp extraction
        timestamp_match = re.match(r'^(\S+\s+\S+\s+\S+).*', log_line)
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
        """Process a log file with its matching label file."""
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

    def batch_process(self):
        """Process all valid log files in the logs directory using optimized batch processing."""
        # Use generator for memory efficiency
        log_files = (f for f in self.logs_dir.rglob('*') 
                    if f.is_file() and not f.name.startswith('.'))

        logger.info("Starting batch processing...")
        processed_count = 0
        log_type_counts = {}
        
        for log_file in log_files:
            if self.is_text_file(log_file):
                self.process_file(log_file)
                processed_count += 1
                
                log_type = self.determine_log_type(log_file)
                log_type_counts[log_type] = log_type_counts.get(log_type, 0) + 1
                
        logger.info(f"Batch processing complete. Processed {processed_count} text files.")
        logger.info(f"Log type distribution: {log_type_counts}")

def main():
    preprocessor = LogPreprocessor()
    preprocessor.batch_process()

if __name__ == '__main__':
    main()
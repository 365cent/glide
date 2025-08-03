
import tensorflow as tf
import json
from pathlib import Path
import logging
from src.config import PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format=\'%(asctime)s - %(levelname)s - %(message)s\')
logger = logging.getLogger("DataValidator")

def validate_tfrecord_file(filepath):
    """Validates the structure and content of a single TFRecord file."""
    logger.info(f"Validating {filepath}...")
    try:
        raw_dataset = tf.data.TFRecordDataset(str(filepath), compression_type=\"GZIP\")
        
        # Define how to parse the example
        feature_description = {
            \"log\": tf.io.FixedLenFeature([], tf.string),
            \"labels\": tf.io.FixedLenFeature([], tf.string),
            \"log_type\": tf.io.FixedLenFeature([], tf.string),
        }

        def _parse_function(example_proto):
            return tf.io.parse_single_example(example_proto, feature_description)

        parsed_dataset = raw_dataset.map(_parse_function)

        for i, parsed_record in enumerate(parsed_dataset.take(5)):  # Check first 5 records
            log_content = parsed_record[\"log\"].numpy().decode(\"utf-8\")
            labels_json = parsed_record[\"labels\"].numpy().decode(\"utf-8\")
            log_type = parsed_record[\"log_type\"].numpy().decode(\"utf-8\")

            # Basic checks
            assert isinstance(log_content, str) and len(log_content) > 0, f\"Record {i}: Log content is invalid.\"
            assert isinstance(labels_json, str), f\"Record {i}: Labels are not a string.\"
            try:
                labels = json.loads(labels_json)
                assert isinstance(labels, list), f\"Record {i}: Decoded labels are not a list.\"
            except json.JSONDecodeError:
                raise ValueError(f\"Record {i}: Labels JSON is malformed.\")
            assert isinstance(log_type, str) and len(log_type) > 0, f\"Record {i}: Log type is invalid.\"
            
            logger.debug(f\"Record {i} - Log: {log_content[:50]}..., Labels: {labels}, Log Type: {log_type}\")

        logger.info(f"Validation successful for {filepath}. First {i+1} records checked.")
        return True
    except Exception as e:
        logger.error(f"Validation failed for {filepath}: {e}")
        return False

def validate_all_tfrecords(base_dir=PROCESSED_DIR):
    """Validates all TFRecord files found in the processed directory."""
    logger.info(f"Starting validation of all TFRecord files in {base_dir}")
    tfrecord_files = list(base_dir.rglob(\"*.tfrecord\"))
    
    if not tfrecord_files:
        logger.warning(f"No TFRecord files found in {base_dir}. Please run preprocessing first.")
        return False

    all_valid = True
    for tf_file in tfrecord_files:
        if not validate_tfrecord_file(tf_file):
            all_valid = False
            
    if all_valid:
        logger.info("All TFRecord files validated successfully.")
    else:
        logger.error("Some TFRecord files failed validation.")
        
    return all_valid

if __name__ == \"__main__\":
    # Example usage: Ensure you have some TFRecord files in PROCESSED_DIR
    # For testing, you might want to run preprocessing.py first
    validate_all_tfrecords()



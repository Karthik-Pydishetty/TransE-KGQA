#!/usr/bin/env python3
"""
Configuration settings for the LC-QuAD preprocessing pipeline.
"""

import os
from typing import Dict, Any

class Config:
    """Configuration class for the preprocessing pipeline."""
    
    # Data paths
    DATA_DIR = "data"
    LC_QUAD_DIR = os.path.join(DATA_DIR, "LC-QuAD")
    TRAIN_DATA_PATH = os.path.join(LC_QUAD_DIR, "train-data.json")
    TEST_DATA_PATH = os.path.join(LC_QUAD_DIR, "test-data.json")
    OUTPUT_DIR = os.path.join(DATA_DIR, "processed")
    
    # DBpedia endpoints (alternatives for reliability)
    DBPEDIA_ENDPOINTS = [
        "https://dbpedia.org/sparql",          # Main endpoint (HTTPS)
        "http://dbpedia.org/sparql",           # HTTP fallback
        "https://linked.opendata.cz/sparql",   # Alternative endpoint
    ]
    
    # Processing settings
    DEFAULT_CONFIG = {
        'request_delay': 0.3,      # Delay between requests (seconds)
        'batch_size': 20,          # Number of queries per batch
        'max_retries': 3,          # Maximum retry attempts for failed queries
        'timeout': 30,             # Request timeout in seconds
        'ssl_verify': False,       # SSL verification (set to False for macOS issues)
    }
    
    # Test configuration
    TEST_CONFIG = {
        'request_delay': 0.5,      # Slower for testing
        'batch_size': 1,           # Process one at a time for testing
        'max_retries': 1,          # Fewer retries for testing
        'timeout': 15,             # Shorter timeout for testing
        'ssl_verify': False,       # Disable SSL verification
        'max_test_queries': 5,     # Limit queries for testing
    }
    
    # Output files
    OUTPUT_FILES = {
        'triples': 'knowledge_graph_triples.txt',
        'train_triples': 'train_triples.txt',
        'test_triples': 'test_triples.txt',
        'valid_triples': 'valid_triples.txt',
        'statistics': 'preprocessing_stats.json',
        'failed_queries': 'failed_queries.json',
        'entity_mapping': 'entity2id.json',
        'relation_mapping': 'relation2id.json',
    }
    
    @classmethod
    def get_config(cls, mode: str = "default") -> Dict[str, Any]:
        """Get configuration for specified mode."""
        if mode == "test":
            return cls.TEST_CONFIG.copy()
        else:
            return cls.DEFAULT_CONFIG.copy()
    
    @classmethod
    def get_output_path(cls, filename: str) -> str:
        """Get full output path for a file."""
        return os.path.join(cls.OUTPUT_DIR, filename)
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        
        # Create subdirectories for different splits
        os.makedirs(os.path.join(cls.OUTPUT_DIR, 'splits'), exist_ok=True)
        os.makedirs(os.path.join(cls.OUTPUT_DIR, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(cls.OUTPUT_DIR, 'stats'), exist_ok=True)

# Environment-specific settings
DEVELOPMENT_MODE = os.getenv('DEV_MODE', 'true').lower() == 'true'
VERBOSE_LOGGING = os.getenv('VERBOSE', 'false').lower() == 'true'

# Create a default configuration instance
config = Config() 
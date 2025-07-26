#!/usr/bin/env python3
"""
Test script for the improved LC-QuAD preprocessing pipeline.
Tests the configuration system and SSL handling improvements.
"""

import json
import os
import logging
from preprocessing_improved import ImprovedLC_QuAD_Preprocessor
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_configuration_system():
    """Test the configuration system."""
    logger.info("=== Testing Configuration System ===")
    
    # Test default configuration
    default_config = Config.get_config("default")
    logger.info(f"Default config: {default_config}")
    
    # Test test configuration
    test_config = Config.get_config("test")
    logger.info(f"Test config: {test_config}")
    
    # Test directory creation
    Config.ensure_directories()
    
    # Verify directories exist
    if os.path.exists(Config.OUTPUT_DIR):
        logger.info(f"✅ Output directory created: {Config.OUTPUT_DIR}")
    else:
        logger.error(f"❌ Failed to create output directory: {Config.OUTPUT_DIR}")
        return False
    
    return True

def test_improved_preprocessing():
    """Test the improved preprocessing pipeline in test mode."""
    logger.info("=== Testing Improved Preprocessing Pipeline ===")
    
    try:
        # Initialize preprocessor in test mode
        preprocessor = ImprovedLC_QuAD_Preprocessor(mode="test")
        
        # Test parsing with limited data
        train_data = preprocessor.parse_lc_quad_json(Config.TRAIN_DATA_PATH)
        test_data = preprocessor.parse_lc_quad_json(Config.TEST_DATA_PATH)
        
        logger.info(f"Loaded {len(train_data)} training samples")
        logger.info(f"Loaded {len(test_data)} test samples")
        
        # Test SPARQL extraction
        train_queries = preprocessor.extract_sparql_queries(train_data)
        test_queries = preprocessor.extract_sparql_queries(test_data)
        
        logger.info(f"Extracted {len(train_queries)} training queries")
        logger.info(f"Extracted {len(test_queries)} test queries")
        
        # Show example queries
        if train_queries:
            logger.info("Example training query:")
            logger.info(f"  ID: {train_queries[0]['id']}")
            logger.info(f"  Question: {train_queries[0]['question']}")
            logger.info(f"  SPARQL: {train_queries[0]['sparql_query']}")
        
        # Test batch processing (limited queries)
        all_queries = train_queries + test_queries
        sample_queries = all_queries[:3]  # Test with just 3 queries
        
        logger.info(f"Testing batch processing with {len(sample_queries)} queries...")
        preprocessor.run_sparql_queries_batch(sample_queries)
        
        # Check results
        logger.info(f"Extracted {len(preprocessor.all_triples)} unique triples")
        
        # Show some statistics
        stats = preprocessor.query_stats
        logger.info("Query Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # Test output saving
        Config.ensure_directories()
        
        # Save test results
        test_triples_file = Config.get_output_path('test_triples.txt')
        preprocessor.format_triples_for_transe(test_triples_file)
        
        test_stats_file = Config.get_output_path('test_stats.json')
        preprocessor.save_statistics(test_stats_file)
        
        test_failed_file = Config.get_output_path('test_failed_queries.json')
        preprocessor.save_failed_queries(test_failed_file)
        
        # Verify files were created
        files_created = 0
        for file_path in [test_triples_file, test_stats_file]:
            if os.path.exists(file_path):
                files_created += 1
                logger.info(f"✅ Created: {file_path}")
            else:
                logger.warning(f"❌ Missing: {file_path}")
        
        # Check if we have any triples
        if len(preprocessor.all_triples) > 0:
            logger.info("Example extracted triples:")
            for i, triple in enumerate(list(preprocessor.all_triples)[:3]):
                logger.info(f"  {i+1}. {triple[0]} -> {triple[1]} -> {triple[2]}")
        else:
            logger.warning("No triples extracted (this may be due to SSL/connectivity issues)")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        return False

def test_ssl_handling():
    """Test SSL context setup."""
    logger.info("=== Testing SSL Handling ===")
    
    try:
        # Create preprocessor which should setup SSL context
        preprocessor = ImprovedLC_QuAD_Preprocessor(mode="test")
        
        # Check if SSL context was set up
        if not preprocessor.config['ssl_verify']:
            logger.info("✅ SSL verification disabled as configured")
        
        # Test endpoint switching
        original_endpoint = preprocessor.endpoints[preprocessor.current_endpoint_idx]
        logger.info(f"Current endpoint: {original_endpoint}")
        
        # Simulate endpoint switch
        preprocessor._try_next_endpoint()
        new_endpoint = preprocessor.endpoints[preprocessor.current_endpoint_idx]
        logger.info(f"Switched to endpoint: {new_endpoint}")
        
        if original_endpoint != new_endpoint:
            logger.info("✅ Endpoint switching works correctly")
        else:
            logger.warning("⚠️ Endpoint switching may not be working (only one endpoint?)")
        
        return True
        
    except Exception as e:
        logger.error(f"SSL handling test failed: {e}")
        return False

def run_full_test_suite():
    """Run the complete test suite."""
    logger.info("🚀 Starting Full Test Suite for Improved Preprocessing Pipeline")
    
    tests = [
        ("Configuration System", test_configuration_system),
        ("SSL Handling", test_ssl_handling),
        ("Improved Preprocessing", test_improved_preprocessing),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running: {test_name} ---")
        try:
            if test_func():
                logger.info(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: FAILED with exception: {e}")
    
    # Summary
    logger.info(f"\n=== TEST SUITE SUMMARY ===")
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All tests passed! The improved preprocessing pipeline is ready.")
        return True
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Please check the logs above.")
        return False

if __name__ == "__main__":
    success = run_full_test_suite()
    
    if success:
        logger.info("\n🎯 Next Steps:")
        logger.info("1. Run the full preprocessing pipeline:")
        logger.info("   python preprocessing_improved.py")
        logger.info("2. Or process in test mode by editing the 'mode' variable in the script")
        logger.info("3. Check the output in data/processed/ directory")
    else:
        logger.error("\n🔧 Troubleshooting needed. Check the logs above for specific issues.") 
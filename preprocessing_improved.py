#!/usr/bin/env python3
"""
Improved LC-QuAD preprocessing pipeline with better error handling and SSL support.
"""

import json
import logging
import time
import ssl
import urllib.request
from typing import List, Dict, Set, Tuple, Optional
from urllib.parse import quote
import requests
from SPARQLWrapper import SPARQLWrapper, JSON, POST, GET
import re
import os
from collections import defaultdict, Counter

from config import Config, config

# Ensure directories exist before setting up logging
Config.ensure_directories()

# Configure logging
log_handlers = [logging.StreamHandler()]
try:
    log_handlers.append(logging.FileHandler(config.get_output_path('preprocessing.log')))
except Exception:
    # If file handler fails, just use console logging
    pass

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=log_handlers
)
logger = logging.getLogger(__name__)

class ImprovedLC_QuAD_Preprocessor:
    """Improved data preprocessing pipeline for LC-QuAD dataset with better error handling."""
    
    def __init__(self, mode: str = "default"):
        """
        Initialize the preprocessor.
        
        Args:
            mode: Configuration mode ("default" or "test")
        """
        self.mode = mode
        self.config = Config.get_config(mode)
        self.endpoints = Config.DBPEDIA_ENDPOINTS.copy()
        self.current_endpoint_idx = 0
        
        # Setup SPARQL wrapper
        self.sparql = None
        self._setup_sparql_wrapper()
        
        # Storage for extracted data
        self.all_triples = set()
        self.failed_queries = []
        self.query_stats = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'empty_results': 0,
            'ssl_errors': 0,
            'timeout_errors': 0,
            'retries_used': 0
        }
        
        # SSL context setup for macOS
        if not self.config['ssl_verify']:
            self._setup_ssl_context()

    def _setup_ssl_context(self):
        """Setup SSL context to handle certificate verification issues."""
        try:
            # Create SSL context that doesn't verify certificates
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Install SSL handler
            https_handler = urllib.request.HTTPSHandler(context=ssl_context)
            opener = urllib.request.build_opener(https_handler)
            urllib.request.install_opener(opener)
            
            logger.info("SSL verification disabled for macOS compatibility")
        except Exception as e:
            logger.warning(f"Could not setup SSL context: {e}")

    def _setup_sparql_wrapper(self):
        """Setup SPARQL wrapper with current endpoint."""
        endpoint = self.endpoints[self.current_endpoint_idx]
        logger.info(f"Setting up SPARQL endpoint: {endpoint}")
        
        self.sparql = SPARQLWrapper(endpoint)
        self.sparql.setReturnFormat(JSON)
        self.sparql.setMethod(GET)
        
        # Set timeout
        if hasattr(self.sparql, 'setTimeout'):
            self.sparql.setTimeout(self.config['timeout'])

    def _try_next_endpoint(self):
        """Switch to next available endpoint."""
        self.current_endpoint_idx = (self.current_endpoint_idx + 1) % len(self.endpoints)
        self._setup_sparql_wrapper()
        logger.info(f"Switched to endpoint: {self.endpoints[self.current_endpoint_idx]}")

    def parse_lc_quad_json(self, file_path: str) -> List[Dict]:
        """Parse LC-QuAD JSON file and extract question-SPARQL pairs."""
        logger.info(f"Parsing LC-QuAD file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Limit queries in test mode
        if self.mode == "test" and 'max_test_queries' in self.config:
            data = data[:self.config['max_test_queries']]
            logger.info(f"Limited to {len(data)} queries for testing")
        
        logger.info(f"Loaded {len(data)} questions from {file_path}")
        return data

    def extract_sparql_queries(self, lc_quad_data: List[Dict]) -> List[Dict]:
        """Extract SPARQL queries from LC-QuAD data."""
        queries = []
        
        for item in lc_quad_data:
            try:
                query_info = {
                    'id': item['_id'],
                    'question': item.get('corrected_question', ''),
                    'sparql_query': item.get('sparql_query', '').strip(),
                    'template_id': item.get('sparql_template_id', None)
                }
                
                if query_info['sparql_query']:
                    queries.append(query_info)
                else:
                    logger.warning(f"Empty SPARQL query for question ID: {item['_id']}")
                    
            except KeyError as e:
                logger.error(f"Missing key {e} in question ID: {item.get('_id', 'unknown')}")
                continue
        
        logger.info(f"Extracted {len(queries)} valid SPARQL queries")
        return queries

    def execute_sparql_query(self, query: str, query_id: str) -> Optional[List[Dict]]:
        """Execute a single SPARQL query with retry logic."""
        for attempt in range(self.config['max_retries']):
            try:
                # Clean and validate query
                query = query.strip()
                if not query:
                    return None
                
                self.sparql.setQuery(query)
                
                # Add delay to avoid overwhelming the server
                time.sleep(self.config['request_delay'])
                
                results = self.sparql.query().convert()
                
                if 'results' in results and 'bindings' in results['results']:
                    return results['results']['bindings']
                else:
                    logger.warning(f"Unexpected result format for query {query_id}")
                    return None
                    
            except Exception as e:
                error_msg = str(e)
                self.query_stats['retries_used'] += 1
                
                # Categorize error types
                if 'SSL' in error_msg or 'certificate' in error_msg.lower():
                    self.query_stats['ssl_errors'] += 1
                    logger.warning(f"SSL error for query {query_id} (attempt {attempt + 1}): {error_msg}")
                elif 'timeout' in error_msg.lower():
                    self.query_stats['timeout_errors'] += 1
                    logger.warning(f"Timeout error for query {query_id} (attempt {attempt + 1}): {error_msg}")
                else:
                    logger.warning(f"Error for query {query_id} (attempt {attempt + 1}): {error_msg}")
                
                # Try next endpoint on SSL/connection errors
                if attempt < self.config['max_retries'] - 1:
                    if 'SSL' in error_msg or 'certificate' in error_msg.lower() or 'timeout' in error_msg.lower():
                        self._try_next_endpoint()
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    # Final failure
                    logger.error(f"Failed to execute query {query_id} after {self.config['max_retries']} attempts: {error_msg}")
                    self.failed_queries.append({
                        'id': query_id,
                        'query': query,
                        'error': error_msg,
                        'attempts': attempt + 1
                    })
                    return None

    def extract_triples_from_results(self, results: List[Dict], original_query: str) -> Set[Tuple[str, str, str]]:
        """Extract RDF triples from SPARQL query results."""
        triples = set()
        
        # Try to extract triples from the query pattern
        query_triples = self.extract_triples_from_query_pattern(original_query)
        
        # For each result binding, try to construct concrete triples
        for binding in results:
            for var_name, var_value in binding.items():
                if 'value' in var_value:
                    entity_uri = var_value['value']
                    
                    # Add triples from query pattern with this entity
                    for subj, pred, obj in query_triples:
                        if f"?{var_name}" in [subj, obj]:
                            # Replace variable with actual entity
                            concrete_subj = entity_uri if subj == f"?{var_name}" else subj
                            concrete_obj = entity_uri if obj == f"?{var_name}" else obj
                            
                            # Clean URIs
                            concrete_subj = self.clean_uri(concrete_subj)
                            concrete_obj = self.clean_uri(concrete_obj)
                            pred = self.clean_uri(pred)
                            
                            if concrete_subj and pred and concrete_obj:
                                triples.add((concrete_subj, pred, concrete_obj))
        
        return triples

    def extract_triples_from_query_pattern(self, query: str) -> List[Tuple[str, str, str]]:
        """Extract triple patterns from SPARQL WHERE clause."""
        triples = []
        
        try:
            # Extract WHERE clause content
            where_match = re.search(r'WHERE\s*\{(.*)\}', query, re.DOTALL | re.IGNORECASE)
            if not where_match:
                return triples
            
            where_content = where_match.group(1)
            
            # Find triple patterns (subject predicate object)
            triple_pattern = r'([?<][^>\s]+[>]?)\s+([<][^>\s]+[>])\s+([?<][^>\s.]+[>]?)'
            
            matches = re.findall(triple_pattern, where_content)
            
            for match in matches:
                subj, pred, obj = match
                # Clean up the matched strings
                subj = subj.strip()
                pred = pred.strip()
                obj = obj.strip()
                
                if subj and pred and obj:
                    triples.append((subj, pred, obj))
        
        except Exception as e:
            logger.warning(f"Failed to extract patterns from query: {str(e)}")
        
        return triples

    def clean_uri(self, uri: str) -> Optional[str]:
        """Clean and validate URI strings."""
        if not uri:
            return None
        
        # Remove angle brackets
        uri = uri.strip('<>')
        
        # Skip variables
        if uri.startswith('?'):
            return None
        
        # Validate that it's a proper URI
        if uri.startswith('http://') or uri.startswith('https://'):
            # Extract the entity name from DBpedia URIs
            if 'dbpedia.org/resource/' in uri:
                return uri.split('/')[-1]
            elif 'dbpedia.org/ontology/' in uri or 'dbpedia.org/property/' in uri:
                return uri.split('/')[-1]
            else:
                return uri.split('/')[-1] if '/' in uri else uri
        
        return uri

    def run_sparql_queries_batch(self, queries: List[Dict]) -> None:
        """Execute SPARQL queries in batches to extract triples."""
        batch_size = self.config['batch_size']
        logger.info(f"Starting batch processing of {len(queries)} queries (batch size: {batch_size})")
        
        total_batches = (len(queries) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(queries), batch_size):
            batch = queries[batch_idx:batch_idx + batch_size]
            current_batch_num = (batch_idx // batch_size) + 1
            
            logger.info(f"Processing batch {current_batch_num}/{total_batches} ({len(batch)} queries)")
            
            for query_info in batch:
                self.query_stats['total'] += 1
                
                results = self.execute_sparql_query(
                    query_info['sparql_query'], 
                    query_info['id']
                )
                
                if results is not None:
                    if len(results) > 0:
                        triples = self.extract_triples_from_results(
                            results, 
                            query_info['sparql_query']
                        )
                        self.all_triples.update(triples)
                        self.query_stats['successful'] += 1
                    else:
                        self.query_stats['empty_results'] += 1
                else:
                    self.query_stats['failed'] += 1
            
            # Log progress
            logger.info(f"Batch {current_batch_num} completed. "
                       f"Total triples so far: {len(self.all_triples)}")

    def format_triples_for_transe(self, output_file: str) -> None:
        """Format extracted triples for TransE training."""
        logger.info(f"Formatting {len(self.all_triples)} triples for TransE")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for head, relation, tail in self.all_triples:
                f.write(f"{head}\t{relation}\t{tail}\n")
        
        logger.info(f"Triples saved to {output_file}")

    def save_statistics(self, stats_file: str) -> None:
        """Save processing statistics to file."""
        stats = {
            'query_statistics': self.query_stats,
            'total_unique_triples': len(self.all_triples),
            'failed_queries_count': len(self.failed_queries),
            'processing_details': {
                'mode': self.mode,
                'endpoints_used': self.endpoints,
                'final_endpoint': self.endpoints[self.current_endpoint_idx],
                'config': self.config
            }
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics saved to {stats_file}")

    def save_failed_queries(self, failed_queries_file: str) -> None:
        """Save failed queries for debugging."""
        if self.failed_queries:
            with open(failed_queries_file, 'w', encoding='utf-8') as f:
                json.dump(self.failed_queries, f, indent=2)
            logger.info(f"Failed queries saved to {failed_queries_file}")

def main():
    """Main preprocessing pipeline."""
    
    # Setup directories
    Config.ensure_directories()
    
    # Initialize preprocessor
    mode = "default"  # Change to "test" for testing
    preprocessor = ImprovedLC_QuAD_Preprocessor(mode=mode)
    
    logger.info(f"=== Starting LC-QuAD Preprocessing (Mode: {mode}) ===")
    
    # Process training data
    logger.info("=== Processing Training Data ===")
    train_data = preprocessor.parse_lc_quad_json(Config.TRAIN_DATA_PATH)
    train_queries = preprocessor.extract_sparql_queries(train_data)
    
    # Execute queries and extract triples
    preprocessor.run_sparql_queries_batch(train_queries)
    
    # Process test data
    logger.info("=== Processing Test Data ===")
    test_data = preprocessor.parse_lc_quad_json(Config.TEST_DATA_PATH)
    test_queries = preprocessor.extract_sparql_queries(test_data)
    
    # Execute test queries (add to the same triple set)
    preprocessor.run_sparql_queries_batch(test_queries)
    
    # Save results
    logger.info("=== Saving Results ===")
    
    # Format and save triples for TransE
    triples_file = Config.get_output_path(Config.OUTPUT_FILES['triples'])
    preprocessor.format_triples_for_transe(triples_file)
    
    # Save statistics
    stats_file = Config.get_output_path(Config.OUTPUT_FILES['statistics'])
    preprocessor.save_statistics(stats_file)
    
    # Save failed queries for debugging
    failed_queries_file = Config.get_output_path(Config.OUTPUT_FILES['failed_queries'])
    preprocessor.save_failed_queries(failed_queries_file)
    
    # Print summary
    logger.info("=== PREPROCESSING SUMMARY ===")
    logger.info(f"Mode: {mode}")
    logger.info(f"Total queries processed: {preprocessor.query_stats['total']}")
    logger.info(f"Successful queries: {preprocessor.query_stats['successful']}")
    logger.info(f"Failed queries: {preprocessor.query_stats['failed']}")
    logger.info(f"Empty result queries: {preprocessor.query_stats['empty_results']}")
    logger.info(f"SSL errors: {preprocessor.query_stats['ssl_errors']}")
    logger.info(f"Timeout errors: {preprocessor.query_stats['timeout_errors']}")
    logger.info(f"Total retries used: {preprocessor.query_stats['retries_used']}")
    logger.info(f"Total unique triples extracted: {len(preprocessor.all_triples)}")
    logger.info(f"Output files saved to: {Config.OUTPUT_DIR}")

if __name__ == "__main__":
    main() 
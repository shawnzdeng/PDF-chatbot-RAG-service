#!/usr/bin/env python3
"""
Debug script to inspect actual field names in Qdrant collection payload
"""

import sys
from pathlib import Path
import json

# Add parent directory to path for imports
parent_dir = Path(__file__).parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from config import Config
from qdrant_client import QdrantClient

def debug_qdrant_collection():
    """Check actual field names in Qdrant collection"""
    try:
        Config.validate_env_vars()
        
        # Initialize Qdrant client
        client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY
        )
        
        # Get collection name
        collection_name = Config.qdrant_config.collection_name if Config.qdrant_config else Config.DEFAULT_COLLECTION_NAME
        print(f"Checking collection: {collection_name}")
        
        # Get collection info
        try:
            info = client.get_collection(collection_name)
            print(f"Collection points count: {info.points_count}")
            print(f"Collection status: {info.status}")
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return
        
        # Search for a few sample documents
        print("\n" + "="*50)
        print("SAMPLE DOCUMENT PAYLOADS:")
        print("="*50)
        
        # Use a simple query vector (zeros) to get any documents
        import numpy as np
        dummy_vector = np.zeros(3072).tolist()  # Correct dimension based on error message
        
        results = client.search(
            collection_name=collection_name,
            query_vector=dummy_vector,
            limit=3,
            score_threshold=0.0  # Very low threshold to get any results
        )
        
        if not results:
            print("No documents found in collection")
            return
        
        for i, result in enumerate(results, 1):
            print(f"\nDocument {i} (Score: {result.score:.3f}):")
            print(f"Payload keys: {list(result.payload.keys())}")
            
            # Print each field and its content preview
            for key, value in result.payload.items():
                if isinstance(value, str):
                    preview = value[:100] + "..." if len(value) > 100 else value
                    print(f"  {key}: {repr(preview)}")
                else:
                    print(f"  {key}: {type(value).__name__} = {value}")
            print("-" * 40)
        
        # Try to identify the content field
        print("\n" + "="*50)
        print("CONTENT FIELD ANALYSIS:")
        print("="*50)
        
        content_candidates = ['text', 'content', 'page_content', 'chunk', 'document', 'body']
        
        for result in results[:1]:  # Check first document
            payload = result.payload
            print(f"Available fields: {list(payload.keys())}")
            
            for candidate in content_candidates:
                if candidate in payload:
                    content = payload[candidate]
                    if isinstance(content, str) and len(content) > 50:
                        print(f"✓ Found content field '{candidate}': {len(content)} chars")
                        print(f"  Preview: {content[:200]}...")
                        break
            else:
                # Find the field with the longest string content
                longest_field = None
                longest_length = 0
                
                for key, value in payload.items():
                    if isinstance(value, str) and len(value) > longest_length:
                        longest_field = key
                        longest_length = len(value)
                
                if longest_field:
                    print(f"✓ Likely content field '{longest_field}': {longest_length} chars")
                    print(f"  Preview: {payload[longest_field][:200]}...")
        
    except Exception as e:
        print(f"Error debugging collection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_qdrant_collection()

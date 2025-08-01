#!/usr/bin/env python3
"""
Simple debug script to inspect Qdrant document fields
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from config import Config
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings

def inspect_documents():
    """Use the actual RAG system to get documents and inspect their structure"""
    try:
        Config.validate_env_vars()
        
        # Initialize components
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY
        )
        
        # Get collection name
        collection_name = Config.qdrant_config.collection_name if Config.qdrant_config else Config.DEFAULT_COLLECTION_NAME
        print(f"Inspecting collection: {collection_name}")
        
        # Generate a real query embedding
        query = "What is machine learning?"
        print(f"Query: {query}")
        
        query_vector = embeddings.embed_query(query)
        print(f"Query vector dimension: {len(query_vector)}")
        
        # Search for documents
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=3
        )
        
        print(f"\nFound {len(results)} documents:")
        print("="*60)
        
        for i, result in enumerate(results, 1):
            print(f"\nDocument {i} (Score: {result.score:.3f}):")
            print(f"Payload keys: {list(result.payload.keys())}")
            
            # Check each field
            for key, value in result.payload.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {len(value)} characters")
                    print(f"    Preview: {repr(value[:150])}...")
                else:
                    print(f"  {key}: {repr(value)}")
            
            # Check what field contains the main content
            content_candidates = ['text', 'content', 'page_content', 'chunk', 'document', 'body']
            main_content_field = None
            
            for candidate in content_candidates:
                if candidate in result.payload:
                    content = result.payload[candidate]
                    if isinstance(content, str) and len(content) > 100:
                        main_content_field = candidate
                        break
            
            if main_content_field:
                print(f"  --> MAIN CONTENT FIELD: '{main_content_field}'")
            else:
                print(f"  --> No obvious content field found!")
            
            print("-" * 40)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_documents()

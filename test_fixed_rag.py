#!/usr/bin/env python3
"""
Test the fixed RAG system with debug output
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Set up logging to see debug messages
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

def test_fixed_rag_system():
    """Test the fixed RAG system and print document content"""
    try:
        from src.rag_engine.qdrant_rag import QdrantRAG
        
        print("Creating RAG system from production config...")
        rag = QdrantRAG.from_production_config()
        
        # Test with a simple question
        query = "What is machine learning?"
        print(f"\nQuery: {query}")
        
        # Get documents directly to see their structure
        print("\nRetrieving documents...")
        retrieved_docs = rag.retrieve_documents(query)
        
        print(f"\nRetrieved {len(retrieved_docs)} documents")
        
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"\nDocument {i} (Score: {doc.score:.3f}):")
            print(f"Content length: {len(doc.content)}")
            if doc.content:
                print(f"Content preview: {repr(doc.content[:200])}")
            else:
                print("NO CONTENT FOUND!")
                print(f"Available metadata fields: {list(doc.metadata.keys())}")
                # Show content field specifically
                if 'content' in doc.metadata:
                    content_field = doc.metadata['content']
                    print(f"Metadata 'content' field: {len(content_field)} chars")
                    print(f"Metadata 'content' preview: {repr(content_field[:200])}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_rag_system()

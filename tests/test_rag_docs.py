#!/usr/bin/env python3
"""
Test the actual RAG system to see document content
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

def test_rag_system():
    """Test the actual RAG system and print document content"""
    try:
        from src.rag_engine.qdrant_rag import QdrantRAG
        
        print("Creating RAG system from production config...")
        rag = QdrantRAG.from_production_config()
        
        # Test with a simple question
        query = "What is machine learning?"
        print(f"Query: {query}")
        
        # Get documents directly to see their structure
        print("\nRetrieving documents...")
        retrieved_docs = rag.retrieve_documents(query)
        
        print(f"Retrieved {len(retrieved_docs)} documents")
        
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"\nDocument {i} (Score: {doc.score:.3f}):")
            print(f"Content length: {len(doc.content)}")
            print(f"Content preview: {repr(doc.content[:200])}")
            print(f"Metadata keys: {list(doc.metadata.keys())}")
            
            # Show all metadata fields
            for key, value in doc.metadata.items():
                if isinstance(value, str) and len(value) > 200:
                    print(f"  {key}: {len(value)} chars - {repr(value[:100])}...")
                else:
                    print(f"  {key}: {repr(value)}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rag_system()

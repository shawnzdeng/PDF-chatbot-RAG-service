#!/usr/bin/env python3
"""
Test the complete RAG system with a full question
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

def test_complete_rag():
    """Test the complete RAG system with a question"""
    try:
        from src.rag_engine.qdrant_rag import QdrantRAG
        
        print("Creating RAG system from production config...")
        rag = QdrantRAG.from_production_config()
        
        # Test with a machine learning question
        question = "What is machine learning?"
        print(f"\nQuestion: {question}")
        print("=" * 60)
        
        # Get complete answer
        result = rag.answer_question(question)
        
        print(f"Answer: {result['answer']}")
        print(f"\nRetrieved {result['retrieved_documents']} documents")
        print(f"Average relevance score: {result['average_relevance_score']:.3f}")
        
        # Show detailed sources
        print(f"\nDetailed Sources:")
        for source in result['detailed_sources']:
            print(f"  [{source['rank']}] {source['source']} (Page {source['page']}) - Score: {source['relevance_score']:.3f}")
            print(f"      Excerpt: {source['excerpt']}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_complete_rag()

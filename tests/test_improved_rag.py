"""
Test the improved RAG system with domain-specific prompts
"""

import sys
from pathlib import Path
import logging

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.rag_engine.qdrant_rag import QdrantRAG
from config import Config

def test_domain_specific_prompt():
    """Test the domain-specific prompt generation"""
    print("=== Testing Domain-Specific Prompt Generation ===")
    
    try:
        # Check if we have production config with document metadata
        doc_metadata = Config.get_document_metadata()
        print(f"Document metadata available: {bool(doc_metadata)}")
        
        if doc_metadata:
            print(f"File description: {doc_metadata.get('file_description', 'N/A')[:100]}...")
            print(f"Example questions count: {len(doc_metadata.get('example_questions', []))}")
            
            # Generate domain-specific template
            template = QdrantRAG._generate_domain_specific_prompt_template(doc_metadata)
            print(f"Generated template length: {len(template)} characters")
            
            # Check if it's machine learning focused
            if "machine learning" in template.lower():
                print("✓ Template is machine learning focused")
            else:
                print("⚠️ Template is not machine learning focused")
                
            # Show template preview
            print("\nTemplate preview:")
            print("=" * 50)
            print(template[:300] + "...")
            print("=" * 50)
        else:
            print("No document metadata found in production config")
            
    except Exception as e:
        print(f"Error testing domain-specific prompt: {e}")
        import traceback
        traceback.print_exc()

def test_improved_rag_system():
    """Test the improved RAG system"""
    print("\n=== Testing Improved RAG System ===")
    
    try:
        # Initialize RAG system from production config
        rag = QdrantRAG.from_production_config()
        print("✓ RAG system initialized from production config")
        
        # Test with machine learning questions
        test_questions = [
            "What is the difference between supervised and unsupervised learning?",
            "Explain how neural networks work",
            "What are decision trees?",
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n--- Test Question {i}: {question} ---")
            
            try:
                # Get answer
                result = rag.answer_question(question, top_k=3, save_to_memory=False)
                answer = result.get("answer", "No answer generated")
                retrieved_docs = result.get("retrieved_documents", 0)
                avg_score = result.get("average_relevance_score", 0)
                
                print(f"Retrieved documents: {retrieved_docs}")
                print(f"Average relevance score: {avg_score:.3f}")
                print(f"Answer preview: {answer[:200]}...")
                
                # Check if we're still getting the problematic response
                if "I don't have enough information" in answer and "provided documents" in answer:
                    print("⚠️ Still getting restrictive response - may need prompt adjustment")
                elif "I don't have enough information" in answer:
                    print("⚠️ Getting 'not enough information' response")
                elif len(answer.strip()) < 50:
                    print("⚠️ Very short answer - may indicate retrieval issues")
                else:
                    print("✓ Generated meaningful answer")
                    
            except Exception as e:
                print(f"✗ Error processing question: {e}")
        
        # Show system configuration
        print(f"\n--- System Configuration ---")
        print(f"Prompt template type: {'Domain-specific' if Config.get_document_metadata() else 'Default'}")
        print(f"Score threshold: {rag.score_threshold}")
        print(f"Reranking enabled: {rag.enable_reranking}")
        print(f"Conversation memory enabled: {rag.enable_conversation_memory}")
        
    except Exception as e:
        print(f"✗ Error testing RAG system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_domain_specific_prompt()
    test_improved_rag_system()

"""
Test script to verify production config loading
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config
from src.rag_engine.qdrant_rag import QdrantRAG

def test_config_loading():
    """Test that production config loads correctly"""
    print("=== Testing Production Config Loading ===")
    
    # Check if production config was loaded
    print(f"Production ready: {Config.is_production_ready()}")
    print(f"RAG config loaded: {Config.rag_config is not None}")
    print(f"Qdrant config loaded: {Config.qdrant_config is not None}")
    print(f"Reranker config loaded: {Config.reranker_config is not None}")
    
    if Config.rag_config:
        print(f"\nRAG Configuration:")
        print(f"  - Embedding model: {Config.rag_config.embedding_model}")
        print(f"  - LLM model: {Config.rag_config.llm_model}")
        print(f"  - Temperature: {Config.rag_config.temperature}")
        print(f"  - Top K: {Config.rag_config.top_k_retrieval}")
    
    if Config.qdrant_config:
        print(f"\nQdrant Configuration:")
        print(f"  - Collection: {Config.qdrant_config.collection_name}")
        print(f"  - URL: {Config.qdrant_config.url}")
        print(f"  - Vector size: {Config.qdrant_config.vector_size}")
    
    if Config.reranker_config:
        print(f"\nReranker Configuration:")
        print(f"  - Enabled: {Config.reranker_config.enabled}")
        print(f"  - Model: {Config.reranker_config.model}")
        print(f"  - Top K before rerank: {Config.reranker_config.top_k_before_rerank}")
        print(f"  - Hybrid scoring: {Config.reranker_config.hybrid_scoring}")
        print(f"  - Embedding weight: {Config.reranker_config.embedding_weight}")
        print(f"  - Cross-encoder weight: {Config.reranker_config.cross_encoder_weight}")
    
    # Test production metrics
    metrics = Config.get_production_metrics()
    if metrics:
        print(f"\nProduction Metrics:")
        print(f"  - Composite score: {metrics.get('composite_score', 'N/A')}")
        print(f"  - Faithfulness: {metrics.get('faithfulness', 'N/A')}")
        print(f"  - Answer relevancy: {metrics.get('answer_relevancy', 'N/A')}")
        print(f"  - Context precision: {metrics.get('context_precision', 'N/A')}")
        print(f"  - Context recall: {metrics.get('context_recall', 'N/A')}")
    
    # Test optimization info
    opt_info = Config.get_optimization_info()
    if opt_info:
        print(f"\nOptimization Info:")
        print(f"  - Tuning date: {opt_info.get('tuning_date', 'N/A')}")
        print(f"  - Combinations tested: {opt_info.get('total_combinations_tested', 'N/A')}")
        print(f"  - Ready for production: {opt_info.get('ready_for_production', 'N/A')}")

def test_rag_creation():
    """Test creating RAG instance from production config"""
    print("\n=== Testing RAG Creation ===")
    
    try:
        # Test creation with defaults (should use production config)
        rag_defaults = QdrantRAG.create_with_defaults()
        print("✓ Successfully created RAG with defaults")
        
        # Test creation from production config
        rag_production = QdrantRAG.from_production_config()
        print("✓ Successfully created RAG from production config")
        
        # Compare configurations
        config_defaults = rag_defaults.get_current_config()
        config_production = rag_production.get_current_config()
        
        print(f"\nConfiguration comparison:")
        print(f"  Collection name: {config_defaults['collection_name']} vs {config_production['collection_name']}")
        print(f"  Embedding model: {config_defaults['embedding_model']} vs {config_production['embedding_model']}")
        print(f"  LLM model: {config_defaults['llm_model']} vs {config_production['llm_model']}")
        print(f"  Temperature: {config_defaults['temperature']} vs {config_production['temperature']}")
        print(f"  Top K: {config_defaults['top_k']} vs {config_production['top_k']}")
        
        # Show production info
        prod_info = config_production.get('production_info', {})
        if prod_info:
            print(f"\nProduction info included: {prod_info.get('is_production_ready', 'N/A')}")
        
    except Exception as e:
        print(f"✗ Error creating RAG instances: {e}")

if __name__ == "__main__":
    test_config_loading()
    test_rag_creation()

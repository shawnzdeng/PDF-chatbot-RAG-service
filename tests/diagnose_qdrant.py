"""
Diagnostic script to check the current state of the RAG system
"""

import os
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

def check_environment():
    """Check environment variables and setup"""
    print("=== Environment Check ===")
    
    required_vars = ["OPENAI_API_KEY", "QDRANT_API_KEY", "QDRANT_URL"]
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✓ {var}: {'*' * 10} (set)")
        else:
            print(f"✗ {var}: Not set")

def check_files():
    """Check required files"""
    print("\n=== File Check ===")
    
    required_files = [
        "config.py",
        "src/rag_engine/qdrant_rag.py",
        "src/rag_engine/conversation_memory.py",
        "production_config/production_rag_config_20250731_214206.json"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}: Found")
        else:
            print(f"✗ {file_path}: Missing")

def check_production_config():
    """Check production config content"""
    print("\n=== Production Config Check ===")
    
    config_file = Path("production_config/production_rag_config_20250731_214206.json")
    if config_file.exists():
        try:
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            print(f"✓ Production config loaded")
            print(f"  - Has document metadata: {'document_metadata' in config}")
            print(f"  - Has RAG config: {'rag_system_config' in config}")
            print(f"  - Has Qdrant config: {'qdrant_config' in config}")
            print(f"  - Ready for production: {config.get('optimization_info', {}).get('ready_for_production', False)}")
            
            if 'document_metadata' in config:
                doc_meta = config['document_metadata']
                print(f"  - File description available: {'file_description' in doc_meta}")
                print(f"  - Example questions: {len(doc_meta.get('example_questions', []))}")
                
        except Exception as e:
            print(f"✗ Error reading production config: {e}")
    else:
        print("✗ Production config file not found")

def check_qdrant_collection():
    """Check Qdrant collection structure and sample data"""
    print("\n=== Qdrant Collection Check ===")
    
    try:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check if environment variables are available
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url or not qdrant_key:
            print("✗ QDRANT_URL or QDRANT_API_KEY not set")
            return
            
        from qdrant_client import QdrantClient
        
        # Initialize Qdrant client
        client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        
        # Get collection name from production config
        import json
        config_file = Path("production_config/production_rag_config_20250731_214206.json")
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        collection_name = config.get('qdrant_config', {}).get('collection_name', 'rag_collection')
        print(f"Collection name: {collection_name}")
        
        # Check if collection exists
        collections = client.get_collections()
        collection_exists = any(col.name == collection_name for col in collections.collections)
        
        if not collection_exists:
            print(f"✗ Collection '{collection_name}' does not exist")
            print(f"Available collections: {[col.name for col in collections.collections]}")
            return
        
        print(f"✓ Collection '{collection_name}' exists")
        
        # Get collection info
        collection_info = client.get_collection(collection_name)
        print(f"  - Vector size: {collection_info.config.params.vectors.size}")
        print(f"  - Points count: {collection_info.points_count}")
        
        # Get a sample point to check payload structure
        if collection_info.points_count > 0:
            sample_points = client.scroll(
                collection_name=collection_name,
                limit=1,
                with_payload=True,
                with_vectors=False
            )[0]
            
            if sample_points:
                sample_point = sample_points[0]
                print(f"  - Sample payload keys: {list(sample_point.payload.keys())}")
                
                # Check for common content field names
                content_fields = ['text', 'content', 'page_content', 'chunk', 'document']
                found_content_field = None
                
                for field in content_fields:
                    if field in sample_point.payload:
                        found_content_field = field
                        content_preview = str(sample_point.payload[field])[:100]
                        print(f"  - Content field '{field}': {content_preview}...")
                        break
                
                if not found_content_field:
                    print(f"  ⚠️ No common content field found. Available fields: {list(sample_point.payload.keys())}")
                    # Show a sample of each field
                    for key, value in sample_point.payload.items():
                        value_preview = str(value)[:50]
                        print(f"    - {key}: {value_preview}...")
                else:
                    print(f"  ✓ Content field found: '{found_content_field}'")
            else:
                print("  - No sample points available")
        else:
            print("  - Collection is empty")
            
    except ImportError as e:
        print(f"✗ Missing dependencies: {e}")
        print("Run: pip install qdrant-client python-dotenv")
    except Exception as e:
        print(f"✗ Error checking Qdrant collection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_environment()
    check_files()
    check_production_config()
    check_qdrant_collection()

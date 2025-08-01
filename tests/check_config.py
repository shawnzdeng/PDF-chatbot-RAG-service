"""
Configuration Checker for RAG Chatbot Streamlit App
Verifies that all dependencies and configurations are properly set up
"""

import os
import sys
from pathlib import Path
import json

def check_environment_variables():
    """Check if required environment variables are set"""
    print("🔍 Checking Environment Variables...")
    
    required_vars = ["OPENAI_API_KEY", "QDRANT_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {'*' * (len(value) - 8) + value[-8:] if len(value) > 8 else '***'}")
        else:
            print(f"❌ {var}: Not found")
            missing_vars.append(var)
    
    # Optional variables
    optional_vars = ["QDRANT_URL"]
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"ℹ️  {var}: {value}")
        else:
            print(f"⚠️  {var}: Not set (will use default)")
    
    return missing_vars

def check_dependencies():
    """Check if required Python packages are installed"""
    print("\n📦 Checking Dependencies...")
    
    required_packages = [
        "streamlit",
        "langchain",
        "langchain_openai", 
        "qdrant_client",
        "openai",
        "python-dotenv"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}: Installed")
        except ImportError:
            print(f"❌ {package}: Not installed")
            missing_packages.append(package)
    
    return missing_packages

def check_production_config():
    """Check if production configuration exists and is valid"""
    print("\n⚙️ Checking Production Configuration...")
    
    config_dir = Path("production_config")
    if not config_dir.exists():
        print("❌ production_config directory not found")
        return False
    
    config_files = list(config_dir.glob("production_rag_config_*.json"))
    if not config_files:
        print("❌ No production config files found")
        return False
    
    # Check the most recent config file
    latest_config = max(config_files, key=lambda p: p.stat().st_mtime)
    print(f"✅ Found config file: {latest_config.name}")
    
    try:
        with open(latest_config, 'r') as f:
            config = json.load(f)
        
        # Check required sections
        required_sections = [
            "document_metadata",
            "rag_system_config", 
            "qdrant_config",
            "performance_metrics"
        ]
        
        for section in required_sections:
            if section in config:
                print(f"✅ {section}: Present")
            else:
                print(f"❌ {section}: Missing")
        
        # Check document metadata specifically
        doc_meta = config.get("document_metadata", {})
        if doc_meta.get("file_description"):
            print(f"✅ Document description: Available ({len(doc_meta['file_description'])} chars)")
        else:
            print("⚠️  Document description: Not found")
        
        if doc_meta.get("example_questions"):
            print(f"✅ Example questions: {len(doc_meta['example_questions'])} available")
        else:
            print("⚠️  Example questions: Not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return False

def check_rag_system():
    """Check if RAG system can be initialized"""
    print("\n🤖 Checking RAG System...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, str(Path.cwd()))
        
        from config import Config
        print("✅ Config module: Imported successfully")
        
        # Try to load production config
        Config.load_from_production_config()
        if Config.is_production_ready():
            print("✅ Production config: Loaded and ready")
        else:
            print("⚠️  Production config: Loaded but not marked as ready")
        
        # Try to validate environment
        Config.validate_env_vars()
        print("✅ Environment validation: Passed")
        
        # Try to initialize RAG system
        from src.rag_engine.qdrant_rag import QdrantRAG
        rag = QdrantRAG.from_production_config()
        print("✅ RAG system: Initialized successfully")
        
        # Get collection stats
        stats = rag.get_collection_stats()
        if stats and stats.get('total_points', 0) > 0:
            print(f"✅ Qdrant collection: {stats['total_points']} documents available")
        else:
            print("⚠️  Qdrant collection: No documents found or connection failed")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG system error: {e}")
        return False

def main():
    """Run all configuration checks"""
    print("🔧 RAG Chatbot Configuration Checker")
    print("=" * 50)
    
    # Check environment variables
    missing_env = check_environment_variables()
    
    # Check dependencies
    missing_deps = check_dependencies()
    
    # Check production config
    config_ok = check_production_config()
    
    # Check RAG system (only if other checks pass)
    rag_ok = False
    if not missing_env and not missing_deps and config_ok:
        rag_ok = check_rag_system()
    else:
        print("\n🤖 Skipping RAG system check due to missing requirements")
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    
    if missing_env:
        print(f"❌ Missing environment variables: {', '.join(missing_env)}")
        print("   💡 Set these in your .env file or system environment")
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("   💡 Run: pip install -r requirements.txt")
    
    if not config_ok:
        print("❌ Production configuration issues detected")
        print("   💡 Check production_config directory and files")
    
    if missing_env or missing_deps or not config_ok:
        print("\n🚫 Setup incomplete - please fix the issues above before running the app")
        return False
    
    if rag_ok:
        print("✅ All checks passed! You're ready to run the Streamlit app")
        print("   💡 Run: streamlit run src/ui/streamlit_app.py")
        print("   💡 Or use: python launch_ui.py")
        return True
    else:
        print("⚠️  Basic setup complete but RAG system has issues")
        print("   💡 Check Qdrant connection and collection setup")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

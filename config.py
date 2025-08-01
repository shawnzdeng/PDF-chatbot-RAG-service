"""
Configuration management for RAG system
Supports loading from production config JSON files and environment variables
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

@dataclass
class RAGConfig:
    """Configuration for RAG system parameters"""
    chunk_size: int = 500
    chunk_overlap: int = 100
    embedding_model: str = "text-embedding-3-large"
    llm_model: str = "gpt-4o"
    temperature: float = 0.0
    top_k_retrieval: int = 5

@dataclass
class QdrantConfig:
    """Configuration for Qdrant vector database"""
    collection_name: str = "rag_collection"
    url: str = "http://localhost:6333"
    vector_size: int = 3072

@dataclass
class RerankerConfig:
    """Configuration for reranking system"""
    enabled: bool = True
    embedding_weight: float = 0.3
    cross_encoder_weight: float = 0.7
    top_k_before_rerank: int = 10
    hybrid_scoring: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    score_threshold: float = 0.3

class Config:
    """Main configuration class for RAG system"""
    
    # Environment variables
    OPENAI_API_KEY = None
    QDRANT_API_KEY = None
    QDRANT_URL = None
    DEFAULT_COLLECTION_NAME = "rag_collection"
    
    # Configuration objects
    rag_config: RAGConfig = None
    qdrant_config: QdrantConfig = None
    reranker_config: RerankerConfig = None
    
    # Production config data
    _production_config: Dict[str, Any] = None
    
    @classmethod
    def load_from_production_config(cls, config_file_path: str = None) -> None:
        """
        Load configuration from production config JSON file
        
        Args:
            config_file_path: Path to production config JSON file
                            If None, will look for the latest file in production_config/
        """
        if config_file_path is None:
            # Look for production config file
            production_dir = Path("production_config")
            if production_dir.exists():
                config_files = list(production_dir.glob("production_rag_config_*.json"))
                if config_files:
                    # Use the most recent file
                    config_file_path = max(config_files, key=lambda p: p.stat().st_mtime)
                    logger.info(f"Using production config: {config_file_path}")
                else:
                    logger.warning("No production config files found in production_config/")
                    return
            else:
                logger.warning("production_config directory not found")
                return
        
        try:
            with open(config_file_path, 'r') as f:
                cls._production_config = json.load(f)
            
            logger.info(f"Loaded production config from {config_file_path}")
            cls._apply_production_config()
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load production config from {config_file_path}: {e}")
            cls._load_defaults()
    
    @classmethod
    def _apply_production_config(cls) -> None:
        """Apply production configuration to config objects"""
        if not cls._production_config:
            logger.warning("No production config loaded, using defaults")
            cls._load_defaults()
            return
        
        # Load RAG system config
        rag_sys_config = cls._production_config.get("rag_system_config", {})
        cls.rag_config = RAGConfig(
            chunk_size=rag_sys_config.get("chunk_size", 500),
            chunk_overlap=rag_sys_config.get("chunk_overlap", 100),
            embedding_model=rag_sys_config.get("embedding_model", "text-embedding-3-large"),
            llm_model=rag_sys_config.get("llm_model", "gpt-4o"),
            temperature=rag_sys_config.get("temperature", 0.0),
            top_k_retrieval=rag_sys_config.get("top_k_retrieval", 5)
        )
        
        # Load Qdrant config
        qdrant_conf = cls._production_config.get("qdrant_config", {})
        cls.qdrant_config = QdrantConfig(
            collection_name=qdrant_conf.get("collection_name", "rag_collection"),
            url=qdrant_conf.get("url", "http://localhost:6333"),
            vector_size=qdrant_conf.get("vector_size", 3072)
        )
        
        # Load reranker config
        reranker_conf = cls._production_config.get("reranker_config", {})
        cls.reranker_config = RerankerConfig(
            enabled=reranker_conf.get("enabled", True),
            embedding_weight=reranker_conf.get("embedding_weight", 0.3),
            cross_encoder_weight=reranker_conf.get("cross_encoder_weight", 0.7),
            top_k_before_rerank=int(reranker_conf.get("top_k_before_rerank", 10)),
            hybrid_scoring=reranker_conf.get("hybrid_scoring", True),
            model=reranker_conf.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            score_threshold=reranker_conf.get("score_threshold", 0.3)
        )
        
        # Update class attributes for backward compatibility
        cls.DEFAULT_COLLECTION_NAME = cls.qdrant_config.collection_name
        
        logger.info("Applied production configuration successfully")
    
    @classmethod
    def _load_defaults(cls) -> None:
        """Load default configuration"""
        cls.rag_config = RAGConfig()
        cls.qdrant_config = QdrantConfig()
        cls.reranker_config = RerankerConfig()
        cls.DEFAULT_COLLECTION_NAME = cls.qdrant_config.collection_name
    
    @classmethod
    def validate_env_vars(cls) -> None:
        """Validate required environment variables"""
        # Load environment variables
        cls.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        cls.QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
        cls.QDRANT_URL = os.getenv("QDRANT_URL")
        
        # Use Qdrant URL from config if not in environment
        if not cls.QDRANT_URL and cls.qdrant_config:
            cls.QDRANT_URL = cls.qdrant_config.url
        
        # Validate required variables
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        if not cls.QDRANT_API_KEY:
            logger.warning("QDRANT_API_KEY not found in environment variables")
        
        if not cls.QDRANT_URL:
            logger.warning("QDRANT_URL not found in environment or config, using default")
            cls.QDRANT_URL = "http://localhost:6333"
    
    @classmethod
    def get_production_metrics(cls) -> Dict[str, Any]:
        """Get performance metrics from production config"""
        if cls._production_config:
            return cls._production_config.get("performance_metrics", {})
        return {}
    
    @classmethod
    def get_optimization_info(cls) -> Dict[str, Any]:
        """Get optimization information from production config"""
        if cls._production_config:
            return cls._production_config.get("optimization_info", {})
        return {}
    
    @classmethod
    def get_document_metadata(cls) -> Dict[str, Any]:
        """Get document metadata from production config"""
        if cls._production_config:
            return cls._production_config.get("document_metadata", {})
        return {}
    
    @classmethod
    def is_production_ready(cls) -> bool:
        """Check if configuration is marked as production ready"""
        if cls._production_config:
            opt_info = cls._production_config.get("optimization_info", {})
            return opt_info.get("ready_for_production", False)
        return False

# Initialize configuration on import
try:
    Config.load_from_production_config()
except Exception as e:
    logger.warning(f"Failed to load production config on import: {e}")
    Config._load_defaults()

# Validate environment variables
try:
    Config.validate_env_vars()
except Exception as e:
    logger.error(f"Environment validation failed: {e}")

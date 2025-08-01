"""
Simple RAG implementation using Qdrant and OpenAI
Provides question-answering capabilities over PDF documents
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Add parent directory to path for config import
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from config import Config
from .conversation_memory import ConversationMemory

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Data class for retrieval results"""
    content: str
    score: float
    metadata: Dict[str, Any]

class QdrantRAG:
    """RAG system using Qdrant for retrieval and OpenAI for generation"""
    
    def __init__(self,
                 collection_name: str = None,
                 embedding_model: str = None,
                 llm_model: str = None,
                 temperature: float = None,
                 top_k: int = 5,
                 prompt_template: str = None,
                 score_threshold: float = 0.3,
                 enable_reranking: bool = True,
                 reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 enable_conversation_memory: bool = True,
                 memory_max_turns: int = 10,
                 memory_summarization_threshold: int = 8):
        """
        Initialize RAG system
        
        Args:
            collection_name: Qdrant collection name (required)
            embedding_model: OpenAI embedding model (required)
            llm_model: OpenAI LLM model (required)
            temperature: LLM temperature (required)
            top_k: Number of documents to retrieve
            prompt_template: Custom prompt template for RAG (required)
            score_threshold: Minimum similarity threshold for document retrieval
            enable_reranking: Whether to use document reranking for improved relevance
            reranking_model: Cross-encoder model for reranking
            enable_conversation_memory: Whether to enable conversation memory for continuous conversations
            memory_max_turns: Maximum number of conversation turns to keep in active memory
            memory_summarization_threshold: Number of turns before triggering memory summarization
        """
        Config.validate_env_vars()
        
        # Validate required parameters
        if collection_name is None:
            raise ValueError("collection_name is required and cannot be None")
        if embedding_model is None:
            raise ValueError("embedding_model is required and cannot be None")
        if llm_model is None:
            raise ValueError("llm_model is required and cannot be None")
        if temperature is None:
            raise ValueError("temperature is required and cannot be None")
        if prompt_template is None:
            raise ValueError("prompt_template is required and cannot be None")
        
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.temperature = temperature
        self.top_k = top_k
        self.prompt_template_str = prompt_template
        self.score_threshold = score_threshold
        self.enable_reranking = enable_reranking
        self.reranking_model = reranking_model
        self.enable_conversation_memory = enable_conversation_memory
        self.memory_max_turns = memory_max_turns
        self.memory_summarization_threshold = memory_summarization_threshold
        
        # Initialize conversation memory if enabled
        if self.enable_conversation_memory:
            self.conversation_memory = ConversationMemory(
                max_turns=memory_max_turns,
                summarization_threshold=memory_summarization_threshold,
                enable_summarization=True
            )
            logger.info(f"Conversation memory enabled with max_turns={memory_max_turns}")
        else:
            self.conversation_memory = None
        
        # Initialize reranker if enabled
        if self.enable_reranking:
            try:
                from .reranker import DocumentReranker, RerankingConfig
                rerank_config = RerankingConfig.from_production_config()
                # Override top_k_after_rerank with the RAG system's top_k
                rerank_config.top_k_after_rerank = top_k
                # Respect the top_k_before_rerank from config instead of overriding
                # Only override if the config value seems too small relative to desired output
                if rerank_config.top_k_before_rerank < top_k:
                    logger.warning(f"Config top_k_before_rerank ({rerank_config.top_k_before_rerank}) is less than top_k ({top_k}). "
                                 f"Setting to max(top_k * 2, 20) = {max(top_k * 2, 20)}")
                    rerank_config.top_k_before_rerank = max(top_k * 2, 20)
                # Override model if specified
                if self.reranking_model != "cross-encoder/ms-marco-MiniLM-L-6-v2":
                    rerank_config.model_name = self.reranking_model
                
                self.reranker = DocumentReranker(rerank_config)
                logger.info(f"Reranking enabled with model: {rerank_config.model_name}, "
                           f"top_k_before_rerank: {rerank_config.top_k_before_rerank}, "
                           f"top_k_after_rerank: {rerank_config.top_k_after_rerank}, "
                           f"hybrid_scoring: {rerank_config.enable_hybrid_scoring}")
            except ImportError as e:
                logger.warning(f"Could not import reranker: {e}. Continuing without reranking.")
                self.enable_reranking = False
                self.reranker = None
        else:
            self.reranker = None
        
        # Initialize clients
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        self.llm = ChatOpenAI(
            model=self.llm_model,
            temperature=self.temperature,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        self.qdrant_client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY
        )
        
        # Setup prompt template
        self.prompt_template = self._create_prompt_template()
        
        logger.info(f"Initialized QdrantRAG with model={self.llm_model}, "
                   f"temp={self.temperature}, top_k={self.top_k}, "
                   f"score_threshold={self.score_threshold}, "
                   f"reranking_enabled={self.enable_reranking}, "
                   f"prompt_template_length={len(self.prompt_template_str)}")
    
    @classmethod
    def create_with_defaults(cls,
                           collection_name: str = None,
                           embedding_model: str = None,
                           llm_model: str = None,
                           temperature: float = None,
                           top_k: int = None,
                           prompt_template: str = None,
                           score_threshold: float = None,
                           enable_reranking: bool = None,
                           enable_conversation_memory: bool = True,
                           memory_max_turns: int = 10,
                           memory_summarization_threshold: int = 8):
        """
        Create QdrantRAG instance with defaults from production config
        
        Args:
            collection_name: Qdrant collection name
            embedding_model: OpenAI embedding model  
            llm_model: OpenAI LLM model
            temperature: LLM temperature
            top_k: Number of documents to retrieve
            prompt_template: Custom prompt template for RAG
            score_threshold: Minimum similarity threshold for document retrieval
            enable_reranking: Whether to enable document reranking
            enable_conversation_memory: Whether to enable conversation memory
            memory_max_turns: Maximum number of conversation turns to keep
            memory_summarization_threshold: Number of turns before summarization
        """
        # Use production config values as defaults
        collection_name = collection_name or (Config.qdrant_config.collection_name if Config.qdrant_config else Config.DEFAULT_COLLECTION_NAME)
        embedding_model = embedding_model or (Config.rag_config.embedding_model if Config.rag_config else "text-embedding-3-large")
        llm_model = llm_model or (Config.rag_config.llm_model if Config.rag_config else "gpt-4o")
        temperature = temperature if temperature is not None else (Config.rag_config.temperature if Config.rag_config else 0.0)
        top_k = top_k if top_k is not None else (Config.rag_config.top_k_retrieval if Config.rag_config else 5)
        score_threshold = score_threshold if score_threshold is not None else (Config.reranker_config.score_threshold if Config.reranker_config else 0.3)
        enable_reranking = enable_reranking if enable_reranking is not None else (Config.reranker_config.enabled if Config.reranker_config else True)
        prompt_template = prompt_template or cls._get_default_prompt_template_static()
        
        return cls(
            collection_name=collection_name,
            embedding_model=embedding_model,
            llm_model=llm_model,
            temperature=temperature,
            top_k=top_k,
            prompt_template=prompt_template,
            score_threshold=score_threshold,
            enable_reranking=enable_reranking,
            enable_conversation_memory=enable_conversation_memory,
            memory_max_turns=memory_max_turns,
            memory_summarization_threshold=memory_summarization_threshold
        )
    
    @classmethod
    def from_production_config(cls) -> 'QdrantRAG':
        """
        Create QdrantRAG instance using production configuration
        
        Returns:
            QdrantRAG instance configured with production parameters
        """
        if not Config.is_production_ready():
            logger.warning("Production config is not marked as ready. Proceeding anyway...")
        
        # Get configuration from production config
        collection_name = Config.qdrant_config.collection_name if Config.qdrant_config else Config.DEFAULT_COLLECTION_NAME
        embedding_model = Config.rag_config.embedding_model if Config.rag_config else "text-embedding-3-large"
        llm_model = Config.rag_config.llm_model if Config.rag_config else "gpt-4o"
        temperature = Config.rag_config.temperature if Config.rag_config else 0.0
        top_k = Config.rag_config.top_k_retrieval if Config.rag_config else 5
        score_threshold = Config.reranker_config.score_threshold if Config.reranker_config else 0.3
        enable_reranking = Config.reranker_config.enabled if Config.reranker_config else True
        
        # Use default prompt template
        prompt_template = cls._get_default_prompt_template_static()
        
        logger.info("Creating QdrantRAG from production configuration")
        
        # Log production metrics if available
        metrics = Config.get_production_metrics()
        if metrics:
            logger.info(f"Production metrics - Composite Score: {metrics.get('composite_score', 'N/A'):.3f}, "
                       f"Faithfulness: {metrics.get('faithfulness', 'N/A'):.3f}, "
                       f"Answer Relevancy: {metrics.get('answer_relevancy', 'N/A'):.3f}")
        
        return cls(
            collection_name=collection_name,
            embedding_model=embedding_model,
            llm_model=llm_model,
            temperature=temperature,
            top_k=top_k,
            prompt_template=prompt_template,
            score_threshold=score_threshold,
            enable_reranking=enable_reranking,
            enable_conversation_memory=True,  # Enable by default for production
            memory_max_turns=10,
            memory_summarization_threshold=8
        )
    
    @classmethod
    def from_config(cls,
                   collection_name: str,
                   embedding_model: str,
                   llm_model: str,
                   temperature: float,
                   top_k: int,
                   prompt_template: str,
                   score_threshold: float = 0.3,
                   enable_reranking: bool = True):
        """
        Create QdrantRAG instance from specific configuration parameters
        
        Args:
            collection_name: Qdrant collection name
            embedding_model: OpenAI embedding model
            llm_model: OpenAI LLM model  
            temperature: LLM temperature
            top_k: Number of documents to retrieve
            prompt_template: Custom prompt template for RAG
            score_threshold: Minimum similarity threshold for document retrieval
            enable_reranking: Whether to enable document reranking
            
        Returns:
            QdrantRAG instance with specified parameters
        """
        return cls(
            collection_name=collection_name,
            embedding_model=embedding_model,
            llm_model=llm_model,
            temperature=temperature,
            top_k=top_k,
            prompt_template=prompt_template,
            score_threshold=score_threshold,
            enable_reranking=enable_reranking
        )
    
    @staticmethod
    def _get_default_prompt_template_static() -> str:
        """Get the default prompt template string (static version)"""
        return """You are a helpful assistant that answers questions based on the provided context from PDF documents.

Context from documents:
{context}

Question: {question}

Instructions:
1. Answer the question based ONLY on the information provided in the context
2. If the answer is not found in the context, clearly state "I don't have enough information in the provided documents to answer this question"
3. Be concise and accurate
4. If relevant, mention which part of the document the information comes from

Answer:"""

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for RAG"""
        # Always use the original template since conversation context is handled dynamically
        return ChatPromptTemplate.from_template(self.prompt_template_str)
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for query
        
        Args:
            query: User question
            
        Returns:
            Query embedding vector
        """
        try:
            embedding = self.embeddings.embed_query(query)
            return embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise
    
    def retrieve_documents(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """
        Retrieve relevant documents from Qdrant with optional reranking
        
        Args:
            query: User question
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with scores (reranked if enabled)
        """
        top_k = top_k or self.top_k
        
        # If reranking is enabled, retrieve more documents for reranking
        search_limit = top_k
        if self.enable_reranking and self.reranker is not None:
            # Use the configured top_k_before_rerank value instead of hardcoded calculation
            search_limit = self.reranker.config.top_k_before_rerank
            logger.debug(f"Using configured top_k_before_rerank: {search_limit} (final top_k: {top_k})")
        
        try:
            # Generate query embedding
            query_embedding = self.embed_query(query)
            
            # Search in Qdrant with configurable score threshold
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=search_limit,
                score_threshold=self.score_threshold
            )
            
            # If no results with threshold, try without threshold as fallback
            if not search_result and self.score_threshold > 0:
                logger.warning(f"No documents found with score_threshold={self.score_threshold}, trying without threshold")
                search_result = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=search_limit
                )
            
            # Convert to RetrievalResult objects
            results = []
            for result in search_result:
                retrieval_result = RetrievalResult(
                    content=result.payload.get("content", ""),
                    score=result.score,
                    metadata={
                        "source": result.payload.get("source", ""),
                        "page": result.payload.get("page", 0),
                        "chunk_index": result.payload.get("chunk_index", 0)
                    }
                )
                results.append(retrieval_result)
            
            # Apply reranking if enabled
            if self.enable_reranking and self.reranker is not None and results:
                logger.debug(f"Applying reranking to {len(results)} documents")
                results = self.reranker.rerank_documents(query, results, top_k)
                logger.info(f"Reranked documents, returning top {len(results)}")
            
            logger.info(f"Retrieved {len(results)} documents for query (threshold={self.score_threshold}, reranking={self.enable_reranking})")
            if results and len(results) > 0:
                logger.debug(f"Score range: {min(r.score for r in results):.3f} - {max(r.score for r in results):.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def format_context(self, results: List[RetrievalResult]) -> str:
        """
        Format retrieved documents into context string
        
        Args:
            results: Retrieved documents
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant documents found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            source_info = f"Document {i} (Score: {result.score:.3f})"
            if result.metadata.get("source"):
                source_info += f" from {result.metadata['source']}"
            if result.metadata.get("page"):
                source_info += f", Page {result.metadata['page']}"
            
            context_parts.append(f"{source_info}:\n{result.content}\n")
        
        return "\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str, use_conversation_memory: bool = True) -> str:
        """
        Generate answer using LLM with optional conversation memory
        
        Args:
            query: User question
            context: Retrieved context
            use_conversation_memory: Whether to include conversation context
            
        Returns:
            Generated answer
        """
        try:
            # Get conversation context if enabled
            conversation_context = ""
            has_conversation_context = False
            
            if (use_conversation_memory and 
                self.enable_conversation_memory and 
                self.conversation_memory and 
                self.conversation_memory.has_conversation_history()):
                
                conversation_context = self.conversation_memory.get_conversation_context(
                    current_question=query,
                    include_summaries=True,
                    max_recent_turns=3
                )
                has_conversation_context = bool(conversation_context.strip())
            
            # Choose the appropriate prompt template and chain based on conversation context
            if has_conversation_context:
                # Use conversation-aware template
                enhanced_template = """You are a helpful AI assistant that answers questions based on provided documents and conversation history.

{conversation_context}

Context from documents:
{context}

Question: {question}

Instructions:
1. Answer the question based ONLY on the information provided in the context and any relevant conversation history
2. If the answer is not found in the context, clearly state "I don't have enough information in the provided documents to answer this question"
3. Be concise and accurate
4. If relevant, mention which part of the document the information comes from
5. Consider the conversation history to provide contextual and coherent responses
6. If the question refers to something mentioned earlier in the conversation, acknowledge that context

Answer:"""
                
                from langchain.prompts import ChatPromptTemplate
                conv_prompt_template = ChatPromptTemplate.from_template(enhanced_template)
                
                # Create conversation-aware chain
                chain = (
                    {
                        "context": RunnablePassthrough(), 
                        "question": RunnablePassthrough(),
                        "conversation_context": RunnablePassthrough()
                    }
                    | conv_prompt_template
                    | self.llm
                    | StrOutputParser()
                )
                
                # Generate answer with conversation context
                answer = chain.invoke({
                    "context": context, 
                    "question": query,
                    "conversation_context": conversation_context
                })
            else:
                # Use original template without conversation context
                chain = (
                    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                    | self.prompt_template
                    | self.llm
                    | StrOutputParser()
                )
                
                # Generate answer
                answer = chain.invoke({"context": context, "question": query})
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def answer_question(self, question: str, top_k: int = None, save_to_memory: bool = True) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve and generate answer with conversation memory
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            save_to_memory: Whether to save this interaction to conversation memory
            
        Returns:
            Dictionary with answer, context, and metadata
        """
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retrieve_documents(question, top_k)
            
            # Format context
            context = self.format_context(retrieved_docs)
            
            # Generate answer with conversation awareness
            answer = self.generate_answer(question, context, use_conversation_memory=True)
            
            # Save to conversation memory if enabled
            if (save_to_memory and 
                self.enable_conversation_memory and 
                self.conversation_memory):
                
                sources = [doc.metadata.get("source", "") for doc in retrieved_docs]
                self.conversation_memory.add_turn(
                    user_message=question,
                    assistant_response=answer,
                    retrieved_context=context,
                    sources=sources,
                    metadata={
                        "retrieved_documents": len(retrieved_docs),
                        "average_relevance_score": sum(doc.score for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0,
                        "model_params": {
                            "llm_model": self.llm_model,
                            "embedding_model": self.embedding_model,
                            "temperature": self.temperature,
                            "top_k": top_k or self.top_k
                        }
                    }
                )
            
            # Calculate average relevance score
            avg_score = sum(doc.score for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0
            
            result = {
                "question": question,
                "answer": answer,
                "context": context,
                "retrieved_documents": len(retrieved_docs),
                "average_relevance_score": avg_score,
                "sources": [doc.metadata.get("source", "") for doc in retrieved_docs],
                "model_params": {
                    "llm_model": self.llm_model,
                    "embedding_model": self.embedding_model,
                    "temperature": self.temperature,
                    "top_k": top_k or self.top_k,
                    "prompt_template": self.prompt_template_str
                }
            }
            
            # Add conversation memory stats if enabled
            if self.enable_conversation_memory and self.conversation_memory:
                result["conversation_stats"] = self.conversation_memory.get_memory_stats()
            
            logger.info(f"Generated answer for question with {len(retrieved_docs)} retrieved docs")
            return result
            
        except Exception as e:
            logger.error(f"Error in answer_question: {e}")
            return {
                "question": question,
                "answer": f"Error processing question: {str(e)}",
                "context": "",
                "retrieved_documents": 0,
                "average_relevance_score": 0,
                "sources": [],
                "model_params": {}
            }
    
    def update_reranking_config(self, rerank_config_params: Dict[str, Any]) -> None:
        """
        Update reranking configuration for parameter tuning
        
        Args:
            rerank_config_params: Dictionary with reranking parameters to update
                - top_k_before_rerank: Number of documents to retrieve before reranking
                - enable_hybrid_scoring: Whether to use hybrid scoring
                - cross_encoder_weight: Weight for cross-encoder scores
                - embedding_weight: Weight for embedding scores
                - model_name: Cross-encoder model name
        """
        if not self.enable_reranking or self.reranker is None:
            logger.warning("Reranking not enabled, cannot update config")
            return
        
        # Update configuration
        config = self.reranker.config
        
        if 'top_k_before_rerank' in rerank_config_params:
            config.top_k_before_rerank = rerank_config_params['top_k_before_rerank']
            logger.debug(f"Updated top_k_before_rerank to {config.top_k_before_rerank}")
        
        if 'enable_hybrid_scoring' in rerank_config_params:
            config.enable_hybrid_scoring = rerank_config_params['enable_hybrid_scoring']
            logger.debug(f"Updated enable_hybrid_scoring to {config.enable_hybrid_scoring}")
        
        if 'cross_encoder_weight' in rerank_config_params:
            config.cross_encoder_weight = rerank_config_params['cross_encoder_weight']
            logger.debug(f"Updated cross_encoder_weight to {config.cross_encoder_weight}")
        
        if 'embedding_weight' in rerank_config_params:
            config.embedding_weight = rerank_config_params['embedding_weight']
            logger.debug(f"Updated embedding_weight to {config.embedding_weight}")
        
        if 'model_name' in rerank_config_params and rerank_config_params['model_name'] != config.model_name:
            # Need to reinitialize the cross-encoder with new model
            try:
                from .reranker import DocumentReranker
                config.model_name = rerank_config_params['model_name']
                self.reranker = DocumentReranker(config)
                logger.info(f"Reinitialized reranker with model: {config.model_name}")
            except Exception as e:
                logger.error(f"Failed to update reranker model: {e}")

    def get_current_config(self) -> Dict[str, Any]:
        """
        Get current RAG configuration including reranking settings and production info
        
        Returns:
            Dictionary with current configuration
        """
        config = {
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
            "enable_reranking": self.enable_reranking,
            "reranking_model": self.reranking_model,
            "enable_conversation_memory": self.enable_conversation_memory,
            "memory_max_turns": self.memory_max_turns,
            "memory_summarization_threshold": self.memory_summarization_threshold
        }
        
        # Add reranking configuration if available
        if self.enable_reranking and self.reranker is not None:
            rerank_config = self.reranker.config
            config["reranking_config"] = {
                "model_name": rerank_config.model_name,
                "top_k_before_rerank": rerank_config.top_k_before_rerank,
                "top_k_after_rerank": rerank_config.top_k_after_rerank,
                "enable_hybrid_scoring": rerank_config.enable_hybrid_scoring,
                "cross_encoder_weight": rerank_config.cross_encoder_weight,
                "embedding_weight": rerank_config.embedding_weight
            }
        
        # Add production configuration information
        # Add conversation memory stats if enabled
        if self.enable_conversation_memory and self.conversation_memory:
            config["conversation_memory_stats"] = self.conversation_memory.get_memory_stats()
        
        config["production_info"] = {
            "is_production_ready": Config.is_production_ready(),
            "performance_metrics": Config.get_production_metrics(),
            "optimization_info": Config.get_optimization_info(),
            "document_metadata": Config.get_document_metadata()
        }
        
        return config

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the Qdrant collection"""
        try:
            info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "total_points": info.points_count,
                "vectors_count": info.vectors_count,
                "indexed_vectors": info.indexed_vectors_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    # Conversation Management Methods
    
    def start_conversation(self, conversation_id: str = None) -> str:
        """
        Start a new conversation session
        
        Args:
            conversation_id: Optional custom conversation ID
            
        Returns:
            Conversation ID
        """
        if not self.enable_conversation_memory or not self.conversation_memory:
            logger.warning("Conversation memory is not enabled")
            return None
            
        return self.conversation_memory.start_new_conversation(conversation_id)
    
    def clear_conversation_memory(self) -> None:
        """Clear all conversation memory"""
        if self.enable_conversation_memory and self.conversation_memory:
            self.conversation_memory.clear_memory()
            logger.info("Conversation memory cleared")
        else:
            logger.warning("Conversation memory is not enabled")
    
    def get_conversation_history(self) -> Dict[str, Any]:
        """
        Get current conversation history and statistics
        
        Returns:
            Dictionary with conversation history and stats
        """
        if not self.enable_conversation_memory or not self.conversation_memory:
            return {"error": "Conversation memory is not enabled"}
        
        stats = self.conversation_memory.get_memory_stats()
        
        # Get recent conversation turns for display
        recent_turns = []
        if self.conversation_memory.conversation_history:
            for turn in list(self.conversation_memory.conversation_history):
                recent_turns.append({
                    "user_message": turn.user_message,
                    "assistant_response": turn.assistant_response,
                    "timestamp": turn.timestamp.isoformat(),
                    "sources": turn.sources
                })
        
        return {
            "conversation_stats": stats,
            "recent_turns": recent_turns,
            "has_summaries": len(self.conversation_memory.conversation_summaries) > 0
        }
    
    def export_conversation_memory(self) -> Dict[str, Any]:
        """
        Export conversation memory for persistence
        
        Returns:
            Dictionary representation of conversation memory
        """
        if not self.enable_conversation_memory or not self.conversation_memory:
            return {"error": "Conversation memory is not enabled"}
        
        return self.conversation_memory.save_to_dict()
    
    def import_conversation_memory(self, memory_data: Dict[str, Any]) -> bool:
        """
        Import conversation memory from saved data
        
        Args:
            memory_data: Dictionary with conversation memory data
            
        Returns:
            True if import was successful, False otherwise
        """
        if not self.enable_conversation_memory:
            logger.warning("Conversation memory is not enabled")
            return False
        
        try:
            self.conversation_memory = ConversationMemory.load_from_dict(memory_data)
            logger.info("Conversation memory imported successfully")
            return True
        except Exception as e:
            logger.error(f"Error importing conversation memory: {e}")
            return False
    
    def chat(self, message: str, top_k: int = None) -> Dict[str, Any]:
        """
        Chat interface that automatically manages conversation memory
        
        Args:
            message: User message
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with response and conversation context
        """
        # If this is the first message and no conversation is started, start one
        if (self.enable_conversation_memory and 
            self.conversation_memory and 
            not self.conversation_memory.conversation_id):
            self.start_conversation()
        
        # Process the question with conversation memory
        result = self.answer_question(message, top_k=top_k, save_to_memory=True)
        
        # Add conversation context to result
        if self.enable_conversation_memory and self.conversation_memory:
            result["conversation_id"] = self.conversation_memory.conversation_id
            result["conversation_turn"] = len(self.conversation_memory.conversation_history)
        
        return result


def main():
    """Main function for testing"""
    # Use production config if available, otherwise fall back to defaults
    try:
        rag = QdrantRAG.from_production_config()
        print("Using production configuration")
        
        # Show production info if available
        opt_info = Config.get_optimization_info()
        if opt_info:
            print(f"Optimization date: {opt_info.get('tuning_date', 'N/A')}")
            print(f"Ready for production: {opt_info.get('ready_for_production', 'N/A')}")
        
        doc_metadata = Config.get_document_metadata()
        if doc_metadata:
            print(f"Document description: {doc_metadata.get('file_description', 'N/A')[:100]}...")
            
    except Exception as e:
        logger.warning(f"Failed to create from production config: {e}")
        print("Falling back to default configuration")
        rag = QdrantRAG.create_with_defaults()
    
    # Test questions
    test_questions = [
        "What is machine learning?",
        "What are the types of learning?",
        "What is PAC learning?",
        "Describe the goal of reinforcement learning."
    ]
    
    print(f"Collection stats: {rag.get_collection_stats()}")
    print("\n" + "="*50)
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        print("-" * 50)
        
        result = rag.answer_question(question)
        print(f"Answer: {result['answer']}")
        print(f"Retrieved docs: {result['retrieved_documents']}")
        print(f"Avg relevance: {result['average_relevance_score']:.3f}")


if __name__ == "__main__":
    main()

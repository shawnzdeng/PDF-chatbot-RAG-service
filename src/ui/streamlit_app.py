"""
Streamlit UI for RAG Chatbot
Provides a user-friendly interface for document-based question answering with conversation memory
"""

import streamlit as st
import logging
from typing import Dict, Any, List
import json
from datetime import datetime
import traceback
import sys
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
root_dir = src_dir.parent

sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(src_dir))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import RAG components
from rag_engine.qdrant_rag import QdrantRAG
from config import Config

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        color: #1f77b4;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
        background-color: #f0f2f6;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
    }
    .source-info {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .example-question {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.25rem;
        padding: 0.5rem;
        margin: 0.25rem 0;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    .example-question:hover {
        background-color: #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

def initialize_rag_system():
    """Initialize the RAG system with error handling"""
    try:
        if 'rag_system' not in st.session_state:
            with st.spinner("Initializing RAG system..."):
                rag = QdrantRAG.from_production_config()
                st.session_state.rag_system = rag
                st.session_state.conversation_started = False
                logger.info("RAG system initialized successfully")
        return st.session_state.rag_system
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        logger.error(f"RAG initialization error: {traceback.format_exc()}")
        return None

def get_document_metadata():
    """Get document metadata from production config"""
    try:
        metadata = Config.get_document_metadata()
        return metadata
    except Exception as e:
        logger.error(f"Error getting document metadata: {e}")
        return {}

def display_chat_message(role: str, content: str, sources: List[Dict] = None, detailed_sources: List[Dict] = None):
    """Display a chat message with sources"""
    message_class = "user-message" if role == "user" else "assistant-message"
    
    with st.container():
        st.markdown(f'<div class="chat-message {message_class}">', unsafe_allow_html=True)
        
        # Role indicator
        icon = "👤" if role == "user" else "🤖"
        st.markdown(f"**{icon} {role.title()}:**")
        
        # Message content
        st.markdown(content)
        
        # Display sources if available
        if role == "assistant" and detailed_sources:
            with st.expander(f"📚 View Sources ({len(detailed_sources)} documents)", expanded=False):
                for i, source in enumerate(detailed_sources, 1):
                    st.markdown(f"**Source {source['rank']}:** {source['source']}")
                    if source.get('page'):
                        st.markdown(f"*Page {source['page']}*")
                    st.markdown(f"**Relevance Score:** {source['relevance_score']:.3f}")
                    
                    # Show excerpt in a collapsible section
                    with st.expander(f"📄 Document Excerpt {i}", expanded=False):
                        st.markdown(f"```\n{source['excerpt']}\n```")
                    
                    if i < len(detailed_sources):
                        st.divider()
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_sidebar():
    """Display sidebar with system information and controls"""
    with st.sidebar:
        st.markdown("## 🔧 System Information")
        
        # Document metadata
        metadata = get_document_metadata()
        if metadata:
            st.markdown("### 📖 Document Overview")
            description = metadata.get('file_description', 'No description available')
            with st.expander("📝 Description", expanded=False):
                st.write(description)
        
        # Conversation controls
        st.markdown("### 💬 Conversation")
        
        if st.button("🔄 Clear Conversation", use_container_width=True):
            if 'rag_system' in st.session_state and st.session_state.rag_system:
                st.session_state.rag_system.clear_conversation_memory()
            st.session_state.messages = []
            st.session_state.conversation_started = False
            st.rerun()
        
        # Display conversation stats
        if 'rag_system' in st.session_state and st.session_state.rag_system:
            try:
                conv_stats = st.session_state.rag_system.get_conversation_history()
                if conv_stats and 'conversation_stats' in conv_stats:
                    stats = conv_stats['conversation_stats']
                    st.markdown("**Conversation Stats:**")
                    st.text(f"Turns: {stats.get('active_turns', 0)}")
                    st.text(f"Summaries: {stats.get('summarized_segments', 0)}")
                    if stats.get('conversation_id'):
                        st.text(f"ID: {stats['conversation_id'][:8]}...")
                    
                    # Show enhanced memory info
                    with st.expander("🧠 Memory Details", expanded=False):
                        if conv_stats.get('recent_turns'):
                            st.markdown("**Recent Topics:**")
                            for i, turn in enumerate(conv_stats['recent_turns'][-3:], 1):
                                if turn.get('sources'):
                                    st.text(f"Turn {i}: {len(turn['sources'])} sources")
                                if 'extracted_topics' in turn.get('metadata', {}):
                                    topics = turn['metadata']['extracted_topics'][:3]
                                    if topics:
                                        st.text(f"  Topics: {', '.join(topics)}")
            except Exception as e:
                logger.error(f"Error getting conversation stats: {e}")
        
        # System configuration
        if st.expander("⚙️ Configuration", expanded=False):
            if 'rag_system' in st.session_state and st.session_state.rag_system:
                try:
                    config = st.session_state.rag_system.get_current_config()
                    st.json({
                        "model": config.get('llm_model'),
                        "embedding": config.get('embedding_model'),
                        "temperature": config.get('temperature'),
                        "top_k": config.get('top_k'),
                        "reranking": config.get('enable_reranking'),
                        "conversation_memory": config.get('enable_conversation_memory')
                    })
                except Exception as e:
                    st.error(f"Error loading config: {e}")

def display_example_questions():
    """Display example questions from production config"""
    metadata = get_document_metadata()
    example_questions = metadata.get('example_questions', [])
    
    if example_questions:
        st.markdown("### 💡 Example Questions")
        st.markdown("*Click on any question to ask it:*")
        
        for i, question in enumerate(example_questions):
            if st.button(
                question, 
                key=f"example_q_{i}",
                use_container_width=True,
                help="Click to ask this question"
            ):
                # Add the question to chat and process it
                st.session_state.messages.append({"role": "user", "content": question})
                with st.spinner("Thinking..."):
                    try:
                        rag = st.session_state.rag_system
                        if not st.session_state.conversation_started:
                            rag.start_conversation()
                            st.session_state.conversation_started = True
                        
                        result = rag.chat(question, include_sources_in_answer=False)
                        
                        # Add assistant response
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": result['raw_answer'],
                            "sources": result.get('sources', []),
                            "detailed_sources": result.get('detailed_sources', [])
                        })
                        
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")
                        logger.error(f"Error in example question: {traceback.format_exc()}")
                
                st.rerun()

def main():
    """Main Streamlit application"""
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Page header
    st.markdown('<h1 class="main-header">🤖 RAG Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("*Ask questions about the documents and get intelligent answers with source references*")
    
    # Initialize RAG system
    rag = initialize_rag_system()
    if not rag:
        st.stop()
    
    # Sidebar
    display_sidebar()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 💬 Chat")
        
        # Chat container
        chat_container = st.container()
        
        # Display chat messages
        with chat_container:
            for message in st.session_state.messages:
                display_chat_message(
                    message["role"], 
                    message["content"],
                    sources=message.get("sources"),
                    detailed_sources=message.get("detailed_sources")
                )
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the documents..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Process with RAG system
            with st.spinner("Thinking..."):
                try:
                    # Start conversation if not already started
                    if not st.session_state.conversation_started:
                        rag.start_conversation()
                        st.session_state.conversation_started = True
                    
                    # Get response using chat method for conversation continuity
                    result = rag.chat(prompt, include_sources_in_answer=False)
                    
                    # Add assistant response
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": result['raw_answer'],
                        "sources": result.get('sources', []),
                        "detailed_sources": result.get('detailed_sources', [])
                    })
                    
                except Exception as e:
                    st.error(f"Error processing your question: {str(e)}")
                    logger.error(f"Error in chat processing: {traceback.format_exc()}")
                    
                    # Add error message
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"I'm sorry, I encountered an error while processing your question: {str(e)}",
                        "sources": [],
                        "detailed_sources": []
                    })
            
            st.rerun()
    
    with col2:
        # Example questions
        display_example_questions()
        
        # Additional info
        st.markdown("### ℹ️ How to Use")
        st.markdown("""
        1. **Ask Questions**: Type your question in the chat input
        2. **View Sources**: Click on source references to see supporting documents
        3. **Continuous Conversation**: The system remembers your conversation context
        4. **Example Questions**: Try the suggested questions on the right
        5. **Clear History**: Use the sidebar button to start fresh
        """)
        
        # Tips
        with st.expander("💡 Tips for Better Results", expanded=False):
            st.markdown("""
            - **Be specific** in your questions
            - **Reference previous discussions** - say "the first one", "that topic", etc.
            - **Ask follow-up questions** for deeper understanding
            - **Use conversation context** - the system remembers what you've discussed
            - **Check source documents** for more details
            - **Use example questions** as starting points
            - **Ask for comparisons** between previously mentioned topics
            """)
        
        # Enhanced conversation features info
        with st.expander("🔧 Enhanced Features", expanded=False):
            st.markdown("""
            **Improved Conversation Memory:**
            - ✅ Tracks key topics from each response
            - ✅ Resolves references like "the first one", "that topic"
            - ✅ Stores context and sources for better continuity
            - ✅ Enhanced prompt templates for better understanding
            - ✅ Query expansion using conversation history
            
            **Better Reference Resolution:**
            - The system now understands when you refer to previously mentioned topics
            - It can connect "give me more details about the first one" to specific concepts
            - Conversation context is preserved across multiple turns
            """)
        
        # Conversation examples
        with st.expander("💬 Conversation Examples", expanded=False):
            st.markdown("""
            **Example conversation flow:**
            
            1. **You:** "What are the different types of machine learning?"
            2. **Assistant:** Explains supervised, unsupervised, reinforcement learning
            3. **You:** "Tell me more about the first one"
            4. **Assistant:** Now understands you want details about supervised learning!
            
            **Other reference phrases that work:**
            - "Can you elaborate on the second type?"
            - "What about that algorithm you mentioned?"
            - "How does it compare to the previous method?"
            """)

if __name__ == "__main__":
    main()

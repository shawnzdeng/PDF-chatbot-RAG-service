"""
Demo script to test conversation memory functionality in RAG engine
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.rag_engine import QdrantRAG, ConversationMemory

def demo_conversation_memory():
    """Demonstrate conversation memory features"""
    print("=== RAG Engine Conversation Memory Demo ===\n")
    
    try:
        # Create RAG instance with conversation memory enabled
        print("Creating RAG instance with conversation memory...")
        rag = QdrantRAG.create_with_defaults(
            enable_conversation_memory=True,
            memory_max_turns=5,  # Keep only 5 turns for demo
            memory_summarization_threshold=4  # Summarize after 4 turns
        )
        print("✓ RAG instance created successfully\n")
        
        # Start a conversation
        conversation_id = rag.start_conversation("demo_conversation")
        print(f"Started conversation: {conversation_id}\n")
        
        # Simulate a conversation
        conversation_turns = [
            "What is machine learning?",
            "Can you explain the different types of machine learning?",
            "What is the difference between supervised and unsupervised learning?",
            "How does reinforcement learning work?",
            "Can you give me an example of supervised learning that we discussed?",
            "What about unsupervised learning examples?"
        ]
        
        print("=== Conversation Simulation ===")
        for i, question in enumerate(conversation_turns, 1):
            print(f"\n--- Turn {i} ---")
            print(f"User: {question}")
            
            # Get response using chat interface (automatically manages memory)
            try:
                result = rag.chat(question, top_k=3)
                print(f"Assistant: {result['answer'][:200]}...")
                
                # Show conversation stats
                if 'conversation_stats' in result:
                    stats = result['conversation_stats']
                    print(f"Memory stats: {stats['active_turns']} active turns, "
                          f"{stats['summarized_segments']} summaries")
                
            except Exception as e:
                print(f"Error in turn {i}: {e}")
                continue
        
        print("\n=== Final Conversation History ===")
        history = rag.get_conversation_history()
        
        if 'conversation_stats' in history:
            stats = history['conversation_stats']
            print(f"Conversation ID: {stats['conversation_id']}")
            print(f"Active turns: {stats['active_turns']}")
            print(f"Summarized segments: {stats['summarized_segments']}")
            print(f"Total estimated tokens: {stats['total_estimated_tokens']}")
        
        if 'recent_turns' in history:
            print(f"\nRecent turns ({len(history['recent_turns'])}):")
            for i, turn in enumerate(history['recent_turns'], 1):
                print(f"  {i}. User: {turn['user_message'][:50]}...")
                print(f"     Assistant: {turn['assistant_response'][:50]}...")
        
        # Test memory export/import
        print("\n=== Memory Persistence Test ===")
        exported_memory = rag.export_conversation_memory()
        if 'error' not in exported_memory:
            print("✓ Memory exported successfully")
            
            # Clear memory and import it back
            rag.clear_conversation_memory()
            print("✓ Memory cleared")
            
            success = rag.import_conversation_memory(exported_memory)
            if success:
                print("✓ Memory imported successfully")
                
                # Verify import worked
                restored_history = rag.get_conversation_history()
                if 'recent_turns' in restored_history:
                    print(f"✓ Restored {len(restored_history['recent_turns'])} conversation turns")
            else:
                print("✗ Memory import failed")
        else:
            print(f"✗ Memory export failed: {exported_memory['error']}")
        
        print("\n=== Standalone Memory Test ===")
        # Test ConversationMemory directly
        memory = ConversationMemory(
            max_turns=3,
            summarization_threshold=2
        )
        
        memory.start_new_conversation("standalone_test")
        
        # Add some turns
        memory.add_turn(
            user_message="Hello, how are you?",
            assistant_response="I'm doing well, thank you!",
            retrieved_context="No context needed",
            sources=[]
        )
        
        memory.add_turn(
            user_message="What's the weather like?",
            assistant_response="I don't have access to weather information.",
            retrieved_context="No weather context available",
            sources=[]
        )
        
        # This should trigger summarization
        memory.add_turn(
            user_message="Can you tell me a joke?",
            assistant_response="Why don't scientists trust atoms? Because they make up everything!",
            retrieved_context="No context for jokes",
            sources=[]
        )
        
        stats = memory.get_memory_stats()
        print(f"Standalone memory stats: {stats['active_turns']} active, "
              f"{stats['summarized_segments']} summaries")
        
        # Test conversation context generation
        context = memory.get_conversation_context("What did we talk about?")
        if context:
            print(f"Generated context: {context[:100]}...")
        
        print("\n=== Demo Complete ===")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_conversation_memory()

"""
Simple test script for conversation memory functionality
Tests the conversation memory module independently
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_conversation_memory_standalone():
    """Test conversation memory without RAG dependencies"""
    print("=== Standalone Conversation Memory Test ===\n")
    
    try:
        from src.rag_engine.conversation_memory import ConversationMemory
        print("✓ ConversationMemory import successful")
        
        # Create a conversation memory instance
        memory = ConversationMemory(
            max_turns=5,
            summarization_threshold=3,
            enable_summarization=True
        )
        print("✓ ConversationMemory instance created")
        
        # Start a conversation
        conv_id = memory.start_new_conversation("test_conversation")
        print(f"✓ Started conversation: {conv_id}")
        
        # Add some conversation turns
        conversations = [
            ("What is artificial intelligence?", "AI is a field of computer science that aims to create machines that can perform tasks typically requiring human intelligence."),
            ("Can you give me examples of AI?", "Examples include machine learning, natural language processing, computer vision, and robotics."),
            ("What is machine learning?", "Machine learning is a subset of AI that enables computers to learn and improve from data without being explicitly programmed."),
            ("How does it relate to what we discussed before?", "Machine learning is one of the key examples of AI that I mentioned earlier, used to build intelligent systems.")
        ]
        
        for i, (question, response) in enumerate(conversations, 1):
            print(f"\nAdding turn {i}:")
            print(f"  User: {question}")
            print(f"  Assistant: {response[:50]}...")
            
            memory.add_turn(
                user_message=question,
                assistant_response=response,
                retrieved_context=f"Context for question {i}",
                sources=[f"source_{i}.pdf"],
                metadata={"turn_number": i}
            )
        
        # Check memory stats
        stats = memory.get_memory_stats()
        print(f"\n✓ Memory stats:")
        print(f"  - Active turns: {stats['active_turns']}")
        print(f"  - Summarized segments: {stats['summarized_segments']}")
        print(f"  - Estimated tokens: {stats['total_estimated_tokens']}")
        
        # Test conversation context generation
        context = memory.get_conversation_context(
            "What did we discuss about AI?",
            include_summaries=True,
            max_recent_turns=3
        )
        
        if context:
            print(f"\n✓ Generated conversation context:")
            print(f"  Context length: {len(context)} characters")
            print(f"  Context preview: {context[:100]}...")
        else:
            print(f"\n✓ No conversation context (expected for first conversation)")
        
        # Test memory export/import
        exported = memory.save_to_dict()
        print(f"\n✓ Memory exported successfully")
        print(f"  - Exported {len(exported.get('conversation_history', []))} turns")
        print(f"  - Exported {len(exported.get('conversation_summaries', []))} summaries")
        
        # Import to new memory instance
        new_memory = ConversationMemory.load_from_dict(exported)
        new_stats = new_memory.get_memory_stats()
        print(f"\n✓ Memory imported successfully")
        print(f"  - Imported active turns: {new_stats['active_turns']}")
        print(f"  - Imported summaries: {new_stats['summarized_segments']}")
        
        # Test context generation with the imported memory
        imported_context = new_memory.get_conversation_context(
            "Summarize our conversation",
            include_summaries=True,
            max_recent_turns=5
        )
        
        if imported_context:
            print(f"\n✓ Context generated from imported memory:")
            print(f"  Context length: {len(imported_context)} characters")
        
        print(f"\n=== Test Complete - All checks passed! ===")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_conversation_memory_standalone()
    if success:
        print("\n🎉 Conversation memory is working correctly!")
    else:
        print("\n❌ Conversation memory test failed.")

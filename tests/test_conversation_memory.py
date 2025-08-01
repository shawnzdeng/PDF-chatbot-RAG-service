"""
Comprehensive test suite for conversation memory functionality in RAG engine
Tests both the ConversationMemory class and RAG integration
"""

import sys
import unittest
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.rag_engine.conversation_memory import ConversationMemory, ConversationTurn, ConversationSummary

class TestConversationMemory(unittest.TestCase):
    """Test cases for ConversationMemory class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.memory = ConversationMemory(
            max_turns=5,
            summarization_threshold=3,
            enable_summarization=True,
            memory_decay_hours=24
        )
    
    def test_conversation_memory_initialization(self):
        """Test ConversationMemory initialization"""
        self.assertEqual(self.memory.max_turns, 5)
        self.assertEqual(self.memory.summarization_threshold, 3)
        self.assertTrue(self.memory.enable_summarization)
        self.assertEqual(self.memory.memory_decay_hours, 24)
        self.assertEqual(len(self.memory.conversation_history), 0)
        self.assertEqual(len(self.memory.conversation_summaries), 0)
        self.assertIsNone(self.memory.conversation_id)
    
    def test_start_new_conversation(self):
        """Test starting a new conversation"""
        # Test with auto-generated ID
        conv_id = self.memory.start_new_conversation()
        self.assertIsNotNone(conv_id)
        self.assertTrue(conv_id.startswith("conv_"))
        self.assertEqual(self.memory.conversation_id, conv_id)
        self.assertIsNotNone(self.memory.conversation_start_time)
        
        # Test with custom ID
        custom_id = "test_conversation_123"
        conv_id2 = self.memory.start_new_conversation(custom_id)
        self.assertEqual(conv_id2, custom_id)
        self.assertEqual(self.memory.conversation_id, custom_id)
    
    def test_add_turn_basic(self):
        """Test adding conversation turns"""
        self.memory.start_new_conversation("test_conv")
        
        # Add first turn with explicit storage enabled
        self.memory.add_turn(
            user_message="Hello, how are you?",
            assistant_response="I'm doing well, thank you!",
            retrieved_context="No context needed",
            sources=["test_source.pdf"],
            metadata={"test": "value"},
            store_sources_in_memory=True,
            store_context_in_memory=True
        )
        
        self.assertEqual(len(self.memory.conversation_history), 1)
        turn = self.memory.conversation_history[0]
        self.assertEqual(turn.user_message, "Hello, how are you?")
        self.assertEqual(turn.assistant_response, "I'm doing well, thank you!")
        self.assertEqual(turn.retrieved_context, "No context needed")
        self.assertEqual(turn.sources, ["test_source.pdf"])
        self.assertEqual(turn.metadata, {"test": "value"})
        self.assertIsInstance(turn.timestamp, datetime)
    
    def test_add_turn_with_storage_options(self):
        """Test adding turns with source/context storage options"""
        self.memory.start_new_conversation("test_conv")
        
        # Add turn without storing sources/context
        self.memory.add_turn(
            user_message="Test question",
            assistant_response="Test answer",
            retrieved_context="Long context that shouldn't be stored",
            sources=["doc1.pdf", "doc2.pdf"],
            store_sources_in_memory=False,
            store_context_in_memory=False
        )
        
        turn = self.memory.conversation_history[0]
        self.assertEqual(turn.sources, [])  # Sources not stored
        self.assertEqual(turn.retrieved_context, "")  # Context not stored
        
        # Add turn with storing sources/context
        self.memory.add_turn(
            user_message="Test question 2",
            assistant_response="Test answer 2",
            retrieved_context="Context to be stored",
            sources=["doc3.pdf"],
            store_sources_in_memory=True,
            store_context_in_memory=True
        )
        
        turn2 = self.memory.conversation_history[1]
        self.assertEqual(turn2.sources, ["doc3.pdf"])  # Sources stored
        self.assertEqual(turn2.retrieved_context, "Context to be stored")  # Context stored
    
    def test_conversation_context_generation(self):
        """Test conversation context generation"""
        self.memory.start_new_conversation("test_conv")
        
        # Add some turns with storage enabled to test context generation
        self.memory.add_turn(
            "What is AI?", 
            "AI is artificial intelligence.", 
            "AI context",
            store_context_in_memory=True
        )
        self.memory.add_turn(
            "Tell me more", 
            "AI involves machine learning.", 
            "ML context",
            store_context_in_memory=True
        )
        
        # Get conversation context
        context = self.memory.get_conversation_context(
            "How does it work?",
            include_summaries=True,
            max_recent_turns=2
        )
        
        self.assertIn("What is AI?", context)
        self.assertIn("AI is artificial intelligence", context)
        self.assertIn("Tell me more", context)
    
    def test_conversation_context_empty(self):
        """Test conversation context when no history exists"""
        context = self.memory.get_conversation_context("Test question")
        self.assertEqual(context, "")
    
    def test_has_conversation_history(self):
        """Test checking if conversation history exists"""
        # Initially no history
        self.assertFalse(self.memory.has_conversation_history())
        
        # Add a turn
        self.memory.start_new_conversation()
        self.memory.add_turn("Test", "Response", "", store_context_in_memory=True)
        self.assertTrue(self.memory.has_conversation_history())
        
        # Add a summary
        self.memory.conversation_summaries.append(
            ConversationSummary("Test summary", 1, (datetime.now(), datetime.now()))
        )
        
        # Clear history but keep summaries
        self.memory.conversation_history.clear()
        self.assertTrue(self.memory.has_conversation_history())  # Still true due to summaries
    
    def test_memory_stats(self):
        """Test memory statistics generation"""
        self.memory.start_new_conversation("test_conv")
        self.memory.add_turn(
            "Question 1", 
            "Answer 1", 
            "Context 1",
            store_context_in_memory=True
        )
        self.memory.add_turn(
            "Question 2", 
            "Answer 2", 
            "Context 2",
            store_context_in_memory=True
        )
        
        stats = self.memory.get_memory_stats()
        
        self.assertEqual(stats["conversation_id"], "test_conv")
        self.assertEqual(stats["active_turns"], 2)
        self.assertEqual(stats["summarized_segments"], 0)
        self.assertGreater(stats["total_estimated_tokens"], 0)
        self.assertGreaterEqual(stats["memory_age_hours"], 0)
    
    def test_clear_memory(self):
        """Test clearing conversation memory"""
        self.memory.start_new_conversation("test_conv")
        self.memory.add_turn("Test", "Response", "Context", store_context_in_memory=True)
        
        # Verify data exists
        self.assertEqual(len(self.memory.conversation_history), 1)
        self.assertIsNotNone(self.memory.conversation_id)
        
        # Clear memory
        self.memory.clear_memory()
        
        # Verify data is cleared
        self.assertEqual(len(self.memory.conversation_history), 0)
        self.assertEqual(len(self.memory.conversation_summaries), 0)
        self.assertIsNone(self.memory.conversation_id)
        self.assertIsNone(self.memory.conversation_start_time)
    
    def test_memory_export_import(self):
        """Test memory export and import functionality"""
        # Set up memory with data
        self.memory.start_new_conversation("export_test")
        self.memory.add_turn(
            "Question 1", 
            "Answer 1", 
            "Context 1", 
            ["source1.pdf"],
            store_sources_in_memory=True,
            store_context_in_memory=True
        )
        self.memory.add_turn(
            "Question 2", 
            "Answer 2", 
            "Context 2", 
            ["source2.pdf"],
            store_sources_in_memory=True,
            store_context_in_memory=True
        )
        
        # Export memory
        exported_data = self.memory.save_to_dict()
        
        # Verify export structure
        self.assertIn("conversation_id", exported_data)
        self.assertIn("conversation_history", exported_data)
        self.assertIn("config", exported_data)
        self.assertEqual(len(exported_data["conversation_history"]), 2)
        
        # Create new memory and import
        new_memory = ConversationMemory.load_from_dict(exported_data)
        
        # Verify import
        self.assertEqual(new_memory.conversation_id, "export_test")
        self.assertEqual(len(new_memory.conversation_history), 2)
        self.assertEqual(new_memory.max_turns, self.memory.max_turns)
        
        # Verify turn data
        imported_turn = new_memory.conversation_history[0]
        self.assertEqual(imported_turn.user_message, "Question 1")
        self.assertEqual(imported_turn.assistant_response, "Answer 1")


class TestRAGConversationIntegration(unittest.TestCase):
    """Test cases for RAG engine conversation memory integration"""
    
    def setUp(self):
        """Set up test fixtures with mocked dependencies"""
        self.mock_config_patcher = patch('src.rag_engine.qdrant_rag.Config')
        self.mock_config = self.mock_config_patcher.start()
        
        # Mock config values
        self.mock_config.validate_env_vars.return_value = None
        self.mock_config.OPENAI_API_KEY = "test_key"
        self.mock_config.QDRANT_URL = "http://test_url"
        self.mock_config.QDRANT_API_KEY = "test_qdrant_key"
        
        # Mock embedding and LLM
        self.embeddings_patcher = patch('src.rag_engine.qdrant_rag.OpenAIEmbeddings')
        self.llm_patcher = patch('src.rag_engine.qdrant_rag.ChatOpenAI')
        self.qdrant_patcher = patch('src.rag_engine.qdrant_rag.QdrantClient')
        
        self.mock_embeddings = self.embeddings_patcher.start()
        self.mock_llm = self.llm_patcher.start()
        self.mock_qdrant = self.qdrant_patcher.start()
        
        # Mock embedding and LLM responses
        self.mock_embeddings.return_value.embed_query.return_value = [0.1] * 1536
        self.mock_llm.return_value = Mock()
        
        # Import after patching
        from src.rag_engine.qdrant_rag import QdrantRAG
        self.QdrantRAG = QdrantRAG
    
    def tearDown(self):
        """Clean up patches"""
        self.mock_config_patcher.stop()
        self.embeddings_patcher.stop()
        self.llm_patcher.stop()
        self.qdrant_patcher.stop()
    
    def test_rag_initialization_with_conversation_memory(self):
        """Test RAG initialization with conversation memory enabled"""
        rag = self.QdrantRAG(
            collection_name="test_collection",
            embedding_model="text-embedding-3-large",
            llm_model="gpt-4o",
            temperature=0.0,
            prompt_template="Test template: {context}\nQuestion: {question}\nAnswer:",
            enable_conversation_memory=True,
            memory_max_turns=5,
            memory_summarization_threshold=3
        )
        
        self.assertTrue(rag.enable_conversation_memory)
        self.assertIsNotNone(rag.conversation_memory)
        self.assertEqual(rag.memory_max_turns, 5)
        self.assertEqual(rag.memory_summarization_threshold, 3)
        self.assertEqual(rag.conversation_memory.max_turns, 5)
        self.assertEqual(rag.conversation_memory.summarization_threshold, 3)
    
    def test_rag_initialization_without_conversation_memory(self):
        """Test RAG initialization with conversation memory disabled"""
        rag = self.QdrantRAG(
            collection_name="test_collection",
            embedding_model="text-embedding-3-large",
            llm_model="gpt-4o",
            temperature=0.0,
            prompt_template="Test template: {context}\nQuestion: {question}\nAnswer:",
            enable_conversation_memory=False
        )
        
        self.assertFalse(rag.enable_conversation_memory)
        self.assertIsNone(rag.conversation_memory)
    
    @patch('src.rag_engine.qdrant_rag.QdrantRAG.retrieve_documents')
    def test_answer_question_with_sources(self, mock_retrieve):
        """Test answer_question with source formatting"""
        # Setup RAG
        rag = self.QdrantRAG(
            collection_name="test_collection",
            embedding_model="text-embedding-3-large",
            llm_model="gpt-4o",
            temperature=0.0,
            prompt_template="Test template: {context}\nQuestion: {question}\nAnswer:",
            enable_conversation_memory=True
        )
        
        # Mock retrieved documents
        from src.rag_engine.qdrant_rag import RetrievalResult
        mock_docs = [
            RetrievalResult(
                content="Machine learning is a subset of AI.",
                score=0.95,
                metadata={"source": "ml_guide.pdf", "page": 5}
            ),
            RetrievalResult(
                content="Deep learning uses neural networks.",
                score=0.87,
                metadata={"source": "dl_book.pdf", "page": 12}
            )
        ]
        mock_retrieve.return_value = mock_docs
        
        # Mock LLM response
        with patch.object(rag, 'generate_answer') as mock_generate:
            mock_generate.return_value = "Machine learning is a type of artificial intelligence."
            
            # Test with sources in answer
            result = rag.answer_question(
                "What is machine learning?", 
                include_sources_in_answer=True,
                save_to_memory=False  # Disable to avoid memory interactions in this test
            )
            
            # Verify response structure
            self.assertIn("question", result)
            self.assertIn("answer", result)
            self.assertIn("raw_answer", result)
            self.assertIn("sources", result)
            self.assertIn("detailed_sources", result)
            self.assertIn("source_references", result)
            
            # Verify sources are included in formatted answer
            self.assertIn("**Sources:**", result["answer"])
            self.assertIn("ml_guide.pdf", result["answer"])
            self.assertIn("Page 5", result["answer"])
            
            # Verify raw answer doesn't have sources
            self.assertNotIn("**Sources:**", result["raw_answer"])
            
            # Verify detailed sources
            self.assertEqual(len(result["detailed_sources"]), 2)
            self.assertEqual(result["detailed_sources"][0]["source"], "ml_guide.pdf")
            self.assertEqual(result["detailed_sources"][0]["page"], 5)
            self.assertEqual(result["detailed_sources"][0]["rank"], 1)
            
            # Verify source references
            self.assertEqual(len(result["source_references"]), 2)
            self.assertEqual(result["source_references"][0], "[1] ml_guide.pdf (p. 5)")
            self.assertEqual(result["source_references"][1], "[2] dl_book.pdf (p. 12)")
    
    @patch('src.rag_engine.qdrant_rag.QdrantRAG.retrieve_documents')
    def test_conversation_flow(self, mock_retrieve):
        """Test full conversation flow with memory"""
        # Setup RAG
        rag = self.QdrantRAG(
            collection_name="test_collection",
            embedding_model="text-embedding-3-large",
            llm_model="gpt-4o",
            temperature=0.0,
            prompt_template="Test template: {context}\nQuestion: {question}\nAnswer:",
            enable_conversation_memory=True,
            memory_max_turns=3
        )
        
        # Mock retrieved documents (minimal for this test)
        from src.rag_engine.qdrant_rag import RetrievalResult
        mock_retrieve.return_value = [
            RetrievalResult("Test content", 0.9, {"source": "test.pdf"})
        ]
        
        # Mock LLM responses
        responses = [
            "AI stands for Artificial Intelligence.",
            "Machine learning is a subset of AI that enables computers to learn.",
            "Yes, as I mentioned, machine learning is part of AI."
        ]
        
        with patch.object(rag, 'generate_answer') as mock_generate:
            mock_generate.side_effect = responses
            
            # Start conversation
            conv_id = rag.start_conversation("test_conversation")
            self.assertEqual(conv_id, "test_conversation")
            
            # First question
            result1 = rag.chat("What is AI?")
            self.assertIn("conversation_id", result1)
            self.assertIn("conversation_turn", result1)
            self.assertEqual(result1["conversation_turn"], 1)
            
            # Second question
            result2 = rag.chat("What is machine learning?")
            self.assertEqual(result2["conversation_turn"], 2)
            
            # Third question with reference to previous context
            result3 = rag.chat("How does it relate to what we discussed?")
            self.assertEqual(result3["conversation_turn"], 3)
            
            # Verify conversation memory
            history = rag.get_conversation_history()
            self.assertEqual(len(history["recent_turns"]), 3)
            self.assertEqual(history["conversation_stats"]["active_turns"], 3)
    
    def test_conversation_memory_management_methods(self):
        """Test conversation memory management methods"""
        rag = self.QdrantRAG(
            collection_name="test_collection",
            embedding_model="text-embedding-3-large",
            llm_model="gpt-4o",
            temperature=0.0,
            prompt_template="Test template: {context}\nQuestion: {question}\nAnswer:",
            enable_conversation_memory=True
        )
        
        # Test start conversation
        conv_id = rag.start_conversation()
        self.assertIsNotNone(conv_id)
        
        # Test get conversation history (empty)
        history = rag.get_conversation_history()
        self.assertEqual(history["conversation_stats"]["active_turns"], 0)
        
        # Test export conversation memory (empty)
        exported = rag.export_conversation_memory()
        self.assertNotIn("error", exported)
        
        # Test clear conversation memory
        rag.clear_conversation_memory()
        
        # Test import conversation memory
        success = rag.import_conversation_memory(exported)
        self.assertTrue(success)
    
    def test_conversation_memory_disabled_methods(self):
        """Test conversation memory methods when disabled"""
        rag = self.QdrantRAG(
            collection_name="test_collection",
            embedding_model="text-embedding-3-large",
            llm_model="gpt-4o",
            temperature=0.0,
            prompt_template="Test template: {context}\nQuestion: {question}\nAnswer:",
            enable_conversation_memory=False
        )
        
        # Test methods return appropriate responses when memory is disabled
        self.assertIsNone(rag.start_conversation())
        
        history = rag.get_conversation_history()
        self.assertIn("error", history)
        
        exported = rag.export_conversation_memory()
        self.assertIn("error", exported)
        
        success = rag.import_conversation_memory({})
        self.assertFalse(success)


class TestConversationMemoryEdgeCases(unittest.TestCase):
    """Test edge cases and error scenarios"""
    
    def test_conversation_turn_creation(self):
        """Test ConversationTurn data class"""
        turn = ConversationTurn(
            user_message="Test message",
            assistant_response="Test response",
            retrieved_context="Test context",
            timestamp=datetime.now(),
            sources=["test.pdf"],
            metadata={"key": "value"}
        )
        
        self.assertEqual(turn.user_message, "Test message")
        self.assertEqual(turn.assistant_response, "Test response")
        self.assertEqual(turn.sources, ["test.pdf"])
        self.assertEqual(turn.metadata, {"key": "value"})
    
    def test_conversation_summary_creation(self):
        """Test ConversationSummary data class"""
        now = datetime.now()
        summary = ConversationSummary(
            summary_text="Test summary",
            turn_count=5,
            time_range=(now, now + timedelta(hours=1)),
            key_topics=["AI", "ML"]
        )
        
        self.assertEqual(summary.summary_text, "Test summary")
        self.assertEqual(summary.turn_count, 5)
        self.assertEqual(summary.key_topics, ["AI", "ML"])
    
    def test_memory_with_max_turns_limit(self):
        """Test memory behavior when max_turns limit is reached"""
        memory = ConversationMemory(max_turns=2, summarization_threshold=5)
        memory.start_new_conversation()
        
        # Add turns beyond max_turns (with explicit storage for testing)
        memory.add_turn("Q1", "A1", "C1", store_context_in_memory=True)
        memory.add_turn("Q2", "A2", "C2", store_context_in_memory=True)
        memory.add_turn("Q3", "A3", "C3", store_context_in_memory=True)  # Should evict oldest
        
        # Should only keep the last 2 turns
        self.assertEqual(len(memory.conversation_history), 2)
        self.assertEqual(memory.conversation_history[0].user_message, "Q2")
        self.assertEqual(memory.conversation_history[1].user_message, "Q3")
    
    def test_memory_token_estimation(self):
        """Test token estimation functionality"""
        memory = ConversationMemory()
        memory.start_new_conversation()
        
        # Add a turn with known content (with storage for testing)
        long_message = "This is a test message. " * 100  # ~500 chars
        memory.add_turn(long_message, "Short response", "", store_context_in_memory=True)
        
        stats = memory.get_memory_stats()
        estimated_tokens = stats["total_estimated_tokens"]
        
        # Should be roughly chars/4 (rough approximation)
        expected_tokens = len(long_message + "Short response") // 4
        self.assertGreater(estimated_tokens, 0)
        self.assertLessEqual(abs(estimated_tokens - expected_tokens), 50)  # Allow some variance


def run_tests():
    """Run all tests and return results"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestConversationMemory,
        TestRAGConversationIntegration,
        TestConversationMemoryEdgeCases
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("Running Conversation Memory Test Suite")
    print("=" * 60)
    
    # Run the tests
    result = run_tests()
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    if result.wasSuccessful():
        print("\n🎉 All tests passed! Conversation memory is working correctly.")
        exit(0)
    else:
        print("\n❌ Some tests failed. Please review the issues above.")
        exit(1)

"""
Conversation memory management for RAG engine
Handles conversation history, context management, and memory optimization
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation"""
    user_message: str
    assistant_response: str
    retrieved_context: str
    timestamp: datetime
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConversationSummary:
    """Represents a summarized portion of conversation history"""
    summary_text: str
    turn_count: int
    time_range: Tuple[datetime, datetime]
    key_topics: List[str] = field(default_factory=list)

class ConversationMemory:
    """
    Manages conversation history and memory for continuous RAG conversations
    
    Features:
    - Maintains conversation history with context
    - Automatic memory summarization when history gets too long
    - Context-aware prompt enhancement
    - Memory persistence and retrieval
    """
    
    def __init__(self,
                 max_turns: int = 10,
                 max_tokens_estimate: int = 3000,
                 summarization_threshold: int = 8,
                 enable_summarization: bool = True,
                 memory_decay_hours: int = 24):
        """
        Initialize conversation memory
        
        Args:
            max_turns: Maximum number of conversation turns to keep in active memory
            max_tokens_estimate: Estimated maximum tokens to include in context
            summarization_threshold: Number of turns before triggering summarization
            enable_summarization: Whether to enable automatic summarization
            memory_decay_hours: Hours after which old memories can be purged
        """
        self.max_turns = max_turns
        self.max_tokens_estimate = max_tokens_estimate
        self.summarization_threshold = summarization_threshold
        self.enable_summarization = enable_summarization
        self.memory_decay_hours = memory_decay_hours
        
        # Active conversation history (recent turns)
        self.conversation_history: deque = deque(maxlen=max_turns)
        
        # Summarized conversation segments
        self.conversation_summaries: List[ConversationSummary] = []
        
        # Current conversation metadata
        self.conversation_id: Optional[str] = None
        self.conversation_start_time: Optional[datetime] = None
        
        logger.info(f"Initialized ConversationMemory with max_turns={max_turns}, "
                   f"summarization_threshold={summarization_threshold}")
    
    def start_new_conversation(self, conversation_id: str = None) -> str:
        """
        Start a new conversation session
        
        Args:
            conversation_id: Optional custom conversation ID
            
        Returns:
            Conversation ID
        """
        if conversation_id is None:
            conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.conversation_id = conversation_id
        self.conversation_start_time = datetime.now()
        self.conversation_history.clear()
        
        logger.info(f"Started new conversation: {conversation_id}")
        return conversation_id
    
    def add_turn(self,
                 user_message: str,
                 assistant_response: str,
                 retrieved_context: str,
                 sources: List[str] = None,
                 metadata: Dict[str, Any] = None) -> None:
        """
        Add a conversation turn to memory
        
        Args:
            user_message: User's input message
            assistant_response: Assistant's response
            retrieved_context: Retrieved RAG context
            sources: List of source documents
            metadata: Additional metadata
        """
        turn = ConversationTurn(
            user_message=user_message,
            assistant_response=assistant_response,
            retrieved_context=retrieved_context,
            timestamp=datetime.now(),
            sources=sources or [],
            metadata=metadata or {}
        )
        
        self.conversation_history.append(turn)
        
        # Check if summarization is needed
        if (self.enable_summarization and 
            len(self.conversation_history) >= self.summarization_threshold):
            self._maybe_summarize_history()
        
        logger.debug(f"Added conversation turn. History length: {len(self.conversation_history)}")
    
    def get_conversation_context(self, 
                               current_question: str,
                               include_summaries: bool = True,
                               max_recent_turns: int = None) -> str:
        """
        Get conversation context for the current question
        
        Args:
            current_question: Current user question
            include_summaries: Whether to include conversation summaries
            max_recent_turns: Maximum number of recent turns to include
            
        Returns:
            Formatted conversation context
        """
        context_parts = []
        
        # Add conversation summaries if enabled
        if include_summaries and self.conversation_summaries:
            summary_text = self._format_summaries()
            if summary_text:
                context_parts.append(f"Previous conversation summary:\n{summary_text}")
        
        # Add recent conversation history
        recent_turns = max_recent_turns or min(len(self.conversation_history), 5)
        if recent_turns > 0 and self.conversation_history:
            recent_history = self._format_recent_history(recent_turns)
            if recent_history:
                context_parts.append(f"Recent conversation:\n{recent_history}")
        
        return "\n\n".join(context_parts) if context_parts else ""
    
    def get_enhanced_prompt_template(self, base_template: str) -> str:
        """
        Enhance the base prompt template with conversation awareness
        
        Args:
            base_template: Base RAG prompt template
            
        Returns:
            Enhanced prompt template with conversation context
        """
        if not self.has_conversation_history():
            return base_template
        
        # Insert conversation context into the template
        conversation_section = """
Previous conversation context:
{conversation_context}

"""
        
        # Find a good place to insert conversation context
        # Look for common patterns in RAG templates
        insertion_patterns = [
            "Context:",
            "Use the following context",
            "Based on the context",
            "Given the context"
        ]
        
        enhanced_template = base_template
        for pattern in insertion_patterns:
            if pattern in base_template.lower():
                # Insert conversation context before the main context
                enhanced_template = base_template.replace(
                    pattern, 
                    conversation_section + pattern,
                    1  # Only replace first occurrence
                )
                break
        
        # If no pattern found, prepend conversation context
        if enhanced_template == base_template:
            enhanced_template = conversation_section + base_template
        
        return enhanced_template
    
    def has_conversation_history(self) -> bool:
        """Check if there's any conversation history"""
        return len(self.conversation_history) > 0 or len(self.conversation_summaries) > 0
    
    def clear_memory(self) -> None:
        """Clear all conversation memory"""
        self.conversation_history.clear()
        self.conversation_summaries.clear()
        self.conversation_id = None
        self.conversation_start_time = None
        logger.info("Cleared conversation memory")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about current memory usage"""
        return {
            "conversation_id": self.conversation_id,
            "conversation_start_time": self.conversation_start_time.isoformat() if self.conversation_start_time else None,
            "active_turns": len(self.conversation_history),
            "summarized_segments": len(self.conversation_summaries),
            "total_estimated_tokens": self._estimate_memory_tokens(),
            "memory_age_hours": self._get_memory_age_hours()
        }
    
    def _format_recent_history(self, max_turns: int) -> str:
        """Format recent conversation history"""
        if not self.conversation_history:
            return ""
        
        recent_turns = list(self.conversation_history)[-max_turns:]
        formatted_turns = []
        
        for i, turn in enumerate(recent_turns, 1):
            formatted_turn = f"Turn {i}:\nUser: {turn.user_message}\nAssistant: {turn.assistant_response}"
            formatted_turns.append(formatted_turn)
        
        return "\n\n".join(formatted_turns)
    
    def _format_summaries(self) -> str:
        """Format conversation summaries"""
        if not self.conversation_summaries:
            return ""
        
        summary_parts = []
        for i, summary in enumerate(self.conversation_summaries, 1):
            summary_text = f"Summary {i}: {summary.summary_text}"
            if summary.key_topics:
                summary_text += f" (Topics: {', '.join(summary.key_topics)})"
            summary_parts.append(summary_text)
        
        return "\n".join(summary_parts)
    
    def _maybe_summarize_history(self) -> None:
        """
        Check if history should be summarized and perform summarization
        """
        if len(self.conversation_history) < self.summarization_threshold:
            return
        
        try:
            # For now, create a simple summary
            # In a production system, you might use an LLM for better summarization
            turns_to_summarize = list(self.conversation_history)[:self.summarization_threshold//2]
            
            if turns_to_summarize:
                summary_text = self._create_simple_summary(turns_to_summarize)
                time_range = (turns_to_summarize[0].timestamp, turns_to_summarize[-1].timestamp)
                
                summary = ConversationSummary(
                    summary_text=summary_text,
                    turn_count=len(turns_to_summarize),
                    time_range=time_range,
                    key_topics=self._extract_key_topics(turns_to_summarize)
                )
                
                self.conversation_summaries.append(summary)
                
                # Remove summarized turns from active history
                for _ in range(len(turns_to_summarize)):
                    if self.conversation_history:
                        self.conversation_history.popleft()
                
                logger.info(f"Summarized {len(turns_to_summarize)} conversation turns")
        
        except Exception as e:
            logger.error(f"Error during conversation summarization: {e}")
    
    def _create_simple_summary(self, turns: List[ConversationTurn]) -> str:
        """Create a simple summary of conversation turns"""
        if not turns:
            return ""
        
        # Extract main topics and questions
        questions = [turn.user_message for turn in turns]
        
        # Create a simple summary
        summary = f"Discussed {len(turns)} topics including: "
        summary += ", ".join(questions[:3])  # First 3 questions
        
        if len(questions) > 3:
            summary += f" and {len(questions) - 3} other topics"
        
        return summary
    
    def _extract_key_topics(self, turns: List[ConversationTurn]) -> List[str]:
        """Extract key topics from conversation turns"""
        # Simple keyword extraction
        # In production, you might use NLP techniques
        topics = set()
        
        for turn in turns:
            # Extract potential topics from user messages
            words = turn.user_message.lower().split()
            # Filter out common words and extract potential topics
            potential_topics = [word for word in words 
                              if len(word) > 4 and word.isalpha()]
            topics.update(potential_topics[:2])  # Take first 2 potential topics per turn
        
        return list(topics)[:5]  # Return up to 5 key topics
    
    def _estimate_memory_tokens(self) -> int:
        """Estimate total tokens in memory (rough approximation)"""
        total_chars = 0
        
        # Count characters in active history
        for turn in self.conversation_history:
            total_chars += len(turn.user_message) + len(turn.assistant_response)
        
        # Count characters in summaries
        for summary in self.conversation_summaries:
            total_chars += len(summary.summary_text)
        
        # Rough approximation: 4 characters per token
        return total_chars // 4
    
    def _get_memory_age_hours(self) -> float:
        """Get age of oldest memory in hours"""
        if not self.conversation_start_time:
            return 0.0
        
        return (datetime.now() - self.conversation_start_time).total_seconds() / 3600
    
    def save_to_dict(self) -> Dict[str, Any]:
        """Save conversation memory to dictionary for persistence"""
        return {
            "conversation_id": self.conversation_id,
            "conversation_start_time": self.conversation_start_time.isoformat() if self.conversation_start_time else None,
            "conversation_history": [
                {
                    "user_message": turn.user_message,
                    "assistant_response": turn.assistant_response,
                    "retrieved_context": turn.retrieved_context,
                    "timestamp": turn.timestamp.isoformat(),
                    "sources": turn.sources,
                    "metadata": turn.metadata
                }
                for turn in self.conversation_history
            ],
            "conversation_summaries": [
                {
                    "summary_text": summary.summary_text,
                    "turn_count": summary.turn_count,
                    "time_range": [summary.time_range[0].isoformat(), summary.time_range[1].isoformat()],
                    "key_topics": summary.key_topics
                }
                for summary in self.conversation_summaries
            ],
            "config": {
                "max_turns": self.max_turns,
                "max_tokens_estimate": self.max_tokens_estimate,
                "summarization_threshold": self.summarization_threshold,
                "enable_summarization": self.enable_summarization,
                "memory_decay_hours": self.memory_decay_hours
            }
        }
    
    @classmethod
    def load_from_dict(cls, data: Dict[str, Any]) -> 'ConversationMemory':
        """Load conversation memory from dictionary"""
        config = data.get("config", {})
        memory = cls(
            max_turns=config.get("max_turns", 10),
            max_tokens_estimate=config.get("max_tokens_estimate", 3000),
            summarization_threshold=config.get("summarization_threshold", 8),
            enable_summarization=config.get("enable_summarization", True),
            memory_decay_hours=config.get("memory_decay_hours", 24)
        )
        
        # Load conversation metadata
        memory.conversation_id = data.get("conversation_id")
        if data.get("conversation_start_time"):
            memory.conversation_start_time = datetime.fromisoformat(data["conversation_start_time"])
        
        # Load conversation history
        for turn_data in data.get("conversation_history", []):
            turn = ConversationTurn(
                user_message=turn_data["user_message"],
                assistant_response=turn_data["assistant_response"],
                retrieved_context=turn_data["retrieved_context"],
                timestamp=datetime.fromisoformat(turn_data["timestamp"]),
                sources=turn_data.get("sources", []),
                metadata=turn_data.get("metadata", {})
            )
            memory.conversation_history.append(turn)
        
        # Load conversation summaries
        for summary_data in data.get("conversation_summaries", []):
            time_range = tuple(datetime.fromisoformat(t) for t in summary_data["time_range"])
            summary = ConversationSummary(
                summary_text=summary_data["summary_text"],
                turn_count=summary_data["turn_count"],
                time_range=time_range,
                key_topics=summary_data.get("key_topics", [])
            )
            memory.conversation_summaries.append(summary)
        
        return memory

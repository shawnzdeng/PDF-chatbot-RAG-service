# Conversation Memory for RAG Engine

## Overview

The RAG engine now includes sophisticated conversation memory capabilities that enable continuous, context-aware conversations while maintaining efficient memory usage.

## Key Features

### 1. **Conversation Memory Management**
- **Automatic Memory**: Tracks conversation history with configurable turn limits
- **Smart Summarization**: Automatically summarizes older conversations when memory limits are reached
- **Memory Persistence**: Export/import conversation memory for session restoration
- **Lean Storage**: Sources and retrieved context are excluded from memory by default to keep it efficient

### 2. **Source Document Handling**
- **Always Return Sources**: Source documents are always included in responses with detailed information
- **Multiple Source Formats**: 
  - Simple source list (backward compatibility)
  - Detailed sources with excerpts and relevance scores
  - Formatted source references for easy display
- **Memory Efficiency**: Sources are not stored in conversation memory to keep it lean
- **Integrated Display**: Sources can be automatically appended to answers

### 3. **Enhanced Response Format**
```python
{
    "question": "user question",
    "answer": "response with source references appended",
    "raw_answer": "response without source references", 
    "sources": ["simple list for backward compatibility"],
    "detailed_sources": [
        {
            "rank": 1,
            "source": "document.pdf",
            "page": 5,
            "relevance_score": 0.85,
            "excerpt": "document excerpt..."
        }
    ],
    "source_references": ["[1] document.pdf (p. 5)", "[2] another.pdf (p. 3)"],
    "conversation_stats": {
        "conversation_id": "conv_20250801_123456",
        "active_turns": 3,
        "summarized_segments": 1
    }
}
```

## Configuration Options

### RAG Engine Initialization
```python
rag = QdrantRAG(
    # ... existing parameters ...
    enable_conversation_memory=True,        # Enable conversation memory
    memory_max_turns=10,                   # Keep 10 recent turns in active memory
    memory_summarization_threshold=8       # Summarize after 8 turns
)
```

### ConversationMemory Settings
```python
memory = ConversationMemory(
    max_turns=10,                          # Maximum active conversation turns
    summarization_threshold=8,             # Turns before summarization
    enable_summarization=True,             # Enable automatic summarization
    memory_decay_hours=24                  # Hours before old memories can be purged
)
```

## Usage Examples

### Basic Chat with Memory
```python
# Initialize RAG with conversation memory
rag = QdrantRAG.create_with_defaults(enable_conversation_memory=True)

# Start conversation (automatic)
result1 = rag.chat("What is machine learning?")
print(result1['answer'])  # Includes source references

# Continue conversation (remembers context)
result2 = rag.chat("Can you give me examples of what we just discussed?")
print(result2['answer'])  # Will reference previous context
```

### Manual Memory Management
```python
# Start a named conversation
conversation_id = rag.start_conversation("ml_discussion")

# Get conversation history
history = rag.get_conversation_history()

# Export memory for persistence
memory_data = rag.export_conversation_memory()

# Clear and restore memory
rag.clear_conversation_memory()
rag.import_conversation_memory(memory_data)
```

### Controlling Source Display
```python
# Include sources in answer text (default)
result = rag.chat("What is AI?", include_sources_in_answer=True)

# Exclude sources from answer text (sources still in metadata)
result = rag.chat("What is AI?", include_sources_in_answer=False)

# Access different source formats
sources = result['sources']                    # Simple list
detailed = result['detailed_sources']         # With excerpts and scores
references = result['source_references']      # Formatted for display
```

## Memory Efficiency Features

### Lean Conversation Storage
- **User Messages**: Stored in full for context
- **Assistant Responses**: Stored without source references
- **Sources**: Not stored in memory (available in response only)
- **Retrieved Context**: Not stored in memory (saves significant space)

### Automatic Summarization
- Triggers when conversation exceeds threshold
- Creates concise summaries of older turns
- Maintains key topics and context
- Removes detailed source information from summaries

### Memory Statistics
```python
stats = rag.get_conversation_history()['conversation_stats']
print(f"Active turns: {stats['active_turns']}")
print(f"Summarized segments: {stats['summarized_segments']}")
print(f"Memory tokens: {stats['total_estimated_tokens']}")
```

## Conversation Context Enhancement

The system automatically enhances prompts with conversation context when:
1. Conversation memory is enabled
2. There's existing conversation history
3. The context is relevant to the current question

### Context-Aware Prompts
When conversation history exists, the system uses an enhanced prompt template that:
- Includes previous conversation context
- Maintains coherence across turns
- References earlier topics when relevant
- Preserves document-grounded responses

## Best Practices

### 1. **Memory Configuration**
- Use moderate `max_turns` (5-15) for most applications
- Set `summarization_threshold` to 60-80% of `max_turns`
- Enable summarization for long conversations

### 2. **Source Management**
- Always include sources in responses for transparency
- Use `detailed_sources` for rich applications
- Use `source_references` for simple displays

### 3. **Performance Optimization**
- Keep conversation memory lean by not storing sources
- Export/import memory for session persistence
- Monitor memory token usage for very long conversations

### 4. **Error Handling**
- Check for conversation memory availability
- Handle memory export/import failures gracefully
- Provide fallbacks when memory is disabled

## Implementation Notes

### Thread Safety
- ConversationMemory is not thread-safe by default
- Use separate instances for concurrent conversations
- Consider adding locking for multi-threaded applications

### Persistence
- Memory export/import uses JSON-serializable format
- Timestamps are stored in ISO format
- Configuration is included in exported data

### Backward Compatibility
- All existing RAG functionality remains unchanged
- New features are opt-in via parameters
- Original response format is preserved in `raw_answer`

This implementation provides a robust foundation for conversation-aware RAG applications while maintaining efficiency and flexibility.

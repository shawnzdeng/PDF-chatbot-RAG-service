# RAG Chatbot Streamlit UI

A user-friendly web interface for the PDF chatbot RAG (Retrieval-Augmented Generation) service, built with Streamlit.

## Features

### 🤖 Intelligent Conversation
- **Continuous Memory**: The chatbot remembers your conversation context for more coherent interactions
- **Document-Based Answers**: All responses are grounded in the provided PDF documents
- **Source Attribution**: Every answer includes expandable source references with relevance scores

### 📚 Document Integration
- **Production Configuration**: Automatically loads document metadata and example questions from production config
- **Smart Retrieval**: Uses optimized Qdrant vector search with reranking for better relevance
- **Context Display**: View document excerpts and page references for each source

### 🎨 User Experience
- **Clean Interface**: Modern, responsive design with intuitive navigation
- **Example Questions**: Quick-start with pre-configured example questions
- **Performance Metrics**: View system performance scores in the sidebar
- **Real-time Chat**: Seamless conversation flow with typing indicators

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
Create a `.env` file or set environment variables:
```bash
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_API_KEY=your_qdrant_api_key_here
```

### 3. Run the Application

**Option A: Use the provided scripts**
- Windows: Double-click `run_streamlit.bat` or `run_streamlit.ps1`
- The scripts will check your environment and start the app

**Option B: Use the Python launcher**
```bash
python launch_ui.py
```

**Option C: Manual command**
```bash
streamlit run src/ui/streamlit_app.py
```

### 4. Open in Browser
The app will automatically open in your default browser at `http://localhost:8501`

## Interface Overview

### Main Chat Area
- **Chat Input**: Type your questions at the bottom
- **Message History**: View the full conversation with context
- **Source Expansion**: Click "View Sources" to see supporting documents
- **Document Excerpts**: Expand individual sources to read relevant text

### Sidebar Information
- **Document Overview**: Description of the loaded documents
- **Performance Metrics**: System evaluation scores
- **Conversation Controls**: Clear history and view stats
- **System Configuration**: Current model and parameter settings

### Example Questions Panel
- **Quick Start**: Pre-configured questions from production config
- **One-Click Ask**: Click any example question to immediately ask it
- **Usage Tips**: Guidance for getting better results

## Production Configuration

The app automatically loads settings from `production_config/production_rag_config_*.json`:

- **Document Metadata**: File description and example questions
- **Model Settings**: Optimized LLM and embedding model configurations
- **Performance Metrics**: Evaluation scores (faithfulness, relevancy, etc.)
- **Retrieval Settings**: Vector search and reranking parameters

## Advanced Features

### Conversation Memory
- **Context Awareness**: References previous messages in the conversation
- **Memory Management**: Automatic summarization for long conversations
- **Session Persistence**: Memory maintained throughout the session

### Document Reranking
- **Hybrid Scoring**: Combines embedding similarity with cross-encoder relevance
- **Relevance Optimization**: Shows most relevant document sections first
- **Score Transparency**: Display relevance scores for each source

### Source Attribution
- **Document References**: File names and page numbers for each source
- **Relevance Scoring**: Numerical scores showing how well sources match the query
- **Text Excerpts**: Preview of relevant document sections
- **Expandable Details**: Full context available on demand

## Troubleshooting

### Common Issues

**App won't start:**
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify environment variables are set correctly
- Ensure Python version is compatible (3.8+)

**No responses from chatbot:**
- Verify OPENAI_API_KEY is valid and has sufficient credits
- Check QDRANT_API_KEY and collection exists
- Look at the terminal/console for error messages

**Sources not displaying:**
- Ensure the Qdrant collection has documents with proper metadata
- Check that the collection name in config matches your setup
- Verify document chunks have source and page information

### Performance Optimization

**For better response times:**
- Use a local Qdrant instance if possible
- Adjust `top_k` values in configuration
- Consider using faster embedding models for development

**For better answer quality:**
- Use the optimized production configuration
- Enable reranking for better source relevance
- Tune temperature settings for your use case

## Development

### Customization
- Modify `src/ui/streamlit_app.py` to change the UI layout or styling
- Update CSS in the `st.markdown()` sections for visual changes
- Add new features by extending the RAG system integration

### Integration
- The app uses the existing `QdrantRAG` class for all backend operations
- Configuration is managed through the `Config` class
- Session state manages conversation continuity

## Support

For issues or questions:
1. Check the logs in the terminal where you started the app
2. Verify your environment variables and configuration
3. Review the production config file for correct settings
4. Check the RAG system components are working independently

## License

This project is part of the PDF-chatbot-RAG-service and follows the same licensing terms.

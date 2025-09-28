# 🏺 Pharaoh Tour Guide - AI-Powered Egyptian History RAG System

An intelligent tour guide system that uses RAG (Retrieval-Augmented Generation) to provide expert knowledge about ancient Egyptian history, pharaohs, monuments, and culture.

## 🚀 Quick Start

1. **Complete Setup** (Recommended):
   ```bash
   cd src
   python pharaoh_rag_setup.py setup
   ```

2. **Interactive Chat**:
   ```bash
   python pharaoh_rag_setup.py chat
   ```

## 📚 What's Included

- **DataLoader System**: Automatically processes PDFs, DOCX, TXT files with Egyptian historical content
- **RAG Pipeline**: Semantic search through historical documents using vector embeddings
- **Pharaoh-Focused Prompts**: Specialized prompts for tour guide responses about Egyptian history
- **Sample Historical Content**: Pre-loaded knowledge about pharaohs, pyramids, culture, and monuments

## 🔧 Architecture

```
DataLoader → DocumentProcessor → VectorStore → RAG Model → Tour Guide Responses
```

## 📖 Documentation

- [Quick Start Guide](src/QUICK_START_GUIDE.md) - Complete setup and usage instructions
- [DataLoader Documentation](src/controllers/DataLoader_README.md) - Detailed file loading system docs
- [Usage Examples](src/dataloader_examples.py) - Code examples and patterns

## 🎯 Sample Queries

- "Who was Tutankhamun and why is he famous?"
- "What are the main pyramids at Giza?"
- "Tell me about ancient Egyptian gods and religion"
- "How were pharaohs mummified?"

## 🏺 Features

- **Multi-format Support**: PDF, DOCX, TXT, PPTX files
- **Semantic Search**: Find relevant historical information using vector embeddings
- **Tour Guide Persona**: Engaging, knowledgeable responses about Egyptian history
- **Extensible**: Easy to add new historical documents and expand knowledge base

Ready to explore ancient Egypt with AI? See the [Quick Start Guide](src/QUICK_START_GUIDE.md) to begin! 🏺
# Agentic AI

A project implementing AI agents with task planning, execution, and verification capabilities.

## Features

- **Basic Agent**: Task planning and execution using LangGraph
- **RAG Agent**: Document retrieval and enhanced task execution
- **Vector Database**: FAISS-based document indexing and search

## Project Structure

```
agentic-AI-1/
├── basic-agent.py          # Basic agent implementation
├── rag-agent.py            # RAG-enhanced agent
├── vector_db.py            # Vector database utilities
├── documents.pkl           # Preprocessed documents
├── faiss_index.bin         # FAISS vector index
├── basic-agent-HLD.png     # Basic agent architecture diagram
├── rag-agent-HLD.png       # RAG agent architecture diagram
├── pyproject.toml          # Project dependencies
└── README.md               # This file
```

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Set up environment variables:
```bash
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

## Usage

Run the basic agent:
```bash
python basic-agent.py
```

Run the RAG-enhanced agent:
```bash
python rag-agent.py
```

## Dependencies

- LangChain & LangGraph for agent orchestration
- Groq for LLM inference
- FAISS for vector search
- DuckDuckGo for web search
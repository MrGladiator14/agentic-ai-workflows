# Agentic AI

A project implementing AI agents with task planning, execution, and verification capabilities.

## Features

- **Basic Agent**: Task planning and execution using LangGraph
- **RAG Agent**: Document retrieval and enhanced task execution
- **Memory Agent**: Long-term memory with thread isolation using ChromaDB
- **Advanced RAG Agent**: Combines RAG, memory, and tools with conditional routing
- **Vector Database**: FAISS-based document indexing and search

## Project Structure

```
agentic-AI-1/
├── basic-agent.py          # Basic agent implementation
├── rag-agent.py            # RAG-enhanced agent
├── memory-agent.py         # Agent with long-term memory
├── adv_rag_agent.py        # Advanced RAG agent with memory, tools, and routing
├── vector_db.py            # Vector database utilities
├── documents.pkl           # Preprocessed documents
├── faiss_index.bin         # FAISS vector index
├── basic-agent-HLD.png     # Basic agent architecture diagram
├── rag-agent-HLD.png       # RAG agent architecture diagram
├── transcript.md           # 4-turn transcript demonstrating adv_rag_agent
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

Run the memory agent:
```bash
python memory-agent.py
```

Run the advanced RAG agent with memory and tools:
```bash
python adv_rag_agent.py
```

## Advanced RAG Agent Use Cases

The `adv_rag_agent.py` combines RAG, memory, and tools with intelligent routing. Here are two realistic use cases:

### 1. Research Assistant with Context Awareness
A researcher can interact with the agent across multiple sessions about AI/ML topics. The agent:
- Uses RAG to retrieve technical information from the knowledge base (e.g., differences between BERT and GPT)
- Remembers previous conversations across sessions using thread-isolated memory
- Performs calculations when needed (e.g., computing metrics, statistical analysis)
- Analyzes text documents (word counts, sentence structure) for literature reviews
- Searches the web for current research papers and developments

**Example workflow**: The researcher asks about transformer architectures, later asks to calculate performance metrics, then references the earlier discussion - the agent maintains context and provides coherent responses.

### 2. Educational Tutor with Multi-Modal Capabilities
An educational platform can use this agent to help students learn technical subjects:
- Explains complex concepts using the RAG knowledge base (NLP, deep learning, etc.)
- Remembers each student's learning progress and past questions via thread isolation
- Helps with mathematical problems using the calculator tool
- Analyzes student essays or code snippets using the text analyzer
- Provides current information from web search when questions require up-to-date data

**Example workflow**: A student asks about neural networks, requests calculation of gradient descent steps, submits an essay for analysis, and later asks follow-up questions - the agent provides personalized assistance based on the student's interaction history.

## Dependencies

- LangChain & LangGraph for agent orchestration
- Groq for LLM inference
- FAISS for vector search
- DuckDuckGo for web search
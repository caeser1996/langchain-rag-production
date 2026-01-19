# LangChain RAG Production Pipeline

A production-ready Retrieval-Augmented Generation (RAG) pipeline template built with LangChain, LangGraph, and FastAPI.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Gateway                          │
├─────────────────────────────────────────────────────────────────┤
│                    LangGraph Orchestrator                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Ingestion│  │ Retrieval│  │ Response │  │   Eval   │       │
│  │  Agent   │  │  Agent   │  │  Agent   │  │  Agent   │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
├───────┼─────────────┼─────────────┼─────────────┼──────────────┤
│       │             │             │             │              │
│  ┌────▼─────┐  ┌────▼─────┐  ┌────▼─────┐  ┌────▼─────┐       │
│  │  PDF/    │  │ Pinecone │  │   LLM    │  │ Metrics  │       │
│  │  DOCX    │  │ ChromaDB │  │  (GPT/   │  │  Store   │       │
│  │  Parser  │  │          │  │  Claude) │  │          │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
├─────────────────────────────────────────────────────────────────┤
│                      Redis Cache Layer                          │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **Document Ingestion**: Support for PDF, DOCX, and medical records
- **Vector Search**: Pinecone and ChromaDB integration
- **Multi-Agent Orchestration**: LangGraph-based agent coordination
- **Query Optimization**: Intelligent query rewriting and caching
- **Evaluation Framework**: Built-in metrics and evaluation tools
- **Production Ready**: Docker and Kubernetes deployment configs

## Tech Stack

- **Framework**: LangChain, LangGraph
- **API**: FastAPI
- **Vector Stores**: Pinecone, ChromaDB
- **Cache**: Redis
- **Deployment**: Docker, Kubernetes

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env

# Run the API server
uvicorn src.main:app --reload
```

## Project Structure

```
langchain-rag-production/
├── src/
│   ├── agents/          # LangGraph agents
│   ├── ingestion/       # Document processors
│   ├── retrieval/       # Vector store integrations
│   ├── cache/           # Redis caching layer
│   └── evaluation/      # Evaluation framework
├── config/              # Configuration files
├── tests/               # Test suite
├── docker/              # Docker configurations
└── k8s/                 # Kubernetes manifests
```

## License

MIT

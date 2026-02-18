# RAG Vector Database System

A lightweight Retrieval-Augmented Generation (RAG) system built with FastAPI, Qdrant vector database, and BGE-M3 text embeddings.

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│   Client    │────▶│   FastAPI (RAG)  │────▶│   Qdrant    │
│  (curl/HTTP)│     │   Port: 5121     │     │ Port: 6333  │
└─────────────┘     └──────────────────┘     └─────────────┘
                           │
                           ▼
                    ┌──────────────────┐
                    │  BGE-M3-TEI      │
                    │  Port: 6400      │
                    └──────────────────┘
```

## Components

| Service | Port | Description |
|---------|------|-------------|
| **RAG API** | 5121 | FastAPI HTTP interface for adding/querying documents |
| **Qdrant** | 6333 | Vector database for storing embeddings |
| **BGE-M3-TEI** | 6400 | Text Embeddings Inference server (BAAI/bge-m3 model) |

## Quick Start

### Prerequisites

- Docker & Docker Compose
- At least 4GB RAM for the embedding model

### Setup

1. **Clone and configure:**
   ```bash
   cp .env.example .env
   ```

2. **Start all services:**
   ```bash
   docker compose up -d
   ```

3. **Verify services are running:**
   ```bash
   docker compose ps
   curl http://localhost:5121/health
   ```

## API Usage

### Add a Document

```bash
curl -X POST http://localhost:5121/add \
  -H "Content-Type: application/json" \
  -d '{
    "id": "doc1",
    "content": "Your document text here",
    "metadata": {"category": "example"}
  }'
```

**Response:**
```json
{
  "success": true,
  "document_id": "doc1",
  "message": "Document added successfully"
}
```

### Query Documents

```bash
curl -X POST http://localhost:5121/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Python?",
    "top_k": 5
  }'
```

**Response:**
```json
{
  "results": [
    {
      "id": "doc1",
      "content": "Python is a high-level programming language...",
      "metadata": {"category": "programming"},
      "score": 0.75
    }
  ]
}
```

### Health Check

```bash
curl http://localhost:5121/health
```

**Response:**
```json
{"status": "healthy"}
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/add` | Add a document to the vector database |
| `POST` | `/query` | Search for similar documents |
| `GET` | `/health` | Health check endpoint |

### Request Schemas

**POST /add**
```json
{
  "id": "string (optional, auto-generated if not provided)",
  "content": "string (required)",
  "metadata": "object (optional)"
}
```

**POST /query**
```json
{
  "query": "string (required)",
  "top_k": "integer (optional, default: 5)"
}
```

## Configuration

Edit `.env` to customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_API_PORT` | 5121 | HTTP port for RAG API |
| `QDRANT_HTTP_PORT` | 6333 | HTTP port for Qdrant |
| `QDRANT_GRPS_PORT` | 6334 | gRPC port for Qdrant |
| `EMBEDDING_PORT` | 6400 | Port for embedding service |
| `COLLECTION_NAME` | documents | Qdrant collection name |
| `HF_TOKEN` | (your token) | HuggingFace token (optional) |

## Development

### Run API locally (without Docker)

```bash
pip install -r requirements.txt
uvicorn server.main:app --reload --port 5121
```

Ensure Qdrant and embedding services are running and update environment variables:

```bash
export QDRANT_URL=http://localhost:6333
export EMBEDDING_SERVICE_URL=http://localhost:6400
```

### Project Structure

```
.
├── docker-compose.yml      # Service orchestration
├── .env                    # Environment variables
├── .env.example            # Environment template
├── requirements.txt        # Python dependencies
├── server/
│   ├── main.py            # FastAPI application
│   └── Dockerfile         # API container config
└── storage/
    ├── embedding/         # Cached embedding models
    └── qdrant_storage/    # Vector database files
```

## Model Information

- **Model:** BAAI/bge-m3
- **Embedding Dimension:** 1024
- **Max Tokens:** 8192
- **Pooling:** CLS
- **License:** MIT

The BGE-M3 model supports:
- Dense retrieval
- Sparse retrieval
- Multi-vector retrieval

## License

MIT

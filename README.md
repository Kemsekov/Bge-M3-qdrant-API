# RAG Vector Database System

A lightweight Retrieval-Augmented Generation (RAG) system built with FastAPI, Qdrant vector database, BGE-M3 text embeddings, and a web-based UI for document management.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│   Web UI    │────▶│    nginx    │────▶│   FastAPI (RAG)  │────▶│   Qdrant    │
│  Port:3000  │     │  (reverse   │     │   Port: 5121     │     │ Port: 6333  │
│             │     │   proxy)    │     │                  │     │             │
└─────────────┘     └─────────────┘     └──────────────────┘     └─────────────┘
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
| **Web UI** | 3000 | Nginx-served frontend for document management and RAG queries |
| **RAG API** | 5121 | FastAPI HTTP interface for adding/querying documents |
| **Qdrant** | 6333 | Vector database for storing embeddings |
| **BGE-M3-TEI** | 6400 | Text Embeddings Inference server (BAAI/bge-m3 model) |

## Quick Start

### Prerequisites

- Docker & Docker Compose
- **At least 4GB RAM** for the embedding model (BGE-M3 uses ~3.7GB with low-memory config)

### Setup

1. **Clone and configure:**
   ```bash
   cp .env.example .env
   # Edit .env with your settings (RAG_MODEL, RAG_URL, etc.)
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

4. **Access the Web UI:**
   - Open browser: `http://localhost:3000`
   - Configure API Base URL if needed (default: `/api`)
   - Use the **Test Connection** button to verify backend connectivity

## Web UI Features

| Tab | Description |
|-----|-------------|
| **➕ Add Document** | Add documents with ID, type, content, and custom metadata |
| **📄 View Documents** | Browse documents by type with pagination (configurable page size) |
| **🔍 Search** | Semantic search using vector similarity |
| **🤖 Ask LLM** | RAG-powered Q&A with query rephrasing and context retrieval |

### API Base URL Configuration

The Web UI includes a settings bar to configure the backend API URL:
- **Default:** `/api` (uses nginx reverse proxy, same-origin)
- **Custom:** Set any backend URL (e.g., `http://localhost:5121`)
- URL is saved in browser localStorage for persistence
- Use **Reset** button to restore default `/api`

## API Usage

### Add a Document

```bash
curl -X POST http://localhost:5121/add \
  -H "Content-Type: application/json" \
  -d '{
    "id": "doc1",
    "content": "Your document text here",
    "metadata": {"type": "programming", "author": "John"}
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

### Query Documents (Semantic Search)

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
      "metadata": {"type": "programming"},
      "score": 0.75
    }
  ]
}
```

### Answer Questions with RAG

```bash
curl -X POST http://localhost:5121/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is Python?",
    "top_k": 5,
    "rephrases": 3
  }'
```

**Response:**
```json
{
  "answer": "Python is a high-level programming language...",
  "original_question": "What is Python?",
  "rephrases": ["Define Python programming", "Explain Python language"],
  "contexts": [...]
}
```

### Get Paged Documents by Type

```bash
curl "http://localhost:5121/get_docs_paged?doc_type=programming&page_size=100&page_number=1"
```

**Response:**
```json
{
  "documents": [...],
  "total_count": 250,
  "page_number": 1,
  "page_size": 100,
  "total_pages": 3
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
| `POST` | `/query` | Search for similar documents using semantic search |
| `POST` | `/answer` | Answer questions using RAG with context retrieval and rephrasing |
| `GET` | `/get_docs_paged` | Get paged documents filtered by document type |
| `GET` | `/health` | Health check endpoint |
| `GET` | `/docs` | Swagger UI documentation |
| `GET` | `/redoc` | ReDoc documentation |

### Request Schemas

**POST /add**
```json
{
  "id": "string (optional, auto-generated if not provided)",
  "content": "string (required)",
  "metadata": {
    "type": "programming",
    "author": "John",
    "version": "1.0"
  }
}
```

**POST /query**
```json
{
  "query": "string (required)",
  "top_k": "integer (optional, default: 5, max: 100)"
}
```

**POST /answer**
```json
{
  "question": "string (required)",
  "top_k": "integer (optional, default: 5, max: 100)",
  "rephrases": "integer (optional, default: 0, max: 10)"
}
```

**GET /get_docs_paged**
```
GET /get_docs_paged?doc_type=programming&page_size=100&page_number=1

Query Parameters:
- doc_type (required): Filter by document type (from metadata.type)
- page_size (optional, default: 100): Documents per page (max: 1000)
- page_number (optional, default: 1): Page number (1-indexed)
```

## Configuration

Edit `.env` to customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_API_PORT` | 5121 | HTTP port for RAG API |
| `WEB_PORT` | 3000 | HTTP port for Web UI |
| `QDRANT_HTTP_PORT` | 6333 | HTTP port for Qdrant |
| `QDRANT_GRPS_PORT` | 6334 | gRPC port for Qdrant |
| `EMBEDDING_PORT` | 6400 | Port for embedding service |
| `COLLECTION_NAME` | documents | Qdrant collection name |
| `RAG_MODEL` | (required) | LLM model name for RAG |
| `RAG_URL` | (required) | LLM API endpoint URL |
| `RAG_API_KEY` | (required) | LLM API key |
| `RAG_MAX_TOKENS` | 1024 | Max tokens for LLM response |
| `RAG_TEMPERATURE` | 1.0 | LLM temperature (0.0-2.0) |
| `HF_TOKEN` | (optional) | HuggingFace token for model downloads |

### BGE-M3 Low Memory Configuration

The `.env` file includes optimized parameters that **reduce RAM usage from ~14GB to ~3.7GB**:

| Parameter | Low-Mem Value | Default | Savings |
|-----------|---------------|---------|---------|
| `EMBEDDING_MAX_CONCURRENT_REQUESTS` | 64 | 512 | Less queue memory |
| `EMBEDDING_MAX_BATCH_TOKENS` | 4096 | 16384 | ~4GB |
| `EMBEDDING_MAX_CLIENT_BATCH_SIZE` | 8 | 32 | ~1GB |
| `EMBEDDING_MAX_BATCH_REQUESTS` | 16 | unset | ~1GB |
| `EMBEDDING_TOKENIZATION_WORKERS` | 2 | CPU cores | ~2-3GB |

**Trade-off**: Lower throughput under heavy concurrent load. For single-user RAG or low-traffic deployments, this is perfectly fine with no noticeable difference.

## Development

### Run API locally (without Docker)

```bash
pip install -r requirements.txt
uvicorn server.main:app --reload --port 5121
```

Ensure Qdrant and embedding services are running:

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
├── README.md              # This file
├── server/
│   ├── main.py            # FastAPI application
│   ├── models.py          # Pydantic models
│   ├── rag.py             # RAG orchestration
│   ├── settings.py        # Configuration
│   ├── prompts.json       # LLM prompts
│   └── Dockerfile         # API container config
├── web/
│   ├── index.html         # Web UI HTML
│   ├── css/
│   │   └── styles.css     # Styles
│   ├── js/
│   │   └── app.js         # Frontend logic
│   ├── nginx.conf         # Nginx reverse proxy config
│   └── Dockerfile         # Web container config
└── storage/
    ├── embedding/         # Cached embedding models
    └── qdrant_storage/    # Vector database files
```

## Model Information

### Embedding Model
- **Model:** BAAI/bge-m3
- **Embedding Dimension:** 1024
- **Max Tokens:** 8192
- **Pooling:** CLS
- **License:** MIT

The BGE-M3 model supports:
- Dense retrieval
- Sparse retrieval
- Multi-vector retrieval

### LLM Integration
- Compatible with Ollama, OpenAI, and OpenAI-compatible APIs
- Supports query rephrasing for improved retrieval
- Configurable temperature and max tokens

## License

MIT

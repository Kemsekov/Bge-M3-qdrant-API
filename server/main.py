import os
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

app = FastAPI(title="RAG API")

# Configuration
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://bge-m3-embedding:80")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
EMBEDDING_DIMENSION = 1024  # BAAI/bge-m3 embedding dimension


class DocumentInput(BaseModel):
    id: Optional[str] = None
    content: str
    metadata: Optional[dict] = None


class QueryInput(BaseModel):
    query: str
    top_k: int = 5


class QueryResult(BaseModel):
    id: str
    content: str
    metadata: Optional[dict] = None
    score: float


class AddResponse(BaseModel):
    success: bool
    document_id: str
    message: str


class QueryResponse(BaseModel):
    results: list[QueryResult]


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def ensure_collection_exists(client: QdrantClient) -> None:
    collections = client.get_collections().collections
    if not any(c.name == COLLECTION_NAME for c in collections):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION, distance=Distance.COSINE
            ),
        )


async def get_embedding(text: str) -> list[float]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{EMBEDDING_SERVICE_URL}/embed",
            json={"inputs": [text]},
            timeout=30.0,
        )
        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Embedding service error: {response.status_code}",
            )
        result = response.json()
        return result[0] if isinstance(result, list) else result["embeddings"][0]


@app.post("/add", response_model=AddResponse)
async def add_document(doc: DocumentInput):
    """Add a document to the vector database.
    
    Embeds the document content using BGE-M3 and stores it in Qdrant.
    """
    try:
        qdrant_client = get_qdrant_client()
        ensure_collection_exists(qdrant_client)

        # Generate embedding
        embedding = await get_embedding(doc.content)

        # Create document ID if not provided
        document_id = doc.id or f"doc_{len(embedding)}_{hash(doc.content)}"

        # Store in Qdrant
        point = PointStruct(
            id=hash(document_id) % (2**63 - 1),  # Qdrant requires int IDs
            vector=embedding,
            payload={
                "document_id": document_id,
                "content": doc.content,
                "metadata": doc.metadata or {},
            },
        )

        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[point])

        return AddResponse(
            success=True,
            document_id=document_id,
            message="Document added successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_documents(query_input: QueryInput):
    """Query the vector database for similar documents.
    
    Embeds the query and returns top_k most similar documents.
    """
    try:
        qdrant_client = get_qdrant_client()
        ensure_collection_exists(qdrant_client)

        # Generate embedding for query
        query_embedding = await get_embedding(query_input.query)

        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=query_input.top_k,
        )

        results = [
            QueryResult(
                id=hit.payload.get("document_id", str(hit.id)),
                content=hit.payload.get("content", ""),
                metadata=hit.payload.get("metadata"),
                score=hit.score,
            )
            for hit in search_results
        ]

        return QueryResponse(results=results)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

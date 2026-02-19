import numpy as np
from pathlib import Path
from server.settings import *
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from server.models import (
    DocumentInput,
    QueryInput,
    QueryResult,
    AddResponse,
    QueryResponse,
    AnswerInput,
    AnswerResponse,
    PagedDocsInput,
    PagedDocsResponse,
)
from server.rag import Rag, initialize_rag
import httpx

app = FastAPI(
    title="RAG Vector Database API",
    description="""
## Overview

A unified RAG (Retrieval-Augmented Generation) API that combines:
- **Vector Database** (Qdrant) for document storage and similarity search
- **Text Embeddings** (BGE-M3) for semantic understanding
- **LLM Integration** for intelligent query rephrasing and answer generation

## Features

- **Document Management**: Add and query documents with metadata
- **Semantic Search**: Find similar documents using vector embeddings
- **RAG-Powered Answers**: Get AI-generated answers based on retrieved context
- **Query Rephrasing**: Automatically rephrase questions for better retrieval

## Architecture

```
Client → FastAPI → Qdrant + BGE-M3 Embedding + LLM
```
    """,
    version="1.0.0",
    contact={
        "name": "RAG API Support",
    },
    license_info={
        "name": "MIT",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG instance
rag: Rag | None = initialize_rag(
    model=RAG_MODEL,
    url=RAG_URL,
    api_key=RAG_API_KEY,
    max_tokens=RAG_MAX_TOKENS,
    temperature=RAG_TEMPERATURE,
    prompts_path="/app/server/prompts.json",
)

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


async def retrieve_context(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve context from the vector database."""
    qdrant_client = get_qdrant_client()
    ensure_collection_exists(qdrant_client)
    query_embedding = await get_embedding(query)
    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
    )
    return [
        {
            "id": hit.payload.get("document_id", str(hit.id)),
            "content": hit.payload.get("content", ""),
            "metadata": hit.payload.get("metadata"),
            "score": hit.score,
        }
        for hit in search_results
    ]


async def retrieve_context_with_embeddings(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve context from the vector database along with document vectors."""
    qdrant_client = get_qdrant_client()
    ensure_collection_exists(qdrant_client)
    query_embedding = await get_embedding(query)
    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
        with_vectors=True,
    )
    return [
        {
            "id": hit.payload.get("document_id", str(hit.id)),
            "content": hit.payload.get("content", ""),
            "metadata": hit.payload.get("metadata"),
            "score": hit.score,
            "vector": hit.vector,
        }
        for hit in search_results
    ]


@app.post("/add", response_model=AddResponse, tags=["Documents"])
async def add_document(doc: DocumentInput):
    """
    Add a document to the vector database.
    
    Embeds the document content using BGE-M3 and stores it in Qdrant.
    The document can then be retrieved via semantic search.
    """
    try:
        qdrant_client = get_qdrant_client()
        ensure_collection_exists(qdrant_client)
        embedding = await get_embedding(doc.content)
        document_id = doc.id or f"doc_{len(embedding)}_{hash(doc.content)}"
        point = PointStruct(
            id=hash(document_id) % (2**63 - 1),
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


@app.post("/query", response_model=QueryResponse, tags=["Documents"])
async def query_documents(query_input: QueryInput):
    """
    Query the vector database for similar documents.
    
    Embeds the query using BGE-M3 and returns the top_k most similar documents
    based on cosine similarity.
    """
    try:
        qdrant_client = get_qdrant_client()
        ensure_collection_exists(qdrant_client)
        query_embedding = await get_embedding(query_input.query)
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


@app.post("/answer", response_model=AnswerResponse, tags=["RAG"])
async def answer_question(answer_input: AnswerInput):
    """
    Answer a question using RAG with query rephrasing and context retrieval.
    
    This endpoint performs the following steps:
    1. **Rephrase**: Generates 3 alternative phrasings of the question using LLM
    2. **Retrieve**: Searches for relevant contexts using original + rephrased queries
    3. **Rank**: Sorts all contexts by similarity score and deduplicates
    4. **Generate**: Creates an answer using the top_k contexts and LLM
    
    **Note**: Requires LLM configuration (RAG_MODEL, RAG_URL) to be set.
    """
    print(f"Answer request with question {answer_input}")
    if rag is None:
        raise HTTPException(
            status_code=503,
            detail="RAG LLM not configured. Set RAG_MODEL, RAG_URL environment variables."
        )
    
    question = answer_input.question
    top_k = answer_input.top_k
    
    # Step 1: Rephrase the question
    if answer_input.rephrases>0:
        rephrases = rag.rephrase(question)[:answer_input.rephrases]
    else:
        rephrases=[]
    
    # Step 2: Retrieve contexts for original question and all rephrases
    all_contexts = []
    queries = [question] + rephrases

    for query in queries:
        contexts = await retrieve_context(query, top_k=top_k)
        for ctx in contexts:
            ctx["source_query"] = query
            all_contexts.append(ctx)

    # Sort by original score and deduplicate
    all_contexts.sort(key=lambda x: x.get("score", 0), reverse=True)
    all_contexts=all_contexts[:top_k]
    
    seen_contents = set()
    unique_contexts = []
    for ctx in all_contexts:
        content = ctx.get("content", "")
        if content not in seen_contents:
            seen_contents.add(content)
            unique_contexts.append(ctx)
        if len(unique_contexts) >= top_k:
            break

    top_contexts = unique_contexts[:top_k]
    
    # Step 4: Generate answer using original question and top contexts
    context_texts = [ctx.get("content", "") for ctx in top_contexts]
    answer_text = rag.answer(question=question, context=context_texts)
    
    return AnswerResponse(
        answer=answer_text,
        original_question=question,
        rephrases=rephrases,
        contexts=top_contexts,
    )


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns the current status of the API.
    """
    return {"status": "healthy"}


@app.get("/get_docs_paged", response_model=PagedDocsResponse, tags=["Documents"])
async def get_docs_paged(
    doc_type: str,
    page_size: int = 100,
    page_number: int = 1
):
    """
    Get paged documents from the vector database filtered by document type.

    Documents are filtered by the `type` or `doc_type` field in metadata using Qdrant filters.
    Returns pagination info including total count and total pages.

    Note: For optimal performance, create a payload index on metadata.type:
        client.create_payload_index(collection_name, "metadata.type", field_type="keyword")
    """
    try:
        qdrant_client = get_qdrant_client()
        ensure_collection_exists(qdrant_client)

        # Use Qdrant filter for efficient server-side filtering
        from qdrant_client.http.models import FieldCondition, MatchValue, Filter

        # Filter by both 'type' and 'doc_type' fields for backwards compatibility
        doc_type_filter = Filter(
            should=[
                FieldCondition(
                    key="metadata.type",
                    match=MatchValue(value=doc_type)
                ),
                FieldCondition(
                    key="metadata.doc_type",
                    match=MatchValue(value=doc_type)
                )
            ]
        )

        # Get total count using count API with filter
        count_result = qdrant_client.count(
            collection_name=COLLECTION_NAME,
            count_filter=doc_type_filter,
            exact=True
        )
        total_count = count_result.count

        if total_count == 0:
            return PagedDocsResponse(
                documents=[],
                total_count=0,
                page_number=page_number,
                page_size=page_size,
                total_pages=0,
            )

        total_pages = (total_count + page_size - 1) // page_size

        # Validate page number
        if page_number > total_pages:
            return PagedDocsResponse(
                documents=[],
                total_count=total_count,
                page_number=page_number,
                page_size=page_size,
                total_pages=total_pages,
            )

        # Calculate offset for pagination
        offset = (page_number - 1) * page_size

        # Scroll with filter to get documents for current page
        records, _ = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=doc_type_filter,
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        documents = [
            QueryResult(
                id=record.payload.get("document_id", str(record.id)),
                content=record.payload.get("content", ""),
                metadata=record.payload.get("metadata", {}),
                score=1.0,  # No score for scroll results
            )
            for record in records
        ]

        return PagedDocsResponse(
            documents=documents,
            total_count=total_count,
            page_number=page_number,
            page_size=page_size,
            total_pages=total_pages,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from typing import Optional, List
from pydantic import BaseModel, Field


class DocumentInput(BaseModel):
    id: Optional[str] = Field(default=None, description="Optional document ID. Auto-generated if not provided.")
    content: str = Field(..., description="The text content of the document to be stored.", min_length=1)
    metadata: Optional[dict] = Field(default=None, description="Optional metadata associated with the document.")


class QueryInput(BaseModel):
    query: str = Field(..., description="The search query text.", min_length=1)
    top_k: int = Field(default=5, description="Number of top results to return.", ge=1, le=100)


class QueryResult(BaseModel):
    id: str = Field(..., description="Document identifier.")
    content: str = Field(..., description="Document content.")
    metadata: Optional[dict] = Field(default=None, description="Document metadata.")
    score: float = Field(..., description="Similarity score (higher is more similar).", ge=0, le=1)


class AddResponse(BaseModel):
    success: bool = Field(..., description="Whether the operation was successful.")
    document_id: str = Field(..., description="The ID of the added document.")
    message: str = Field(..., description="Status message.")


class QueryResponse(BaseModel):
    results: List[QueryResult] = Field(..., description="List of matching documents.")


class AnswerInput(BaseModel):
    question: str = Field(..., description="The question to answer.", min_length=1)
    top_k: int = Field(default=5, description="Number of top contexts to retrieve for answer generation.", ge=1, le=100)
    rephrases: int = Field(default=0, description="Number of top rephrases to apply to input question. Larger values will improve context retrieval", ge=0, le=100)

class AnswerResponse(BaseModel):
    answer: str = Field(..., description="The LLM-generated answer based on the retrieved contexts.")
    original_question: str = Field(..., description="The original question asked.")
    rephrases: List[str] = Field(..., description="List of rephrased versions of the question.")
    contexts: List[dict] = Field(..., description="Top-k contexts retrieved, sorted by relevance score.")


class PagedDocsInput(BaseModel):
    doc_type: str = Field(..., description="The document type to filter by (from metadata).", min_length=1)
    page_size: int = Field(default=100, description="Number of documents per page.", ge=1, le=1000)
    page_number: int = Field(default=1, description="Page number (1-indexed).", ge=1)


class PagedDocsResponse(BaseModel):
    documents: List[QueryResult] = Field(..., description="List of documents on the requested page.")
    total_count: int = Field(..., description="Total number of documents of this type.")
    page_number: int = Field(..., description="Current page number.")
    page_size: int = Field(..., description="Number of documents per page.")
    total_pages: int = Field(..., description="Total number of pages available.")


class DocTypesResponse(BaseModel):
    doc_types: List[str] = Field(..., description="List of all unique document types in the database.")

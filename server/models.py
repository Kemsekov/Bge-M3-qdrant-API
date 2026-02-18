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


class AnswerResponse(BaseModel):
    answer: str = Field(..., description="The LLM-generated answer based on the retrieved contexts.")
    original_question: str = Field(..., description="The original question asked.")
    rephrases: List[str] = Field(..., description="List of rephrased versions of the question.")
    contexts: List[dict] = Field(..., description="Top-k contexts retrieved, sorted by relevance score.")

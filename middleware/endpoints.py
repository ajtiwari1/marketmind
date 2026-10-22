"""HTTP endpoints for asking questions against indexed documents."""

import logging

from fastapi import APIRouter, HTTPException
from google.genai.errors import APIError
from qdrant_client.http.exceptions import ApiException, ResponseHandlingException, UnexpectedResponse

from chat.generation import (
    GeminiConfigurationError,
    answer_question,
)
from chat.retrieval import RetrievalFilters
from .schemas import AskRequest, AskResponse, SourceResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["RAG"])


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Return a grounded Gemini answer and the retrieved source chunks."""
    filters = RetrievalFilters(
        document_ids=tuple(request.document_ids),
        file_types=tuple(request.file_types),
    )
    try:
        result = answer_question(
            request.question,
            top_k=request.top_k,
            filters=filters,
        )
    except GeminiConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (
        APIError,
        ApiException,
        OSError,
        ResponseHandlingException,
        RuntimeError,
        UnexpectedResponse,
    ) as error:
        logger.error("Question answering failed: %s", type(error).__name__)
        raise HTTPException(
            status_code=502,
            detail="Retrieval or answer generation service failed",
        ) from error

    return AskResponse(
        answer=result.answer,
        model=result.model,
        sources=[
            SourceResponse(
                chunk_id=source.chunk_id,
                document_id=source.document_id,
                text=source.text,
                source=source.source,
                file_name=source.file_name,
                file_type=source.file_type,
                chunk_index=source.chunk_index,
                score=source.score,
                metadata=source.metadata,
            )
            for source in result.sources
        ],
    )

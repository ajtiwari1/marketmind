"""HTTP endpoints for asking questions against indexed documents."""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from google.genai.errors import APIError
from qdrant_client.http.exceptions import ApiException, ResponseHandlingException, UnexpectedResponse

from chat.generation import (
    GeminiConfigurationError,
    answer_question,
    answer_with_sources,
)
from chat.retrieval import RetrievedChunk, RetrievalFilters, retrieve
from search_tool import TavilyMCPClient, WebSearchResult
from search_tool.client import SearchToolConfigurationError, SearchToolError
from .schemas import AskRequest, AskResponse, SourceResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["RAG"])


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """Return a grounded Gemini answer using the requested search sources."""
    filters = RetrievalFilters(
        document_ids=tuple(request.document_ids),
        file_types=tuple(request.file_types),
    )
    try:
        if request.search_mode == "local":
            result = await run_in_threadpool(
                answer_question,
                request.question,
                top_k=request.top_k,
                filters=filters,
            )
        else:
            sources = []
            if request.search_mode == "hybrid":
                local_sources = await run_in_threadpool(
                    retrieve,
                    request.question,
                    top_k=request.top_k,
                    filters=filters,
                )
                sources.extend(local_sources)
            elif request.document_ids or request.file_types:
                raise HTTPException(
                    status_code=422,
                    detail="document_ids and file_types only apply to local retrieval",
                )

            web_sources = await TavilyMCPClient().search(
                request.question,
                max_results=request.top_k,
            )
            sources.extend(web_sources)
            result = await run_in_threadpool(
                answer_with_sources,
                request.question,
                sources,
            )
    except GeminiConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except SearchToolConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (
        APIError,
        ApiException,
        OSError,
        ResponseHandlingException,
        RuntimeError,
        UnexpectedResponse,
        SearchToolError,
    ) as error:
        logger.error("Question answering failed: %s", type(error).__name__)
        raise HTTPException(
            status_code=502,
            detail="Retrieval or answer generation service failed",
        ) from error

    return AskResponse(
        answer=result.answer,
        model=result.model,
        sources=[_source_response(source) for source in result.sources],
    )


def _source_response(source: RetrievedChunk | WebSearchResult) -> SourceResponse:
    if isinstance(source, WebSearchResult):
        return SourceResponse(
            source_type="web",
            text=source.snippet,
            source=source.url,
            file_name=source.title,
            score=source.score,
            metadata={},
            url=source.url,
            published_date=source.published_date,
        )
    return SourceResponse(
        source_type="local",
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

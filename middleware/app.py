"""FastAPI application for the RAG question-answering service."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from libs.langfuse import flush_langfuse
from .endpoints import router


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    yield
    flush_langfuse()


app = FastAPI(
    title="MarketMind RAG API",
    description="Hybrid document retrieval and Gemini-grounded answers",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(router)

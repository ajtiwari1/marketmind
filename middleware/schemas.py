"""Request and response models for the RAG HTTP API."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: Annotated[str, Field(min_length=1, max_length=20_000)]
    top_k: Annotated[int, Field(ge=1, le=20)] = 5
    search_mode: Literal["local", "web", "hybrid"] = "local"
    document_ids: list[Annotated[str, Field(min_length=1)]] = Field(
        default_factory=list
    )
    file_types: list[Annotated[str, Field(min_length=1)]] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("question must not be blank")
        return question


class SourceResponse(BaseModel):
    source_type: Literal["local", "web"]
    chunk_id: str | None = None
    document_id: str | None = None
    text: str
    source: str
    file_name: str
    file_type: str | None = None
    chunk_index: int | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    url: str | None = None
    published_date: str | None = None


class AskResponse(BaseModel):
    answer: str
    model: str
    sources: list[SourceResponse]

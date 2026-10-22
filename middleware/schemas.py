"""Request and response models for the RAG HTTP API."""

from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: Annotated[str, Field(min_length=1, max_length=20_000)]
    top_k: Annotated[int, Field(ge=1, le=20)] = 5
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
    chunk_id: str
    document_id: str
    text: str
    source: str
    file_name: str
    file_type: str
    chunk_index: int
    score: float
    metadata: dict[str, Any]


class AskResponse(BaseModel):
    answer: str
    model: str
    sources: list[SourceResponse]

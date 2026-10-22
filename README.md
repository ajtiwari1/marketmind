# marketmind

## Reusable Langfuse utilities

The Langfuse helpers live in `libs.langfuse` and use the official Python SDK.
For a Langfuse server/platform v4, this project depends on Python SDK v5.

Install [uv](https://docs.astral.sh/uv/) and Python 3.11 or newer, then sync
the project dependencies and create the virtual environment and lockfile:

```bash
uv sync
```

Add your Langfuse project credentials to the root `.env` file. Keep this file
local and never commit it. For Langfuse Cloud, the SDK's default host can be
used; set `LANGFUSE_HOST` in `.env` only when using a custom/self-hosted host:

```dotenv
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
# Optional:
# LANGFUSE_HOST=https://your-langfuse-host
```

Run commands with the `.env` file loaded by uv:

```bash
uv run --env-file .env python your_app.py
```

To send a real example trace, generation, and Python log messages to Langfuse:

```bash
uv run --env-file .env python examples/langfuse_trace_demo.py
```

The demo creates a request trace with a simulated generation. Its dedicated
logger writes to the console and records each demo log message as a child span.
Do not send secrets or sensitive data in trace inputs or log messages.

Use the helpers in an application:

```python
from libs.langfuse import (
    flush_langfuse,
    propagate_attributes,
    trace_logs,
)
import logging

logger = logging.getLogger("my_app.worker")

@trace_logs(logger_name="my_app.worker", name="answer-question")
def answer_question(question: str) -> str:
    logger.info("Starting answer generation")
    return "..."

with propagate_attributes(user_id="user-123"):
    answer_question("What is Langfuse?")

# Flush buffered events during application shutdown.
flush_langfuse()
```

`@trace_logs` wraps the function in a Langfuse span and records messages from
the named logger (and its child loggers) as child spans for that call. Use
`@observe` directly when you want function tracing without log capture.

The SDK manages client construction and configuration from environment
variables. Importing the helpers does not create a client; call
`get_langfuse_client()` when the application needs it. Use
`uv run --env-file .env python -m unittest discover -s tests -v` to run the
tests.

## RAG document ingestion

The initial RAG ingestion pipeline lives under `chat/`. It supports PDF, DOCX,
PPTX, XLSX, CSV, HTML, Markdown, plain text, and JSON files. Chunks retain
document and section provenance and are embedded with both configured dense
and sparse models before being written to local Qdrant.

Each Qdrant point stores a named `dense` vector from Sentence Transformers and a
named `sparse` vector from FastEmbed's `Qdrant/bm25` model. The default Qdrant URL
is `http://localhost:6333`, the collection is `marketmind_hybrid_documents`,
and the dense embedding model is
`sentence-transformers/all-MiniLM-L6-v2`. Embedding models are downloaded on
first use. The collection must be created explicitly; the pipeline validates
its vector schema and configured model names and never deletes/recreates it
automatically. Existing dense-only collections are left untouched; use a new
collection name for dense+sparse ingestion.

When Langfuse credentials are configured in `.env`, each ingested document
creates a `rag.ingest_document` span with nested dense embedding, sparse
embedding, and Qdrant upsert spans. Trace inputs include counts and model or
collection identifiers, not document text. The CLI flushes Langfuse events
before exiting.

Preview parsing and chunking without writing to Qdrant:

```bash
uv run --env-file .env marketmind-ingest ./documents --dry-run
```

Create the configured collection if it does not exist and ingest a directory:

```bash
uv run --env-file .env marketmind-ingest ./documents --create-collection
```

Set optional configuration in the ignored `.env` file:

```dotenv
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=marketmind_hybrid_documents
# Optional if the Qdrant instance requires authentication:
# QDRANT_API_KEY=
RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_SPARSE_EMBEDDING_MODEL=Qdrant/bm25
RAG_CHUNK_SIZE_TOKENS=500
RAG_CHUNK_OVERLAP_TOKENS=75
RAG_EMBEDDING_BATCH_SIZE=32
```

Change the collection name when switching to a model with a different vector
dimension. Scanned PDFs without extractable text fail with an actionable OCR
message; OCR, hybrid retrieval, and reranking are not included in this phase.
Run the ingestion unit tests with:

```bash
uv run --env-file .env python -m unittest tests.test_rag_ingestion -v
```

Run the optional test against the existing local Qdrant service. It creates
and removes a uniquely named temporary collection:

```bash
RUN_QDRANT_INTEGRATION=1 uv run --env-file .env python -m unittest tests.test_rag_qdrant -v
```

## Hybrid retrieval

Search the configured dense+sparse collection with reciprocal rank fusion:

```bash
uv run --env-file .env marketmind-search "your search question" --top-k 5
```

Optional `--document-id` and `--file-type` filters may be repeated. The Python
API returns typed results with each chunk's text, source, metadata, and fused
score:

```python
from chat import RetrievalFilters, retrieve

results = retrieve(
    "your search question",
    top_k=5,
    filters=RetrievalFilters(file_types=("text/markdown",)),
)
```

The query uses the same dense and sparse models configured for ingestion. Search
does not create or alter collections and rejects embedding-model mismatches.
Langfuse records nested query embedding and Qdrant retrieval observations without
including query or document text.

## Gemini answer generation

Add your Gemini key and model to the ignored `.env` file:

```dotenv
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-3.8-flash
```

The model is configurable. Ask a question using retrieved chunks as grounding
context:

```bash
uv run --env-file .env marketmind-ask "How do I configure the application?" --top-k 5
```

The generator instructs Gemini to use only the retrieved excerpts and cite them
as `[1]`, `[2]`, and so on. The CLI prints the corresponding source list. If
retrieval returns no chunks, it responds that it could not find relevant
information without making an LLM call.

```python
from chat import answer_question

result = answer_question("How do I configure the application?", top_k=5)
print(result.answer)
for source in result.sources:
    print(source.file_name, source.source)
```

Langfuse records generation metadata and lengths only; it does not receive the
question, retrieved text, or generated answer. Gemini receives the question and
retrieved excerpts to produce the grounded response. Requests use the Gemini
Interactions API with `store=False`.

## FastAPI question endpoint

Start the API after configuring `.env` and ensuring the hybrid Qdrant collection
and Gemini API key are available:

```bash
uv run --env-file .env uvicorn middleware.app:app --reload
```

Send a question to `POST /api/ask`:

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I configure the application?",
    "top_k": 5,
    "document_ids": [],
    "file_types": []
  }'
```

The JSON response contains `answer`, the Gemini `model`, and `sources` with
each retrieved chunk's text, source, file type, score, and metadata. `top_k`
must be between 1 and 20, and document ID and MIME type filters are optional.
Interactive API documentation is available at `http://127.0.0.1:8000/docs`.
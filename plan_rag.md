# Plan: RAG document reading, chunking, and indexing

## Goal and scope

Build the ingestion foundation for a future advanced RAG system: read multiple
document formats, extract structure-aware text, create useful chunks, embed
them with a Sentence Transformers model, and persist vectors and metadata in the
already-running local Qdrant instance.

This plan covers the ingestion foundation. Dense and sparse vectors are stored
for every chunk to support hybrid retrieval. Retrieval is tracked separately in
[plan_retrieval.md](./plan_retrieval.md); reranking, query processing, and
answer generation remain later phases.

## Proposed module layout

```text
chat/
  __init__.py
  config.py
  documents/
    __init__.py
    models.py
    loader.py
    parsers/
      __init__.py
      pdf.py
      office.py
      html.py
      text.py
      structured.py
  chunking/
    __init__.py
    base.py
    text.py
    markdown.py
    tables.py
  embeddings/
    __init__.py
    sentence_transformer.py
  vector_store/
    __init__.py
    qdrant.py
  pipeline.py
tests/
  test_rag_ingestion.py
  test_rag_qdrant_unit.py
  test_rag_qdrant.py
```

Keep parsers, chunkers, embedding generation, and Qdrant persistence behind
small interfaces so a format, strategy, or storage backend can be extended
without coupling it to the ingestion pipeline.

## Supported inputs for the first implementation

Use explicit extension/MIME-type dispatch and reject unsupported inputs with a
clear error rather than silently treating binary files as text.

| Input type | Initial support | Extraction and structure to preserve |
| --- | --- | --- |
| PDF | Digital PDFs; scanned PDFs when OCR is configured | Page boundaries, page numbers, headings where detectable, reading order |
| Word (`.docx`) | Paragraphs and tables | Heading levels, paragraph order, table boundaries |
| PowerPoint (`.pptx`) | Slides, text boxes, and tables | Slide number, title, content order |
| Excel (`.xlsx`) and CSV | Sheets/tables and rows | Workbook, sheet/table name, column headers, row boundaries |
| HTML | Local HTML and saved pages | Title, headings, sections, links where useful; remove scripts/styles/navigation noise |
| Markdown | `.md` files | Heading hierarchy, code fences, lists, tables |
| Plain text | `.txt`, `.rst`, and `.log` | Paragraphs and line boundaries |
| JSON | `.json` | Key paths and array/object boundaries; produce readable labeled text |

Add OCR as an optional parser capability, not an unconditional dependency. In
the initial implementation, report scanned/unextractable PDFs clearly when OCR
is not installed or configured. Treat email, image, audio, and video ingestion
as later additions unless real input requirements call for them.

## Ingestion and chunking strategy

1. **Represent extracted content consistently** — implemented
   - Define typed `Document`, `DocumentSection`, and `Chunk` models.
   - Store source URI/path, stable document ID, format, page/slide/sheet, heading
     path, table/row identifiers, and extraction warnings as available.
   - Separate content text from metadata so payloads remain filterable and
     content can be updated without losing provenance.

2. **Load and normalize** — implemented for the listed local file formats
   - Dispatch to a parser based on file type; validate files and apply
     configurable size limits.
   - Normalize line endings and whitespace while retaining paragraph, heading,
     code, and table boundaries.
   - Preserve source offsets or page/section references when a parser exposes
     them; do not merge unrelated pages, slides, sheets, or documents.
   - Record parser/OCR errors with the source identifier and fail that file
     explicitly; do not emit empty or success-shaped documents.

3. **Chunk according to structure** — implemented; initial tests use a fake
   tokenizer, and production chunk counts use the embedding model tokenizer
   - Prefer semantic/structural boundaries first, then split oversized
     sections to fit a configured token budget.
   - Use a tokenizer compatible with the selected embedding model when
     calculating chunk size; do not treat character count as token count.
   - Start with configurable defaults around 400–600 tokens per chunk and
     50–100 tokens of overlap for prose, then tune against representative data.
   - Repeat a concise heading path in each prose chunk so its context survives
     retrieval; retain the original body separately if practical.
   - Keep Markdown headings with their sections, and avoid splitting fenced
     code blocks unless an individual block exceeds the limit.
   - Keep tables coherent: preserve headers with row groups, split very large
     tables by rows, and include table/sheet and column labels in each chunk.
   - Chunk CSV/Excel by header plus row groups, not arbitrary character spans.
   - Serialize JSON with key paths and keep related object/array records
     together; split large arrays at record boundaries.
   - For slide decks, keep slide-local content together unless one slide is too
     large. Keep page/slide/sheet identifiers on every resulting chunk.
   - Make chunk size, overlap, and strategy configurable per format and
     collection rather than hard-coded.

4. **Generate embeddings** — implemented with a lazy Sentence Transformers
   adapter; first model use downloads the configured model
   - Use the `sentence-transformers` package and a configurable model name.
   - Start with a compact, general-purpose model such as
     `sentence-transformers/all-MiniLM-L6-v2` for local development; document
     that changing models changes vector dimensions and can require a new
     collection/re-embedding.
   - Load the model once per process, batch embedding requests, and expose batch
     size/device configuration. Keep the model ID in collection configuration
     and chunk payloads.
   - Determine vector size from the loaded model rather than duplicating a
     dimension constant; use the same model and preprocessing for all indexed
     chunks in a collection.

5. **Persist to local Qdrant** — implemented with named dense and sparse vectors
   - Reuse the existing local Qdrant service; do not start, install, or replace
     the service as part of the first ingestion implementation.
   - Read connection URL and collection name from configuration/environment.
     Confirm the running service URL and collection naming before coding; use
     `http://localhost:6333` only as a documented development default if it
     matches the existing setup.
   - Create a collection only when explicitly configured to do so. Validate an
     existing collection's vector size and distance metric before writing;
     never silently delete/recreate a populated collection.
   - Store a named dense cosine vector and named sparse vector plus payload
     fields for chunk text, document ID, source, format, chunk index,
     page/slide/sheet/section, heading path, content hash, and both model IDs.
   - Use stable point IDs derived from document identity and chunk identity so
     re-ingesting an unchanged document is idempotent. Define replacement and
     deletion behavior for changed documents; avoid leaving stale chunks.
   - Use FastEmbed's configurable `Qdrant/bm25` sparse model by default.
   - Keep hybrid querying and result fusion out of scope for this phase.

6. **Provide a usable ingestion entry point** — implemented as library
   functions and the `marketmind-ingest` CLI
   - Expose a library API such as `ingest_file(path)` and `ingest_directory(path)`
     with explicit include/exclude patterns and recursive traversal behavior.
   - Return an ingestion summary: accepted files, skipped files, extracted
     sections, chunks upserted, and per-file errors.
   - Add a small CLI only if it fits the project workflow; allow a dry-run mode
     that parses/chunks and reports counts without writing to Qdrant.

## Configuration and dependencies

- Manage new dependencies with `uv` in `pyproject.toml` and update `uv.lock`.
- Evaluate focused parser packages for each required format rather than adding
  a large all-formats dependency without need (for example, PDF, DOCX/PPTX/XLSX,
  HTML, and Qdrant client libraries).
- Add `sentence-transformers` and the Qdrant Python client only when
  implementing the relevant phases.
- Configure at least the embedding model, Qdrant URL, collection, batch size,
  chunk size/overlap, and optional OCR behavior.
- Keep credentials and local configuration in the ignored `.env`; the local
  Qdrant setup may not require credentials.
- Avoid recording secrets or full sensitive document contents in logs.

## Validation and acceptance criteria

The initial implementation creates small parser fixtures in temporary
directories and has an opt-in integration test that creates and removes only
its uniquely named test collection. OCR, real-world multilingual tuning, and
hybrid-search benchmarking remain future work.

- Small fixtures cover every supported format, including a scanned/unreadable
  PDF case and malformed/unsupported input cases.
- Parser tests verify extracted text and provenance (page, slide, sheet,
  heading, table/row) rather than only asserting that parsing succeeded.
- Chunk tests verify configured token limits, overlap, structural boundaries,
  stable ordering, and metadata preservation for prose, code, tables, and JSON.
- Pipeline tests verify vector count and finite values; a live model smoke test
  confirmed the configured model's output dimension.
- Qdrant integration tests target the existing local instance, use an isolated
  test collection, verify upsert/re-ingest behavior, and clean up only that
  explicitly named test collection.
- Dry-run ingestion requires no Qdrant writes.
- Run focused unit tests with `uv run --env-file .env python -m unittest
  discover -s tests -v`; keep local-service integration tests separately
  selectable.

## Out of scope for this phase

- Hybrid/dense+sparse search, query rewriting, and fusion algorithms.
- Reranking, answer generation, citations, and conversational memory.
- Production OCR, image understanding, audio/video transcription, and remote
  document connectors.
- Replacing or provisioning the existing local Qdrant service.

## Follow-up decisions

1. The local Qdrant service was reachable at `http://localhost:6333` and had
   no collections when implementation began. The URL and collection name are
   configurable; collection creation requires the explicit CLI flag.
2. Tune the default model and chunk settings against representative documents
   and the target hardware/languages.
3. Decide whether scanned-PDF OCR is required in the first release.

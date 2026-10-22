# Plan: Hybrid RAG retrieval

## Goal

Retrieve relevant chunks from the existing Qdrant dense+sparse collection using
Sentence Transformers and FastEmbed query embeddings, then combine both ranked
lists with Qdrant's Reciprocal Rank Fusion (RRF). Keep retrieval usable as a
Python API and a small CLI, then ground Gemini answers in the retrieved
passages. Do not modify the existing dense-only collection.

## Implementation stages

1. **Define the retrieval API** — implemented
   - Accept a non-empty query, a positive result count, and optional metadata
     filters for document IDs and file types.
   - Return typed chunk results with text, source, metadata, and fused score.
   - Reject invalid embedding output and incompatible collection/model settings
     explicitly.

2. **Generate query embeddings** — implemented
   - Reuse the configured dense Sentence Transformers and sparse FastEmbed
     models used during ingestion.
   - Validate one dense and one non-empty sparse vector before searching.
   - Never include raw query text or document text in Langfuse trace inputs.

3. **Run hybrid search and fusion** — implemented
   - Prefetch candidates independently from named `dense` and `sparse` vectors.
   - Fuse ranks with Qdrant RRF and return the requested number of chunks.
   - Apply the same metadata filters to both candidate searches.

4. **Expose entry points and tracing** — implemented
   - Add `chat.retrieve(...)` for application use and a `marketmind-search`
     command for local checks.
   - Record retrieval, query embedding, and Qdrant search as nested Langfuse
     observations; flush at CLI shutdown.

5. **Test and document** — implemented
   - Cover query validation, filters, fusion request construction, output
     mapping, and model/schema safety with unit tests.
   - Add an opt-in integration test using a temporary Qdrant collection.
   - Document API/CLI usage and limitations.

6. **Generate grounded answers with Gemini** — implemented
   - Use a configurable Gemini model and API key from environment variables.
   - Build the context only from retrieved chunks, ask for numbered source
     citations, and return the source objects alongside the answer.
   - Do not call the model when no relevant chunks were retrieved.
   - Trace generation metadata without transmitting the question, retrieved
     text, or generated answer to Langfuse.
   - Provide a Python API and `marketmind-ask` CLI.

The planned retrieval and initial Gemini generation phases are implemented.
Model evaluation, reranking, and query rewriting remain future work.

## Non-goals for this phase

- Cross-encoder reranking or LLM-based query rewriting.
- Conversation memory.
- Automatic collection creation, migration, or mutation during search.
- Benchmarking against a labeled relevance dataset.

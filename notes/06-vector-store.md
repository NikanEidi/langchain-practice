# 06: Vector store and retrieval

## What this does

Indexes three unrelated documents in an in-memory vector store, then runs a semantic similarity search for a query that shares no words with the correct document — proving retrieval works on meaning, not keyword matching.

```python
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

embeddings = OllamaEmbeddings(model="nomic-embed-text")

docs = [
    Document(page_content="LangChain is a framework for building LLM applications."),
    Document(page_content="Kubernetes orchestrates containers across a cluster of machines."),
    Document(page_content="Ruff is a fast Python linter written in Rust."),
]

vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(docs)

query = "What tool helps manage containers?"
results = vector_store.similarity_search(query, k=1)

print(results[0].page_content)
```

## Result (actual run)

```
Kubernetes orchestrates containers across a cluster of machines.
```

Correct, despite zero shared words between "manage containers" and "orchestrates containers across a cluster of machines" beyond "containers" itself.

## The connection to CineVision

This is the same core mechanism as CineVision's recommender (TF-IDF + cosine similarity): convert text to numbers, compare vectors, return the closest match. The difference is what does the converting — TF-IDF counts word frequency (so it mostly catches shared vocabulary); a neural embedding model like `nomic-embed-text` captures meaning, which is why "manage" and "orchestrates" end up close together even with no shared words.

## What `add_documents` actually does

Each `Document.page_content` gets passed through the embedding model (the same `embed_query`-style call from Part 5, run once per document) and the resulting 768-number vector is stored alongside the original text. This happens once, upfront — the indexing step. In a real system with thousands of documents, this is the expensive one-time cost; querying afterward is cheap by comparison.

## What `similarity_search` actually does

1. Embeds the query string the same way every document was embedded (same model, so the vectors live in the same space and are comparable).
2. Computes `cosine_similarity` between the query vector and every stored document vector — a standard measure of how close two vectors point in the same direction, independent of their length.
3. Returns the `k` documents with the highest similarity score, as a list of `Document` objects (`k=1` here still returns a list with one item, not a bare `Document`).

## Dependency note (the error hit mid-session)

`InMemoryVectorStore` relies on `numpy` to compute `cosine_similarity`, but doesn't install it as a hard dependency — it surfaces as an `ImportError` only when `similarity_search` actually runs, not at import time. Fixed with `pip install numpy`. Worth remembering: a clean `import` succeeding doesn't guarantee every code path's dependencies are satisfied.

## `InMemoryVectorStore` vs. a real vector store

This is RAM-only — nothing is saved to disk, and it's gone when the script ends. Fine for learning and small demos. A real project would use something persistent (Chroma, Pinecone, pgvector, etc.) so the expensive indexing step doesn't have to be redone on every run.

## One-line interview answer

> "A vector store holds documents as embeddings and lets you query by semantic similarity instead of keyword match — I verified this concretely by retrieving a Kubernetes document for a query about 'managing containers' with zero shared vocabulary beyond the word itself, which only works because the embedding captures meaning, not words."

# 05: Embeddings

## What this does

Converts a piece of text into a vector (a list of numbers) using a dedicated embedding model, `nomic-embed-text`. This is the foundation RAG is built on — nothing is retrieved yet, this part just proves embedding works and shows what the output actually looks like.

```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

text = "LangChain is a framework for building LLM applications."
vector = embeddings.embed_query(text)

print(f"Vector length: {len(vector)}")
print(f"First 5 numbers: {vector[:5]}")
```

## Result (actual run)

```
Vector length: 768
First 5 numbers: [0.005853893, 0.02392343, -0.14927709, -0.029465826, 0.055638984]
```

## Why a separate model for embeddings

`nomic-embed-text` is not a chat model — it has no concept of a conversation or an answer. Its only job is text-in, vector-out. This is a deliberate split in the ecosystem: embedding models are smaller, faster, and optimized purely for producing a good vector representation, not for generating fluent text. Using a chat model to generate embeddings is possible with some providers but is not the normal pattern.

## What the 768 numbers actually mean

Nothing individually interpretable by a human. What matters is **distance between vectors**: two pieces of text with similar meaning produce vectors that are close together in this 768-dimensional space (measured with something like cosine similarity), even if they share no words in common. This is the entire mechanism retrieval relies on — not keyword matching, but "is this chunk's vector close to the query's vector."

768 is fixed per model — every call to `embed_query` on this model returns exactly 768 numbers, whether the input is one word or several paragraphs.

## `embed_query` vs. `embed_documents` (not shown yet, worth knowing)

`embed_query` is for embedding a single piece of text (typically the user's question). A separate method, `embed_documents`, takes a list of texts and embeds all of them — used once, upfront, to index a whole document set. Same underlying model, different method depending on whether you're embedding one query or many chunks at indexing time.

## One-line interview answer

> "An embedding model converts text into a fixed-length numeric vector that captures meaning, not just words — texts with similar meaning end up with vectors close together in that space. RAG retrieval works by embedding the query and finding the closest stored vectors, not by keyword search."

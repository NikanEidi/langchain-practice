# 07: Full RAG (retrieval + generation)

## What this does

Combines Part 6's retrieval with actual generation: the retrieved document becomes the model's only source of truth for answering the question, instead of the model guessing from its own training.

```python
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2:3b")

docs = [
    Document(page_content="LangChain is a framework for building LLM applications."),
    Document(page_content="Kubernetes orchestrates containers across a cluster of machines."),
    Document(page_content="Ruff is a fast Python linter written in Rust."),
]

vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(docs)

query = "What tool helps manage containers?"
results = vector_store.similarity_search(query, k=1)
context = results[0].page_content

prompt = (
    "You must answer using ONLY the information in the context below. "
    "Do not use any outside knowledge. If the context contains the answer, "
    "state it directly and do not mention any other tools or options.\n\n"
    f"Context: {context}\n\n"
    f"Question: {query}\n\n"
    "Answer:"
)

response = llm.invoke([HumanMessage(prompt)])
print(response.content)
```

## Real result: a failure, then a fix (both worth keeping)

**First attempt — softer instruction** (`"Answer the question using only the context below."`):

```
Based on the provided context, it can be inferred that Docker is the tool that
helps manage containers. However, since the name "Docker" is not mentioned in
the context, we should look for other possibilities.

In this case, another popular container management tool is Podman (also known
as Crank). But again, its mention isn't present here.

Given the lack of explicit information and focusing on what's commonly
associated with container orchestration and management in Kubernetes: one
could infer that tools like kubectl are likely involved but aren't used to
manage containers themselves directly.
```

Completely wrong, despite the correct context ("Kubernetes orchestrates containers across a cluster of machines.") being right there in the prompt. The model ignored it and hallucinated Docker, Podman, and kubectl — none of which were ever mentioned.

**Second attempt — stricter instruction** (the version above, `"You must answer using ONLY..."` + `"do not mention any other tools or options"` + a trailing `"Answer:"`):

```
Kubernetes
```

Correct, direct, no hallucination.

## Why the first version failed — this is not a code bug

Retrieval was correct both times (verified in Part 6 — the right document was found). The failure was entirely in **generation**: a soft instruction ("using only the context") is a suggestion, not an enforced constraint, and `llama3.2:3b` is a small (3B parameter) model with noticeably weaker instruction-following than larger hosted models. It had the right answer sitting in front of it and still reached for outside "knowledge" instead.

## What actually changed between the two prompts

1. **"You must"** instead of a plain descriptive instruction — more directive phrasing.
2. **"Do not use any outside knowledge"** — made explicit what was previously only implied by "using only the context."
3. **"do not mention any other tools or options"** — directly blocks the exact failure mode observed (listing alternatives not in the context).
4. **A trailing `"Answer:"`** — signals "stop reasoning, respond now," which tends to cut down on open-ended, hedging continuations.

None of these are LangChain-specific — this is general prompt engineering, and it mattered more here than any code change would have.

## The real lesson: RAG does not guarantee correctness

Giving a model the right context is necessary but **not sufficient**. The model still has to (a) actually read and prioritize that context over its own training, and (b) follow the instruction to stay within it. Both can fail independently, especially on smaller models. This is a genuine, well-known limitation — not something unique to this toy example — and it's exactly why production RAG systems add things like output validation, citation-checking, or simply using a more capable model for the generation step.

## One-line interview answer

> "RAG retrieval finding the right document doesn't guarantee the model uses it — I hit this directly: the same correct context produced a hallucinated answer with a soft instruction, and a correct answer once the prompt explicitly forbade outside knowledge. Smaller models need much more directive prompting to actually stay grounded in retrieved context."

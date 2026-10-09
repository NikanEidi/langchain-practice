# langchain-practice

Hands-on LangChain practice, from a plain model call to a real tool-calling agent. Same format as [`k8s-practice`](https://github.com/NikanEidi/k8s-practice): small parts, real code, a short note per part explaining what it does and why.

## Setup

- Python 3, [Ollama](https://ollama.com) running locally with `llama3.2:3b` pulled
- `python3 -m venv venv && source venv/bin/activate`
- `pip install langchain langchain-ollama`

## Structure

```
langchain-practice/
├── README.md
├── code/     — the actual scripts, one per part
└── notes/    — one note per part: what it does, why, key lines explained
```

## Parts

| # | Part | Code | Note |
|---|---|---|---|
| 1 | First model call | [`code/01_first_call.py`](code/01_first_call.py) | [`notes/01-first-call.md`](notes/01-first-call.md) |
| 2 | System prompt | [`code/02_system_prompt.py`](code/02_system_prompt.py) | [`notes/02-system-prompt.md`](notes/02-system-prompt.md) |
| 3 | Tool calling (single round) | [`code/03_tool_calling.py`](code/03_tool_calling.py) | [`notes/03-tool-calling.md`](notes/03-tool-calling.md) |
| 4 | Real agent loop (multi-step, multi-tool) | [`code/04_agent_loop.py`](code/04_agent_loop.py) | [`notes/04-agent-loop.md`](notes/04-agent-loop.md) |
| 5 | Embeddings | [`code/05_embeddings.py`](code/05_embeddings.py) | [`notes/05-embeddings.md`](notes/05-embeddings.md) |
| 6 | Vector store and retrieval | [`code/06_vector_store.py`](code/06_vector_store.py) | [`notes/06-vector-store.md`](notes/06-vector-store.md) |
| 7 | Full RAG (retrieval + generation) | [`code/07_rag.py`](code/07_rag.py) | [`notes/07-rag.md`](notes/07-rag.md) |

**Real project built from this practice:** [code-review-agent](https://github.com/NikanEidi/code-review-agent) — a LangChain agent using real tools (`ruff`, `ast`) to review Python files.

More parts get added as we go: full RAG (retrieval + generation combined), then LangGraph.

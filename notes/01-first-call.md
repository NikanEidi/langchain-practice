# 01: First model call

## What this does

Sends one request to a local LLM (`llama3.2:3b`, served by Ollama) through LangChain's chat model interface and prints the text of the response.

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2:3b")

response = llm.invoke("What is LangChain in one sentence?")

print(response.content)
```

## What `ChatOllama(...)` actually is

`ChatOllama` is a **chat model wrapper** — a class that implements LangChain's `BaseChatModel` interface on top of Ollama's local HTTP API (Ollama runs a server on `localhost:11434`; this class is an HTTP client for it). Constructing it does nothing over the network yet — it just configures the object: which model, which host, and which generation parameters to use on every call.

Parameters worth knowing now, even unset (they default, but control real behavior):
- `temperature` — randomness. `0` = near-deterministic, same prompt tends to give the same answer. Higher (`0.7`–`1.0`) = more varied phrasing. Default varies by provider.
- `num_predict` (Ollama's name for max output tokens) — caps how long a response can get.
- `base_url` — defaults to `http://localhost:11434`; you'd change this if Ollama runs elsewhere (e.g. a container, a remote machine).

Every LangChain chat model class — `ChatOllama`, `ChatOpenAI`, `ChatAnthropic` — implements the same `invoke()` / `.content` interface. That uniform interface is the actual value LangChain adds here: swapping providers is a one-line change, not a rewrite.

## What `.invoke()` returns

`response` is not a string — it's an `AIMessage` object. `.content` is one field on it. The object also carries:
- `response_metadata` — provider-specific details (for Ollama: things like `eval_count`, `eval_duration`, total time). For a hosted provider this is where token usage and billing-relevant counts live.
- `id` — a generated message id, used later when messages need to reference each other (e.g. tool call results).
- `usage_metadata` — a normalized `{input_tokens, output_tokens, total_tokens}` shape, provider-independent.

This matters because in a real application you rarely print `.content` and stop — you log token usage, you check `response_metadata` for truncation/finish reasons, and later (tool calling) you inspect other fields on this same object (`.tool_calls`).

## Error modes worth knowing now

- **Ollama not running** → connection refused. The model class does not start Ollama for you.
- **Model not pulled** (`ollama pull llama3.2:3b` never run) → a 404-style error from the local API.
- **No exception handling here on purpose** — this is a learning script. A real call site wraps `.invoke()` in retry/error handling, because network and local-server calls fail in ways pure function calls don't.

## Why a local model for learning, not a hosted one

No API key, no cost, no internet dependency once pulled. The LangChain code is nearly identical either way — this is a deliberate choice to remove payment/account friction while learning the framework, not a claim that local models are how this would run in production for most companies.

## One-line interview answer

> "`ChatOllama.invoke()` sends a request to a local model server and returns an `AIMessage` object — the text is one field on it, alongside token usage and metadata. Every LangChain chat model implements the same interface, so switching from a local model to a hosted one is a one-line change."

# 01: First model call

## What this does

Sends one question to a local LLM (`llama3.2:3b`, running through Ollama) and prints the answer. No tools, no memory, no system prompt — the smallest possible LangChain program.

## Code

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2:3b")

response = llm.invoke("What is LangChain in one sentence?")

print(response.content)
```

## Why each line

- `ChatOllama(model="llama3.2:3b")` — creates a handle to the model. Nothing is sent yet, this just says "I want to talk to this model."
- `llm.invoke("...")` — the actual request. Sends the text, waits, gets a response object back.
- `response.content` — the response is an object, not a plain string. `.content` is the actual text inside it.

## Why Ollama, not OpenAI

Ollama runs the model locally — free, no API key, no internet dependency once the model is pulled. Good for learning; a real job would likely use a hosted model (OpenAI, Anthropic), but the LangChain code barely changes — just swap `ChatOllama` for `ChatOpenAI` or similar.

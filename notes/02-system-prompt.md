# 02: System prompt

## What this does

Controls the model's behavior with a fixed instruction (a `SystemMessage`), separate from the actual question (`HumanMessage`). Proves the system prompt works by asking the model to always answer in 2 sentences or less.

## Code

```python
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOllama(model="llama3.2:3b")

messages = [
    SystemMessage("You are a senior backend engineer. Always answer in 2 sentences or less, no fluff."),
    HumanMessage("What is LangChain?"),
]

response = llm.invoke(messages)
print(response.content)
```

## Why each line

- `SystemMessage(...)` — a fixed rule the model follows for the whole conversation. Not something the user typed; it's set by the code.
- `HumanMessage(...)` — the actual user question.
- `messages` is a **list**, not a single string — `invoke` can take a whole conversation, not just one question. This matters later, when we keep a running history.
- `llm.invoke(messages)` — same call as Part 1, but now the model sees both the rule and the question.

## Message types (so far)

| Type | Who writes it |
|---|---|
| `SystemMessage` | The code — a fixed rule or persona |
| `HumanMessage` | The user |
| `AIMessage` | The model's own reply (used once we keep conversation history — not yet here) |

## Key takeaway

The system prompt is how you give a LangChain app a personality or a hard constraint, without the user ever seeing or typing it.

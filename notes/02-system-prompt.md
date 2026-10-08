# 02: System prompt

## What this does

Separates a fixed behavioral rule (`SystemMessage`) from the user's actual question (`HumanMessage`), and sends both as a conversation list instead of a single string.

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

## Why a list, not a string

Part 1 passed a plain string to `.invoke()`. LangChain accepted it as shorthand and silently converted it into a single `HumanMessage` behind the scenes. Here, passing an explicit list is what makes a multi-role conversation possible at all — a plain string has no way to express "this part is a rule, this part is the user."

The list is also the mechanism conversation memory is built on later: a real chat app appends each new `HumanMessage` and the model's `AIMessage` reply to the same list and resends the whole thing every call, because these models are stateless between requests — nothing is remembered server-side unless you resend it.

## The three message roles

| Type | Who writes it | Sent to the model as |
|---|---|---|
| `SystemMessage` | The application code | A `"system"` role message |
| `HumanMessage` | The end user | A `"user"` role message |
| `AIMessage` | The model's own prior reply | An `"assistant"` role message, when replayed back in history |

Under the hood, LangChain converts these into the role-tagged JSON format every chat API actually expects (`{"role": "system", "content": "..."}`, etc.). The `SystemMessage`/`HumanMessage` classes exist so your code reads clearly and stays provider-agnostic — you're not hand-building that JSON or memorizing each provider's exact field names.

## Order and position matter

The system message is conventionally placed **first** in the list. Most providers treat position as meaningful — a system instruction appearing after user turns is weaker or ignored by some models, because it reads more like "the user said this" than "this is a standing rule." Put `SystemMessage` first unless you have a specific reason not to.

## Production considerations (not visible in this toy example)

- **Cost:** the system prompt is resent on every single call in a conversation — it is not "set once." A long system prompt multiplies token cost across every turn.
- **Not a security boundary:** a system prompt is a strong suggestion, not an enforced rule. It can be overridden or leaked by adversarial user input ("prompt injection") — never put secrets in a system prompt, and never treat it as a guarantee for safety-critical behavior.
- **Model-dependent strength:** smaller/local models (like `llama3.2:3b` here) follow system instructions less reliably than larger hosted models. If output doesn't match the instruction, that's a real, known limitation — not necessarily a bug in the code.

## One-line interview answer

> "A system prompt is a `SystemMessage` placed first in the message list — it sets a standing rule the model should follow for the whole conversation, separate from what the user typed. It's not a security boundary; it's a strong instruction that a determined adversarial prompt can still override."

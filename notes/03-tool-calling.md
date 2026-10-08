# 03: Tool calling (single round)

## What this does

Defines a Python function as a tool the model can request, lets the model decide it needs that tool, executes it in our own code, feeds the result back, and gets a final answer. One round only — not yet a loop.

```python
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from datetime import datetime

@tool
def get_current_time() -> str:
    """Returns the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

llm = ChatOllama(model="llama3.2:3b")
llm_with_tools = llm.bind_tools([get_current_time])

messages = [HumanMessage("What time is it right now?")]

ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

for call in ai_msg.tool_calls:
    result = get_current_time.invoke(call["args"])
    messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

final = llm_with_tools.invoke(messages)
print(final.content)
```

## The core fact this part proves

**The model never executes code.** It returns structured data describing a function name and arguments it wants called. Running that function is entirely our own code's job (`get_current_time.invoke(call["args"])`). This is the actual security and architecture boundary in any tool-using LLM system: the model proposes, your code decides whether to execute and with what guardrails.

## What `@tool` generates

`@tool` inspects the function's name, type hints, and docstring, and builds a `StructuredTool` wrapping it — carrying a JSON Schema derived from those hints. The docstring is not documentation for humans here; it's sent to the model and is the primary signal it uses to decide *when* to call this tool versus answering directly. A vague docstring produces unreliable tool selection — this is a real, common bug source, not a style nitpick.

## `bind_tools` — what actually crosses the wire

`llm.bind_tools([get_current_time])` returns a new model object that, on every `.invoke()`, attaches the tools' JSON Schemas to the outgoing API request. It returns a new object rather than mutating `llm` — this immutable/chainable pattern shows up throughout LangChain (`.bind_tools()`, later `.bind()`, `.with_structured_output()` all follow it).

## The two-call cycle, and why it's two calls

1. **First `.invoke(messages)`** — the model sees the question and the available tool schemas. It replies with an `AIMessage` whose `.content` is typically empty and whose `.tool_calls` is a non-empty list of `{name, args, id}` dicts. Nothing has executed yet.
2. **Our loop executes the real function**, using `call["args"]` as keyword arguments to the tool, and wraps the result in a `ToolMessage` tagged with `tool_call_id=call["id"]`. That id is what lets the model (and us) correctly associate a result with its originating request when there's more than one tool call in a single turn.
3. **Second `.invoke(messages)`** — the model now sees the full exchange (question → its own tool request → the real result) and produces a final natural-language answer, because `messages` carries full history and these models are stateless between calls.

## Why the `for` loop matters even with one tool call

`ai_msg.tool_calls` is always a list, because a single model turn can request multiple tool calls at once (e.g. "what time is it, and what's 5+7?" with two tools bound). The loop is what makes the code correct for 0, 1, or N simultaneous tool calls without special-casing any of them — not an artifact of this specific example.

## Why this is not yet "an agent"

This script makes exactly two `.invoke()` calls, hardcoded. If the model, after seeing the tool result, decided it needed *another* tool call to finish the job, there is no mechanism here to continue — the script would just print whatever `final.content` happens to be. A real agent replaces the fixed two-step sequence with a loop that keeps calling until the model stops requesting tools. That's the next part.

## Real risk worth knowing

The model can hallucinate arguments — wrong types, missing fields, or values outside any sane range. `call["args"]` is fed into `.invoke()` without any validation here. For a toy read-only tool like this, low stakes. For a tool that writes a file, calls a paid API, or moves data, this gap is a real production concern: validate (or clamp/allow-list) arguments before executing, don't trust them just because they came from a structured-looking response.

## One-line interview answer

> "Tool calling works in two round trips: the model requests a function call as structured data, my own code executes the real function and feeds the result back as a `ToolMessage`, and a second call to the model produces the final answer. The model never runs code itself — that boundary is the actual security model."

# 04: Real agent loop (multi-step, multi-tool)

## What this does

Replaces the fixed two-call sequence from Part 3 with a loop that keeps calling tools until the model decides it's done, and adds a second tool so the model has to choose between them (or use both).

```python
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from datetime import datetime

@tool
def get_current_time() -> str:
    """Returns the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def add(a: int, b: int) -> int:
    """Adds two numbers and returns the result."""
    return a + b

llm = ChatOllama(model="llama3.2:3b")
tools = [get_current_time, add]
tools_by_name = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

messages = [HumanMessage("What time is it, and what is 12 plus 30?")]

max_iterations = 5
for i in range(max_iterations):
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    if not ai_msg.tool_calls:
        print("FINAL:", ai_msg.content)
        break

    for call in ai_msg.tool_calls:
        tool_fn = tools_by_name[call["name"]]
        result = tool_fn.invoke(call["args"])
        messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
else:
    print("Stopped after max_iterations without a final answer.")
```

## Result (actual run)

```
FINAL: The current time is 6:10 PM and the result of 12 + 30 is 42.
```

Both tools were called from a single user message, and the final answer combined both results correctly.

## What's actually new vs. Part 3

| | Part 3 | Part 5 |
|---|---|---|
| Tool count | 1 | 2 |
| Tool dispatch | hardcoded `get_current_time.invoke(...)` | looked up by name: `tools_by_name[call["name"]]` |
| Number of `.invoke()` rounds | exactly 2, fixed | unbounded, up to `max_iterations`, decided by the model |
| Stop condition | implicit (script just ends) | explicit: `if not ai_msg.tool_calls: break` |

## `tools_by_name` — why a dict, not an `if/elif` chain

```python
tools_by_name = {t.name: t for t in tools}
```

A dict comprehension building `{"get_current_time": <tool>, "add": <tool>}`. With one tool, hardcoding the call was fine. With N tools, the model returns `call["name"]` as a string, and the dict gives O(1) lookup of the matching tool object — the alternative, a chain of `if call["name"] == "...": ... elif ...`, doesn't scale and is exactly the kind of code a dict lookup replaces. This is the general pattern for any dispatch-by-name situation, not LangChain-specific.

## The stop condition is the actual "agent" part

```python
if not ai_msg.tool_calls:
    print("FINAL:", ai_msg.content)
    break
```

An empty `tool_calls` list means the model looked at everything in `messages` so far — original question, its own prior tool requests, the real results — and decided no further action is needed. This decision is made by the model each iteration; nothing in the code tells it how many tool calls to expect. That's the actual definition of "agentic" here: the control flow is driven by the model's own judgment, not by a fixed script.

## `max_iterations` and `for...else` — the safety net

A model can loop: request a tool, get a result, request the same (or another) tool again, indefinitely, without ever emitting an empty `tool_calls`. `max_iterations` is a hard ceiling against that — a real cost/safety control, not a style choice. The `for...else` clause only runs if the loop completed all `max_iterations` rounds **without** hitting `break` — i.e., it's the "we gave up, no final answer" branch, distinct from "we got an answer and stopped early." This Python construct (`for`/`else`) is uncommon but exactly fits this situation: distinguishing a loop that finished normally from one that exited via `break`.

## Why `str(result)` matters here specifically

`add` returns an `int` (e.g. `42`), but `ToolMessage(content=...)` expects a string. Part 3's tool (`get_current_time`) already returned a string, so this bug was invisible there — it only surfaces once a tool returns a non-string type. Good general rule: tool functions' return types are not automatically coerced for you by `ToolMessage`.

## One-line interview answer

> "The loop keeps calling the model until its response has no tool calls left — that's the model's own signal that it's done, not a fixed number of steps. A `max_iterations` cap exists because nothing guarantees the model won't loop indefinitely requesting tools; that's a real cost and reliability concern in production agents, not just a toy-example detail."

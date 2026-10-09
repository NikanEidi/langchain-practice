# 08: LangGraph (the same agent, as a graph)

## What this does

Rebuilds Part 4's manual tool-calling loop using LangGraph: the same logic (call the model, run any requested tools, repeat until no more tools are requested), expressed as a graph of nodes and edges instead of a Python `for` loop with a `break`.

```python
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END

@tool
def get_current_time() -> str:
    """Returns the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

llm = ChatOllama(model="llama3.2:3b")
tools = [get_current_time]
tools_by_name = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

def call_model(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def call_tools(state: MessagesState):
    last_message = state["messages"][-1]
    results = []
    for call in last_message.tool_calls:
        tool_fn = tools_by_name[call["name"]]
        result = tool_fn.invoke(call["args"])
        results.append(ToolMessage(content=result, tool_call_id=call["id"]))
    return {"messages": results}

def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

graph = StateGraph(MessagesState)
graph.add_node("agent", call_model)
graph.add_node("tools", call_tools)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

app = graph.compile()

result = app.invoke({"messages": [HumanMessage("What time is it right now?")]})
print(result["messages"][-1].content)
```

## Result (actual run)

```
The current time is 5:22 PM.
```

## Direct mapping to Part 4's manual loop

| Part 4 (manual `for` loop) | Part 8 (LangGraph) |
|---|---|
| `ai_msg = llm_with_tools.invoke(messages)` + `messages.append(ai_msg)` | the `"agent"` node (`call_model`) |
| the inner `for call in ai_msg.tool_calls:` block | the `"tools"` node (`call_tools`) |
| `if not ai_msg.tool_calls: break` | the `should_continue` function |
| the loop itself | the edge from `"tools"` back to `"agent"` |
| `max_iterations` safety cap | not present here — LangGraph has its own recursion-limit mechanism, not shown in this minimal example |

Nothing about the underlying agent logic changed. What changed is *how the control flow is expressed* — as a declared graph structure instead of imperative Python control flow.

## New concepts, precisely

- **`StateGraph(MessagesState)`**: creates a graph whose shared state is a dict with a `"messages"` key. `MessagesState` is a built-in state schema — using it means LangGraph automatically **appends** new messages a node returns to the existing list, rather than replacing it. A node returning `{"messages": [response]}` doesn't overwrite history; it extends it.
- **A node is just a function** that takes `state` and returns a partial state update (a dict). `call_model` and `call_tools` are both ordinary Python functions — nothing LangGraph-specific inside them beyond reading/writing `state["messages"]`.
- **`graph.add_edge(A, B)`**: an unconditional transition — always go from `A` to `B`.
- **`graph.add_conditional_edges(A, fn)`**: after node `A`, call `fn(state)`; whatever string it returns names the next node (or `END`). This is how the stop condition from Part 4 (`if not ai_msg.tool_calls: break`) gets expressed in graph form.
- **`START` / `END`**: sentinel values marking the graph's entry point and termination, not real nodes you write logic for.
- **`graph.compile()`**: turns the declared structure into a runnable object (`app`). The graph is just a description until compiled.
- **`app.invoke({"messages": [...]})`**: runs the whole graph, not a single model call. Internally this is doing exactly what Part 4's `for` loop did — repeatedly calling `call_model` then `call_tools` — but the looping is handled by the graph's edges, not by Python control flow in the calling code.

## Why bother, if the behavior is identical to Part 4

For this toy example, it genuinely isn't a practical improvement — same logic, more ceremony. The payoff shows up at larger scale: branching logic more complex than a single yes/no loop, persisting state across sessions (checkpointing), pausing for human approval mid-run, or visualizing the flow. LangGraph is also what LangChain's current top-level `create_agent` API is built on internally — understanding the graph model underneath makes that higher-level API legible instead of magic.

## One-line interview answer

> "LangGraph expresses an agent's control flow as a graph of nodes and conditional edges instead of a hand-written loop. I built the identical agent both ways — a manual Python loop in one part, the same logic as a LangGraph graph in another — specifically to see that the underlying behavior doesn't change, only how the looping and branching get declared."

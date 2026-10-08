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
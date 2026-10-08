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
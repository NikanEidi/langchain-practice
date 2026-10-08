from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOllama(model="llama3.2:3b")

messages = [
    SystemMessage("You are a senior backend engineer. Always answer in 2 sentences or less, no fluff."),
    HumanMessage("What is LangChain?"),
]

response = llm.invoke(messages)
print(response.content)
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2:3b")

docs = [
    Document(page_content="LangChain is a framework for building LLM applications."),
    Document(page_content="Kubernetes orchestrates containers across a cluster of machines."),
    Document(page_content="Ruff is a fast Python linter written in Rust."),
]

vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(docs)

query = "What tool helps manage containers?"

results = vector_store.similarity_search(query, k=1)
context = results[0].page_content

prompt = (
    "You must answer using ONLY the information in the context below. "
    "Do not use any outside knowledge. If the context contains the answer, "
    "state it directly and do not mention any other tools or options.\n\n"
    f"Context: {context}\n\n"
    f"Question: {query}\n\n"
    "Answer:"
)

response = llm.invoke([HumanMessage(prompt)])
print(response.content)
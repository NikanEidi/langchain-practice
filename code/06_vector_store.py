from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

embeddings = OllamaEmbeddings(model="nomic-embed-text")

docs = [
    Document(page_content="LangChain is a framework for building LLM applications."),
    Document(page_content="Kubernetes orchestrates containers across a cluster of machines."),
    Document(page_content="Ruff is a fast Python linter written in Rust."),
]

vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(docs)

query = "What tool helps manage containers?"
results = vector_store.similarity_search(query, k=1)

print(results[0].page_content)
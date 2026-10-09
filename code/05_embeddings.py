from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

text = "LangChain is a framework for building LLM applications."
vector = embeddings.embed_query(text)

print(f"Vector length: {len(vector)}")
print(f"First 5 numbers: {vector[:5]}")
#Compare MiniLM vs MPNet embedding functions in ChromaDB

import time
import chromadb
from chromadb.utils import embedding_functions

documents = [
    "I love dogs",
    "I like puppies",
    "The stock market crashed today",
    "Cats are cute pets",
    "The economy is in recession"
]

query = "puppy lovers"

client = chromadb.Client()

# --- MiniLM collection ---
minilm_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
minilm_collection = client.create_collection(
    name="minilm_collection",
    embedding_function=minilm_embedder
)

start = time.time()
minilm_collection.add(documents=documents, ids=[f"m{i}" for i in range(len(documents))])
minilm_time = time.time() - start

minilm_results = minilm_collection.query(query_texts=[query], n_results=3)

# --- MPNet collection ---
mpnet_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2"
)
mpnet_collection = client.create_collection(
    name="mpnet_collection",
    embedding_function=mpnet_embedder
)

start = time.time()
mpnet_collection.add(documents=documents, ids=[f"p{i}" for i in range(len(documents))])
mpnet_time = time.time() - start

mpnet_results = mpnet_collection.query(query_texts=[query], n_results=3)

# --- Compare ---
print(f"MiniLM add time: {minilm_time:.3f}s")
print(f"MPNet  add time: {mpnet_time:.3f}s\n")

print("MiniLM results:")
for doc, distance in zip(minilm_results["documents"][0], minilm_results["distances"][0]):
    print(f"  Distance: {distance:.4f}  |  {doc}")

print("\nMPNet results:")
for doc, distance in zip(mpnet_results["documents"][0], mpnet_results["distances"][0]):
    print(f"  Distance: {distance:.4f}  |  {doc}")
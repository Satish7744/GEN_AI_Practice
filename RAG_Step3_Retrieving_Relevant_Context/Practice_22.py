from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions


documents = {
    "ai_overview.txt": """Artificial Intelligence: An Overview

Artificial Intelligence, or AI, is the field of computer science focused on building systems
that can perform tasks which normally require human intelligence. These tasks include
understanding language, recognizing images, making decisions, and learning from experience.

Machine Learning is a subset of AI where systems learn patterns from data instead of following
hardcoded rules. Instead of programming every decision by hand, we feed the system many examples,
and it learns the underlying pattern on its own.

Deep Learning takes Machine Learning further by using neural networks with many layers. These
layered networks can automatically discover complex patterns in large amounts of data, such as
recognizing faces in photos or understanding the meaning of a sentence.

Retrieval-Augmented Generation, or RAG, is a technique that combines a search system with a
language model. Instead of relying only on what a language model memorized during training, RAG
first retrieves relevant, up-to-date information from a document collection, then passes that
information to the model so it can generate a more accurate, grounded answer.
""",
    "embeddings_in_practice.txt": """Embeddings and Vector Search in Practice

Modern search engines increasingly rely on embeddings rather than simple keyword matching.
An embedding model converts a search query into the same vector space as the stored documents,
allowing the system to find results that match the meaning of the query, not just its exact words.

This is especially powerful for customer support systems, where a user might phrase a question very
differently from how it appears in a help article, yet still expect to find the right answer.

Vector databases such as ChromaDB store these embeddings and allow fast similarity search across
thousands or millions of documents at once.
""",
}


splitter = RecursiveCharacterTextSplitter(
    chunk_size=250, chunk_overlap=40, separators=["\n\n", "\n", ". ", " ", ""]
)

client = chromadb.Client()
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.create_collection(name="rag_chunks", embedding_function=embedder)

for source, text in documents.items():
    doc_chunks = splitter.split_text(text)
    collection.add(
        documents=doc_chunks,
        metadatas=[{"source": source, "chunk_index": i} for i in range(len(doc_chunks))],
        ids=[f"{source}_chunk_{i}" for i in range(len(doc_chunks))],
    )

print("Total chunks stored:", collection.count())


def distance_to_similarity(distance):
    return 1 - distance


def retrieve_with_threshold(query, collection, n_results=5, similarity_threshold=0.3):
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    filtered = []
    for doc, meta, distance in zip(results["documents"][0],
                                   results["metadatas"][0],
                                   results["distances"][0]):
        similarity = distance_to_similarity(distance)
        if similarity >= similarity_threshold:
            filtered.append({"text": doc, "metadata": meta, "similarity": similarity})
    return filtered



query = "What is the best recipe for chocolate cake?"
print(f"\nQuery: {query}\n")

for t in [0.3, 0.25, 0.1, 0.0]:
    matches = retrieve_with_threshold(query, collection, n_results=5, similarity_threshold=t)
    print(f"threshold={t:<4} -> {len(matches)} chunk(s) passed")
    for m in matches:
        print(f"    sim={m['similarity']:.4f} | {m['metadata']['source']} | {m['text'][:60]}...")
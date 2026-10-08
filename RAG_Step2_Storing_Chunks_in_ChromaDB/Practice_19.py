
import os
from collections import Counter
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

article_text = """Artificial Intelligence: An Overview

Artificial Intelligence, or AI, is the field of computer science focused on building systems
that can perform tasks which normally require human intelligence. These tasks include
understanding language, recognizing images, making decisions, and learning from experience.

Machine Learning is a subset of AI where systems learn patterns from data instead of following
hardcoded rules. Instead of programming every decision by hand, we feed the system many examples,
and it learns the underlying pattern on its own. This approach powers everything from spam filters
to recommendation systems.

Deep Learning takes Machine Learning further by using neural networks with many layers. These
layered networks can automatically discover complex patterns in large amounts of data, such as
recognizing faces in photos or understanding the meaning of a sentence.

Natural Language Processing, or NLP, is the branch of AI that focuses specifically on human
language. NLP powers chatbots, translation tools, and search engines. A key building block of
modern NLP is the embedding, which converts words or sentences into numeric vectors that capture
meaning.

Retrieval-Augmented Generation, or RAG, is a technique that combines a search system with a
language model. Instead of relying only on what a language model memorized during training, RAG
first retrieves relevant, up-to-date information from a document collection, then passes that
information to the model so it can generate a more accurate, grounded answer.

Vector Databases store embeddings and allow fast similarity search across millions of documents.
They are a core infrastructure piece behind modern RAG systems, chatbots, and recommendation
engines used by companies around the world today.
"""

second_doc_text = """Embeddings and Vector Search in Practice

Modern search engines increasingly rely on embeddings rather than simple keyword matching.
An embedding model converts a search query into the same vector space as the stored documents,
allowing the system to find results that match the meaning of the query, not just its exact words.

This is especially powerful for customer support systems, where a user might phrase a question very
differently from how it appears in a help article, yet still expect to find the right answer.
"""

third_doc_text = """Transformers and Large Language Models

The transformer is a neural network architecture built around a mechanism called self-attention.
Self-attention lets the model weigh how relevant every word in a sentence is to every other word,
which helps it capture context over long distances.

Large Language Models, or LLMs, are transformers trained on huge text datasets. They can write,
summarize, translate, and answer questions. However, they can also produce confident but incorrect
statements, which is one of the main reasons techniques like RAG are used to ground their answers.
"""


os.makedirs("docs", exist_ok=True)
docs = {
    "ai_overview.txt": article_text,
    "embeddings_in_practice.txt": second_doc_text,
    "transformers_and_llms.txt": third_doc_text,
}
for name, text in docs.items():
    with open(os.path.join("docs", name), "w", encoding="utf-8") as f:
        f.write(text)


client = chromadb.Client()
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.create_collection(name="rag_chunks", embedding_function=embedder)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", ". ", " ", ""]
)


def store(text, source_name):
    chunks = splitter.split_text(text)
    prefix = source_name.rsplit(".", 1)[0]
    collection.add(
        documents=chunks,
        metadatas=[{"source": source_name, "chunk_index": i, "char_count": len(c)}
                   for i, c in enumerate(chunks)],
        ids=[f"{prefix}_chunk_{i}" for i in range(len(chunks))],
    )
    print(f"Stored {len(chunks)} chunks from {source_name}")


# Store the first two documents
store(article_text, "ai_overview.txt")
store(second_doc_text, "embeddings_in_practice.txt")
before = collection.count()
print("Chunks before adding the third document:", before)


store(third_doc_text, "transformers_and_llms.txt")
print("Chunks after adding the third document:", collection.count())


metas = collection.get(include=["metadatas"])["metadatas"]
print("Chunks per source:", dict(Counter(m["source"] for m in metas)))

res = collection.query(
    query_texts=["why do LLMs need RAG"],
    n_results=3,
    include=["documents", "metadatas", "distances"],
)
print("\nQuery: why do LLMs need RAG")
for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
    print(f"{dist:.4f} | {meta['source']} | chunk #{meta['chunk_index']}")
    print(doc)
    print("-" * 60)
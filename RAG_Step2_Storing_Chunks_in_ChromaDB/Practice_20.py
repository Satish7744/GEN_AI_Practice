
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

client = chromadb.Client()
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.create_collection(name="rag_chunks", embedding_function=embedder)

splitter_300 = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", ". ", " ", ""]
)
splitter_150 = RecursiveCharacterTextSplitter(
    chunk_size=150, chunk_overlap=25, separators=["\n\n", "\n", ". ", " ", ""]
)


def store(splitter, text, source_name, id_prefix, version):
    chunks = splitter.split_text(text)
    collection.add(
        documents=chunks,
        metadatas=[{"source": source_name, "chunk_index": i, "char_count": len(c), "version": version}
                   for i, c in enumerate(chunks)],
        ids=[f"{id_prefix}_chunk_{i}" for i in range(len(chunks))],
    )
    print(f"{version}: stored {len(chunks)} chunks (ids start with '{id_prefix}_')")



store(splitter_300, article_text, "ai_overview.txt", "ai_overview", "v1_size300")
store(splitter_150, article_text, "ai_overview.txt", "ai_overview_v2", "v2_size150")
print("Total chunks in collection:", collection.count())


def compare(query, n=3):
    print(f"\n{'=' * 70}\nQUERY: {query}")
    for version in ["v1_size300", "v2_size150"]:
        res = collection.query(
            query_texts=[query],
            n_results=n,
            where={"version": version},
            include=["documents", "distances"],
        )
        print(f"\n  [{version}]")
        for cid, doc, dist in zip(res["ids"][0], res["documents"][0], res["distances"][0]):
            print(f"   {dist:.4f} | {cid} | {len(doc)} chars")
            print(f"      {doc[:110].replace(chr(10), ' ')}...")


compare("What is deep learning?")
compare("How does RAG work?")
compare("What do vector databases do?")


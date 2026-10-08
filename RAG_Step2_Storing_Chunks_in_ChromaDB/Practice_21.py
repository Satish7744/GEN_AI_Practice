
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter


def add_document(collection, splitter, text, source_name, id_prefix=None, extra_metadata=None):
    """Chunk `text`, build metadata + ids, and store everything in `collection`.

    Returns the list of ids that were stored.
    """
    id_prefix = id_prefix or source_name.rsplit(".", 1)[0]   # "ai_overview.txt" -> "ai_overview"
    doc_chunks = splitter.split_text(text)

    metadatas = []
    for i, chunk in enumerate(doc_chunks):
        meta = {"source": source_name, "chunk_index": i, "char_count": len(chunk)}
        if extra_metadata:
            meta.update(extra_metadata)
        metadatas.append(meta)

    ids = [f"{id_prefix}_chunk_{i}" for i in range(len(doc_chunks))]
    collection.add(documents=doc_chunks, metadatas=metadatas, ids=ids)
    print(f"Added {len(doc_chunks)} chunks from '{source_name}'. Total in collection: {collection.count()}")
    return ids


if __name__ == "__main__":
    client = chromadb.Client()
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.create_collection(name="rag_chunks", embedding_function=embedder)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", ". ", " ", ""]
    )

    doc_a = """Machine Learning is a subset of AI where systems learn patterns from data instead of
following hardcoded rules. Instead of programming every decision by hand, we feed the system many
examples, and it learns the underlying pattern on its own. This approach powers everything from
spam filters to recommendation systems.

Deep Learning takes Machine Learning further by using neural networks with many layers. These
layered networks can automatically discover complex patterns in large amounts of data."""

    doc_b = """Embeddings and Vector Search in Practice

Modern search engines increasingly rely on embeddings rather than simple keyword matching.
An embedding model converts a search query into the same vector space as the stored documents,
allowing the system to find results that match the meaning of the query, not just its exact words."""

    doc_c = """The transformer is a neural network architecture built around a mechanism called
self-attention. Self-attention lets the model weigh how relevant every word in a sentence is to
every other word, which helps it capture context over long distances."""


    add_document(collection, splitter, doc_a, "machine_learning.txt")
    add_document(collection, splitter, doc_b, "embeddings_in_practice.txt")
    add_document(collection, splitter, doc_c, "transformers.txt", extra_metadata={"topic": "nlp"})

  
    res = collection.query(
        query_texts=["how do search engines understand meaning"],
        n_results=2,
        include=["documents", "metadatas", "distances"],
    )
    print("\nQuery: how do search engines understand meaning")
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        print(f"{dist:.4f} | {meta['source']} | chunk #{meta['chunk_index']}")
        print(doc)
        print("-" * 60)

   
    res = collection.query(
        query_texts=["self-attention"],
        n_results=1,
        where={"source": "transformers.txt"},
        include=["documents", "metadatas"],
    )
    print("\nFiltered to transformers.txt:", res["documents"][0][0][:100], "...")
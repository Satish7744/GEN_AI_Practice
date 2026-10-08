import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi


documents = {
    "ai_overview.txt": """Artificial Intelligence: An Overview

Artificial Intelligence, or AI, is the field of computer science focused on building systems
that can perform tasks which normally require human intelligence.

Machine Learning is a subset of AI where systems learn patterns from data instead of following
hardcoded rules.

Deep Learning takes Machine Learning further by using neural networks with many layers to
automatically discover complex patterns in data.

Retrieval-Augmented Generation, or RAG, combines a search system with a language model so answers
are grounded in real retrieved documents instead of only the model's memorized training data.
""",
    "embeddings_in_practice.txt": """Embeddings and Vector Search in Practice

Modern search engines increasingly rely on embeddings rather than simple keyword matching, allowing
systems to find results that match the meaning of a query, not just its exact words.

Vector databases such as ChromaDB store these embeddings and allow fast similarity search across
thousands or millions of documents at once.
""",
    "product_catalog.txt": """Product Catalog Notes

The wireless noise-cancelling headphones are listed under product code SKU-4521 and are currently
our best-selling audio accessory.

The standing desk converter is listed under product code SKU-7788 and ships within three business
days.

Return requests for any product must reference the exact SKU code so our warehouse team can locate
the correct item quickly.
""",
}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=220, chunk_overlap=30, separators=["\n\n", "\n", ". ", " ", ""]
)

client = chromadb.Client()
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.create_collection(name="rag_hybrid_demo", embedding_function=embedder)


all_chunks, all_chunk_meta = [], []
bm25 = None

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())      # "SKU-4521" -> ["sku", "4521"]

def rebuild_bm25():
    global bm25
    bm25 = BM25Okapi([tokenize(c) for c in all_chunks])

def add_document(source, text):
    chunks = splitter.split_text(text)
    ids = [f"{source}_chunk_{i}" for i in range(len(chunks))]
    meta = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
    collection.add(documents=chunks, metadatas=meta, ids=ids)
    all_chunks.extend(chunks)
    all_chunk_meta.extend(meta)
    rebuild_bm25()          # IMPORTANT: BM25 does not see new chunks until it is rebuilt
    return len(chunks)

for src, txt in documents.items():
    add_document(src, txt)
print("Total chunks stored:", collection.count())


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "am", "do", "does", "did",
    "what", "whats", "which", "who", "how", "why", "when", "where", "can", "could", "would",
    "should", "you", "your", "me", "my", "i", "we", "our", "us", "it", "its", "this", "that",
    "of", "for", "to", "in", "on", "at", "by", "with", "about", "and", "or", "please", "tell",
    "s",
}

def remove_stopwords(query):
    return " ".join(t for t in tokenize(query) if t not in STOPWORDS)


def semantic_search(query, n_results=3):
    n = min(n_results, collection.count())
    results = collection.query(
        query_texts=[query], n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": d, "source": m["source"], "similarity": round(1 - dist, 4)}
        for d, m, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ]

def bm25_search(query, n_results=3, strip_stopwords=False):
    tokens = tokenize(query)
    if strip_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
    return [
        {"text": all_chunks[i], "source": all_chunk_meta[i]["source"], "bm25_score": round(float(scores[i]), 4)}
        for i in ranked if scores[i] > 0
    ]

def hybrid_search(query, n_results=3, k=60, min_similarity=0.0, strip_stopwords=False):
    semantic = [r for r in semantic_search(query, n_results=10) if r["similarity"] >= min_similarity]
    keyword = bm25_search(query, n_results=10, strip_stopwords=strip_stopwords)

    rrf, lookup = {}, {}
    for rank, r in enumerate(semantic):
        rrf[r["text"]] = rrf.get(r["text"], 0) + 1 / (k + rank + 1)
        lookup[r["text"]] = r
    for rank, r in enumerate(keyword):
        rrf[r["text"]] = rrf.get(r["text"], 0) + 1 / (k + rank + 1)
        lookup.setdefault(r["text"], r)

    ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:n_results]
    return [{"text": t, "source": lookup[t]["source"], "rrf_score": round(s, 5)} for t, s in ranked]

def retrieve_with_threshold(query, n_results=5, similarity_threshold=0.3):
    return [r for r in semantic_search(query, n_results=n_results)
            if r["similarity"] >= similarity_threshold]


KNOWLEDGE_BASE_TOPICS = sorted({m["source"] for m in all_chunk_meta})

def smart_retrieve(query, n_results=4, similarity_threshold=0.3):
    primary = retrieve_with_threshold(query, n_results=n_results, similarity_threshold=similarity_threshold)
    if primary:
        return {"status": "ok", "method": "semantic", "chunks": primary}

    hybrid = hybrid_search(query, n_results=n_results)
    if hybrid:
        return {"status": "ok", "method": "hybrid_fallback", "chunks": hybrid}

    return {
        "status": "empty", "method": None, "chunks": [],
        "message": "I don't know based on the provided documents. "
                   f"This knowledge base currently covers: {', '.join(KNOWLEDGE_BASE_TOPICS)}.",
    }
    
def smart_retrieve_v2(query, n_results=4, similarity_threshold=0.3, semantic_floor=0.15):
    
    primary = retrieve_with_threshold(query, n_results=n_results, similarity_threshold=similarity_threshold)
    if primary:
        return {"status": "ok", "method": "semantic", "chunks": primary}

   
    hybrid = hybrid_search(query, n_results=n_results, min_similarity=semantic_floor)
    if hybrid:
        return {"status": "ok", "method": "hybrid_fallback", "chunks": hybrid}

    
    cleaned = remove_stopwords(query)
    if cleaned and cleaned != " ".join(tokenize(query)):
        primary = retrieve_with_threshold(cleaned, n_results=n_results, similarity_threshold=similarity_threshold)
        if primary:
            return {"status": "ok", "method": "stopword_semantic", "chunks": primary, "cleaned_query": cleaned}

        hybrid = hybrid_search(cleaned, n_results=n_results, min_similarity=semantic_floor, strip_stopwords=True)
        if hybrid:
            return {"status": "ok", "method": "stopword_hybrid", "chunks": hybrid, "cleaned_query": cleaned}

   
    return {
        "status": "empty", "method": None, "chunks": [],
        "message": "I don't know based on the provided documents. "
                   f"This knowledge base currently covers: {', '.join(sorted({m['source'] for m in all_chunk_meta}))}.",
    }


tests = [
    "What is deep learning?",                                   # step 1
    "Tell me about SKU-4521",                                   # step 1 or 2
    "Could you please tell me what the SKU-7788 is about?",     # lots of filler words
    "What's the best recipe for chocolate cake?",               # out of scope
]

for q in tests:
    res = smart_retrieve_v2(q)
    print(f"\nQuery  : {q}")
    print(f"Status : {res['status']} | method: {res['method']}")
    if "cleaned_query" in res:
        print(f"Cleaned: {res['cleaned_query']}")
    if res["status"] == "empty":
        print(res["message"])
    for c in res["chunks"][:2]:
        print("   ", c["source"], "|", c["text"][:60].replace("\n", " "), "...")


cake = "What's the best recipe for chocolate cake?"
print("\nBM25 hits for the cake query, raw:     ", len(bm25_search(cake, n_results=10)))
print("BM25 hits for the cake query, stripped:", len(bm25_search(cake, n_results=10, strip_stopwords=True)))
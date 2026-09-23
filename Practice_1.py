from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

# A new, bigger "document" database spanning multiple topics
my_documents = [
    "Python is widely used for data science and machine learning.",
    "Neural networks are inspired by the structure of the human brain.",
    "A balanced diet with vegetables and protein supports good health.",
    "Regular exercise like running or swimming improves cardiovascular fitness.",
    "The Great Wall of China stretches thousands of kilometers.",
    "Tokyo is one of the most densely populated cities in the world.",
    "Cricket and football are two of the most popular sports in India.",
    "A home-cooked meal of rice and curry is a comfort food for many.",
    "Cloud computing lets companies rent servers instead of owning hardware.",
    "Climate change is causing more frequent extreme weather events."
]

my_doc_embeddings = model.encode(my_documents)

def semantic_search_custom(query, docs, doc_embeddings, top_k=3):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, doc_embeddings)[0]
    ranked_idx = scores.argsort()[::-1][:top_k]

    print(f"Query: {query}\n")
    for idx in ranked_idx:
        print(f"Score: {scores[idx]:.3f}  |  {docs[idx]}")
    print()

semantic_search_custom("How do I stay physically fit?", my_documents, my_doc_embeddings)
semantic_search_custom("Tell me about AI and neural networks", my_documents, my_doc_embeddings)
semantic_search_custom("What are some famous places to visit?", my_documents, my_doc_embeddings)
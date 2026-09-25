import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer('all-MiniLM-L6-v2')
model_mpnet = SentenceTransformer('all-mpnet-base-v2')
print("Both models loaded successfully!")

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

start = time.time()
minilm_embeddings = model.encode(my_documents)
minilm_time = time.time() - start

start = time.time()
mpnet_embeddings = model_mpnet.encode(my_documents)
mpnet_time = time.time() - start

print(f"MiniLM embedding shape: {minilm_embeddings.shape}  |  Time: {minilm_time:.3f}s")
print(f"MPNet  embedding shape: {mpnet_embeddings.shape}  |  Time: {mpnet_time:.3f}s")

def compare_models(query, docs, top_k=3):
    print(f"Query: {query}\n")

    q_mini = model.encode([query])
    scores_mini = cosine_similarity(q_mini, minilm_embeddings)[0]
    ranked_mini = scores_mini.argsort()[::-1][:top_k]

    q_mpnet = model_mpnet.encode([query])
    scores_mpnet = cosine_similarity(q_mpnet, mpnet_embeddings)[0]
    ranked_mpnet = scores_mpnet.argsort()[::-1][:top_k]

    print("MiniLM (fast, 384-dim):")
    for idx in ranked_mini:
        print(f"  {scores_mini[idx]:.3f}  |  {docs[idx]}")

    print("\nMPNet (accurate, 768-dim):")
    for idx in ranked_mpnet:
        print(f"  {scores_mpnet[idx]:.3f}  |  {docs[idx]}")
    print()

compare_models("How do I stay physically fit?", my_documents)
compare_models("What are some famous places to visit?", my_documents)

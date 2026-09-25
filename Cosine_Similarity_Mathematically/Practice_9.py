#Real sentence embeddings instead of toy vectors

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import math

def dot_product(A, B):
    return sum(a * b for a, b in zip(A, B))

def magnitude(A):
    return math.sqrt(sum(a ** 2 for a in A))

def cosine_similarity_manual(A, B):
    dp = dot_product(A, B)
    mag_A = magnitude(A)
    mag_B = magnitude(B)
    return dp / (mag_A * mag_B)

model = SentenceTransformer('all-MiniLM-L6-v2')

sentence_a = "I love learning about Artificial Intelligence."
sentence_b = "I enjoy studying Machine Learning."

embedding_a = model.encode(sentence_a)
embedding_b = model.encode(sentence_b)

# Manual calculation on real embeddings
manual_result = cosine_similarity_manual(embedding_a, embedding_b)

# sklearn calculation for comparison
sklearn_result = cosine_similarity([embedding_a], [embedding_b])[0][0]

print("Manual cosine similarity:", round(manual_result, 4))
print("scikit-learn result:     ", round(sklearn_result, 4))
print("Match?", round(manual_result, 4) == round(sklearn_result, 4))
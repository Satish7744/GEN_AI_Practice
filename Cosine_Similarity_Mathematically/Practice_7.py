#List comprehension version

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

vector_A = [2, 4, 1]
vector_B = [1, 3, 5]

result = cosine_similarity_manual(vector_A, vector_B)
print("Cosine Similarity (list comprehension version):", round(result, 4))
#Predict, then verify

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

# Prediction: [1,1,1] and [5,5,5] point in the exact same direction,
# just scaled by 5x — cosine similarity should be 1.0
vector_A = [1, 1, 1]
vector_B = [5, 5, 5]

result = cosine_similarity_manual(vector_A, vector_B)
print("Cosine Similarity:", round(result, 4))
print("Matches prediction of 1.0?", round(result, 4) == 1.0)


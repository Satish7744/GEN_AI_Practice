#Build a word similarity checker function

from gensim.models import Word2Vec

sentences = [
    ["python", "is", "a", "language"],
    ["java", "is", "a", "language"],
    ["python", "is", "easy"],
    ["java", "is", "easy"],
    ["python", "is", "powerful"],
    ["java", "is", "powerful"],
    ["c", "is", "a", "language"],
    ["c", "is", "powerful"],
]

model = Word2Vec(sentences, vector_size=10, window=2, min_count=1, workers=1)

def check_similarity(word1, word2):
    score = model.wv.similarity(word1, word2)
    print(f"Similarity between '{word1}' and '{word2}': {score:.3f}")

check_similarity('python', 'java')
check_similarity('python', 'c')
check_similarity('is', 'a')
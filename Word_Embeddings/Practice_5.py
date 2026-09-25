#Word2Vec with a bigger custom corpus

from gensim.models import Word2Vec

sentences = [
    ["dog", "is", "a", "loyal", "animal"],
    ["cat", "is", "a", "independent", "animal"],
    ["dog", "loves", "to", "play"],
    ["cat", "loves", "to", "sleep"],
    ["dog", "is", "friendly"],
    ["cat", "is", "friendly"],
    ["dog", "and", "cat", "are", "pets"],
]

model = Word2Vec(sentences, vector_size=10, window=2, min_count=1, workers=1)

print(model.wv.most_similar('dog', topn=3))
print(model.wv.most_similar('cat', topn=3))
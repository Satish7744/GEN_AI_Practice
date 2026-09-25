
#One-Hot Encoding with my own words
from sklearn.preprocessing import OneHotEncoder

words = [['red'], ['blue'], ['green'], ['yellow'], ['red'], ['blue']]

encoder = OneHotEncoder(sparse_output=False)
result = encoder.fit_transform(words)

print("words :", encoder.categories_[0])
print("Embeddings:")
print(result)
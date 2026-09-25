#Delete by category filter, confirm with count

import chromadb

client = chromadb.Client()
collection = client.create_collection(name="delete_by_category_demo")

collection.add(
    documents=[
        "Python is a popular programming language for AI.",
        "The weather today is sunny and warm.",
        "Machine learning models learn patterns from data.",
        "I enjoy playing football on weekends.",
        "Deep learning uses neural networks with many layers."
    ],
    metadatas=[
        {"category": "tech"},
        {"category": "weather"},
        {"category": "tech"},
        {"category": "sports"},
        {"category": "tech"}
    ],
    ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
)

print("Count before delete:", collection.count())

# Delete every document where category == "tech"
collection.delete(where={"category": "tech"})

print("Count after delete:", collection.count())
print("Remaining ids:", collection.get()["ids"])
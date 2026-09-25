#Add 5 documents, then upsert to update 2 and create 1 new

import chromadb

client = chromadb.Client()
collection = client.create_collection(name="upsert_demo")

# Adding 5 initial documents
collection.add(
    documents=[
        "Python is a popular programming language for AI.",
        "The weather today is sunny and warm.",
        "Machine learning models learn patterns from data.",
        "I enjoy playing football on weekends.",
        "Deep learning uses neural networks with many layers."
    ],
    ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
)

print("After initial add, count:", collection.count())


collection.upsert(
    documents=[
        "Python is a versatile programming language used in AI and web development.",
        "Machine learning models learn patterns from data and improve with experience.",
        "The Great Barrier Reef is the largest coral reef system in the world."
    ],
    ids=["doc1", "doc3", "doc6"]
)

print("After upsert, count:", collection.count())

result = collection.get(ids=["doc1", "doc3", "doc6"])
for doc_id, doc in zip(result["ids"], result["documents"]):
    print(f"{doc_id} -> {doc}")
#safe_add() function using upsert internally

import chromadb

client = chromadb.Client()
collection = client.create_collection(name="safe_add_demo")

def safe_add(collection, id, text):
    """Adds a document if the id is new, or updates it if the id already exists.
    Never raises a duplicate-id error, unlike collection.add()."""
    collection.upsert(
        documents=[text],
        ids=[id]
    )
    print(f"Upserted id='{id}'")

# First call creates doc1
safe_add(collection, "doc1", "Python is a popular programming language for AI.")

# Second call with the same id updates it instead of erroring
safe_add(collection, "doc1", "Python is widely used in AI, web development, and automation.")

# A new id just creates a new document
safe_add(collection, "doc2", "The stock market fluctuates based on investor sentiment.")

print("\nFinal count:", collection.count())
print("Final state:", collection.get()["ids"])
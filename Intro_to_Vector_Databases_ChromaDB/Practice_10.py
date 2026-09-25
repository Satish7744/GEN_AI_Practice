#My own 5 sentences in a new collection

import chromadb

client = chromadb.Client()
collection = client.create_collection(name="my_own_collection")

collection.add(
    documents=[
        "The chef prepared a delicious pasta dish.",
        "Basketball players need good stamina and agility.",
        "The new laptop has a fast processor and long battery life.",
        "Volcanoes erupt when magma pressure builds up underground.",
        "Reading books before bed can improve sleep quality."
    ],
    ids=["s1", "s2", "s3", "s4", "s5"]
)

print("Documents added! Total documents:", collection.count())

results = collection.query(
    query_texts=["What sports require physical fitness?"],
    n_results=2
)

for doc, distance in zip(results["documents"][0], results["distances"][0]):
    print(f"Distance: {distance:.4f}  |  {doc}")
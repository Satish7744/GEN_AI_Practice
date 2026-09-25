#Metadata categories + filtering across two categories


import chromadb

client = chromadb.Client()
collection = client.create_collection(name="categorized_collection")

collection.add(
    documents=[
        "The chef prepared a delicious pasta dish.",
        "A balanced diet includes fruits and vegetables.",
        "Basketball players need good stamina and agility.",
        "Marathon runners train for months before a race.",
        "The new laptop has a fast processor and long battery life.",
        "Cloud computing lets you rent servers instead of buying them.",
        "Volcanoes erupt when magma pressure builds up underground.",
        "Earthquakes are measured using the Richter scale."
    ],
    metadatas=[
        {"category": "food"},
        {"category": "food"},
        {"category": "sports"},
        {"category": "sports"},
        {"category": "technology"},
        {"category": "technology"},
        {"category": "science"},
        {"category": "science"}
    ],
    ids=["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8"]
)

print("Documents added! Total documents:", collection.count())

# Filter within "food" category
print("\n--- Query filtered to 'food' ---")
results_food = collection.query(
    query_texts=["What should I eat for a healthy lifestyle?"],
    n_results=2,
    where={"category": "food"}
)
for doc, distance in zip(results_food["documents"][0], results_food["distances"][0]):
    print(f"Distance: {distance:.4f}  |  {doc}")

# Filter within "technology" category
print("\n--- Query filtered to 'technology' ---")
results_tech = collection.query(
    query_texts=["Tell me about computers and servers"],
    n_results=2,
    where={"category": "technology"}
)
for doc, distance in zip(results_tech["documents"][0], results_tech["distances"][0]):
    print(f"Distance: {distance:.4f}  |  {doc}")
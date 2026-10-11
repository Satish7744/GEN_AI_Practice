from Practice_28 import collection, store_document, rag_answer

third_doc = """Transformers and Large Language Models

The transformer is a neural network architecture built around a mechanism called self-attention.
Self-attention lets the model weigh how relevant every word in a sentence is to every other word,
which helps it capture context over long distances.

Large Language Models, or LLMs, are transformers trained on huge text datasets. They can write,
summarize, translate, and answer questions. However, they can also produce confident but incorrect
statements, which is why techniques like RAG are used to ground their answers.
"""

store_document("transformers_and_llms.txt", third_doc)
print("Total chunks now:", collection.count())

for q in ["What is self-attention?",
          "Why do LLMs sometimes give wrong answers?",
          "What is the best recipe for chocolate cake?"]:      
    print("\nQuestion:", q)
    print(rag_answer(q))
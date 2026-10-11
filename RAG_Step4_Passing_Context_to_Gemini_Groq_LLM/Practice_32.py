from Practice_28 import retrieve_context, rag_answer

q2 = "How do embeddings and vector databases help RAG find useful information?"

for n in [4, 1]:
    chunks = retrieve_context(q2, n_results=n)
    print("=" * 60)
    print(f"n_results={n} -> {len(chunks)} chunk(s) retrieved")
    for c in chunks:
        print(f"   [{c['similarity']}] {c['source']} #{c['chunk_index']}: {c['text'][:70]}...")
    print("\nAnswer:\n" + rag_answer(q2, n_results=n))
    print()
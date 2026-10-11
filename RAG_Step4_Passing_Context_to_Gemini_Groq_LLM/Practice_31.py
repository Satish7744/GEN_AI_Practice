from Practice_28 import rag_answer

BULLET_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context. "
    "If the answer is not in the context, reply exactly: \"I don't know based on the provided documents.\" "
    "Format rules: respond with EXACTLY 3 bullet points, each starting with '- ' and each one short "
    "sentence. Write nothing before or after the bullets, except a final line 'Sources: <file names>'."
)

for q in ["What is deep learning?",
          "How do embeddings and vector databases help RAG find useful information?"]:
    answer = rag_answer(q, system_prompt=BULLET_PROMPT, show_sources=False)
    bullets = sum(1 for line in answer.splitlines() if line.strip().startswith("- "))
    print(f"\nQuestion: {q}\n{answer}\nBullet count: {bullets}")
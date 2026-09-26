#Compare sentences_per_chunk=1 vs sentences_per_chunk=4

import re

article_text = """Artificial Intelligence, or AI, is the field of computer science focused on building systems
that can perform tasks which normally require human intelligence. These tasks include
understanding language, recognizing images, making decisions, and learning from experience.
Machine Learning is a subset of AI where systems learn patterns from data instead of following
hardcoded rules. Instead of programming every decision by hand, we feed the system many examples,
and it learns the underlying pattern on its own. This approach powers everything from spam filters
to recommendation systems. Deep Learning takes Machine Learning further by using neural networks
with many layers. These layered networks can automatically discover complex patterns in large
amounts of data, such as recognizing faces in photos or understanding the meaning of a sentence."""

def sentence_chunking(text, sentences_per_chunk=2):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s]

    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)
    return chunks

for n in [1, 4]:
    chunks = sentence_chunking(article_text, sentences_per_chunk=n)
    print(f"\nsentences_per_chunk={n} -> {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"  --- Chunk {i+1} ---")
        print(" ", chunk)
        
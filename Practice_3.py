# 5 Q&A pairs for our FAQ bot
faq = [
    ("What is machine learning?", "Machine learning is a field of AI where models learn patterns from data instead of being explicitly programmed."),
    ("How do I install Python?", "Download Python from python.org and follow the installer instructions for your operating system."),
    ("What is an embedding?", "An embedding is a numeric vector that represents the meaning of a piece of text, image, or other data."),
    ("How much does Hugging Face cost?", "Many Hugging Face models, including sentence-transformers models, are free and open-source to use."),
    ("What is cosine similarity?", "Cosine similarity measures how close two vectors are in direction, giving a score between -1 and 1."),
]

faq_questions = [q for q, a in faq]
faq_answers = [a for q, a in faq]

# Encode all FAQ questions once
faq_embeddings = model.encode(faq_questions)

def faq_bot(user_question, threshold=0.45):
    query_embedding = model.encode([user_question])
    scores = cosine_similarity(query_embedding, faq_embeddings)[0]

    best_idx = scores.argmax()
    best_score = scores[best_idx]

    print(f"You asked: {user_question}")
    if best_score < threshold:
        print(f"Bot: Sorry, I don't have an answer for that. (best match score: {best_score:.3f})")
    else:
        print(f"Bot: {faq_answers[best_idx]}")
        print(f"(matched: '{faq_questions[best_idx]}', score: {best_score:.3f})")
    print()

# Try it with questions phrased differently from the stored ones
faq_bot("How can I set up Python on my laptop?")
faq_bot("What's a text vector used for?")
faq_bot("Is Hugging Face free to use?")
faq_bot("What's the weather like today?")  # should trigger the 'don't know' fallback
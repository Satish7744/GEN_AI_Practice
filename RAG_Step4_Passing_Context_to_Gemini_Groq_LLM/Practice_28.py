# Shared Setup

import os
import getpass
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from google import genai
from google.genai import types

load_dotenv()

def get_key(name):
    key = os.environ.get(name, "")
    if not key:
        key = getpass.getpass(f"Enter {name} (press Enter to skip): ").strip()
    return key

GROQ_API_KEY = get_key("GROQ_API_KEY")
GEMINI_API_KEY = get_key("GEMINI_API_KEY")

if not GROQ_API_KEY and not GEMINI_API_KEY:
    raise SystemExit("No API key provided. Add GROQ_API_KEY to your .env file and run again.")

default_provider = "groq" if GROQ_API_KEY else "gemini"
print("Using provider:", default_provider)


documents = {
    "ai_overview.txt": """Artificial Intelligence: An Overview

Artificial Intelligence, or AI, is the field of computer science focused on building systems
that can perform tasks which normally require human intelligence. These tasks include
understanding language, recognizing images, making decisions, and learning from experience.

Machine Learning is a subset of AI where systems learn patterns from data instead of following
hardcoded rules. Instead of programming every decision by hand, we feed the system many examples,
and it learns the underlying pattern on its own.

Deep Learning takes Machine Learning further by using neural networks with many layers. These
layered networks can automatically discover complex patterns in large amounts of data, such as
recognizing faces in photos or understanding the meaning of a sentence.

Retrieval-Augmented Generation, or RAG, is a technique that combines a search system with a
language model. Instead of relying only on what a language model memorized during training, RAG
first retrieves relevant, up-to-date information from a document collection, then passes that
information to the model so it can generate a more accurate, grounded answer.
""",
    "embeddings_in_practice.txt": """Embeddings and Vector Search in Practice

Modern search engines increasingly rely on embeddings rather than simple keyword matching.
An embedding model converts a search query into the same vector space as the stored documents,
allowing the system to find results that match the meaning of the query, not just its exact words.

This is especially powerful for customer support systems, where a user might phrase a question very
differently from how it appears in a help article, yet still expect to find the right answer.

Vector databases such as ChromaDB store these embeddings and allow fast similarity search across
thousands or millions of documents at once.
""",
}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=250, chunk_overlap=40, separators=["\n\n", "\n", ". ", " ", ""]
)

client = chromadb.Client()
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.create_collection(name="rag_chunks", embedding_function=embedder)

def store_document(source, text):
    chunks = splitter.split_text(text)
    collection.add(
        documents=chunks,
        metadatas=[{"source": source, "chunk_index": i} for i in range(len(chunks))],
        ids=[f"{source}_chunk_{i}" for i in range(len(chunks))],
    )

for source, text in documents.items():
    store_document(source, text)
print("Total chunks stored:", collection.count())


def retrieve_context(query, n_results=4, similarity_threshold=0.25):
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    context_chunks = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        similarity = 1 - dist
        if similarity >= similarity_threshold:
            context_chunks.append({
                "text": doc, "source": meta["source"],
                "chunk_index": meta["chunk_index"], "similarity": round(similarity, 4),
            })
    return context_chunks


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context. "
    "If the answer is not in the context, reply exactly: \"I don't know based on the provided documents.\" "
    "Keep the answer short and clear. At the end, mention the source file name(s) you used."
)

def build_user_prompt(query, context_chunks):
    context_text = "\n\n".join(
        f"[Source: {c['source']}, chunk {c['chunk_index']}]\n{c['text']}" for c in context_chunks
    )
    return f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"


GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-20b"]
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash-lite"]

def ask_groq(user_prompt, system_prompt=SYSTEM_PROMPT, temperature=0.1):
    if not GROQ_API_KEY:
        return "[Groq skipped: no API key provided]"
    groq_client = Groq(api_key=GROQ_API_KEY)
    last_error = None
    for model_name in GROQ_MODELS:
        try:
            response = groq_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                temperature=temperature, max_tokens=600,
            )
            answer = response.choices[0].message.content
            if answer:
                return answer.strip()
        except Exception as e:
            last_error = e
    return f"[Groq error: {last_error}]"

def ask_gemini(user_prompt, system_prompt=SYSTEM_PROMPT, temperature=0.1):
    if not GEMINI_API_KEY:
        return "[Gemini skipped: no API key provided]"
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    last_error = None
    for model_name in GEMINI_MODELS:
        try:
            response = gemini_client.models.generate_content(
                model=model_name, contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature, max_output_tokens=1500,
                ),
            )
            if response.text:
                return response.text.strip()
        except Exception as e:
            last_error = e
    return f"[Gemini error: {last_error}]"

ask_llm = ask_gemini if default_provider == "gemini" else ask_groq

def rag_answer(query, n_results=4, system_prompt=SYSTEM_PROMPT, temperature=0.1, show_sources=True):
    context_chunks = retrieve_context(query, n_results=n_results)
    if not context_chunks:
        return "I don't know based on the provided documents. (No relevant chunks were retrieved.)"

    answer = ask_llm(build_user_prompt(query, context_chunks),
                     system_prompt=system_prompt, temperature=temperature)
    if show_sources:
        sources = sorted({f"{c['source']} #{c['chunk_index']}" for c in context_chunks})
        answer += "\n\nRetrieved from: " + ", ".join(sources)
    return answer
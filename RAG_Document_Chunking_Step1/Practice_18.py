#Running all 5 strategies on your own .txt file


import os
import re
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
import tiktoken

# Point this to your own file (create a .txt file with some paragraphs first)
file_path = "my_document.txt"

if not os.path.exists(file_path):
    # Creates a placeholder file so the script runs even if you haven't made one yet
    with open(file_path, "w") as f:
        f.write("""Replace this with your own multi-paragraph text.

Add a few paragraphs about any topic you like.

The more text you add, the more interesting the chunking comparison will be.
""")
    print(f"No {file_path} found — created a placeholder. Edit it with your own content and rerun.")

with open(file_path, "r") as f:
    my_text = f.read()

print("Loaded document, total characters:", len(my_text))

# Strategy 1: Fixed-size (from scratch)
def fixed_size_chunking(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

fixed_chunks = fixed_size_chunking(my_text)

# Strategy 2: CharacterTextSplitter
char_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=300, chunk_overlap=50)
char_chunks = char_splitter.split_text(my_text)

# Strategy 3: RecursiveCharacterTextSplitter
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", ". ", " ", ""]
)
recursive_chunks = recursive_splitter.split_text(my_text)

# Strategy 4: Sentence-based
def sentence_chunking(text, sentences_per_chunk=2):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s]
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunks.append(" ".join(sentences[i:i + sentences_per_chunk]))
    return chunks

sentence_chunks = sentence_chunking(my_text)

# Strategy 5: Token-based
encoding = tiktoken.get_encoding("cl100k_base")

def token_chunking(text, max_tokens=60, overlap_tokens=10):
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunks.append(encoding.decode(tokens[start:end]))
        start += max_tokens - overlap_tokens
    return chunks

token_chunks = token_chunking(my_text)

# Compare all 5
print("\n--- Comparison ---")
print(f"Fixed-size (scratch):            {len(fixed_chunks)} chunks")
print(f"CharacterTextSplitter:            {len(char_chunks)} chunks")
print(f"RecursiveCharacterTextSplitter:   {len(recursive_chunks)} chunks")
print(f"Sentence-based:                   {len(sentence_chunks)} chunks")
print(f"Token-based:                      {len(token_chunks)} chunks")
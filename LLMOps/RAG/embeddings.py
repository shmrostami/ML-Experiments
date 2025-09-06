# Embedding with llama3.2:3b
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# --------------------
# Setup
# --------------------
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
os.environ.setdefault("HTTPX_NO_PROXY", "127.0.0.1,localhost")

# persist_dir = "llama-emb"
# embeddings = OllamaEmbeddings(model="llama3.2:3b")

persist_dir = "nomic-emb"
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load documents only if we need to create embeddings
if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
    print("⚡ No existing DB found. Creating new embeddings...")

    # Load and split text
    loader = TextLoader("facts.txt")
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(docs)

    # Create Chroma DB and persist
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
else:
    print("✅ Found existing DB. Loading without re-embedding...")
    db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

# --------------------
# Run a test query
# --------------------
query = "What is an interesting fact about the English language?"
results = db.similarity_search(query, k=3)

for i, result in enumerate(results, start=1):
    print(f"\nResult {i}:")
    print(result.page_content)
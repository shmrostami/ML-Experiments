from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Split text into chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0
)

# Load text file
loader = TextLoader("facts.txt")
docs = loader.load_and_split(text_splitter=text_splitter)

# Initialize embeddings with Ollama
model_name = "llama3.2:3b"
embeddings = OllamaEmbeddings(model=model_name)

# Create or load Chroma vector store
db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="emb"  # stored in ./emb
)

# # load Chroma vecto store
# db = Chroma(persist_directory="emb", embedding_function=embeddings)

# Run similarity search
results = db.similarity_search(
    "What is an interesting fact about the English language?"
)

# Print results
for result in results:
    print("\n")
    print(result.page_content)

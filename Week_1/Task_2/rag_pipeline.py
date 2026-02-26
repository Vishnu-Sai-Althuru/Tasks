import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma

# -----------------------------
# 1. Load PDF safely
# -----------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(base_dir, "Data", "AI-NOTES-UNIT-1.pdf")

loader = PyPDFLoader(pdf_path)
documents = loader.load()

# -----------------------------
# 2. Split into chunks
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
 chunk_size=500,
 chunk_overlap=50
)
chunks = splitter.split_documents(documents)

# -----------------------------
# 3. Create embeddings
# -----------------------------
embedding = OllamaEmbeddings(model="nomic-embed-text:latest")

# -----------------------------
# 4. Store in vector DB
# -----------------------------
vectordb = Chroma.from_documents(chunks, embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# connecting Ollama
llm = OllamaLLM(model="llama3", temperature=0)

## Context

## Query
question = "What is AI?"

docs = retriever.invoke(question)
context = "\n".join([doc.page_content for doc in docs])

prompt = f"""
You are a factual AI assistant.
Answer ONLY from the context below.
If the answer is not present, say "Not found".

Context:
{context}

Question:
{question}
"""

response = llm.invoke(prompt)
print("\n--- RESPONSE ---\n")
print(response)
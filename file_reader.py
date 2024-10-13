from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain


loader = PyPDFLoader("/content/drive/MyDrive/FE-Delhi 17 Aug 2024.pdf")
data = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = [t.page_content for t in text_splitter.split_documents(data)]

# Set up the embedding function
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding = HuggingFaceEmbeddings(model_name=model_name)

# Initialize ChromaDB vector store
vectorstore = Chroma(embedding_function=embedding, collection_name="ppp")

# Add texts and their embeddings to the ChromaDB vector store
vectorstore.add_texts(texts=texts)

query = "when will conduct to j&k vote for a new assembly"
docsearch = vectorstore.similarity_search(query, k=5)  # Adjust k to the number of results you want

# Print the search results
for result in docsearch:
    print(result)

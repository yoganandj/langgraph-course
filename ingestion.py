from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [ UnstructuredLoader(web_url=url, chunking_strategy="basic", max_characters=1000000).load() for url in urls ]
print(f"Loaded {len(docs)} documents from {len(urls)} URLs.")
docs_list = [doc for sublist in docs for doc in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

docs_splits = text_splitter.split_documents(docs_list)

print(f"Split {len(docs_list)} documents into {len(docs_splits)} chunks.")

vector_store = Chroma.from_documents(
    documents = docs_splits,
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    persist_directory = "./.chroma",
    collection_name = "rag-chroma",
)


retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
).as_retriever()
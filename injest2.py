"""
## Scope for further optimization - https://www.neum.ai/post/llm-spreadsheets
"""

import csv
from langchain.docstore.document import Document 
from langchain.document_loaders import CSVLoader
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
# from langchain.pipelines.preprocessors import (
#     LowercasePreprocessor,
#     PunctuationPreprocessor,
#     StopWordPreprocessor,
# )
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant

# Define CSV paths and columns
qns_column_name = "QUESTIONS"
ans_column_name = "ANSWERS"
# csv_path = "example.csv"
csv_path = ".\\data\\faqs_data.csv"

# Define Hugging Face model and preprocessors
# model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_name = "BAAI/bge-large-en"
model_kwargs = {"device": 'cpu'}
encode_kwargs = {"normalize_embeddings":False}
# preprocessors = [
#     LowercasePreprocessor(),
#     PunctuationPreprocessor(),
#     StopWordPreprocessor(),
# ]

# Create loader, splitter, and embedder
# loader = CSVLoader(
#     file_path=csv_path,
#     source_column="LINKS",
#     csv_args={
#         "delimiter": ",",
#         "quotechar": '"',
#         "fieldnames": [qns_column_name, ans_column_name],
#     },
# )

# documents = loader.load()

cols_to_embed = [qns_column_name, ans_column_name]
cols_to_metadata = [qns_column_name, ans_column_name,"LINKS"]

docs = []
with open(csv_path, newline="", encoding='utf-8') as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for i, row in enumerate(csv_reader):
        to_metadata = {col: row[col] for col in cols_to_metadata if col in row}
        values_to_embed = {k: row[k] for k in cols_to_embed if k in row}
        to_embed = "\n".join(f"{k.strip()}: {v.strip()}" if v is not None else "" for k, v in values_to_embed.items())
        newDoc = Document(page_content=to_embed, metadata=to_metadata)
        docs.append(newDoc)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)
texts = text_splitter.split_documents(docs)
# texts = text_splitter.split_documents(documents)
# loader = CSVLoader(path=csv_path, columns=[qns_column_name, ans_column_name])
# splitter = SentenceTextSplitter()
# embedder = HuggingFaceEmbeddings(model_name=model_name)

embeddings = HuggingFaceBgeEmbeddings(
    model_kwargs = model_kwargs,
    model_name = model_name,
    encode_kwargs = encode_kwargs
)

print("Embedding Model Loaded.....")

url = "http://localhost:6333"
collection_name = "test_csv_db"

qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url = url,
    prefer_grpc = False,
    collection_name = collection_name
)

print("Qdrant VectorDB created........")

# Preprocess, split, and embed data
# texts = [text for row in loader for text in row]
# preprocessing_pipeline = Pipeline(preprocessors)
# preprocessed_texts = preprocessing_pipeline.apply(texts)
# split_texts = splitter.split_documents(preprocessed_texts)
# embeddings = embedder.encode(split_texts)

# Connect to Qdrant and create collection
# qdrant_url = "http://localhost:6334"
# client = QdrantClient(url=qdrant_url)
# collection_name = "my_collection"
# collection = Qdrant(client=client, collection_name=collection_name)

# # Create payloads for Qdrant (optional)
# payloads = [{"qns": row[0], "ans": row[1]} for row in loader]

# # Store embeddings and payloads in Qdrant
# collection.add(embeddings=embeddings, payloads=payloads)

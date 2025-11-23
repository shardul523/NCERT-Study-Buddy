import json
import os
import shutil
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore, EncoderBackedStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
VECTOR_DB_PATH = "/kaggle/working/chroma_db"
DOC_STORE_PATH = "/kaggle/working/doc_store"

class PickleSerializer:
    def dumps(self, obj): return pickle.dumps(obj)
    def loads(self, data): return pickle.loads(data)

# Cleanup for fresh run
if os.path.exists(VECTOR_DB_PATH): shutil.rmtree(VECTOR_DB_PATH)
if os.path.exists(DOC_STORE_PATH): shutil.rmtree(DOC_STORE_PATH)

with open('/kaggle/input/ncert-history-textbooks-chunks/sections.json', 'r', encoding='utf-8') as file:
    sections_data = json.load(file)

with open('/kaggle/input/ncert-history-textbooks-chunks/paragraphs.json', 'r', encoding='utf-8') as file:
    paragraphs_data = json.load(file)

# Use BGE-M3 for better entity retrieval and multi-granularity support
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'}, # Use 'cuda' if you have a GPU
    encode_kwargs={'normalize_embeddings': True}
)

vector_store = Chroma(
    collection_name='chunked_paragraphs',
    embedding_function=embeddings,
    persist_directory=VECTOR_DB_PATH
)

# store = LocalFileStore(DOC_STORE_PATH)
store = EncoderBackedStore(
    store=LocalFileStore(DOC_STORE_PATH),
    key_encoder=lambda x: str(x),
    value_deserializer=pickle.loads,
    value_serializer=pickle.dumps
)

print('Writing sections into Docstore')
section_docs_to_store = []

for id, data in sections_data.items():
    section_content = ""
    for chunk in paragraphs_data:
        if chunk['parent_id'] == id:
            section_content += chunk['content'] + '\n'
    doc = Document(page_content=section_content, metadata=data)
    section_docs_to_store.append((id, doc))

store.mset(section_docs_to_store)

print('Wiring paragraphs into Vector store')
para_docs_to_store = []

for chunk in paragraphs_data:
    metadata = {
        'doc_id': chunk['parent_id'],
        'chunk_id': chunk['id']
    }

    doc = Document(page_content=chunk['content'], metadata=metadata)
    para_docs_to_store.append(doc)

vector_store.add_documents(para_docs_to_store)

retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=store,
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0),
    parent_splitter=None
)

# query = 'Plassey'

# print('\nQuerying: {query}')
# results = retriever.invoke(query)

# if results:
#     print(f'Retrieved parent: {results[0].page_content}')
# else:
#     print('No results found')

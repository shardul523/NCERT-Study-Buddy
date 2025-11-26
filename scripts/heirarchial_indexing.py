import json
import uuid
import pickle
from typing import List, Dict

# LangChain Imports
from flashrank import Ranker, RerankRequest  # <--- CRITICAL IMPORT
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.storage import LocalFileStore, EncoderBackedStore
from langchain_classic.retrievers import ParentDocumentRetriever, EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.document_compressors import FlashrankRerank
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter


# ==========================================
# 1. SETUP
# ==========================================

with open('data/paragraphs.json', 'r', encoding='utf-8') as file:
    children_data = json.load(file)

with open('data/sections.json', 'r', encoding='utf-8') as file:
    parents_data = json.load(file)

VECTOR_DB_PATH = "data/chroma_db"
DOC_STORE_PATH = "data/doc_store"

# ==========================================
# 2. INGESTION: The "BYO-Chunks" Logic
# ==========================================
def build_retriever():
    print("--- Building Stores ---")
    
    # A. Setup Embedding Model (Optimized for Study Data)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # B. Setup Stores
    # Parent Store: Holds the full text (RAM or Redis/FileStore for persistence)
    parent_store = EncoderBackedStore(
        store=LocalFileStore(DOC_STORE_PATH),
        key_encoder=lambda x: str(x),
        value_deserializer=pickle.loads,
        value_serializer=pickle.dumps
    )
    # Child Store: The Vector Database
    vectorstore = Chroma(
        collection_name="study_chunks",
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_PATH # Use disk persistence
    )

    # C. Ingest Parents (The "Answers")
    # We must format them as (key, value) pairs for mset
    parent_docs_to_store = []
    parent_docs_list_for_bm25 = [] # Keep a list for BM25 indexing later

    for p_id, p_data in parents_data.items():
        doc = Document(
            page_content=p_data["content"],
            metadata=p_data["metadata"]
        )
        parent_docs_to_store.append((p_id, doc))
        parent_docs_list_for_bm25.append(doc)
    
    parent_store.mset(parent_docs_to_store)

    # D. Ingest Children (The "Search Index")
    child_docs_to_store = []
    for c_data in children_data:
        doc = Document(
            page_content=c_data["content"],
            metadata=c_data["metadata"]
        )
        # CRITICAL: Link Child -> Parent
        # The PDR looks for the key 'doc_id' in metadata to find the parent
        doc.metadata["doc_id"] = c_data["parent_id"]
        child_docs_to_store.append(doc)
    
    # Add children to VectorDB
    # Note: We use add_documents directly on the vectorstore, bypassing the 
    # automatic splitting of the PDR since we already have chunks.
    vectorstore.add_documents(child_docs_to_store)

    print(f"--- Ingested {len(parent_docs_to_store)} Parents and {len(child_docs_to_store)} Children ---")

    # ==========================================
    # 3. RETRIEVER ASSEMBLY
    # ==========================================
    
    # A. The Vector Retriever (Parent Document Retriever)
    # Since we manually populated the stores, we just initialize the class
    # to act as the interface.
    pdr_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=parent_store,
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0), # Disable auto-splitting
        parent_splitter=None # Disable auto-splitting
    )
    # Force it to return the Top 5 Parents
    pdr_retriever.search_kwargs = {"k": 5}

    # B. The Keyword Retriever (BM25)
    # We index the PARENT documents here. 
    # Why? So that both retrievers return full sections to the Reranker.
    bm25_retriever = BM25Retriever.from_documents(parent_docs_list_for_bm25)
    bm25_retriever.k = 5

    # C. The Hybrid Retriever (Ensemble)
    # Weights: 0.6 Vector (Concepts) / 0.4 Keyword (Exact Terms)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[pdr_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )

    # D. The Reranker (FlashRank)
    # The "Judge" that picks the absolute best answer from the mixed list
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
    
    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

    return final_retriever

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    retriever = build_retriever()
    
    query = "Tell me about trade in the Harappan civilization"
    
    # Example 1: Standard Search
    print(f"\nQuery: {query}")
    docs = retriever.invoke(query)
    
    for i, doc in enumerate(docs):
        print(f"\n[Result {i+1}] (Score: {doc.metadata.get('relevance_score', 'N/A')}):")
        print(f"Source: {doc.metadata.get('book')} - Ch {doc.metadata.get('chapter')}")
        print(f"Content Preview: {doc.page_content[:100]}...")

    # Example 2: Metadata Filtered Search (For your Self-Ask Agent)
    # NOTE: To pass filters to the PDR inside an Ensemble/Compression pipeline is tricky.
    # The robust way for your Agent is to access the underlying PDR directly when filtering is needed.
    
    print("\n--- Testing Filtered Access (Direct PDR) ---")
    # Access the vector retriever inside the stack
    # 1. Get the Ensemble Retriever from the generic wrapper
    # We use getattr to safely get it, defaulting to None if missing
    ensemble = getattr(retriever, "base_retriever", None)

    # 2. Get the list of retrievers from the Ensemble
    # The EnsembleRetriever stores them in the attribute 'retrievers'
    underlying_retrievers = getattr(ensemble, "retrievers", [])

    # 3. Access the first one (Vector Store / PDR)
    pdr = underlying_retrievers[0]
    # pdr = retriever.base_retriever.retrievers[0] 
    
    # We filter on the CHILD chunks (vector store), but get PARENT docs back
    filtered_docs = pdr.vectorstore.similarity_search(
        query, 
        k=3, 
        filter={"book": "History 6"} # This matches the metadata we added to children
    )
    # Note: similarity_search returns children. To get parents with filtering, 
    # you use the pdr.invoke with pre-filtering, but PDR doesn't natively support 
    # dynamic filters easily in invoke(). 
    
    # PRO TIP for Agents:
    # Use the 'vectorstore' directly for filtered checks if the Agent knows the book:
    print(f"Direct Vector Hit: {filtered_docs[0].page_content}")

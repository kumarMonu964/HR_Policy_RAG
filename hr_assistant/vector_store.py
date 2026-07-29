"""STEP 4: Stores the vectors embeddings in FAISS for semantic searching"""

import os
from langchain_community.vectorstores import FAISS

from hr_assistant import config
from hr_assistant.embeddings import get_embeddings_model


# building a vector store using FAISS
def build_vector_store(chunks):
    """embedd every chunk and build
    a searchable FAISS index in memory.
    """
    embeddings_model = get_embeddings_model()
    return FAISS.from_documents(chunks,embeddings_model)

# Save the vector store
def save_vector_store(vector_store , path:str = config.vector_store_path)-> None:
    """
    Save the FAISS index to disk
    so we dont have to rebuild it every time.
    """
    vector_store.save_local(path)

# load the vector store
def load_vector_store(path: str = config.vector_store_path):
    """
    load a previously saved FAISS index frmo disk.
    """
    embeddings_model = get_embeddings_model()
    #allow_dangerous_deserialization=Trueis safe here because we only ever load
    # an index that same app created and saved
    return FAISS.load_local(path, embeddings_model,allow_dangerous_deserialization=True)

# check if vector store exists
def vector_store_exists(path:str = config.vector_store_path) -> bool:
    """
    check if vector store exists on disk
    """
    return os.path.exists(os.path.join(path,"index.faiss"))


# function to get the retriever
def get_retriever(vector_store, k: int = config.top_k_results):
    """
    turn a vector store into a retriever that returns the top-k
    matching chunks
    """
    return vector_store.as_retriever(search_kwargs= {"k":k})


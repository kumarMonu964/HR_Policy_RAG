"""All the settings for the app live here in one place"""


import os
from dotenv import load_dotenv

load_dotenv()

## secret keys

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
jina_api_key = os.getenv("JINA_API_KEY")

## define the data path, and vector store path

## vector stores are of two types: 
# In-Memory (vanishes after shutting down vs code) and 
# Persistent Memory (stays permanent in the folder in the project)
# Cloud Memory
data_file_path = os.path.join("data","hr_policy.txt")

vector_store_path = os.path.join("data","faiss_index")


## Models
# LLM and Embedding Model

llm_model_name = "openai/gpt-oss-20b"
embedding_model_name = "jina-embeddings-v2-base-en"

## Chunk/ Text-Splitting Config

chunk_size = 500
chunk_overlap = 50

## Retrieval Results
top_k_results = 3

## System instructions
system_prompt = "\n".join([
    "You are a friendly HR assistant.",
    "Always use the search_hr_policy tool to look up the facts before answering.",
    "If the answer isn't in the search results, say you don't know, no guessing allowed."
])

## Check API keys

def check_api_keys() -> None:
    """Stop early with a clear message if a required API key is missing"""
    if not GROQ_API_KEY:
        raise ValueError("Missing GROQ_API_KEY.")
    if not jina_api_key:
        raise ValueError("Missing JINA_API_KEY.")
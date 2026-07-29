"""STEP 8: wires all the compenent together into one ready-to-use agent
This is the single entry point that main.py (CLI) and app.py (streamlit)
both call. Each step is handled by its own small module.
"""

from hr_assistant import config
from hr_assistant.agent import create_hr_agent
from hr_assistant.document_loader import load_document
from hr_assistant.llm import get_llm
from hr_assistant.data_splitter import split_into_chunks
from hr_assistant.tools import create_search_tool
from hr_assistant.vector_store import (
    build_vector_store,
    save_vector_store,
    load_vector_store,
    vector_store_exists,
    get_retriever
)

def build_vector_store_for_document(file_path: str = config.data_file_path):
    if vector_store_exists():
        print("Found the saved vector store on disk, loading it (fast, no re-embedding required.)")
        return load_vector_store()

    print("No saved vector store found, building one from scratch...")
    documents = load_document(file_path)
    chunks = split_into_chunks(documents)
    print(f"Loaded '{file_path}' and split it into {len(chunks)} chunks.")

    vector_store = build_vector_store(chunks)
    save_vector_store(vector_store)
    print("Vector store built and saved to disk for next time.")
    return vector_store

def build_hr_assistant(file_path: str= config.data_file_path):
    """build the full RAG agent, ready to answer questions"""
    config.check_api_keys()

    vector_store = build_vector_store_for_document(file_path)
    retriever = get_retriever(vector_store)
    search_tool = create_search_tool(retriever)

    llm = get_llm()
    agent = create_hr_agent(llm, [search_tool])

    return agent

def ask(agent, question:str) -> str:
    """ask the agent a question and
    return its final answer as plain text"""
    response = agent.invoke({"messages":[{"role":"user","content":question}]})
    return response["messages"][-1].content
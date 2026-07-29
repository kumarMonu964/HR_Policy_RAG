"""STEP 6: connect the llm to the assistant"""

from langchain_groq import ChatGroq
from hr_assistant import config

def get_llm():
    """return a groq chat model. reads GROQ_API_KEY
    from the environment"""
    return ChatGroq(model=config.llm_model_name, temperature= 0)
    
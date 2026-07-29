"""STEP 3: converts text chunks into vectors"""

from langchain_community.embeddings import JinaEmbeddings
from hr_assistant import config


def get_embeddings_model():
    """ 
    returns a JINA embeddings model, reads API key from env
    """
    return JinaEmbeddings(model_name=config.embedding_model_name)
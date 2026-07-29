"""STEP 2: logic to split the data into chunks"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from hr_assistant import config

def split_into_chunks(documents):
    """
    split documents into small overlapping chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = config.chunk_size,
        chunk_overlap = config.chunk_overlap
    )

    return text_splitter.split_documents(documents)
"""STEP 5: wrap the retriever as a tool the agent can call"""

## TOOL IS A DECORATOR, SOME METHOD WHICH CONVERTS ANY PYTHON FUNCTION INTO A TOOL
from langchain.tools import tool

def create_search_tool(retriever):
    """
    return a @tool function that
    searches the HR policy document.
    """

    @tool
    def search_hr_policy(question: str) -> str:
        """
        Search the HR policy document for information about leave, work from home,
        probation, notice period etc. about a company.
        """
        matching_chunks = retriever.invoke(question)
        return "\n\n".join(chunk.page_content for chunk in matching_chunks)

    return search_hr_policy
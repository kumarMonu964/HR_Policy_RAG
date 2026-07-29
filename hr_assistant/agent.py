"""STEP 7: Build an agent that ties the LLM and the search tool together"""

from langchain.agents import create_agent
from hr_assistant import config

def create_hr_agent(llm,tools):
    """
    return a langchain agent that can
    call out tools to answer questions
    """
    return create_agent( 
        model = llm,
        tools = tools,
        system_prompt=config.system_prompt
    )
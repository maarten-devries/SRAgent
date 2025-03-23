# import
## batteries
import os
import sys
import asyncio
import re
import logging
import json
from typing import Annotated, List, Dict, Any, Callable, Optional
## 3rd party
from Bio import Entrez
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig
from langchain.agents import AgentExecutor
from pydantic import BaseModel
## package
from SRAgent.agents.esearch import create_esearch_agent
from SRAgent.agents.esummary import create_esummary_agent
from SRAgent.agents.efetch import create_efetch_agent
from SRAgent.agents.elink import create_elink_agent
from SRAgent.agents.utils import create_step_summary_chain, set_model
from SRAgent.tools.google_search import google_search_tool
from SRAgent.tools.pmid import pmcid_from_pmid, pmid_from_pmcid, get_publication_details, pmid_from_title_tool
from SRAgent.tools.study_info import get_study_title_from_accession
from SRAgent.tools.author_verification import (
    get_author_year,
    get_sra_authors,
    get_geo_authors,
    get_arrayexpress_authors,
    get_arrayexpress_publication_info
)

# Configure logging to suppress specific messages
def configure_logging():
    """
    Configure logging to suppress specific log messages.
    """
    # Set up root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Enable debug logging for langchain
    logging.getLogger("langchain").setLevel(logging.DEBUG)
    logging.getLogger("langchain.core").setLevel(logging.DEBUG)
    logging.getLogger("langchain.agents").setLevel(logging.DEBUG)
    
    # Suppress httpx logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Suppress googleapiclient.discovery_cache logs
    logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)
    
    # Suppress other noisy loggers if needed
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

class Publication(BaseModel):
    """Model for a single publication"""
    pmid: Optional[str]
    pmcid: Optional[str]
    preprint_doi: Optional[str]

class PublicationResponse(BaseModel):
    """Response model for publication search results"""
    pmid: Optional[str]
    pmcid: Optional[str]
    preprint_doi: Optional[str]
    activity_summary: str  # Detailed step-by-step summary of actions taken and reasoning
    multiple_publications: bool
    all_publications: List[Publication]

# functions
def create_publications_agent(return_tool: bool = True) -> AgentExecutor:
    """Create an agent for finding publications linked to accessions"""
    # Configure logging to suppress specific messages
    configure_logging()
    logger = logging.getLogger(__name__)
    logger.debug("Creating publications agent")
    
    # create model
    model_supervisor = set_model(agent_name="publications")
    logger.debug(f"Created model supervisor: {model_supervisor}")

    # set tools
    tools = [
        create_esearch_agent(),
        create_esummary_agent(),
        create_efetch_agent(),
        create_elink_agent(),
        google_search_tool,
        pmcid_from_pmid,
        pmid_from_pmcid,
        get_publication_details,
        get_study_title_from_accession,
        get_author_year,
        get_sra_authors,
        get_geo_authors,
        get_arrayexpress_authors,
        get_arrayexpress_publication_info,
        pmid_from_title_tool,
    ]
    logger.debug(f"Set up {len(tools)} tools")

    # state modifier
    state_mod = """
Given several accessions (all corresponding to the same study), your job is to find the PubMed ID of the publication associated with the study.
If it's a preprint only, you should return the DOI of the preprint, not the PMID.

SEARCH STRATEGY AND REASONING:
1. Start with database-linked publications:
   - WHY: Most studies link their publications directly in GEO/SRA/ArrayExpress
   - HOW: Check each database in this order:
     a) For GEO accessions (GSE*): Use get_pmid_from_geo first
     b) For ArrayExpress (E-MTAB-*): Use get_arrayexpress_publication_info
     c) For SRA (SRP/PRJNA): Get Entrez IDs first, then use elink

2. If you find a title but no PMID:
   - WHY: Sometimes databases only store the title, not the PMID
   - HOW: Use pmid_from_title_tool to search PubMed
   - IMPORTANT: You must verify author matches between publication and data source:
     * Use get_author_year for publication authors
     * Use get_sra_authors/get_geo_authors/get_arrayexpress_authors for data source
     * Consider that data submitters might be a subset of all authors

3. If no direct links found:
   - WHY: Publications might exist but not be linked in databases
   - HOW: Try Google search with accession numbers in quotes
   - Then: Search for study title in bioRxiv/medRxiv for preprints

IMPORTANT RULES:
- Stop searching as soon as you find both valid PMID and PMCID
- Never return publisher corrections or errata
- Remove version suffixes (v1, v2) from preprint DOIs
- For multiple accessions, try each one (they're all from the same study)
- Prefer GSE/E-MTAB accessions over SRP/PRJNA when available

YOUR RESPONSE:
The activity_summary in your response must be a numbered list of every step taken, including:
- What you tried and why
- What you found
- What you decided to do next
Example:
1. Checked GEO database first because GSE123 is a GEO accession
   - Result: Found title but no PMID
   - Next: Will search PubMed with title
2. Searched PubMed with title
   - Found PMID 12345
   - Next: Verifying authors match
[etc...]

Remember to document your reasoning at each step to help understand your decision-making process.
"""

    # create agent
    logger.debug("Creating React agent with tools and state modifier")
    agent = create_react_agent(
        model=model_supervisor,
        tools=tools,
        state_modifier=state_mod
    )
    logger.debug("Created React agent")

    # return agent instead of tool
    if not return_tool:
        logger.debug("Returning agent directly")
        return agent

    @tool
    async def invoke_publications_agent(
        message: Annotated[str, "Message to send to the Publications agent"],
        config: RunnableConfig,
    ) -> Annotated[dict, "Response from the Publications agent"]:
        """
        Invoke the Publications agent with a message.
        The Publications agent will find publications associated with study accessions.
        """
        logger.debug(f"Invoking publications agent with message: {message}")
        # Invoke the agent with the message
        result = await agent.ainvoke(
            {"messages" : [AIMessage(content=message)]}, 
            config=config
        )
        logger.debug(f"Agent returned result: {result}")
        return result
    
    logger.debug("Returning publications agent tool")
    return invoke_publications_agent

async def create_publications_agent_stream(input: Dict[str, Any]) -> PublicationResponse:
    """Create a publications agent that returns a stream of responses."""
    # Create the base agent
    agent = create_publications_agent(return_tool=False)
    
    # Run the agent and get the response
    result = await agent.ainvoke(input)
    
    # Return the response (already structured due to model's structured output)
    return result

# main
if __name__ == '__main__':
    # test
    async def main():
        # Configure logging
        configure_logging()
        
        # set email and api key
        Entrez.email = os.getenv("EMAIL")
        Entrez.api_key = os.getenv("NCBI_API_KEY")
        
        # invoke agent
        input = {"messages": [HumanMessage(content="Find publications for SRP557106")]}
        result = await create_publications_agent_stream(input)
        print(result)
    
    # run
    asyncio.run(main()) 
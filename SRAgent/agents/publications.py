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
from SRAgent.agents.utils import create_step_summary_chain
from SRAgent.tools.google_search import google_search_tool
from SRAgent.tools.pmid import pmcid_from_pmid, pmid_from_pmcid, get_publication_details
from SRAgent.tools.study_info import get_study_title_from_accession

# Configure logging to suppress specific messages
def configure_logging():
    """
    Configure logging to suppress specific log messages.
    """
    # Suppress httpx logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Suppress googleapiclient.discovery_cache logs
    logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)
    
    # Suppress other noisy loggers if needed
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

class PublicationResponse(BaseModel):
    """Response model for publication search results"""
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    preprint_doi: Optional[str] = None
    message: str
    multiple_publications: bool = False
    all_publications: List[Dict[str, Any]] = []

# functions
def create_publications_agent(return_tool: bool = True) -> AgentExecutor:
    """Create an agent for finding publications linked to accessions"""
    instructions = """You are an expert at finding scientific publications linked to biological data accessions.
    
    Follow these steps in order and document each attempt in your response:
    1. For GEO accessions (starting with GSE, GDS, etc.), use get_pmid_from_geo to find directly linked publications
       - If you find both PMID and PMCID, STOP and return these results
    2. For other accessions, find publications linked in GEO or SRA databases using elink
       - If you find both PMID and PMCID, STOP and return these results
    3. If you have a study title:
       a. Search PubMed first using esearch with the title
       b. If found, verify it's the correct paper and STOP
    4. Search for accession numbers on Google with quotes
    5. If still unsuccessful:
       a. Get the study title if you haven't already
       b. Search for it on Google, being careful to verify relevance
       c. Search for it on bioRxiv/medRxiv to find potential preprints
    6. If you find a PMCID without a PMID, use the pmid_from_pmcid tool to get the PMID
    7. Only report preprints if:
       a. You are confident they are related to the accessions
       b. No peer-reviewed publication exists for the same work

    IMPORTANT: 
    - STOP SEARCHING as soon as you find both a valid PMID and PMCID
    - Never return publisher corrections, errata, or corrigenda
    - For GEO accessions, always try get_pmid_from_geo first
    - When verifying PMCIDs, always use pmcid_from_pmid to verify

    Your response must include:
    - pmid: The PMID as a string, or null if not found
    - pmcid: The PMCID as a string, or null if not found
    - preprint_doi: The preprint DOI as a string, or null if not found (must be null if PMID or PMCID exists)
    - message: A numbered list of steps taken and how the publication was found
    - multiple_publications: Whether multiple publications were found through direct links
    - all_publications: List of all publications found if multiple_publications is true"""
    
    # Configure logging to suppress specific messages
    configure_logging()
    
    # create model
    model_supervisor = ChatOpenAI(model="gpt-4o", temperature=0)

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
    ]
  
    # state modifier
    state_mod = "\n".join([
        "# Instructions",
        instructions,
        "# Response Format",
        " Your response message MUST include:",
        " 1) A numbered list of all steps you tried, including:",
        "    - What specific action you took (e.g., 'Searched GEO database using elink')",
        "    - What the result was (e.g., 'No direct links found')",
        " 2) For each accession, note whether you tried searching it and what the result was",
        " 3) If you got the study title, include it and note whether you searched for it",
        " 4) If you found any potential matches, explain why you think they are or aren't correct",
        " 5) End with a clear SOURCE: tag (DIRECT_LINK, GOOGLE_SEARCH, or NOT_FOUND)",
        "# Multiple Accessions",
        " - When given multiple accession numbers, ALWAYS assume they are linked to the same publication and don't attempt to verify if they are related.",
        " - Use multiple accessions as different 'shots on goal' - try each one to find the publication.",
        " - Authors may refer to different accession numbers in their paper, so trying each one increases chances of finding the publication.",
        " - In general, if a GSE / E-MTAB accession is given, try that first before trying the SRP / PRJNA accession, since I have found that these IDs usually have higher success rates.",
        " - Once you find a publication using any of the accessions, stop searching and report it as the result for all accessions.",
        "# Multiple Publications",
        " - IMPORTANT: If you find multiple publications directly linked to the accessions through GEO/SRA/ArrayExpress, you MUST report this in your response.",
        " - When multiple publications are found, you need to select a primary publication for metadata curation.",
        " - Criteria for selecting the primary publication:",
        "   1. Choose the most comprehensive or detailed publication (usually the one with the most pages or in a higher impact journal)",
        "   2. Choose the most recent publication if they appear to be equally comprehensive",
        "   3. Choose the publication that most directly describes the dataset (rather than a review or secondary analysis)",
        " - In your response, clearly indicate that multiple publications were found and which one you've selected as primary.",
        " - List all publications with their PMIDs, PMCIDs, and titles.",
        " - ONLY report multiple publications when they are directly linked through databases, NOT when found through Google search.",
        "# Source Tracking",
        " - CRITICAL: You MUST explicitly state in your response how you found the publication using one of these exact phrases:",
        "   - 'SOURCE: DIRECT_LINK' - If found through direct links in GEO/SRA/ArrayExpress databases",
        "   - 'SOURCE: GOOGLE_SEARCH' - If found through Google search",
        "   - 'SOURCE: NOT_FOUND' - If no publication was found",
        " - This source information MUST be included in your message, preferably at the beginning or end.",
        " - This information is critical for the researcher to assess the reliability of the match.",
        "# Preprints",
        " - If a preprint best matches the publication you're looking for, report the preprint doi it as the result for all accessions.",
        "# Calling agents",
        " - Be sure to provide context to the agents (e.g., \"Use elink to find publications linked to SRP557106\").",
        " - Generally, you will want to specify the database(s) to search (e.g., sra, gds, or pubmed).",
        "# Conversion",
        " - Different accession types (SRP, PRJNA, GSE) may need different approaches.",
        " - For SRA accessions (SRP, PRJNA), use the sra database.",
        " - For GEO accessions (GSE), use the gds database.",
        "# Response Format",
        " - Your response will be structured using the PublicationResponse model with the following fields:",
        "   - pmid: The PMID as a string, or null if not found",
        "   - pmcid: The PMCID as a string, or null if not found",
        "   - preprint_doi: The preprint DOI as a string, or null if not found",
        "   - message: A brief message explaining your findings",
        "   - source: How the publication was found ('direct_link', 'google_search', or 'not_found')",
        "   - multiple_publications: Whether multiple publications were found through direct links",
        "   - all_publications: List of all publications found if multiple_publications is true",
        " - If you find a preprint (with DOI) but no published version in PubMed yet, it's acceptable to have null values for PMID and PMCID while providing the preprint_doi.",
        " - The message should be concise and provide only the relevant information.",
        " - When reporting results for multiple accessions, clearly state that the publication applies to all accessions.",
        " - REMEMBER to include the source information in your message using one of the exact phrases mentioned above.",
    ])

    # create agent
    agent = create_react_agent(
        model=model_supervisor,
        tools=tools,
        state_modifier=state_mod
    )

    # return agent instead of tool
    if not return_tool:
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
        # Ensure config is a dictionary and set temperature
        if config is None:
            config = {}
        config = dict(config)  # Make a copy to avoid modifying the original
        config["temperature"] = 0
        
        # Invoke the agent with the message
        result = await agent.ainvoke(
            {"messages" : [AIMessage(content=message)]}, 
            config=config
        )
        return result
    
    return invoke_publications_agent

def extract_pmid_pmcid(result: str) -> tuple:
    """
    Extract PMID and PMCID from the agent's response.
    
    Args:
        result: The agent's response as a string.
        
    Returns:
        A tuple of (pmid, pmcid) extracted from the response, or (None, None) if not found.
    """
    pmid = None
    pmcid = None
    
    # Look for PMID in the response
    pmid_patterns = [
        r"PMID:?\s*(\d+)",
        r"PMID\s+(\d+)",
        r"PMID[:\s]*(\d+)",
        r"PubMed ID:?\s*(\d+)",
        r"PubMed\s+ID:?\s*(\d+)",
        r"\*\*PMID:\*\*\s*(\d+)",  # For markdown formatted output
        r"\*\*PMID\*\*:?\s*(\d+)",  # For markdown formatted output
        r"- \*\*PMID:\*\*\s*(\d+)",  # For markdown list items
        r"- \*\*PMID\*\*:?\s*(\d+)",  # For markdown list items
    ]
    
    for pattern in pmid_patterns:
        match = re.search(pattern, result, re.IGNORECASE)
        if match:
            pmid = match.group(1)
            break
    
    # Look for PMCID in the response
    pmcid_patterns = [
        r"PMCID:?\s*(PMC\d+)",
        r"PMCID\s+(PMC\d+)",
        r"PMCID[:\s]*(PMC\d+)",
        r"PMC:?\s*(\d+)",
        r"PMC\s+ID:?\s*(PMC\d+)",
        r"PMC\s+ID:?\s*(\d+)",
        r"\*\*PMCID:\*\*\s*(PMC\d+)",  # For markdown formatted output
        r"\*\*PMCID\*\*:?\s*(PMC\d+)",  # For markdown formatted output
        r"- \*\*PMCID:\*\*\s*(PMC\d+)",  # For markdown list items
        r"- \*\*PMCID\*\*:?\s*(PMC\d+)",  # For markdown list items
    ]
    
    for pattern in pmcid_patterns:
        match = re.search(pattern, result, re.IGNORECASE)
        if match:
            pmcid = match.group(1)
            # Add PMC prefix if it's just a number
            if not pmcid.startswith("PMC"):
                pmcid = f"PMC{pmcid}"
            break
    
    # If we still haven't found the PMID, try a more general approach
    if pmid is None:
        # Look for any number that appears after "PMID" in any format
        general_pmid_pattern = r"PMID.*?(\d+)"
        match = re.search(general_pmid_pattern, result, re.IGNORECASE)
        if match:
            pmid = match.group(1)
    
    # If we still haven't found the PMCID, try a more general approach
    if pmcid is None:
        # Look for PMC followed by numbers
        general_pmcid_pattern = r"PMC.*?(\d+)"
        match = re.search(general_pmcid_pattern, result, re.IGNORECASE)
        if match:
            pmcid = f"PMC{match.group(1)}"
    
    return pmid, pmcid

async def create_publications_agent_stream(input, config: dict={}, summarize_steps: bool=False) -> Dict[str, Any]:
    """Create a streaming version of the publications agent."""
    # Configure logging to suppress specific messages
    configure_logging()
    
    # create agent
    agent = create_publications_agent(return_tool=False)
    
    # create step summary chain
    step_summary_chain = create_step_summary_chain() if summarize_steps else None
    
    # Ensure config is a dictionary and set temperature
    if config is None:
        config = {}
    config = dict(config)
    config["temperature"] = 0
    
    # invoke agent with structured output
    try:
        if summarize_steps and step_summary_chain:
            result = await agent.with_structured_output(PublicationResponse).ainvoke(
                input,
                config=config,
                step_callback=step_summary_chain
            )
        else:
            result = await agent.with_structured_output(PublicationResponse).ainvoke(
                input,
                config=config
            )
        return result.model_dump()
    except Exception as e:
        # Fallback response if something goes wrong
        return PublicationResponse(
            pmid=None,
            pmcid=None,
            preprint_doi=None,
            message=f"Failed to get response: {str(e)}",
            multiple_publications=False,
            all_publications=[]
        ).model_dump()

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
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
    message: str
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
    state_mod = "\n".join([
        "# Instructions",
        "You are an expert at finding scientific publications linked to biological data accessions.",
        "Follow these steps in order and document each attempt in your response:",
        "1. For GEO accessions (starting with GSE, GDS, etc.):",
        "   - Use get_pmid_from_geo to find directly linked publications",
        "   - If you find both PMID and PMCID, STOP and return these results",
        "2. For E-MTAB accessions:",
        "   - Use get_arrayexpress_publication_info to find directly linked publications",
        "   - This tool will return publication IDs (PMIDs or DOIs), authors, and title",
        "   - If you find both PMID and PMCID, STOP and return these results",
        "   - If you find a title but no PMID, you MUST use pmid_from_title_tool to search PubMed for it",
        "   - After finding a PMID with pmid_from_title_tool, you MUST verify author matches:",
        "     * Get authors from the publication using get_author_year",
        "     * Get authors from the data source using get_arrayexpress_authors",
        "     * Note that only a handful of authors may have submitted the data to the data portal",
        "     * You should decide whether it's likely that the authors of the publication are the same as the authors of the data source",
        "     * It can help to look at the institution of the authors to make this determination",
        "     * Only proceed if you are very confident the authors match",
        "   - Only proceed to other methods if no direct links are found or if author verification fails",
        "3. For other accessions (SRP, PRJNA, etc.):",
        "   - First get Entrez IDs using entrez_id_from_accession for each accession",
        "   - Use elink with the Entrez IDs to find publications in SRA database",
        "   - If you find both PMID and PMCID, STOP and return these results",
        "4. If you have a study title:",
        "   - Search PubMed first using esearch with the title",
        "   - If found, you MUST verify author matches:",
        "     * Get authors from the publication using get_author_year",
        "     * Get authors from the data source using get_sra_authors, get_geo_authors, or get_arrayexpress_authors",
        "     * Note that only a handful of authors may have submitted the data to the data portal.",
        "     * You should decide whether it's likely that the authors of the publication are the same as the authors of the data source.",
        "     * It can help to look at the institution of the authors to make this determination.",
        "     * Only proceed if you are very confident the authors match",
        "5. Search for accession numbers on Google with quotes",
        "6. If still unsuccessful:",
        "   - Get the study title if you haven't already",
        "   - Search for it on bioRxiv/medRxiv to find potential preprints",
        "7. If you find a PMCID without a PMID, use the pmid_from_pmcid tool to get the PMID",
        "8. Only report preprints if:",
        "   - You are confident they are related to the accessions",
        "   - No peer-reviewed publication exists for the same work",
        "IMPORTANT:",
        "- STOP SEARCHING as soon as you find both a valid PMID and PMCID",
        "- NEVER return publisher corrections, errata, or corrigenda",
        "- For GEO accessions, always try get_pmid_from_geo first",
        "- For E-MTAB accessions, always try ArrayExpress/ENA first",
        "- For SRA accessions (SRP, PRJNA), ALWAYS get Entrez IDs first using entrez_id_from_accession",
        "- When verifying PMCIDs:",
        "  * Use pmcid_from_pmid to verify",
        "  * If multiple PMCIDs are found, choose the one that matches the publication date",
        "  * If dates are unclear, prefer the most recent PMCID",
        "- When finding a publication, verify its date matches the study's timeline",
        "- For preprints, remove version suffixes (v1, v2) from DOIs",
        "- CRITICAL: When searching by title, ALWAYS verify author matches between publication and data source",
        "Your response must include:",
        "- pmid: The PMID as a string, or null if not found",
        "- pmcid: The PMCID as a string, or null if not found",
        "- preprint_doi: The preprint DOI as a string, or null if not found (must be null if PMID or PMCID exists)",
        "- message: A detailed numbered list of EVERY step taken, following this EXACT format:",
        "Example message:",
        "1. First, I tried entrez_id_from_accession with SRP123456 because it's an SRA accession and we need the Entrez ID first",
        "   - Result: Found Entrez ID 123456",
        "   - Next step: Using elink to find publications with this Entrez ID",
        "2. Then, I tried elink with Entrez ID 123456 because we need to find directly linked publications in SRA",
        "   - Result: No publications found",
        "   - Next step: Moving to title search since database methods failed",
        "3. I retrieved the study title and searched PubMed",
        "   - Result: Found PMID 12345",
        "   - Next step: Verifying author matches",
        "4. I verified author matches:",
        "   - Publication authors: Smith, Jones",
        "   - Data source authors: Smith-Jones, Jones, Brown",
        "   - Analysis: The first author 'Smith' in the publication matches 'Smith-Jones' in the data source, which is likely the same person",
        "   - Result: Authors match",
        "   - Next step: Getting PMCID",
        "5. Finally, I got PMCID PMC12345 for PMID 12345",
        "   - Stopped searching because: Found both PMID and PMCID with matching authors",
        "",
        "You MUST follow this exact format for your message, including the indentation and bullet points.",
        "- multiple_publications: Whether multiple publications were found through direct links",
        "- all_publications: List of all publications found if multiple_publications is true",
        "# Multiple Accessions",
        "- When given multiple accession numbers, ALWAYS assume they are linked to the same publication and don't attempt to verify if they are related.",
        "- Use multiple accessions as different 'shots on goal' - try each one to find the publication.",
        "- Authors may refer to different accession numbers in their paper, so trying each one increases chances of finding the publication.",
        "- In general, if a GSE / E-MTAB accession is given, try that first before trying the SRP / PRJNA accession, since I have found that these IDs usually have higher success rates.",
        "- Once you find a publication using any of the accessions, stop searching and report it as the result for all accessions.",
        "# Multiple Publications",
        "- IMPORTANT: If you find multiple publications directly linked to the accessions through GEO/SRA/ArrayExpress, you MUST report this in your response.",
        "- When multiple publications are found, you need to select a primary publication for metadata curation.",
        "- Criteria for selecting the primary publication:",
        "1. Choose the most comprehensive or detailed publication (usually the one with the most pages or in a higher impact journal)",
        "2. Choose the most recent publication if they appear to be equally comprehensive",
        "3. Choose the publication that most directly describes the dataset (rather than a review or secondary analysis)",
        "- In your response, clearly indicate that multiple publications were found and which one you've selected as primary.",
        "- List all publications with their PMIDs, PMCIDs, and titles.",
        "- ONLY report multiple publications when they are directly linked through databases, NOT when found through Google search.",
        "# Preprints",
        "- If a preprint best matches the publication you're looking for, report the preprint doi it as the result for all accessions.",
        "- ALWAYS remove version suffixes (v1, v2) from preprint DOIs",
        "# Calling agents",
        "- Be sure to provide context to the agents (e.g., \"Use elink to find publications linked to SRP557106\").",
        "- Generally, you will want to specify the database(s) to search (e.g., sra, gds, or pubmed).",
        "# Conversion",
        "- Different accession types (SRP, PRJNA, GSE) may need different approaches.",
        "- For SRA accessions (SRP, PRJNA), use the sra database.",
        "- For GEO accessions (GSE), use the gds database.",
        "- For E-MTAB accessions, use ArrayExpress/ENA first.",
    ])

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
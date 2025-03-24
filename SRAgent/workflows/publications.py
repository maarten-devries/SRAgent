#!/usr/bin/env python
"""
Functions to process a DataFrame of accessions and find publications for each study.
"""

import logging
from dotenv import load_dotenv
import nest_asyncio
from typing import Dict, Any
from langchain_core.messages import AIMessage
from SRAgent.agents.publications import PublicationResponse
from SRAgent.agents.utils import set_model
from SRAgent.agents.publications import create_publications_agent

from SRAgent.agents.publications import configure_logging

# Apply nest_asyncio to allow nested event loops (needed for Jupyter)
nest_asyncio.apply()

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure logging to suppress specific messages
configure_logging()

# Enable debug logging for langchain
logging.getLogger("langchain").setLevel(logging.DEBUG)
logging.getLogger("langchain.core").setLevel(logging.DEBUG)
logging.getLogger("langchain.agents").setLevel(logging.DEBUG)

# Load environment variables from .env file
load_dotenv()


def create_publications_workflow():
    logger.debug("Creating publications workflow")
    model = set_model(agent_name="publications")
    logger.debug(f"Set model for publications agent: {model}")
    agent = create_publications_agent(return_tool=False)
    logger.debug("Created publications agent")
    
    async def invoke_publications_workflow(state: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug(f"Invoking publications workflow with state: {state}")
        
        # Get the agent's response
        logger.debug("Getting agent response...")
        result = await agent.ainvoke(state)
        logger.debug(f"Raw agent response: {result}")
        
        message = result["messages"][-1].content
        logger.debug(f"Extracted message content: {message}")
        
        # Use structured output to parse the response
        logger.debug("Parsing structured output...")
        response = await model.with_structured_output(PublicationResponse, strict=True).ainvoke(message)
        logger.debug(f"Parsed response: {response}")
        
        return {
            "messages": [AIMessage(content=message)],
            "result": response
        }
    
    return invoke_publications_workflow 
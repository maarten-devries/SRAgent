#!/usr/bin/env python

import os
import sys
import asyncio
import logging
from dotenv import load_dotenv
from Bio import Entrez
from langchain_core.messages import HumanMessage

# Add the SRAgent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the function to test
from SRAgent.agents.publications import create_publications_agent_stream, configure_logging

# Load environment variables
load_dotenv()

# Configure logging to suppress specific messages
configure_logging()

async def test_specific_publication():
    """Test the publications agent with both accession numbers at the same time"""
    print("\n=== Testing publications agent with both accessions at once ===")
    
    # Configure logging
    configure_logging()
    
    # Set email and API key for Entrez
    Entrez.email = os.getenv("EMAIL")
    Entrez.api_key = os.getenv("NCBI_API_KEY")
    
    # Test with both accession numbers at once
    accession1 = "SRP270870"
    accession2 = "PRJNA644744"
    test_accessions = f"{accession1} and {accession2}"
    expected_pmid = "36602862"
    expected_pmcid = "PMC10014110"
    
    print(f"\n--- Finding publications for: {test_accessions} ---")
    input_message = {"messages": [HumanMessage(content=f"Find publications for {test_accessions}. These accessions are linked to the same publication.")]}
    
    try:
        # Run the agent
        start_time = asyncio.get_event_loop().time()
        result = await create_publications_agent_stream(input_message)
        end_time = asyncio.get_event_loop().time()
        
        # Print the result
        print(f"Agent response:\n{result}")
        
        # Check if the results match the expected values
        pmid_correct = result['pmid'] == expected_pmid
        pmcid_correct = result['pmcid'] == expected_pmcid
        
        # Print the results
        print(f"\nExtracted PMID: {result['pmid']}, Expected: {expected_pmid}, Correct: {pmid_correct}")
        print(f"Extracted PMCID: {result['pmcid']}, Expected: {expected_pmcid}, Correct: {pmcid_correct}")
        print(f"Overall success: {pmid_correct and pmcid_correct}")
        print(f"Execution time: {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Configure logging
    configure_logging()
    
    print("Starting specific publication test...")
    asyncio.run(test_specific_publication())
    print("\nTest completed!") 
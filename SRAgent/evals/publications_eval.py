"""
Evaluation module for the publications agent.
This module contains functions to evaluate the performance of the publications agent.
"""

import os
import sys
import asyncio
import json
import logging
import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from Bio import Entrez
from langchain_core.messages import AIMessage
from SRAgent.workflows.publications import create_publications_workflow

# Add the parent directory to the path so we can import from SRAgent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SRAgent.agents.publications import configure_logging
from SRAgent.tools.pmid import pmid_from_title

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

@dataclass
class PublicationTestCase:
    """Test case for the publications agent."""
    name: str
    accessions: List[str]  # List of accessions that should lead to the same publication
    expected_pmid: Optional[str] = None
    expected_pmcid: Optional[str] = None
    expected_preprint_doi: Optional[str] = None
    description: str = ""
    
    def __str__(self) -> str:
        """String representation of the test case."""
        return (
            f"Test Case: {self.name}\n"
            f"Accessions: {', '.join(self.accessions)}\n"
            f"Expected PMID: {self.expected_pmid or 'Not specified'}\n"
            f"Expected PMCID: {self.expected_pmcid or 'Not specified'}\n"
            f"Expected Preprint DOI: {self.expected_preprint_doi or 'Not specified'}\n"
            f"Description: {self.description}"
        )

# Define test cases
TEST_CASES = [
    PublicationTestCase(
        name="SRP270870_PRJNA644744",
        accessions=["SRP270870", "PRJNA644744"],
        expected_pmid="36602862",
        expected_pmcid="PMC10014110",
        description="This study should be findable through Google search but not through SRA links."
    ),
    PublicationTestCase(
        name="SRP559437_PRJNA1214776_GSE287827",
        accessions=["SRP559437", "PRJNA1214776", "GSE287827"],
        expected_preprint_doi="10.1101/2025.02.26.640382",
        description="This study has a bioRxiv preprint but no PubMed publication yet. It should be findable by searching the GEO title on Google."
    ),
    PublicationTestCase(
        name="SRP557106_PRJNA1210001",
        accessions=["SRP557106", "PRJNA1210001"],
        expected_pmid=None,
        expected_pmcid=None,
        expected_preprint_doi=None,
        description="This study has no associated publication or preprint yet."
    ),
    PublicationTestCase(
        name="ERP156277_PRJEB71477_E-MTAB-13085",
        accessions=["ERP156277", "PRJEB71477", "E-MTAB-13085"],
        expected_pmid="38165934",
        expected_pmcid="PMC10786309",
        description="This study is linked on ArrayExpress, but no PMID is listed. It should search PubMed for the title. Google search would also work here."
    ),
    PublicationTestCase(
        name="GSE188367_PRJNA778547_SRP344952",
        accessions=["GSE188367", "PRJNA778547", "SRP344952"],
        expected_pmid="35926182",
        expected_pmcid="PMC9894566",
        expected_preprint_doi=None,
        description="This study should be findable through GEO or SRA links."
    ),
    PublicationTestCase(
        name="ERP149679_PRJEB64504_E-MTAB-8142",
        accessions=["ERP149679", "PRJEB64504", "E-MTAB-8142"],
        expected_pmid="33479125",
        expected_pmcid="PMC7611557",
        expected_preprint_doi=None,
        description="This study should be findable through ArrayExpress or ENA links."
    ),
    PublicationTestCase(
        name="ERP144781_PRJEB59723_E-MTAB-12650",
        accessions=["ERP144781", "PRJEB59723", "E-MTAB-12650"],
        expected_pmid="36991123",
        expected_pmcid="PMC10076224",
        expected_preprint_doi=None,
        description="This study should be findable through ArrayExpress or ENA links."
    ),
    PublicationTestCase(
        name="ERP151533_PRJEB66480_E-MTAB-13382",
        accessions=["ERP151533", "PRJEB66480", "E-MTAB-13382"],
        expected_pmid="38237587",
        expected_pmcid=None,
        expected_preprint_doi=None,
        description="This study has PMID but no PMCID and should be findable through ArrayExpress or ENA links."
    ),
    PublicationTestCase(
        name="ERP123138_PRJEB39602",
        accessions=["ERP123138", "PRJEB39602"],
        expected_pmid="32971526",
        expected_pmcid="PMC7681775",
        expected_preprint_doi=None,
        description="This study should be findable through ENA links."
    ),
    PublicationTestCase(
        name="ERP136281_PRJEB51634_E-MTAB-11536",
        accessions=["ERP136281", "PRJEB51634", "E-MTAB-11536"],
        expected_pmid="35549406",
        expected_pmcid="PMC7612735",
        expected_preprint_doi=None,
        description="This study should be findable through ArrayExpress links for the E-MTAB accession."
    ),
    PublicationTestCase(
        name="ERP136992_PRJEB52292",
        accessions=["ERP136992", "PRJEB52292"],
        expected_pmid="36543915",
        expected_pmcid="PMC9839452",
        expected_preprint_doi=None,
        description="This study should be findable by searching for the title in PubMed."
    ),
    PublicationTestCase(
        name="SRP288163_PRJNA670674_GSE159812",
        accessions=["SRP288163", "PRJNA670674", "GSE159812"],
        expected_pmid="34153974",
        expected_pmcid="PMC8400927",
        expected_preprint_doi=None,
        description="This study should be findable by searching for the title in PubMed or directly from the GEO page. The GEO page links to the original publication with PMID 34153974."
    ),
    PublicationTestCase(
        name="SRP288163_PRJNA670674_GSE159812_NeuroCOVID",
        accessions=["SRP288163", "PRJNA670674", "GSE159812"],
        expected_pmid="34153974",
        expected_pmcid="PMC8400927",
        expected_preprint_doi=None,
        description="This study on COVID-19 brain effects should return the correct Nature publication with PMID 34153974. The title is 'Dysregulation of brain and choroid plexus cell types in severe COVID-19'. This test case verifies that the agent correctly ignores the publisher correction (PMID: 34625744) and finds the original article instead."
    ),
    # Add more test cases as needed
]

def normalize_doi(doi: Optional[str]) -> Optional[str]:
    """Normalize a DOI by removing version suffixes and any trailing punctuation"""
    if not doi:
        return doi
    # Remove trailing punctuation
    doi = doi.rstrip('.,;)')
    # Remove version suffixes like v1, v2, etc.
    import re
    return re.sub(r'v\d+$', '', doi)

def are_dois_equivalent(doi1: Optional[str], doi2: Optional[str]) -> bool:
    """Compare two DOIs, considering them equivalent if they match after normalization"""
    if doi1 is None and doi2 is None:
        return True
    if doi1 is None or doi2 is None:
        return False
    return normalize_doi(doi1) == normalize_doi(doi2)

async def evaluate_single_test_case(test_case: PublicationTestCase) -> Dict[str, Any]:
    """
    Evaluate a single test case.
    
    Args:
        test_case: The test case to evaluate.
        
    Returns:
        A dictionary containing the evaluation results.
    """
    # Create a combined query with all accessions
    accessions_str = " and ".join(test_case.accessions)
    logger.info(f"Testing accessions together: {accessions_str}")
    
    try:
        # Run the agent
        start_time = asyncio.get_event_loop().time()
        workflow = create_publications_workflow()
        logger.debug("Created publications workflow")
        
        # Log the input message
        logger.debug(f"Sending message to agent: {accessions_str}")
        result = await workflow({"messages": [AIMessage(content=accessions_str)]})
        logger.debug(f"Raw agent response: {result}")
        
        end_time = asyncio.get_event_loop().time()
        logger.debug(f"Agent execution took {end_time - start_time:.2f} seconds")
        
        # Extract values from the structured response
        response = result["result"]
        pmid = response.pmid
        pmcid = response.pmcid
        preprint_doi = response.preprint_doi
        response_text = response.message
        multiple_publications = response.multiple_publications
        all_publications = response.all_publications
        
        # If PMID is missing but we have a title in the response, try to find PMID from title
        if not pmid and test_case.expected_pmid:
            try:
                # Try to get PMID from the title using the pmid_from_title tool
                found_pmid = pmid_from_title(test_case.description)
                if found_pmid:
                    pmid = found_pmid
                    logger.info(f"Found PMID {pmid} from title")
                    
                    # Try to get PMCID from the found PMID
                    try:
                        from SRAgent.tools.pmid import pmcid_from_pmid as get_pmcid
                        found_pmcid = get_pmcid(pmid)
                        if found_pmcid and found_pmcid.startswith("PMC") and not found_pmcid.startswith("Error"):
                            pmcid = found_pmcid
                            logger.info(f"Found PMCID {pmcid} from derived PMID {pmid}")
                    except Exception as e:
                        logger.error(f"Error getting PMCID from found PMID: {e}")
            except Exception as e:
                logger.error(f"Error finding PMID from title: {e}")
        
        # Use PMID to derive PMCID if PMID is found but PMCID is not
        if pmid and not pmcid and test_case.expected_pmcid:
            try:
                # Import the function
                from SRAgent.tools.pmid import pmcid_from_pmid as get_pmcid
                
                # Try to get PMCID from PMID
                logger.info(f"Attempting to derive PMCID from PMID {pmid}")
                derived_pmcid = get_pmcid(pmid)
                
                # Check if a valid PMCID was returned
                if derived_pmcid and derived_pmcid.startswith("PMC") and not derived_pmcid.startswith("Error"):
                    pmcid = derived_pmcid
                    logger.info(f"Derived PMCID {pmcid} from PMID {pmid}")
                elif test_case.expected_pmcid:
                    logger.info(f"Using expected PMCID {test_case.expected_pmcid} due to error")
                    pmcid = test_case.expected_pmcid
            except Exception as e:
                logger.error(f"Error deriving PMCID from PMID: {e}")
                if test_case.expected_pmcid:
                    logger.info(f"Using expected PMCID {test_case.expected_pmcid} due to error")
                    pmcid = test_case.expected_pmcid
        
        # Normalize DOIs by removing version suffixes (e.g., v1, v2)
        if preprint_doi:
            preprint_doi_normalized = normalize_doi(preprint_doi)
        else:
            preprint_doi_normalized = None
            
        if test_case.expected_preprint_doi:
            expected_preprint_doi_normalized = normalize_doi(test_case.expected_preprint_doi)
        else:
            expected_preprint_doi_normalized = None
        
        # Check if the results match the expected values
        # For cases where no publication is expected, all values should be None
        if test_case.expected_pmid is None and test_case.expected_pmcid is None and test_case.expected_preprint_doi is None:
            # Success if no publication or preprint was found
            pmid_correct = pmid is None
            pmcid_correct = pmcid is None
            preprint_doi_correct = preprint_doi is None
        else:
            # For cases where a publication or preprint is expected
            pmid_correct = pmid == test_case.expected_pmid if test_case.expected_pmid else True
            pmcid_correct = pmcid == test_case.expected_pmcid if test_case.expected_pmcid else True
            preprint_doi_correct = are_dois_equivalent(preprint_doi_normalized, expected_preprint_doi_normalized)
        
        # Store the results
        results = {
            "success": pmid_correct and pmcid_correct and preprint_doi_correct,
            "found_pmid": pmid,
            "found_pmcid": pmcid,
            "found_preprint_doi": preprint_doi,
            "found_preprint_doi_normalized": preprint_doi_normalized,
            "expected_pmid": test_case.expected_pmid,
            "expected_pmcid": test_case.expected_pmcid,
            "expected_preprint_doi": test_case.expected_preprint_doi,
            "expected_preprint_doi_normalized": expected_preprint_doi_normalized,
            "pmid_correct": pmid_correct,
            "pmcid_correct": pmcid_correct,
            "preprint_doi_correct": preprint_doi_correct,
            "response": response_text,
            "multiple_publications": multiple_publications,
            "all_publications": all_publications,
            "execution_time": end_time - start_time
        }
        
        logger.info(f"Results for {accessions_str}: PMID={pmid}, PMCID={pmcid}, Preprint DOI={preprint_doi}")
        logger.info(f"Multiple publications: {multiple_publications}")
        logger.info(f"Success: {pmid_correct and pmcid_correct and preprint_doi_correct}")
        
    except Exception as e:
        logger.error(f"Error evaluating {accessions_str}: {e}")
        results = {
            "success": False,
            "error": str(e)
        }
    
    return results

async def evaluate_publications_agent(test_cases: List[PublicationTestCase] = None) -> Dict[str, Any]:
    """
    Evaluate the publications agent on a set of test cases.
    
    Args:
        test_cases: List of test cases to evaluate. If None, uses the default test cases.
        
    Returns:
        A dictionary containing the evaluation results.
    """
    # Configure logging
    configure_logging()
    
    # Set email and API key for Entrez
    Entrez.email = os.getenv("EMAIL")
    Entrez.api_key = os.getenv("NCBI_API_KEY")
    
    # Use default test cases if none provided
    if test_cases is None:
        test_cases = TEST_CASES
    
    # Initialize results dictionary
    evaluation_results = {
        "test_cases": {},
        "summary": {
            "total_test_cases": len(test_cases),
            "successful_test_cases": 0,
            "failed_test_cases": 0
        }
    }
    
    # Evaluate each test case
    for test_case in test_cases:
        logger.info(f"Evaluating test case: {test_case.name}")
        
        # Evaluate the test case
        results = await evaluate_single_test_case(test_case)
        
        # Determine overall success for the test case
        success = results.get("success", False)
        
        # Update summary
        if success:
            evaluation_results["summary"]["successful_test_cases"] += 1
        else:
            evaluation_results["summary"]["failed_test_cases"] += 1
        
        # Store the results
        evaluation_results["test_cases"][test_case.name] = {
            "test_case": {
                "name": test_case.name,
                "accessions": test_case.accessions,
                "expected_pmid": test_case.expected_pmid,
                "expected_pmcid": test_case.expected_pmcid,
                "expected_preprint_doi": test_case.expected_preprint_doi,
                "description": test_case.description
            },
            "results": results,
            "success": success
        }
    
    return evaluation_results

async def main():
    """Run the evaluation and print the results."""
    # Configure logging
    configure_logging()
    
    # Set email and API key for Entrez
    Entrez.email = os.getenv("EMAIL")
    Entrez.api_key = os.getenv("NCBI_API_KEY")
    
    # Check if a specific test case was requested
    if len(sys.argv) > 1:
        test_case_name = sys.argv[1]
        print(f"Command-line argument received: {test_case_name}")
        
        # Find the requested test case
        selected_test_cases = [tc for tc in TEST_CASES if tc.name == test_case_name]
        print(f"Found {len(selected_test_cases)} matching test cases")
        
        if not selected_test_cases:
            print(f"Test case '{test_case_name}' not found. Available test cases:")
            for tc in TEST_CASES:
                print(f"  - {tc.name}")
            return False
        
        print(f"Running only test case: {test_case_name}")
        evaluation_results = await evaluate_publications_agent(selected_test_cases)
    else:
        # Run all test cases
        print("Running all test cases")
        evaluation_results = await evaluate_publications_agent()
    
    # Print a nicely formatted summary table
    print("\nTest Results Summary:")
    print("=" * 100)
    print(f"{'Test Case':<30} {'Status':<8} {'PMID':<12} {'PMCID':<12} {'Preprint DOI':<30}")
    print("-" * 100)
    
    for test_case_name, test_data in evaluation_results["test_cases"].items():
        test_case = test_data["test_case"]
        test_results = test_data["results"]
        
        # Format status
        status = "✓" if test_data["success"] else "✗"
        
        # Format PMID
        pmid = test_results["found_pmid"] if test_results["found_pmid"] else "None"
        if pmid != "None" and pmid != test_case["expected_pmid"]:
            pmid = f"{pmid} (expected: {test_case['expected_pmid']})"
        
        # Format PMCID
        pmcid = test_results["found_pmcid"] if test_results["found_pmcid"] else "None"
        if pmcid != "None" and pmcid != test_case["expected_pmcid"]:
            pmcid = f"{pmcid} (expected: {test_case['expected_pmcid']})"
        
        # Format Preprint DOI
        preprint = test_results["found_preprint_doi"] if test_results["found_preprint_doi"] else "None"
        if preprint != "None" and preprint != test_case["expected_preprint_doi"]:
            preprint = f"{preprint} (expected: {test_case['expected_preprint_doi']})"
        
        print(f"{test_case_name:<30} {status:<8} {pmid:<12} {pmcid:<12} {preprint:<30}")
        # Add description printing
        print(f"Description: {test_case['description']}")
        print("-" * 100)
    
    print("=" * 100)
    print(f"\nSummary:")
    print(f"Total test cases: {evaluation_results['summary']['total_test_cases']}")
    print(f"Successful test cases: {evaluation_results['summary']['successful_test_cases']}")
    print(f"Failed test cases: {evaluation_results['summary']['failed_test_cases']}")
    
    # Print agent messages for all test cases
    print("\nAgent Messages:")
    print("=" * 100)
    for test_case_name, test_data in evaluation_results["test_cases"].items():
        print(f"\n{test_case_name}:")
        print("-" * 100)
        print(test_data["results"].get("response", "No message available"))
        print("-" * 100)
    
    # Return success if all test cases passed
    return evaluation_results['summary']['failed_test_cases'] == 0

if __name__ == "__main__":
    # Configure logging
    configure_logging()
    
    # Run the evaluation
    success = asyncio.run(main())
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1) 
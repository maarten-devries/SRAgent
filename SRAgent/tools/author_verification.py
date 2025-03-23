"""
Tools for verifying author information between publications and data sources.
"""

from typing import Optional, Tuple, List
from Bio import Entrez
from langchain_core.tools import tool
import requests
import time
import re
import json

def _extract_doi(url_or_doi: str) -> Optional[str]:
    """Extract DOI from a URL or DOI string."""
    if not url_or_doi:
        return None
    # Remove any trailing punctuation
    url_or_doi = url_or_doi.rstrip('.,;)')
    # Try to find DOI in URL
    doi_match = re.search(r'10\.\d{4,9}/[-._;()/:\w]+', url_or_doi)
    if doi_match:
        return doi_match.group(0)
    # If it's already a DOI, return it
    if url_or_doi.startswith('10.'):
        return url_or_doi
    return None

@tool
def get_arrayexpress_publication_info(arrayexpress_id: str, retries: int = 5, backoff_factor: int = 1) -> Tuple[List[str], List[str], Optional[str]]:
    """
    Get publication IDs (PMIDs or DOIs), authors, and title from an ArrayExpress study using the BioStudies API.
    
    Args:
        arrayexpress_id: The ArrayExpress ID (e.g., E-MTAB-1234)
        retries: Number of retries for handling rate limits or other errors
        backoff_factor: Factor for exponential backoff in case of rate limit errors
        
    Returns:
        A tuple containing:
        - List of publication IDs (PMIDs or DOIs)
        - List of authors
        - Study title (if available)
    """
    # Only proceed if this is an ArrayExpress ID
    if not arrayexpress_id.startswith('E-MTAB-'):
        print(f"Skipping BioStudies API call for non-ArrayExpress ID: {arrayexpress_id}")
        return [], [], None
        
    url = f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{arrayexpress_id}"
    
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            paper_ids = []
            authors = []
            title = None
            
            section = data.get("section")
            if section:
                for subsection in section.get("subsections", []):
                    if isinstance(subsection, dict):
                        if subsection.get("type") == "Publication":
                            attr = subsection.get("attributes", [])
                            for a in attr:
                                if a.get("name") == "Title":
                                    title = a.get("value")
                                    if title:
                                        paper_ids.append(title)
                                elif a.get("name") == "Authors":
                                    authors.extend(a.get("value").split(", "))
                                elif a.get("name") == "DOI":
                                    doi = _extract_doi(a.get("value"))
                                    if doi:
                                        paper_ids.append(doi)
                            # Try to get DOI from links
                            links = subsection.get("links", [])
                            for link in links:
                                if "doi.org" in link.get("url"):
                                    doi = _extract_doi(link.get("url"))
                                    if doi:
                                        paper_ids.append(doi)
                            # Remove duplicates
                            paper_ids = list(set(paper_ids))
                                
                        elif subsection.get("type") == "Author":
                            attr = subsection.get("attributes", [])
                            for a in attr:
                                if a.get("name") == "Name":
                                    authors.append(a.get("value"))
                                    
                        elif subsection.get("type") == "Title":
                            attr = subsection.get("attributes", [])
                            for a in attr:
                                if a.get("name") == "Text":
                                    title = a.get("value")
            
            return paper_ids, authors, title
            
        except requests.exceptions.RequestException as e:
            if response.status_code == 429:  # Rate limit error
                wait_time = backoff_factor * (2**attempt)
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Request failed: {e}")
                break
                
    return [], [], None

@tool
def get_author_year(pmid: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Get the first author and publication year of a paper using its PMID.
    
    Args:
        pmid: The PubMed ID of the paper
        
    Returns:
        A tuple of (first_author_full_name, publication_year)
    """
    # Use the esummary utility to get the summary of the article
    handle = Entrez.esummary(db="pubmed", id=pmid)
    summary = Entrez.read(handle)
    handle.close()

    # Extract the first author from the summary
    authors = summary[0].get("AuthorList")
    first_author = authors[0] if authors else None

    pub_date = summary[0].get("PubDate")
    pub_year = (
        int(pub_date.split()[0]) if pub_date else None
    )  # Assuming PubDate is in the format 'YYYY MMM DD'

    return first_author, pub_year

@tool
def get_sra_authors(entrez_id: str) -> List[str]:
    """
    Get the list of authors from an SRA study using its Entrez ID.
    
    Args:
        entrez_id: The Entrez ID of the SRA study
        
    Returns:
        List of full author names
    """
    handle = Entrez.esummary(db="sra", id=entrez_id)
    summary = Entrez.read(handle)
    handle.close()
    
    authors = summary[0].get("Authors", "")
    if not authors:
        return []
    return [author.strip() for author in authors.split(',')]

@tool
def get_geo_authors(entrez_id: str) -> List[str]:
    """
    Get the list of authors from a GEO study using its Entrez ID.
    
    Args:
        entrez_id: The Entrez ID of the GEO study
        
    Returns:
        List of full author names
    """
    handle = Entrez.esummary(db="gds", id=entrez_id)
    summary = Entrez.read(handle)
    handle.close()
    
    authors = summary[0].get("Authors", "")
    if not authors:
        return []
    return [author.strip() for author in authors.split(',')]

@tool
def get_arrayexpress_authors(accession: str) -> List[str]:
    """
    Get the list of authors from an ArrayExpress study using its accession.
    
    Args:
        accession: The ArrayExpress accession (e.g., E-MTAB-1234)
        
    Returns:
        List of full author names
    """
    # ArrayExpress API endpoint
    base_url = "https://www.ebi.ac.uk/arrayexpress/json/v3/experiments"
    
    try:
        # Make request to ArrayExpress API
        response = requests.get(f"{base_url}/{accession}")
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Print response content for debugging
        print(f"ArrayExpress API Response for {accession}:")
        print(f"Status code: {response.status_code}")
        print(f"Response content: {response.text[:500]}...")  # Print first 500 chars
        
        # Parse JSON response
        data = response.json()
        
        # Extract authors from the response
        authors = data.get("experiment", {}).get("authors", [])
        
        if not authors:
            print(f"No authors found in response for {accession}")
            return []
            
        # Return full author names
        return [author.strip() for author in authors]
        
    except requests.exceptions.RequestException as e:
        print(f"Request error fetching authors from ArrayExpress: {e}")
        print(f"Response content: {response.text if 'response' in locals() else 'No response'}")
        return []
    except json.JSONDecodeError as e:
        print(f"JSON decode error for {accession}: {e}")
        print(f"Response content: {response.text if 'response' in locals() else 'No response'}")
        return []
    except Exception as e:
        print(f"Unexpected error in get_arrayexpress_authors for {accession}: {e}")
        print(f"Response content: {response.text if 'response' in locals() else 'No response'}")
        return []

@tool
def check_biorxiv_published_status(doi: str) -> Optional[str]:
    """
    Check if a bioRxiv preprint has been published and return the published DOI if it exists.
    
    Args:
        doi: The bioRxiv DOI to check
        
    Returns:
        The published DOI if the preprint has been published, None otherwise
    """
    api_url = f"https://api.biorxiv.org/details/biorxiv/{doi}"
    
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if "collection" in data and data["collection"]:
            for preprint in data["collection"]:
                if preprint.get("published"):
                    if preprint["published"] == "NA":
                        return None
                    return preprint["published"]
        return None
    except Exception as e:
        print(f"Error checking bioRxiv published status: {e}")
        return None

@tool
def check_medrxiv_published_status(doi: str) -> Optional[str]:
    """
    Check if a medRxiv preprint has been published and return the published DOI if it exists.
    
    Args:
        doi: The medRxiv DOI to check
        
    Returns:
        The published DOI if the preprint has been published, None otherwise
    """
    api_url = f"https://api.biorxiv.org/details/medrxiv/{doi}"
    
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if "collection" in data and data["collection"]:
            for preprint in data["collection"]:
                if preprint.get("published"):
                    if preprint["published"] == "NA":
                        return None
                    return preprint["published"]
        return None
    except Exception as e:
        print(f"Error checking medRxiv published status: {e}")
        return None

@tool
def check_preprint_published_status(doi: str) -> Optional[str]:
    """
    Check if a preprint has been published by checking both bioRxiv and medRxiv.
    
    Args:
        doi: The preprint DOI to check
        
    Returns:
        The published DOI if the preprint has been published, None otherwise
    """
    # First try bioRxiv
    published_doi = check_biorxiv_published_status(doi)
    if published_doi:
        return published_doi
        
    # Then try medRxiv
    published_doi = check_medrxiv_published_status(doi)
    if published_doi:
        return published_doi
        
    return None 
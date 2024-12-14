# import
## batteries
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Annotated, List, Dict, Optional
## 3rd party
from Bio import Entrez
from langchain_core.tools import tool
## package
from SRAgent.tools.utils import set_entrez_access

# functions
def esearch_batch(esearch_query: str, database: str, max_ids: Optional[int], verbose: bool=False) -> List[str]:
    # query 
    ids = []
    retstart = 0
    retmax = 10000
    while True:
        try:
            # search
            search_handle = Entrez.esearch(
                db=database, 
                term=esearch_query, 
                retstart=retstart, 
                retmax=retmax
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()
            # add IDs to 
            ids.extend(search_results.get("IdList", []))
            retstart += retmax
            time.sleep(0.34)
            if verbose:
                print(f"No. of IDs found: {len(ids)}", file=sys.stderr)
            if max_ids and len(ids) >= max_ids:
                break
            if retstart >= int(search_results['Count']):
                break
        except Exception as e:
            print(f"Error searching {database} with query: {esearch_query}: {str(e)}")
            break 
        
    # just unique IDs
    ids = list(set(ids))
        
    # return IDs
    if max_ids:
        ids = ids[:max_ids]
    return ids

@tool 
def esearch_scrna(
    esearch_query: Annotated[str, "Entrez query string."],
    database: Annotated[str, "Database name ('sra' or 'gds')"],
    previous_days: Annotated[int, "Number of days to search back."]=7,
    )-> Annotated[List[str], "Entrez IDs of database records"]:
    """
    Find recent single cell RNA-seq studies in SRA or GEO.
    sra : Sequence Read Archive (SRA)
    gds : Gene Expression Omnibus (GEO)
    
    Example query for single cell RNA-seq:
    '("single cell RNA sequencing" OR "single cell RNA-seq")'

    Args:
        esearch_query: Entrez query string.
        database: Database name ('sra' or 'gds').
        previous_days: Number of days to search back (default = 7).
    """
    set_entrez_access()
    # add date range
    start_date = datetime.now() - timedelta(days=previous_days)
    end_date = datetime.now()
    date_range = f"{start_date.strftime('%Y/%m/%d')}:{end_date.strftime('%Y/%m/%d')}[PDAT]"
    esearch_query += f" AND {date_range}"

    # add mouse and human organism
    esearch_query += " AND (Homo sapiens[Organism] OR Mus musculus[Organism])"

    # debug model
    max_ids = 1 if os.getenv("DEBUG_MODE") == "TRUE" else None

    # return entrez IDs 
    return esearch_batch(esearch_query, database, max_ids)

@tool 
def esearch(
    esearch_query: Annotated[str, "Entrez query string."],
    database: Annotated[str, "Database name (e.g., sra, gds, or pubmed)"],
    )-> Annotated[List[str], "Entrez IDs of database records"]:
    """
    Run an Entrez search query and return the Entrez IDs of the results.
    Example query for single cell RNA-seq:
        `("single cell"[Title] OR "single-cell"[Title] OR "scRNA-seq"[Title])`
    Example query for an ENA accession number (database = sra):
        `ERX13336121`
    Example query for a GEO accession number (database = gds):
        `GSE51372`
    """
    set_entrez_access()

    # debug model
    max_records = 2 if os.getenv("DEBUG_MODE") == "TRUE" else None

    # check input
    if esearch_query == "":
        return "Please provide a valid query."
    for x in ["SRR", "ERR", "GSE", "GSM", "GDS", "ERX", "DRR", "PRJ", "SAM", "SRP", "SRX"]:
        if esearch_query == x:
            return f"Invalid query: {esearch_query}"
    
    # query
    records = []
    retstart = 0
    retmax = 50
    while True:
        try:
            search_handle = Entrez.esearch(
                db=database, 
                term=esearch_query, 
                retstart=retstart, 
                retmax=retmax
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()
            # delete unneeded keys
            to_rm = ["RetMax", "RetStart"]
            for key in to_rm:
                if key in search_results.keys():
                    del search_results[key]
            # add to records
            records.append(str(search_results))
            # update retstart
            retstart += retmax
            time.sleep(0.34)
            if max_records and len(records) >= max_records:
                break
            if retstart >= int(search_results['Count']):
                break
        except Exception as e:
            print(f"Error searching {database} with query: {esearch_query}: {str(e)}", file=sys.stderr)
            break 
        
    # return records
    if len(records) == 0:
        return f"No records found for query: {esearch_query}"
    if max_records:
        records = records[:max_records]  # debug
    return str(records)


if __name__ == "__main__":
    # setup
    from dotenv import load_dotenv
    load_dotenv()
    Entrez.email = os.getenv("EMAIL")

    # scRNA-seq
    #query = '("single cell RNA sequencing" OR "single cell RNA-seq")'
    query = '("bulk RNA sequencing")'
    #input = {"esearch_query" : query, "database" : "sra", "previous_days" : 90}
    input = {"esearch_query" : query, "database" : "gds", "previous_days" : 60}
    # print(esearch_scrna.invoke(input))

    # esearch accession
    input = {"esearch_query" : "GSE51372", "database" : "sra"}
    input = {"esearch_query" : "GSE121737", "database" : "gds"}
    input = {"esearch_query" : "35447314", "database" : "sra"}
    input = {"esearch_query" : "35447314", "database" : "gds"}
    #print(esearch.invoke(input))


# import
## batteries
import os
import time
import json
from pprint import pprint
from typing import Annotated, List, Dict, Tuple, Optional, Union, Any, Callable
## 3rd party
import xml.etree.ElementTree as ET
from Bio import Entrez
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
## package
from SRAgent.tools.entrez_db import which_entrez_databases
from SRAgent.tools.utils import batch_ids, truncate_values, xml2json

def elink_error_check(batch_record):
    # Parse XML
    try:
        root = ET.fromstring(batch_record)
    except ET.ParseError as e:
        return f"XML Parsing Error: {e}"

    print(root.find("ERROR").text);

    # Check for errors in the response
    return root.find("ERROR")

@tool
def elink(
    entrez_ids: Annotated[List[str], "List of Entrez IDs"],
    source_db: Annotated[str, "Source database (e.g., 'sra')"],
    target_db: Annotated[str, "Target database (e.g., 'bioproject', 'biosample', 'pubmed')"],
) -> Annotated[str, "eLink results in XML format"]:
    """
    Find related entries between Entrez databases, particularly useful for finding
    BioProject, BioSample, or publication records related to SRA entries.
    """
    ntries = 6
    batch_size = 200  # Maximum number of IDs per request as per NCBI guidelines
    records = []

    for id_batch in batch_ids(entrez_ids, batch_size):
        time.sleep(0.34)  # Respect NCBI's rate limits (no more than 3 requests per second)
        id_str = ",".join(id_batch)
        
        for i in range(ntries):
            try:
                handle = Entrez.elink(
                    id=id_str,
                    dbfrom=source_db,
                    db=target_db,
                    retmode="xml",
                    timeout=10
                )
                batch_record = handle.read()
                handle.close()
            except Entrez.Parser.ValidationError:
                batch_record = f"ERROR: Failed to find links for IDs: {id_str}"
                continue
            except Exception as e:
                batch_record = f"ERROR: {e}"
                continue
            finally:
                try:
                    handle.close()
                except:
                    pass  # Handle cases where the handle might not be open
            # Check for errors in the response
            if elink_error_check(batch_record) is not None:
                batch_record = f"ERROR: Failed to find links for IDs: {id_str}. Verify database names ({source_db}, {target_db}) and Entrez IDs."
                time.sleep(i+1)
                continue
            else:
                break
    
        # Decode the record if necessary
        if isinstance(batch_record, str) and batch_record.startswith("ERROR:"):
            records.append(batch_record)
            continue
        elif isinstance(batch_record, bytes):
            try:
                batch_record = batch_record.decode("utf-8")
            except Exception as e:
                records.append(f"Decoding error: {e}")
                continue

        # Truncate long values in the record
        batch_record = truncate_values(batch_record, max_length=1000)

        # convert to XML to JSON
        batch_record = xml2json(batch_record)

        # Check for errors in the response
        if "ERROR" in batch_record.upper():
            batch_record = f"Failed to find links for IDs: {id_str}. Verify database names ({source_db}, {target_db}) and Entrez IDs."

        # Append the batch record to the list of records
        records.append(batch_record)
    
    # Combine all batch records into a single string
    return "\n".join(records)


if __name__ == "__main__":
    # setup
    from dotenv import load_dotenv
    load_dotenv()
    Entrez.email = os.getenv("EMAIL")
    Entrez.api_key = os.getenv('NCBI_API_KEY')

    # test esummary
    #input = {"entrez_ids" : ["35966237", "200254051"], "source_db" : "gds", "target_db" : "sra"}
    #input = {"entrez_ids" : ['200121737', '100024679', '303444964', '303444963', '303444962'], "source_db" : "gds", "target_db" : "sra"}
    input = {"entrez_ids" : ["200277303"], "source_db" : "gds", "target_db" : "sra"}
    print(elink.invoke(input))

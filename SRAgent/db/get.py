# import
## batteries
import os
from typing import List, Dict, Any, Tuple, Optional
## 3rd party
import psycopg2
import pandas as pd
from pypika import Query, Table, Field, Column, Criterion
from psycopg2.extras import execute_values
from psycopg2.extensions import connection

# functions
def db_get_srx_records(conn: connection, column: str="entrez_id", database: str="sra") -> List[int]:
    """
    Get the entrez_id values of all SRX records that have not been processed.
    Args:
        conn: Connection to the database.
        database: Name of the database to query.
    Returns:
        List of entrez_id values of SRX records that have not been processed.
    """
    srx_metadata = Table("srx_metadata")
    target_column = getattr(srx_metadata, column)
    stmt = Query \
        .from_(srx_metadata) \
        .select(target_column) \
        .distinct() \
        .where(srx_metadata.database == database)
        
    # Fetch the results and return a list of {target_column} values
    return [row[0] for row in execute_query(stmt, conn)]

def db_get_unprocessed_records(conn: connection, database: str="sra", max_records: int=3) -> pd.DataFrame:
    """
    Get all suitable unprocessed SRX records
    Args:
        conn: Connection to the database.
        database: Name of the database to query.
    Returns:
        Table of unprocessed SRX records.
    """
    srx_metadata = Table("srx_metadata")
    srx_srr = Table("srx_srr")

    stmt = Query \
        .from_(srx_metadata) \
        .inner_join(srx_srr) \
        .on(srx_metadata.srx_accession == srx_srr.srx_accession) \
        .where(Criterion.all([
            srx_metadata.screcounter_status.isnull(),
            srx_metadata.database == database,
            srx_metadata.is_illumina == "yes",
            srx_metadata.is_single_cell == "yes",
            srx_metadata.is_paired_end == "yes",
            srx_metadata.is_10x.isin(["yes", "unsure"])
        ])) \
        .select(
            srx_metadata.srx_accession.as_("sample"),
            srx_srr.srr_accession.as_("accession"),
            srx_metadata.entrez_id.as_("entrez_id"),
            srx_metadata.tech_10x.as_("lib_prep_method"),
            srx_metadata.organism.as_("organism")
        ) \
        .limit(max_records)
        
    # fetch as pandas dataframe
    return pd.read_sql(str(stmt), conn)
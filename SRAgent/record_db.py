# import
## batteries
import os
import warnings
from typing import List, Dict, Any, Tuple, Optional
## 3rd party
import psycopg2
import pandas as pd
from pypika import Query, Table, Field, Column
from psycopg2.extras import execute_values
from psycopg2.extensions import connection

# Suppress the specific warning
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")

# functions
def db_connect() -> connection:
    """Connect to the sql database"""
    host = os.path.join(os.path.expanduser("~"), "cloudsql", os.environ["GCP_SQL_DB_HOST"])
    db_params = {
        'host': host,
        'database': os.environ["GCP_SQL_DB_NAME"],
        'user': os.environ["GCP_SQL_DB_USERNAME"],
        'password': os.environ["GCP_SQL_DB_PASSWORD"],
        'port': '5432',
        'connect_timeout': 10 
    }
    return psycopg2.connect(**db_params)

def db_list_tables(conn: connection) -> List[Tuple[str]]:
    """
    List all tables in the public schema of the database.
    Args:
        conn: Connection to the database.
    Returns:
        List of table names in the public schema.
    """
    tables = Table('tables', schema='information_schema')
    query = Query.from_(tables).select('table_name').where(tables.table_schema == 'public')
    with conn.cursor() as cur:
        cur.execute(str(query))
        tables = cur.fetchall()
        return tables

def db_glimpse_tables(conn: connection) -> None:
    """
    Print the first 5 rows of each table in the database.
    Args:
        conn: Connection to the database.
    """
    for table in db_list_tables(conn):
        table_name = table[0]
        print(f"#-- Table: {table[0]} --#")
        df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 5", conn)
        print(df)

def execute_query(stmt, conn: connection) -> Optional[List[Tuple]]:
    """
    Execute a query and return the results, if any.
    Args:
        stmt: Query to execute.
        conn: Connection to the database.
    Returns:
        Results of the query, if any.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(str(stmt))
            conn.commit() 
            # Return the results of the query, if any
            try:
                return cur.fetchall()
            except psycopg2.ProgrammingError:
                return None
    except psycopg2.errors.DuplicateTable as e:
        print(f"Table already exists: {e}")

def db_get_processed_records(conn: connection, column: str="entrez_id", database: str="sra") -> List[int]:
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
        .where((srx_metadata.processed != "complete") | (srx_metadata.processed.isnull())) \
        .where(srx_metadata.database == database)
        
    # Fetch the results and return a list of {target_column} values
    return [row[0] for row in execute_query(stmt, conn)]

def db_add(data_list: List[Dict[str, Any]], table: str, conn: connection) -> None:
    """
    Add a list of dictionaries to a table in the database.
    Args:
        data_list: List of dictionaries to add to the table. If a key is missing from a dictionary, the corresponding column in the table will be NULL.
        table: Name of the table to add the data to.
        conn: Connection to the database.
    """
    if not data_list:
        return

    # Table and columns
    srx_metadata = Table(table)
    columns = list(data_list[0].keys())

    # Values for batch insert
    values = [tuple(d.get(col) for col in columns) for d in data_list]

    # Build the query
    query = f"""
        INSERT INTO {srx_metadata} ({', '.join(columns)})
        VALUES %s
    """
    try:
        with conn.cursor() as cur:
            execute_values(cur, query, values)  # Batch insert
            conn.commit()
    except psycopg2.errors.UniqueViolation:
        pass
    

# main
if __name__ == '__main__':
    # connect to database
    from dotenv import load_dotenv
    load_dotenv()

    # glimpse tables
    with db_connect() as conn:
        db_glimpse_tables(conn)
    exit();

    # get processed entrez ids
    # with db_connect() as conn:
    #     print(db_get_processed_records(conn, "srx_accession"))

    data = [{
        "database": "sra",
        "entrez_id": 123456,
        "srx_accession": "test",
        "is_illumina": "unsure",
        "is_single_cell": "unsure",
        "is_paired_end": "unsure",
        "is_10x": "unsure",
        "tech_10x": "other",
        "organism": "other"
    }]
    with db_connect() as conn:
        db_add(data, "srx_metadata", conn)

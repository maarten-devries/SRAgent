# import
## batteries
import os
import operator
from functools import partial
from typing import Annotated, List, Dict, Any, TypedDict, Sequence
## 3rd party
import pandas as pd
import gspread
from gspread_dataframe import set_with_dataframe
from langgraph.types import Send
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph
## package
from SRAgent.workflows.convert import create_convert_graph, invoke_convert_graph
from SRAgent.workflows.metadata import create_metadata_graph, invoke_metadata_graph, get_metadata_items
from SRAgent.record_db import db_connect, db_add, db_get_processed_records

# classes
class GraphState(TypedDict):
    """
    Shared state of the agents in the graph
    """
    # messages
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # database
    database: str
    # dataset entrez ID
    entrez_id: str
    # accessions
    SRX: Annotated[List[str], operator.add]
    # is_illumina
    is_illumina: Annotated[List[str], operator.add]
    # is_single_cell
    is_single_cell: Annotated[List[str], operator.add]
    # is_paired_end
    is_paired_end: Annotated[List[str], operator.add]
    # is_10x
    is_10x: Annotated[List[str], operator.add]
    # 10X tech
    tech_10x: Annotated[List[str], operator.add]
    # organism
    organism: Annotated[List[str], operator.add]


# functions
def create_convert_graph_node():
    """
    Create a convert graph node
    """
    graph = create_convert_graph()
    def invoke_convert_graph_node(state: GraphState) -> Dict[str, Any]:
        entrez_id = state["entrez_id"]
        database = state["database"]
        message = "\n".join([
            f"Convert Entrez ID {entrez_id} to SRX (or ERX) accessions.",
            f"The Entrez ID is associated with the {database} database."
        ])
        input = {"messages": [HumanMessage(message)]}
        return graph.invoke(input)
    return invoke_convert_graph_node

def continue_to_metadata(state: GraphState) -> List[Dict[str, Any]]:
    """
    Parallel invoke of the metadata graph
    Return:
        List of metadata graph outputs
    """
    # format the prompt for the metadata graph
    prompt = "\n".join([
        "For the SRA accession {SRX_accession}, find the following information:",
        ] + list(get_metadata_items().values())
    )
    
    # submit each accession to the metadata graph    
    ## filter out existing SRX accessions
    SRX_filt = []
    with db_connect() as conn:
        existing_srx = set(db_get_processed_records(conn, column="srx_accession", database=state["database"]))
        SRX_filt = [x for x in state["SRX"] if x not in existing_srx]

    ## handle case where no SRX accessions are found
    if len(SRX_filt) == 0:
        add_entrez_id_to_db(state["entrez_id"], state["database"])
        return []

    ## submit each SRX accession to the metadata graph
    responses = []
    for SRX_accession in SRX_filt:
        input = {
            "database": state["database"],
            "entrez_id": state["entrez_id"],
            "SRX": SRX_accession,
            "messages": [HumanMessage(prompt.format(SRX_accession=SRX_accession))]
        }
        responses.append(Send("metadata_graph_node", input))
    return responses

def add_entrez_id_to_db(entrez_id: int, database: str) -> None:
    """
    Add the entrez ID to the record database.
    Args:
        entrez_id: Entrez ID to add
        database: Entrez database (e.g., sra or gds)
    """
    data = [{
        "database": database,
        "entrez_id": entrez_id,
    }]
    with db_connect() as conn:
        db_add(data, "srx_metadata", conn)

def final_state(state: GraphState) -> Dict[str, Any]:
    """
    Return the final state of the graph
    """
    # filter to messages that 
    messages = []
    for msg in state["messages"]:
        try:
            msg = [msg.content]
        except AttributeError:
            msg = [x.content for x in msg]
        for x in msg:
            if x.startswith("# SRX accession: "):
                messages.append(x)
    # final message
    message = "\n".join(messages)
    return {
        "messages": [AIMessage(content=message)]
    }

def create_SRX_info_graph(db_add:bool = True):
    # metadata subgraph
    invoke_metadata_graph_p = partial(
        invoke_metadata_graph,
        graph=create_metadata_graph(db_add=db_add),
        to_return=["messages"]
    )

    #-- graph --#
    workflow = StateGraph(GraphState)

    # nodes
    workflow.add_node("convert_graph_node", create_convert_graph_node())
    workflow.add_node("metadata_graph_node", invoke_metadata_graph_p)
    workflow.add_node("final_state_node", final_state)

    # edges
    workflow.add_edge(START, "convert_graph_node")
    workflow.add_conditional_edges("convert_graph_node", continue_to_metadata, ["metadata_graph_node"])
    workflow.add_edge("metadata_graph_node", "final_state_node")
    workflow.add_edge("final_state_node", END)

    # compile the graph
    graph = workflow.compile()
    return graph

# main
if __name__ == "__main__":
    from functools import partial
    from Bio import Entrez

    #-- setup --#
    from dotenv import load_dotenv
    load_dotenv()
    Entrez.email = os.getenv("EMAIL")

    #-- graph --#
    #input = {"entrez_id": 25576380, "database": "sra"}
    input = {"entrez_id": 36106630, "database": "sra"}
    graph = create_SRX_info_graph()
    for step in graph.stream(input, config={"max_concurrency" : 3, "recursion_limit": 100}):
        print(step)

    # save the graph
    # from SRAgent.utils import save_graph_image
    # save_graph_image(graph)
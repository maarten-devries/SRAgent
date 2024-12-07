# import
## batteries
import os
from typing import Annotated, List, Dict, Tuple, Optional, Union, Any, Callable
## 3rd party
from google.cloud import bigquery
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
## package
from utils import to_json, join_accs

# functions
def create_get_study_metadata(client):
    @tool
    def get_study_metadata(
        study_accessions: Annotated[List[str], "A list of SRA study accession numbers (SRP)"]
        ) -> Annotated[str, "JSON string of SRA experiment metadata"]:
        """
        Get study-level metadata for a list of SRA study accessions.
        The metadata fields returned:
        - sra_study: SRA study accession (the query accession)
        - bioproject: BioProject accession (parent of study)
        - experiments: Comma-separated list of associated experiment accessions (SRX)
        """
        query = f"""
        WITH distinct_values AS (
            SELECT DISTINCT
                m.sra_study,
                m.bioproject,
                m.experiment
            FROM `nih-sra-datastore.sra.metadata` as m
            WHERE m.sra_study IN ({join_accs(study_accessions)})
        )
        SELECT 
            sra_study,
            bioproject,
            STRING_AGG(experiment, ',') as experiments
        FROM distinct_values
        GROUP BY sra_study, bioproject
        """
        return to_json(client.query(query))
    return get_study_metadata

def create_get_experiment_metadata(client):
    @tool
    def get_experiment_metadata(
        experiment_accessions: Annotated[List[str], "A list of SRA experiment accession numbers (SRX)"]
        ) -> Annotated[str, "JSON string of SRA experiment metadata"]:
        """
        Get experiment-level metadata for a list of SRA experiment accessions.
        The metadata fields returned:
        - experiment: SRA experiment accession (the query accession)
        - sra_study: SRA study accession (parent of experiment)
        - library_name: Library name (e.g., 1, 2, 3)
        - librarylayout: Library layout (e.g., single, paired)
        - libraryselection: Library selection (e.g., random, PCR)
        - librarysource: Library source (e.g., transcriptomic, genomic)
        - platform: Sequencing platform (e.g., Illumina, PacBio)
        - instrument: Sequencing instrument (e.g., HiSeq, NovaSeq)
        - acc: Comma-separated list of associated run accessions (SRR)
        """
        query = f"""
        WITH distinct_values AS (
            SELECT DISTINCT
                m.experiment,
                m.sra_study,
                m.library_name, 
                m.librarylayout,
                m.libraryselection, 
                m.librarysource,
                m.platform,
                m.instrument,
                m.acc,
            FROM `nih-sra-datastore.sra.metadata` as m
            WHERE m.experiment IN ({join_accs(experiment_accessions)})
        )
        SELECT
            experiment,
            library_name,
            librarylayout,
            libraryselection,
            librarysource,
            platform,
            instrument,
            STRING_AGG(acc, ',') as acc
        FROM distinct_values
        GROUP BY experiment, library_name, librarylayout, libraryselection, librarysource, platform, instrument
        """
        return to_json(client.query(query))
    return get_experiment_metadata

def create_get_run_metadata(client):
    @tool
    def get_run_metadata(
        run_accessions: Annotated[List[str], "A list of SRA run accession numbers (SRR)"]
        ) -> Annotated[str, "JSON string of SRA run metadata"]:
        """
        Get run-level metadata for a list of SRA run accessions.
        The metadata fields returned:
        - acc: SRA run accession (the query accession)
        - experiment: SRA experiment accession (parent of run)
        - biosample: BioSample accession (parent of run)
        - organism: Organism name
        - assay_type: Assay type (e.g., RNA-Seq, ChIP-Seq)
        - mbases: Total bases sequenced (in megabases)
        - avgspotlen: Average spot length (in base pairs)
        - insertsize: Insert size (in base pairs)
        """
        query = f"""
        SELECT 
            m.acc,
            m.experiment,
            m.biosample,
            m.organism,
            m.assay_type,
            m.mbases,
            m.avgspotlen,
            m.insertsize,            
        FROM `nih-sra-datastore.sra.metadata` as m
        WHERE m.acc IN ({join_accs(run_accessions)})
        """
        return to_json(client.query(query))
    return get_run_metadata


def create_get_study_experiment_run(client):
    @tool
    def get_study_experiment_run(
        accessions: Annotated[List[str], "A list of SRA study accession numbers"]
        ) -> Annotated[str, "JSON string of SRA experiment metadata"]:
        """
        Get study, experiment, and run accessions for a list of SRA and/or ENA accessions.
        The accessions can be from any level of the SRA hierarchy: study, experiment, or run.
        The metadata fields returned:
        - study_accession: SRA or ENA study accession (SRP or PRJNA)
        - experiment_accession: SRA or ENA experiment accession (SRX or ERX)
        - run_accession: SRA or ENA run accession (SRR or ERR)
        """
        # get study accessions
        study_acc = [x for x in accessions if x.startswith("SRP") or x.startswith("PRJNA")]
        exp_acc = [x for x in accessions if x.startswith("SRX") or x.startswith("ERX")]        
        run_acc = [x for x in accessions if x.startswith("SRR") or x.startswith("ERR")]

        # create WHERE query
        study_query = f"m.sra_study IN ({join_accs(study_acc)})" if len(study_acc) > 0 else None
        exp_query =  f"m.experiment IN ({join_accs(exp_acc)})" if len(exp_acc) > 0 else None
        run_query = f"m.acc IN ({join_accs(run_acc)})" if len(run_acc) > 0 else None
        query = " OR ".join([x for x in [study_query, exp_query, run_query] if x is not None])

        # if empty
        if query is None or query == "":
            return "No valid accessions provided."

        # create full query
        query = f"""
        SELECT DISTINCT
            m.sra_study AS study_accession,
            m.experiment AS experiment_accession,
            m.acc AS run_accession
        FROM `nih-sra-datastore.sra.metadata` as m
        WHERE {query}
        """

        # return query results
        return to_json(client.query(query))
    return get_study_experiment_run

def create_bigquery_agent(model_name="gpt-4o") -> Callable:
    # create model
    model = ChatOpenAI(model=model_name, temperature=0.1)

    # init client
    client = bigquery.Client()

    # set tools
    tools = [
        create_get_study_experiment_run(client),
        create_get_study_metadata(client),
        create_get_experiment_metadata(client),
        create_get_run_metadata(client)
    ]
  
    # state modifier
    state_mod = "\n".join([
        # Role and Purpose
        "You are an expert bioinformatician specialized in querying the Sequence Read Archive (SRA) database.",
        "Your purpose is to retrieve and analyze metadata across SRA's hierarchical structure: studies (SRP) → experiments (SRX) → runs (SRR).",
        # Tool Capabilities
        "You have access to four specialized tools:",
        " 1. get_study_experiment_run: Retrieves study, experiment, and run accessions",
        " 2. get_study_metadata: Retrieves study and associated experiment accessions",
        " 3. get_experiment_metadata: Retrieves experiment details and associated run accessions",
        " 4. get_run_metadata: Retrieves detailed run-level information",
        # Tool Usage
        "Use the get_study_experiment_run tool to convert accessions between study, experiment, and run levels.",
        "Use the get_*_metadata tools to retrieve metadata for a specific accession type.",
        "Chain the tools as needed to gather complete information for a given study, experiment, or run.",
        # Response Guidelines
        "When responding:",
        " - If the query mentions one accession type but asks about another, automatically perform the necessary conversions",
        " - Chain multiple tool calls when needed to gather complete information",
        " - If you receive an error, explain it clearly and suggest alternatives",
        # Output Format
        "Keep responses concise and structured:",
        " - Present metadata as key-value pairs",
        " - Group related information",
        " - Include accession IDs in outputs",
        " - No markdown formatting",
    ])

    # create agent
    agent = create_react_agent(
        model=model,
        tools=tools,
        state_modifier=state_mod
    )
    @tool
    def invoke_bigquery_agent(
        message: Annotated[str, "Message to send to the BigQuery agent"],
    ) -> Annotated[dict, "Response from the BigQuery agent"]:
        """
        Invoke the BigQuery agent with a message.
        The BigQuery agent will search the SRA database with BigQuery.
        """
        # Invoke the agent with the message
        result = agent.invoke({"messages" : [AIMessage(content=message)]})
        return {
            "messages": [AIMessage(content=result["messages"][-1].content, name="bigquery_agent")]
        }
    return invoke_bigquery_agent

if __name__ == "__main__":
    # setup
    from dotenv import load_dotenv
    load_dotenv()
    client = bigquery.Client()

    # test agent
    bigquery_agent = create_bigquery_agent()
    # print(bigquery_agent.invoke({"message" : "Get study metadata for SRP548813"}))
    # print(bigquery_agent.invoke({"message" : "Get experiment metadata for SRP548813"}))
    # print(bigquery_agent.invoke({"message" : "Get the number of base pairs for all runs in SRP548813"}))
    # print(bigquery_agent.invoke({"message" : "Convert SRP548813 to SRR"}))

    # test tools
    ## get_study_experiment_run
    # get_study_experiment_run = create_get_study_experiment_run(client)
    # print(get_study_experiment_run.invoke({"accessions" : ["SRP548813", "SRX26939191", "SRR31573627"]}))
    # print(get_study_experiment_run.invoke({"accessions" : ["XXX"]}))

    ## get_study_metadata
    # get_study_metadata = create_get_study_metadata(client)
    # print(get_study_metadata.invoke({"study_accessions" : ["SRP548813"]}))
    # print(get_study_metadata.invoke({"study_accessions" : ["XXX"]}))

    ## get_experiment_metadata
    # get_experiment_metadata = create_get_experiment_metadata(client)
    # print(get_experiment_metadata.invoke({"experiment_accessions" : ["SRX26939191"]}))

    ## get_run_metadata
    # get_run_metadata = create_get_run_metadata(client)
    # print(get_run_metadata.invoke({"run_accessions" : ["SRR31573627", "SRR31573628"]}))
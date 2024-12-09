# import
## batteries
import os
import sys
import asyncio
import argparse
from Bio import Entrez
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from SRAgent.cli.utils import CustomFormatter
from SRAgent.agents_OLD.fetch import create_entrez_agent_stream

# functions
def entrez_agent_parser(subparsers):
    help = 'Entrez Agent: general agent for working with Entrez databases.'
    desc = """
    # Example prompts:
    1. "Convert GSE121737 to SRX accessions"
    2. "Obtain any available publications for GSE196830"
    3. "Obtain the SRR accessions for SRX4967527"
    4. "Is SRR8147022 paired-end Illumina data?"
    5. "Is SRP309720 10X Genomics data?"
    """
    sub_parser = subparsers.add_parser(
        'entrez', help=help, description=desc, formatter_class=CustomFormatter
    )
    sub_parser.set_defaults(func=entrez_agent_main)
    sub_parser.add_argument('prompt', type=str, help='Prompt for the agent')    
    sub_parser.add_argument('--no-summaries', action='store_true', default=False,
                            help='No LLM summaries')
    sub_parser.add_argument('--max-concurrency', type=int, default=3, 
                            help='Maximum number of concurrent processes')
    sub_parser.add_argument('--recursion-limit', type=int, default=40,
                            help='Maximum recursion limit')
    
def entrez_agent_main(args):
    """
    Main function for invoking the entrez agent
    """
    # set email and api key
    Entrez.email = os.getenv("EMAIL")
    Entrez.api_key = os.getenv("NCBI_API_KEY")

    # invoke agent with streaming
    config = {
        "max_concurrency" : args.max_concurrency,
        "recursion_limit": args.recursion_limit
    }
    input = {"messages": [HumanMessage(content=args.prompt)]}
    results = asyncio.run(create_entrez_agent_stream(input, config, summarize_steps=True))
    print(results)
            

# main
if __name__ == '__main__':
    pass
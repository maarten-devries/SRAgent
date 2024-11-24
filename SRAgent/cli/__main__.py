#!/usr/bin/env python
# libraries
## batteries
import os
import sys
import argparse
## 3rd party
from dotenv import load_dotenv
## package
from SRAgent.cli.utils import CustomFormatter
from SRAgent.cli.entrez_agent import entrez_agent_parser, entrez_agent_main
from SRAgent.cli.metadata_agent import metadata_agent_parser, metadata_agent_main


# functions
def arg_parse(args=None) -> dict:
    """
    Parse command line arguments.
    """
    desc = "SRAgent: A multi-agent tool for working with the SRA"
    epi = """DESCRIPTION:
    SRAgent is a multi-agent tool for working with the Sequence Read Archive (SRA) database
    and other Entriz databases. It is designed to be a flexible and easy-to-use tool for
    interacting with the SRA and Entrez.
    """
    # check for OP
    if os.getenv("OPENAI_API_KEY") is None:
        raise ValueError("OPENAI_API_KEY not found in environment")
    
    # main parser
    parser = argparse.ArgumentParser(
        description=desc,
        epilog=epi,
        formatter_class=CustomFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # subparsers
    ## Entrez agent
    entrez_agent_parser(subparsers)
    ## Metadata agent
    metadata_agent_parser(subparsers)
    
    # parsing args
    return parser.parse_args()

def main():
    # load environment variables
    load_dotenv()
    # parsing args
    args = arg_parse()
    
    # which subcommand
    if args.command == "entrez":
        entrez_agent_main(args)
    elif args.command == "metadata":
        metadata_agent_main(args)
    else:
        print("No command specified. Exiting ...")
        sys.exit(0)

    
if __name__ == "__main__":
    main()
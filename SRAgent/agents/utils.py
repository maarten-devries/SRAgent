import os
import sys
from importlib import resources
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dynaconf import Dynaconf

def load_settings() -> Dict[str, Any]:
    """
    Load settings from settings.yml file
    
    Args:
        env: Environment to load settings for ('test' or 'prod')
        
    Returns:
        Dictionary containing settings for the specified environment
    """
    # get path to settings
    if os.getenv("DYNACONF_SETTINGS_PATH"):
        s_path = os.getenv("DYNACONF_SETTINGS_PATH")
    else:
        s_path = str(resources.files("SRAgent").joinpath("settings.yml"))
    if not os.path.exists(s_path):
        raise FileNotFoundError(f"Settings file not found: {s_path}")
    settings = Dynaconf(
        settings_files=[s_path], 
        environments=True, 
        env_switcher="DYNACONF"
    )
    return settings

def set_model(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
    agent_name: str = "default",
) -> ChatOpenAI:
    """
    Create a model instance with settings from configuration
    
    Args:
        model_name: Override model name from settings
        temperature: Override temperature from settings
        reasoning_effort: Override reasoning effort from settings
        agent_name: Name of the agent to get settings for
    Returns:
        Configured model instance
    """
    # Load settings
    settings = load_settings()
    
    # Use provided params or get from settings
    model_name = model_name or settings["models"].get(agent_name, settings["models"]["default"])
    temp = temperature or settings["temperature"].get(agent_name, settings["temperature"]["default"])
    effort = reasoning_effort or settings["reasoning_effort"].get(agent_name, settings["reasoning_effort"]["default"])

    # Only reasoning effort can be set for certain models
    if model_name.startswith("gpt-4o"):
        effort = None
    elif model_name.startswith("o1") or model_name.startswith("o3"):
        # Always set temperature to 0 for o1/o3 models if not explicitly provided
        temp = temperature if temperature is not None else 0
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    # Initialize model
    model = ChatOpenAI(model_name=model_name, temperature=temp, reasoning_effort=effort)

    return model

def create_step_summary_chain(model: str="gpt-4o-mini", max_tokens: int=45):
    """
    Create a chain of tools to summarize each step in a workflow.
    Args:
        model: The OpenAI model to use for the language model.
        max_tokens: The maximum number of tokens to use for the summary.
    Returns:
        A chain of tools to summarize each step in a workflow.
    """
    # Create the prompt template
    template = "\n".join([
        "Concisely summarize the provided step in the langgraph workflow.",
        f"The summary must be {max_tokens} tokens or less.",
        "Do not use introductory words such as \"The workflow step involves\"",
        "Write your output as plain text instead of markdown.",
        "#-- The workflow step --#",
        "{step}"
    ])
    prompt = PromptTemplate(input_variables=["step"], template=template)

    # Initialize the language model
    llm = ChatOpenAI(model_name=model, temperature=0, max_tokens=max_tokens)

    # Return the LLM chain
    return prompt | llm


async def create_agent_stream(
    input,  
    create_agent_func,
    config: dict={}, 
    summarize_steps: bool=False
) -> str:
    """
    Create an Entrez agent and stream the steps.
    Args:
        input: Input message to the agent.
        create_agent_func: Function to create the agent.
        config: Configuration for the agent.
        summarize_steps: Whether to summarize the steps.
    Returns:
        The final step message.
    """
    # create entrez agent
    agent = create_agent_func(return_tool=False)

    # create step summary chain
    step_summary_chain = create_step_summary_chain() if summarize_steps else None
    
    # invoke agent
    step_cnt = 0
    final_step = ""
    async for step in agent.astream(input, stream_mode="values", config=config):
        step_cnt += 1
        final_step = step
        # summarize step
        if step_summary_chain:
            msg = step_summary_chain.invoke({"step": step.get("messages")})
            print(f"Step {step_cnt}: {msg.content}", file=sys.stderr)
        else:
            try:
                if "messages" in step and step["messages"]:
                    last_msg = step["messages"][-1].content
                    if last_msg != "":
                        print(f"Step {step_cnt}: {last_msg}", file=sys.stderr)
                    else:
                        step_cnt -= 1
            except (KeyError, IndexError, AttributeError):
                print(f"Step {step_cnt}: {step}", file=sys.stderr)
    # get final step, and handle different types
    try:
        final_step = final_step["agent"]["messages"][-1].content
    except KeyError:
        try:
            final_step = final_step["messages"][-1].content
        except (KeyError, IndexError, AttributeError):
            if isinstance(final_step, str):
                return final_step
            return str(final_step)
    except TypeError:
        return str(final_step)
    return final_step

# main
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)
    load_settings()
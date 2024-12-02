import asyncio
import operator
import json

from tavily import TavilyClient, AsyncTavilyClient
from langchain_community.document_loaders import WebBaseLoader

from langchain_anthropic import ChatAnthropic

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableConfig

from langgraph.graph import START, END, StateGraph

from pydantic import BaseModel, Field
from typing_extensions import Annotated, Any, List, Optional
from dataclasses import dataclass, field

import configuration

# -----------------------------------------------------------------------------
# LLMs
rate_limiter = InMemoryRateLimiter(
    requests_per_second=4,
    check_every_n_seconds=0.1,
    max_bucket_size=10,  # Controls the maximum burst size.
)
claude_3_5_sonnet = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0, rate_limiter=rate_limiter)

# -----------------------------------------------------------------------------
# Search
tavily_client = TavilyClient()
tavily_async_client = AsyncTavilyClient()

# -----------------------------------------------------------------------------
# Utils


def deduplicate_and_format_sources(
    search_response, max_tokens_per_source, include_raw_content=True
):
    """
    Takes either a single search response or list of responses from Tavily API and formats them.
    Limits the raw_content to approximately max_tokens_per_source.
    include_raw_content specifies whether to include the raw_content from Tavily in the formatted string.

    Args:
        search_response: Either:
            - A dict with a 'results' key containing a list of search results
            - A list of dicts, each containing search results

    Returns:
        str: Formatted string with deduplicated sources
    """
    # Convert input to list of results
    if isinstance(search_response, dict):
        sources_list = search_response["results"]
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and "results" in response:
                sources_list.extend(response["results"])
            else:
                sources_list.extend(response)
    else:
        raise ValueError(
            "Input must be either a dict with 'results' or a list of search results"
        )

    # Deduplicate by URL
    unique_sources = {}
    for source in sources_list:
        if source["url"] not in unique_sources:
            unique_sources[source["url"]] = source

    # Format output
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += (
            f"Most relevant content from source: {source['content']}\n===\n"
        )
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get("raw_content", "")
            if raw_content is None:
                raw_content = ""
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"

    return formatted_text.strip()


def format_all_notes(completed_notes: list[str]) -> str:
    """Format a list of notes into a string"""
    formatted_str = ""
    for idx, people_notes in enumerate(completed_notes, 1):
        formatted_str += f"""
{'='*60}
People {idx}:
{'='*60}
Notes from research:
{people_notes}"""
    return formatted_str


# -----------------------------------------------------------------------------
# Schema


class Person(BaseModel):
    """A class representing a person to research."""

    name: Optional[str] = None
    """The name of the person."""
    company: Optional[str] = None
    """The current company of the person."""
    linkedin: Optional[str] = None
    """The Linkedin URL of the person."""
    email: str
    """The email of the person."""
    role: Optional[str] = None
    """The current title of the person."""


class Queries(BaseModel):
    queries: list[str] = Field(
        description="List of search queries.",
    )


DEFAULT_EXTRACTION_SCHEMA = {
    "type": "object",
    "required": [
        "Years-Experience",
        "Company",
        "Role",
        "Prior-Companies",
    ],
    "properties": {
        "Role": {"type": "string", "description": "Current role of the person."},
        "Years-Experience": {
            "type": "number",
            "description": "How many years of full time work experience (excluding internships) does this person have.",
        },
        "Company": {
            "type": "string",
            "description": "The name of the current company the person works at.",
        },
        "Prior-Companies": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of previous companies where the person has worked",
        },
    },
    "description": "Person information",
    "title": "Person-Schema",
}


@dataclass(kw_only=True)
class InputState:
    """Input state defines the interface between the graph and the user (external API)."""

    person: Person
    "Person to research."

    extraction_schema: dict[str, Any] = field(
        default_factory=lambda: DEFAULT_EXTRACTION_SCHEMA
    )
    "The json schema defines the information the agent is tasked with filling out."

    user_notes: Optional[dict[str, Any]] = field(default=None)
    "Any notes from the user to start the research process."


@dataclass(kw_only=True)
class OverallResearchState:
    """Input state defines the interface between the graph and the user (external API)."""

    person: Person
    "Person to research provided by the user."

    extraction_schema: dict[str, Any] = field(
        default_factory=lambda: DEFAULT_EXTRACTION_SCHEMA
    )
    "The json schema defines the information the agent is tasked with filling out."

    user_notes: str = field(default=None)
    "Any notes from the user to start the research process."

    # Add default values for required fields
    completed_notes: Annotated[list, operator.add] = field(default_factory=list)
    "Notes from completed research related to the schema"

    info: dict[str, Any] = field(default=None)
    """
    A dictionary containing the extracted and processed information
    based on the user's query and the graph's execution.
    This is the primary output of the enrichment process.
    """
    reflection: str = field(default=None)
    """
    A string that determines whether we need to reflect.
    """
    queries: list[str] = field(default=None)
    """
    A list of the search queries that were executed
    """


@dataclass(kw_only=True)
class OutputState:
    """The response object for the end user.

    This class defines the structure of the output that will be provided
    to the user after the graph's execution is complete.
    """

    info: dict[str, Any]
    """
    A dictionary containing the extracted and processed information
    based on the user's query and the graph's execution.
    This is the primary output of the enrichment process.
    """


# -----------------------------------------------------------------------------
# Prompts

extraction_prompt = """Your task is to take notes gather from web research

and extract them into the following schema. 

<schema>
{info}
</schema>

Here are all the notes from research:

<Web research notes>
{notes}
<Web research notes>
"""

query_writer_instructions = """You are a search query generator tasked with creating targeted search queries to gather specific information about people.

Here are the people you are researching: {people}

Generate at most {max_search_queries} search queries that will help gather the following information:

<schema>
{info}
</schema>

<User notes>
{user_notes}
</User notes>

Your query should:
1. Make sure to look up the right name
2. Use context clues as to the company the person works at (if it isn't concretely provided)
3. Do not hallucinate search terms that will make you miss the persons profile entirely
4. Take advantage of the Linkedin URL if it exists, you can include the raw URL in your search query as that will lead you to the correct page guaranteed.

Create a focused query that will maximize the chances of finding schema-relevant information about the person.
Remember we are interested in determining their work experience mainly."""

_INFO_PROMPT = """You are doing web research on people, {people}. 

The following schema shows the type of information we're interested in:

<schema>
{info}
</schema>

You have just scraped website content. Your task is to take clear, organized notes about the people, focusing on topics relevant to our interests.

<Website contents>
{content}
</Website contents>

Here are any additional notes from the user:
<User notes>
{user_notes}
</User notes>

Please provide detailed research notes that:
1. Are well-organized and easy to read
2. Focus on topics mentioned in the schema
3. Include specific facts, dates, and figures when available
4. Maintain accuracy of the original content
5. Note when important information appears to be missing or unclear

Remember: Don't try to format the output to match the schema - just take clear notes that capture all relevant information."""

# -----------------------------------------------------------------------------
# Nodes


async def research_people(state: OverallResearchState, config: RunnableConfig) -> str:
    """Execute a multi-step web search and information extraction process.

    This function performs the following steps:
    1. Generates multiple search queries based on the input query
    2. Executes concurrent web searches using the Tavily API
    3. Deduplicates and formats the search results
    4. Extracts structured information based on the provided schema

    Args:
        query: The initial search query string
        state: Injected application state containing the extraction schema
        config: Runtime configuration for the search process

    Returns:
        str: Structured notes from the search results that are
         relevant to the extraction schema in state.extraction_schema

    Note:
        The function uses concurrent execution for multiple search queries to improve
        performance and combines results from various sources for comprehensive coverage.
    """

    # Get configuration
    configurable = configuration.Configuration.from_runnable_config(config)
    max_search_queries = configurable.max_search_queries
    max_search_results = configurable.max_search_results

    # Generate search queries
    structured_llm = claude_3_5_sonnet.with_structured_output(Queries)

    # Format system instructions
    people_str = f"Email: {state.person['email']}"
    if "name" in state.person:
        people_str += f" Name: {state.person['name']}"
    if "linkedin" in state.person:
        people_str += f" LinkedIn URL: {state.person['linkedin']}"
    if "role" in state.person:
        people_str += f" Role: {state.person['role']}"
    if "company" in state.person:
        people_str += f" Company: {state.person['company']}"
    query_instructions = query_writer_instructions.format(
        people=people_str,
        info=json.dumps(state.extraction_schema, indent=2),
        user_notes=state.user_notes,
        max_search_queries=max_search_queries,
    )

    # Generate queries
    if state.reflection is not None:
        human_messages = [
            HumanMessage(
                content=f"Please generate a list of search queries related to the schema that you want to populate. Make them different to these prior queries {state.queries}"
            )
        ]
    else:
        human_messages = [
            HumanMessage(
                content=f"Please generate a list of search queries related to the schema that you want to populate."
            )
        ]
    results = structured_llm.invoke(
        [SystemMessage(content=query_instructions)] + human_messages
    )

    # Search client
    tavily_async_client = AsyncTavilyClient()

    # Web search
    query_list = [query for query in results.queries]
    search_tasks = []
    for query in query_list:
        search_tasks.append(
            tavily_async_client.search(
                query,
                days=360,
                max_results=max_search_results,
                include_raw_content=True,
                topic="general",
            )
        )

    # Execute all searches concurrently
    search_docs = await asyncio.gather(*search_tasks)

    # Deduplicate and format sources
    source_str = deduplicate_and_format_sources(
        search_docs, max_tokens_per_source=1000, include_raw_content=True
    )

    # Grab data from raw LinkedIn URL if it exists
    if "linkedin" in state.person and state.person["linkedin"] is not None:
        search_docs += WebBaseLoader(state.person["linkedin"]).load()[0].page_content

    # Generate structured notes relevant to the extraction schema
    p = _INFO_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2),
        content=source_str,
        people=state.person,
        user_notes=state.user_notes,
    )
    result = await claude_3_5_sonnet.ainvoke(p)
    return {"completed_notes": [str(result.content)], "queries": query_list}


def gather_notes_extract_schema(state: OverallResearchState) -> dict[str, Any]:
    """Gather notes from the web search and extract the schema fields."""

    # Format all notes
    notes = format_all_notes(state.completed_notes)

    # Extract schema fields
    system_prompt = extraction_prompt.format(
        info=json.dumps(state.extraction_schema, indent=2), notes=notes
    )
    structured_llm = claude_3_5_sonnet.with_structured_output(state.extraction_schema)
    result = structured_llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Produce a structured output from these notes."),
        ]
    )
    return {"info": result}


reflect_prompt = """You are an AI assistant tasked with analyzing the quality and completeness of research gathered about a person.

Review the following extracted information and assess whether additional research is needed:

<Extracted Information>
{extracted_info}
</Extracted Information>

Based on the required schema:
<Schema>
{schema}
</Schema>

Please analyze:
1. What information is missing or incomplete?
2. What information seems uncertain or needs verification?
3. Are there any contradictions in the gathered data?
4. Is the information recent and relevant?

Please respond YES if we need to redo the research, and NO if we can finish. Only return the strings YES or NO, no other strings.
"""


def reflect_node(state: OverallResearchState) -> dict[str, Any]:
    """Reflect on gathered information and determine if more research is needed."""
    # Don't reflect more than once
    if state.reflection is not None:
        return {"reflection": "NO"}
    system_prompt = reflect_prompt.format(
        extracted_info=json.dumps(state.info, indent=2),
        schema=json.dumps(state.extraction_schema, indent=2),
    )

    reflection = claude_3_5_sonnet.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Analyze the research quality and completeness."),
        ]
    )

    return {
        "reflection": str(reflection.content),
    }


def decide_whether_to_research_again(state: OverallResearchState):
    """Determine whether additional research is needed based on reflection results."""
    # If confidence score is below threshold or there are missing required fields
    if state.reflection == "YES":
        return "research_people"
    else:
        return "__end__"


# Add nodes and edges
builder = StateGraph(
    OverallResearchState,
    input=InputState,
    output=OutputState,
    config_schema=configuration.Configuration,
)
builder.add_node("gather_notes_extract_schema", gather_notes_extract_schema)
builder.add_node("research_people", research_people)
builder.add_node("reflect", reflect_node)


builder.add_edge(START, "research_people")
builder.add_edge("research_people", "gather_notes_extract_schema")
builder.add_edge("gather_notes_extract_schema", "reflect")
builder.add_conditional_edges(
    "reflect", decide_whether_to_research_again, [END, "research_people"]
)

# Compile
graph = builder.compile()

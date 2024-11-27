from langsmith import Client, evaluate
from Levenshtein import ratio
from langgraph.pregel.remote import RemoteGraph
import os
from pydantic import BaseModel
from typing import Optional

# Defaults
EXPERIMENT_PREFIX = "People mAIstro "
TOLERANCE = 0.15  # should match within 15%
NUMERIC_FIELDS = ("Years-Experience")
FUZZY_MATCH_FIELDS = ("Role","Company")
LIST_OF_STRING_FIELDS = ("Prior-Companies")

os.environ["OPENAI_API_KEY"] = "..."
os.environ["LANGSMITH_API_KEY"] = "..."
os.environ["TAVILY_API_KEY"] = "..."

client = Client()

# Load dataset
people_dataset = client.read_dataset(dataset_name="Person Researcher Dataset")

# Run some evals manually
MAIN_PROMPT = """You are a people researcher doing web research on behalf of a user. You are trying to collect this information about people:

<info>
{info}
</info>

You have access to the following tools:

- `Search`: call a search tool and get back some results
- `ScrapeWebsite`: scrape a website and get relevant notes about the given request. This will update the notes above.
- `Info`: call this when you are done and have gathered all the relevant info

Gather info for this person, company: {topic}"""


graph = RemoteGraph("people_maistro", url="https://ht-abandoned-cynic-27-d4d35e0b052a570a9c5cb83f703881f4.default.us.langgraph.app", api_key=None)

extraction_schema = {
    "type": "object",
    "required": [
      "Years-Experience",
      "Company",
      "Role", 
      "Prior-Companies",
    ],
    "properties": {
      "Role": {
        "type": "string",
        "description": "Current role of the person."
      },
      "Years-Experience": {
        "type": "number",
        "description": "How many years of full time work experience (excluding internships) does this person have."
      },
      "Company": {
        "type": "string",
        "description": "The name of the current company the person works at."
      },
      "Prior-Companies": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "List of previous companies where the person has worked"
      }
    },
    "description": "Person information",
    "title": "Person-Schema",
}

DEFAULT_CONFIG = {
    "configurable": {
        "prompt": MAIN_PROMPT,
        "max_loops": 3
    }
}

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

def run_agent(inputs, config=None):
    agent_inputs = {"person": Person(name=inputs["Person"],email=inputs["Work-Email"],linkedin=inputs['Linkedin']),
                    "extraction_schema": extraction_schema}
    response = graph.invoke(agent_inputs, config or DEFAULT_CONFIG)
    return {"info": response["extracted_information"]}

def evaluate_list_of_string_fields(outputs: dict, reference_outputs: dict):
    field_to_score = {}
    for k in LIST_OF_STRING_FIELDS:
        if k not in reference_outputs:
            continue
            
        output_list = outputs.get(k, [])
        reference_list = reference_outputs[k].split(", ")
        
        # Convert to lists if needed
        if isinstance(output_list, str):
            output_list = [output_list]
        if isinstance(reference_list, str):
            reference_list = [reference_list]
            
        # Convert to lowercase
        output_list = [str(item).lower() for item in output_list]
        reference_list = [str(item).lower() for item in reference_list]
        
        if not output_list or not reference_list:
            score = 0.0
        else:
            # For each reference item, find the best ratio match in output
            scores = []
            for ref_item in reference_list:
                best_ratio = max(ratio(ref_item, out_item) for out_item in output_list)
                scores.append(best_ratio)
            
            # Average the ratios
            score = sum(scores) / len(scores)
            
        field_to_score[k] = score
    return field_to_score

def evaluate_numeric_fields(outputs: dict, reference_outputs: dict):
    lower_bound = 1 - TOLERANCE
    upper_bound = 1 + TOLERANCE
    field_to_score = {}
    for k in NUMERIC_FIELDS:
        if k not in reference_outputs:
            continue

        raw_field_value = outputs.get(k, 0)
        try:
            score = float(lower_bound <= int(raw_field_value) / reference_outputs[k] <= upper_bound)
        except ValueError:
            score = 0.0

        field_to_score[k] = score
    return field_to_score

def evaluate_fuzzy_match_fields(outputs: dict, reference_outputs: dict):
    return {
        k: ratio(outputs.get(k, "").lower(), reference_outputs[k].lower())
        for k in FUZZY_MATCH_FIELDS
        if k in reference_outputs
    }

# effectively fraction of matching fields 
def evaluate_agent(outputs: dict, reference_outputs: dict):    
    if "info" not in outputs or not isinstance(outputs["info"], dict):
        return 0.0

    actual_person_info = outputs["info"]
    expected_person_info = reference_outputs

    results = {
        **evaluate_numeric_fields(actual_person_info, expected_person_info),
        **evaluate_fuzzy_match_fields(actual_person_info, expected_person_info),
        **evaluate_list_of_string_fields(actual_person_info, expected_person_info)
    }
    return sum(results.values()) / len(results)

def run_evals(
    *,
    max_concurrency: int = 2,
    experiment_prefix: str = EXPERIMENT_PREFIX
):
    res = evaluate(
        run_agent,
        data=people_dataset,
        evaluators=[evaluate_agent],
        experiment_prefix=experiment_prefix,
        max_concurrency=max_concurrency,
        blocking=True
    )

    return res

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=2,
        help="Maximum number of concurrent runs during evaluation",
    )
    parser.add_argument(
        "--experiment-prefix",
        type=str,
        default=EXPERIMENT_PREFIX,
        help="Experiment prefix for the evaluation",
    )
    args = parser.parse_args()

    run_evals(max_concurrency=args.max_concurrency, experiment_prefix=args.experiment_prefix)
    
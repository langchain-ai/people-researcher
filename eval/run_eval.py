import argparse
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langsmith import Client, evaluate
from langsmith.evaluation import EvaluationResults
from pydantic import BaseModel, Field

from langgraph.pregel.remote import RemoteGraph

# Defaults
EXPERIMENT_PREFIX = "People mAIstro "
NUMERIC_FIELDS = ("years_experience",)
FUZZY_MATCH_FIELDS = ("role", "current_company",)
LIST_FIELDS = ("prior_companies",)
TOTAL_FIELDS = len(NUMERIC_FIELDS + FUZZY_MATCH_FIELDS + LIST_FIELDS)

DEFAULT_DATASET_NAME = "People Data Enrichment"
DEFAULT_GRAPH_ID = "people_researcher"
DEFAULT_AGENT_URL = "http://localhost:2024"

client = Client()

extraction_schema = {
    "type": "object",
    "required": [
        "years_experience",
        "current_company",
        "role",
        "prior_companies",
    ],
    "properties": {
        "role": {"type": "string", "description": "Current role of the person."},
        "years_experience": {
            "type": "number",
            "description": "How many years of full time work experience (excluding internships) does this person have.",
        },
        "current_company": {
            "type": "string",
            "description": "The name of the current company the person works at.",
        },
        "prior_companies": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of previous companies where the person has worked",
        },
    },
    "description": "Person information",
    "title": "Person",
}


judge_llm = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0)

EVALUATION_PROMPT = f"""You are an evaluator tasked with assessing the accuracy of an agent's output compared to the expected output. Follow these instructions:

1. **Numeric Fields Evaluation**: For fields {NUMERIC_FIELDS}, check if the agent's output is within 10% of the expected value. Score 1 if yes, 0 if no.
2. **Fuzzy Match Evaluation**: For fields {FUZZY_MATCH_FIELDS}, check if the agent's output matches the expected output APPROXIMATELY. Score 1 if yes, 0 if no.
3. **List Fields Evaluation**: For fields {LIST_FIELDS}, check if at least one item in the agent's output overlaps with the expected output. Score 1 if yes, 0 if no.
4. **Overall Evaluation**: Return final score = number of fields with score of 1 / total number of fields ({TOTAL_FIELDS}). For example, if 1/{TOTAL_FIELDS} fields has score of 1, the final score is {1/TOTAL_FIELDS:.2f}."""


def evaluate_agent(outputs: dict, reference_outputs: dict):
    if "info" not in outputs:
        raise ValueError("Agent output must contain 'info' key")

    class Score(BaseModel):
        """Evaluate the agent's output against the expected output."""

        score: float = Field(
            description="A score between 0 and 1 indicating the accuracy of the agent's output compared to the expected output. 1 is a perfect match."
        )
        reason: str = Field(
            description="A brief explanation for why you scored the agent's output as you did."
        )

    score = judge_llm.with_structured_output(Score).invoke(
        [
            {
                "role": "system",
                "content": EVALUATION_PROMPT,
            },
            {
                "role": "user",
                "content": f'Actual output: {outputs["info"]}\nExpected output: {reference_outputs}',
            },
        ]
    )
    return score.score


def transform_dataset_inputs(inputs: dict) -> dict:
    """Transform LangSmith dataset inputs to match the agent's input schema before invoking the agent."""
    # see the `Example input` in the README for reference on what `inputs` dict should look like
    return {
        "person": {
            "name": inputs["name"],
            "email": inputs["work_email"],
            "linkedin": inputs["linkedin_profile"],
        },
        "extraction_schema": extraction_schema,
    }


def transform_agent_outputs(outputs: dict) -> dict:
    """Transform agent outputs to match the LangSmith dataset output schema."""
    # see the `Example output` in the README for reference on what the output should look like
    return {"info": outputs["info"]}


def make_agent_runner(graph_id: str, agent_url: str):
    """Wrapper that transforms inputs/outputs to match the expected eval schema and invokes the agent."""
    agent_graph = RemoteGraph(graph_id, url=agent_url)

    def run_agent(inputs: dict) -> dict:
        """Run the agent on the inputs from the LangSmith dataset record, return outputs conforming to the LangSmith dataset output schema."""
        transformed_inputs = transform_dataset_inputs(inputs)
        response = agent_graph.invoke(transformed_inputs)
        return transform_agent_outputs(response)

    return run_agent


def run_eval(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    graph_id: str = DEFAULT_GRAPH_ID,
    agent_url: str = DEFAULT_AGENT_URL,
    experiment_prefix: Optional[str] = None,
) -> EvaluationResults:
    dataset = client.read_dataset(dataset_name=dataset_name)
    run_agent = make_agent_runner(graph_id, agent_url)
    results = evaluate(
        run_agent,
        data=dataset,
        evaluators=[evaluate_agent],
        experiment_prefix=experiment_prefix,
    )
    return results


# Update main block
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="Name of the dataset to evaluate against",
    )
    parser.add_argument(
        "--graph-id",
        type=str,
        default=DEFAULT_GRAPH_ID,
        help="ID of the graph to evaluate",
    )
    parser.add_argument(
        "--agent-url",
        type=str,
        default=DEFAULT_AGENT_URL,
        help="URL of the deployed agent to evaluate",
    )
    parser.add_argument(
        "--experiment-prefix",
        type=str,
        help="Experiment prefix for the evaluation",
    )
    args = parser.parse_args()

    run_eval(
        dataset_name=args.dataset_name,
        graph_id=args.graph_id,
        agent_url=args.agent_url,
        experiment_prefix=args.experiment_prefix,
    )

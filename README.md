# People mAIstro

People mAIstro researches information about a user-supplied list of people, and returns it in any user-defined schema.

## Overview

 People mAIstro follows a [plan-and-execute workflow](https://github.com/assafelovic/gpt-researcher) that separates planning from research, allowing for better resource management and significantly reducing overall research time:

   - **Planning Phase**: An LLM analyzes the user's set of people to research and returns a list. 
   - **Research Phase**: The system parallelizes web research across all people in parallel:
     - Uses [Tavily API](https://tavily.com/) for targeted web searches, performing up to `max_search_queries` queries per person.
     - Performs web searches for each person in parallel and returns up to `max_search_results` results per query.
   - **Extract Schema**: After research is complete, the system uses an LLM to extract the information from the research in the user-defined schema.


## Inputs 

The user inputs are: 

```
* people: People - A person to research (People are dictionaries with email (required), name (optional), company (optional), LinkedIn URL (optional))
* schema: str - A JSON schema for the output
* user_notes: Optional[str] - Any additional notes about the people from the user
```

Here is an example schema that can be supplied: 

```
{
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
```


## Dataset

There is one dataset for public evaluation in LangSmith:

- [People Dataset](https://smith.langchain.com/public/2af89d2a-93f6-4c84-80ac-70defcfd14c8/d). This dataset has a list of people to extract the following fields for:
  - `Years-Experience`
  - `Company`
  - `Role`
  - `Prior-Companies`


### Running evals

To evaluate the People mAIstro agent, you can run the `run_eval.py` script. This will create new experiments in LangSmith for the [dataset](#dataset) mentioned above.

**Basic usage:**

```shell
python evals/test_agent.py
```

By default this will use the `Person Researcher Dataset` dataset & `People mAIstro` agent by LangChain.

**Advanced usage:**

You can pass the following parameters to customize the evaluation:

- `--dataset-name`: Name of the dataset to evaluate against. Defaults to `Person Researcher Dataset` dataset.
- `--agent-id`: ID of the agent to evaluate. Defaults to `people_maistro`.
- `--agent-url`: URL of the deployed agent to evaluate. Defaults to `People mAIstro` deployment.
- `--experiment-prefix`: Prefix for the experiment name.
- `--min-score`: Minimum acceptable score for evaluation. If specified, the script will raise an assertion error if the average score is below this threshold.

```shell
python evals/test_agent.py --experiment-prefix "My custom prefix" --min-score 0.9
```

### Using different schema

#### Different Extraction Schema

If you want to use a different extraction schema, you can modify the `extraction_schema` variable in `evals/test_agent.py` to match whatever extraction schema you are looking for.

#### Different Agent Schema

If your agent uses a schema that's different from the [example one above](#agent-schema), you can modify `make_agent_runner` in `evals/test_agent.py` in the following way:

```python
def make_agent_runner(agent_id: str, agent_url: str):
    agent_graph = RemoteGraph(agent_id, url=agent_url)

    def run_agent(inputs: dict):
        # transform the inputs (single LangSmith dataset record) to match the agent's schema
        transformed_inputs = {"my_agent_key": inputs["Person"], ...}
        response = agent_graph.invoke(transformed_inputs)
        # transform the agent outputs to match expected eval schema
        transformed_outputs = {"info": response["my_agent_output_key"]}
        return transformed_outputs

    return run_agent
```
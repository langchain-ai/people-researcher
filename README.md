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
* people: List[str] - A list of people to research
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

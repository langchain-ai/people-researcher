# Company mAIstro

Company mAIstro researches information about a user-supplied list of people, and returns it in any user-defined schema.

## Quickstart

1. Populate the `.env` file: 
```
$ cp .env.example .env
```

2. Load this folder in [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio?tab=readme-ov-file#download) 

3. Provide a schema for the output, and pass in a list of people. 

4. Run the graph!

![Screenshot 2024-11-26 at 2 33 07 PM](https://github.com/user-attachments/assets/7b52ac0a-fe4a-414c-8936-9a2d8abaea46)


## Overview

 Company mAIstro follows a [plan-and-execute workflow](https://github.com/assafelovic/gpt-researcher) that separates planning from research, allowing for better resource management and significantly reducing overall research time:

   - **Planning Phase**: An LLM analyzes the user's set of people to research and returns a list. 
   - **Research Phase**: The system parallelizes web research across all people in parallel:
     - Uses [Tavily API](https://tavily.com/) for targeted web searches, performing up to `max_search_queries` queries per person.
     - Performs web searches for each person in parallel and returns up to `max_search_results` results per query.
   - **Extract Schema**: After research is complete, the system uses an LLM to extract the information from the research in the user-defined schema.

## Configuration

The configuration for Company mAIstro is defined in the `configuration.py` file: 
* `max_search_queries`: int = 3 # Max search queries per person
* `max_search_results`: int = 3 # Max search results per query

These can be added in Studio:

![Screenshot 2024-11-26 at 2 33 14 PM](https://github.com/user-attachments/assets/305cf2ad-a664-4cb6-99e5-e86bd024b065)

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
    "title": "people_info",
    "description": "Information about multiple people",
    "type": "array",
    "items": {
    "type": "object",
    "required": [
      "years_experience",
      "role",
      "linkedin",
      "skills",
      "prior_companies",
      "instagram",
      "twitter",
      "work_email"
    ],
    "properties": {
      "role": {
        "type": "string",
        "description": "Current role of the person."
      },
      "years_experience": {
        "type": "number",
        "description": "How many years of full time work experience (excluding internships) does this person have."
      },
      "linkedin": {
        "type": "string",
        "description": "URL for the LinkedIn page of the person. N/A if you cannot find it."
      },
      "instagram": {
        "type": "string",
        "description": "URL for the Instagram page of the person. N/A if you cannot find it."
      },
      "twitter": {
        "type": "string",
        "description": "URL for the Twitter page of the person. N/A if you cannot find it."
      },
      "work_email": {
        "type": "string",
        "description": "Work email address of the person. N/A if you cannot find it."
      },
      "prior_companies": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "List of previous companies where the person has worked"
      },
      "skills": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "List of skills the person has such as Marketing, Software Engineering, etc. Do not put specific things like Javascript or Google Ads."
      }
    },
    "description": "Person information"
  }
}
```

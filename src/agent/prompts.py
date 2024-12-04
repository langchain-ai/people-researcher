EXTRACTION_PROMPT = """Your task is to take notes gathered from web research and extract them into the following schema.

<schema>
{info}
</schema>

Here are all the notes from research:

<web_research_notes>
{notes}
</web_research_notes>
"""

QUERY_WRITER_PROMPT = """You are a search query generator tasked with creating targeted search queries to gather specific information about a person.

Here is the person you are researching: {person}

Generate at most {max_search_queries} search queries that will help gather the following information:

<schema>
{info}
</schema>

<user_notes>
{user_notes}
</user_notes>

Your query should:
1. Make sure to look up the right name
2. Use context clues as to the company the person works at (if it isn't concretely provided)
3. Do not hallucinate search terms that will make you miss the persons profile entirely
4. Take advantage of the Linkedin URL if it exists, you can include the raw URL in your search query as that will lead you to the correct page guaranteed.

Create a focused query that will maximize the chances of finding schema-relevant information about the person.
Remember we are interested in determining their work experience mainly."""

INFO_PROMPT = """You are doing web research on people, {people}. 

The following schema shows the type of information we're interested in:

<schema>
{info}
</schema>

You have just scraped website content. Your task is to take clear, organized notes about a person, focusing on topics relevant to our interests.

<Website contents>
{content}
</Website contents>

Here are any additional notes from the user:
<user_notes>
{user_notes}
</user_notes>

Please provide detailed research notes that:
1. Are well-organized and easy to read
2. Focus on topics mentioned in the schema
3. Include specific facts, dates, and figures when available
4. Maintain accuracy of the original content
5. Note when important information appears to be missing or unclear

Remember: Don't try to format the output to match the schema - just take clear notes that capture all relevant information."""


REFLECTION_PROMPT = """You are a research analyst tasked with reviewing the quality and completeness of extracted person information.

Compare the extracted information with the required schema:

<Schema>
{schema}
</Schema>

Here is the extracted information:
<extracted_info>
{info}
</extracted_info>

Analyze if all required fields are present and sufficiently populated. Consider:
1. Are any required fields missing?
2. Are any fields incomplete or containing uncertain information?
3. Are there fields with placeholder values or "unknown" markers?
"""

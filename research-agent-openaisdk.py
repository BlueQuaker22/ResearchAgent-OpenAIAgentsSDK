#!pip install ddgs trafilatura
#!pip install lxml_html_clean
#!pip install openai-agents

import os
import json
import asyncio
from dotenv import load_dotenv 
from pprint import pprint
from IPython.display import Markdown, display

from ddgs import DDGS
import trafilatura

from agents import Agent, Runner, function_tool

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise Exception("API Key is missing")

Model = "gpt-4.1-mini"

@function_tool
def search_web_ddg(query:str): #def search_web(query:str, num_results:int):
    """Search the web using the DuckDuckGo browser, return results"""
    ddgs = DDGS()
    results = ddgs.text(query, max_results = 3) #results = ddgs.text(query, max_results = num_results)
    print(f"  \u2705 Got results")
    return json.dumps(results, indent=2)

@function_tool
def fetch_url(url:str):
    """Fetch the content of URL using Trafilatura tool"""
    #url = "https://finance.yahoo.com/news/qualcomm-weighs-ai-export-rules-150751298.html"
    download_text= trafilatura.fetch_url(url)
    #print(download_text)

    if download_text:
        extract_text = trafilatura.extract(download_text)
        if extract_text:
            print(f". \u2705 Got text:{len(extract_text)} chars")
            return extract_text
    print(f". \u274c Failed to fetch or extract text from {url}")
    return (f"Could not extract text from {url}. Try a different source")

RESEARCH_AGENT_SYSTEM_PROMPT = """
You are a research specialist that follows all instructions excels at searching the web for news on a given topic and summarizing your findings into a 
comprehensive research brief.

You MUST gather information from at least 6 distinct sources before delivering your brief. 
If you have fewer than 6 sources, keep searching. You need to pause and reflect on the results after you get the information 
from 3 sources so that you can identify the best next 3 sources to retrieve.

VERY IMPORTANT: ALWAYS look for recent news when possible. DO NOT use older news articles if newer ones from reputable sources exist.
Prioritize sources such as press releases or articles directly from the company and organization that is being searched if relevant. 
Also prioritize sources from reputable news sources. 

You have access to (1) a search_web tool that you can use to search the web for information and 
(2) a fetch_url tool to extract and read the text from the URLs of the webpages in the search results.

Your typical process:
1. Search for the topic to find the best sources to fetch
2. Reflect on the search results — which sources look most relevant and why?
3. Fetch the full content of the best URLs based on the max number of results requested
4. Reflect on what you have gathered. Do you have enough? Are there gaps? Are you unable to access or fetch text from any of the URLs?
5. If there are gaps, search again with a different query that captures the intent of the user
6. When you have enough information from at least 3 different sources, synthesize into a research brief

Your research brief MUST include:
- Main themes, headlines, and important text from the sources
- Key facts, data points, and statistics
- Summary
- Source URLs for attribution and dates of the content if from news sources

Until you are ready, just keep working — search, fetch, think, reflect.
Do not rush. Take time to reflect between tool calls before deciding your next step.
Not every response needs a tool call — sometimes just thinking through what you have is the right move. 

"""

research_agent = Agent(
    name = "Research Agent",
    model = Model,
    tools = [search_web_ddg, fetch_url],
    instructions=RESEARCH_AGENT_SYSTEM_PROMPT
)

async def main():
    research_brief = await Runner.run(
        research_agent,
        input="Conduct research and produce a comprehensive brief on the following topic: Nvidia GTC 2026",
        max_turns=20,
    )
    print(research_brief.final_output)


if __name__ == "__main__":
    asyncio.run(main())
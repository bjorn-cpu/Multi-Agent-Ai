"""
Research Agent - Agentic AI with tool-use loop
Uses Google Gemini to autonomously search, fetch, and summarize
"""

import json
import urllib.request
import urllib.parse
from datetime import datetime
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
load_dotenv()

# =========================================================
# Gemini Client
# =========================================================

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)


# =========================================================
# Tool Definitions
# =========================================================

TOOLS = [
    {
        "name": "web_search",
        "description": (
            "Search Wikipedia for information on a topic. "
            "Returns titles, URLs, and snippets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "fetch_page",
        "description": (
            "Fetch and read content from a URL."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch"
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "summarize_findings",
        "description": (
            "Save structured research output."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string"
                },
                "key_points": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string"
                            },
                            "url": {
                                "type": "string"
                            }
                        }
                    }
                },
                "summary": {
                    "type": "string"
                }
            },
            "required": [
                "title",
                "key_points",
                "sources",
                "summary"
            ]
        }
    }
]


# =========================================================
# Tool Implementations
# =========================================================

def web_search(query: str) -> dict:
    """
    Search Wikipedia
    """

    try:

        search_url = (
            "https://en.wikipedia.org/w/api.php?"
            "action=query"
            "&list=search"
            f"&srsearch={urllib.parse.quote(query)}"
            "&format=json"
            "&srlimit=5"
        )

        req = urllib.request.Request(
            search_url,
            headers={
                "User-Agent": "ResearchAgent/1.0"
            }
        )

        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())

        hits = data.get("query", {}).get("search", [])

        results = []

        for hit in hits:

            title = hit.get("title", "")

            snippet = (
                hit.get("snippet", "")
                .replace('<span class="searchmatch">', "")
                .replace("</span>", "")
            )

            url = (
                "https://en.wikipedia.org/wiki/"
                f"{urllib.parse.quote(title.replace(' ', '_'))}"
            )

            results.append({
                "title": title,
                "url": url,
                "snippet": snippet
            })

        return {
            "query": query,
            "results": results
        }

    except Exception as e:

        return {
            "error": str(e),
            "results": []
        }


def fetch_page(url: str) -> dict:
    """
    Fetch Wikipedia article content
    """

    try:

        if "wikipedia.org/wiki/" in url:

            title = url.split("/wiki/")[-1]

            api_url = (
                "https://en.wikipedia.org/w/api.php?"
                "action=query"
                "&prop=extracts"
                f"&titles={title}"
                "&format=json"
                "&explaintext=true"
                "&exintro=true"
            )

            req = urllib.request.Request(
                api_url,
                headers={
                    "User-Agent": "ResearchAgent/1.0"
                }
            )

            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            pages = data.get("query", {}).get("pages", {})
            page = next(iter(pages.values()))

            text = page.get(
                "extract",
                "No content found."
            )

            return {
                "url": url,
                "content": text[:3000],
                "length": len(text)
            }

        return {
            "url": url,
            "content": "Non-Wikipedia URLs are not supported yet."
        }

    except Exception as e:

        return {
            "error": str(e),
            "url": url
        }


def summarize_findings(
    title: str,
    key_points: list,
    sources: list,
    summary: str
) -> dict:
    """
    Store final structured output
    """

    return {
        "status": "saved",
        "title": title,
        "key_points": key_points,
        "sources": sources,
        "summary": summary,
        "timestamp": datetime.now().isoformat()
    }


# =========================================================
# Tool Dispatcher
# =========================================================

def execute_tool(
    tool_name: str,
    tool_input: dict
) -> dict:

    if tool_name == "web_search":

        return web_search(
            tool_input["query"]
        )

    elif tool_name == "fetch_page":

        return fetch_page(
            tool_input["url"]
        )

    elif tool_name == "summarize_findings":

        return summarize_findings(
            tool_input["title"],
            tool_input["key_points"],
            tool_input["sources"],
            tool_input["summary"]
        )

    return {
        "error": f"Unknown tool: {tool_name}"
    }


# =========================================================
# Core Agent Loop
# =========================================================

def run_research_agent(
    query: str,
    on_step=None
):

    system_prompt = """
You are an expert autonomous research agent.

Workflow:
1. Search for relevant information
2. Fetch article content
3. Analyze information
4. Summarize findings
5. Provide final answer

Guidelines:
- Use multiple searches when useful
- Fetch the most relevant pages
- Be concise but informative
- Always cite sources
- Use summarize_findings before final response
"""

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[
            types.Tool(
                function_declarations=TOOLS
            )
        ]
    )

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(
                    text=query
                )
            ]
        )
    ]

    steps = []

    final_result = None

    max_iterations = 10

    for iteration in range(max_iterations):

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=config
        )

        candidate = response.candidates[0]

        content = candidate.content

        contents.append(content)

        tool_calls = []

        text_parts = []

        for part in content.parts:

            if part.function_call:
                tool_calls.append(
                    part.function_call
                )

            elif part.text:
                text_parts.append(
                    part.text
                )

        step_info = {
            "iteration": iteration + 1,
            "text": " ".join(text_parts),
            "tools_called": []
        }

        # =================================================
        # TOOL CALLS
        # =================================================

        if tool_calls:

            tool_response_parts = []

            for fn_call in tool_calls:

                tool_name = fn_call.name

                tool_args = dict(fn_call.args)

                result = execute_tool(
                    tool_name,
                    tool_args
                )

                step_info["tools_called"].append({
                    "name": tool_name,
                    "input": tool_args,
                    "result_preview": json.dumps(result)[:200]
                })

                if tool_name == "summarize_findings":
                    final_result = result

                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={
                            "result": result
                        }
                    )
                )

            steps.append(step_info)

            if on_step:
                on_step(step_info)

            contents.append(
                types.Content(
                    role="tool",
                    parts=tool_response_parts
                )
            )

        # =================================================
        # FINAL RESPONSE
        # =================================================

        else:

            steps.append(step_info)

            if on_step:
                on_step(step_info)

            return {
                "success": True,
                "query": query,
                "steps": steps,
                "final_answer": step_info["text"],
                "structured": final_result,
                "total_iterations": iteration + 1
            }

    return {
        "success": False,
        "query": query,
        "steps": steps,
        "final_answer": (
            "Research stopped after reaching "
            "maximum iterations."
        ),
        "structured": final_result,
        "total_iterations": max_iterations
    }


# =========================================================
# Direct Run
# =========================================================

if __name__ == "__main__":

    result = run_research_agent(
        "What is retrieval augmented generation?"
    )

    print(json.dumps(result, indent=2))
import json
import urllib.request
import urllib.parse
from groq import Groq
from agents.utils import generate_with_retry
from agents.tavily_search import tavily_search, fetch_page as tavily_fetch
import os

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "qwen/qwen3.6-27b"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "wiki_search",
            "description": "Search the web for a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wiki_fetch",
            "description": "Fetch the full content of a URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_wiki_answer",
            "description": "Save a deep-dive structured answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "overview": {"type": "string"},
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "heading": {"type": "string"},
                                "content": {"type": "string"}
                            }
                        }
                    },
                    "fun_facts": {"type": "array", "items": {"type": "string"}},
                    "sources": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "url": {"type": "string"}
                            }
                        }
                    }
                },
                "required": ["topic", "overview", "sections", "fun_facts", "sources"]
            }
        }
    }
]

_history = []

def reset():
    global _history
    _history = []


def do_search(query):
    if os.environ.get("TAVILY_API_KEY"):
        result = tavily_search(query)
        if result.get("results"):
            return result
    try:
        url = (
            "https://en.wikipedia.org/w/api.php?action=query&list=search"
            f"&srsearch={urllib.parse.quote(query)}&format=json&srlimit=5"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "WikiAgent/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        hits = data.get("query", {}).get("search", [])
        results = []
        for hit in hits:
            title = hit.get("title", "")
            snippet = (hit.get("snippet", "")
                .replace('<span class="searchmatch">', "")
                .replace("</span>", ""))
            wiki_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
            results.append({"title": title, "url": wiki_url, "snippet": snippet})
        return {"results": results}
    except Exception as e:
        return {"error": str(e), "results": []}


def do_fetch(url):
    if os.environ.get("TAVILY_API_KEY"):
        return tavily_fetch(url)
    try:
        if "wikipedia.org/wiki/" in url:
            title = url.split("/wiki/")[-1]
            api_url = (
                f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
                f"&exintro=false&titles={title}&format=json&explaintext=true"
            )
            req = urllib.request.Request(api_url, headers={"User-Agent": "WikiAgent/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            pages = data.get("query", {}).get("pages", {})
            page = next(iter(pages.values()))
            text = page.get("extract", "No content found.")
            return {"url": url, "content": text[:4000], "length": len(text)}
        return {"error": "Cannot fetch non-Wikipedia URLs without Tavily", "url": url}
    except Exception as e:
        return {"error": str(e), "url": url}


def execute_tool(tool_name, tool_input):
    if tool_name == "wiki_search":
        return do_search(tool_input["query"])
    elif tool_name == "wiki_fetch":
        return do_fetch(tool_input["url"])
    elif tool_name == "save_wiki_answer":
        return {"status": "saved", **tool_input}
    return {"error": f"Unknown tool: {tool_name}"}


def run_agent(query, on_step=None):
    global _history

    system_prompt = """You are a deep-dive research expert with memory of the full conversation.
When given a topic:
1. Use wiki_search to find the most relevant sources (2-3 searches)
2. Use wiki_fetch to read them in depth
3. Call save_wiki_answer with rich structured breakdown including sections and fun facts
4. Give an engaging final response
For follow-ups use context from earlier. Go deep."""

    _history.append({"role": "user", "content": query})
    steps = []
    final_result = None

    for iteration in range(10):
        messages = [{"role": "system", "content": system_prompt}] + _history

        response = generate_with_retry(lambda: client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=2000
        ))

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        step_info = {
            "iteration": iteration + 1,
            "stop_reason": finish_reason,
            "text": msg.content or "",
            "tools_called": []
        }

        if msg.tool_calls:
            _history.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in msg.tool_calls
                ]
            })

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_input = json.loads(tc.function.arguments)
                except:
                    tool_input = {}

                result = execute_tool(tool_name, tool_input)

                step_info["tools_called"].append({
                    "name": tool_name,
                    "input": tool_input,
                    "result_preview": json.dumps(result)[:200]
                })

                if tool_name == "save_wiki_answer":
                    final_result = result

                _history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result)
                })

            steps.append(step_info)
            if on_step: on_step(step_info)

        else:
            _history.append({"role": "assistant", "content": msg.content or ""})
            steps.append(step_info)
            if on_step: on_step(step_info)
            return {
                "success": True, "query": query, "steps": steps,
                "final_answer": msg.content or "",
                "structured": final_result, "total_iterations": iteration + 1
            }

    return {
        "success": False, "query": query, "steps": steps,
        "final_answer": "Hit max iterations.",
        "structured": final_result, "total_iterations": 10
    }
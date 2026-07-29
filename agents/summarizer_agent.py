import json
from groq import Groq
from agents.utils import generate_with_retry
import os

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "qwen/qwen3.6-27b"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "save_summary",
            "description": "Save the structured summary of the provided text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "one_liner": {"type": "string"},
                    "key_points": {"type": "array", "items": {"type": "string"}},
                    "conclusion": {"type": "string"}
                },
                "required": ["title", "one_liner", "key_points", "conclusion"]
            }
        }
    }
]

_history = []

def reset():
    global _history
    _history = []


def execute_tool(tool_name, tool_input):
    if tool_name == "save_summary":
        return {
            "status": "saved",
            "title": tool_input["title"],
            "one_liner": tool_input["one_liner"],
            "key_points": tool_input["key_points"],
            "conclusion": tool_input["conclusion"]
        }
    return {"error": f"Unknown tool: {tool_name}"}


def run_agent(query, on_step=None):
    global _history

    system_prompt = """You are an expert summarizer with memory of the full conversation.
- Summarize any text given to you using save_summary
- For follow-ups like 'expand on point 3' use earlier context
- Be concise and capture the most important information"""

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

                if tool_name == "save_summary":
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
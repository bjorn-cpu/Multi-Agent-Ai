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
            "name": "analyze_code",
            "description": "Analyze a piece of code and return structured findings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {"type": "string"},
                    "issues": {"type": "array", "items": {"type": "string"}},
                    "suggestions": {"type": "array", "items": {"type": "string"}},
                    "explanation": {"type": "string"}
                },
                "required": ["language", "issues", "suggestions", "explanation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_code",
            "description": "Write a code solution for a given problem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {"type": "string"},
                    "code": {"type": "string"},
                    "explanation": {"type": "string"}
                },
                "required": ["language", "code", "explanation"]
            }
        }
    }
]

_history = []

def reset():
    global _history
    _history = []


def execute_tool(tool_name, tool_input):
    if tool_name == "analyze_code":
        return {
            "status": "analyzed",
            "language": tool_input["language"],
            "issues": tool_input["issues"],
            "suggestions": tool_input["suggestions"],
            "explanation": tool_input["explanation"]
        }
    elif tool_name == "write_code":
        return {
            "status": "written",
            "language": tool_input["language"],
            "code": tool_input["code"],
            "explanation": tool_input["explanation"]
        }
    return {"error": f"Unknown tool: {tool_name}"}


def run_agent(query, on_step=None):
    global _history

    system_prompt = """You are an expert coding assistant with memory of the full conversation.
- If the user shares code to review: use analyze_code
- If the user wants code written: use write_code
- For follow-ups like 'now do it in JavaScript' use context from earlier
- Always explain in simple terms
- Support all programming languages"""

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

                if tool_name in ("analyze_code", "write_code"):
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
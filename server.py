"""
Flask backend - streams agent steps to the browser in real-time via SSE
"""

import json
import queue
import threading
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from agent import run_research_agent

app = Flask(__name__, static_folder="static")
CORS(app)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/research", methods=["POST"])
def research():
    """Standard JSON endpoint - returns full result when done"""
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Query is required"}), 400
    result = run_research_agent(query)
    return jsonify(result)


@app.route("/api/research/stream")
def research_stream():
    """SSE endpoint - streams each agent step as it happens"""
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Query param 'q' is required"}), 400

    step_queue = queue.Queue()

    def run_agent():
        def on_step(step):
            step_queue.put({"type": "step", "data": step})
        result = run_research_agent(query, on_step=on_step)
        step_queue.put({"type": "done", "data": result})

    thread = threading.Thread(target=run_agent, daemon=True)
    thread.start()

    def generate():
        yield f"data: {json.dumps({'type': 'start', 'query': query})}\n\n"
        while True:
            try:
                event = step_queue.get(timeout=60)
                yield f"data: {json.dumps(event)}\n\n"
                if event["type"] == "done":
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'timeout'})}\n\n"
                break

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


if __name__ == "__main__":
    print("\n🔬 Research Agent running at http://localhost:5000\n")
    app.run(debug=True, port=5000, threaded=True)
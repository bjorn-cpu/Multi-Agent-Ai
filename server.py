import json
import queue
import threading
import os
import re
import unicodedata
import logging
from datetime import datetime
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

load_dotenv()

from agents import research_agent, code_agent, summarizer_agent, math_agent, wiki_agent


# ── Logging ──────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
log_file = f"logs/{datetime.now().strftime('%Y-%m-%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("multi-agent")


# ── Startup env check ─────────────────────────────────────────────────────────
def check_env():
    required = ["GROQ_API_KEY"]
    optional = ["TAVILY_API_KEY"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"\n❌ Missing required environment variables: {', '.join(missing)}")
        print("Create a .env file with your API keys.\n")
        exit(1)
    for k in optional:
        if not os.environ.get(k):
            print(f"⚠️  {k} not set — Wikipedia fallback will be used")

check_env()


# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static")
CORS(app, origins=["http://localhost:5000", "http://127.0.0.1:5000"])

app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["500 per day", "100 per hour"],
    storage_uri="memory://"
)


# ── Constants ─────────────────────────────────────────────────────────────────
MAX_QUERY_LENGTH = 8000
MAX_SAVED_CHATS_SIZE_MB = 100
ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.docx'}
ALLOWED_AGENTS = {'research', 'code', 'summarizer', 'math', 'wiki'}

AGENTS = {
    "research":   research_agent.run_agent,
    "code":       code_agent.run_agent,
    "summarizer": summarizer_agent.run_agent,
    "math":       math_agent.run_agent,
    "wiki":       wiki_agent.run_agent,
}

RESET = {
    "research":   research_agent.reset,
    "code":       code_agent.reset,
    "summarizer": summarizer_agent.reset,
    "math":       math_agent.reset,
    "wiki":       wiki_agent.reset,
}

SAVE_DIR = "saved_chats"
os.makedirs(SAVE_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_filename(name: str) -> str:
    name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
    return name[:80] if name else "chat"


def sanitize_input(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cc" or ch in "\n\t")
    return text.strip()


def check_disk_usage() -> float:
    total = 0
    for f in os.listdir(SAVE_DIR):
        fp = os.path.join(SAVE_DIR, f)
        if os.path.isfile(fp):
            total += os.path.getsize(fp)
    return total / (1024 * 1024)


def safe_filepath(filename: str) -> str | None:
    """Returns filepath only if it stays inside SAVE_DIR, else None."""
    if not re.match(r'^[a-zA-Z0-9_\-]+\.json$', filename):
        return None
    filepath = os.path.join(SAVE_DIR, filename)
    if not os.path.abspath(filepath).startswith(os.path.abspath(SAVE_DIR)):
        return None
    return filepath


# ── Security headers ──────────────────────────────────────────────────────────
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
    return response


# ── Error handlers ────────────────────────────────────────────────────────────
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Max size is 10MB."}), 413

@app.errorhandler(429)
def rate_limited(e):
    return jsonify({"error": "Too many requests. Please slow down."}), 429

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {str(e)}")
    return jsonify({"error": "Internal server error"}), 500


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/reset", methods=["POST"])
def reset():
    data = request.get_json(silent=True) or {}
    agent_id = data.get("agent", "research")
    if agent_id not in ALLOWED_AGENTS:
        return jsonify({"error": "Invalid agent"}), 400
    if agent_id in RESET:
        RESET[agent_id]()
    logger.info(f"Reset — agent={agent_id}")
    return jsonify({"status": "cleared", "agent": agent_id})


@app.route("/api/upload_file", methods=["POST"])
@limiter.limit("20 per hour")
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    filename = file.filename.lower()
    ext = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""

    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"File type not allowed. Use: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    logger.info(f"File upload — filename={file.filename} ext={ext}")

    content = ""
    try:
        if filename.endswith(".txt"):
            content = file.read().decode("utf-8", errors="replace")

        elif filename.endswith(".pdf"):
            import fitz
            data = file.read()
            doc = fitz.open(stream=data, filetype="pdf")
            pages = [page.get_text() for page in doc]
            content = "\n\n".join(pages)
            doc.close()

        elif filename.endswith(".docx"):
            try:
                import zipfile, io
                data = file.read()
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    with z.open("word/document.xml") as f:
                        xml = f.read().decode("utf-8")
                content = re.sub(r"<[^>]+>", " ", xml)
                content = re.sub(r"\s+", " ", content).strip()
            except Exception:
                content = "Could not read .docx file."

        if len(content) > 12000:
            content = content[:12000] + "\n\n[File truncated]"

        return jsonify({"content": content, "filename": file.filename, "length": len(content)})

    except Exception as e:
        logger.error(f"File processing error: {str(e)}")
        return jsonify({"error": "Could not process file"}), 500


@app.route("/api/save_chat", methods=["POST"])
def save_chat():
    data = request.get_json(silent=True) or {}
    name = safe_filename(data.get("name", "chat").replace(" ", "_"))
    messages = data.get("messages", [])
    agent = data.get("agent", "research")

    if agent not in ALLOWED_AGENTS:
        return jsonify({"error": "Invalid agent"}), 400
    if not isinstance(messages, list):
        return jsonify({"error": "Invalid messages"}), 400
    if len(messages) > 200:
        messages = messages[-200:]

    total_size = len(json.dumps(messages))
    if total_size > 5 * 1024 * 1024:
        return jsonify({"error": "Chat too large to save"}), 400

    if check_disk_usage() > MAX_SAVED_CHATS_SIZE_MB:
        return jsonify({"error": "Storage limit reached. Delete some saved chats first."}), 507

    filename = f"{name}_{agent}.json"
    filepath = safe_filepath(filename)
    if not filepath:
        return jsonify({"error": "Invalid filename"}), 400

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({"agent": agent, "name": name, "messages": messages}, f, indent=2, ensure_ascii=False)

    logger.info(f"Chat saved — {filename}")
    return jsonify({"status": "saved", "filename": filename})


@app.route("/api/list_chats", methods=["GET"])
def list_chats():
    files = []
    for f in os.listdir(SAVE_DIR):
        if f.endswith(".json"):
            path = os.path.join(SAVE_DIR, f)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    chat_data = json.load(fh)
                files.append({
                    "filename": f,
                    "name": chat_data.get("name", f),
                    "agent": chat_data.get("agent", "unknown"),
                    "message_count": len(chat_data.get("messages", []))
                })
            except Exception:
                pass
    return jsonify({"chats": sorted(files, key=lambda x: x["filename"], reverse=True)})


@app.route("/api/load_chat", methods=["POST"])
def load_chat():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename", "")
    filepath = safe_filepath(filename)
    if not filepath:
        return jsonify({"error": "Invalid filename"}), 400
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    with open(filepath, "r", encoding="utf-8") as f:
        chat_data = json.load(f)
    logger.info(f"Chat loaded — {filename}")
    return jsonify(chat_data)


@app.route("/api/delete_chat", methods=["POST"])
def delete_chat():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename", "")
    filepath = safe_filepath(filename)
    if not filepath:
        return jsonify({"error": "Invalid filename"}), 400

    if os.path.exists(filepath):
        os.remove(filepath)
        logger.info(f"Chat deleted — {filename}")
    return jsonify({"status": "deleted"})


@app.route("/api/run", methods=["POST"])
@limiter.limit("30 per minute")
def run():
    data = request.get_json(silent=True) or {}
    query = sanitize_input(data.get("query", ""))
    agent_id = data.get("agent", "research")

    if not query:
        return jsonify({"error": "Query is required"}), 400
    if len(query) > MAX_QUERY_LENGTH:
        return jsonify({"error": f"Query too long. Max {MAX_QUERY_LENGTH} characters."}), 400
    if agent_id not in AGENTS:
        return jsonify({"error": f"Unknown agent: {agent_id}"}), 400

    logger.info(f"Run — agent={agent_id} query_len={len(query)}")
    result = AGENTS[agent_id](query)
    return jsonify(result)


@app.route("/api/stream")
@limiter.limit("30 per minute")
def stream():
    query = sanitize_input(request.args.get("q", ""))
    agent_id = request.args.get("agent", "research")

    if not query:
        return jsonify({"error": "Query param 'q' is required"}), 400
    if len(query) > MAX_QUERY_LENGTH:
        return jsonify({"error": f"Query too long. Max {MAX_QUERY_LENGTH} characters."}), 400
    if agent_id not in AGENTS:
        return jsonify({"error": f"Unknown agent: {agent_id}"}), 400

    logger.info(f"Stream — agent={agent_id} query_len={len(query)}")
    step_queue = queue.Queue()

    def run_agent():
        def on_step(step):
            step_queue.put({"type": "step", "data": step})
        try:
            result = AGENTS[agent_id](query, on_step=on_step)
        except Exception as e:
            logger.error(f"Agent error — agent={agent_id} error={str(e)}")
            result = {
                "success": False,
                "final_answer": "An error occurred processing your request.",
                "structured": None, "steps": [], "total_iterations": 0
            }
        step_queue.put({"type": "done", "data": result})

    threading.Thread(target=run_agent, daemon=True).start()

    def generate():
        yield f"data: {json.dumps({'type': 'start', 'query': query, 'agent': agent_id})}\n\n"
        while True:
            try:
                event = step_queue.get(timeout=300)
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
    print("\n🤖 Multi-Agent AI running at http://localhost:5000\n")
    app.run(debug=True, port=5000, threaded=True)
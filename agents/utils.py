import time
import re

def generate_with_retry(func, max_retries=5):
    delay = 10

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            err_str = str(e)
            is_429 = "429" in err_str or "rate_limit" in err_str.lower() or "rate limit" in err_str.lower()
            is_503 = "503" in err_str or "unavailable" in err_str.lower()
            is_tool_fail = "tool_use_failed" in err_str or "Failed to call a function" in err_str

            if (is_429 or is_503) and attempt < max_retries - 1:
                wait = delay
                match = re.search(r'try again in ([\d.]+)s', err_str.lower())
                if match:
                    wait = float(match.group(1)) + 2
                wait = min(wait, 60)
                print(f"[retry] {'429 rate limit' if is_429 else '503 overload'} — waiting {wait:.0f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                delay = min(delay * 2, 60)

            elif is_tool_fail and attempt < max_retries - 1:
                # Tool call malformed — wait briefly and retry
                wait = 3
                print(f"[retry] tool_use_failed — retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)

            else:
                raise

    raise RuntimeError("Max retries exceeded")
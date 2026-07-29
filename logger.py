import logging
import os
from datetime import datetime

def setup_logger():
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
    return logging.getLogger("multi-agent")

logger = setup_logger()
import json
import time
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("api_logs.jsonl")

class APILogger:

    @staticmethod
    def log_request_response(input_data, output_data, latency_ms):
        log = {
            "timestamp": datetime.utcnow().isoformat(),
            "input": input_data,
            "output": output_data,
            "latency_ms": latency_ms
        }

        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log) + "\n")

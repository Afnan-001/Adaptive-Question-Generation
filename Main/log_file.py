# log_file.py

import os
import json
import time
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Dict, Any, List, Optional

class LLMCallLogger:
    def __init__(
        self,
        log_path: str = "llm_calls.jsonl",
        rotate_max_bytes: int = 5 * 1024 * 1024,
        rotate_backup_count: int = 5,
        redact_keys: Optional[List[str]] = None,
    ):
        self.log_path = log_path
        self.redact_keys = set(k.lower() for k in (redact_keys or []))

        self.logger = logging.getLogger("LLMLogger")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = RotatingFileHandler(log_path, maxBytes=rotate_max_bytes, backupCount=rotate_backup_count)
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)

    def _redact(self, record: Dict[str, Any]) -> Dict[str, Any]:
        secret_markers = ["key", "secret", "token", "authorization", "password", "cookie", "bearer"]
        redacted = {}
        for k, v in record.items():
            if k.lower() in self.redact_keys or any(m in k.lower() for m in secret_markers):
                redacted[k] = "***REDACTED***"
            else:
                redacted[k] = v
        return redacted

    def log(self, record: Dict[str, Any]):
        safe_record = self._redact(record)
        self.logger.info(json.dumps(safe_record, ensure_ascii=False))

    def log_success(
        self,
        provider: str,
        model_or_deployment: str,
        messages: List[Dict[str, Any]],
        response_text: str,
        usage: Dict[str, Any],
        latency_ms: int,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.log({
            "ts": datetime.utcnow().isoformat() + "Z",
            "event": "LLM_CALL",
            "provider": provider,
            "model_or_deployment": model_or_deployment,
            "messages": messages,
            "response_text": response_text,
            "usage": usage,
            "latency_ms": latency_ms,
            "request_id": request_id,
            "metadata": metadata or {},
        })

    def log_error(
        self,
        provider: str,
        model_or_deployment: str,
        messages: List[Dict[str, Any]],
        error: str,
        latency_ms: int,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.log({
            "ts": datetime.utcnow().isoformat() + "Z",
            "event": "LLM_ERROR",
            "provider": provider,
            "model_or_deployment": model_or_deployment,
            "messages": messages,
            "error": error,
            "latency_ms": latency_ms,
            "request_id": request_id,
            "metadata": metadata or {},
        })
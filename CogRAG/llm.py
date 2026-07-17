import litellm
import time
import os
import json

from CogRAG.log_file import LLMCallLogger


# ----------------------------
# Helper: Make JSON Safe
# ----------------------------

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif hasattr(obj, "__dict__"):
        return make_json_safe(vars(obj))
    else:
        try:
            json.dumps(obj)
            return obj
        except:
            return str(obj)


# ----------------------------
# LLM Class
# ----------------------------

class LLM:

    def __init__(
        self,
        model='gpt-4',
        sys_msg="You are an AI assistant that helps people find information.",
        log_path='common_log.jsonl',
        temperature=0.7,
        top_p=1
    ):
        self.model = "azure/gpt-4o"
        self.api_base = "https://ctonpeuiaopenai.openai.azure.com/"
        self.api_version = "2025-01-01-preview"
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.sys_msg = sys_msg
        self.temperature = temperature
        self.top_p = top_p

        self.logger = LLMCallLogger(log_path)

    # ----------------------------
    # LLM Call
    # ----------------------------

    def get_response(self, usr_msg, sys_msg=None):

        if sys_msg:
            self.sys_msg = sys_msg

        start = time.time()

        messages = [
            {
                "role": "system",
                "content": self.sys_msg
            },
            {
                "role": "user",
                "content": usr_msg
            }
        ]

        try:
            response = litellm.completion(
                model=self.model,
                api_base=self.api_base,
                api_version=self.api_version,
                api_key=self.api_key,
                temperature=self.temperature,
                top_p=self.top_p,
                messages=messages
            )

            latency_ms = int((time.time() - start) * 1000)

            text = response["choices"][0]["message"]["content"]

            # ✅ ✅ FIX: Make everything JSON-safe
            usage = make_json_safe(response.get("usage", {}))
            safe_messages = make_json_safe(messages)

            # ✅ ✅ SUCCESS LOG
            self.logger.log_success(
                provider="azure-openai",
                model_or_deployment=self.model,
                messages=safe_messages,
                response_text=text,
                usage=usage,
                latency_ms=latency_ms
            )

            return text

        except Exception as e:

            latency_ms = int((time.time() - start) * 1000)

            # ✅ ✅ ERROR LOG
            self.logger.log_error(
                provider="azure-openai",
                model_or_deployment=self.model,
                messages=make_json_safe(messages),
                error=str(e),
                latency_ms=latency_ms
            )

            raise

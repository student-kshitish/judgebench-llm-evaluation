import httpx


class OllamaProvider:
    def __init__(
        self,
        model,
        base_url="http://localhost:11434",
        timeout_seconds=180,
        temperature=0.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.call_count = 0
        self.total_duration_ns = 0

    def generate(self, prompt):
        self.call_count += 1

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        with httpx.Client(
            timeout=self.timeout_seconds
        ) as client:
            response = client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )

            response.raise_for_status()
            data = response.json()

        self.total_duration_ns += data.get(
            "total_duration",
            0,
        )

        return {
            "text": data.get("response", ""),
            "model": data.get("model", self.model),
            "prompt_tokens": data.get("prompt_eval_count"),
            "completion_tokens": data.get("eval_count"),
            "duration_ns": data.get("total_duration"),
        }

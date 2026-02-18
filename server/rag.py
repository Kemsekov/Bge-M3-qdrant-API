import json
from typing import List
from pathlib import Path

import httpx


class SimpleOpenAI:
    """Simple chat completion class compatible with Ollama and OpenAI-served models."""
    
    def __init__(
        self, 
        base_url: str, 
        api_key: str, 
        temperature: float = 0.7, 
        max_tokens: int = 1024, 
        timeout: float = 120.0
    ):
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.http = httpx.Client(
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
                keepalive_expiry=120,
            ),
        )
        self.api_key = api_key

    def completion(self, model: str, messages: list) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept-Language": "en-US,en"
        }
        resp = self.http.post(self.base_url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        if "choices" in data:
            return data["choices"][0]["message"]["content"].strip()
        else:
            return data["message"]["content"].strip()

    def close(self):
        self.http.close()

    def __enter__(self): 
        return self
    
    def __exit__(self, *a): 
        self.close()


def _extract_keywords_from_string(text: str) -> List[str]:
    start = text.find('[')
    end = text.find(']')
    if start == -1 or end == -1 or end <= start:
        return []
    json_str = text[start:end + 1]
    return json_str[1:-1].replace("\"", "").split(',')


class Rag:
    """RAG orchestration class for managing LLM interactions and embeddings."""

    def __init__(
        self,
        system_prompt: str,
        message_prompt: str,
        keyword_prompt: str,
        self_evaluation_prompt: str,
        rephrase_prompt: str,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> None:
        self.system_prompt = system_prompt
        self.message_prompt = message_prompt
        self.provider = base_url
        self.client = SimpleOpenAI(
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.keyword_prompt = keyword_prompt
        self.self_evaluation_prompt = self_evaluation_prompt
        self.rephrase_prompt = rephrase_prompt

    def answer(self, question: str, context: List[str]) -> str:
        message = self.message_prompt.format(
            question=question,
            context="\n".join(context)
        )
        response = self.client.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": message}
            ],
        )
        return response

    def get_keywords(self, message: str) -> List[str]:
        response = self.client.completion(
            model=self.model,
            messages=[
                {"role": "user", "content": self.keyword_prompt.format(message=message)}
            ],
        )
        return _extract_keywords_from_string(response)

    def self_eval(self, message: str) -> dict:
        prompt = self.self_evaluation_prompt.format(message=message)
        content = self.client.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = content[start:end + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        return {"answer_score": 0, "context_score": 0}

    def rephrase(self, message: str) -> List[str]:
        prompt = self.rephrase_prompt.format(message=message)
        content = self.client.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        start = content.find('[')
        end = content.rfind(']')
        if start != -1 and end != -1 and end > start:
            json_str = content[start:end + 1]
            return json.loads(json_str)
        raise RuntimeError("Failed to rephrase")


def initialize_rag(
    model: str,
    url: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    prompts_path: str,
) -> Rag | None:
    """Initialize RAG instance if LLM is configured."""
    if not model or not url:
        return None
    
    prompts = json.load(open(prompts_path, 'r'))
    print("Initialized rag model")
    return Rag(
        system_prompt=prompts["system_prompt"]["template"],
        message_prompt=prompts["message_prompt"]["template"],
        keyword_prompt=prompts["keyword_prompt"]["template"],
        rephrase_prompt=prompts["rephrase_prompt"]["template"],
        self_evaluation_prompt=prompts["self_evaluation_prompt"]["template"],
        base_url=url,
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )

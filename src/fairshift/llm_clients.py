from __future__ import annotations

import os
import time
from dataclasses import dataclass

import anthropic
from dotenv import load_dotenv
from mistralai import Mistral
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


@dataclass
class LLMClients:
    chatgpt: OpenAI | None = None
    claude: anthropic.Anthropic | None = None
    mistral: Mistral | None = None
    local: OpenAI | None = None


def build_clients(local_base_url: str = "http://localhost:1234/v1") -> LLMClients:
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    mistral_key = os.getenv("MISTRAL_API_KEY")

    return LLMClients(
        chatgpt=OpenAI(api_key=openai_key) if openai_key else None,
        claude=anthropic.Anthropic(api_key=anthropic_key) if anthropic_key else None,
        mistral=Mistral(api_key=mistral_key) if mistral_key else None,
        local=OpenAI(base_url=local_base_url, api_key="not-needed"),
    )


def _missing(name: str) -> ValueError:
    return ValueError(f"{name} client is not configured. Check your .env file or local model server.")


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=5, min=30, max=300),
    stop=stop_after_attempt(5),
)
def get_llm_response(
    prompt: str,
    model_name: str,
    clients: LLMClients,
    provider: str = "api",
    max_tokens: int = 50,
) -> str:
    print(f"Making new API call to {model_name}")

    try:
        if model_name == "ChatGPT":
            if clients.chatgpt is None:
                raise _missing("OpenAI")
            response = clients.chatgpt.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
                top_p=0.1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            return response.choices[0].message.content.strip()

        if model_name == "Claude":
            if clients.claude is None:
                raise _missing("Anthropic")
            response = clients.claude.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=max_tokens,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()

        if model_name == "Mistral" and provider == "api":
            if clients.mistral is None:
                raise _missing("Mistral")
            time.sleep(4)
            response = clients.mistral.chat.complete(
                model="open-mistral-7b",
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()

        if model_name == "Mistral" and provider == "local":
            if clients.local is None:
                raise _missing("local Mistral")
            response = clients.local.chat.completions.create(
                model="mistral-7b-instruct-v0.3",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=max_tokens,
                top_p=0.1,
            )
            return response.choices[0].message.content.strip()

        if model_name.startswith("Gemma"):
            if clients.local is None:
                raise _missing("local Gemma")
            response = clients.local.chat.completions.create(
                model={"Gemma2B": "gemma-2-2b-it", "Gemma9B": "gemma-2-9b-it"}.get(
                    model_name, "gemma-2-27b-it"
                ),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=max_tokens,
                top_p=0.1,
            )
            return response.choices[0].message.content.strip()

        raise ValueError(f"Unsupported model/provider combination: {model_name}/{provider}")

    except Exception as exc:
        if "rate limit" in str(exc).lower():
            print("Rate limit hit; retrying with exponential backoff...")
            raise
        print(f"Error with {model_name}: {exc}")
        return "Error"

import os
from typing import Optional, List

from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic


def gpt(prompt: str, system: Optional[str] = None, model: Optional[str] = "gpt-4.1-mini") -> str:
    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "developer",
                "content": system
            },
            {
                "role": "user",
                "content": prompt
            },
        ]
    )
    return response.output_text


def claude(prompt: str, system: str = "You are a helpful assistant", model: str = "claude-sonnet-4-20250514") -> str:
    load_dotenv()
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model=model,
        system=system,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        max_tokens=10_000,
    )

    return response.content[0].text

import json
import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

with (Path(__file__).resolve().parent.parent / "prompts.json").open('r') as f:
    PROMPTS: dict = json.load(f)

def ping_backend(prompt: str, system: str = PROMPTS["START_REASONING"], model: str = "claude-sonnet-4-20250514") -> str:
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
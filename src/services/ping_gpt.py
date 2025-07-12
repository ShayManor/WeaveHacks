import os
from typing import Optional, List

from dotenv import load_dotenv
from openai import OpenAI


def gpt(prompt: str, system: Optional[str] = None, model: Optional[str] = "gpt-4.1-mini") -> str:
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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

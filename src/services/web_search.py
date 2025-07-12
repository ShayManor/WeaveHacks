import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from exa_py import Exa
from bs4 import BeautifulSoup
import dotenv
from exa_py.api import ResultWithText

from src.services.ping_gpt import gpt

_LINK_RE = re.compile(
    r"""
      # 1) Markdown images  ![alt](url)
      !\[[^\]]*]          \( [^)]+ \) |

      # 2) Markdown links  [text](url)  ⇒ keep "text"
      \[ ([^\]]+) ]       \( [^)]+ \) |

      # 3) HTML anchors    <a href="url">text</a>  ⇒ keep "text"
      <a\s+[^>]*href=     ["'][^"']+["'][^>]*> (.*?) </a> |

      # 4) bare URLs / mailto
      (?:https?://|www\.) \S+ |
      mailto:\S+
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

with (Path(__file__).resolve().parent.parent / "prompts.json").open('r') as f:
    PROMPTS: dict = json.load(f)


def strip_links(text: str) -> str:
    def _replacer(match: re.Match) -> str:
        return match.group(1) or match.group(2) or ''

    return _LINK_RE.sub(_replacer, text)


def clean_web_results(raw_data: list[ResultWithText], k: int = 10) -> list[dict[str, str | Any]]:
    results = getattr(raw_data, "results", raw_data)
    good = []
    for hit in results[:k]:
        text = hit.text or ""
        text = BeautifulSoup(text, "lxml").get_text(separator=" ", strip=True)
        good.append(
            {
                "title": hit.title,
                "snippet": strip_links(text),
            }
        )
    return good


def search_web(prompt: str):
    load_dotenv()
    exa = Exa(os.getenv("EXA_API_KEY"))
    raw_result = exa.search_and_contents(
        prompt,
        text=True,
        num_results=5
    ).results
    raw_result = sorted(raw_result, key=lambda h: h.published_date, reverse=True)
    clean_result = clean_web_results(raw_result, 10)
    print("cleaned")
    summary = gpt(prompt=str(clean_result), system=PROMPTS["WEB_SUMMARIZER"], model="gpt-4.1-mini")
    return summary

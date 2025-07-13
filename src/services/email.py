import os
import uuid

import requests
from dotenv import load_dotenv
from langchain_community.agent_toolkits.load_tools import load_tools


def execute(recipient: str, content: str, subject: str) -> str:
    """
    Sends email to given recipient.
    :param recipient: email address to send to
    :return: "Email sent"
    """
    load_dotenv()
    api_key = os.getenv("EXA_API_KEY")
    print(api_key)
    if not api_key:
        return "Error: EXA_API_KEY not set"
    url = os.getenv("MCP_EMAIL_URL", "http://localhost:8000")
    # Build JSON-RPC 2.0 payload
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "send_email",
        "params": {
            "receiver": [recipient],
            "subject": subject,
            "body": content
        }
    }
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            return f"Error from MCP: {data['error']}"
        return "Email sent"
    except requests.exceptions.RequestException as e:
        return f"Error sending email: {e}"

if __name__ == '__main__':
    print(execute('shay.manor@gmail.com', 'Shalom', 'EMAIL RULES'))
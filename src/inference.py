import json
from pathlib import Path

import requests
from typing import Dict, Any, List
import importlib
import os
from anthropic import Anthropic
from dotenv import load_dotenv

from src.crew_integration import HomeMateCrewIntegration
from src.services import crew_manager

with (Path(__file__).resolve().parent / "prompts.json").open('r') as f:
    PROMPTS: dict = json.load(f)


class HomeMateAgent:
    def __init__(self, use_claude=True, model="claude-3-5-haiku-latest"):
        self.use_claude = use_claude
        self.model = model
        self.tools = self._load_tools()
        self.conversation_history = []
        self.crew_integration = HomeMateCrewIntegration(self)
        crew_manager.crew_integration = self.crew_integration
        self.tools["crew_manager"] = crew_manager.execute

        if use_claude:
            load_dotenv()
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _convert_to_claude_format(self, messages: List[Dict]) -> List[Dict]:
        claude_messages = []

        # Process all messages after system
        for i, msg in enumerate(messages[1:], 1):
            if msg["role"] == "user":
                claude_messages.append({
                    "role": "user",
                    "content": msg["content"]
                })
            elif msg["role"] == "assistant":
                # Include assistant messages as-is
                claude_messages.append({
                    "role": "assistant",
                    "content": msg["content"]
                })
            elif msg["role"] == "tool":
                # Tool results need to be from user in Claude format
                claude_messages.append({
                    "role": "user",
                    "content": f"[Tool '{msg['name']}' returned: {msg['content']}]"
                })

        if claude_messages and claude_messages[-1]["role"] == "assistant":
            claude_messages.append({
                "role": "user",
                "content": "Continue with the next step based on the tool result above."
            })

        return claude_messages

    def _parse_response(self, response_text: str) -> Dict:
        """Parse response with better JSON extraction"""
        import re

        result = {"content": "", "tool_calls": []}

        # Check for final answer
        if "final_answer:" in response_text:
            result["content"] = response_text
            return result

        # Extract thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=tool_call:|$)', response_text, re.DOTALL | re.IGNORECASE)
        if thought_match:
            result["content"] = f"Thought: {thought_match.group(1).strip()}"

        # Method 1: Find tool_call: and extract everything after it
        tool_call_match = re.search(r'tool[_\s]call:\s*(.+)', response_text, re.IGNORECASE)
        if tool_call_match:
            json_str = tool_call_match.group(1).strip()

            if json_str.startswith('{'):
                brace_count = 0
                end_pos = 0

                for i, char in enumerate(json_str):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break

                if end_pos > 0:
                    try:
                        json_obj = json_str[:end_pos]
                        tool_data = json.loads(json_obj)
                        result["tool_calls"].append({
                            "name": tool_data["name"],
                            "arguments": json.dumps(tool_data.get("arguments", {}))
                        })
                        print(f"Successfully parsed tool call: {tool_data['name']}")
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                        print(f"Attempted to parse: {json_obj}")

        # Method 2: If method 1 fails, try to find any valid tool JSON
        if not result["tool_calls"]:
            # Look for pattern like {"name": "...", "arguments": {...}}
            all_json_matches = re.finditer(r'\{[^{]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{', response_text)

            for match in all_json_matches:
                start = match.start()
                # Find the closing brace for the complete JSON
                brace_count = 0
                end_pos = start

                for i in range(start, len(response_text)):
                    if response_text[i] == '{':
                        brace_count += 1
                    elif response_text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break

                if end_pos > start:
                    try:
                        json_str = response_text[start:end_pos]
                        tool_data = json.loads(json_str)
                        result["tool_calls"].append({
                            "name": tool_data["name"],
                            "arguments": json.dumps(tool_data.get("arguments", {}))
                        })
                        print(f"Successfully parsed tool call (method 2): {tool_data['name']}")
                        break
                    except:
                        continue

        print(f"Parse result - Content: {result['content'][:50]}...")
        print(f"Parse result - Tool calls: {result['tool_calls']}")

        return result

    def _load_tools(self) -> Dict[str, callable]:
        """Dynamically load all tools from services folder"""
        tools = {}
        services_dir = "services"

        print(f"Loading tools from {services_dir}...")

        for filename in os.listdir(services_dir):
            if filename.endswith(".py") and filename != "__init__.py" and filename != "ping_gpt.py":
                module_name = filename[:-3]
                try:
                    module = importlib.import_module(f"{services_dir}.{module_name}")

                    if hasattr(module, 'execute'):
                        tools[module_name] = module.execute
                        print(f"✓ Loaded tool: {module_name}")
                    else:
                        print(f"✗ No execute function in {module_name}")
                except Exception as e:
                    print(f"✗ Failed to load {module_name}: {e}")

        print(f"Total tools loaded: {len(tools)}")
        print(f"Available tools: {list(tools.keys())}")
        return tools


    def _call_llm(self, messages: List[Dict]) -> Dict:
        """Call Claude or local Llama model and return structured response"""
        if self.use_claude:
            last_tool_result = None
            for msg in reversed(messages):
                if msg["role"] == "tool":
                    last_tool_result = msg
                    break

            if last_tool_result and messages[-1]["role"] == "tool":
                claude_messages = self._convert_to_claude_format(messages)
                claude_messages.append({
                    "role": "user",
                    "content": "The tool has completed. What's next? If all tasks are done, provide final_answer:, otherwise continue with the next step."
                })
            else:
                claude_messages = self._convert_to_claude_format(messages)

            response = self.client.messages.create(
                model=self.model,
                system=PROMPTS["STEP_BY_STEP_SYSTEM"],
                # system=messages[0]["content"],
                messages=claude_messages,
                max_tokens=8192,
            )
            if not response.content:
                return {"content": "Waiting for tool response...", "tool_calls": []}
            print(response.content)
            return self._parse_response(response.content[0].text.replace("\n", ""))

    def _parse_claude_response(self, response_text: str) -> Dict:
        """Parse Claude's text response into structured format"""
        result = {"content": "", "tool_calls": []}

        if "Thought:" in response_text:
            thought_start = response_text.find("Thought:")
            thought_end = response_text.find('"tool_calls"') if '"tool_calls"' in response_text else len(response_text)
            result["content"] = response_text[thought_start:thought_end].strip()
        else:
            result["content"] = response_text.strip()

        # Extract tool_calls if present
        if '"tool_calls":' in response_text:
            try:
                start = response_text.find('"tool_calls":') + len('"tool_calls":')
                end = response_text.find(']', start) + 1
                tool_calls_json = response_text[start:end]
                result["tool_calls"] = json.loads(tool_calls_json)
            except:
                pass

        return result

    def run(self, user_input: str) -> str:
        """Main reasoning loop"""
        messages = [
            {"role": "system",
             "content": "You are HomeMate OS, an autonomous assistant that reasons step-by-step using Thought, Action, Observation loops until final_answer."},
            {"role": "user", "content": user_input}
        ]

        max_iterations = 10

        for iteration in range(max_iterations):
            print(f"\n=== Iteration {iteration + 1} ===")
            print(f"Current message count: {len(messages)}")

            response = self._call_llm(messages)

            content = response.get("content", "")
            tool_calls = response.get("tool_calls", [])


            if not content.strip() and not tool_calls:
                print("Empty response, skipping...")
                continue

            assistant_msg = {"role": "assistant", "content": content}
            print(assistant_msg)
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            if "final_answer:" in content:
                return content.split("final_answer:")[-1].strip()

            if tool_calls:
                for tool_call in tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = json.loads(tool_call.get("arguments", "{}"))

                    print(f"Executing tool: {tool_name} with args: {tool_args}")

                    if tool_name in self.tools:
                        try:
                            result = self.tools[tool_name](**tool_args)
                            print(f"Tool result: {result}")

                            messages.append({
                                "role": "tool",
                                "name": tool_name,
                                "content": str(result)
                            })
                        except Exception as e:
                            print(f"Tool execution error: {e}")
                            messages.append({
                                "role": "tool",
                                "name": tool_name,
                                "content": f"Error: {str(e)}"
                            })

        return "Maximum iterations reached"


if __name__ == "__main__":
    agent = HomeMateAgent(True)
    res = agent.run(
        user_input="Good morning! Call my dentist and make an appointment for a root canal next week. Call james at 13478347434 and tell him I can't make it to his barbeque today.")
    print('====================Final Result====================')
    print(res)

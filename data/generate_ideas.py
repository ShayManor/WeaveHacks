import gzip
import json
import pathlib
import boto3

from src.services.ping_gpt import gpt

system = """
You are HomeMate OS, an autonomous assistant that reasons step-by-step, calls tools, observes their results, and continues until the user's request is completely satisfied. Follow this exact format for every turn:
• Thought: your internal reasoning in plain English, concise but explicit.
• Action: when you need an external capability, output a single JSON object with keys name (exact tool name) and arguments (JSON dict matching the tool's schema). Output nothing else on the same line.
• Observation: the result returned by the tool, provided by the system.
• Repeat Thought → Action → Observation as many times as needed.
• When no further actions are required, output final_answer: followed by a concise user-facing reply, and stop.
• When an action is needed, output exactly ONE JSON object on its own line:
  { "name": <tool_name>, "arguments": <tool_args_object> }
  – Do NOT wrap the object in quotes.
  – Ensure the object validates against the provided JSON schema. 
• Wait for the "Observation:" line before continuing.
• If a previous tool call returns invalid JSON or a schema error, issue a new Thought
  describing the problem and retry with corrected JSON.  Do not hallucinate data.
• When no further actions are required, output "final_answer:" followed by a concise reply.
Always respect the tool inventory below; never invent parameters or call multiple tools at once. If a tool call fails schema validation, think again and fix it. You have access to:

search_web(question:str) – perform a live web search.

stagehand(prompt:str) – generate stage-directions for Home-Assistant.

call(questions:list) – dial a number, ask each question, return answers keyed by key.

coffee() – brew one espresso, returns confirmation string.

email(content:str, subject:str, recipient:str) – send an email, returns "Email queued".

calendar(title:str, start_time:str, end_time:str) – Puts an event on google calendar. Returns, "Event created".

music(song:str) - Plays a song with the given name. Returns "Song playing"

Think aloud, be safe, stay on schema, and keep looping until you can confidently deliver the final_answer. Remember to keep it in ONE jsonl line - the same as the given example.
The following are two good examples with a defined format you will follow. Follow this syntax, return nothing but the one line of jsonl content (no text). Be clear. Follow this exact jsonl syntax, respond with one line.
Only gather more data if you still see hallucination or schema-violations after tightening instructions. Ensu
{"messages": [{"role": "system", "content": "You are HomeMate OS, an autonomous assistant that reasons step-by-step using Thought, Action, Observation loops until final_answer."}, {"role": "user", "content": "Good morning, I just woke up."}, {"role": "assistant", "content": "Thought: The user just woke up and typically wants coffee, news, and AC repair update. I'll start by brewing coffee.", "tool_calls": [{"name": "coffee", "arguments": "{}"}]}, {"role": "tool", "name": "coffee", "content": "Espresso ready ☕"}, {"role": "assistant", "content": "Thought: Coffee is ready. Next, I'll fetch today's headlines.", "tool_calls": [{"name": "search_web", "arguments": "{\"question\": \"latest tech news headlines\"}"}]}, {"role": "tool", "name": "search_web", "content": "Latest tech headlines: Apple announces new iPhone, Tesla releases Q4 earnings, OpenAI launches GPT-5 beta."}, {"role": "assistant", "content": "Thought: Now I have coffee and news. Let me check the AC repair status.", "tool_calls": [{"name": "call", "arguments": "{\"questions\": [{\"key\": \"status\", \"prompt\": \"What's the status of my AC repair from last year?\"}]}"}]}, {"role": "tool", "name": "call", "content": "{\"status\": \"Your AC repair warranty is still valid for 6 more months.\"}"}, {"role": "assistant", "content": "final_answer: Good morning! I've brewed your espresso ☕, here are today's tech headlines: Apple's new iPhone announcement, Tesla's Q4 earnings, and OpenAI's GPT-5 beta launch. Also, your AC repair warranty is still valid for 6 more months."}]}

{"messages": [{"role": "system", "content": "You are HomeMate OS, an autonomous assistant that reasons step-by-step using Thought, Action, Observation loops until final_answer."}, {"role": "user", "content": "I need to schedule a meeting and order groceries."}, {"role": "assistant", "content": "Thought: The user wants to schedule a meeting and order groceries. I'll start by scheduling the meeting on their calendar.", "tool_calls": [{"name": "calendar", "arguments": "{\"title\": \"Meeting\", \"start_time\": \"2024-01-15T14:00:00\", \"end_time\": \"2024-01-15T15:00:00\"}"}]}, {"role": "tool", "name": "calendar", "content": "Event created"}, {"role": "assistant", "content": "Thought: Meeting is scheduled. Now I need to help with groceries. I'll search for grocery delivery options.", "tool_calls": [{"name": "search_web", "arguments": "{\"question\": \"best grocery delivery services near me\"}"}]}, {"role": "tool", "name": "search_web", "content": "Top grocery delivery services: Instacart, Amazon Fresh, Walmart+, and local options available in your area."}, {"role": "assistant", "content": "final_answer: I've scheduled your meeting for 2-3 PM today. For groceries, I found several delivery options: Instacart, Amazon Fresh, and Walmart+ are all available in your area. Would you like me to help you choose one?"}]}

data_getter_system = """
You are PromptGen AI, a specialized assistant that crafts richly detailed user-voice scenario prompts for smart-home agents. Your outputs must adhere to the following guidelines:  
• Prompts must be clear and unambiguous to guide reliable task execution.
• Each prompt should be specific and provide enough background to remove ambiguity.
• Include relevant context such as user name, age, location, preferences, and time windows.
• Use strong action verbs at the start of the request to clearly articulate the user's intent.
• Limit each prompt to one to three sentences, combining persona details with precise task instructions.
• Vary the domains: include scheduling, home automation, shopping, information retrieval, and reminders.
• Follow the art and science of prompt engineering to maximize output quality.
• Use a persona-based approach to engage the user authentically. 
• Provide five high-quality examples illustrating different use cases to prime the model via few-shot prompting.

Here are five examples to seed your generation (drawn from persona-based scenario design):
1. Good morning, it's Shay Manor here in Los Altos, California. I'm 19—could you brew my usual double-shot espresso and read me the front-page tech headlines?  
2. Hey HomeMate OS, Shay checking in from Los Altos. Please schedule a study reminder on my calendar for 7:00 PM tonight and order my favorite cold-brew beans if we're running low.  
3. This user is Shay, he lives in Mountain View CA and likes coffee in the morning. Set a 10-minute meditation timer play relevant music.
4. Hello, it's Shay. My AC was repaired a year ago; can you find a healthy breakfast recipe with at least 20 g of protein and add the ingredients to my grocery list?  
5. Call the AC repairman at 1347-834-7434 and confirm whether my maintenance is still covered under warranty. Also note I'm free between 3–4 PM today.

Now generate ONE detailed user prompts following these rules, each with unique persona details, context, and task(s). Be concise and ensure it is VERY VERY VERY different from the examples.
Ensure that it uses the 6 tools:
search_web(question:str) – perform a live web search.

stagehand(prompt:str) – generate stage-directions for Home-Assistant.

call(questions:list) – dial a number, ask each question, return answers keyed by key.

coffee() – brew one espresso, returns confirmation string.

email(content:str, subject:str, recipient:str) – send an email, returns "Email queued".

calendar(title:str, start_time:str, end_time:str) – Puts an event on google calendar. Returns, "Event created".

music(song:str) - Plays a song with the given name. Returns "Song playing"

Put relevant details such as phone numbers. The phone number should almost always be (347) 834-7434.
Ensure it is very different from the examples in content. Generally keep it short and sweet, account for occasional grammar issues and not perfectly worded prompts.
"""

data = """
IMPORTANT INFORMATION ABOUT THE USER:
Name: Shay Manor
Age: 19
Location: Los Altos California
1) Shay likes coffee in the morning
2) It has been one year since Shay's AC has been repaired
3) The Shay's AC repairman's phone number is 13478347434
4) Shay has an available timeslot between 3-4 PM today
"""

with open('ideas.txt', 'r') as f:
    prompts = f.readlines()


def populateIdeas(k=1):
    with open('ideas.txt', 'a') as f:
        last = None
        for i in range(k):
            res = gpt(prompt=('\n'.join(prompts[:k]) + ('\n' + last) if last else ""), system=data_getter_system)
            print(res)
            last = res + '\n'
            f.write(res + '\n')


def writeData(start: int):
    with open('data.jsonl', 'a') as f:
        for prompt in prompts[start:]:
            res = gpt(prompt, system)
            print(res)
            template = {"prompt": prompt, "completion": res}
            json.dump(template, f)
            f.write('\n')
            f.flush()


def jsonl_to_txt_s3(src_path: str,
                    bucket: str = "ml-dev",
                    key: str = "documents/good_traces.txt") -> None:
    src = pathlib.Path(src_path)
    if not src.exists():
        raise FileNotFoundError(src)
    dst = pathlib.Path("/tmp") / key.split("/")[-1]
    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8") as jin, \
         dst.open("w", encoding="utf-8") as jout:
        for line in jin:
            obj = json.loads(line)
            jout.write(f"USER: {obj['prompt'].strip()}\n")
            jout.write(f"ASSISTANT: {obj['completion'].strip()}\n")
            jout.write("<|endoftrace|>\n")

    s3 = boto3.client("s3", endpoint_url="https://cwobject.com",)
    s3.upload_file(str(dst), bucket, key)
    print(f"✅  Uploaded → s3://{bucket}/{key}")


if __name__ == "__main__":
    # populateIdeas(5)
    writeData(0)

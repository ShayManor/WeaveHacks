# crew_ivr_survey.py
"""
CrewAI-powered IVR survey.
Call phone_survey('+15558675309',
                  [{"key":"name","prompt":"May I have your full name?"},
                   {"key":"needs","prompt":"What service are you looking for?"},
                   {"key":"slot","prompt":"Is morning or afternoon better?"}])
returns {"name": "...", "needs": "...", "slot": "..."}
"""
import os, threading, time, json, sys
from typing import List, Dict
from dotenv import load_dotenv
from flask import Flask, request, Response, jsonify
from twilio.rest import Client
import twilio
from twilio.twiml.voice_response import VoiceResponse, Gather
from pyngrok import ngrok                    # easier than raw ngrok CLI
from crewai import Agent, Task, Crew, Process

# ─────────────────────────  Twilio / Flask plumbing  ──────────────────────────
load_dotenv()
TWILIO_USER_SID="USf92da53d58f7a16ea50085211495485d"
TWILIO_SID="SK1b378aed0c18f8a009c9dca483c0086e"
TWILIO_TOKEN="8sHkOfs9XbxwM0ldcXeYeItaZ56cGMII"
TWILIO_FROM  = "17473640171"

TWILIO       = Client(TWILIO_SID, TWILIO_TOKEN)

app        = Flask(__name__)
CALL_DATA  = {}   # {CallSid: {answers}}
QUESTIONS  = []   # populated per survey run

@app.route("/ivr/ask")
def ivr_ask():
    """Ask the question at ?index=i and repeat if no input."""
    idx = int(request.args["index"])
    sid = request.values.get("CallSid")
    if sid and sid not in CALL_DATA:
        CALL_DATA[sid] = {}
    vr  = VoiceResponse()
    g   = Gather(
            input="speech dtmf",
            action=f"/ivr/handle?index={idx}",
            timeout=6,
            hints="yes,no,morning,afternoon,consultation,cleaning")
    g.say(QUESTIONS[idx]["prompt"])
    vr.append(g)
    vr.say("I didn’t catch that.")
    vr.redirect(f"/ivr/ask?index={idx}")
    return Response(str(vr), mimetype="text/xml")

@app.route("/ivr/handle", methods=["POST"])
def ivr_handle():
    """Store the answer then move to next or finish."""
    idx  = int(request.args["index"])
    sid  = request.values["CallSid"]
    ans  = (request.values.get("SpeechResult") or
            request.values.get("Digits","")).strip()
    CALL_DATA[sid][QUESTIONS[idx]["key"]] = ans
    nxt  = idx + 1
    vr   = VoiceResponse()
    if nxt < len(QUESTIONS):
        vr.redirect(f"/ivr/ask?index={nxt}")
    else:
        vr.say("Thank you, goodbye.")
        vr.hangup()
    return Response(str(vr), mimetype="text/xml")

@app.route("/ping")
def ping(): return jsonify({"ping":"pong"})

def _run_flask():
    app.run(port=8080, threaded=True)

# ───────────────────────────── CrewAI Tool ────────────────────────────────────
class TwilioIVRTool:
    """Exposes dial() so Crew agents can start the survey."""
    def __init__(self, base_url: str):
        self.base_url = base_url
    def dial(self, to_number: str) -> str:
        call = TWILIO.calls.create(
            to=to_number,
            from_=TWILIO_FROM,
            url=f"{self.base_url}/ivr/ask?index=0")
        return call.sid

# ─────────────────────────── public API function ─────────────────────────────
def phone_survey(to_number: str,
                 questions: List[Dict[str,str]],
                 poll_interval: float = 1.0,
                 timeout: int = 180) -> Dict[str,str]:
    """
    High-level one-shot helper.  Launches the IVR and
    blocks until all questions are answered or timeout.
    """
    global QUESTIONS
    QUESTIONS[:] = questions

    # 1️⃣ expose Flask over internet
    public_url   = ngrok.connect(8080, bind_tls=True).public_url
    tool         = TwilioIVRTool(public_url)
    threading.Thread(target=_run_flask, daemon=True).start()

    # 2️⃣ build minimal Crew
    caller = Agent(role="Caller",
                   goal="Ask scripted questions over the phone and store answers.",
                   tools=[tool], allow_delegation=False)
    dial_task = Task(description="Dial the target number and conduct the IVR survey.",
                     agent=caller,
                     expected_output="CallSid string returned from Twilio")
    crew = Crew(agents=[caller], tasks=[dial_task], process=Process.sequential)

    # 3️⃣ kick off crew (returns CallSid), then poll shared dict
    call_sid = crew.kickoff(inputs={"to_number": to_number})  # result is str
    deadline = time.time() + timeout
    while time.time() < deadline:
        if call_sid in CALL_DATA and \
           len(CALL_DATA[call_sid]) == len(questions):
            return CALL_DATA[call_sid]
        time.sleep(poll_interval)
    raise TimeoutError(f"Survey timed-out after {timeout}s")

# ───────────────────────────── simple demo ───────────────────────────────────
if __name__ == "__main__":
    q = [
        {"key": "name",  "prompt": "Hi. May I have your full name?"},
        {"key": "needs", "prompt": "What service are you looking for?"},
        {"key": "slot",  "prompt": "Is morning or afternoon better for you?"}
    ]
    print(phone_survey(sys.argv[1] if len(sys.argv)>1 else input("Number: "), q))

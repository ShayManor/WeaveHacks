# minimal_ivr.py
import threading
import time
from typing import Dict, List

import ngrok
from dotenv import load_dotenv
from flask import Flask, request, Response, jsonify
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather
import os, datetime, json, requests

load_dotenv()
TWILIO_SID = os.environ["TWILIO_SID"]
TWILIO_TOKEN = os.environ["TWILIO_TOKEN"]
TWILIO_FROM = '17473640171'
BASE_URL = 'https://a09a76ce0649.ngrok-free.app'

QUESTIONS = [
    {"key": "name", "prompt": "Hi. May I have your full name?"},
    {"key": "needs", "prompt": "What service are you looking for?"},
    {"key": "preferred", "prompt": "Morning or afternoon is better?"}
]

CALL_DATA = {}  # {CallSid: {"name": "...", ...}}

app = Flask(__name__)
tw = Client(TWILIO_SID, TWILIO_TOKEN)


def start_call(to_number: str):
    call = tw.calls.create(
        to=to_number,
        from_=TWILIO_FROM,
        url=f"{BASE_URL}/ask?index=0"
    )
    return call.sid


@app.route("/ask")
def ask():
    idx = int(request.args["index"])
    call_sid = request.values.get("CallSid")  # null on first hop
    if call_sid and call_sid not in CALL_DATA:
        CALL_DATA[call_sid] = {}

    q_text = QUESTIONS[idx]["prompt"]
    vr = VoiceResponse()
    gather = Gather(
        input="speech dtmf",
        action=f"/handle?index={idx}",
        timeout=6,
        hints="yes,no,morning,afternoon,consultation,cleaning"
    )
    gather.say(q_text)
    vr.append(gather)
    vr.say("I didn't catch that.")
    vr.redirect(f"/ask?index={idx}")  # repeat same Q
    return Response(str(vr), mimetype="text/xml")


@app.route("/handle", methods=["POST"])
def handle():
    idx = int(request.args["index"])
    call_sid = request.values["CallSid"]
    answer = (request.values.get("SpeechResult") or
              request.values.get("Digits", "")).strip()
    CALL_DATA[call_sid][QUESTIONS[idx]["key"]] = answer

    next_idx = idx + 1
    if next_idx < len(QUESTIONS):
        vr = VoiceResponse()
        vr.redirect(f"/ask?index={next_idx}")
        return Response(str(vr), mimetype="text/xml")

    requests.post(f"{BASE_URL}/finished", json={
        "call_sid": call_sid,
        "answers": CALL_DATA[call_sid]
    })
    vr = VoiceResponse()
    vr.say("Thank you. We have all we need. Goodbye.")
    vr.hangup()
    return Response(str(vr), mimetype="text/xml")


@app.route("/finished", methods=["POST"])
def finished():
    print(json.dumps(request.json, indent=2))
    return json.dumps(request.json, indent=2)


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"ping": "pong"})


if __name__ == '__main__':
    app.run(port=8080, debug=True, threaded=True)


def phone_survey(
        to_number: str,
        questions: List[Dict[str, str]],
        *,
        poll_interval: float = 1.0,
        timeout: int = 180
) -> Dict[str, str]:
    """
    Dial `to_number`, ask `questions` (list of {'key','prompt'}), return {key: answer}.
    Requires minimal_ivr.py API to be running in the same interpreter.
    """
    global QUESTIONS
    QUESTIONS[:] = questions  # thread-safe list mutation is fine here

    # 2️⃣   Launch the outbound call and capture its CallSid
    call_sid = start_call(to_number)  # Twilio returns a unique 34-char ID

    deadline = time.time() + timeout
    while time.time() < deadline:
        answers = CALL_DATA.get(call_sid, {})
        if len(answers) == len(questions):
            return answers  #  finished
        time.sleep(poll_interval)

    raise TimeoutError(
        f"Survey for {to_number} did not complete within {timeout}s."
    )

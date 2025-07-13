import json
import os
import traceback
from typing import Optional

from flask import Flask, jsonify, request
from flask_cors import CORS
from src.services.ping_gpt import gpt
from src.inference import HomeMateAgent

app = Flask(__name__)


CORS(app)


@app.route('/')
def ping():
    return jsonify({"ping": "pong"})


@app.route("/health/<data>", methods=["GET"])
@app.route("/health", defaults={'data': None}, methods=["GET"])
def health(data: Optional[str]):
    return jsonify({"health": data}) if data else jsonify({"health": "healthy"})


@app.route("/prompt", methods=["POST"])
def prompt():
    user_query = request.args.get("prompt", "").strip()
    if not user_query:
        data = request.get_json()
        user_query = data.get('prompt') or data.get('query', '')
    if not user_query:
        user_query = request.form.get('prompt') or request.form.get('query', '')
    
    print(f"Received prompt: {user_query}")
    try:
        agent = HomeMateAgent(True)
        res = agent.run(user_query)
        return jsonify({'response': res})
    except Exception as e:
        traceback.print_exc()
        print(f"Exception: {e}")
        return jsonify({'response': e.__str__()})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(debug=True, host='0.0.0.0', port=port)

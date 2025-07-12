import json
from typing import Optional

from flask import Flask, jsonify, request
from flask_cors import CORS

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
def complete_user_prompt():
    prompt = request.form.get('prompt')
    print(prompt)
    return jsonify({"response": prompt})


if __name__ == "__main__":
    app.run(debug=True, port=5000)

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
def prompt():
    # Handle both form data and JSON
    if request.is_json:
        data = request.get_json()
        prompt = data.get('prompt') or data.get('query', '')
    else:
        prompt = request.form.get('prompt') or request.form.get('query', '')
    
    print(f"Received prompt: {prompt}")
    
    # For now, just echo back the prompt
    # You can add your AI logic here later
    response = f"I received your message: {prompt}"
    
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')

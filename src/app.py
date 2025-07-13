import json
from typing import Optional

from flask import Flask, jsonify, request
from flask_cors import CORS
from services.ping_gpt import gpt

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
        user_query = data.get('prompt') or data.get('query', '')
    else:
        user_query = request.form.get('prompt') or request.form.get('query', '')
    
    print(f"Received prompt: {user_query}")
    
    try:
        # Use OpenAI GPT to generate a response
        system_prompt = "You are a helpful AI assistant integrated with Google Home. Provide clear, concise, and helpful responses that would be appropriate for voice output."
        
        ai_response = gpt(
            prompt=user_query,
            system=system_prompt,
            model="gpt-4.1-mini"
        )
        
        print(f"AI response: {ai_response}")
        
        return jsonify({"response": ai_response})
        
    except Exception as e:
        print(f"Error calling OpenAI: {e}")
        # Fallback response if AI fails
        fallback_response = f"I received your message: {user_query}. I'm having trouble processing that right now."
        return jsonify({"response": fallback_response})


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')

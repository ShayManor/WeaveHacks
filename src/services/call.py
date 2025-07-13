import os
import json
import time
from typing import List, Dict, Any
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather
from crewai import Agent, Task, Crew
import uuid
import sqlite3
from contextlib import contextmanager


class PhoneSurveyAgent:
    def __init__(self, account_sid: str, auth_token: str, twilio_phone: str):
        """
        Initialize the phone survey agent

        Args:
            account_sid: Twilio Account SID
            auth_token: Twilio Auth Token
            twilio_phone: Your Twilio phone number
        """
        self.client = Client(account_sid, auth_token)
        self.twilio_phone = twilio_phone

        # Initialize database for storing survey state
        self.db_path = "survey_data.db"
        self.init_database()

        # CrewAI Agent for processing responses
        self.response_agent = Agent(
            role="Phone Survey Response Processor",
            goal="Process and extract key information from phone call responses",
            backstory="You are an expert at analyzing phone conversation responses and extracting structured data.",
            verbose=False,
            allow_delegation=False
        )

    def init_database(self):
        """Initialize SQLite database for survey state"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS surveys (
                    id TEXT PRIMARY KEY,
                    phone_number TEXT,
                    questions TEXT,
                    responses TEXT,
                    status TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def create_survey_session(self, phone_number: str, questions: List[Dict[str, str]]) -> str:
        """Create a new survey session"""
        survey_id = str(uuid.uuid4())

        with self.get_db_connection() as conn:
            conn.execute('''
                INSERT INTO surveys (id, phone_number, questions, responses, status)
                VALUES (?, ?, ?, ?, ?)
            ''', (survey_id, phone_number, json.dumps(questions), json.dumps({}), 'pending'))
            conn.commit()

        return survey_id

    def get_survey_session(self, survey_id: str) -> Dict[str, Any]:
        """Get survey session data"""
        with self.get_db_connection() as conn:
            cursor = conn.execute('''
                SELECT phone_number, questions, responses, status 
                FROM surveys WHERE id = ?
            ''', (survey_id,))
            row = cursor.fetchone()

            if row:
                return {
                    'phone_number': row[0],
                    'questions': json.loads(row[1]),
                    'responses': json.loads(row[2]),
                    'status': row[3]
                }
            return None

    def update_survey_response(self, survey_id: str, key: str, response: str):
        """Update survey response"""
        session = self.get_survey_session(survey_id)
        if session:
            responses = session['responses']
            responses[key] = response

            with self.get_db_connection() as conn:
                conn.execute('''
                    UPDATE surveys SET responses = ? WHERE id = ?
                ''', (json.dumps(responses), survey_id))
                conn.commit()

    def complete_survey(self, survey_id: str):
        """Mark survey as completed"""
        with self.get_db_connection() as conn:
            conn.execute('''
                UPDATE surveys SET status = 'completed' WHERE id = ?
            ''', (survey_id,))
            conn.commit()

    def generate_twiml_for_question(self, question: Dict[str, str], survey_id: str, question_index: int) -> str:
        """Generate TwiML for a specific question"""
        response = VoiceResponse()

        # Create a gather to collect speech
        gather = Gather(
            input='speech',
            action=f'https://handler.twilio.com/twiml/EH{survey_id}?question_index={question_index}',
            method='POST',
            speech_timeout='auto',
            timeout=10,
            language='en-US'
        )

        gather.say(question['prompt'], voice='alice')
        response.append(gather)

        # Fallback if no speech detected
        response.say("I didn't hear a response. Thank you for your time. Goodbye.", voice='alice')
        response.hangup()

        return str(response)

    def process_response_with_ai(self, question: str, raw_response: str) -> str:
        """Use CrewAI to process and clean the response"""
        try:
            task = Task(
                description=f"""
                Clean and extract key information from this phone survey response:

                Question: {question}
                Response: {raw_response}

                Provide a concise, clean answer that captures the essential information.
                Remove filler words and irrelevant information.
                If the response is unclear or empty, return "No clear response provided".
                """,
                agent=self.response_agent,
                expected_output="A clean, structured response"
            )

            crew = Crew(
                agents=[self.response_agent],
                tasks=[task],
                verbose=False
            )

            result = crew.kickoff()
            return str(result).strip()

        except Exception as e:
            print(f"AI processing error: {e}")
            return raw_response  # Return raw response if AI processing fails

    def make_survey_call(self, phone_number: str, questions: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Make a phone call to conduct a survey using Twilio Studio or TwiML Bins

        Args:
            phone_number: Phone number to call (format: +1234567890)
            questions: List of question dictionaries with 'key' and 'prompt'

        Returns:
            Dictionary with survey responses
        """

        # Create survey session
        survey_id = self.create_survey_session(phone_number, questions)

        # Create TwiML for the first question
        first_question_twiml = self.generate_twiml_for_question(questions[0], survey_id, 0)

        # Create a TwiML Bin for this survey (simplified approach)
        try:
            # For this example, we'll use a simpler approach with recorded responses
            # In production, you'd set up proper webhooks

            twiml_response = VoiceResponse()

            # Ask all questions in sequence with pauses for responses
            for i, question in enumerate(questions):
                twiml_response.say(f"Question {i + 1}: {question['prompt']}", voice='alice')
                twiml_response.pause(length=5)  # Give time for response
                twiml_response.say("Thank you for your response.", voice='alice')
                twiml_response.pause(length=1)

            twiml_response.say("Thank you for completing our survey. Goodbye!", voice='alice')
            twiml_response.hangup()

            # Make the call
            call = self.client.calls.create(
                twiml=str(twiml_response),
                to=phone_number,
                from_=self.twilio_phone
            )

            print(f"Call initiated: {call.sid}")
            print(f"Calling {phone_number}...")

            # Wait for call to complete
            max_wait = 120  # 2 minutes
            wait_time = 0

            while wait_time < max_wait:
                call = self.client.calls(call.sid).fetch()
                if call.status in ['completed', 'failed', 'canceled']:
                    break
                time.sleep(5)
                wait_time += 5

            # For demonstration, create mock responses
            # In production, these would come from actual call transcription
            mock_responses = {}
            for question in questions:
                mock_responses[question['key']] = f"Mock response for: {question['prompt']}"

            # Update database with mock responses
            with self.get_db_connection() as conn:
                conn.execute('''
                    UPDATE surveys SET responses = ?, status = 'completed' WHERE id = ?
                ''', (json.dumps(mock_responses), survey_id))
                conn.commit()

            return {
                'survey_id': survey_id,
                'call_sid': call.sid,
                'call_status': call.status,
                'responses': mock_responses
            }

        except Exception as e:
            print(f"Error making call: {e}")
            return {
                'survey_id': survey_id,
                'error': str(e),
                'responses': {}
            }

    def make_interactive_survey_call(self, phone_number: str, questions: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Alternative method using Twilio's Record verb for capturing responses
        """

        survey_id = self.create_survey_session(phone_number, questions)

        # Create TwiML that records responses
        twiml_response = VoiceResponse()

        twiml_response.say("Hello! Thank you for participating in our survey.", voice='alice')
        twiml_response.pause(length=1)

        # Ask each question and record response
        for i, question in enumerate(questions):
            twiml_response.say(question['prompt'], voice='alice')
            twiml_response.say("Please speak your answer after the beep.", voice='alice')

            # Record the response
            twiml_response.record(
                action=f'/handle_recording/{survey_id}/{question["key"]}',
                method='POST',
                max_length=30,
                finish_on_key='#',
                transcribe=True
            )

            twiml_response.say("Thank you.", voice='alice')
            twiml_response.pause(length=1)

        twiml_response.say("Thank you for completing our survey. Goodbye!", voice='alice')
        twiml_response.hangup()

        try:
            # Make the call
            call = self.client.calls.create(
                twiml=str(twiml_response),
                to=phone_number,
                from_=self.twilio_phone
            )

            print(f"Interactive call initiated: {call.sid}")
            return {
                'survey_id': survey_id,
                'call_sid': call.sid,
                'status': 'initiated'
            }

        except Exception as e:
            print(f"Error making interactive call: {e}")
            return {
                'survey_id': survey_id,
                'error': str(e)
            }

    def get_survey_results(self, survey_id: str) -> Dict[str, Any]:
        """Get results for a completed survey"""
        return self.get_survey_session(survey_id)

    def cleanup_database(self):
        """Clean up old survey data"""
        with self.get_db_connection() as conn:
            conn.execute('''
                DELETE FROM surveys 
                WHERE created_at < datetime('now', '-7 days')
            ''')
            conn.commit()


# Standalone function for simple usage
def make_phone_survey(account_sid: str, auth_token: str, twilio_phone: str,
                      target_phone: str, questions: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Simple function to make a phone survey call

    Args:
        account_sid: Twilio Account SID
        auth_token: Twilio Auth Token
        twilio_phone: Your Twilio phone number
        target_phone: Phone number to call
        questions: List of question dictionaries

    Returns:
        Dictionary with survey results
    """
    agent = PhoneSurveyAgent(account_sid, auth_token, twilio_phone)
    return agent.make_survey_call(target_phone, questions)


# Example usage
if __name__ == "__main__":
    TWILIO_SID = "SK1b378aed0c18f8a009c9dca483c0086e"
    TWILIO_TOKEN = "8sHkOfs9XbxwM0ldcXeYeItaZ56cGMII"
    TWILIO_FROM = "+17473640171"  # e.g. +17473640171
    TWILIO_ACCOUNT_SID = "SK1b378aed0c18f8a009c9dca483c0086e"
    TWILIO_AUTH_TOKEN = "8sHkOfs9XbxwM0ldcXeYeItaZ56cGMII"
    TWILIO_PHONE_NUMBER = "+17473640171"  # Your Twilio phone number
    WEBHOOK_URL = "https://b6681bc89ca7.ngrok-free.app"  # Use ngrok for local testing

    # Example questions
    QUESTIONS = [
        {"key": "name", "prompt": "Hi, may I have your full name?"},
        {"key": "needs", "prompt": "What service are you looking for?"},
        {"key": "slot", "prompt": "Is morning or afternoon better for you?"}
    ]

    # Simple usage
    target_phone = "+13478347434"  # Replace with actual number

    # Method 1: Using the class
    survey_agent = PhoneSurveyAgent(
        account_sid=TWILIO_ACCOUNT_SID,
        auth_token=TWILIO_AUTH_TOKEN,
        twilio_phone=TWILIO_PHONE_NUMBER
    )

    results = survey_agent.make_survey_call(target_phone, QUESTIONS)
    print("Survey Results:")
    print(json.dumps(results, indent=2))

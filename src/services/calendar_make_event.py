import datetime
import os.path
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.oauth2.gdch_credentials import ServiceAccountCredentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ['https://www.googleapis.com/auth/calendar']


def execute(title: str, description: str, start_time: str, end_time: str):
    """
    Creates event on google calendar. All times are in PST.
    :return: "Event created"
    """
    path = Path(__file__).resolve().parent.parent
    SCOPES = ['https://www.googleapis.com/auth/calendar']
    creds = None
    if os.path.exists(path / 'token.json'):
        creds = Credentials.from_authorized_user_file((path / 'token.json').__str__(), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file((path / 'gcsecret.json').__str__(), SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('calendar', 'v3', credentials=creds)
    event = {
      'summary': title,
      'description': description,
      'start': {
        'dateTime': start_time,
        'timeZone': 'America/Los_Angeles',
      },
      'end': {
        'dateTime': end_time,
        'timeZone': 'America/Los_Angeles',
      },
      'attendees': [
        {'email': 'dana.e.manor@gmail.com'},
        {'email': 'shay.manor@gmail.com'},
      ],
      'reminders': {
        'useDefault': True,
      },
    }

    event = service.events().insert(calendarId='primary', body=event).execute()
    return "Event created"

if __name__ == '__main__':
    execute("Winning Hackathon!!!", "We r gunna win", '2025-07-13T09:00:00-07:00', '2025-07-13T09:00:00-12:00')
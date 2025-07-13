import datetime
import os.path
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]


def execute(num_events: int):
    """
    Get next n events from google calendar
    :param num_events: Number of next events to get
    :return: Returns title and times of events
    """
    path = Path(__file__).resolve().parent.parent
    SCOPES = ['https://www.googleapis.com/auth/calendar']
    creds = None
    if os.path.exists(path / 'cal_token.json'):
        creds = Credentials.from_authorized_user_file((path / 'cal_token.json').__str__(), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file((path / 'cal_gcsecret.json').__str__(), SCOPES)
            creds = flow.run_local_server(port=0)
        with open('cal_token.json', 'w') as token:
            token.write(creds.to_json())


    try:
        service = build("calendar", "v3", credentials=creds)

        # Call the Calendar API
        now = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        print("Getting the upcoming 10 events")
        events_result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=now,
                maxResults=num_events,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        events = events_result.get("items", [])

        if not events:
            print("No upcoming events found.")
            return

        # Prints the start and name of the next 10 events
        for event in events:
            start = event["start"].get("dateTime", event["start"].get("date"))
            print(start, event["summary"])
        return events

    except HttpError as error:
        print(f"An error occurred: {error}")

if __name__ == "__main__":
    execute(5)

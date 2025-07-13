import base64
import os.path
from email.message import EmailMessage
from pathlib import Path

import google
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def execute(recipient: str, subject: str, content: str):
    """
    Sends email to given recipient.
    :param recipient: email address to send to
    :return: "Email sent"
    """

    path = Path(__file__).resolve().parent.parent
    SCOPES = ['https://mail.google.com/']
    creds = None
    if os.path.exists(path / 'mail_token.json'):
        creds = Credentials.from_authorized_user_file((path / 'mail_token.json').__str__(), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file((path / 'cal_gcsecret.json').__str__(), SCOPES)
            creds = flow.run_local_server(port=0)
        with open('../mail_token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build("gmail", "v1", credentials=creds)
        message = EmailMessage()

        message.set_content(content)

        message["To"] = recipient
        message["From"] = "shay.manor@gmail.com"
        message["Subject"] = subject

        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        create_message = {"raw": encoded_message}
        send_message = (
            service.users()
            .messages()
            .send(userId="me", body=create_message)
            .execute()
        )
        print(f'Message Id: {send_message["id"]}')
    except HttpError as error:
        print(f"An error occurred: {error}")
        send_message = None
    return send_message


if __name__ == '__main__':
    print(execute('shay.manor@gmail.com', 'Automated Test Email', 'This is an automated test email for WeaveHacks.'))

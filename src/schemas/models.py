from typing import List

from pydantic import BaseModel, Field


class CalendarCreateArgs(BaseModel):
    title: str
    summary: str
    start: str = Field(rexex=r".+Z$")
    end: str = Field(regex=r".+Z$")


class SearchWebArgs(BaseModel):
    question: str = Field(..., description="Plain-English web search query")


class StagehandArgs(BaseModel):
    prompt: str = Field(..., description="Detailed stage-direction prompt")


class Question(BaseModel):
    key: str = Field(..., description="Identifier for the answer")
    prompt: str = Field(..., description="Spoken question for the callee")


class CallArgs(BaseModel):
    questions: List[Question] = Field(
        ..., description="Structured IVR questions to ask sequentially"
    )


class CoffeeArgs(BaseModel):
    # no parameters
    pass


class EmailArgs(BaseModel):
    content: str
    subject: str
    recipient: str


tools = [{
    "name": "google_calendar_create",
    "description": "Create a calendar event in the user's Google Calendar",
    "parameters": CalendarCreateArgs.model_json_schema()
}]

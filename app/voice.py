"""Twilio voice webhook handlers (TwiML generation).

Point your Twilio phone number's voice webhook at POST /voice.
Twilio transcribes the caller's speech (Gather input="speech") and posts the
text back to the same endpoint as SpeechResult; we run it through the agent
and speak the answer back.
"""
from twilio.twiml.voice_response import Gather, VoiceResponse

from app.agent import run_agent

MAX_SPOKEN_CHARS = 500  # keep voice answers short


def greeting_twiml() -> str:
    """Initial greeting + speech capture."""
    response = VoiceResponse()
    gather = Gather(
        input="speech",
        action="/voice",
        method="POST",
        timeout=4,
        speech_timeout="auto",
        speech_model="phone_call",
    )
    gather.say("Hello! I am your A I business assistant. How can I help you today?")
    response.append(gather)
    response.say("I didn't catch that. Goodbye!")
    response.hangup()
    return str(response)


def answer_twiml(speech_text: str, caller_id: str = "voice") -> str:
    """Run the transcribed speech through the agent and speak the reply."""
    try:
        answer = run_agent(speech_text, session_id=f"voice-{caller_id}")
    except Exception:
        answer = "Sorry, I ran into a problem answering that. Please try again later."

    response = VoiceResponse()
    response.say(answer[:MAX_SPOKEN_CHARS])

    # Let the caller ask a follow-up question
    gather = Gather(
        input="speech",
        action="/voice",
        method="POST",
        timeout=4,
        speech_timeout="auto",
        speech_model="phone_call",
    )
    gather.say("Is there anything else I can help with?")
    response.append(gather)
    response.say("Thank you for calling. Goodbye!")
    response.hangup()
    return str(response)

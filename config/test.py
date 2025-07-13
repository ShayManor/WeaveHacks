import os
import time
import threading
import http.server
import socket
import sounddevice as sd
import soundfile as sf
import pychromecast
from google.cloud import speech, texttospeech

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DEVICE_NAME = "Living room"
RECORD_FS = 16000  # sampling rate
RECORD_CH = 1  # mono
RECORD_DUR = 5  # seconds per utterance
HTTP_PORT = 8080
API_ENDPOINT = "https://weavehacks-833206055650.us-central1.run.app/process"


# ── HTTP SERVER (for serving TTS files) ────────────────────────────────────────
def start_http_server():
    os.chdir(os.path.dirname(__file__))
    server = http.server.HTTPServer(('0.0.0.0', HTTP_PORT),
                                    http.server.SimpleHTTPRequestHandler)
    server.serve_forever()


server_thread = threading.Thread(target=start_http_server, daemon=True)
server_thread.start()


# ── AUDIO CAPTURE ─────────────────────────────────────────────────────────────
def record_to_file(filename: str, duration: float):
    """Record microphone audio to a WAV file."""
    print(f"[RECORD] Capturing {duration}s of audio...")
    data = sd.rec(int(duration * RECORD_FS), samplerate=RECORD_FS, channels=RECORD_CH)
    sd.wait()
    sf.write(filename, data, RECORD_FS)
    print(f"[RECORD] Saved to {filename}")


# ── SPEECH-TO-TEXT ─────────────────────────────────────────────────────────────
speech_client = speech.SpeechClient()


def transcribe_file(filename: str) -> str:
    with open(filename, "rb") as f:
        audio = speech.RecognitionAudio(content=f.read())
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RECORD_FS,
        language_code="en-US",
    )
    resp = speech_client.recognize(config=config, audio=audio)
    if not resp.results:
        return ""
    text = resp.results[0].alternatives[0].transcript
    print(f"[STT] Transcript: {text}")
    return text


# ── CALL YOUR API ─────────────────────────────────────────────────────────────
import requests


def call_my_api(query: str) -> str:
    return "Chicken nugget RAT"
    resp = requests.post(API_ENDPOINT, params={"prompt": query})
    print(f"Response: {resp.text}")
    reply = resp.json().get("response", "")
    print(f"[API] Reply JSON: {reply}")
    return reply


# ── TEXT-TO-SPEECH ─────────────────────────────────────────────────────────────
tts_client = texttospeech.TextToSpeechClient()


def synthesize_text(text: str, out_mp3: str) -> None:
    print(f"[TTS] Synthesizing speech for: {text}")
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    with open(out_mp3, "wb") as f:
        f.write(response.audio_content)
    print(f"[TTS] Written to {out_mp3}")


# ── CAST TO GOOGLE HOME ────────────────────────────────────────────────────────
def cast_mp3_to_google_home(mp3_path: str, device_name: str):
    chromecasts, _ = pychromecast.get_listed_chromecasts(friendly_names=[device_name])
    if not chromecasts:
        raise RuntimeError(f"No Chromecast named {device_name}")
    cast = chromecasts[0]
    cast.wait()
    local_ip = socket.gethostbyname(socket.gethostname())
    url = f"http://{local_ip}:{HTTP_PORT}/{mp3_path}"
    print(f"[CAST] Playing URL: {url}")
    mc = cast.media_controller
    mc.play_media(url, 'audio/mp3')
    mc.block_until_active()
    # Wait approximate duration
    duration = text_duration_seconds(mp3_path)
    time.sleep(duration + 0.5)


def text_duration_seconds(mp3_path: str) -> float:
    """Rough estimate: 150 words/minute → seconds."""
    # Fallback if you have no better way to get duration
    return os.path.getsize(mp3_path) / (50_000)  # ~50 KB/sec at 128kbps


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
def main():
    while True:
        # 1) Record user utterance
        wav_file = "input.wav"
        record_to_file(wav_file, RECORD_DUR)

        # 2) Transcribe
        query = transcribe_file(wav_file)
        if not query:
            print("[MAIN] No speech detected, retrying...")
            continue

        # 3) Call your API
        reply_text = call_my_api(query) or "Sorry, I didn't get that."

        # 4) Synthesize reply
        mp3_file = "response.mp3"
        synthesize_text(reply_text, mp3_file)

        # 5) Cast reply to Google Home
        cast_mp3_to_google_home(mp3_file, DEVICE_NAME)

        # 6) Cleanup
        os.remove(wav_file)
        os.remove(mp3_file)


if __name__ == "__main__":
    main()

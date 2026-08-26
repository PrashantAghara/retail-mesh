from groq import Groq

from app.domain.voice.constants import (
    STT_MODEL,
    TTS_MODEL,
    TTS_RESPONSE_FORMAT,
    TTS_VOICE,
)


def transcribe_audio(groq_client: Groq, audio_path: str) -> str:
    with open(audio_path, "rb") as audio_file:
        transcription = groq_client.audio.transcriptions.create(
            file=audio_file,
            model=STT_MODEL,
        )
    return transcription.text.strip()


def synthesize_speech(groq_client: Groq, text: str, output_path: str) -> str:
    speech_response = groq_client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        response_format=TTS_RESPONSE_FORMAT,
    )
    speech_response.write_to_file(output_path)
    return output_path

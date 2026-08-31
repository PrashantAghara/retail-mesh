import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from app.api.deps import get_voice_models
from app.domain.voice.model_registry import VoiceModelsContainer
from app.domain.voice.service import synthesize_speech, transcribe_audio

router = APIRouter(prefix="/voice", tags=["voice"])
TEMP_AUDIO_DIR = Path(tempfile.gettempdir())
MAX_FILE_SIZE = 25 * 1024 * 1024
ALLOWED_AUDIO_MIME_TYPES = {
    "audio/wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/webm",
    "audio/flac",
}


class TranscribeResponse(BaseModel):
    text: str


class SynthesizeRequest(BaseModel):
    text: str


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(...),
    voice_models: VoiceModelsContainer = Depends(get_voice_models),
):
    if file.content_type not in ALLOWED_AUDIO_MIME_TYPES:
        raise HTTPException(400, f"Unsupported audio MIME type '{file.content_type}'.")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File exceeds {MAX_FILE_SIZE // (1024 * 1024)}MB.")

    suffix = Path(file.filename).suffix or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        input_path = tmp.name

    try:
        text = transcribe_audio(voice_models.groq_client, input_path)
    finally:
        Path(input_path).unlink(missing_ok=True)

    return TranscribeResponse(text=text)


@router.post("/synthesize")
async def synthesize(
    request: SynthesizeRequest,
    voice_models: VoiceModelsContainer = Depends(get_voice_models),
):
    output_path = str(TEMP_AUDIO_DIR / f"voice_response_{uuid.uuid4().hex}.wav")
    synthesize_speech(voice_models.groq_client, request.text, output_path)

    return FileResponse(
        output_path,
        media_type="audio/wav",
        background=BackgroundTask(lambda: Path(output_path).unlink(missing_ok=True)),
    )

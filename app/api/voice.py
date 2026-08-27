import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.api.deps import get_supervisor_models, get_voice_models
from app.domain.supervisor.model_registry import SupervisorModelsContainer
from app.domain.voice.model_registry import VoiceModelsContainer
from app.domain.voice.service import synthesize_speech, transcribe_audio

router = APIRouter(prefix="/voice", tags=["voice"])

MAX_FILE_SIZE = 25 * 1024 * 1024
ALLOWED_AUDIO_MIME_TYPES = {
    "audio/wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/webm",
    "audio/flac",
}


@contextmanager
def temp_file(suffix: str = ""):
    """Context manager for temporary file with guaranteed cleanup."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            yield tmp
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


@contextmanager
def temp_output_file(suffix: str = ".wav"):
    """Context manager for temporary output file that persists until explicitly cleaned."""
    tmp_path = (
        Path(tempfile.gettempdir()) / f"voice_response_{uuid.uuid4().hex}{suffix}"
    )
    try:
        yield str(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/query")
async def query_voice(
    file: UploadFile = File(...),  # noqa: B008
    voice_models: VoiceModelsContainer = Depends(get_voice_models),  # noqa: B008
    supervisor_models: SupervisorModelsContainer = Depends(get_supervisor_models),  # noqa: B008
):
    suffix = Path(file.filename).suffix or ".wav"

    if file.content_type not in ALLOWED_AUDIO_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio MIME type '{file.content_type}'. Allowed: {', '.join(ALLOWED_AUDIO_MIME_TYPES)}",
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds maximum allowed ({MAX_FILE_SIZE // (1024 * 1024)}MB)",
        )

    with temp_file(suffix=suffix) as tmp:
        tmp.write(contents)
        tmp.flush()
        input_path = tmp.name

        transcribed_text = transcribe_audio(voice_models.groq_client, input_path)

        result = supervisor_models.graph.invoke(
            {
                "query": transcribed_text,
                "image_path": None,
                "category": None,
                "confidence": None,
                "method": None,
                "response": None,
                "needs_image": False,
            }
        )
        response_text = result["response"]

        with temp_output_file() as output_path:
            synthesize_speech(voice_models.groq_client, response_text, output_path)

            return FileResponse(
                output_path,
                media_type="audio/wav",
                headers={
                    "X-Transcribed-Query": transcribed_text,
                    "X-Category": result.get("category") or "",
                },
            )

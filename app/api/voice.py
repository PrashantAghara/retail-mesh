import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import FileResponse

from app.api.deps import get_supervisor_models, get_voice_models
from app.domain.supervisor.model_registry import SupervisorModelsContainer
from app.domain.voice.model_registry import VoiceModelsContainer
from app.domain.voice.service import synthesize_speech, transcribe_audio

router = APIRouter(prefix="/voice", tags=["voice"])

TEMP_AUDIO_DIR = Path(tempfile.gettempdir())


@router.post("/query")
async def query_voice(
    file: UploadFile = File(...),  # noqa: B008
    voice_models: VoiceModelsContainer = Depends(get_voice_models),  # noqa: B008
    supervisor_models: SupervisorModelsContainer = Depends(get_supervisor_models),  # noqa: B008
):
    suffix = Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        input_path = tmp.name

    try:
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

        output_path = str(TEMP_AUDIO_DIR / f"voice_response_{uuid.uuid4().hex}.wav")
        synthesize_speech(voice_models.groq_client, response_text, output_path)
    finally:
        Path(input_path).unlink(missing_ok=True)

    return FileResponse(
        output_path,
        media_type="audio/wav",
        headers={
            "X-Transcribed-Query": transcribed_text,
            "X-Category": result.get("category") or "",
        },
    )

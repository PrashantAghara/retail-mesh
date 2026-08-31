import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.api.deps import get_supervisor_models
from app.domain.supervisor.model_registry import SupervisorModelsContainer

router = APIRouter(prefix="/chat", tags=["supervisor"])

STEP_LABELS = {
    "classify_intent": "Analyzing your question…",
    "call_support": "Checking policy & product info…",
    "call_fulfillment": "Checking orders & inventory…",
    "call_vision": "Analyzing shelf image…",
}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/stream")
async def stream_chat(
    query: str = Form(...),
    file: UploadFile | None = File(None),
    models: SupervisorModelsContainer = Depends(get_supervisor_models),
):
    image_path = None

    if file is not None:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(400, f"Unsupported file type '{suffix}'.")
        if file.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(400, f"Unsupported MIME type '{file.content_type}'.")

        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                413, f"File exceeds {MAX_FILE_SIZE // (1024 * 1024)}MB."
            )

        # Write and fully close the handle here — do NOT delete yet.
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            image_path = tmp.name

    async def event_generator():
        initial_state = {
            "query": query,
            "image_path": image_path,
            "category": None,
            "confidence": None,
            "method": None,
            "response": None,
            "needs_image": False,
        }
        final_state = dict(initial_state)
        try:
            for step_output in models.graph.stream(initial_state):
                for node_name, node_state in step_output.items():
                    label = STEP_LABELS.get(node_name, node_name)
                    yield f"data: {json.dumps({'type': 'step', 'node': node_name, 'label': label})}\n\n"
                    final_state.update(node_state)

            done_event = {
                "type": "done",
                "category": final_state.get("category"),
                "response": final_state.get("response"),
                "needs_image": final_state.get("needs_image", False),
            }
            yield f"data: {json.dumps(done_event)}\n\n"
        finally:
            # Only deleted AFTER streaming (and thus Vision's file read) is fully done.
            if image_path:
                Path(image_path).unlink(missing_ok=True)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

import tempfile
from contextlib import contextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import get_vision_models
from app.domain.vision.model_registry import VisionModelsContainer
from app.domain.vision.schemas import ShelfQueryResponse

router = APIRouter(prefix="/vision", tags=["vision"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


@contextmanager
def temp_file(suffix: str):
    """Context manager for temporary file with guaranteed cleanup."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            yield tmp
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


@router.post("/query", response_model=ShelfQueryResponse)
async def query_vision(
    file: UploadFile = File(...),  # noqa: B008
    models: VisionModelsContainer = Depends(get_vision_models),  # noqa: B008
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported MIME type '{file.content_type}'. Allowed: {', '.join(ALLOWED_MIME_TYPES)}",
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
        result = models.agent.invoke(
            {"messages": [("user", f"Check the status of shelf image at {tmp.name}")]}
        )
        answer = result["messages"][-1].content

    return ShelfQueryResponse(image_path=file.filename, answer=answer)

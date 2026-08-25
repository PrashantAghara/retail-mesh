import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.deps import get_vision_models
from app.domain.vision.model_registry import VisionModelsContainer
from app.domain.vision.schemas import ShelfQueryResponse

router = APIRouter(prefix="/vision", tags=["vision"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@router.post("/query", response_model=ShelfQueryResponse)
async def query_vision(
    file: UploadFile = File(...),  # noqa: B008
    models: VisionModelsContainer = Depends(get_vision_models),  # noqa: B008
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return ShelfQueryResponse(
            image_path=file.filename,
            answer=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Save to a temp file — YOLO's predict() needs a real file path, not raw bytes in memory
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = models.agent.invoke(
            {"messages": [("user", f"Check the status of shelf image at {tmp_path}")]}
        )
        answer = result["messages"][-1].content
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return ShelfQueryResponse(image_path=file.filename, answer=answer)

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.deps import get_supervisor_models
from app.domain.supervisor.model_registry import SupervisorModelsContainer
from app.domain.supervisor.schemas import SupervisorResponse

router = APIRouter(prefix="/chat", tags=["supervisor"])

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


@router.post("/query", response_model=SupervisorResponse)
async def query_supervisor(
    query: str = Form(...),
    file: Optional[UploadFile] = File(None),
    models: SupervisorModelsContainer = Depends(get_supervisor_models),
):
    image_path = None

    if file is not None:
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
            image_path = tmp.name

            result = models.graph.invoke(
                {
                    "query": query,
                    "image_path": image_path,
                    "category": None,
                    "confidence": None,
                    "method": None,
                    "response": None,
                    "needs_image": False,
                }
            )
    else:
        result = models.graph.invoke(
            {
                "query": query,
                "image_path": None,
                "category": None,
                "confidence": None,
                "method": None,
                "response": None,
                "needs_image": False,
            }
        )

    return SupervisorResponse(
        query=query,
        category=result.get("category"),
        response=result["response"],
        needs_image=result.get("needs_image", False),
    )

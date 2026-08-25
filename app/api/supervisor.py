import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import get_supervisor_models
from app.domain.supervisor.model_registry import SupervisorModelsContainer
from app.domain.supervisor.schemas import SupervisorResponse

router = APIRouter(prefix="/chat", tags=["supervisor"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@router.post("/query", response_model=SupervisorResponse)
async def query_supervisor(
    query: str = Form(...),
    file: Optional[UploadFile] = File(None),
    models: SupervisorModelsContainer = Depends(get_supervisor_models),
):
    image_path = None
    tmp_path = None

    if file is not None:
        suffix = Path(file.filename).suffix.lower()
        if suffix in ALLOWED_EXTENSIONS:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                contents = await file.read()
                tmp.write(contents)
                tmp_path = tmp.name
                image_path = tmp_path

    try:
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
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    return SupervisorResponse(
        query=query,
        category=result.get("category"),
        response=result["response"],
        needs_image=result.get("needs_image", False),
    )

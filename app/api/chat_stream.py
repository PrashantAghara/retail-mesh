import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import get_supervisor_models
from app.domain.supervisor.model_registry import SupervisorModelsContainer
from app.domain.supervisor.schemas import SupervisorRequest

router = APIRouter(prefix="/chat", tags=["supervisor"])

STEP_LABELS = {
    "classify_intent": "Analyzing your question…",
    "call_support": "Checking policy & product info…",
    "call_fulfillment": "Checking orders & inventory…",
    "call_vision": "Analyzing shelf image…",
}


@router.post("/stream")
async def stream_chat(
    request: SupervisorRequest,
    models: SupervisorModelsContainer = Depends(get_supervisor_models),
):
    async def event_generator():
        initial_state = {
            "query": request.query,
            "image_path": request.image_path,
            "category": None,
            "confidence": None,
            "method": None,
            "response": None,
            "needs_image": False,
        }

        final_state = None
        for step_output in models.graph.stream(initial_state):
            for node_name, node_state in step_output.items():
                label = STEP_LABELS.get(node_name, node_name)
                event = {"type": "step", "node": node_name, "label": label}
                yield f"data: {json.dumps(event)}\n\n"
                final_state = (
                    {**initial_state, **node_state}
                    if final_state is None
                    else {**final_state, **node_state}
                )

        done_event = {
            "type": "done",
            "category": final_state.get("category") if final_state else None,
            "response": final_state.get("response") if final_state else None,
            "needs_image": final_state.get("needs_image", False)
            if final_state
            else False,
        }
        yield f"data: {json.dumps(done_event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

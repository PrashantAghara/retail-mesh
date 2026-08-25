from app.core.config import Settings
from app.domain.fulfillment.model_registry import FulfillmentModelsContainer
from app.domain.nlp.model_registry import NLPModelsContainer
from app.domain.nlp.service import get_intent
from app.domain.rag.model_registry import RAGModelsContainer
from app.domain.rag.service import support_agent
from app.domain.supervisor.schemas import SupervisorState
from app.domain.vision.model_registry import VisionModelsContainer


def build_nodes(
    nlp_models: NLPModelsContainer,
    rag_models: RAGModelsContainer,
    fulfillment_models: FulfillmentModelsContainer,
    vision_models: VisionModelsContainer,
    settings: Settings,
):
    """Nodes are built as closures over the already-loaded model containers —
    same dependency-discipline as the tool-building pattern used elsewhere."""

    def classify_intent_node(state: SupervisorState) -> dict:
        result = get_intent(state["query"], nlp_models, settings)
        return {
            "category": result.category,
            "confidence": result.confidence,
            "method": result.method,
        }

    def call_support_node(state: SupervisorState) -> dict:
        result = support_agent(state["query"], rag_models)
        return {"response": result.answer}

    def call_fulfillment_node(state: SupervisorState) -> dict:
        result = fulfillment_models.agent.invoke(
            {"messages": [("user", state["query"])]}
        )
        return {"response": result["messages"][-1].content}

    def call_vision_node(state: SupervisorState) -> dict:
        if not state.get("image_path"):
            return {
                "needs_image": True,
                "response": (
                    "This looks like a shelf/inventory question, but I don't have an image to check. "
                    "Please upload a shelf photo, or rephrase your question if you meant something else."
                ),
            }
        message = f"Check the status of shelf image at {state['image_path']}"
        result = vision_models.agent.invoke({"messages": [("user", message)]})
        return {"needs_image": False, "response": result["messages"][-1].content}

    def route_by_category(state: SupervisorState) -> str:
        category = state["category"]
        if category == "support":
            return "call_support"
        elif category == "fulfillment":
            return "call_fulfillment"
        elif category == "vision":
            return "call_vision"
        return "call_support"  # safe fallback

    return {
        "classify_intent": classify_intent_node,
        "call_support": call_support_node,
        "call_fulfillment": call_fulfillment_node,
        "call_vision": call_vision_node,
        "route_by_category": route_by_category,
    }

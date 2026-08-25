from dataclasses import dataclass

from langgraph.graph import END, StateGraph

from app.core.config import Settings
from app.core.exceptions import ModelLoadError
from app.core.models import SharedModels
from app.domain.fulfillment.model_registry import FulfillmentModelsContainer
from app.domain.nlp.model_registry import NLPModelsContainer
from app.domain.rag.model_registry import RAGModelsContainer
from app.domain.supervisor.nodes import build_nodes
from app.domain.supervisor.schemas import SupervisorState
from app.domain.vision.model_registry import VisionModelsContainer


@dataclass
class SupervisorModelsContainer:
    graph: object


def load_supervisor_models(
    settings: Settings,
    shared: SharedModels,
    nlp_models: NLPModelsContainer,
    rag_models: RAGModelsContainer,
    fulfillment_models: FulfillmentModelsContainer,
    vision_models: VisionModelsContainer,
) -> SupervisorModelsContainer:
    """Compose the four already-loaded domain agents into one routed graph.

    Unlike other domains, the Supervisor loads no models of its own — it
    only wires together containers that were already loaded concurrently.
    """
    try:
        nodes = build_nodes(
            nlp_models, rag_models, fulfillment_models, vision_models, settings
        )

        builder = StateGraph(SupervisorState)
        builder.add_node("classify_intent", nodes["classify_intent"])
        builder.add_node("call_support", nodes["call_support"])
        builder.add_node("call_fulfillment", nodes["call_fulfillment"])
        builder.add_node("call_vision", nodes["call_vision"])

        builder.set_entry_point("classify_intent")
        builder.add_conditional_edges(
            "classify_intent",
            nodes["route_by_category"],
            {
                "call_support": "call_support",
                "call_fulfillment": "call_fulfillment",
                "call_vision": "call_vision",
            },
        )
        builder.add_edge("call_support", END)
        builder.add_edge("call_fulfillment", END)
        builder.add_edge("call_vision", END)

        graph = builder.compile()
        return SupervisorModelsContainer(graph=graph)
    except Exception as e:
        raise ModelLoadError(f"Failed to build supervisor graph: {e}") from e

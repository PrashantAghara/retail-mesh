from langchain_core.prompts import ChatPromptTemplate
from ultralytics import YOLO

from app.core.models import SharedModels
from app.domain.vision.constants import SUMMARY_SYSTEM_PROMPT
from app.domain.vision.service import analyze_shelf


def build_vision_tools(shelf_model: YOLO, shared: SharedModels):
    from langchain_core.tools import tool

    summary_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SUMMARY_SYSTEM_PROMPT),
            (
                "human",
                "Shelf analysis:\n- Items detected: {num_items_detected}\n- Status: {status}",
            ),
        ]
    )
    summary_chain = summary_prompt | shared.llm

    @tool
    def check_shelf_status(image_path: str) -> str:
        """Analyze a shelf image to check stock status and detect low-stock gaps.
        Use this when asked to check inventory, shelf status, or stock levels from an image."""
        if not image_path or not image_path.strip():
            return "Image path is required."
        metrics = analyze_shelf(shelf_model, image_path)
        summary = summary_chain.invoke(metrics)
        return f"{summary.content} (Items detected: {metrics['num_items_detected']}, Status: {metrics['status']})"

    return [check_shelf_status]

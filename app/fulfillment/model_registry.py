from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_groq import ChatGroq

from app.core.config import Settings
from app.fulfillment.db import get_engine
from app.fulfillment.tools import build_fulfillment_tools

FULFILLMENT_SYSTEM_PROMPT = """You are RetailMesh's fulfillment assistant. You help customers check product
availability, place orders, track existing orders, and cancel orders. Always use the
provided tools to get real data — never guess stock levels or order statuses. Be concise
and friendly in your responses."""


@dataclass
class FulfillmentModels:
    agent: object


def load_fulfillment_models(settings: Settings) -> FulfillmentModels:
    engine = get_engine(settings.database_url)
    tools = build_fulfillment_tools(engine)
    llm = ChatGroq(api_key=settings.groq_api_key, model=settings.intent_llm_model)

    agent = create_agent(
        model=llm, tools=tools, system_prompt=FULFILLMENT_SYSTEM_PROMPT
    )
    return FulfillmentModels(agent=agent)

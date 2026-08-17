INTENT_EXAMPLES = {
    "support": [
        "What is your return policy?",
        "Do you offer a warranty on this product?",
        "How do I get a refund?",
        "What are your store hours?",
    ],
    "fulfillment": [
        "Is this item in stock?",
        "I want to place an order",
        "Where is my order?",
        "Can I cancel my order?",
    ],
    "vision": [
        "Check the shelf status in aisle 3",
        "Is there a stock gap on this shelf?",
        "Analyze this shelf image",
        "How full is the inventory display?",
    ],
}

SENTIMENT_LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}

INTENT_SYSTEM_PROMPT = """Classify the customer query into one category:
- support: questions about policies, returns, warranties, general help
- fulfillment: questions about orders, stock availability, placing/tracking orders
- vision: questions about shelf status, inventory display, physical store monitoring
Also determine the sentiment."""

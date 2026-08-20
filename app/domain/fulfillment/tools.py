import uuid
from typing import Any

from langchain_core.tools import tool
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.core.config import get_settings


def find_products(product_name: str, engine: Engine) -> list[tuple[Any, ...]]:
    """Find products matching the given name (partial match).
    
    Args:
        product_name: Product name to search for.
        engine: SQLAlchemy engine.
        
    Returns:
        List of product tuples (id, title, stock_count, price, aisle).
    """
    settings = get_settings()
    words = product_name.strip().split()
    if not words:
        return []
    
    # Build parameterized query safely
    placeholders = [f"title ILIKE :word{i}" for i in range(len(words))]
    conditions = " AND ".join(placeholders)
    params = {f"word{i}": f"%{w}%" for i, w in enumerate(words)}
    
    query = text(
        f"SELECT id, title, stock_count, price, aisle FROM products WHERE {conditions} LIMIT {settings.fulfillment_max_product_results}"
    )
    
    with engine.connect() as conn:
        return conn.execute(query, params).fetchall()


def _generate_order_id() -> str:
    """Generate a unique order ID using configured prefix and length."""
    settings = get_settings()
    return f"{settings.fulfillment_order_id_prefix}{uuid.uuid4().hex[:settings.fulfillment_order_id_length]}"


def build_fulfillment_tools(engine: Engine):
    """Build fulfillment tools with database engine closure.
    
    Args:
        engine: SQLAlchemy engine for database operations.
        
    Returns:
        List of LangChain tools for fulfillment operations.
    """
    non_cancellable_statuses = ("out_for_delivery", "delivered", "cancelled")

    @tool
    def check_stock(product_name: str) -> str:
        """Check current stock availability for a product by name (partial match supported).
        Use this when a customer asks if an item is in stock or available."""
        if not product_name or not product_name.strip():
            return "Product name is required."
            
        rows = find_products(product_name, engine)
        if not rows:
            return f"No product found matching '{product_name}'."
        lines = []
        for _, title, stock, price, aisle in rows:
            status = f"{stock} in stock" if stock > 0 else "OUT OF STOCK"
            lines.append(f"{title}: {status}, ₹{price}, located at {aisle}")
        return "\n".join(lines)

    @tool
    def place_order(product_name: str, quantity: int) -> str:
        """Place a new order for a product by name and quantity.
        Checks stock availability before confirming the order."""
        if not product_name or not product_name.strip():
            return "Product name is required."
        if quantity <= 0:
            return "Quantity must be positive."
            
        rows = find_products(product_name, engine)
        if not rows:
            return f"No product found matching '{product_name}'."
        product_id, title, stock, _, _ = rows[0]
        if stock < quantity:
            return f"Cannot place order — only {stock} units of '{title}' available, requested {quantity}."

        order_id = _generate_order_id()
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO orders (order_id, product_id, quantity, status) VALUES (:o, :p, :q, 'placed')"
                ),
                {"o": order_id, "p": product_id, "q": quantity},
            )
            conn.execute(
                text(
                    "UPDATE products SET stock_count = stock_count - :q WHERE id = :p"
                ),
                {"q": quantity, "p": product_id},
            )
        return f"Order placed! Order ID: {order_id} — {quantity}x {title}."

    @tool
    def track_order(order_id: str) -> str:
        """Look up the current status of an existing order by its order ID."""
        if not order_id or not order_id.strip():
            return "Order ID is required."
            
        with engine.connect() as conn:
            result = conn.execute(
                text("""SELECT o.status, o.quantity, p.title, o.created_at
                        FROM orders o JOIN products p ON o.product_id = p.id
                        WHERE o.order_id = :order_id"""),
                {"order_id": order_id},
            ).fetchone()
        if not result:
            return f"No order found with ID '{order_id}'."
        status, quantity, title, created_at = result
        return f"Order {order_id}: {quantity}x {title}, status = '{status}', placed on {created_at.strftime('%Y-%m-%d %H:%M')}."

    @tool
    def cancel_order(order_id: str) -> str:
        """Cancel an existing order by its order ID. Only orders not yet out for delivery can be cancelled."""
        if not order_id or not order_id.strip():
            return "Order ID is required."
            
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT status, product_id, quantity FROM orders WHERE order_id = :order_id"
                ),
                {"order_id": order_id},
            ).fetchone()
        if not result:
            return f"No order found with ID '{order_id}'."
        status, product_id, quantity = result
        if status in non_cancellable_statuses:
            return (
                f"Order {order_id} cannot be cancelled — current status is '{status}'."
            )
        with engine.begin() as conn:
            conn.execute(
                text("UPDATE orders SET status = 'cancelled' WHERE order_id = :o"),
                {"o": order_id},
            )
            conn.execute(
                text(
                    "UPDATE products SET stock_count = stock_count + :q WHERE id = :p"
                ),
                {"q": quantity, "p": product_id},
            )
        return f"Order {order_id} has been cancelled and refunded."

    return [check_stock, place_order, track_order, cancel_order]

import uuid
from sqlalchemy import text
from sqlalchemy.engine import Engine
from langchain_core.tools import tool

from app.fulfillment.db import find_products


def build_fulfillment_tools(engine: Engine):
    """Tools are built as a closure over `engine` so they don't rely on module-level
    globals — same dependency-discipline as the NLP/RAG layers."""

    @tool
    def check_stock(product_name: str) -> str:
        """Check current stock availability for a product by name (partial match supported).
        Use this when a customer asks if an item is in stock or available."""
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
        rows = find_products(product_name, engine)
        if not rows:
            return f"No product found matching '{product_name}'."
        product_id, title, stock, _, _ = rows[0]
        if stock < quantity:
            return f"Cannot place order — only {stock} units of '{title}' available, requested {quantity}."

        order_id = f"order_{uuid.uuid4().hex[:8]}"
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
        if status in ("out_for_delivery", "delivered", "cancelled"):
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

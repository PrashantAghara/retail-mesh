from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


def get_engine(database_url: str) -> Engine:
    return create_engine(database_url)


def find_products(product_name: str, engine: Engine):
    words = product_name.strip().split()
    if not words:
        return []
    conditions = " AND ".join([f"title ILIKE :word{i}" for i in range(len(words))])
    params = {f"word{i}": f"%{w}%" for i, w in enumerate(words)}
    with engine.connect() as conn:
        return conn.execute(
            text(
                f"SELECT id, title, stock_count, price, aisle FROM products WHERE {conditions} LIMIT 3"
            ),
            params,
        ).fetchall()

from pathlib import Path
import sqlite3
import uuid
from contextlib import contextmanager

DB_PATH = Path(__file__).parent / "tenants.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tenants (
                id        TEXT PRIMARY KEY,
                api_key   TEXT UNIQUE NOT NULL,
                name      TEXT,
                faq_path  TEXT
            )
        """)
        conn.commit()

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()

def create_tenant(name: str) -> dict:
    api_key = "pk_" + uuid.uuid4().hex         # public key
    tid     = uuid.uuid4().hex
    faq_path = f"data/{tid}/faq.json"          # default location
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO tenants (id, api_key, name, faq_path) VALUES (?,?,?,?)",
            (tid, api_key, name, faq_path),
        )
        conn.commit()
    return {"id": tid, "api_key": api_key, "faq_path": faq_path}

def get_tenant(api_key: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, faq_path FROM tenants WHERE api_key = ?",
            (api_key,),
        ).fetchone()
    if row:
        return {"id": row[0], "faq_path": row[1]}
    return None

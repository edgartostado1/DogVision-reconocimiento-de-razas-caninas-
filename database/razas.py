"""
Módulo de acceso a datos de razas (SQLite).
"""
import sqlite3
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "razas.db"


def obtener_raza(clave: str) -> Optional[dict]:
    """
    Busca una raza por su clave (ej: 'golden_retriever').
    Devuelve el dict de la ficha, o None si no existe. La clave coincide
    con los nombres en CATEGORIAS de modelo/red_neuronal.py.
    """
    if not DB_PATH.exists():
        return None

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.row_factory = sqlite3.Row  # permite dict(row) en vez de tupla posicional
        row = conn.execute(
            "SELECT * FROM razas WHERE clave = ?", (clave,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def obtener_todas() -> list:
    """Devuelve todas las razas ordenadas por nombre."""
    if not DB_PATH.exists():
        return []

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.row_factory = sqlite3.Row  # permite dict(row) en vez de tupla posicional
        rows = conn.execute("SELECT * FROM razas ORDER BY nombre;").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
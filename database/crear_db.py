"""
Crea la base de datos SQLite 'razas.db' a partir de 'razas.json'.
Ejecutar una sola vez o cuando cambien los datos:
    python -m database.crear_db
"""
import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "razas.db"
JSON_PATH = Path(__file__).parent / "razas.json"


def crear_tabla(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS razas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clave TEXT UNIQUE NOT NULL,
            nombre TEXT NOT NULL,
            origen TEXT,
            tamano TEXT,
            peso TEXT,
            esperanza_vida TEXT,
            temperamento TEXT,
            descripcion TEXT
        );
    """)


def main():
    if not JSON_PATH.exists():
        print(f"ERROR: No se encontró {JSON_PATH}")
        return

    with open(JSON_PATH, encoding="utf-8") as f:
        razas = json.load(f)

    conn = sqlite3.connect(DB_PATH)
    try:
        crear_tabla(conn)
        # Borrar todo antes de insertar para que sea re-ejecutable sin duplicados
        conn.execute("DELETE FROM razas;")

        conn.executemany(
            """
            INSERT INTO razas (clave, nombre, origen, tamano, peso, esperanza_vida, temperamento, descripcion)
            VALUES (:clave, :nombre, :origen, :tamano, :peso, :esperanza_vida, :temperamento, :descripcion);
            """,
            razas
        )
        conn.commit()
        print(f"OK -> {DB_PATH} creada con {len(razas)} razas.")

        total = conn.execute("SELECT COUNT(*) FROM razas;").fetchone()[0]
        print(f"Verificación: {total} registros en la tabla 'razas'.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
import os
import sqlite3
from contextlib import contextmanager
from typing import Iterator, List, Optional, Tuple


DB_FILENAME = "face_diet.sqlite3"


def get_db_path(database_path: Optional[str] = None) -> str:
    if database_path:
        return database_path
    return os.path.join(os.path.dirname(__file__), DB_FILENAME)


@contextmanager
def connect(database_path: Optional[str] = None) -> Iterator[sqlite3.Connection]:
    db_path = get_db_path(database_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
        conn.commit()
    finally:
        conn.close()


def initialize_schema(database_path: Optional[str] = None) -> None:
    with connect(database_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                width INTEGER,
                height INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                w INTEGER NOT NULL,
                h INTEGER NOT NULL,
                score REAL,
                embedding BLOB,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
            );
            """
        )


def upsert_image(
    image_path: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    database_path: Optional[str] = None,
) -> int:
    with connect(database_path) as conn:
        cur = conn.execute(
            "INSERT INTO images(path, width, height) VALUES (?, ?, ?)\n"
            "ON CONFLICT(path) DO UPDATE SET width=excluded.width, height=excluded.height\n"
            "RETURNING id;",
            (image_path, width, height),
        )
        row = cur.fetchone()
    return int(row[0])


def insert_faces(
    image_id: int,
    boxes: List[Tuple[int, int, int, int]],
    scores: Optional[List[float]] = None,
    embeddings: Optional[List[bytes]] = None,
    database_path: Optional[str] = None,
) -> List[int]:
    inserted_ids: List[int] = []
    with connect(database_path) as conn:
        for idx, (x, y, w, h) in enumerate(boxes):
            score = scores[idx] if scores is not None and idx < len(scores) else None
            embedding = embeddings[idx] if embeddings is not None and idx < len(embeddings) else None
            cur = conn.execute(
                "INSERT INTO faces(image_id, x, y, w, h, score, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)\n"
                "RETURNING id;",
                (image_id, x, y, w, h, score, embedding),
            )
            row = cur.fetchone()
            inserted_ids.append(int(row[0]))
    return inserted_ids


def list_images(database_path: Optional[str] = None) -> List[Tuple[int, str]]:
    with connect(database_path) as conn:
        cur = conn.execute("SELECT id, path FROM images ORDER BY created_at DESC;")
        return [(int(r[0]), str(r[1])) for r in cur.fetchall()]


def list_faces_for_image(image_id: int, database_path: Optional[str] = None) -> List[Tuple[int, int, int, int, float]]:
    with connect(database_path) as conn:
        cur = conn.execute(
            "SELECT x, y, w, h, COALESCE(score, 0.0) FROM faces WHERE image_id = ? ORDER BY id ASC;",
            (image_id,),
        )
        rows = cur.fetchall()
        result: List[Tuple[int, int, int, int, float]] = []
        for x, y, w, h, score in rows:
            result.append((int(x), int(y), int(w), int(h), float(score)))
        return result



"""
Batch compute and populate 384-dimensional dense embeddings for all listings in PostgreSQL.
Uses sentence-transformers/all-MiniLM-L6-v2 on Apple Silicon MPS / CPU.
"""

import sys
import time
import logging
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from db.connection import get_db_connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("populate_embeddings")


def build_listing_text(row: dict) -> str:
    title = row.get("title") or ""
    locality = row.get("locality") or ""
    city = row.get("city") or ""
    bedrooms = row.get("bedrooms") or ""
    ptype = row.get("property_type") or ""
    furnishing = row.get("furnishing_status") or ""
    facing = row.get("facing_direction") or "Not specified"
    desc = row.get("description") or ""
    return (
        f"{title} in {locality}, {city}. "
        f"{bedrooms} BHK {ptype}, {furnishing}. Facing: {facing}. {desc}"
    ).strip()


def populate_embeddings():
    logger.info("Initializing SentenceTransformer on available device...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using compute device: {device}")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, title, locality, city, bedrooms, property_type, furnishing_status, facing_direction, description
                FROM listings
                WHERE embedding IS NULL
                ORDER BY id ASC;
            """)
            rows = cur.fetchall()

    total = len(rows)
    logger.info(f"Found {total} listings needing embeddings.")
    if total == 0:
        logger.info("All listings already have embeddings populated!")
        return

    batch_size = 256
    start_time = time.time()
    updated = 0

    for i in range(0, total, batch_size):
        batch = rows[i : i + batch_size]
        texts = [build_listing_text(r) for r in batch]
        ids = [r["id"] for r in batch]

        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)

        # Batch update into PostgreSQL
        update_data = [
            (f"[{','.join(f'{x:.6f}' for x in emb)}]", row_id)
            for emb, row_id in zip(embeddings, ids)
        ]

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "UPDATE listings SET embedding = %s::vector WHERE id = %s;",
                    update_data,
                )

        updated += len(batch)
        elapsed = time.time() - start_time
        rate = updated / elapsed
        if updated % 1024 == 0 or updated == total:
            logger.info(f"Populated {updated}/{total} embeddings ({rate:.1f} listings/sec, elapsed: {elapsed:.1f}s)")

    logger.info(f"Successfully finished embedding all {updated} listings in {time.time() - start_time:.2f} seconds!")


if __name__ == "__main__":
    populate_embeddings()

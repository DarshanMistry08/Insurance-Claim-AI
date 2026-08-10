"""
Load and embed the synthetic policy document into Postgres.

Reads data/policy_specimen.txt, splits it into paragraphs (chunks),
embeds them using sentence-transformers, and inserts them into pgvector.

Usage:
    python scripts/load_policy.py
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed; rely on env vars being set externally

import psycopg2
from rich.console import Console
from sentence_transformers import SentenceTransformer

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
console = Console(force_terminal=True, highlight=False)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = PROJECT_ROOT / "data" / "policy_specimen.txt"

# Database connection
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "claims_db",
    "user": "claims",
    "password": os.getenv("DB_PASSWORD", ""),
}

def main():
    console.rule("[bold blue]Policy Document Loader (RAG Setup)[/bold blue]")

    if not POLICY_PATH.exists():
        console.print(f"[red]Policy file not found:[/red] {POLICY_PATH}")
        return

    with open(POLICY_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    # Simple paragraph chunking
    chunks = [c.strip() for c in text.split("\n\n") if c.strip() and not c.startswith("## ")]
    console.print(f"[OK] Extracted [green]{len(chunks)}[/green] chunks from policy document.")

    # Initialize embedding model (MiniLM is small and fast)
    console.print("[>>] Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    console.print("[>>] Computing embeddings...")
    embeddings = model.encode(chunks)

    # Insert into database
    console.print(f"[DB] Connecting to Postgres at {DB_CONFIG['host']}:{DB_CONFIG['port']}...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = False
        cursor = conn.cursor()
        
        # 1. Create document record
        cursor.execute(
            """INSERT INTO documents (source_dataset, file_path, doc_type)
               VALUES ('synthetic', 'policy_specimen.txt', 'policy')
               RETURNING document_id"""
        )
        doc_id = cursor.fetchone()[0]

        # 2. Insert chunks
        inserted = 0
        for i, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
            # convert numpy array to list for pgvector
            emb_list = emb.tolist()
            cursor.execute(
                """INSERT INTO chunks (document_id, chunk_index, chunk_text, embedding)
                   VALUES (%s, %s, %s, %s)""",
                (doc_id, i, chunk_text, emb_list)
            )
            inserted += 1

        conn.commit()
        console.print(f"[OK] Successfully inserted {inserted} chunks into Postgres.")

    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        console.print(f"[red]Database error:[/red] {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()

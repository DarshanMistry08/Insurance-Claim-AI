"""
Seed the golden_labels table in Postgres from data/golden_set.csv

Reads the hand-labeled CSV, validates field names against SPEC.md,
and inserts rows into the database.

Usage:
    python scripts/seed_golden_set.py
    python scripts/seed_golden_set.py --csv data/golden_set.csv --dry-run
"""

import os
import csv
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed; rely on env vars being set externally

import click
import psycopg2
from rich.console import Console
from rich.table import Table

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
console = Console(force_terminal=True, highlight=False)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Valid field names from SPEC.md
VALID_FIELDS = {
    "claimant_name",
    "policy_number",
    "incident_date",
    "claim_filing_date",
    "incident_type",
    "incident_description",
    "amount_claimed",
    "claimant_address",
    "adjuster_name",
    "supporting_doc_count",
}

# Database connection defaults (match docker-compose.yml)
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "claims_db",
    "user": "claims",
    "password": os.getenv("DB_PASSWORD", ""),
}


def validate_csv(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Validate CSV rows against SPEC.md field definitions."""
    valid_rows = []
    errors = []

    for i, row in enumerate(rows, start=2):  # row 1 is header
        doc_id = row.get("doc_id", "").strip()
        field_name = row.get("field_name", "").strip()
        value = row.get("ground_truth_value", "").strip()

        if not doc_id:
            errors.append(f"Row {i}: missing doc_id")
            continue

        if not field_name:
            errors.append(f"Row {i}: missing field_name")
            continue

        if field_name not in VALID_FIELDS:
            errors.append(f"Row {i}: unknown field '{field_name}' (valid: {', '.join(sorted(VALID_FIELDS))})")
            continue

        # Skip rows with no ground truth value (not yet labeled)
        if not value:
            continue

        valid_rows.append({
            "doc_id": doc_id,
            "field_name": field_name,
            "ground_truth_value": value,
            "confidence_note": row.get("confidence_note", "").strip() or None,
        })

    return valid_rows, errors


def ensure_document_exists(cursor, doc_id: str) -> int:
    """Get or create a document record, return document_id."""
    cursor.execute(
        "SELECT document_id FROM documents WHERE file_path = %s",
        (doc_id,)
    )
    result = cursor.fetchone()

    if result:
        return result[0]

    # Create a placeholder document record
    cursor.execute(
        """INSERT INTO documents (source_dataset, file_path, doc_type, metadata)
           VALUES ('golden_set', %s, 'claim_form', '{}')
           RETURNING document_id""",
        (doc_id,)
    )
    return cursor.fetchone()[0]


@click.command()
@click.option("--csv-path", default=None, type=click.Path(exists=True),
              help="Path to golden_set.csv (default: data/golden_set.csv)")
@click.option("--dry-run", is_flag=True, help="Validate only, don't insert into DB")
def main(csv_path: str | None, dry_run: bool):
    """Load golden set labels from CSV into Postgres."""
    console.rule("[bold blue]Golden Set Seeder[/bold blue]")

    csv_file = Path(csv_path) if csv_path else PROJECT_ROOT / "data" / "golden_set.csv"

    if not csv_file.exists():
        console.print(f"[red]CSV not found:[/red] {csv_file}")
        sys.exit(1)

    # Read CSV
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    console.print(f"[DOC] Read [green]{len(rows)}[/green] rows from [cyan]{csv_file}[/cyan]")

    # Validate
    valid_rows, errors = validate_csv(rows)

    if errors:
        console.print(f"\n[yellow]WARNING: {len(errors)} validation warnings:[/yellow]")
        for err in errors[:10]:
            console.print(f"  - {err}")
        if len(errors) > 10:
            console.print(f"  - ... and {len(errors) - 10} more")

    if not valid_rows:
        console.print("\n[yellow]No labeled rows found (ground_truth_value is empty for all rows).[/yellow]")
        console.print("Label some documents in data/golden_set.csv first!")
        sys.exit(0)

    # Show preview
    preview_table = Table(title=f"Labels to Insert ({len(valid_rows)} rows)")
    preview_table.add_column("Doc ID", style="cyan")
    preview_table.add_column("Field", style="white")
    preview_table.add_column("Value", style="green", max_width=40)
    preview_table.add_column("Note", style="dim")

    for row in valid_rows[:15]:
        preview_table.add_row(
            row["doc_id"],
            row["field_name"],
            row["ground_truth_value"][:40],
            row["confidence_note"] or "",
        )
    if len(valid_rows) > 15:
        preview_table.add_row("...", "...", f"({len(valid_rows) - 15} more)", "")

    console.print(preview_table)

    if dry_run:
        console.print("\n[yellow]DRY RUN -- no database changes made.[/yellow]")
        return

    # Insert into Postgres
    console.print(f"\n[DB] Connecting to Postgres at [cyan]{DB_CONFIG['host']}:{DB_CONFIG['port']}[/cyan]...")

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = False
        cursor = conn.cursor()
    except psycopg2.OperationalError as e:
        console.print(f"[red]Cannot connect to Postgres:[/red] {e}")
        console.print("[yellow]Is Docker running? Try: docker-compose up -d[/yellow]")
        sys.exit(1)

    inserted = 0
    skipped = 0

    try:
        for row in valid_rows:
            doc_db_id = ensure_document_exists(cursor, row["doc_id"])

            try:
                cursor.execute(
                    """INSERT INTO golden_labels (document_id, field_name, ground_truth_value, confidence_note)
                       VALUES (%s, %s, %s, %s)
                       ON CONFLICT (document_id, field_name)
                       DO UPDATE SET ground_truth_value = EXCLUDED.ground_truth_value,
                                     confidence_note = EXCLUDED.confidence_note""",
                    (doc_db_id, row["field_name"], row["ground_truth_value"], row["confidence_note"])
                )
                inserted += 1
            except Exception as e:
                console.print(f"[red]Error inserting {row['doc_id']}/{row['field_name']}:[/red] {e}")
                skipped += 1

        conn.commit()
    except Exception as e:
        conn.rollback()
        console.print(f"[red]Transaction failed, rolled back:[/red] {e}")
        sys.exit(1)
    finally:
        cursor.close()
        conn.close()

    # Final summary
    console.print()
    summary = Table(title="Seed Results")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Count", style="green", justify="right")
    summary.add_row("Inserted/updated", str(inserted))
    summary.add_row("Skipped (errors)", str(skipped))
    summary.add_row("Total valid rows", str(len(valid_rows)))
    console.print(summary)

    console.print("\n[OK] Golden set seeded successfully!")
    console.print("   Verify: [cyan]docker exec -it claims-postgres psql -U claims -d claims_db -c \"SELECT COUNT(*) FROM golden_labels;\"[/cyan]")


if __name__ == "__main__":
    main()

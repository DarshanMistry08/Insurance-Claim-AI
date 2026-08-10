"""
Inspect downloaded datasets — browse documents, view annotations, print stats.

Usage:
    python scripts/inspect_dataset.py --dataset funsd --limit 5
    python scripts/inspect_dataset.py --dataset docvqa --limit 10
    python scripts/inspect_dataset.py --dataset funsd --doc-id funsd_train_0003
"""

import os
import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
console = Console(force_terminal=True, highlight=False)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def inspect_funsd(limit: int, doc_id: str | None):
    """Inspect FUNSD dataset."""
    funsd_dir = DATA_DIR / "funsd"
    images_dir = funsd_dir / "images"
    annotations_dir = funsd_dir / "annotations"

    if not annotations_dir.exists():
        console.print("[red]FUNSD not found.[/red] Run: python scripts/pull_funsd.py")
        return

    annotation_files = sorted(annotations_dir.glob("*.json"))

    if doc_id:
        annotation_files = [f for f in annotation_files if doc_id in f.stem]
        if not annotation_files:
            console.print(f"[red]Document '{doc_id}' not found.[/red]")
            return

    annotation_files = annotation_files[:limit]

    console.rule("[bold blue]FUNSD Dataset Inspector[/bold blue]")
    console.print(f"[DIR] Data dir: [cyan]{funsd_dir}[/cyan]")
    console.print(f"[DOC] Showing [green]{len(annotation_files)}[/green] documents\n")

    # Summary table
    table = Table(title="Document Overview")
    table.add_column("Doc ID", style="cyan")
    table.add_column("Image", style="dim")
    table.add_column("Entities", justify="right", style="green")
    table.add_column("Headers", justify="right")
    table.add_column("Questions", justify="right")
    table.add_column("Answers", justify="right")
    table.add_column("Other", justify="right")

    for ann_file in annotation_files:
        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        doc = data["doc_id"]
        entities = data.get("entities", [])

        # Count by tag type
        counts = {}
        for ent in entities:
            tag = ent.get("tag", "O")
            base_tag = tag.split("-")[-1] if "-" in tag else tag
            counts[base_tag] = counts.get(base_tag, 0) + 1

        image_path = images_dir / f"{doc}.png"
        image_exists = "Y" if image_path.exists() else "N"

        table.add_row(
            doc,
            image_exists,
            str(len(entities)),
            str(counts.get("HEADER", 0)),
            str(counts.get("QUESTION", 0)),
            str(counts.get("ANSWER", 0)),
            str(counts.get("O", 0)),
        )

    console.print(table)

    # Show detailed view of first doc
    if annotation_files:
        with open(annotation_files[0], "r", encoding="utf-8") as f:
            data = json.load(f)

        console.print()
        console.rule(f"[bold yellow]Detail: {data['doc_id']}[/bold yellow]")

        # Show first 15 entities
        detail_table = Table(title=f"Entities (first 15 of {data['num_entities']})")
        detail_table.add_column("Word", style="white")
        detail_table.add_column("Tag", style="cyan")
        detail_table.add_column("BBox", style="dim")

        for ent in data["entities"][:15]:
            detail_table.add_row(
                ent["word"],
                ent["tag"],
                str(ent["bbox"]),
            )

        console.print(detail_table)
        image_path = images_dir / f"{data['doc_id']}.png"
        console.print(f"\n[IMG] Image: [link=file:///{image_path}]{image_path}[/link]")


def inspect_docvqa(limit: int, doc_id: str | None):
    """Inspect DocVQA dataset."""
    docvqa_dir = DATA_DIR / "docvqa"
    images_dir = docvqa_dir / "images"
    qa_dir = docvqa_dir / "qa_pairs"

    if not qa_dir.exists():
        console.print("[red]DocVQA not found.[/red] Run: python scripts/pull_docvqa.py")
        return

    qa_files = sorted(qa_dir.glob("*.json"))

    if doc_id:
        qa_files = [f for f in qa_files if doc_id in f.stem]
        if not qa_files:
            console.print(f"[red]Document '{doc_id}' not found.[/red]")
            return

    qa_files = qa_files[:limit]

    console.rule("[bold blue]DocVQA Dataset Inspector[/bold blue]")
    console.print(f"[DIR] Data dir: [cyan]{docvqa_dir}[/cyan]")
    console.print(f"[DOC] Showing [green]{len(qa_files)}[/green] documents\n")

    table = Table(title="QA Pairs Overview")
    table.add_column("Doc ID", style="cyan")
    table.add_column("Image", style="dim")
    table.add_column("Question", style="white", max_width=50)
    table.add_column("Answer(s)", style="green", max_width=30)

    for qa_file in qa_files:
        with open(qa_file, "r", encoding="utf-8") as f:
            qa = json.load(f)

        image_path = images_dir / f"{qa['doc_id']}.png"
        image_exists = "Y" if image_path.exists() else "N"

        answers_str = ", ".join(qa.get("answers", [])[:2])
        if len(qa.get("answers", [])) > 2:
            answers_str += "…"

        table.add_row(
            qa["doc_id"],
            image_exists,
            qa.get("question", "N/A")[:50],
            answers_str[:30],
        )

    console.print(table)


@click.command()
@click.option("--dataset", required=True, type=click.Choice(["funsd", "docvqa"]),
              help="Which dataset to inspect")
@click.option("--limit", default=10, type=int, help="Max documents to show")
@click.option("--doc-id", default=None, type=str, help="Specific document ID to inspect")
def main(dataset: str, limit: int, doc_id: str | None):
    """Browse downloaded dataset documents, annotations, and stats."""
    if dataset == "funsd":
        inspect_funsd(limit, doc_id)
    elif dataset == "docvqa":
        inspect_docvqa(limit, doc_id)
    else:
        console.print(f"[red]Unknown dataset: {dataset}[/red]")


if __name__ == "__main__":
    main()

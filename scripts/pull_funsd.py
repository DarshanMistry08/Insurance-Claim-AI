"""
Pull FUNSD dataset from HuggingFace and save to data/funsd/

FUNSD contains 199 scanned form images with semantic entity annotations
(header, question, answer, other) — the closest public proxy to insurance forms.

Usage:
    python scripts/pull_funsd.py
    python scripts/pull_funsd.py --limit 20
"""

import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.progress import track

# Force UTF-8 to avoid Windows cp1252 emoji issues
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
console = Console(force_terminal=True, highlight=False)

# Project root = parent of scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "funsd"


@click.command()
@click.option("--limit", default=None, type=int, help="Max number of samples to download (default: all)")
@click.option("--split", default="train", type=click.Choice(["train", "test"]), help="Dataset split")
def main(limit: int | None, split: str):
    """Download FUNSD dataset and save images + annotations locally."""
    console.rule("[bold blue]FUNSD Dataset Pull[/bold blue]")

    # Lazy import so --help is fast
    from datasets import load_dataset

    console.print(f"[>>] Loading FUNSD split=[cyan]{split}[/cyan] from HuggingFace...")

    try:
        ds = load_dataset("nielsr/funsd", split=split)
    except Exception as e:
        console.print(f"[red]Failed to load dataset:[/red] {e}")
        console.print("[yellow]Try: huggingface-cli login[/yellow]")
        sys.exit(1)

    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    console.print(f"[OK] Loaded [green]{len(ds)}[/green] samples")
    console.print(f"[DIR] Saving to [cyan]{DATA_DIR}[/cyan]")

    # Create output dirs
    images_dir = DATA_DIR / "images"
    annotations_dir = DATA_DIR / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # Track annotation stats
    total_entities = 0
    label_counts = {}

    for i, sample in enumerate(track(ds, description="Saving documents...")):
        doc_id = f"funsd_{split}_{i:04d}"

        # Save image
        image = sample["image"]
        image_path = images_dir / f"{doc_id}.png"
        image.save(str(image_path))

        # Parse and save annotations
        words = sample.get("words", [])
        bboxes = sample.get("bboxes", [])
        ner_tags = sample.get("ner_tags", [])

        # FUNSD NER tag mapping
        tag_map = {0: "O", 1: "B-HEADER", 2: "I-HEADER", 3: "B-QUESTION",
                   4: "I-QUESTION", 5: "B-ANSWER", 6: "I-ANSWER"}

        entities = []
        for word, bbox, tag_id in zip(words, bboxes, ner_tags):
            tag_name = tag_map.get(tag_id, f"UNK-{tag_id}")
            entities.append({
                "word": word,
                "bbox": bbox,
                "tag": tag_name
            })
            label_counts[tag_name] = label_counts.get(tag_name, 0) + 1

        total_entities += len(entities)

        # Save annotation as JSON
        import json
        annotation_path = annotations_dir / f"{doc_id}.json"
        with open(annotation_path, "w", encoding="utf-8") as f:
            json.dump({
                "doc_id": doc_id,
                "source": "funsd",
                "split": split,
                "num_entities": len(entities),
                "entities": entities
            }, f, indent=2)

    # Summary table
    console.print()
    console.rule("[bold green]Summary[/bold green]")

    table = Table(title="FUNSD Download Stats")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Split", split)
    table.add_row("Documents saved", str(len(ds)))
    table.add_row("Total entities", str(total_entities))
    table.add_row("Images dir", str(images_dir))
    table.add_row("Annotations dir", str(annotations_dir))

    console.print(table)

    # Label distribution
    if label_counts:
        label_table = Table(title="Entity Label Distribution")
        label_table.add_column("Label", style="cyan")
        label_table.add_column("Count", style="green", justify="right")
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            label_table.add_row(label, str(count))
        console.print(label_table)

    console.print(f"\n[OK] Done! Inspect with: [cyan]python scripts/inspect_dataset.py --dataset funsd --limit 5[/cyan]")


if __name__ == "__main__":
    main()

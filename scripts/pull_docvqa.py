"""
Pull DocVQA dataset (subset) from HuggingFace and save to data/docvqa/

DocVQA contains document images with question-answer pairs — useful for
evaluating document understanding and field extraction capabilities.

Usage:
    python scripts/pull_docvqa.py
    python scripts/pull_docvqa.py --limit 50
"""

import os
import sys
import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.progress import track

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
console = Console(force_terminal=True, highlight=False)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "docvqa"


@click.command()
@click.option("--limit", default=50, type=int, help="Max samples to download (default: 50)")
@click.option("--split", default="validation", type=click.Choice(["train", "validation", "test"]),
              help="Dataset split (default: validation — smaller and labeled)")
def main(limit: int, split: str):
    """Download a subset of DocVQA and save images + QA pairs locally."""
    console.rule("[bold blue]DocVQA Dataset Pull[/bold blue]")

    from datasets import load_dataset

    console.print(f"[>>] Loading DocVQA split=[cyan]{split}[/cyan], limit=[cyan]{limit}[/cyan]...")

    try:
        # Use the lmms-lab version which is well-maintained
        ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split=split)
    except Exception as e:
        console.print(f"[red]Failed to load dataset:[/red] {e}")
        console.print("[yellow]Try: huggingface-cli login[/yellow]")
        console.print("[yellow]You may need to accept the dataset terms at https://huggingface.co/datasets/lmms-lab/DocVQA[/yellow]")
        sys.exit(1)

    ds = ds.select(range(min(limit, len(ds))))
    console.print(f"[OK] Selected [green]{len(ds)}[/green] samples")

    # Create output dirs
    images_dir = DATA_DIR / "images"
    qa_dir = DATA_DIR / "qa_pairs"
    images_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    # Track stats
    total_qa = 0
    answer_lengths = []

    for i, sample in enumerate(track(ds, description="Saving documents...")):
        doc_id = f"docvqa_{split}_{i:04d}"

        # Save image
        image = sample.get("image")
        if image is not None:
            image_path = images_dir / f"{doc_id}.png"
            image.save(str(image_path))

        # Extract QA data
        question = sample.get("question", "")
        # DocVQA may store answers in different formats
        answers = sample.get("answers", sample.get("answer", []))
        if isinstance(answers, str):
            answers = [answers]

        qa_data = {
            "doc_id": doc_id,
            "source": "docvqa",
            "split": split,
            "question": question,
            "answers": answers,
            "question_id": sample.get("questionId", sample.get("question_id", i)),
        }

        total_qa += 1
        for ans in answers:
            answer_lengths.append(len(str(ans)))

        # Save QA pair
        qa_path = qa_dir / f"{doc_id}.json"
        with open(qa_path, "w", encoding="utf-8") as f:
            json.dump(qa_data, f, indent=2)

    # Summary
    console.print()
    console.rule("[bold green]Summary[/bold green]")

    table = Table(title="DocVQA Download Stats")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Split", split)
    table.add_row("Documents saved", str(len(ds)))
    table.add_row("QA pairs", str(total_qa))
    table.add_row("Avg answer length", f"{sum(answer_lengths)/max(len(answer_lengths),1):.1f} chars")
    table.add_row("Images dir", str(images_dir))
    table.add_row("QA dir", str(qa_dir))

    console.print(table)

    # Show a few example QA pairs
    console.print()
    console.rule("[bold yellow]Sample QA Pairs[/bold yellow]")

    samples_to_show = min(3, len(ds))
    for i in range(samples_to_show):
        qa_path = qa_dir / f"docvqa_{split}_{i:04d}.json"
        with open(qa_path, "r") as f:
            qa = json.load(f)
        console.print(f"\n[bold]Doc:[/bold] {qa['doc_id']}")
        console.print(f"  [cyan]Q:[/cyan] {qa['question']}")
        console.print(f"  [green]A:[/green] {', '.join(qa['answers'][:3])}")

    console.print(f"\n[OK] Done! Inspect with: [cyan]python scripts/inspect_dataset.py --dataset docvqa --limit 5[/cyan]")


if __name__ == "__main__":
    main()

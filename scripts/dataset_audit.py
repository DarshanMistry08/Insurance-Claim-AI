"""
scripts/dataset_audit.py

Dataset Auditing Tool for Insurance Claim Fraud Detection System.
Scans FUNSD and DocVQA datasets for:
1. Class imbalances.
2. Missing or empty annotations.
3. Overly small bounding boxes (OCR unfriendly).
4. Provides actionable retraining/preprocessing recommendations.

Usage:
    python scripts/dataset_audit.py
"""

import os
import json
from pathlib import Path
from collections import defaultdict

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
except ImportError:
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
        def rule(self, *args, **kwargs):
            print("-" * 40)
    console = Console()
    Table = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def audit_funsd():
    funsd_dir = DATA_DIR / "funsd"
    annotations_dir = funsd_dir / "annotations"
    
    if not annotations_dir.exists():
        console.print("[red]FUNSD annotations not found. Skipping.[/red]")
        return

    console.rule("[bold blue]FUNSD Dataset Audit[/bold blue]")
    
    files = list(annotations_dir.glob("*.json"))
    console.print(f"Total documents to audit: {len(files)}")
    
    class_counts = defaultdict(int)
    small_bbox_count = 0
    total_entities = 0
    missing_text_count = 0

    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                continue
        
        entities = data.get("entities", [])
        total_entities += len(entities)
        
        for ent in entities:
            tag = ent.get("tag", "O")
            class_counts[tag] += 1
            
            text = ent.get("text", ent.get("word", ""))
            if not text.strip():
                missing_text_count += 1
                
            bbox = ent.get("bbox", [])
            if len(bbox) == 4:
                # [x1, y1, x2, y2]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                if width < 10 or height < 10:
                    small_bbox_count += 1

    console.print("\n[bold]Class Imbalance (FUNSD):[/bold]")
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_entities) * 100 if total_entities > 0 else 0
        console.print(f"  {cls}: {count} ({pct:.1f}%)")
        
    console.print("\n[bold]Quality Issues:[/bold]")
    console.print(f"  Entities with empty text: {missing_text_count}")
    console.print(f"  Entities with very small bounding boxes (<10px): {small_bbox_count}")
    
    if class_counts.get("O", 0) > total_entities * 0.5:
        console.print("\n[yellow]WARNING: High 'O' (Other) class imbalance. The model may struggle to distinguish relevant fields.[/yellow]")


def audit_docvqa():
    docvqa_dir = DATA_DIR / "docvqa"
    qa_dir = docvqa_dir / "qa_pairs"
    
    if not qa_dir.exists():
        console.print("[red]DocVQA qa_pairs not found. Skipping.[/red]")
        return
        
    console.rule("[bold blue]DocVQA Dataset Audit[/bold blue]")
    
    files = list(qa_dir.glob("*.json"))
    console.print(f"Total documents to audit: {len(files)}")
    
    question_lengths = []
    answer_lengths = []
    missing_answers = 0
    
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                continue
                
        q = data.get("question", "")
        question_lengths.append(len(q.split()))
        
        ans = data.get("answers", [])
        if not ans:
            missing_answers += 1
        else:
            answer_lengths.append(len(ans[0].split()))

    avg_q = sum(question_lengths) / len(question_lengths) if question_lengths else 0
    avg_a = sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0
    
    console.print(f"\n[bold]QA Statistics:[/bold]")
    console.print(f"  Average question length (words): {avg_q:.1f}")
    console.print(f"  Average answer length (words): {avg_a:.1f}")
    console.print(f"  Questions with no answers: {missing_answers}")


def print_recommendations():
    console.rule("[bold green]Retraining & Preprocessing Recommendations[/bold green]")
    recs = """
1. **Preprocessing (Critical for OCR and Donut):**
   - **Deskewing:** Ensure all documents are deskewed. Donut performs poorly on heavily rotated text.
   - **Contrast Enhancement:** Apply adaptive binarization (e.g., cv2.adaptiveThreshold) to fix faded ink.
   - **Resolution:** Resize images to ensure maximum width/height does not exceed the model's max resolution (e.g., 1280x960), but maintain aspect ratio.

2. **Annotation Fixes (Data Centric AI):**
   - **Missing Fields:** Ensure that if a field is physically present, its bounding box exists. If absent, ensure it is NOT annotated (or is annotated as a negative sample if your schema supports it).
   - **Small Bounding Boxes:** Filter or manually fix bounding boxes that are <10px in height or width, as they introduce noise.
   
3. **Model Fine-Tuning Strategy:**
   - The current generic Donut-DocVQA model understands general forms. Fine-tune it specifically on **Insurance Claim Forms** (your domain data).
   - Use a specialized prompt template: `<s_claimant_name>{name}</s_claimant_name><s_amount>{amount}</s_amount>`.
   - Augmentation: Add noise, slight rotations, and blur to the training set to make the model robust against poor-quality mobile scans.
"""
    console.print(recs)


if __name__ == "__main__":
    audit_funsd()
    print("\n")
    audit_docvqa()
    print("\n")
    print_recommendations()

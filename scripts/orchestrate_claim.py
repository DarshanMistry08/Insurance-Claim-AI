"""
Phase 3: Multi-Agent Orchestration with LangGraph

Refactors the linear pipeline into a graph:
1. extract (Donut DocVQA)
2. verify_policy (pgvector RAG)
3. check_duplicates (Postgres similarity on past claims)
4. score_fraud (LangChain + Ollama Zero-shot)
5. summarize (LangChain + Ollama)
"""

import os
import sys
import json
from pathlib import Path
from typing import TypedDict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed; rely on env vars being set externally

import click
import psycopg2
from PIL import Image
from rich.console import Console

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
console = Console(force_terminal=True, highlight=False)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "claims_db",
    "user": "claims",
    "password": os.getenv("DB_PASSWORD", ""),
}

# ----------------- 1. State Definition ----------------- #
class ClaimState(TypedDict):
    doc_id: str
    image_path: str
    extracted_fields: dict
    policy_context: str
    is_duplicate: bool
    fraud_score: str
    final_summary: str

# ----------------- 2. Node Functions ----------------- #

def node_extract(state: ClaimState) -> ClaimState:
    """Extract fields using Donut DocVQA."""
    console.print("\n[bold cyan]Node: extract[/bold cyan]")
    
    try:
        import torch
        from transformers import DonutProcessor, VisionEncoderDecoderModel
        import re
        
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        image = Image.open(state["image_path"]).convert("RGB")
        
        questions = {
            "claimant_name": "What is the name of the person?",
            "incident_type": "What is the type or category?",
            "incident_date": "What is the date?",
            "amount_claimed": "What is the total amount?"
        }
        
        fields = {}
        for field, q in questions.items():
            prompt = f"<s_docvqa><s_question>{q}</s_question><s_answer>"
            pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
            decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=512,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )
            
            seq = processor.batch_decode(outputs.sequences)[0]
            seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
            match = re.search(r"<s_answer>(.*)</s_answer>", seq, re.DOTALL)
            val = match.group(1).strip() if match else "N/A"
            fields[field] = val
            console.print(f"  [green]{field}:[/green] {val}")
            
        return {"extracted_fields": fields}
    except Exception as e:
        console.print(f"[red]Extraction Error:[/red] {e}")
        return {"extracted_fields": {}}

def node_verify_policy(state: ClaimState) -> ClaimState:
    """Retrieve policy context from pgvector."""
    console.print("\n[bold cyan]Node: verify_policy[/bold cyan]")
    from sentence_transformers import SentenceTransformer
    
    incident_type = state.get("extracted_fields", {}).get("incident_type", "unknown")
    query = f"coverage limit for {incident_type} damage"
    
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_emb = emb_model.encode(query).tolist()
    
    context = "No context found."
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT chunk_text FROM chunks ORDER BY embedding <-> %s::vector LIMIT 1;",
            (query_emb,)
        )
        res = cursor.fetchone()
        if res:
            context = res[0]
        cursor.close()
        conn.close()
    except Exception as e:
        console.print(f"[red]DB Error:[/red] {e}")
        
    console.print(f"  [green]Retrieved Policy:[/green] {context[:100]}...")
    return {"policy_context": context}

def node_check_duplicates(state: ClaimState) -> ClaimState:
    """Check against previously processed claims (mocked simple vector check)."""
    console.print("\n[bold cyan]Node: check_duplicates[/bold cyan]")
    # In a real system, we embed the new claim and query `processed_claims` for sim > 0.95
    # For the vertical slice, we'll just check if the exact doc_id exists.
    doc_id = state["doc_id"]
    is_dup = False
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM processed_claims WHERE document_id = (SELECT document_id FROM documents WHERE file_path LIKE %s LIMIT 1);", (f"%{doc_id}%",))
        if cursor.fetchone():
            is_dup = True
        cursor.close()
        conn.close()
    except Exception:
        pass
    
    console.print(f"  [green]Is Duplicate:[/green] {is_dup}")
    return {"is_duplicate": is_dup}

def node_score_fraud(state: ClaimState) -> ClaimState:
    """Zero-shot fraud scoring via Ollama."""
    console.print("\n[bold cyan]Node: score_fraud[/bold cyan]")
    llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434")
    
    prompt = f"""
Analyze these claim fields for potential fraud markers:
{json.dumps(state.get("extracted_fields", {}), indent=2)}

Look for: unusually high amounts, weird dates, missing names.
Respond ONLY with a short sentence describing the risk level (Low, Medium, High).
"""
    try:
        resp = llm.invoke(prompt)
        score = resp.content.strip()
    except Exception as e:
        console.print(f"[red]Ollama Error:[/red] {e}")
        score = "Error"
        
    console.print(f"  [green]Fraud Score:[/green] {score}")
    return {"fraud_score": score}

def node_summarize(state: ClaimState) -> ClaimState:
    """Final LLM summarization."""
    console.print("\n[bold cyan]Node: summarize[/bold cyan]")
    llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434")
    
    prompt = f"""
Generate a short executive summary for this insurance claim.
Fields: {state.get('extracted_fields')}
Policy context: {state.get('policy_context')}
Duplicate Check: {'Failed (Duplicate)' if state.get('is_duplicate') else 'Passed'}
Fraud Score: {state.get('fraud_score')}

Is this claim valid? Yes/No/Why.
"""
    try:
        resp = llm.invoke(prompt)
        summary = resp.content.strip()
    except Exception as e:
        summary = f"Ollama Error: {e}"
        
    console.print(f"\n[bold yellow]--- Final Summary ---[/bold yellow]\n{summary}")
    return {"final_summary": summary}


# ----------------- 3. Graph Orchestration ----------------- #

@click.command()
@click.option("--doc-id", required=True, help="Document ID (e.g. funsd_train_0000)")
@click.option("--image-dir", default=str(PROJECT_ROOT / "data" / "funsd" / "images"))
def main(doc_id: str, image_dir: str):
    console.rule("[bold blue]LangGraph Claim Orchestration (Ollama)[/bold blue]")
    
    image_path = Path(image_dir) / f"{doc_id}.png"
    if not image_path.exists():
        console.print(f"[red]Error: Image not found[/red] {image_path}")
        sys.exit(1)
        
    # Build Graph
    workflow = StateGraph(ClaimState)
    
    workflow.add_node("extract", node_extract)
    workflow.add_node("verify_policy", node_verify_policy)
    workflow.add_node("check_duplicates", node_check_duplicates)
    workflow.add_node("score_fraud", node_score_fraud)
    workflow.add_node("summarize", node_summarize)
    
    workflow.add_edge(START, "extract")
    workflow.add_edge("extract", "verify_policy")
    workflow.add_edge("verify_policy", "check_duplicates")
    workflow.add_edge("check_duplicates", "score_fraud")
    workflow.add_edge("score_fraud", "summarize")
    workflow.add_edge("summarize", END)
    
    app = workflow.compile()
    
    # Run Orchestrator
    initial_state = {
        "doc_id": doc_id,
        "image_path": str(image_path),
        "extracted_fields": {},
        "policy_context": "",
        "is_duplicate": False,
        "fraud_score": "",
        "final_summary": ""
    }
    
    console.print("[>>] Invoking graph...")
    app.invoke(initial_state)

if __name__ == "__main__":
    main()

"""
End-to-End Vertical Slice: Field Extraction -> RAG -> LLM Validation

This script ties together Phase 2:
1. Extraction: Uses `naver-clova-ix/donut-base-finetuned-docvqa` to extract fields from an image.
2. Retrieval: Uses `sentence-transformers` to embed a query and retrieve the top policy chunk from pgvector.
3. Validation: Uses the OpenAI API to evaluate the claim based on the retrieved policy and extracted fields.

Usage:
    set OPENAI_API_KEY=your_key (if you are using OpenAI)
    python scripts/process_claim.py --doc-id funsd_train_0000
"""

import os
import sys
import json
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed; rely on env vars being set externally

import click
import psycopg2
from PIL import Image
from rich.console import Console
from rich.table import Table

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

def load_donut_model():
    """Load the Donut DocVQA model and processor."""
    console.print("[>>] Loading Donut DocVQA model (this may take a minute on first run)...")
    try:
        import torch
        from transformers import DonutProcessor, VisionEncoderDecoderModel
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return processor, model, device
    except ImportError as e:
        console.print(f"[red]Missing dependency for Donut:[/red] {e}")
        sys.exit(1)

def extract_field(processor, model, device, image: Image.Image, question: str) -> str:
    """Run DocVQA on a single question."""
    import re
    # Format required by Donut DocVQA
    prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
    
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=model.config.decoder.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    # Extract the answer part
    answer_match = re.search(r"<s_answer>(.*)</s_answer>", sequence, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return "N/A"

def retrieve_policy_chunk(query: str) -> str:
    """Embed the query and retrieve the most relevant chunk from Postgres."""
    from sentence_transformers import SentenceTransformer
    console.print(f"[>>] Retrieving policy context for query: '{query}'...")
    
    # Normally we'd keep the model loaded in memory, but this is a simple script
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_emb = emb_model.encode(query).tolist()
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Perform vector similarity search (cosine distance <->)
        cursor.execute(
            """
            SELECT chunk_text 
            FROM chunks 
            ORDER BY embedding <-> %s::vector 
            LIMIT 1;
            """,
            (query_emb,)
        )
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if result:
            return result[0]
        return "No policy context found."
    except Exception as e:
        console.print(f"[red]Database error during retrieval:[/red] {e}")
        return "Error retrieving context."

def validate_claim_with_llm(extracted_fields: dict, policy_context: str) -> str:
    """Call OpenAI (or compatible API) to validate the claim."""
    import openai
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print("[yellow]WARNING: OPENAI_API_KEY not set. Using mock LLM response for demonstration.[/yellow]")
        return "MOCK LLM RESPONSE:\nFlagged: No\nExplanation: The claimed amount of $5,000 is well within the coverage limit of $25,000 for water damage as stated in the policy context. The incident dates also fall within the valid policy period."
    
    console.print("[>>] Calling LLM for validation...")
    
    client = openai.OpenAI(api_key=api_key)
    
    prompt = f"""
You are an insurance claims validator. Review the extracted fields from the claim document and compare them against the retrieved policy guidelines.

EXTRACTED CLAIM FIELDS:
{json.dumps(extracted_fields, indent=2)}

POLICY GUIDELINES:
{policy_context}

Task:
1. Does the claim appear valid based on the policy context?
2. Are there any discrepancies (e.g. amount claimed > coverage limit)?
3. Provide a clear Yes/No/Flagged conclusion and a brief explanation.
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Fast and cheap for this task
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM Error: {e}"

@click.command()
@click.option("--doc-id", required=True, help="Document ID to process (e.g., funsd_train_0000)")
@click.option("--image-dir", default=str(PROJECT_ROOT / "data" / "funsd" / "images"), help="Directory containing images")
def main(doc_id: str, image_dir: str):
    console.rule("[bold blue]Vertical Slice: End-to-End Claim Processing[/bold blue]")
    
    image_path = Path(image_dir) / f"{doc_id}.png"
    if not image_path.exists():
        console.print(f"[red]Image not found:[/red] {image_path}")
        sys.exit(1)
        
    console.print(f"[OK] Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    
    # 1. Field Extraction
    processor, model, device = load_donut_model()
    
    questions = {
        "claimant_name": "What is the name of the person?",
        "incident_type": "What is the type or category?",
        "incident_date": "What is the date?",
        "amount_claimed": "What is the total amount?"
    }
    
    extracted_fields = {}
    console.print("[>>] Extracting fields...")
    for field, q in questions.items():
        ans = extract_field(processor, model, device, image, q)
        extracted_fields[field] = ans
        console.print(f"  - {field}: [green]{ans}[/green]")
        
    # 2. RAG Retrieval
    incident_type = extracted_fields.get("incident_type", "")
    query = f"coverage limit for {incident_type} damage"
    policy_context = retrieve_policy_chunk(query)
    console.print(f"[OK] Retrieved context: [cyan]{policy_context}[/cyan]")
    
    # 3. LLM Validation
    llm_response = validate_claim_with_llm(extracted_fields, policy_context)
    
    console.rule("[bold yellow]Final Validation Result[/bold yellow]")
    console.print(llm_response)

if __name__ == "__main__":
    main()

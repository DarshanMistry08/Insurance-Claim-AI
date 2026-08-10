# Building ClaimSight: An Open-Source AI Claim Processing Pipeline

Insurance claim processing is traditionally a slow, manual, and error-prone process. Adjusters spend hours reviewing scanned forms, checking policies, and verifying information across multiple systems. **ClaimSight** is an open-source, local-first AI pipeline that automates the initial steps of this process using cutting-edge models.

In this post, we’ll walk through the architecture, the multi-agent pipeline built with LangGraph, and how you can run this entire system on your own hardware using Docker.

## The Architecture

ClaimSight separates the logic into a backend FastAPI service and a Next.js frontend for easy user interaction. But the real magic happens in the backend with **LangGraph** orchestrating multiple AI agents to process the document end-to-end.

The core stack:
- **Backend**: Python, FastAPI, LangGraph
- **Database**: PostgreSQL with `pgvector` for vector search
- **Local LLM**: Ollama (`llama3.2`) for privacy and fast local inference
- **Frontend**: Next.js 15
- **Orchestration**: Docker Compose

## The Multi-Agent Pipeline

To mimic an insurance adjuster's workflow, we designed a sequential pipeline using LangGraph:

1. **Extraction Agent (Node 1)**: Takes a claim document image (like an ACORD form) and extracts key structured fields: Claimant Name, Incident Date, Amount Claimed, etc.
2. **Policy Retrieval Agent (Node 2)**: Embeds the incident description and searches a `pgvector` database for relevant clauses in the user's insurance policy.
3. **Verification Agent (Node 3)**: Compares the extracted fields against the retrieved policy clauses to determine if the claim falls within coverage limits and dates.
4. **Fraud Scoring Agent (Node 4)**: Cross-references the claim details with a historical database to flag duplicates or suspiciously high claim amounts.
5. **Summary Agent (Node 5)**: Synthesizes the results into a markdown executive summary and provides a final recommendation (Approve, Flag for Review, Reject) along with a confidence score.

Because the system runs on local models (Ollama), it never sends sensitive Personal Identifiable Information (PII) to cloud providers.

## Try It Out

We have containerized the entire stack. You can spin up the Postgres database, Ollama, the Python API, and the Next.js frontend with one command.

```bash
git clone https://github.com/yourusername/claimsight.git
cd claimsight
docker-compose up -d --build
```

Once running, head over to `http://localhost:3000` to interact with the frontend. Upload a sample claim form, and watch the agents work in real-time as they extract, verify, and score the claim.

## What's Next?

Phase 6 of this project will explore adding support for multimodal evidence (like photos of car damage) and further refining the multi-agent collaboration with LangGraph's dynamic routing. Stay tuned!

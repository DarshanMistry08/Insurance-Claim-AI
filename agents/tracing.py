"""
agents/tracing.py — Langfuse + structlog wiring.

All calls are wrapped in try/except so the pipeline runs cleanly
even when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set.
"""

from __future__ import annotations
import os
import time
import structlog

logger = structlog.get_logger()

# ── Langfuse client (lazy, no-op if keys absent) ──────────────── #

_langfuse = None

def _get_langfuse():
    global _langfuse
    if _langfuse is not None:
        return _langfuse
    pk = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    sk = os.getenv("LANGFUSE_SECRET_KEY", "")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    if pk and sk:
        try:
            from langfuse import Langfuse
            _langfuse = Langfuse(public_key=pk, secret_key=sk, host=host)
            logger.info("langfuse.initialized", host=host)
        except Exception as e:
            logger.warning("langfuse.init_failed", error=str(e))
            _langfuse = None
    return _langfuse


# ── Trace context ─────────────────────────────────────────────── #

class NodeTracer:
    """Context manager that opens a Langfuse span and records latency."""

    def __init__(self, node_name: str, trace_id: str = "", inputs: dict | None = None):
        self.node_name = node_name
        self.trace_id = trace_id
        self.inputs = inputs or {}
        self._span = None
        self._t0 = 0.0
        self.span_id = ""

    def __enter__(self):
        self._t0 = time.perf_counter()
        logger.info("node.start", node=self.node_name, doc_id=self.inputs.get("doc_id", ""))
        lf = _get_langfuse()
        if lf and self.trace_id:
            try:
                trace = lf.trace(id=self.trace_id, name="claim_pipeline")
                self._span = trace.span(name=self.node_name, input=self.inputs)
                self.span_id = self._span.id
            except Exception:
                pass
        return self

    def end(self, outputs: dict | None = None, error: str = ""):
        latency_ms = (time.perf_counter() - self._t0) * 1000
        if error:
            logger.error("node.error", node=self.node_name, error=error, latency_ms=round(latency_ms, 1))
        else:
            logger.info("node.done", node=self.node_name, latency_ms=round(latency_ms, 1))
        if self._span:
            try:
                self._span.end(output=outputs or {})
            except Exception:
                pass
        return latency_ms

    def __exit__(self, *_):
        self.end()

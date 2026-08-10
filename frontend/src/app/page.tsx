"use client";

import { useState, useCallback, useRef } from "react";

/* ─── Types ─────────────────────────────────────────────────── */
interface ClaimResult {
  doc_id: string;
  document_type?: string;
  document_type_confidence?: number;
  classification_status?: string;
  extracted_fields: Record<string, any>;
  policy_context: string;
  is_duplicate: boolean;
  fraud_score: string;
  flags: any[];
  final_summary: any;
  recommendation: string;
  confidence_score: number;
  citations: { field_ref: string; clause_ref: string; rationale: string }[];
  retrieved_clauses: any[];
  agent_trace?: any[];
  fraud_result?: any;
  verification_verdict?: any;
}

type PipelineStep = "extract" | "verify_policy" | "check_duplicates" | "score_fraud" | "summarize";
type StepStatus = "waiting" | "active" | "done";

const PIPELINE_STEPS: { key: PipelineStep; label: string; icon: string }[] = [
  { key: "extract", label: "Field Extraction", icon: "🔍" },
  { key: "verify_policy", label: "Policy Verification", icon: "📋" },
  { key: "check_duplicates", label: "Duplicate Check", icon: "🔄" },
  { key: "score_fraud", label: "Fraud Scoring", icon: "⚠️" },
  { key: "summarize", label: "AI Summary", icon: "🤖" },
];

const FIELD_LABELS: Record<string, string> = {
  claimant_name: "Claimant Name",
  incident_type: "Incident Type",
  incident_date: "Incident Date",
  amount_claimed: "Amount Claimed",
  claim_filing_date: "Filing Date",
  incident_description: "Description",
  adjuster_name: "Adjuster Name",
  claimant_address: "Claimant Address",
  supporting_doc_count: "Supporting Docs",
  policy_number: "Policy Number"
};

/* ─── Helpers ────────────────────────────────────────────────── */
function safeRender(val: any): string {
  if (val === null || val === undefined) return "—";
  if (typeof val === "string") return val;
  if (typeof val === "object") {
    if (val.value !== undefined) return String(val.value);
    if (val.description !== undefined) return String(val.description);
    return JSON.stringify(val);
  }
  return String(val);
}

function parsePolicyContext(context: string) {
  if (!context || context.includes("No relevant policy clauses") || context.includes("No policy context")) {
    return null;
  }

  const extractSection = (secName: string, nextSecName: string | null) => {
    const startIdx = context.indexOf(secName);
    if (startIdx === -1) return "";
    const contentStart = startIdx + secName.length;
    let endIdx = context.length;
    if (nextSecName) {
      const nextIdx = context.indexOf(nextSecName);
      if (nextIdx !== -1) {
        endIdx = nextIdx;
      }
    }
    return context.substring(contentStart, endIdx).trim();
  };

  const coveredItems = extractSection("Covered Items:", "Coverage Limit:");
  const coverageLimit = extractSection("Coverage Limit:", "Restrictions:");
  const restrictions = extractSection("Restrictions:", "Exclusions:");
  const exclusions = extractSection("Exclusions:", "Deductibles:");
  const deductibles = extractSection("Deductibles:", "Policy Match Status:");
  const matchStatus = extractSection("Policy Match Status:", null)
    .replace(/^[:\-\s\*\#]+/, "")
    .split("\n")[0]
    .trim();
    
  const clean = (val: string) => val.split('\n').map(line => line.replace(/^[*-•\s]+/, '').trim()).filter(Boolean).join(', ');
  
  return {
    coveredItems: clean(coveredItems) || "Not applicable to policy schedule",
    coverageLimit: clean(coverageLimit) || "Not stated",
    restrictions: clean(restrictions) || "Not applicable to policy schedule",
    exclusions: clean(exclusions) || "Not applicable to policy schedule",
    deductibles: clean(deductibles) || "Not applicable to policy schedule",
    matchStatus: matchStatus || "Uncertain"
  };
}

function parseSummary(summaryText: string) {
  if (!summaryText) return null;
  
  const sections = [
    { key: "overview", patterns: ["1. Claim Overview", "Claim Overview"] },
    { key: "missingFields", patterns: ["2. Missing Fields", "Missing Fields"] },
    { key: "coverageMatch", patterns: ["3. Policy Coverage Match", "Policy Coverage Match"] },
    { key: "fraudRisk", patterns: ["4. Fraud Risk", "Fraud Risk"] },
    { key: "recommendation", patterns: ["5. Recommendation", "Recommendation"] }
  ];

  const parsed: any = {};
  
  for (let i = 0; i < sections.length; i++) {
    const current = sections[i];
    const next = sections[i + 1];
    
    let startIdx = -1;
    for (const pat of current.patterns) {
      const idx = summaryText.indexOf(pat);
      if (idx !== -1) {
        startIdx = idx + pat.length;
        break;
      }
    }
    
    if (startIdx === -1) {
      parsed[current.key] = "";
      continue;
    }

    let endIdx = summaryText.length;
    if (next) {
      for (const pat of next.patterns) {
        const idx = summaryText.indexOf(pat);
        if (idx !== -1) {
          endIdx = idx;
          break;
        }
      }
    }
    
    parsed[current.key] = summaryText.substring(startIdx, endIdx).trim().replace(/^[:\-\s\*\#]+/, '');
  }

  const rawMissing = parsed.missingFields || "";
  parsed.missingFields = rawMissing.split(/[\n,;]/)
    .map((s: string) => s.replace(/^[-*•\s]+/, '').trim())
    .filter((s: string) => s.length > 0 && !s.toLowerCase().includes("no required") && !s.toLowerCase().includes("no fields") && !s.toLowerCase().includes("none") && !s.toLowerCase().includes("no missing"));

  return parsed;
}

/* ─── Component ──────────────────────────────────────────────── */
export default function HomePage() {
  const [dragOver, setDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState<number>(-1);
  const [result, setResult] = useState<ClaimResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stepTimes, setStepTimes] = useState<Record<string, number>>({});
  const [reviewedFlags, setReviewedFlags] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Please upload an image file (PNG or JPG).");
      return;
    }
    setSelectedFile(file);
    setError(null);
    setResult(null);
    setReviewedFlags(false);
    setCurrentStep(-1);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(file);
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const simulatePipelineProgress = async () => {
    setStepTimes({});
    const times: Record<string, number> = {};
    for (let i = 0; i < PIPELINE_STEPS.length; i++) {
      setCurrentStep(i);
      const stepStart = Date.now();
      await new Promise((r) => setTimeout(r, 1500 + Math.random() * 500));
      times[PIPELINE_STEPS[i].key] = Date.now() - stepStart;
      setStepTimes({ ...times });
    }
  };

  const processFile = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setReviewedFlags(false);
    setCurrentStep(0);

    const formData = new FormData();
    formData.append("file", selectedFile);

    const progressPromise = simulatePipelineProgress();

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5 * 60 * 1000);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const response = await fetch(`${apiUrl}/claims/process`, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      await progressPromise;

      if (!response.ok) {
        let detail = "API error";
        try {
          const err = await response.json();
          detail = err.detail || detail;
        } catch {
          detail = `Server returned ${response.status}`;
        }
        throw new Error(detail);
      }

      const data: ClaimResult = await response.json();
      setCurrentStep(PIPELINE_STEPS.length);
      setResult(data);
    } catch (err: unknown) {
      clearTimeout(timeoutId);
      await progressPromise;
      if (err instanceof DOMException && err.name === "AbortError") {
        setError("Request timed out after 5 minutes. The LLM may be overloaded — try again.");
      } else if (err instanceof TypeError && (err.message === "Failed to fetch" || err.message.includes("fetch"))) {
        setError("Cannot reach the API server at localhost:8000. Make sure the backend is running (docker compose up -d).");
      } else {
        setError(err instanceof Error ? err.message : "Failed to process claim.");
      }
      setCurrentStep(-1);
    } finally {
      setLoading(false);
    }
  };

  const getStepStatus = (idx: number): StepStatus => {
    if (currentStep === -1 || (!loading && !result)) return "waiting";
    if (result) return "done";
    if (idx < currentStep) return "done";
    if (idx === currentStep) return "active";
    return "waiting";
  };

  const getDisplayTime = (key: PipelineStep) => {
    if (result && result.agent_trace) {
      const nodeNameMap: Record<string, string[]> = {
        extract: ["extract", "extract_agent"],
        verify_policy: ["verification_agent", "policy_retrieval_agent"],
        check_duplicates: ["fraud_scoring_agent"],
        score_fraud: ["fraud_scoring_agent"],
        summarize: ["summary_agent"]
      };

      const matchingNodes = nodeNameMap[key] || [];
      let totalMs = 0;
      result.agent_trace.forEach((log: any) => {
        if (matchingNodes.includes(log.node_name)) {
          totalMs += log.latency_ms || 0;
        }
      });

      if (totalMs > 0) {
        return `${(totalMs / 1000).toFixed(1)}s`;
      }
    }

    if (stepTimes[key]) {
      return `${(stepTimes[key] / 1000).toFixed(1)}s`;
    }

    return "";
  };

  // Parsers and metrics
  const hasFlags = !!(result && result.flags && result.flags.length > 0);
  const anyHighSeverityFlag = hasFlags && result.flags.some((f: any) => {
    const sev = typeof f === 'object' && f !== null ? f.severity : 'MEDIUM';
    return sev === 'HIGH';
  });

  const parsedPolicy = result ? parsePolicyContext(result.policy_context) : null;
  const parsedSummary = result
    ? (result.final_summary?.summary_text 
        ? parseSummary(result.final_summary.summary_text) 
        : (typeof result.final_summary === 'string' ? parseSummary(result.final_summary) : null))
    : null;

  // Semantic color for status badge
  let statusBadgeStyle = "bg-success-subtle color-success";
  let statusLabel = "Clean";
  if (hasFlags) {
    if (anyHighSeverityFlag) {
      statusBadgeStyle = "bg-danger-subtle color-danger";
      statusLabel = "High Risk";
    } else {
      statusBadgeStyle = "bg-warning-subtle color-warning";
      statusLabel = "Needs Review";
    }
  }

  // Visual meter setup for Fraud Risk
  const fraudLabel = result?.fraud_result?.label || (result?.fraud_score?.toLowerCase().includes("fraud") ? "fraudulent" : result?.fraud_score?.toLowerCase().includes("suspic") ? "suspicious" : "legitimate");
  const fraudConf = result?.fraud_result?.confidence !== undefined ? result.fraud_result.confidence : 0.6;
  const fraudPercent = fraudConf * 100;

  let fraudMeterColor = "#34d399"; // success
  let fraudBg = "bg-success-subtle border-[rgba(16,185,129,0.3)]";
  let fraudText = "Low Risk";
  if (fraudLabel === "fraudulent") {
    fraudMeterColor = "#f87171"; // danger
    fraudBg = "bg-danger-subtle border-[rgba(239,68,68,0.3)]";
    fraudText = "High Risk";
  } else if (fraudLabel === "suspicious") {
    fraudMeterColor = "#fbbf24"; // warning
    fraudBg = "bg-warning-subtle border-[rgba(245,158,11,0.3)]";
    fraudText = "Medium Risk";
  }

  const fraudReason = result?.fraud_result?.reason || "";
  const fraudReasonPoints = fraudReason ? fraudReason.split(/[.;] /).filter((s: string) => s.trim().length > 0) : [];

  const triggerToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3000);
  };

  return (
    <div className="min-h-screen py-10 px-4" style={{ fontFamily: "'Inter', sans-serif" }}>
      {/* Toast Notification */}
      {toast && (
        <div className="fixed bottom-5 right-5 z-50 flex items-center gap-3 p-4 rounded-xl bg-info-subtle border border-[rgba(59,130,246,0.3)] text-tier-2 glow-blue fade-in">
          <span>ℹ️</span> {toast}
        </div>
      )}

      {/* ── Header ── */}
      <div className="max-w-6xl mx-auto mb-10 fade-in">
        <div className="flex items-center gap-3 mb-2">
          <div style={{
            width: 42, height: 42, borderRadius: 12,
            background: "linear-gradient(135deg, #3b82f6, #8b5cf6)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 20, boxShadow: "0 0 20px rgba(59,130,246,0.4)"
          }}>⚖️</div>
          <span style={{ fontSize: 13, fontWeight: 700, letterSpacing: "0.12em", color: "var(--text-muted)", textTransform: "uppercase" }}>Claims AI Pipeline</span>
        </div>
        <h1 style={{ fontSize: 42, fontWeight: 800, lineHeight: 1.1 }} className="gradient-text">
          Intelligent Insurance<br />Claim Processor
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: 16, marginTop: 12, maxWidth: 560 }}>
          Upload a claim document image. Our multi-agent AI pipeline extracts fields, checks policy coverage, detects fraud, and generates a validation report — 100% locally.
        </p>
      </div>

      <div className="max-w-6xl mx-auto" style={{ display: "grid", gridTemplateColumns: result ? "1fr 1.6fr" : "1fr", gap: 24, alignItems: "start" }}>
        {/* ── Left Column: Upload + Pipeline ── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
          {/* Upload Zone */}
          <div className="glass card-padding fade-in">
            <p className="section-label">Document Upload</p>
            <div
              className={`upload-zone ${dragOver ? "drag-over" : ""}`}
              style={{ padding: 40, textAlign: "center", cursor: "pointer", position: "relative" }}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                style={{ display: "none" }}
                onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
              />
              {preview ? (
                <div>
                  <img src={preview} alt="preview" style={{ maxHeight: 220, maxWidth: "100%", borderRadius: 12, objectFit: "contain", marginBottom: 12 }} />
                  <p style={{ color: "var(--text-secondary)", fontSize: 13 }}>
                    📄 <strong style={{ color: "var(--text-primary)" }}>{selectedFile?.name}</strong>
                  </p>
                  <p style={{ color: "var(--text-muted)", fontSize: 12, marginTop: 4 }}>Click to change</p>
                </div>
              ) : (
                <div>
                  <div style={{ fontSize: 48, marginBottom: 16 }}>📂</div>
                  <p style={{ color: "var(--text-primary)", fontWeight: 600, fontSize: 16, marginBottom: 8 }}>
                    Drop your claim document here
                  </p>
                  <p style={{ color: "var(--text-muted)", fontSize: 13 }}>or click to browse — PNG, JPG supported</p>
                </div>
              )}
            </div>

            {error && (
              <div style={{ marginTop: 16, padding: "12px 16px", background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 10, color: "#fca5a5", fontSize: 13 }}>
                ⚠️ {error}
              </div>
            )}

            <button
              className="btn-primary"
              style={{ width: "100%", marginTop: 16, display: "flex", alignItems: "center", justifyContent: "center", gap: 10 }}
              onClick={processFile}
              disabled={!selectedFile || loading}
            >
              {loading ? (
                <><span className="spinner" /> Processing Claim...</>
              ) : (
                <><span>🚀</span> Process Claim</>
              )}
            </button>
          </div>

          {/* Pipeline Steps */}
          {(loading || result) && (
            <div className="glass card-padding fade-in">
              <p className="section-label">Pipeline Progress</p>
              <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
                {PIPELINE_STEPS.map((step, idx) => {
                  const status = getStepStatus(idx);
                  const stepTime = getDisplayTime(step.key);
                  return (
                    <div key={step.key} style={{ display: "flex", alignItems: "flex-start", gap: 16 }}>
                      <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                        <div className={`step-dot ${status}`} style={{ marginTop: 6 }} />
                        {idx < PIPELINE_STEPS.length - 1 && (
                          <div style={{ width: 2, height: 32, background: status === "done" ? "rgba(16,185,129,0.4)" : "rgba(255,255,255,0.07)", marginTop: 4, borderRadius: 1 }} />
                        )}
                      </div>
                      <div style={{ paddingBottom: idx < PIPELINE_STEPS.length - 1 ? 16 : 0, flex: 1 }}>
                        <div className="flex justify-between items-center">
                          <p style={{
                            fontWeight: 600, fontSize: 14,
                            color: status === "done" ? "#6ee7b7" : status === "active" ? "#93c5fd" : "var(--text-muted)"
                          }}>
                            {step.icon} {step.label}
                          </p>
                          {stepTime && (
                            <span className="text-xs text-muted font-bold bg-[rgba(255,255,255,0.03)] px-2 py-0.5 rounded">
                              {stepTime}
                            </span>
                          )}
                        </div>
                        <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 2 }}>
                          {status === "done" ? "✓ Completed" : status === "active" ? "Running…" : "Pending"}
                        </p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Tech Badge */}
          <div className="glass card-padding fade-in">
            <p className="section-label">Powered By</p>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {["LangGraph", "Ollama llama3.2", "Donut DocVQA", "pgvector", "sentence-transformers"].map(tech => (
                <span key={tech} style={{
                  padding: "5px 12px", borderRadius: 20, fontSize: 12, fontWeight: 600,
                  background: "rgba(139,92,246,0.1)", border: "1px solid rgba(139,92,246,0.25)", color: "#c4b5fd"
                }}>{tech}</span>
              ))}
            </div>
          </div>
        </div>

        {/* ── Right Column: Results ── */}
        {result && (
          <div className="flex flex-col card-gap">
            {/* Summary Header */}
            <div className="glass-bright card-padding fade-in glow-blue">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
                <div>
                  <p className="section-label">Validation Result</p>
                  <h2 className="text-tier-1 font-bold">Document Analysis Complete</h2>
                  <div className="flex items-center gap-2 mt-1">
                    <p style={{ color: "var(--text-muted)", fontSize: 13 }}>Doc ID: {result.doc_id}</p>
                    {result.document_type && (
                      <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-[rgba(139,92,246,0.15)] text-[#c4b5fd] border border-[rgba(139,92,246,0.3)]">
                        📄 Document Type: <strong>{result.document_type.replace('_', ' ').toUpperCase()}</strong> ({((result.document_type_confidence || 0.85) * 100).toFixed(0)}% conf)
                      </span>
                    )}
                  </div>
                </div>
                <span className={`px-4 py-1.5 rounded-full fontWeight-bold uppercase tracking-wider text-xs font-black ${statusBadgeStyle}`}>
                  {statusLabel}
                </span>
              </div>

              {/* Extracted Fields Grid */}
              <p className="section-label mt-6">Extracted Fields</p>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 20 }}>
                {Object.entries(result.extracted_fields).map(([key, val]) => {
                  const status = typeof val === "object" && val.status ? val.status : (
                    (!val || val === "N/A" || val === "NOT_FOUND" || (typeof val === "object" && (!val.value || val.value === "N/A"))) ? "NOT_FOUND" : "EXTRACTED"
                  );
                  const displayVal = safeRender(val);
                  const confidence = typeof val === "object" && val.confidence !== undefined ? val.confidence : null;
                  const reason = typeof val === "object" && val.missing_reason ? val.missing_reason : "";
                  const method = typeof val === "object" && val.extraction_method ? val.extraction_method : "";

                  const isFlagged = result.flags.some((f: any) => {
                    if (typeof f === 'object' && f !== null) {
                      return f.field_ref === key;
                    }
                    return false;
                  });

                  let cardStyle = "";
                  let badge = null;
                  let content = null;

                  if (status === "EXTRACTED") {
                    badge = confidence !== null && (
                      <span className="text-[10px] text-[#34d399] bg-success-subtle px-1.5 py-0.5 rounded font-semibold">
                        {(confidence * 100).toFixed(0)}% conf
                      </span>
                    );
                    content = (
                      <p className="text-tier-2 font-bold text-[rgba(255,255,255,0.95)]">
                        {displayVal}
                      </p>
                    );
                  } else if (status === "NOT_FOUND") {
                    cardStyle = "field-missing";
                    badge = <span className="text-xs color-info font-bold flex items-center gap-1">ℹ️ Not present</span>;
                    content = (
                      <p className="text-tier-3 color-info font-normal italic">
                        {reason || "Not present in document"}
                      </p>
                    );
                  } else {
                    cardStyle = isFlagged ? "field-missing-flagged" : "field-missing";
                    badge = <span className="text-xs color-danger font-bold flex items-center gap-1">⚠️ Failed</span>;
                    content = (
                      <p className="text-tier-3 color-danger font-normal italic">
                        {reason || "Extraction failed — needs manual review"}
                      </p>
                    );
                  }

                  return (
                    <div key={key} className={`field-card ${cardStyle} ${isFlagged ? "card-border-connect" : ""}`}>
                      <div className="flex justify-between items-start mb-1">
                        <p className="text-tier-3 text-muted uppercase font-bold tracking-wider" style={{ fontSize: 10 }}>
                          {FIELD_LABELS[key] || key}
                        </p>
                        {badge}
                      </div>
                      {content}
                    </div>
                  );
                })}
              </div>

              {/* Flags */}
              {hasFlags && (
                <>
                  <p className="section-label">Flags Raised</p>
                  <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 20 }}>
                    {result.flags.map((flag: any, i: number) => {
                      const flagSev = typeof flag === 'object' && flag !== null ? flag.severity : 'MEDIUM';
                      let flagStyle = "bg-warning-subtle color-warning border-[rgba(245,158,11,0.3)]";
                      if (flagSev === 'HIGH') {
                        flagStyle = "bg-danger-subtle color-danger border-[rgba(239,68,68,0.3)]";
                      }
                      return (
                        <div key={i} className={`p-4 rounded-xl border flex gap-3 ${flagStyle}`}>
                          <span className="text-lg">⚑</span>
                          <div>
                            <p className="font-bold text-tier-2">{typeof flag === 'object' ? flag.code : 'FLAG'}</p>
                            <p className="text-tier-3 opacity-90 mt-1">{typeof flag === 'object' ? flag.description : String(flag)}</p>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </>
              )}

              {/* Fraud Score Visual Gauge */}
              {result.fraud_score && (
                <div className={`p-5 rounded-xl border mb-5 ${fraudBg}`}>
                  <div className="flex justify-between items-center mb-3">
                    <p className="text-tier-2 font-bold">Fraud Assessment: <span style={{ color: fraudMeterColor }}>{fraudText}</span></p>
                    <p className="text-tier-2 font-bold" style={{ color: fraudMeterColor }}>{fraudPercent.toFixed(0)}% Confidence</p>
                  </div>
                  
                  <div className="fraud-gauge-container mb-2">
                    <div className="fraud-gauge-bar" style={{ width: `${fraudPercent}%`, backgroundColor: fraudMeterColor }} />
                    <div className="fraud-gauge-marker" style={{ left: `${fraudPercent}%`, borderColor: fraudMeterColor }} />
                  </div>
                  
                  <div className="flex justify-between text-[10px] text-muted font-bold mb-4">
                    <span>LOW RISK</span>
                    <span>SUSPICIOUS</span>
                    <span>HIGH RISK</span>
                  </div>

                  {fraudReasonPoints.length > 0 && (
                    <ul className="list-disc pl-5 text-tier-3 text-[rgba(255,255,255,0.7)] flex flex-col gap-1">
                      {fraudReasonPoints.map((point: string, idx: number) => (
                        <li key={idx}>{point.replace(/^[*-•\s]+/, "")}</li>
                      ))}
                    </ul>
                  )}
                </div>
              )}

              {/* Duplicate */}
              <div className={`p-4 rounded-xl mb-4 text-tier-3 font-semibold ${
                result.is_duplicate 
                  ? "bg-warning-subtle color-warning border-[rgba(245,158,11,0.3)]" 
                  : "bg-success-subtle color-success border-[rgba(16,185,129,0.3)]"
              }`}>
                {result.is_duplicate ? "⚠️ Duplicate claim detected in database" : "✓ No duplicate found in database"}
              </div>
            </div>

            {/* Policy Context */}
            {result.policy_context && (
              <>
                {parsedPolicy ? (
                  <div className="glass card-padding fade-in fade-in-delay-1 flex flex-col gap-4">
                    <div className="flex justify-between items-center border-b border-[rgba(255,255,255,0.06)] pb-4">
                      <div>
                        <p className="section-label">Policy Verification</p>
                        <h3 className="text-tier-2 font-bold">Retrieved Policy Analysis</h3>
                      </div>
                      <span className={`px-4 py-1.5 rounded-full text-xs font-black uppercase tracking-wide ${
                        parsedPolicy.matchStatus.toLowerCase().includes("verified") || parsedPolicy.matchStatus.toLowerCase().includes("covered")
                          ? "bg-success-subtle color-success"
                          : parsedPolicy.matchStatus.toLowerCase().includes("partially")
                          ? "bg-warning-subtle color-warning"
                          : parsedPolicy.matchStatus.toLowerCase().includes("not covered") || parsedPolicy.matchStatus.toLowerCase().includes("uncertain")
                          ? "bg-danger-subtle color-danger"
                          : "bg-success-subtle color-success"
                      }`}>
                        {parsedPolicy.matchStatus}
                      </span>
                    </div>
                    
                    <div className="flex flex-col gap-3">
                      <div className="flex justify-between items-center py-2 border-b border-[rgba(255,255,255,0.03)]">
                        <span className="text-tier-3 text-muted font-semibold">COVERED ITEMS</span>
                        <span className="text-tier-3 font-semibold text-[rgba(255,255,255,0.9)] text-right">{parsedPolicy.coveredItems}</span>
                      </div>
                      <div className="flex justify-between items-center py-2 border-b border-[rgba(255,255,255,0.03)]">
                        <span className="text-tier-3 text-muted font-semibold">COVERAGE LIMIT</span>
                        <span className="text-tier-3 font-semibold text-[rgba(255,255,255,0.9)] text-right">{parsedPolicy.coverageLimit}</span>
                      </div>
                      <div className="flex justify-between items-center py-2 border-b border-[rgba(255,255,255,0.03)]">
                        <span className="text-tier-3 text-muted font-semibold">RESTRICTIONS</span>
                        <span className="text-tier-3 font-semibold text-[rgba(255,255,255,0.9)] text-right">{parsedPolicy.restrictions}</span>
                      </div>
                      <div className="flex justify-between items-center py-2 border-b border-[rgba(255,255,255,0.03)]">
                        <span className="text-tier-3 text-muted font-semibold">EXCLUSIONS</span>
                        <span className="text-tier-3 font-semibold text-[rgba(255,255,255,0.9)] text-right">{parsedPolicy.exclusions}</span>
                      </div>
                      <div className="flex justify-between items-center py-2">
                        <span className="text-tier-3 text-muted font-semibold">DEDUCTIBLES</span>
                        <span className="text-tier-3 font-semibold text-[rgba(255,255,255,0.9)] text-right">{parsedPolicy.deductibles}</span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="glass card-padding fade-in fade-in-delay-1">
                    <p className="section-label">Policy Verification</p>
                    <div className="p-4 rounded-xl bg-danger-subtle color-danger text-tier-3">
                      No relevant policy clauses were found for this claim.
                    </div>
                  </div>
                )}
              </>
            )}

            {/* AI Summary Structured Card */}
            <div className="glass card-padding fade-in fade-in-delay-2 flex flex-col gap-6">
              <div className="border-b border-[rgba(255,255,255,0.06)] pb-4">
                <p className="section-label">AI Analysis Report</p>
                <h3 className="text-tier-2 font-bold">Executive Summary Report</h3>
              </div>

              {parsedSummary ? (
                <div className="flex flex-col gap-5">
                  {/* Overview */}
                  <div className="flex gap-4 p-4 rounded-xl bg-info-subtle border-l-4 border-l-[#3b82f6]">
                    <span className="text-xl">🔍</span>
                    <div>
                      <p className="text-tier-2 font-bold color-info">1. Claim Overview</p>
                      <p className="text-tier-3 mt-1 text-[rgba(255,255,255,0.8)]">{parsedSummary.overview || "Claim details parsed from document."}</p>
                    </div>
                  </div>

                  {/* Missing Fields */}
                  <div className="flex gap-4 p-4 rounded-xl bg-[rgba(255,255,255,0.02)] border border-[rgba(255,255,255,0.05)]">
                    <span className="text-xl">⚠️</span>
                    <div>
                      <p className="text-tier-2 font-bold color-warning">2. Missing Fields</p>
                      {parsedSummary.missingFields.length > 0 ? (
                        <div className="flex flex-wrap gap-2 mt-2">
                          {parsedSummary.missingFields.map((field: string, idx: number) => (
                            <span key={idx} className="px-2.5 py-1 rounded-md text-xs font-semibold bg-warning-subtle color-warning">
                              {field}
                            </span>
                          ))}
                        </div>
                      ) : (
                        <p className="text-tier-3 mt-1 text-[rgba(255,255,255,0.8)]">No required fields are missing.</p>
                      )}
                    </div>
                  </div>

                  {/* Coverage Match */}
                  <div className="flex gap-4 p-4 rounded-xl bg-[rgba(255,255,255,0.02)] border border-[rgba(255,255,255,0.05)]">
                    <span className="text-xl">📋</span>
                    <div>
                      <p className="text-tier-2 font-bold color-success">3. Policy Coverage Match</p>
                      <p className="text-tier-3 mt-1 text-[rgba(255,255,255,0.8)]">{parsedSummary.coverageMatch}</p>
                    </div>
                  </div>

                  {/* Fraud Risk */}
                  <div className="flex gap-4 p-4 rounded-xl bg-[rgba(255,255,255,0.02)] border border-[rgba(255,255,255,0.05)]">
                    <span className="text-xl">🚨</span>
                    <div>
                      <p className="text-tier-2 font-bold color-danger">4. Fraud Risk Assessment</p>
                      <p className="text-tier-3 mt-1 text-[rgba(255,255,255,0.8)]">{parsedSummary.fraudRisk}</p>
                    </div>
                  </div>

                  {/* Recommendation */}
                  <div className="flex gap-4 p-4 rounded-xl bg-danger-subtle border-l-4 border-l-[#ef4444]">
                    <span className="text-xl">⚖️</span>
                    <div>
                      <p className="text-tier-2 font-bold color-danger">5. Final Recommendation</p>
                      <p className="text-tier-1 font-black tracking-wide mt-1 uppercase" style={{ color: parsedSummary.recommendation.toLowerCase().includes("approve") ? "#34d399" : parsedSummary.recommendation.toLowerCase().includes("review") ? "#fbbf24" : "#f87171" }}>
                        {parsedSummary.recommendation}
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <div style={{ whiteSpace: "pre-wrap", color: "var(--text-secondary)", fontSize: 14 }}>
                  {typeof result.final_summary === 'object' 
                    ? (result.final_summary?.summary_text || JSON.stringify(result.final_summary, null, 2)) 
                    : result.final_summary}
                </div>
              )}

              {/* Citations Display (No raw JSON leaks) */}
              {result.citations && result.citations.length > 0 && (
                <div className="mt-4 border-t border-[rgba(255,255,255,0.06)] pt-4">
                  <p className="section-label">Cited Fields & Policy References</p>
                  <div className="flex flex-col gap-2">
                    {result.citations.map((cit, i) => (
                      <div key={i} className="flex flex-col md:flex-row md:items-start justify-between p-3 rounded-lg bg-[rgba(255,255,255,0.02)] border border-[rgba(255,255,255,0.04)] text-tier-3">
                        <div>
                          <div className="flex flex-wrap gap-2 items-center mb-1">
                            {cit.field_ref && (
                              <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-info-subtle color-info">
                                Field: {cit.field_ref}
                              </span>
                            )}
                            {cit.clause_ref && (
                              <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-warning-subtle color-warning">
                                Clause: {cit.clause_ref}
                              </span>
                            )}
                          </div>
                          <p className="text-tier-3 text-[rgba(255,255,255,0.7)] mt-1">{cit.rationale}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Actions Panel */}
            <div className="glass card-padding fade-in fade-in-delay-3 flex flex-col gap-4">
              {/* Flag Review Acknowledgment */}
              {hasFlags && (
                <label className="flex items-start gap-3 p-4 rounded-xl bg-danger-subtle border border-[rgba(239,68,68,0.2)] cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={reviewedFlags}
                    onChange={(e) => setReviewedFlags(e.target.checked)}
                    className="mt-1 w-4 h-4 rounded text-red-500 focus:ring-red-500 border-gray-600 bg-gray-700"
                  />
                  <div>
                    <span className="text-sm font-semibold color-danger block">I have reviewed and acknowledge the raised flags.</span>
                    <span className="text-xs text-muted block mt-0.5">You must check this box to proceed to another claim review.</span>
                  </div>
                </label>
              )}

              <div className="flex gap-4">
                <button
                  className="btn-primary flex-1"
                  style={{ 
                    padding: "12px 20px", 
                    fontSize: 14, 
                    opacity: (hasFlags && !reviewedFlags) ? 0.4 : 1, 
                    cursor: (hasFlags && !reviewedFlags) ? "not-allowed" : "pointer" 
                  }}
                  onClick={() => {
                    if (hasFlags && !reviewedFlags) return;
                    setResult(null);
                    setSelectedFile(null);
                    setPreview(null);
                    setCurrentStep(-1);
                    setReviewedFlags(false);
                  }}
                  disabled={hasFlags && !reviewedFlags}
                >
                  + Process Another Claim
                </button>
                <button
                  style={{
                    flex: 1, padding: "12px 20px", fontSize: 14, fontWeight: 600,
                    background: "rgba(59,130,246,0.1)", border: "1px solid rgba(59,130,246,0.3)",
                    color: "#93c5fd", borderRadius: 12, cursor: "pointer"
                  }}
                  onClick={() => {
                    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a"); a.href = url;
                    a.download = `claim_${result.doc_id}.json`; a.click();
                    triggerToast("Claim JSON data exported successfully!");
                  }}
                >
                  ⬇ Export JSON
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}


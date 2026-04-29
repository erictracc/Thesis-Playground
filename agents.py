"""
agents.py - v2
--------------
Full pipeline inspired by LEAN-LLM-OPT:
  1. Classification Agent  — RAG → identifies problem type
  2. Instructor Agent      — RAG filtered by type → builds guidance
  3. Worker Agent          — RAG + instructor guidance → formulation
  4. Validator Agent       — checks formulation → PASS / PARTIAL / FAIL + feedback
  5. Repair Loop           — if PARTIAL or FAIL, worker retries with feedback (max 2x)

Local stack: Ollama + Qdrant + sentence-transformers
"""

from __future__ import annotations
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_HOST     = "192.168.195.69"
QDRANT_PORT     = 6333
COLLECTION_NAME = "bpo_problems"
OLLAMA_BASE_URL = "http://192.168.195.69:11434"
WORKER_MODEL    = "llama3.1:latest"
VALIDATOR_MODEL = "phi3:latest"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K_SIMILAR   = 3
MAX_REPAIRS     = 2
MAX_EXAMPLE_CHARS = 800  # truncate RAG examples to avoid token overflow

PROBLEM_TYPES = [
    "Facility Location Problem (FLP)",
    "Network Revenue Management (NRM)",
    "Resource Allocation (RA)",
    "Transportation Problem (TP)",
    "Assignment Problem (AP)",
    "Scheduling Problem (SP)",
    "Other",
]
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class TokenLog:
    classifier_in:  int = 0
    classifier_out: int = 0
    instructor_in:  int = 0
    instructor_out: int = 0
    worker_in:      int = 0
    worker_out:     int = 0
    validator_in:   int = 0
    validator_out:  int = 0
    repair_in:      int = 0
    repair_out:     int = 0

    def total_in(self):
        return (self.classifier_in + self.instructor_in +
                self.worker_in + self.validator_in + self.repair_in)

    def total_out(self):
        return (self.classifier_out + self.instructor_out +
                self.worker_out + self.validator_out + self.repair_out)

    def report(self):
        print("\n=== Token Usage ===")
        print(f"  Classifier  — in: {self.classifier_in:>6}  out: {self.classifier_out:>6}")
        print(f"  Instructor  — in: {self.instructor_in:>6}  out: {self.instructor_out:>6}")
        print(f"  Worker      — in: {self.worker_in:>6}  out: {self.worker_out:>6}")
        print(f"  Validator   — in: {self.validator_in:>6}  out: {self.validator_out:>6}")
        print(f"  Repair      — in: {self.repair_in:>6}  out: {self.repair_out:>6}")
        print(f"  TOTAL       — in: {self.total_in():>6}  out: {self.total_out():>6}")

    def to_dict(self):
        return {
            "classifier_in":  self.classifier_in,
            "classifier_out": self.classifier_out,
            "instructor_in":  self.instructor_in,
            "instructor_out": self.instructor_out,
            "worker_in":      self.worker_in,
            "worker_out":     self.worker_out,
            "validator_in":   self.validator_in,
            "validator_out":  self.validator_out,
            "repair_in":      self.repair_in,
            "repair_out":     self.repair_out,
            "total_in":       self.total_in(),
            "total_out":      self.total_out(),
        }


def get_token_counts(response) -> tuple[int, int]:
    usage = getattr(response, "usage_metadata", None)
    if usage:
        return usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    return 0, 0


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        timeout=30,
        check_compatibility=False,
    )


# ── RAG ───────────────────────────────────────────────────────────────────────

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def retrieve_similar_problems(
    query: str,
    top_k: int = TOP_K_SIMILAR,
    problem_type: str = None,
) -> list[dict]:
    model = get_embedding_model()
    query_vector = model.encode(query).tolist()
    client = get_qdrant_client()

    query_filter = None
    if problem_type and problem_type != "Other":
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        query_filter = Filter(
            must=[FieldCondition(
                key="problem_type",
                match=MatchValue(value=problem_type)
            )]
        )

    try:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )
        points = results.points
    except Exception:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        points = results.points

    return [
        {
            "problem_id":  r.payload["problem_id"],
            "description": r.payload["description"][:MAX_EXAMPLE_CHARS],
            "label":       r.payload["label"][:MAX_EXAMPLE_CHARS],
            "score":       r.score,
        }
        for r in points
    ]


def format_few_shot(similar: list[dict]) -> str:
    context = ""
    for i, s in enumerate(similar):
        context += f"\n--- Example {i+1} (similarity: {s['score']:.3f}) ---\n"
        context += f"PROBLEM:\n{s['description']}\n\n"
        context += f"FORMULATION SNIPPET:\n{s['label']}\n"
    return context


# ── Agent 1: Classifier ───────────────────────────────────────────────────────

def classification_agent(problem_description: str, token_log: TokenLog) -> str:
    print("\n[Classifier] Identifying problem type...")
    similar = retrieve_similar_problems(problem_description, top_k=3)
    few_shot = format_few_shot(similar)

    llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=WORKER_MODEL, temperature=0.0)

    system_msg = SystemMessage(content=(
        "You are an operations research expert. "
        "Classify the given optimization problem into exactly one of these types: "
        f"{', '.join(PROBLEM_TYPES)}. "
        "Reply with ONLY the problem type name, nothing else."
    ))

    human_msg = HumanMessage(content=(
        f"Reference problems:\n{few_shot}\n\n"
        f"Classify this problem:\n{problem_description[:600]}\n\n"
        f"Problem type:"
    ))

    response = llm.invoke([system_msg, human_msg])
    in_tok, out_tok = get_token_counts(response)
    token_log.classifier_in  += in_tok
    token_log.classifier_out += out_tok

    result = response.content.strip()
    problem_type = "Other"
    for pt in PROBLEM_TYPES:
        if any(word in result for word in pt.split()):
            problem_type = pt
            break

    print(f"[Classifier] Type: {problem_type} (tokens in: {in_tok}, out: {out_tok})")
    return problem_type


# ── Agent 2: Instructor ───────────────────────────────────────────────────────

def instructor_agent(
    problem_description: str,
    problem_type: str,
    token_log: TokenLog,
) -> str:
    print(f"\n[Instructor] Retrieving examples for type: {problem_type}...")
    similar = retrieve_similar_problems(
        problem_description, top_k=TOP_K_SIMILAR, problem_type=problem_type
    )
    few_shot = format_few_shot(similar)
    print(f"[Instructor] Retrieved {len(similar)} examples")

    llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=WORKER_MODEL, temperature=0.0)

    system_msg = SystemMessage(content=(
        f"You are an expert optimization instructor for {problem_type} problems. "
        "Using the reference examples, give concise structured instructions "
        "for formulating this problem. Max 250 words. Cover: "
        "(1) decision variables, (2) objective function, (3) constraints."
    ))

    human_msg = HumanMessage(content=(
        f"Reference examples:\n{few_shot}\n\n"
        f"Problem:\n{problem_description[:500]}\n\n"
        f"Modeling instructions:"
    ))

    response = llm.invoke([system_msg, human_msg])
    in_tok, out_tok = get_token_counts(response)
    token_log.instructor_in  += in_tok
    token_log.instructor_out += out_tok

    print(f"[Instructor] Done (tokens in: {in_tok}, out: {out_tok})")
    return response.content


# ── Agent 3: Worker ───────────────────────────────────────────────────────────

def worker_agent(
    problem_description: str,
    problem_type: str,
    instructor_guidance: str,
    token_log: TokenLog,
    feedback: str = None,
    attempt: int = 1,
) -> str:
    label = f"[Worker attempt {attempt}]"
    print(f"\n{label} Generating formulation...")

    similar = retrieve_similar_problems(
        problem_description, top_k=2, problem_type=problem_type
    )
    few_shot = format_few_shot(similar)

    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=WORKER_MODEL,
        temperature=0.1 if attempt == 1 else 0.3,
    )

    system_msg = SystemMessage(content=(
        f"You are an expert OR modeler solving a {problem_type} problem. "
        "Write a complete Python PuLP model. Rules: "
        "Use LpMinimize for minimization, LpMaximize for maximization. "
        "Read all data from CSV files using pandas. "
        "Include EVERY constraint — missing constraints cause infeasibility. "
        "Comment each section clearly."
    ))

    feedback_section = ""
    if feedback:
        feedback_section = (
            f"\n\nVALIDATOR FEEDBACK FROM PREVIOUS ATTEMPT:\n{feedback}\n"
            f"You MUST fix every issue listed above.\n"
        )

    human_msg = HumanMessage(content=(
        f"REFERENCE EXAMPLES:\n{few_shot}\n\n"
        f"INSTRUCTOR GUIDANCE:\n{instructor_guidance}\n"
        f"{feedback_section}"
        f"PROBLEM:\n{problem_description[:500]}\n\n"
        f"Complete Python PuLP formulation:"
    ))

    response = llm.invoke([system_msg, human_msg])
    in_tok, out_tok = get_token_counts(response)

    if attempt == 1:
        token_log.worker_in  += in_tok
        token_log.worker_out += out_tok
    else:
        token_log.repair_in  += in_tok
        token_log.repair_out += out_tok

    print(f"{label} Done (tokens in: {in_tok}, out: {out_tok})")
    return response.content


# ── Agent 4: Validator ────────────────────────────────────────────────────────

def validator_agent(
    problem_description: str,
    formulation: str,
    token_log: TokenLog,
) -> dict:
    print("\n[Validator] Validating formulation...")

    llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=VALIDATOR_MODEL, temperature=0.0)

    system_msg = SystemMessage(content=(
        "You are an optimization model validator. "
        "Check the formulation on 4 dimensions:\n"
        "1. FEASIBILITY — are all constraints present?\n"
        "2. OBJECTIVE — is min/max correct?\n"
        "3. VARIABLES — properly defined?\n"
        "4. COMPLETENESS — anything missing?\n\n"
        "Be specific about what is wrong and how to fix it. "
        "End your response with exactly one of:\n"
        "VERDICT: PASS\nVERDICT: PARTIAL\nVERDICT: FAIL"
    ))

    human_msg = HumanMessage(content=(
        f"PROBLEM:\n{problem_description[:400]}\n\n"
        f"FORMULATION:\n{formulation[:1200]}\n\n"
        f"Validation:"
    ))

    response = llm.invoke([system_msg, human_msg])
    in_tok, out_tok = get_token_counts(response)
    token_log.validator_in  += in_tok
    token_log.validator_out += out_tok

    content = response.content
    verdict = "UNKNOWN"
    for v in ["PASS", "PARTIAL", "FAIL"]:
        if f"VERDICT: {v}" in content.upper():
            verdict = v
            break
    if verdict == "UNKNOWN":
        for v in ["PASS", "PARTIAL", "FAIL"]:
            if v in content.upper():
                verdict = v
                break

    print(f"[Validator] Verdict: {verdict} (tokens in: {in_tok}, out: {out_tok})")
    return {"verdict": verdict, "assessment": content}


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def run_full_pipeline(problem_description: str, token_log: TokenLog) -> dict:
    """
    Classifier → Instructor → Worker → Validator → Repair loop
    """
    problem_type = classification_agent(problem_description, token_log)
    guidance = instructor_agent(problem_description, problem_type, token_log)
    formulation = worker_agent(
        problem_description, problem_type, guidance, token_log, attempt=1
    )
    result = validator_agent(problem_description, formulation, token_log)
    verdict = result["verdict"]
    feedback = result["assessment"]

    repairs = 0
    repair_history = []

    while verdict in ("PARTIAL", "FAIL") and repairs < MAX_REPAIRS:
        repairs += 1
        print(f"\n[Repair {repairs}/{MAX_REPAIRS}] Verdict was {verdict}, retrying...")
        repair_history.append({
            "attempt": repairs,
            "verdict_before": verdict,
            "feedback": feedback[:300],
        })

        formulation = worker_agent(
            problem_description, problem_type, guidance, token_log,
            feedback=feedback, attempt=repairs + 1
        )
        result = validator_agent(problem_description, formulation, token_log)
        verdict = result["verdict"]
        feedback = result["assessment"]

        if verdict == "PASS":
            print(f"[Repair] PASS achieved after {repairs} repair(s)!")
            break

    return {
        "problem_type":   problem_type,
        "guidance":       guidance,
        "formulation":    formulation,
        "verdict":        verdict,
        "assessment":     feedback,
        "repairs":        repairs,
        "repair_history": repair_history,
    }

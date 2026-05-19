"""
agents.py - v3
--------------
Full pipeline inspired by LEAN-LLM-OPT:
  1. Classification Agent   — RAG → identifies problem type
  2. Instructor Agent       — RAG filtered by type → builds guidance
  3. Worker Agent           — RAG + instructor guidance → formulation
  4. Validator Agent        — checks formulation → PASS / PARTIAL / FAIL + feedback
  5. Repair Loop            — if PARTIAL or FAIL, worker retries with feedback (max 2x)

  Triangle Scoring (v3):
  ─────────────────────
  A. Accuracy Validator     — correctness of formulation (weighted most)
  B. Interpretability Score — constraint/variable count + human readability
  C. Efficiency Score       — estimated solver complexity + tightness

Provider switching:
  Set PROVIDER=ollama  → uses local Ollama server
  Set PROVIDER=openai  → uses OpenAI API (GPT-4.1 etc.)
  All config lives in .env — no hardcoded values.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from dotenv import load_dotenv
import os
import re
import pandas as pd
from pathlib import Path

load_dotenv()

# ── Provider Config ────────────────────────────────────────────────────────────
PROVIDER = os.getenv("PROVIDER", "ollama")  # "ollama" or "openai"

QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", "6333"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

WORKER_MODEL = (
    os.getenv("OPENAI_WORKER_MODEL", "gpt-4.1")
    if PROVIDER == "openai"
    else os.getenv("OLLAMA_WORKER_MODEL", "llama3.1:latest")
)
VALIDATOR_MODEL = (
    os.getenv("OPENAI_VALIDATOR_MODEL", "gpt-4.1")
    if PROVIDER == "openai"
    else os.getenv("OLLAMA_VALIDATOR_MODEL", "phi3:latest")
)

# ── Architecture Constants (stay in code — these are design decisions) ─────────
COLLECTION_NAME   = "bpo_problems"
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
TOP_K_SIMILAR     = 3
MAX_REPAIRS       = 2
MAX_EXAMPLE_CHARS = 800

# ── Triangle Hyperparameters ───────────────────────────────────────────────────
# Weights must sum to 1.0. Accuracy is weighted most per Arik's direction.
TRIANGLE_WEIGHT_ACCURACY         = float(os.getenv("TRIANGLE_WEIGHT_ACCURACY",         "0.50"))
TRIANGLE_WEIGHT_INTERPRETABILITY = float(os.getenv("TRIANGLE_WEIGHT_INTERPRETABILITY", "0.30"))
TRIANGLE_WEIGHT_EFFICIENCY       = float(os.getenv("TRIANGLE_WEIGHT_EFFICIENCY",       "0.20"))

# Interpretability thresholds — beyond these, score degrades
MAX_ACCEPTABLE_CONSTRAINTS = int(os.getenv("MAX_ACCEPTABLE_CONSTRAINTS", "50"))
MAX_ACCEPTABLE_VARIABLES   = int(os.getenv("MAX_ACCEPTABLE_VARIABLES",   "100"))

PROBLEM_TYPES = [
    "Facility Location Problem (FLP)",
    "Network Revenue Management (NRM)",
    "Resource Allocation (RA)",
    "Transportation Problem (TP)",
    "Assignment Problem (AP)",
    "Scheduling Problem (SP)",
    "Other",
]

# Data directory — used by read_csv_context to find actual CSV files
DATA_DIR = Path(os.getenv("DATA_DIR", "Large_Scale_Or_Files/Other_example"))


# ── CSV Context Reader (matches paper's data-handling approach) ────────────────

def read_csv_context(problem_id: str) -> str:
    """
    Read actual CSV files from the problem folder and return their
    column names and sample rows as a string for the Worker prompt.

    This matches the LEAN-LLM-OPT paper's approach: the workflow agent
    reads dataset structure at runtime so the model agent never has to
    guess filenames or column names.

    Returns a formatted string like:
      ACTUAL DATA FILES IN THIS PROBLEM:
      File: cost.csv
        Columns: ['plant', 'fixed_cost', 'capacity', 'C1', 'C2', ...]
        Row 1: {'plant': 'F1', 'fixed_cost': 11250, ...}
    """
    folder = DATA_DIR / str(problem_id)
    if not folder.exists():
        return ""

    csv_files = list(folder.glob("*.csv"))
    if not csv_files:
        return ""

    context = "ACTUAL DATA FILES IN THIS PROBLEM:\n"
    for csv_file in sorted(csv_files):
        try:
            df = pd.read_csv(csv_file, nrows=2)
            context += f"\nFile: {csv_file.name}\n"
            context += f"  Columns: {list(df.columns)}\n"
            if len(df) > 0:
                context += f"  Row 1:   {df.iloc[0].to_dict()}\n"
        except Exception as e:
            context += f"\nFile: {csv_file.name} (could not read: {e})\n"

    context += (
        "\nIMPORTANT: Use EXACTLY these filenames and column names in your code. "
        "Do not invent or guess any filenames or column names.\n"
    )
    return context

# ── LLM Factory ───────────────────────────────────────────────────────────────

def get_llm(model: str, temperature: float = 0.0):
    """
    Returns the appropriate LLM based on PROVIDER env var.
    Switch between ollama and openai by changing .env only.
    """
    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=model,
            temperature=temperature,
        )

# ── Token Logging ─────────────────────────────────────────────────────────────

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
        print(f"  Provider    — {PROVIDER.upper()} / worker: {WORKER_MODEL}")
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


# ── Qdrant Client ─────────────────────────────────────────────────────────────

def get_qdrant_client():
    from qdrant_client import QdrantClient
    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        timeout=30,
        check_compatibility=False,
    )


# ── RAG ───────────────────────────────────────────────────────────────────────

_embedding_model = None

def get_embedding_model():
    from sentence_transformers import SentenceTransformer
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
    print(f"\n[Classifier] Identifying problem type... (provider: {PROVIDER})")
    similar = retrieve_similar_problems(problem_description, top_k=3)
    few_shot = format_few_shot(similar)

    from langchain_core.messages import HumanMessage, SystemMessage
    llm = get_llm(WORKER_MODEL, temperature=0.0)

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

    from langchain_core.messages import HumanMessage, SystemMessage
    llm = get_llm(WORKER_MODEL, temperature=0.0)

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
    csv_context: str = "",
) -> str:
    label = f"[Worker attempt {attempt}]"
    print(f"\n{label} Generating formulation... (provider: {PROVIDER})")

    similar = retrieve_similar_problems(
        problem_description, top_k=2, problem_type=problem_type
    )
    few_shot = format_few_shot(similar)

    from langchain_core.messages import HumanMessage, SystemMessage
    llm = get_llm(WORKER_MODEL, temperature=0.1 if attempt == 1 else 0.3)

    system_msg = SystemMessage(content=(
        f"You are an expert OR modeler solving a {problem_type} problem. "
        "Write a complete, EXECUTABLE Python PuLP model. Rules:\n"
        "1. Use LpMinimize for minimization, LpMaximize for maximization.\n"
        "2. Read all data from CSV files using pandas — use EXACT filenames provided.\n"
        "3. Include EVERY constraint — missing constraints cause infeasibility.\n"
        "4. ALWAYS call prob.solve() or model.solve() at the end — never comment it out.\n"
        "5. ALWAYS print the objective value like this after solving:\n"
        "   print(f\'Objective value: {value(model.objective):.4f}\')\n"
        "6. Keep the model compact — avoid redundant constraints.\n"
        "7. Comment each section clearly."
    ))

    feedback_section = ""
    if feedback:
        feedback_section = (
            f"\n\nVALIDATOR FEEDBACK FROM PREVIOUS ATTEMPT:\n{feedback}\n"
            f"You MUST fix every issue listed above.\n"
        )

    csv_section = f"\n{csv_context}\n" if csv_context else ""

    human_msg = HumanMessage(content=(
        f"REFERENCE EXAMPLES:\n{few_shot}\n\n"
        f"INSTRUCTOR GUIDANCE:\n{instructor_guidance}\n"
        f"{csv_section}"
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


# ── Agent 4: Validator (Accuracy) ────────────────────────────────────────────

def validator_agent(
    problem_description: str,
    formulation: str,
    token_log: TokenLog,
) -> dict:
    """
    Accuracy validator — checks logical correctness of the formulation.
    Key insight from Arik: a wrong constraint can still yield the same
    feasible region. The validator must reason rigorously about whether
    the constraints are semantically correct, not just syntactically present.
    """
    print("\n[Validator] Validating formulation (accuracy)...")

    from langchain_core.messages import HumanMessage, SystemMessage
    llm = get_llm(VALIDATOR_MODEL, temperature=0.0)

    system_msg = SystemMessage(content=(
        "You are a rigorous optimization model validator. "
        "Check the formulation on these 4 dimensions:\n\n"
        "1. FEASIBILITY — Are all required constraints present AND semantically correct?\n"
        "   WARNING: A wrong constraint may accidentally preserve the same feasible region.\n"
        "   Check that each constraint encodes exactly what the problem requires.\n\n"
        "2. OBJECTIVE — Is minimize/maximize set correctly? Is the objective function complete?\n\n"
        "3. VARIABLES — Are decision variables properly typed (binary/continuous/integer) "
        "with correct bounds?\n\n"
        "4. COMPLETENESS — Is anything missing? Are all problem requirements captured?\n\n"
        "Be specific about what is wrong and exactly how to fix it. "
        "Do not say PASS if any constraint is semantically wrong, even if the "
        "feasible region appears similar.\n\n"
        "End your response with exactly one of:\n"
        "VERDICT: PASS\nVERDICT: PARTIAL\nVERDICT: FAIL"
    ))

    human_msg = HumanMessage(content=(
        f"PROBLEM:\n{problem_description[:400]}\n\n"
        f"FORMULATION (full code):\n{formulation[:4000]}\n\n"
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

    print(f"[Validator] Accuracy verdict: {verdict} (tokens in: {in_tok}, out: {out_tok})")
    return {"verdict": verdict, "assessment": content}


# ── Triangle Validator A: Accuracy Score ──────────────────────────────────────

def score_accuracy(verdict: str) -> float:
    """
    Converts validator verdict to a numeric accuracy score [0.0, 1.0].
    PASS=1.0, PARTIAL=0.5, FAIL=0.0, UNKNOWN=0.0
    """
    return {"PASS": 1.0, "PARTIAL": 0.5, "FAIL": 0.0}.get(verdict, 0.0)


# ── Triangle Validator B: Interpretability Score ──────────────────────────────

def score_interpretability(formulation: str) -> dict:
    """
    Measures how interpretable/verifiable the formulation is for a human.

    Arik's concern: 5000 constraints + 3000 variables = unverifiable.
    Scoring logic:
      - Count constraints (lines with <= >= == in PuLP code)
      - Count variables (LpVariable declarations)
      - Penalize exponentially beyond thresholds
      - Check for comments (good for interpretability)
      - Check for meaningful variable names (not just x1, x2, x3...)

    Returns score [0.0, 1.0] and breakdown dict.
    """
    lines = formulation.split("\n")

    # Count constraints — lines that add constraints to the model
    constraint_patterns = [
        r"\+\s*=\s*lpSum",   # model += lpSum(...)
        r"\+\s*=\s*\(",    # model += (expr)
        r"addConstraint",    # explicit addConstraint calls
        r"<=|>=|==",         # inequality/equality expressions
    ]
    constraint_count = 0
    for line in lines:
        if any(re.search(p, line) for p in constraint_patterns):
            if "LpVariable" not in line and "LpProblem" not in line:
                constraint_count += 1

    # Count variables — LpVariable declarations
    variable_count = len(re.findall(r"LpVariable", formulation))

    # Count comments — indicator of human readability
    comment_count = len([l for l in lines if l.strip().startswith("#")])

    # Check for meaningful names (not just single letters)
    has_meaningful_names = bool(re.search(
        r"LpVariable\(['\"](?!x\d|y\d|z\d)[a-zA-Z]{2,}", formulation
    ))

    # Score constraints: full marks up to threshold, then degrades
    if constraint_count <= MAX_ACCEPTABLE_CONSTRAINTS:
        constraint_score = 1.0
    else:
        # Exponential penalty beyond threshold
        excess = constraint_count - MAX_ACCEPTABLE_CONSTRAINTS
        constraint_score = max(0.0, 1.0 - (excess / MAX_ACCEPTABLE_CONSTRAINTS))

    # Score variables: same logic
    if variable_count <= MAX_ACCEPTABLE_VARIABLES:
        variable_score = 1.0
    else:
        excess = variable_count - MAX_ACCEPTABLE_VARIABLES
        variable_score = max(0.0, 1.0 - (excess / MAX_ACCEPTABLE_VARIABLES))

    # Comment bonus: up to 0.1 extra weight toward readability
    comment_score = min(1.0, comment_count / 5.0)

    # Name bonus
    name_score = 1.0 if has_meaningful_names else 0.7

    # Composite interpretability score
    interp_score = (
        0.40 * constraint_score +
        0.35 * variable_score   +
        0.15 * comment_score    +
        0.10 * name_score
    )

    return {
        "score":              round(interp_score, 4),
        "constraint_count":   constraint_count,
        "variable_count":     variable_count,
        "comment_count":      comment_count,
        "has_meaningful_names": has_meaningful_names,
        "constraint_score":   round(constraint_score, 4),
        "variable_score":     round(variable_score, 4),
    }


# ── Triangle Validator C: Efficiency Score ────────────────────────────────────

def score_efficiency(formulation: str, problem_type: str) -> dict:
    """
    Estimates how efficiently this MILP will solve.

    Arik's point: interpretable MILPs may still be slow to solve
    (worst case exponential for branch-and-bound). This score estimates:
      - Binary variable count (more = harder)
      - Continuous variable count
      - Constraint tightness (presence of bound-tightening patterns)
      - Problem type complexity (FLP is NP-hard, TP is polynomial)
      - Presence of cutting plane hints in the formulation

    Returns score [0.0, 1.0] and breakdown dict.
    Note: actual solver runtime measurement requires running the code,
    which we do not do here for safety. This is a static estimate.
    """

    # Count binary variables
    binary_count = len(re.findall(
        r"LpVariable\([^)]*cat\s*=\s*['\"]Binary['\"]", formulation
    ))

    # Count integer variables
    integer_count = len(re.findall(
        r"LpVariable\([^)]*cat\s*=\s*['\"]Integer['\"]", formulation
    ))

    # Count continuous variables
    continuous_count = len(re.findall(
        r"LpVariable\([^)]*cat\s*=\s*['\"]Continuous['\"]", formulation
    ))

    # Detect bound tightening (good for efficiency)
    has_bounds = bool(re.search(r"lowBound|upBound", formulation))

    # Detect Big-M patterns (bad for efficiency — weak relaxation)
    has_big_m = bool(re.search(r"\bM\b|\bBIG_M\b|big_m|bigM", formulation))

    # Problem type complexity mapping
    type_complexity = {
        "Transportation Problem (TP)":         0.9,  # polynomial
        "Assignment Problem (AP)":             0.8,  # polynomial (Hungarian)
        "Resource Allocation (RA)":            0.7,  # often LP-relaxable
        "Network Revenue Management (NRM)":    0.7,
        "Facility Location Problem (FLP)":     0.4,  # NP-hard
        "Scheduling Problem (SP)":             0.3,  # NP-hard in general
        "Other":                               0.5,
    }
    type_score = type_complexity.get(problem_type, 0.5)

    # Binary variable penalty
    if binary_count == 0:
        binary_score = 1.0  # pure LP, very fast
    elif binary_count <= 10:
        binary_score = 0.8
    elif binary_count <= 50:
        binary_score = 0.5
    else:
        binary_score = max(0.1, 1.0 - (binary_count / 200))

    # Big-M penalty (weak LP relaxation = slower B&B)
    big_m_penalty = 0.2 if has_big_m else 0.0

    # Bounds bonus
    bounds_bonus = 0.1 if has_bounds else 0.0

    efficiency_score = max(0.0, min(1.0,
        0.40 * type_score   +
        0.40 * binary_score +
        bounds_bonus        -
        big_m_penalty
    ))

    return {
        "score":            round(efficiency_score, 4),
        "binary_count":     binary_count,
        "integer_count":    integer_count,
        "continuous_count": continuous_count,
        "has_bounds":       has_bounds,
        "has_big_m":        has_big_m,
        "type_score":       type_score,
        "binary_score":     round(binary_score, 4),
    }


# ── Triangle Composite Score ───────────────────────────────────────────────────

def compute_triangle_score(
    verdict: str,
    formulation: str,
    problem_type: str,
) -> dict:
    """
    Computes the weighted triangle score from all three validators.

    Weights are set in .env:
      TRIANGLE_WEIGHT_ACCURACY         (default 0.50)
      TRIANGLE_WEIGHT_INTERPRETABILITY (default 0.30)
      TRIANGLE_WEIGHT_EFFICIENCY       (default 0.20)

    Arik's direction: accuracy is most important.
    The weights reflect the relative importance per the triangle framework.
    """
    accuracy_score    = score_accuracy(verdict)
    interp_result     = score_interpretability(formulation)
    efficiency_result = score_efficiency(formulation, problem_type)

    interp_score     = interp_result["score"]
    efficiency_score = efficiency_result["score"]

    weighted_score = (
        TRIANGLE_WEIGHT_ACCURACY         * accuracy_score  +
        TRIANGLE_WEIGHT_INTERPRETABILITY * interp_score    +
        TRIANGLE_WEIGHT_EFFICIENCY       * efficiency_score
    )

    print(f"\n=== Triangle Scores ===")
    print(f"  Accuracy        ({TRIANGLE_WEIGHT_ACCURACY:.0%} weight): {accuracy_score:.4f}  [{verdict}]")
    print(f"  Interpretability({TRIANGLE_WEIGHT_INTERPRETABILITY:.0%} weight): {interp_score:.4f}  "
          f"[{interp_result['constraint_count']} constraints, "
          f"{interp_result['variable_count']} variables]")
    print(f"  Efficiency      ({TRIANGLE_WEIGHT_EFFICIENCY:.0%} weight): {efficiency_score:.4f}  "
          f"[{efficiency_result['binary_count']} binary vars, "
          f"big_m={efficiency_result['has_big_m']}]")
    print(f"  WEIGHTED TOTAL  : {weighted_score:.4f}")

    return {
        "accuracy_score":        accuracy_score,
        "interpretability_score": interp_score,
        "efficiency_score":      efficiency_score,
        "weighted_triangle_score": round(weighted_score, 4),
        "weights": {
            "accuracy":         TRIANGLE_WEIGHT_ACCURACY,
            "interpretability": TRIANGLE_WEIGHT_INTERPRETABILITY,
            "efficiency":       TRIANGLE_WEIGHT_EFFICIENCY,
        },
        "interpretability_detail": interp_result,
        "efficiency_detail":       efficiency_result,
    }


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def run_full_pipeline(problem_description: str, token_log: TokenLog, problem_id: str = None) -> dict:
    """
    Classifier → Instructor → Worker → Validator → Repair loop → Triangle Score

    Provider and model are read from .env — switch PROVIDER=openai/ollama
    without changing any code.
    """
    print(f"\n{'='*60}")
    print(f"  PIPELINE START — provider: {PROVIDER} / model: {WORKER_MODEL}")
    print(f"{'='*60}")

    # Read actual CSV structure — passed to Worker so it never guesses filenames
    csv_context = read_csv_context(problem_id) if problem_id else ""
    if csv_context:
        print(f"[Data] CSV context loaded for problem {problem_id}")

    problem_type = classification_agent(problem_description, token_log)
    guidance     = instructor_agent(problem_description, problem_type, token_log)
    formulation  = worker_agent(
        problem_description, problem_type, guidance, token_log,
        attempt=1, csv_context=csv_context
    )
    result  = validator_agent(problem_description, formulation, token_log)
    verdict = result["verdict"]
    feedback = result["assessment"]

    repairs = 0
    repair_history = []

    while verdict in ("PARTIAL", "FAIL") and repairs < MAX_REPAIRS:
        repairs += 1
        print(f"\n[Repair {repairs}/{MAX_REPAIRS}] Verdict was {verdict}, retrying...")
        repair_history.append({
            "attempt":        repairs,
            "verdict_before": verdict,
            "feedback":       feedback[:300],
        })

        formulation = worker_agent(
            problem_description, problem_type, guidance, token_log,
            feedback=feedback, attempt=repairs + 1, csv_context=csv_context
        )
        result   = validator_agent(problem_description, formulation, token_log)
        verdict  = result["verdict"]
        feedback = result["assessment"]

        if verdict == "PASS":
            print(f"[Repair] PASS achieved after {repairs} repair(s)!")
            break

    # ── Triangle scoring on final formulation ─────────────────────────────────
    triangle = compute_triangle_score(verdict, formulation, problem_type)

    return {
        "provider":       PROVIDER,
        "worker_model":   WORKER_MODEL,
        "problem_type":   problem_type,
        "guidance":       guidance,
        "formulation":    formulation,
        "verdict":        verdict,
        "assessment":     feedback,
        "repairs":        repairs,
        "repair_history": repair_history,
        "triangle":       triangle,
    }
"""
run.py - v3
-----------
Runs a problem through the full v3 pipeline:
Classifier → Instructor → Worker → Validator → Repair loop → Triangle Score → Execution Check

Accuracy measurement (matching the paper):
  - Extracts Python code from Worker output
  - Executes it in a subprocess sandbox with correct working directory
  - Captures the optimal value from solver output
  - Compares to ground truth optimal value (extracted from label.txt if it is
    executable Gurobi code; otherwise falls back to validator verdict)
  - execution_correct = True/False → this is the paper-style accuracy metric

Provider switching:
  Set PROVIDER=openai in .env  → GPT-4.1
  Set PROVIDER=ollama in .env  → local llama3.1

Usage:
    python run.py --problem_id 3
    python run.py --problem_id 3 --label v3_openai
    python run.py --problem_id 3 --label v3_ollama
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from agents import TokenLog, run_full_pipeline, PROVIDER, WORKER_MODEL

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR        = Path("Large_Scale_Or_Files/Other_example")
RESULTS_LOG     = Path("results_log.csv")
DEFAULT_ID      = "3"
EXEC_TIMEOUT    = 60    # seconds to allow generated code to run
OBJ_TOLERANCE   = 0.01  # 1% tolerance when comparing optimal values

# CSV columns — v3 adds triangle scores, provider info, and execution results
CSV_FIELDS = [
    # Run metadata
    "timestamp", "run_label", "problem_id", "provider", "worker_model",
    # Pipeline results
    "problem_type", "verdict", "repairs",
    # Execution-based accuracy (paper-style)
    "execution_attempted", "execution_success",
    "generated_obj_value", "ground_truth_obj_value",
    "execution_correct", "accuracy_pct",
    # Triangle scores
    "accuracy_score", "interpretability_score", "efficiency_score",
    "weighted_triangle_score",
    # Interpretability detail
    "constraint_count", "variable_count", "comment_count",
    # Efficiency detail
    "binary_count", "has_big_m", "has_bounds",
    # Triangle weights used
    "weight_accuracy", "weight_interpretability", "weight_efficiency",
    # Token usage
    "classifier_in", "classifier_out",
    "instructor_in", "instructor_out",
    "worker_in", "worker_out",
    "validator_in", "validator_out",
    "repair_in", "repair_out",
    "total_in", "total_out",
    # Summary
    "assessment_summary",
]
# ──────────────────────────────────────────────────────────────────────────────


def load_problem(problem_id: str) -> tuple[str, str]:
    """Load problem description and ground truth label from disk."""
    folder = DATA_DIR / problem_id
    desc   = (folder / "problem description.txt").read_text(encoding="utf-8").strip()
    label  = (folder / "label.txt").read_text(encoding="utf-8").strip()
    return desc, label


# ── Code Extraction ───────────────────────────────────────────────────────────

def extract_python_code(formulation: str) -> str | None:
    """
    Extract Python code from the Worker output.
    Handles:
      - ```python ... ``` blocks
      - ``` ... ``` blocks
      - Raw Python code without fences
    """
    # Try ```python ... ``` first
    match = re.search(r"```python\s*(.*?)```", formulation, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try ``` ... ```
    match = re.search(r"```\s*(.*?)```", formulation, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if "import" in code or "LpProblem" in code or "pulp" in code.lower():
            return code

    # If it looks like raw Python
    if "from pulp" in formulation or "import pulp" in formulation:
        return formulation.strip()

    return None


# ── Ground Truth Extraction ───────────────────────────────────────────────────

def parse_objective_from_output(output: str) -> float | None:
    """
    Parse the optimal objective value from solver stdout.
    Handles PuLP CBC output and common print patterns.
    """
    output_lower = output.lower()

    # Try specific keyword hints first
    hints = [
        "objective value:",
        "total cost:",
        "optimal cost:",
        "objective:",
        "obj =",
        "objval",
    ]
    for hint in hints:
        idx = output_lower.find(hint)
        if idx >= 0:
            segment = output_lower[idx:idx+80]
            nums = re.findall(r"[-+]?\d+\.\d+", segment)
            for n in nums:
                val = float(n)
                if val > 0:
                    return val

    # Fallback: largest positive float in output (obj values tend to be large)
    floats = [float(f) for f in re.findall(r"[-+]?\d+\.\d+", output) if float(f) > 0]
    if floats:
        return max(floats)

    return None


def extract_ground_truth_value(problem_id: str) -> float | None:
    """
    Load the ground truth optimal value from ground_truth.json.
    Generated by running: python generate_ground_truth.py
    Uses PuLP-solved values -- consistent with the generated code.
    """
    gt_file = Path("ground_truth.json")
    if not gt_file.exists():
        print("[Execution] ground_truth.json not found.")
        print("[Execution] Run:  python generate_ground_truth.py  first.")
        return None

    with open(gt_file) as f:
        import json as _json
        gt = _json.load(f)

    entry = gt.get(str(problem_id), {})
    val   = entry.get("optimal_value")

    if val is not None:
        print(f"[Execution] Ground truth optimal value (PuLP): {val:.4f}")
    else:
        print(f"[Execution] No ground truth available for problem {problem_id}.")
    return val

def execute_generated_code(code: str, problem_folder: Path) -> dict:
    """
    Execute the generated PuLP code in a subprocess sandbox.
    Working directory is set to the problem folder so CSV reads work.
    Safe: uses subprocess, not eval() or exec().
    """
    print(f"\n[Execution] Running generated code (timeout: {EXEC_TIMEOUT}s)...")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=EXEC_TIMEOUT,
            cwd=str(problem_folder),
        )
        os.unlink(tmp_path)

        success   = result.returncode == 0
        obj_value = parse_objective_from_output(result.stdout) if success else None

        if success:
            print(f"[Execution] ✅ Success. Objective value: {obj_value}")
        else:
            print(f"[Execution] ❌ Failed (returncode {result.returncode})")
            if result.stderr:
                print(f"[Execution] Error: {result.stderr[:400]}")

        return {
            "success":   success,
            "obj_value": obj_value,
            "stdout":    result.stdout[:500],
            "stderr":    result.stderr[:500],
        }

    except subprocess.TimeoutExpired:
        print(f"[Execution] ⏱ Timed out after {EXEC_TIMEOUT}s")
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        return {"success": False, "obj_value": None, "stdout": "", "stderr": "Timeout"}

    except Exception as e:
        print(f"[Execution] Exception: {e}")
        return {"success": False, "obj_value": None, "stdout": "", "stderr": str(e)}


# ── Full Accuracy Check ───────────────────────────────────────────────────────

def check_execution_accuracy(
    formulation: str,
    ground_truth: str,
    problem_id: str,
) -> dict:
    """
    Paper-style accuracy check:
      1. Extract Python code from formulation
      2. Get ground truth optimal value from label.txt
      3. Execute generated code
      4. Compare objective values within OBJ_TOLERANCE
    """
    problem_folder = DATA_DIR / problem_id

    result = {
        "execution_attempted":    False,
        "execution_success":      False,
        "generated_obj_value":    None,
        "ground_truth_obj_value": None,
        "execution_correct":      None,
        "accuracy_pct":           None,
        "code_extracted":         False,
        "stdout":                 "",
        "stderr":                 "",
    }

    # Step 1: Extract code
    code = extract_python_code(formulation)
    if code is None:
        print("[Execution] Could not extract Python code from formulation.")
        return result
    result["code_extracted"] = True

    # Step 2: Ground truth value
    gt_value = extract_ground_truth_value(problem_id)
    result["ground_truth_obj_value"] = gt_value

    # Step 3: Execute generated code
    result["execution_attempted"] = True
    exec_out = execute_generated_code(code, problem_folder)
    result["execution_success"] = exec_out["success"]
    result["generated_obj_value"] = exec_out["obj_value"]
    result["stdout"] = exec_out["stdout"]
    result["stderr"] = exec_out["stderr"]

    gen_val = exec_out["obj_value"]

    # Step 4: Compare
    if gen_val is not None and gt_value is not None:
        rel_diff   = abs(gen_val - gt_value) / (abs(gt_value) + 1e-9)
        is_correct = rel_diff <= OBJ_TOLERANCE
        result["execution_correct"] = is_correct
        result["accuracy_pct"]      = 100.0 if is_correct else 0.0
        print(f"\n[Accuracy] Generated: {gen_val:.4f} | Ground truth: {gt_value:.4f}")
        print(f"[Accuracy] Relative diff: {rel_diff:.4%} | Correct: {'✅ YES' if is_correct else '❌ NO'}")

    elif gen_val is not None and gt_value is None:
        print(f"[Accuracy] Code ran (obj={gen_val:.4f}) — no ground truth to compare.")
        result["execution_correct"] = None
        result["accuracy_pct"]      = None

    else:
        print("[Accuracy] Code did not produce an objective value.")
        result["execution_correct"] = False
        result["accuracy_pct"]      = 0.0

    return result


# ── Logging ───────────────────────────────────────────────────────────────────

def log_result(
    problem_id: str,
    run_label: str,
    output: dict,
    token_log: TokenLog,
    exec_result: dict,
):
    """Append one row to results_log.csv."""
    file_exists = RESULTS_LOG.exists()
    triangle    = output.get("triangle", {})
    interp      = triangle.get("interpretability_detail", {})
    eff         = triangle.get("efficiency_detail", {})
    weights     = triangle.get("weights", {})

    with open(RESULTS_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp":               datetime.now().isoformat(),
            "run_label":               run_label,
            "problem_id":              problem_id,
            "provider":                output.get("provider", PROVIDER),
            "worker_model":            output.get("worker_model", WORKER_MODEL),
            "problem_type":            output["problem_type"],
            "verdict":                 output["verdict"],
            "repairs":                 output["repairs"],
            "execution_attempted":     exec_result.get("execution_attempted", False),
            "execution_success":       exec_result.get("execution_success", False),
            "generated_obj_value":     exec_result.get("generated_obj_value", ""),
            "ground_truth_obj_value":  exec_result.get("ground_truth_obj_value", ""),
            "execution_correct":       exec_result.get("execution_correct", ""),
            "accuracy_pct":            exec_result.get("accuracy_pct", ""),
            "accuracy_score":          triangle.get("accuracy_score", ""),
            "interpretability_score":  triangle.get("interpretability_score", ""),
            "efficiency_score":        triangle.get("efficiency_score", ""),
            "weighted_triangle_score": triangle.get("weighted_triangle_score", ""),
            "constraint_count":        interp.get("constraint_count", ""),
            "variable_count":          interp.get("variable_count", ""),
            "comment_count":           interp.get("comment_count", ""),
            "binary_count":            eff.get("binary_count", ""),
            "has_big_m":               eff.get("has_big_m", ""),
            "has_bounds":              eff.get("has_bounds", ""),
            "weight_accuracy":         weights.get("accuracy", ""),
            "weight_interpretability": weights.get("interpretability", ""),
            "weight_efficiency":       weights.get("efficiency", ""),
            "classifier_in":           token_log.classifier_in,
            "classifier_out":          token_log.classifier_out,
            "instructor_in":           token_log.instructor_in,
            "instructor_out":          token_log.instructor_out,
            "worker_in":               token_log.worker_in,
            "worker_out":              token_log.worker_out,
            "validator_in":            token_log.validator_in,
            "validator_out":           token_log.validator_out,
            "repair_in":               token_log.repair_in,
            "repair_out":              token_log.repair_out,
            "total_in":                token_log.total_in(),
            "total_out":               token_log.total_out(),
            "assessment_summary":      output["assessment"][:300].replace("\n", " "),
        })
    print(f"✅ Result logged to {RESULTS_LOG}")


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    problem_id: str,
    problem_description: str,
    ground_truth: str,
    run_label: str,
):
    print(f"\n{'='*60}")
    print(f"  LEAN-LLM-OPT Mini v3 — Problem {problem_id} [{run_label}]")
    print(f"  Provider : {PROVIDER.upper()} / Model: {WORKER_MODEL}")
    print(f"{'='*60}")
    print(f"\nPROBLEM:\n{problem_description[:300]}...\n")

    token_log = TokenLog()
    output    = run_full_pipeline(problem_description, token_log, problem_id=problem_id)

    print(f"\n[Pipeline Complete]")
    print(f"  Provider     : {output.get('provider', PROVIDER).upper()}")
    print(f"  Model        : {output.get('worker_model', WORKER_MODEL)}")
    print(f"  Problem type : {output['problem_type']}")
    print(f"  Final verdict: {output['verdict']}")
    print(f"  Repairs done : {output['repairs']}")

    if output["repair_history"]:
        for r in output["repair_history"]:
            print(f"  Repair {r['attempt']}: was {r['verdict_before']}")

    token_log.report()

    # Execution accuracy
    exec_result = check_execution_accuracy(
        output["formulation"], ground_truth, problem_id
    )

    log_result(problem_id, run_label, output, token_log, exec_result)

    # Save JSON
    ts       = datetime.now().strftime("%H%M%S")
    out_file = Path(f"output_{problem_id}_{run_label}_{ts}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "problem_id":          problem_id,
            "run_label":           run_label,
            "provider":            output.get("provider", PROVIDER),
            "worker_model":        output.get("worker_model", WORKER_MODEL),
            "problem_description": problem_description,
            "ground_truth":        ground_truth,
            "problem_type":        output["problem_type"],
            "guidance":            output["guidance"],
            "formulation":         output["formulation"],
            "verdict":             output["verdict"],
            "assessment":          output["assessment"],
            "repairs":             output["repairs"],
            "repair_history":      output["repair_history"],
            "triangle":            output.get("triangle", {}),
            "execution":           exec_result,
            "tokens":              token_log.to_dict(),
        }, f, indent=2)
    print(f"✅ Full output saved to {out_file}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS — Problem {problem_id} [{run_label}]")
    print(f"{'='*60}")
    print(f"  Validator verdict  : {output['verdict']}")
    print(f"  Code executed      : {exec_result.get('execution_success', False)}")
    correct = exec_result.get("execution_correct")
    if correct is True:
        print(f"  Execution correct  : ✅ YES (within {OBJ_TOLERANCE:.0%} of ground truth)")
    elif correct is False:
        print(f"  Execution correct  : ❌ NO")
    else:
        print(f"  Execution correct  : ⚠️  Cannot determine (no ground truth value in label)")
    print(f"  Triangle score     : {output.get('triangle', {}).get('weighted_triangle_score', 'N/A')}")
    print(f"{'='*60}\n")

    return output["verdict"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LEAN-LLM-OPT Mini v3 — triangle scoring + execution accuracy"
    )
    parser.add_argument(
        "--problem_id", type=str, default=DEFAULT_ID,
        help="Problem ID under Large_Scale_Or_Files/Other_example (e.g. 3, 5, 9, 10)"
    )
    parser.add_argument(
        "--label", type=str, default="v3",
        help="Run label for results_log (e.g. v3_openai, v3_ollama)"
    )
    parser.add_argument(
        "--problem", type=str, default=None,
        help="Custom problem text — overrides --problem_id"
    )
    args = parser.parse_args()

    if args.problem:
        run_pipeline("custom", args.problem, "", args.label)
    else:
        desc, label = load_problem(args.problem_id)
        run_pipeline(args.problem_id, desc, label, args.label)
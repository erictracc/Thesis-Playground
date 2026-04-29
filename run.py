"""
run.py - v2
-----------
Runs a problem through the full v2 pipeline:
Classifier → Instructor → Worker → Validator → Repair loop

Usage:
    python run.py --problem_id 3
    python run.py --problem_id 3 --label v2
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from agents import TokenLog, run_full_pipeline

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = Path("Large_Scale_Or_Files/Other_example")
RESULTS_LOG  = Path("results_log.csv")
DEFAULT_ID   = "3"
# ──────────────────────────────────────────────────────────────────────────────


def load_problem(problem_id: str) -> tuple[str, str]:
    folder = DATA_DIR / problem_id
    desc  = (folder / "problem description.txt").read_text(encoding="utf-8").strip()
    label = (folder / "label.txt").read_text(encoding="utf-8").strip()
    return desc, label


def log_result(problem_id: str, run_label: str, output: dict, token_log: TokenLog):
    file_exists = RESULTS_LOG.exists()
    with open(RESULTS_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "run_label", "problem_id", "problem_type",
            "verdict", "repairs",
            "classifier_in", "classifier_out",
            "instructor_in", "instructor_out",
            "worker_in", "worker_out",
            "validator_in", "validator_out",
            "repair_in", "repair_out",
            "total_in", "total_out",
            "assessment_summary",
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp":          datetime.now().isoformat(),
            "run_label":          run_label,
            "problem_id":         problem_id,
            "problem_type":       output["problem_type"],
            "verdict":            output["verdict"],
            "repairs":            output["repairs"],
            "classifier_in":      token_log.classifier_in,
            "classifier_out":     token_log.classifier_out,
            "instructor_in":      token_log.instructor_in,
            "instructor_out":     token_log.instructor_out,
            "worker_in":          token_log.worker_in,
            "worker_out":         token_log.worker_out,
            "validator_in":       token_log.validator_in,
            "validator_out":      token_log.validator_out,
            "repair_in":          token_log.repair_in,
            "repair_out":         token_log.repair_out,
            "total_in":           token_log.total_in(),
            "total_out":          token_log.total_out(),
            "assessment_summary": output["assessment"][:300].replace("\n", " "),
        })
    print(f"✅ Result logged to {RESULTS_LOG}")


def run_pipeline(problem_id: str, problem_description: str,
                 ground_truth: str, run_label: str):

    print(f"\n{'='*60}")
    print(f"  LEAN-LLM-OPT Mini v2 — Problem {problem_id} [{run_label}]")
    print(f"{'='*60}")
    print(f"\nPROBLEM:\n{problem_description[:300]}...\n")

    token_log = TokenLog()
    output = run_full_pipeline(problem_description, token_log)

    # Print summaries
    print(f"\n[Pipeline Complete]")
    print(f"  Problem type : {output['problem_type']}")
    print(f"  Final verdict: {output['verdict']}")
    print(f"  Repairs done : {output['repairs']}")

    if output["repair_history"]:
        for r in output["repair_history"]:
            print(f"  Repair {r['attempt']}: was {r['verdict_before']}")

    token_log.report()

    # Log CSV
    log_result(problem_id, run_label, output, token_log)

    # Save full JSON
    ts = datetime.now().strftime("%H%M%S")
    out_file = Path(f"output_{problem_id}_{run_label}_{ts}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "problem_id":         problem_id,
            "run_label":          run_label,
            "problem_description": problem_description,
            "ground_truth":       ground_truth,
            "problem_type":       output["problem_type"],
            "guidance":           output["guidance"],
            "formulation":        output["formulation"],
            "verdict":            output["verdict"],
            "assessment":         output["assessment"],
            "repairs":            output["repairs"],
            "repair_history":     output["repair_history"],
            "tokens":             token_log.to_dict(),
        }, f, indent=2)
    print(f"✅ Full output saved to {out_file}")

    return output["verdict"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_id", type=str, default=DEFAULT_ID)
    parser.add_argument("--label", type=str, default="v2",
                        help="Label for this run in results_log (e.g. v1, v2, repair)")
    parser.add_argument("--problem", type=str, default=None,
                        help="Custom problem text (overrides --problem_id)")
    args = parser.parse_args()

    if args.problem:
        run_pipeline("custom", args.problem, "", args.label)
    else:
        desc, label = load_problem(args.problem_id)
        run_pipeline(args.problem_id, desc, label, args.label)

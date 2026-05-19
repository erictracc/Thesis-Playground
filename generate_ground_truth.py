"""
generate_ground_truth.py
------------------------
Solves all 4 test problems (3, 5, 9, 10) using PuLP and saves the
optimal objective values to ground_truth.json.

This gives us paper-style accuracy measurement — we compare the
pipeline's generated code output against these known optimal values.

Run once before running experiments:
    python generate_ground_truth.py

Problems:
  3  — Capacitated Facility Location (15 plants × 15 customers)
  5  — Capacitated Facility Location (10 centres × 15 customers)
  9  — Transportation Problem (10 sources × 20 destinations)
  10 — Assignment Problem (12 machines × 12 tasks)
"""

import json
from pathlib import Path
import pandas as pd
from pulp import (
    LpProblem, LpMinimize, LpVariable, LpBinary, LpInteger,
    lpSum, value, PULP_CBC_CMD, LpStatus
)

DATA_DIR   = Path("Large_Scale_Or_Files/Other_example")
OUT_FILE   = Path("ground_truth.json")
TOLERANCE  = 1e-4


def solve_problem_3() -> float:
    """
    Problem 3 — Capacitated Facility Location
    15 plants (F1-F15), 15 customers (C1-C15)
    CSV: cost.csv (plant, fixed_cost, capacity, C1..C15)
         demand.csv (customer, demand)
    Objective: minimize fixed opening costs + transportation costs
    """
    print("\n[Problem 3] Solving Capacitated FLP (15 plants × 15 customers)...")
    folder = DATA_DIR / "3"

    cost_df   = pd.read_csv(folder / "cost.csv")
    demand_df = pd.read_csv(folder / "demand.csv")

    plants    = cost_df["plant"].tolist()
    customers = demand_df["customer"].tolist()

    fixed_cost = dict(zip(cost_df["plant"], cost_df["fixed_cost"]))
    capacity   = dict(zip(cost_df["plant"], cost_df["capacity"]))
    demand     = dict(zip(demand_df["customer"], demand_df["demand"]))

    # Transport cost: cost_df has columns C1..C15 for each plant row
    transport = {}
    for _, row in cost_df.iterrows():
        p = row["plant"]
        for c in customers:
            transport[(p, c)] = row[c]

    prob = LpProblem("FLP_3", LpMinimize)

    # y[p] = 1 if plant p is opened
    y = {p: LpVariable(f"y_{p}", cat=LpBinary) for p in plants}
    # x[p,c] = units shipped from plant p to customer c
    x = {(p, c): LpVariable(f"x_{p}_{c}", lowBound=0) for p in plants for c in customers}

    # Objective
    prob += (
        lpSum(fixed_cost[p] * y[p] for p in plants) +
        lpSum(transport[(p, c)] * x[(p, c)] for p in plants for c in customers)
    )

    # Demand satisfaction
    for c in customers:
        prob += lpSum(x[(p, c)] for p in plants) >= demand[c], f"demand_{c}"

    # Capacity constraints
    for p in plants:
        prob += lpSum(x[(p, c)] for c in customers) <= capacity[p] * y[p], f"cap_{p}"

    prob.solve(PULP_CBC_CMD(msg=0))

    obj = value(prob.objective)
    print(f"[Problem 3] Status: {LpStatus[prob.status]} | Optimal value: {obj:.4f}")
    return round(obj, 4)


def solve_problem_5() -> float:
    """
    Problem 5 — Capacitated Facility Location
    10 service centres (SC1-SC10), 15 customers (C1-C15)
    Each opened centre can serve at most 4 customers (binary assignment)
    CSVs: service_centers_fixed_costs.csv, expanded_customer_service_costs.csv
    """
    print("\n[Problem 5] Solving Capacitated FLP (10 centres × 15 customers)...")
    folder = DATA_DIR / "5"

    fixed_df  = pd.read_csv(folder / "service_centers_fixed_costs.csv")
    service_df = pd.read_csv(folder / "expanded_customer_service_costs.csv")

    centres   = fixed_df["Service Center"].tolist()
    customers = service_df["Customer"].tolist()

    fixed_cost   = dict(zip(fixed_df["Service Center"], fixed_df["Fixed Opening Cost"]))
    service_cost = {}
    for _, row in service_df.iterrows():
        c = row["Customer"]
        for sc in centres:
            service_cost[(c, sc)] = row[sc]

    CAP = 4  # each centre serves at most 4 customers

    prob = LpProblem("FLP_5", LpMinimize)

    y = {sc: LpVariable(f"y_{sc}", cat=LpBinary) for sc in centres}
    x = {(c, sc): LpVariable(f"x_{c}_{sc}", cat=LpBinary)
         for c in customers for sc in centres}

    # Objective
    prob += (
        lpSum(fixed_cost[sc] * y[sc] for sc in centres) +
        lpSum(service_cost[(c, sc)] * x[(c, sc)] for c in customers for sc in centres)
    )

    # Each customer assigned to exactly one centre
    for c in customers:
        prob += lpSum(x[(c, sc)] for sc in centres) == 1, f"assign_{c}"

    # Assignment only to open centres
    for c in customers:
        for sc in centres:
            prob += x[(c, sc)] <= y[sc], f"open_{c}_{sc}"

    # Capacity: at most 4 customers per centre
    for sc in centres:
        prob += lpSum(x[(c, sc)] for c in customers) <= CAP * y[sc], f"cap_{sc}"

    prob.solve(PULP_CBC_CMD(msg=0))

    obj = value(prob.objective)
    print(f"[Problem 5] Status: {LpStatus[prob.status]} | Optimal value: {obj:.4f}")
    return round(obj, 4)


def solve_problem_9() -> float:
    """
    Problem 9 — Transportation Problem
    10 sources (S1-S10), 20 destinations (D1-D20)
    Integer number of trucks per route (10 units per truck)
    CSVs: expanded_cost_matrix.csv, expanded_sources.csv, expanded_destinations.csv
    """
    print("\n[Problem 9] Solving Transportation Problem (10 sources × 20 destinations)...")
    folder = DATA_DIR / "9"

    cost_df = pd.read_csv(folder / "expanded_cost_matrix.csv")
    src_df  = pd.read_csv(folder / "expanded_sources.csv")
    dst_df  = pd.read_csv(folder / "expanded_destinations.csv")

    sources  = cost_df["source_id"].tolist()
    dests    = [c for c in cost_df.columns if c != "source_id"]

    # Supply and demand
    supply = dict(zip(src_df.iloc[:, 0], src_df.iloc[:, 1]))
    demand = dict(zip(dst_df.iloc[:, 0], dst_df.iloc[:, 1]))

    # Cost matrix
    cost = {}
    for _, row in cost_df.iterrows():
        s = row["source_id"]
        for d in dests:
            cost[(s, d)] = row[d]

    TRUCK_CAPACITY = 10

    prob = LpProblem("TP_9", LpMinimize)

    # x[s,d] = number of trucks dispatched (integer)
    x = {(s, d): LpVariable(f"x_{s}_{d}", lowBound=0, cat=LpInteger)
         for s in sources for d in dests}

    # Objective: minimize total cost (cost is per unit, truck = 10 units)
    prob += lpSum(cost[(s, d)] * TRUCK_CAPACITY * x[(s, d)]
                  for s in sources for d in dests)

    # Supply constraints
    for s in sources:
        if s in supply:
            prob += lpSum(TRUCK_CAPACITY * x[(s, d)] for d in dests) <= supply[s], f"supply_{s}"

    # Demand constraints
    for d in dests:
        if d in demand:
            prob += lpSum(TRUCK_CAPACITY * x[(s, d)] for s in sources) >= demand[d], f"demand_{d}"

    prob.solve(PULP_CBC_CMD(msg=0))

    obj = value(prob.objective)
    print(f"[Problem 9] Status: {LpStatus[prob.status]} | Optimal value: {obj:.4f}")
    return round(obj, 4)


def solve_problem_10() -> float:
    """
    Problem 10 — Assignment Problem
    12 machines (M1-M12) × 12 tasks (A-L)
    Binary: each task assigned to exactly one machine
    CSV: cost_12x12.csv (Machine, A..L)
    """
    print("\n[Problem 10] Solving Assignment Problem (12 machines × 12 tasks)...")
    folder = DATA_DIR / "10"

    cost_df  = pd.read_csv(folder / "cost_12x12.csv")
    machines = cost_df["Machine"].tolist()
    tasks    = [c for c in cost_df.columns if c != "Machine"]

    cost = {}
    for _, row in cost_df.iterrows():
        m = row["Machine"]
        for t in tasks:
            cost[(m, t)] = row[t]

    prob = LpProblem("AP_10", LpMinimize)

    x = {(m, t): LpVariable(f"x_{m}_{t}", cat=LpBinary)
         for m in machines for t in tasks}

    # Objective
    prob += lpSum(cost[(m, t)] * x[(m, t)] for m in machines for t in tasks)

    # Each task assigned to exactly one machine
    for t in tasks:
        prob += lpSum(x[(m, t)] for m in machines) == 1, f"task_{t}"

    # Each machine assigned at most one task
    for m in machines:
        prob += lpSum(x[(m, t)] for t in tasks) <= 1, f"machine_{m}"

    prob.solve(PULP_CBC_CMD(msg=0))

    obj = value(prob.objective)
    print(f"[Problem 10] Status: {LpStatus[prob.status]} | Optimal value: {obj:.4f}")
    return round(obj, 4)


def main():
    print("=" * 60)
    print("  Generating PuLP ground truth optimal values")
    print("  for problems 3, 5, 9, 10")
    print("=" * 60)

    results = {}

    solvers = {
        "3":  solve_problem_3,
        "5":  solve_problem_5,
        "9":  solve_problem_9,
        "10": solve_problem_10,
    }

    for pid, solver in solvers.items():
        try:
            opt_val = solver()
            results[pid] = {
                "optimal_value": opt_val,
                "status": "solved",
            }
        except Exception as e:
            print(f"[Problem {pid}] ❌ Error: {e}")
            results[pid] = {
                "optimal_value": None,
                "status": f"error: {e}",
            }

    # Save to JSON
    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Ground truth values saved to {OUT_FILE}")
    print(f"{'='*60}")
    for pid, r in results.items():
        status = "✅" if r["status"] == "solved" else "❌"
        print(f"  Problem {pid}: {status}  optimal = {r['optimal_value']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
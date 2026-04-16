#!/usr/bin/env python3
"""
Orchestrator — run all experiments sequentially.

Usage (from the openpi directory on the GCP instance):
    cd $OPENPI_DIR
    uv run python $EXPERIMENT_DIR/run_all.py 2>&1 | tee $EXPERIMENT_DIR/results/overnight.log
"""

import subprocess
import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

RESULTS_DIR = os.environ.get(
    "EXPERIMENT_DIR",
    os.path.join(os.environ.get("WORKSPACE", os.path.expanduser("~")), "quantization_experiments"),
) + "/results"
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

LOG_PATH = os.path.join(RESULTS_DIR, "run_log.json")


def run_experiment(script_name, label, timeout_s):
    """Run one experiment script as a subprocess.  Returns (status, elapsed)."""
    script_path = os.path.join(SCRIPT_DIR, script_name)
    if not os.path.exists(script_path):
        return "MISSING", 0.0

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Script:  {script_path}")
    print(f"  Timeout: {timeout_s}s ({timeout_s/3600:.1f}h)")
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            timeout=timeout_s,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        elapsed = time.time() - t0
        status = "SUCCESS" if result.returncode == 0 else f"FAILED(rc={result.returncode})"
    except subprocess.TimeoutExpired:
        elapsed = float(timeout_s)
        status = "TIMEOUT"
    except Exception as e:
        elapsed = time.time() - t0
        status = f"ERROR({e})"

    print(f"\n>>> {label}: {status} ({elapsed:.0f}s / {elapsed/60:.1f}min)")
    return status, elapsed


def main():
    log = {
        "start_time": datetime.now().isoformat(),
        "experiments": [],
    }

    experiments = [
        ("setup_and_verify.py", "Phase 0: Verification",            300),    # 5 min
        ("exp1_activation_stats.py", "Exp 1: Activation Statistics", 3600),   # 1 hour
        ("exp2_layer_sensitivity.py", "Exp 2: Layer Sensitivity",    18000),  # 5 hours
        ("exp3_flow_step_sensitivity.py", "Exp 3: Flow-Step",        10800),  # 3 hours
    ]

    for script, label, timeout in experiments:
        status, elapsed = run_experiment(script, label, timeout)
        log["experiments"].append({
            "script": script,
            "label": label,
            "status": status,
            "elapsed_s": round(elapsed, 1),
            "finished": datetime.now().isoformat(),
        })

        # Save after each experiment (survive crashes)
        log["last_update"] = datetime.now().isoformat()
        with open(LOG_PATH, "w") as f:
            json.dump(log, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # Abort on setup failure
        if script == "setup_and_verify.py" and not status.startswith("SUCCESS"):
            print("\n!!! SETUP FAILED — aborting all experiments !!!")
            break

    log["end_time"] = datetime.now().isoformat()
    total = sum(e["elapsed_s"] for e in log["experiments"])
    log["total_elapsed_s"] = round(total, 1)

    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    print(f"\n{'='*60}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"  Total time: {total/3600:.1f}h")
    print(f"  Log: {LOG_PATH}")
    print(f"{'='*60}")

    for e in log["experiments"]:
        marker = "+" if e["status"].startswith("SUCCESS") else "X"
        print(f"  [{marker}] {e['label']}: {e['status']} ({e['elapsed_s']:.0f}s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

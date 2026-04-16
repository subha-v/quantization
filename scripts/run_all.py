#!/usr/bin/env python3
"""
Orchestrator — run all experiments sequentially with fault tolerance.

- Aborts ONLY if setup_and_verify fails (everything else is unrecoverable).
- Continues through experiment failures so you get partial results.
- Logs everything to both console and results/overnight.log.
- Tracks GPU memory and wall time per experiment.

Usage (from the openpi directory on the Stanford server):
    cd /data/subha2/openpi
    tmux new -s overnight
    CUDA_VISIBLE_DEVICES=0 uv run python /data/subha2/experiments/run_all.py
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

# Ensure GPU pinning even for subprocesses
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

EXPERIMENT_DIR = os.environ.get(
    "EXPERIMENT_DIR",
    os.path.join(os.environ.get("WORKSPACE", "/data/subha2"), "experiments"),
)
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

LOG_PATH = os.path.join(RESULTS_DIR, "run_log.json")
LOG_FILE = os.path.join(RESULTS_DIR, "overnight.log")


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()


def run_experiment(script_name, label, timeout_s):
    """Run one experiment.  Returns (status, elapsed_s, returncode)."""
    script_path = os.path.join(SCRIPT_DIR, script_name)
    if not os.path.exists(script_path):
        log(f"  SKIP: {script_path} not found")
        return "MISSING", 0.0, -1

    log("")
    log("=" * 60)
    log(f"  STARTING: {label}")
    log(f"  Script:   {script_path}")
    log(f"  Timeout:  {timeout_s}s ({timeout_s/3600:.1f}h)")
    log("=" * 60)

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    t0 = time.time()
    try:
        # Stream output to both console and log file
        with open(LOG_FILE, "a") as logf:
            proc = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                logf.write(line)
                logf.flush()

            proc.wait(timeout=timeout_s)
            elapsed = time.time() - t0
            rc = proc.returncode
            status = "SUCCESS" if rc == 0 else f"FAILED(rc={rc})"

    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        elapsed = float(timeout_s)
        rc = -1
        status = "TIMEOUT"
    except Exception as e:
        elapsed = time.time() - t0
        rc = -1
        status = f"ERROR({e})"

    log(f"\n>>> {label}: {status}  ({elapsed:.0f}s / {elapsed/60:.1f}min)")
    return status, elapsed, rc


def main():
    log("\n" + "=" * 60)
    log("  OVERNIGHT EXPERIMENT RUN")
    log(f"  Server:   {os.uname().nodename}")
    log(f"  GPU:      CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
    log(f"  Started:  {datetime.now().isoformat()}")
    log("=" * 60)

    run_log = {
        "server": os.uname().nodename,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "start_time": datetime.now().isoformat(),
        "experiments": [],
    }

    experiments = [
        # (script,                         label,                          timeout_s, abort_on_fail)
        ("setup_and_verify.py",            "Phase 0: Verification",        600,       True),
        ("exp1_activation_stats.py",       "Exp 1: Activation Statistics", 3600,      False),
        ("exp2_layer_sensitivity.py",      "Exp 2: Layer Sensitivity",     21600,     False),
        ("exp3_flow_step_sensitivity.py",  "Exp 3: Flow-Step Sensitivity", 14400,     False),
    ]

    for script, label, timeout, abort_on_fail in experiments:
        status, elapsed, rc = run_experiment(script, label, timeout)

        entry = {
            "script": script,
            "label": label,
            "status": status,
            "elapsed_s": round(elapsed, 1),
            "finished": datetime.now().isoformat(),
        }
        run_log["experiments"].append(entry)

        # Save run_log after EVERY experiment (survive crashes)
        run_log["last_update"] = datetime.now().isoformat()
        _save_log(run_log)

        if abort_on_fail and not status.startswith("SUCCESS"):
            log("\n!!! SETUP FAILED — aborting remaining experiments !!!")
            log("!!! Fix the issue and re-run. Experiments 1-3 depend on setup. !!!")
            break

        if not status.startswith("SUCCESS"):
            log(f"\n*** {label} failed but continuing with next experiment ***\n")

    # ---- Final summary ----
    run_log["end_time"] = datetime.now().isoformat()
    total = sum(e["elapsed_s"] for e in run_log["experiments"])
    run_log["total_elapsed_s"] = round(total, 1)
    _save_log(run_log)

    log("")
    log("=" * 60)
    log("  RUN COMPLETE")
    log(f"  Total time: {total/3600:.1f}h")
    log(f"  Log file:   {LOG_FILE}")
    log(f"  Run log:    {LOG_PATH}")
    log("=" * 60)

    for e in run_log["experiments"]:
        icon = "OK" if e["status"].startswith("SUCCESS") else "XX"
        log(f"  [{icon}] {e['label']}: {e['status']} ({e['elapsed_s']:.0f}s)")

    log("")
    log("Results in: " + RESULTS_DIR)
    log("Plots in:   " + os.path.join(EXPERIMENT_DIR, "plots"))

    return 0


def _save_log(run_log):
    with open(LOG_PATH, "w") as f:
        json.dump(run_log, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


if __name__ == "__main__":
    sys.exit(main())

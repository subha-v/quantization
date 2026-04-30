#!/bin/bash
# Wait for the overnight orchestrator (expD_overnight.sh) to finish, then copy
# its result artifacts into the repo's results/ dir, commit, and push to
# GitHub so a cloud-based scheduled agent can read them.
#
# The cloud agent has no access to /data/subha2/experiments — it can only
# see committed-and-pushed git state. This script bridges that gap.
#
# Run via:
#     nohup setsid bash /data/subha2/experiments/expD_publish_results.sh \
#         > /data/subha2/experiments/logs/expD_publish.log 2>&1 < /dev/null &

set -uo pipefail

LOGDIR=/data/subha2/experiments/logs
RESULTS_REMOTE=/data/subha2/experiments/results
REPO=/data/subha2/quantization
RESULTS_REPO=$REPO/results
BRANCH=overnight-2026-04-29-w4-first

ORCH_PID="${ORCH_PID:-121318}"

mkdir -p "$RESULTS_REPO" "$LOGDIR"

banner() {
    echo ""
    echo "=========================================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "=========================================================================="
}

banner "Waiting for orchestrator pid=$ORCH_PID to exit"
while ps -p "$ORCH_PID" > /dev/null 2>&1; do
    sleep 120
done
banner "Orchestrator exited; publishing results"

# Files to publish: markdown summaries + small JSONLs (rollouts + features) so
# the cloud agent can re-run the conditional partition analysis itself.
FILES=(
    "expD_trialgate_summary__libero_pro_obj_x0.2.md"
    "expD_trialgate_summary__libero_pro_obj_x0.2_n200.md"
    "expD_trialgate_features__libero_pro_obj_x0.2.jsonl"
    "expD_trialgate_features__libero_pro_obj_x0.2_n200.jsonl"
    "expB_w4__libero_pro_obj_x0.2_n200_summary.md"
    "expB_w4__libero_pro_obj_x0.2_rollouts.jsonl"
    "expB_w4__libero_pro_obj_x0.2_n200_rollouts.jsonl"
    "libero_pro_operating_point.md"
)

cd "$REPO"
git pull --ff-only origin "$BRANCH" 2>&1 || true

copied=()
for f in "${FILES[@]}"; do
    src="$RESULTS_REMOTE/$f"
    dst="$RESULTS_REPO/$f"
    if [ -f "$src" ]; then
        cp "$src" "$dst"
        copied+=("$f")
        echo "  copied: $f ($(wc -c < "$src") bytes)"
    else
        echo "  MISSING: $f (skipping)"
    fi
done

if [ "${#copied[@]}" -eq 0 ]; then
    echo "No result files copied; aborting commit"
    exit 1
fi

cd "$REPO"
git add -- "${copied[@]/#/results/}"
git status --short

if ! git commit -m "ExpD overnight: publish n=50 + n=200 trial-gate summaries + Phase B chunk results" 2>&1 | tee /tmp/expD_commit.log; then
    if grep -q "nothing to commit" /tmp/expD_commit.log; then
        echo "Nothing to commit (already up to date) — exiting OK"
        exit 0
    fi
    echo "ERROR: git commit failed for a non-trivial reason; bailing"
    cat /tmp/expD_commit.log
    exit 1
fi

git push origin "$BRANCH" 2>&1 | tail -5
push_rc=${PIPESTATUS[0]}
if [ "$push_rc" -ne 0 ]; then
    echo "ERROR: git push failed (exit $push_rc). The commit is local on $(hostname)." >&2
    echo "Recover with: git fetch ssh://$(whoami)@$(hostname)$REPO $BRANCH && git push origin <commit>" >&2
    exit 1
fi
banner "Publish DONE at $(date)"

#!/usr/bin/env python3
"""
LIBERO closed-loop rollout harness for pi0.5.

In-process (no websocket) so downstream experiments can monkey-patch the model
(attention hooks, weight-swap quantization schedulers, etc).

Lifted from openpi/examples/libero/main.py but restructured around a single
`run_rollout()` function with callback seams for hook-based experiments.

Usage:
  Programmatic:
    from rollout import run_rollout, SUITE_TO_OPENPI_NAME
    rec = run_rollout(policy, task_id=0, suite="Long", seed=0)

  CLI smoke tests:
    python scripts/rollout.py --smoke-render     # headless MuJoCo sanity
    python scripts/rollout.py --single-rollout --suite Object --task-id 20 --seed 0
"""

import argparse
import collections
import dataclasses
import math
import os
import sys
import time
import traceback

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import utils  # sets MUJOCO_GL, TORCHDYNAMO_DISABLE, paths, logging


# ---------------------------------------------------------------------------
# Suite naming
# ---------------------------------------------------------------------------
# Our parquet data uses the task_index → suite mapping in utils.suite_of():
#     0-9  = Long     (task_suite "libero_10")
#     10-19 = Goal    (task_suite "libero_goal")
#     20-29 = Object  (task_suite "libero_object")
#     30-39 = Spatial (task_suite "libero_spatial")
# openpi's LIBERO eval uses the task_suite names directly.
SUITE_TO_OPENPI_NAME = {
    "Long":    "libero_10",
    "Goal":    "libero_goal",
    "Object":  "libero_object",
    "Spatial": "libero_spatial",
}

# Within-suite task index is task_index % 10 (each suite has 10 tasks in the
# standard LIBERO benchmark; they are indexed 0..9 in their own suite).
def task_id_in_suite(global_task_id: int) -> int:
    return global_task_id % 10


# Per-suite max rollout step budget — cribbed from openpi/examples/libero/main.py
MAX_STEPS_BY_SUITE = {
    "Spatial": 220,
    "Object":  280,
    "Goal":    300,
    "Long":    520,
}

LIBERO_ENV_RESOLUTION = 256        # LIBERO's training-time render resolution
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]   # gripper closed, no motion
NUM_STEPS_WAIT = 10                # let objects settle after reset
DEFAULT_RESIZE = 224                # pi0.5 expects 224x224 after resize-with-pad
DEFAULT_REPLAN_STEPS = 5           # execute 5 actions per 10-long chunk, then replan


# ---------------------------------------------------------------------------
# Lazy imports of libero (heavy; only needed when actually rolling out)
# ---------------------------------------------------------------------------
def _import_libero():
    from libero.libero import benchmark, get_libero_path      # noqa: F401
    from libero.libero.envs import OffScreenRenderEnv          # noqa: F401
    return benchmark, get_libero_path, OffScreenRenderEnv


def _import_openpi_image_tools():
    # openpi_client.image_tools has resize_with_pad + convert_to_uint8.
    try:
        from openpi_client import image_tools
        return image_tools
    except Exception as e:
        raise RuntimeError(
            "Could not import openpi_client.image_tools. "
            "Run `uv pip install -e $OPENPI_DIR/packages/openpi-client` on the server."
        ) from e


# ---------------------------------------------------------------------------
# Obs adapter: LIBERO env → openpi policy.infer input dict
# ---------------------------------------------------------------------------
def _quat2axisangle(quat):
    """Convert quaternion (xyzw) to axis-angle (3-vec). Copied from openpi."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _libero_obs_to_openpi(obs, task_description, resize_size=DEFAULT_RESIZE):
    """Translate LIBERO env observation dict into the dict pi0.5 expects.

    LIBERO obs keys used:
      agentview_image            (H, W, 3) uint8  — 3rd-person view
      robot0_eye_in_hand_image   (H, W, 3) uint8  — wrist cam
      robot0_eef_pos             (3,)  — end-effector position
      robot0_eef_quat            (4,)  — end-effector orientation (xyzw)
      robot0_gripper_qpos        (2,)  — gripper joint positions

    Preprocessing: openpi training rotates images 180° and pads to (resize, resize).
    """
    image_tools = _import_openpi_image_tools()

    # IMPORTANT: rotate 180 degrees to match pi0.5 training preprocessing.
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, resize_size, resize_size)
    )
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
    )

    state = np.concatenate([
        obs["robot0_eef_pos"],
        _quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ]).astype(np.float32)

    return {
        "observation/image":       img,
        "observation/wrist_image": wrist_img,
        "observation/state":       state,
        "prompt":                  str(task_description),
    }


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------
def make_libero_env(suite: str, task_id: int, seed: int, resolution: int = LIBERO_ENV_RESOLUTION):
    """Instantiate a LIBERO OffScreenRenderEnv for a given (suite, task_id, seed).

    Returns (env, task_description, initial_states).
    """
    benchmark, get_libero_path, OffScreenRenderEnv = _import_libero()
    import pathlib

    if suite not in SUITE_TO_OPENPI_NAME:
        raise ValueError(f"Unknown suite {suite!r}; expected one of {list(SUITE_TO_OPENPI_NAME)}")
    suite_name = SUITE_TO_OPENPI_NAME[suite]

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()
    n_tasks = task_suite.n_tasks
    local_id = task_id_in_suite(task_id)
    if not (0 <= local_id < n_tasks):
        raise ValueError(
            f"task_id {task_id} → local_id {local_id} out of range [0, {n_tasks}) in suite {suite}"
        )

    task = task_suite.get_task(local_id)
    task_description = task.language
    bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    initial_states = task_suite.get_task_init_states(local_id)

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task_description, initial_states


# ---------------------------------------------------------------------------
# Main rollout function
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class RolloutRecord:
    success: bool
    steps: int
    task_id: int
    suite: str
    seed: int
    episode_idx: int
    task_description: str
    termination_reason: str  # "success" | "timeout" | "error"
    wall_time_s: float
    final_reward: float = 0.0
    exception: str = ""

    def to_dict(self):
        return dataclasses.asdict(self)


def run_rollout(
    policy,
    task_id: int,
    suite: str,
    seed: int = 0,
    episode_idx: int = 0,
    max_steps: int | None = None,
    replan_steps: int = DEFAULT_REPLAN_STEPS,
    resize_size: int = DEFAULT_RESIZE,
    env=None,                  # pre-constructed env; if None, we build one
    initial_states=None,       # if env is provided, pass its initial_states
    task_description=None,     # if env is provided, pass its description
    obs_callback=None,         # fn(step_idx, libero_obs, openpi_obs_dict) -> None
    action_callback=None,      # fn(step_idx, action_chunk_np) -> None
    verbose: bool = False,
) -> RolloutRecord:
    """Run one LIBERO rollout with pi0.5 FP16 (or whatever state `policy` is in).

    If `env` is None, we build a fresh env for (suite, task_id). Reusing an env
    across multiple episodes of the same task is more efficient and matches
    openpi/examples/libero/main.py's pattern — callers that roll out many
    episodes per task should pass a pre-built env.
    """
    import torch  # noqa: F401 — make sure torch env is initialized before sim

    if max_steps is None:
        max_steps = MAX_STEPS_BY_SUITE[suite]

    # Build env lazily if not provided
    own_env = env is None
    if env is None:
        env, task_description, initial_states = make_libero_env(suite, task_id, seed)
    else:
        if task_description is None or initial_states is None:
            raise ValueError(
                "When passing a pre-built env, task_description and initial_states "
                "must also be provided (got None). Use make_libero_env() to get all three."
            )

    t_start = time.time()
    try:
        env.reset()
        if initial_states is None or episode_idx >= len(initial_states):
            raise RuntimeError(
                f"No initial state for episode_idx={episode_idx} "
                f"(have {0 if initial_states is None else len(initial_states)})"
            )
        obs = env.set_init_state(initial_states[episode_idx])

        action_plan: collections.deque = collections.deque()
        done = False
        success = False
        reward = 0.0
        t = 0

        while t < max_steps + NUM_STEPS_WAIT:
            # Wait for objects to settle
            if t < NUM_STEPS_WAIT:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            # If we don't have queued actions, query the policy
            if not action_plan:
                openpi_obs = _libero_obs_to_openpi(obs, task_description, resize_size=resize_size)
                if obs_callback is not None:
                    try:
                        obs_callback(t, obs, openpi_obs)
                    except Exception as cb_e:
                        utils.log(f"[rollout] obs_callback error at t={t}: {cb_e}")

                with __import__("torch").no_grad():
                    result = policy.infer(openpi_obs)
                action_chunk = result["actions"] if isinstance(result, dict) else result
                action_chunk = np.asarray(action_chunk)

                if action_chunk.shape[0] < replan_steps:
                    raise RuntimeError(
                        f"action_chunk length {action_chunk.shape[0]} < replan_steps {replan_steps}"
                    )

                if action_callback is not None:
                    try:
                        action_callback(t, action_chunk)
                    except Exception as cb_e:
                        utils.log(f"[rollout] action_callback error at t={t}: {cb_e}")

                action_plan.extend(action_chunk[:replan_steps])

            action = action_plan.popleft()
            obs, reward, done, info = env.step(action.tolist() if hasattr(action, "tolist") else list(action))

            if verbose and t % 50 == 0:
                utils.log(f"[rollout] t={t} reward={reward:.3f} done={done}")

            if done:
                success = True
                break
            t += 1

        wall_time = time.time() - t_start
        termination = "success" if success else "timeout"
        return RolloutRecord(
            success=success, steps=t, task_id=task_id, suite=suite, seed=seed,
            episode_idx=episode_idx, task_description=str(task_description),
            termination_reason=termination, wall_time_s=wall_time,
            final_reward=float(reward),
        )

    except Exception as e:
        wall_time = time.time() - t_start
        tb = traceback.format_exc()
        utils.log(f"[rollout] exception at task_id={task_id} suite={suite} seed={seed} "
                  f"episode={episode_idx}: {e}\n{tb}")
        return RolloutRecord(
            success=False, steps=0, task_id=task_id, suite=suite, seed=seed,
            episode_idx=episode_idx, task_description=str(task_description or ""),
            termination_reason="error", wall_time_s=wall_time,
            exception=f"{type(e).__name__}: {e}",
        )
    finally:
        if own_env:
            try:
                env.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------
def smoke_render():
    """Verify headless MuJoCo can render a single LIBERO frame."""
    utils.log("[smoke-render] Importing libero...")
    benchmark, get_libero_path, OffScreenRenderEnv = _import_libero()
    import pathlib

    suite_name = "libero_object"
    utils.log(f"[smoke-render] Building env for {suite_name} task 0...")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()
    task = task_suite.get_task(0)
    bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(0)

    utils.log("[smoke-render] Resetting env and rendering...")
    obs = env.reset()
    img = obs.get("agentview_image")
    if img is None:
        raise RuntimeError(f"No agentview_image in obs keys: {list(obs.keys())}")
    if img.shape != (LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION, 3):
        raise RuntimeError(f"Unexpected image shape: {img.shape}")
    if img.dtype != np.uint8:
        raise RuntimeError(f"Image dtype not uint8: {img.dtype}")
    mean_pixel = float(img.mean())
    if mean_pixel < 1.0:
        raise RuntimeError(
            f"Image is all black (mean={mean_pixel:.2f}). "
            f"Likely EGL misconfig. Try MUJOCO_GL=osmesa or MUJOCO_GL=glx."
        )

    utils.log(f"[smoke-render] OK: image shape={img.shape} dtype={img.dtype} "
              f"mean={mean_pixel:.1f} (non-black)")
    env.close()
    return True


def single_rollout_cli(args):
    """Run 1 rollout with verbose logging — quick end-to-end sanity check."""
    policy, _ = utils.load_policy("pi05_libero")
    rec = run_rollout(
        policy,
        task_id=args.task_id,
        suite=args.suite,
        seed=args.seed,
        episode_idx=0,
        verbose=True,
    )
    utils.log("=" * 60)
    utils.log("SINGLE-ROLLOUT RESULT:")
    for k, v in rec.to_dict().items():
        utils.log(f"  {k}: {v}")
    utils.log("=" * 60)
    return 0 if rec.success or rec.termination_reason == "timeout" else 1


def main():
    utils.setup_logging()
    p = argparse.ArgumentParser(description="LIBERO rollout harness smoke tests.")
    p.add_argument("--smoke-render", action="store_true",
                   help="Import libero + render one frame. Validates MUJOCO_GL setup.")
    p.add_argument("--single-rollout", action="store_true",
                   help="Run one full rollout with verbose logging.")
    p.add_argument("--suite", default="Object", choices=list(SUITE_TO_OPENPI_NAME))
    p.add_argument("--task-id", type=int, default=20,
                   help="Global task_id (suite-agnostic; mapped via task_id % 10).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.smoke_render:
        try:
            smoke_render()
            utils.log("[smoke-render] PASSED")
            return 0
        except Exception as e:
            utils.log(f"[smoke-render] FAILED: {e}")
            traceback.print_exc()
            return 1

    if args.single_rollout:
        return single_rollout_cli(args)

    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

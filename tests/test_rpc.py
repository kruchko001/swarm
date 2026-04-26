#!/usr/bin/env python3
"""
test_rpc.py

Local test script for RPC agent submissions. Tests your agent exactly 
like the validator does before you submit.

Usage:
    python tests/test_rpc.py swarm/submission_template/ --seed 42 --gui
    python tests/test_rpc.py swarm/submission_template/ --zip
    python tests/test_rpc.py my_agent/ --log-io-every 50 --depth-gif runs/depth.gif
    python tests/test_rpc.py my_agent/ --rpc-start-timeout 120

Logs are always written under runs/ as test_rpc_<timestamp>_seed<N>.log (UTF-8).
Per simulation step: IN (depth meta + state), OUT (actions), then STEP (position, speed, flags) on stdout and in that log file.
"""

import argparse
import asyncio
import logging
import math
import os
from datetime import datetime
import socket
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

# Repo root (parent of ``tests/``) must be on ``sys.path`` for ``import swarm`` when
# running: ``python tests/test_rpc.py ...`` from any working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_repo_root_str = str(_REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

import numpy as np

logger = logging.getLogger(__name__)

# When --depth-gif is set but --log-io-every is 0, sample depth this often (≈1 s sim at 50 Hz).
_DEFAULT_DEPTH_GIF_INTERVAL = 50

try:
    import capnp
except ImportError:
    print("ERROR: pycapnp not installed. Install with: pip install pycapnp")
    sys.exit(1)

from gym_pybullet_drones.utils.enums import ActionType

from swarm.constants import SIM_DT, SPEED_LIMIT
from swarm.core.drone import track_drone
from swarm.protocol import MapTask, ValidationResult
from swarm.utils.env_factory import make_env
from swarm.validator.reward import flight_reward
from swarm.validator.task_gen import random_task


def _default_run_log_path(seed: int) -> Path:
    """runs/test_rpc_<YYYYMMDD_HHMMSS>_seed<seed>.log under repo root."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _REPO_ROOT / "runs" / f"test_rpc_{stamp}_seed{seed}.log"


class _FlushingStreamHandler(logging.StreamHandler):
    """Line-oriented console output: flush after each record so every timestep is visible immediately."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        try:
            self.flush()
        except OSError:
            pass


def _setup_logging(verbose: bool, *, log_file: Path) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False

    # stdout: per-step I/O prints here; miner continues to use stderr.
    console_handler = _FlushingStreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)


def _summarize_state_vec(state: np.ndarray) -> str:
    """Compact text for logs (never dump huge tensors)."""
    s = np.asarray(state, dtype=np.float64).ravel()
    n = int(s.size)
    if n == 0:
        return "empty"
    head = s[: min(12, n)]
    tail = s[-min(6, n) :]
    return (
        f"len={n} head[:12]={np.array2string(head, precision=3, max_line_width=120)} "
        f"tail={np.array2string(tail, precision=4)} "
        f"min={float(s.min()):.4f} max={float(s.max()):.4f} mean={float(s.mean()):.4f}"
    )


def _summarize_obs_for_log(obs) -> Tuple[str, str]:
    """Returns (depth_meta, state_summary) for dict obs; else single summary."""
    if not isinstance(obs, dict):
        arr = np.asarray(obs)
        return ("n/a", f"vector shape={arr.shape} dtype={arr.dtype}")
    depth_meta = "missing"
    if "depth" in obs:
        d = np.asarray(obs["depth"])
        depth_meta = f"shape={d.shape} dtype={d.dtype} range=[{float(np.min(d)):.4f},{float(np.max(d)):.4f}]"
    state_s = (
        _summarize_state_vec(obs["state"])
        if "state" in obs
        else "missing"
    )
    return depth_meta, state_s


def _depth_to_rgb_u8(depth: np.ndarray) -> np.ndarray:
    """Normalized depth (H,W,1) or (H,W) float → uint8 RGB for GIF."""
    d = np.asarray(depth, dtype=np.float32)
    while d.ndim > 2 and d.shape[-1] == 1:
        d = d[..., 0]
    if d.ndim != 2:
        raise ValueError(f"Expected HxW depth, got shape {d.shape}")
    u8 = (np.clip(d, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def _save_depth_gif(frames: List[np.ndarray], path: Path, *, duration_ms: int) -> None:
    try:
        from PIL import Image
    except ImportError as e:
        logger.error(
            "Pillow is required for --depth-gif (pip install Pillow). Import error: %s",
            e,
        )
        return
    if not frames:
        logger.warning("No depth frames were captured; GIF not written.")
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(f, mode="RGB") for f in frames]
    dur = max(20, int(duration_ms))
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=dur,
        loop=0,
    )
    logger.info(
        "Wrote depth GIF: %s (%s frames, %sms per frame)",
        path.resolve(),
        len(frames),
        dur,
    )


def _check_folder_structure(folder: Path):
    required_files = ["main.py", "agent_server.py", "drone_agent.py", "agent.capnp"]
    missing = [f for f in required_files if not (folder / f).exists()]

    if missing:
        print(f"❌ Missing required files: {', '.join(missing)}")
        return False

    print("✅ Folder structure valid")
    return True


def _wait_for_port(
    port: int = 8000,
    *,
    max_retries: int = 90,
    retry_delay: float = 1.0,
) -> bool:
    """Poll until ``localhost:port`` accepts TCP connections or retries are exhausted."""
    for retry in range(max_retries):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex(("localhost", port))
            sock.close()
            if result == 0:
                return True
        except OSError:
            pass
        time.sleep(retry_delay)
    return False


async def _run_episode(
    task: MapTask,
    uid: int,
    agent,
    gui: bool = False,
    *,
    gif_frame_interval: int = 0,
    depth_gif_path: Optional[Path] = None,
):
    env = make_env(task, gui=gui)
    info: dict = {}
    gif_frames: List[np.ndarray] = []
    if depth_gif_path:
        gif_n = (
            max(1, gif_frame_interval)
            if gif_frame_interval > 0
            else _DEFAULT_DEPTH_GIF_INTERVAL
        )
        frame_duration_ms = max(1, int(round(gif_n * SIM_DT * 1000)))
    else:
        frame_duration_ms = 50

    try:
        obs, _ = env.reset()
        await agent.reset()

        pos0 = np.asarray(task.start, dtype=float)
        t_sim = 0.0
        success = False
        speeds = []
        step_count = 0

        lo, hi = env.action_space.low.flatten(), env.action_space.high.flatten()
        last_pos = pos0

        cli_id = getattr(env, "CLIENT", getattr(env, "_cli", 0))
        frames_per_cam = max(1, int(round(1.0 / (SIM_DT * 60.0))))

        schema_file = (
            Path(__file__).parent.parent
            / "swarm"
            / "submission_template"
            / "agent.capnp"
        )
        agent_capnp = capnp.load(str(schema_file))

        logger.info(
            "Episode: IN/OUT logged every sim step (~%.0f Hz). depth_gif=%s (frame every %s steps)",
            1.0 / SIM_DT,
            str(depth_gif_path.resolve()) if depth_gif_path else "off",
            gif_frame_interval if (depth_gif_path and gif_frame_interval > 0) else (
                _DEFAULT_DEPTH_GIF_INTERVAL if depth_gif_path else "—"
            ),
        )

        while t_sim < task.horizon:
            depth_meta, state_summary = _summarize_obs_for_log(obs)
            gif_interval = (
                gif_frame_interval
                if gif_frame_interval > 0
                else _DEFAULT_DEPTH_GIF_INTERVAL
            )
            if (
                depth_gif_path
                and isinstance(obs, dict)
                and "depth" in obs
                and (step_count % gif_interval == 0)
            ):
                try:
                    gif_frames.append(_depth_to_rgb_u8(obs["depth"]))
                except Exception:
                    logger.exception(
                        "Failed to convert depth to GIF frame at step=%s t=%.3f",
                        step_count,
                        t_sim,
                    )

            try:
                observation = agent_capnp.Observation.new_message()

                if isinstance(obs, dict):
                    entries = observation.init("entries", len(obs))
                    for i, (key, value) in enumerate(obs.items()):
                        arr = np.asarray(value, dtype=np.float32)
                        entries[i].key = key
                        entries[i].tensor.data = arr.tobytes()
                        entries[i].tensor.shape = list(arr.shape)
                        entries[i].tensor.dtype = str(arr.dtype)
                else:
                    arr = np.asarray(obs, dtype=np.float32)
                    entry = observation.init("entries", 1)[0]
                    entry.key = "__value__"
                    entry.tensor.data = arr.tobytes()
                    entry.tensor.shape = list(arr.shape)
                    entry.tensor.dtype = str(arr.dtype)

                action_response = await agent.act(observation)
                action = np.frombuffer(
                    action_response.action.data,
                    dtype=np.dtype(action_response.action.dtype),
                ).reshape(tuple(action_response.action.shape))
            except Exception:
                logger.exception(
                    "agent.act failed at t_sim=%.3fs step=%s (using zero action)",
                    t_sim,
                    step_count,
                )
                action = np.zeros(5, dtype=np.float32)

            rpc_action = np.asarray(action, dtype=np.float32).reshape(-1)
            act = np.clip(rpc_action, lo, hi)
            clipped_action = act.copy()

            if hasattr(env, "ACT_TYPE") and hasattr(env, "SPEED_LIMIT"):
                if env.ACT_TYPE == ActionType.VEL and env.SPEED_LIMIT:
                    n = max(np.linalg.norm(act[:3]), 1e-6)
                    scale = min(1.0, SPEED_LIMIT / n)
                    act[:3] *= scale
                    act = np.clip(act, lo, hi)

            logger.info(
                "IO step=%s t_sim=%.4fs | IN depth: %s | IN state: %s",
                step_count,
                t_sim,
                depth_meta,
                state_summary,
            )
            logger.info(
                "IO step=%s t_sim=%.4fs | OUT action_rpc=%s | OUT action_clipped=%s | OUT action_env=%s",
                step_count,
                t_sim,
                np.array2string(rpc_action, precision=4),
                np.array2string(clipped_action, precision=4),
                np.array2string(act, precision=4),
            )

            prev = last_pos
            obs, _r, terminated, truncated, info = env.step(act[None, :])
            last_pos = env._getDroneStateVector(0)[0:3]

            speed = np.linalg.norm(last_pos - prev) / SIM_DT
            speeds.append(speed)

            t_sim += SIM_DT

            logger.info(
                "IO step=%s t_sim=%.4fs | STEP pos=%s speed=%.4f m/s term=%s trunc=%s",
                step_count,
                t_sim,
                np.array2string(last_pos, precision=3),
                float(speed),
                terminated,
                truncated,
            )

            if gui and step_count % frames_per_cam == 0:
                try:
                    track_drone(cli=cli_id, drone_id=env.DRONE_IDS[0])
                except Exception:
                    pass

            if gui:
                time.sleep(SIM_DT)

            if terminated or truncated:
                success = info.get("success", False)
                break

            step_count += 1

        if not gui:
            env.close()

        if depth_gif_path and gif_frames:
            _save_depth_gif(gif_frames, depth_gif_path, duration_ms=frame_duration_ms)

        min_clearance = info.get("min_clearance", None)
        collision = info.get("collision", False)
        score = flight_reward(
            success=success,
            t=t_sim,
            horizon=task.horizon,
            task=task,
            min_clearance=min_clearance,
            collision=collision,
        )
        avg_speed = np.mean(speeds) if speeds else 0.0

        return ValidationResult(uid, success, t_sim, score), avg_speed

    finally:
        try:
            env.close()
        except Exception:
            pass


async def _test_rpc_agent(
    submission_folder: Path,
    task: MapTask,
    gui: bool = False,
    *,
    gif_frame_interval: int = 0,
    depth_gif_path: Optional[Path] = None,
    rpc_start_timeout_sec: float = 90.0,
):
    # Inherit stdout/stderr so miner tracebacks and logs appear in this console and so a
    # filled PIPE cannot block the child before it binds to port 8000.
    retry_delay = 1.0
    max_retries = max(1, math.ceil(rpc_start_timeout_sec / retry_delay))

    agent_process = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=str(submission_folder),
        stdin=subprocess.DEVNULL,
        stdout=None,
        stderr=None,
        env=os.environ.copy(),
    )

    try:
        print(
            f"🔄 Starting RPC server on port 8000 (waiting up to {rpc_start_timeout_sec:.0f}s)...\n"
            "   Miner logs from main.py appear below.\n"
        )

        if not _wait_for_port(
            8000, max_retries=max_retries, retry_delay=retry_delay
        ):
            exit_code = agent_process.poll()
            print(
                f"❌ RPC server did not listen on port 8000 within {rpc_start_timeout_sec:.0f}s."
            )
            if exit_code is not None:
                print(f"   Agent subprocess exited early with code {exit_code}.")
                print("   Check the traceback above (miner stderr is shared with this terminal).")
            else:
                print(
                    "   Agent subprocess is still running — ONNX/model load may be slow, "
                    "or main.py never reached start_server()."
                )
                print(
                    f"   Try: python tests/test_rpc.py ... --rpc-start-timeout {int(rpc_start_timeout_sec * 2)}"
                )
            return None, 0.0

        if agent_process.poll() is not None:
            exit_code = agent_process.poll()
            print(f"❌ Agent process exited before RPC handshake (code {exit_code}).")
            print("   See miner output above.")
            return None, 0.0

        print("✅ RPC server started")
        print("✅ Connecting via Cap'n Proto...")

        async with capnp.kj_loop():
            stream = await capnp.AsyncIoStream.create_connection(
                host="localhost", port=8000
            )
            client = capnp.TwoPartyClient(stream)
            schema_file = (
                Path(__file__).parent.parent
                / "swarm"
                / "submission_template"
                / "agent.capnp"
            )
            agent_capnp = capnp.load(str(schema_file))
            agent = client.bootstrap().cast_as(agent_capnp.Agent)

            ping_response = await agent.ping("test")
            if ping_response.response != "pong":
                print(f"❌ RPC ping test failed: got {ping_response.response}")
                return None, 0.0

            print("✅ RPC connection established")
            print(f"🚁 Running evaluation (seed: {task.map_seed})...")

            result, avg_speed = await _run_episode(
                task,
                uid=0,
                agent=agent,
                gui=gui,
                gif_frame_interval=gif_frame_interval,
                depth_gif_path=depth_gif_path,
            )

            return result, avg_speed

    finally:
        agent_process.terminate()
        try:
            agent_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            agent_process.kill()
            agent_process.wait()


def _ensure_utf8_stdio() -> None:
    """Avoid UnicodeEncodeError on Windows (cp1252) when printing emoji to the console."""
    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if callable(reconf):
            try:
                reconf(encoding="utf-8", errors="replace")
            except (OSError, ValueError, AttributeError):
                pass


def main():
    _ensure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description="Test RPC agent submission locally (exactly like validator)"
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Path to submission folder (containing main.py, agent_server.py, etc.)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for task generation (default: 42)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Show PyBullet GUI during evaluation",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Create submission.zip after testing",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging on the test_rpc logger",
    )
    parser.add_argument(
        "--log-io-every",
        type=int,
        default=0,
        metavar="N",
        help=(
            "With --depth-gif, append a GIF frame every N simulation steps (default N=50 if "
            "omitted). I/O text logs are written every step regardless of this flag."
        ),
    )
    parser.add_argument(
        "--depth-gif",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Save the agent's normalized depth input as an animated GIF (Pillow). "
            "Frame sampling uses --log-io-every (default N=50 when that flag is 0). "
            "Per-step text logs still go to the run log file and console."
        ),
    )
    parser.add_argument(
        "--rpc-start-timeout",
        type=float,
        default=90.0,
        metavar="SEC",
        help=(
            "Seconds to wait for the miner (main.py) to listen on TCP port 8000 (default: 90). "
            "ONNX or other imports can exceed 15s on cold start."
        ),
    )
    args = parser.parse_args()

    run_log = _default_run_log_path(args.seed)
    _setup_logging(verbose=args.verbose, log_file=run_log)
    print(f"📝 Log file: {run_log.resolve()}\n")
    logger.info("Run log file: %s", run_log.resolve())

    gif_frame_interval = 0
    depth_gif: Optional[Path] = None
    if args.depth_gif is not None:
        depth_gif = Path(args.depth_gif).resolve()
        gif_frame_interval = (
            args.log_io_every
            if args.log_io_every > 0
            else _DEFAULT_DEPTH_GIF_INTERVAL
        )
        if args.log_io_every <= 0:
            logger.info(
                "Using --log-io-every=%s for depth GIF frames only (default; text logs every step).",
                gif_frame_interval,
            )
    if args.log_io_every < 0:
        print("❌ --log-io-every must be >= 0")
        sys.exit(2)

    if args.rpc_start_timeout <= 0:
        print("❌ --rpc-start-timeout must be > 0")
        sys.exit(2)

    if not args.folder.exists():
        print(f"❌ Folder not found: {args.folder}")
        sys.exit(1)

    if not args.folder.is_dir():
        print(f"❌ Not a directory: {args.folder}")
        sys.exit(1)

    print("🔍 Testing RPC Agent Submission\n")

    if not _check_folder_structure(args.folder):
        sys.exit(1)

    task = random_task(sim_dt=SIM_DT, seed=args.seed)

    print("\n📋 Task Details:")
    print(f"   Start: {task.start}")
    print(f"   Goal:  {task.goal}")
    print(f"   Seed:  {task.map_seed}")
    print(f"   Horizon: {task.horizon}s\n")

    result, avg_speed = asyncio.run(
        _test_rpc_agent(
            args.folder,
            task,
            gui=args.gui,
            gif_frame_interval=gif_frame_interval,
            depth_gif_path=depth_gif,
            rpc_start_timeout_sec=args.rpc_start_timeout,
        )
    )

    if result is None:
        print("\n❌ Test failed - agent could not be evaluated")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Success    : {'✅ True' if result.success else '❌ False'}")
    print(f"Time       : {result.time_sec:.2f}s")
    print(f"Score      : {result.score:.3f}")
    print(f"Avg Speed  : {avg_speed:.3f} m/s")
    print("=" * 60)

    if result.success and result.score > 0.7:
        print("\n🎉 Your agent is ready for submission!")
    elif result.success:
        print("\n✅ Mission successful, but try to improve speed for higher scores!")
    else:
        print("\n⚠️  Mission failed. Debug and try again.")

    if args.zip:
        _create_submission_zip(args.folder)

    sys.exit(0 if result.success else 1)


def _create_submission_zip(folder: Path):
    submission_dir = Path(__file__).parent.parent / "Submission"
    submission_dir.mkdir(parents=True, exist_ok=True)

    zip_path = submission_dir / "submission.zip"

    miner_files = ["drone_agent.py", "requirements.txt", "ppo_policy.zip"]
    model_extensions = [".pt", ".pth", ".onnx", ".pkl", ".h5", ".weights"]

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in folder.iterdir():
            if file.is_file():
                if file.name in miner_files:
                    zf.write(file, file.name)
                elif any(file.name.endswith(ext) for ext in model_extensions):
                    zf.write(file, file.name)

    print(f"\n📦 Created submission: {zip_path}")
    print("   Contents:")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            size_kb = info.file_size / 1024
            print(f"   - {info.filename} ({size_kb:.1f} KB)")

    print(f"\n✅ Submission ready at: {zip_path}")
    print("   Miner will read from: Submission/submission.zip")


if __name__ == "__main__":
    main()

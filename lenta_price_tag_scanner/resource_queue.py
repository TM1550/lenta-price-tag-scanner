"""One-shot GPU0 admission for the user-authorized OOF run; no other GPU is used."""

import json
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import click

from .server_video_oof import confined, gpu_preflight


def admission(root, minimum_free_gib=12):
    free = shutil.disk_usage(root).free / 1024**3
    if free < minimum_free_gib:
        return False, f"disk: {free:.2f} GiB free; need {minimum_free_gib}"
    try:
        gpu_preflight()
    except RuntimeError as error:
        return False, str(error)
    return True, f"GPU0 idle and {free:.2f} GiB free"


def save_status(path, status):
    status["updated_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
    temporary.replace(path)


@click.command()
@click.option("--timeout-hours", type=click.FloatRange(min=0.01), default=6.0)
@click.option("--interval-seconds", type=click.IntRange(min=5), default=30)
@click.option("--idle-checks", type=click.IntRange(min=2), default=3)
def main(timeout_hours, interval_seconds, idle_checks):
    import fcntl

    root = confined(Path(os.environ["LENTA_WORKSPACE"]))
    status_path = root / "artifacts/oof_launch_status.json"
    lock = (root / "artifacts/oof_launch.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = dict(
        state="waiting_resources",
        queue_pid=os.getpid(),
        gpu=0,
        gpu1_used=False,
        training_started=False,
        timeout_hours=timeout_hours,
        required_idle_checks=idle_checks,
    )
    deadline, consecutive = time.monotonic() + timeout_hours * 3600, 0
    while time.monotonic() < deadline:
        ready, reason = admission(root)
        consecutive = consecutive + 1 if ready else 0
        status.update(reason=reason, consecutive_idle_checks=consecutive)
        save_status(status_path, status)
        print(json.dumps(status), flush=True)
        if consecutive >= idle_checks:
            break
        time.sleep(interval_seconds)
    else:
        status.update(state="expired_waiting_resources")
        save_status(status_path, status)
        return
    ready, reason = admission(root)
    if not ready:
        status.update(state="admission_lost", reason=reason)
        save_status(status_path, status)
        return
    with (root / "logs/full_oof_launcher.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen(
            ["bash", "scripts/server/launch_oof.sh", "--execute"],
            cwd=root / "project",
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        status.update(
            state="running", runner_pid=child.pid, runner_started=True, training_started=None
        )
        save_status(status_path, status)
        while child.poll() is None:
            free = shutil.disk_usage(root).free / 1024**3
            if free < 2:
                # Stop only the process group created immediately above.
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=60)
                status.update(state="stopped_low_disk", free_disk_gib=free)
                save_status(status_path, status)
                return
            output = root / "results/video_oof_v1"
            status.update(free_disk_gib=free, training_started=(output / "training").exists())
            save_status(status_path, status)
            time.sleep(interval_seconds)
    status.update(
        state="completed" if child.returncode == 0 else "failed",
        returncode=child.returncode,
        report=str(root / "results/video_oof_v1/report.json"),
    )
    save_status(status_path, status)


if __name__ == "__main__":
    main()

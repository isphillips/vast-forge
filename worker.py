"""
Forge worker — the pull side of the forge dispatcher.

Each forge box runs this loop inside the server process (started by server.py at boot when FORGE_CONTROL_URL is
set). It talks OUTBOUND only, to the `forge-worker` Supabase edge function, authenticated with the same
FORGE_TOKEN the push endpoint uses:

    tick     — heartbeat (state, current job, GPU, model loaded, jobs done) and, when idle, claim the oldest
               queued job (image already generated upstream). One call does both so a node costs one request
               per interval.
    complete — hand back the GLB URL for a claimed job (the function stores it, creates the facet, pushes).
    fail     — report an error for a claimed job (the function refunds and releases the concept).

Why pull instead of push: the box's public address rotates (residential Vast host) and any number of boxes can
run the same loop against the same queue — the queue IS the load balancer. Nodes identify themselves by their
Vast instance id (VAST_CONTAINERLABEL=C.<id>), so a new box needs no configuration beyond the shared token.
"""
import os
import socket
import threading
import time
import traceback

import requests

CONTROL_URL = os.environ.get("FORGE_CONTROL_URL", "").rstrip("/")   # https://<ref>.supabase.co/functions/v1/forge-worker
FORGE_TOKEN = os.environ.get("FORGE_TOKEN", "")
IDLE_POLL_S = float(os.environ.get("FORGE_POLL_IDLE_S", "8"))       # quiet queue: heartbeat + claim attempt every 8 s
BUSY_POLL_S = float(os.environ.get("FORGE_POLL_ACTIVE_S", "2.5"))   # after recent work: claim faster (a burst of jobs)
ACTIVE_WINDOW_S = 600                                               # "recent" = a job in the last 10 min
HEARTBEAT_S = 20                                                    # heartbeat cadence while a job is running


def _pid1_env(key: str) -> str:
    """A variable from the container's init environment. Vast sets VAST_CONTAINERLABEL/CONTAINER_ID on PID 1, but
    our launcher (start_server.py) starts uvicorn with a curated env that doesn't carry them."""
    try:
        with open("/proc/1/environ", "rb") as fh:
            for kv in fh.read().split(b"\0"):
                if kv.startswith(key.encode() + b"="):
                    return kv.split(b"=", 1)[1].decode(errors="replace")
    except OSError:
        pass
    return ""


def node_id() -> str:
    explicit = os.environ.get("FORGE_NODE_ID", "").strip()
    if explicit:
        return explicit
    label = os.environ.get("VAST_CONTAINERLABEL") or _pid1_env("VAST_CONTAINERLABEL")   # "C.49344137" on Vast
    if label.startswith("C.") and label[2:].isdigit():
        return f"vast-{label[2:]}"
    cid = os.environ.get("CONTAINER_ID") or _pid1_env("CONTAINER_ID")
    if cid.isdigit():
        return f"vast-{cid}"
    return f"host-{socket.gethostname()}"


def gpu_name() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "unknown"


class Worker(threading.Thread):
    def __init__(self, run_job, model_loaded):
        """run_job(image_url) -> glb_url (raises on failure); model_loaded() -> bool."""
        super().__init__(name="forge-worker", daemon=True)
        self.run_job = run_job
        self.model_loaded = model_loaded
        self.id = node_id()
        self.gpu = gpu_name()
        self.started = time.time()
        self.jobs_done = 0
        self.jobs_failed = 0
        self.current = None          # {"id": ..., "started": ts}
        self.last_work = 0.0
        self.last_error = None

    # ── control-plane calls ──────────────────────────────────────────────────────────────────────────────────
    def _post(self, action: str, body: dict, timeout: float = 20) -> dict:
        r = requests.post(
            CONTROL_URL,
            json={"action": action, "node_id": self.id, **body},
            headers={"Authorization": f"Bearer {FORGE_TOKEN}", "Content-Type": "application/json"},
            timeout=timeout,
        )
        if r.status_code != 200:
            raise RuntimeError(f"{action} → HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    def _status(self) -> dict:
        return {
            "state": "busy" if self.current else "idle",
            "job_id": self.current["id"] if self.current else None,
            "gpu": self.gpu,
            "model_loaded": bool(self.model_loaded()),
            "jobs_done": self.jobs_done,
            "jobs_failed": self.jobs_failed,
            "uptime_s": int(time.time() - self.started),
            "last_error": self.last_error,
            "version": os.environ.get("FORGE_VERSION", "solidify-r2"),
        }

    def tick(self, claim: bool) -> dict | None:
        """Heartbeat; when `claim`, also ask for the next queued job. Returns the job or None."""
        res = self._post("tick", {"status": self._status(), "claim": claim})
        return res.get("job")

    # ── the loop ─────────────────────────────────────────────────────────────────────────────────────────────
    def run(self):
        print(f"[worker] {self.id} ({self.gpu}) pulling from {CONTROL_URL}", flush=True)
        while True:
            try:
                job = self.tick(claim=True)
            except Exception as e:  # noqa: BLE001 — control plane unreachable: keep trying
                print(f"[worker] tick failed: {e}", flush=True)
                time.sleep(IDLE_POLL_S)
                continue
            if not job:
                recent = (time.time() - self.last_work) < ACTIVE_WINDOW_S
                time.sleep(BUSY_POLL_S if recent else IDLE_POLL_S)
                continue
            self.process(job)

    def process(self, job: dict):
        self.current = {"id": job["id"], "started": time.time()}
        self.last_work = time.time()
        print(f"[worker] job {job['id']} claimed ({job.get('prompt', '')[:60]!r})", flush=True)
        stop = threading.Event()
        threading.Thread(target=self._heartbeat_while, args=(stop,), daemon=True).start()
        try:
            glb_url = self.run_job(job["image_url"])
            took = int(time.time() - self.current["started"])
            self._post("complete", {"job_id": job["id"], "glb_url": glb_url, "seconds": took}, timeout=30)
            self.jobs_done += 1
            self.last_error = None
            print(f"[worker] job {job['id']} done in {took}s → {glb_url}", flush=True)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.jobs_failed += 1
            self.last_error = f"{type(e).__name__}: {str(e)[:160]}"
            try:
                self._post("fail", {"job_id": job["id"], "error": self.last_error}, timeout=30)
            except Exception as e2:  # noqa: BLE001 — the reaper will time the job out if this never lands
                print(f"[worker] could not report failure: {e2}", flush=True)
        finally:
            stop.set()
            self.current = None
            self.last_work = time.time()

    def _heartbeat_while(self, stop: threading.Event):
        while not stop.wait(HEARTBEAT_S):
            try:
                self.tick(claim=False)
            except Exception as e:  # noqa: BLE001
                print(f"[worker] heartbeat failed: {e}", flush=True)


def start(run_job, model_loaded) -> Worker | None:
    if not CONTROL_URL:
        print("[worker] FORGE_CONTROL_URL not set — pull worker disabled (push /generate only)", flush=True)
        return None
    if not FORGE_TOKEN:
        print("[worker] FORGE_TOKEN not set — pull worker disabled", flush=True)
        return None
    w = Worker(run_job, model_loaded)
    w.start()
    return w

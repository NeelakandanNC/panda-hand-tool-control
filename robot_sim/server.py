"""Simulation process: owns the MuJoCo model, runs physics (optionally with the viewer window)
and serves robot commands as newline-delimited JSON over TCP.

    scripts/mjpython.sh -m robot_sim.server            # with the 3D window (macOS needs mjpython)
    uv run python -m robot_sim.server --headless       # no window, physics runs as fast as possible

Protocol: one JSON object per line.
    request:  {"id": 1, "cmd": "move_palm", "args": {"position": [0.25, 0, 0.55], "palm": "down"}}
    response: {"id": 1, "ok": true, "result": {...}}   or   {"id": 1, "ok": false, "error": "..."}
Motion commands reply once the motion has finished and the robot has settled; the reply
includes the resulting state.
"""

import argparse
import json
import queue
import socketserver
import threading
import time
from concurrent.futures import Future

import mujoco
import numpy as np

from robot_sim.controller import Controller
from robot_sim.scene import build_model

DEFAULT_PORT = 7777
COMMANDS = ("get_state", "describe", "move_actuators", "move_palm", "hand_pose", "wait", "reset", "snapshot")
MOTION_COMMANDS = ("move_actuators", "move_palm", "hand_pose", "wait")


def _to_json(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (np.ndarray, set)):
        return list(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


class Simulation:
    """Single-threaded owner of model/data. Other threads talk to it only through `submit`."""

    def __init__(self):
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.controller = Controller(self.model, self.data)
        self.requests: queue.Queue = queue.Queue()
        self._pending: tuple[Future, dict] | None = None

    def submit(self, cmd: str, args: dict) -> Future:
        fut: Future = Future()
        self.requests.put((cmd, args, fut))
        return fut

    @property
    def idle(self) -> bool:
        return self._pending is None and not self.controller.busy()

    def _begin(self, cmd: str, args: dict, fut: Future) -> None:
        if cmd not in COMMANDS:
            fut.set_exception(ValueError(f"unknown command {cmd!r}; available: {list(COMMANDS)}"))
            return
        try:
            info = getattr(self.controller, cmd)(**args)
        except Exception as exc:  # bad arguments etc. -> error reply, keep serving
            fut.set_exception(exc)
            return
        if cmd in MOTION_COMMANDS and self.controller.trajectory is not None:
            self._pending = (fut, info)
        else:
            fut.set_result(info)

    def tick(self, block: bool = False) -> None:
        """Accept a new command if free, advance physics one step, and complete finished motions."""
        if self._pending is None:
            try:
                self._begin(*self.requests.get(block=block, timeout=0.2 if block else None))
            except queue.Empty:
                pass
        self.controller.step()
        if self._pending is not None and not self.controller.busy():
            fut, info = self._pending
            self._pending = None
            fut.set_result({**info, "state": self.controller.get_state()})

    def run_command(self, cmd: str, **args) -> dict:
        """Execute one command synchronously in the calling thread (headless/in-process use and tests)."""
        fut = self.submit(cmd, args)
        while not fut.done():
            self.tick()
        return fut.result()

    # ------------------------------------------------------------------ loops
    def run_headless(self, stop: threading.Event | None = None) -> None:
        stop = stop or threading.Event()
        while not stop.is_set():
            # Nothing to do: block for the next command instead of burning CPU on idle physics.
            if self.idle and self.requests.empty():
                try:
                    cmd, args, fut = self.requests.get(timeout=0.2)
                except queue.Empty:
                    continue
                self._begin(cmd, args, fut)
            self.tick()

    def run_with_viewer(self) -> None:
        import mujoco.viewer

        dt = self.model.opt.timestep
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            wall_start, sim_start, last_sync = time.perf_counter(), self.data.time, 0.0
            while viewer.is_running():
                self.tick()
                now = time.perf_counter()
                if now - last_sync > 1 / 60:
                    viewer.sync()
                    last_sync = now
                # Real-time pacing: never run ahead of the wall clock.
                ahead = (self.data.time - sim_start) - (now - wall_start)
                if ahead > dt:
                    time.sleep(ahead)
                elif ahead < -0.25:  # fell behind (e.g. long IK solve); resync instead of fast-forwarding
                    wall_start, sim_start = now, self.data.time


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        sim: Simulation = self.server.sim  # type: ignore[attr-defined]
        for line in self.rfile:
            if not line.strip():
                continue
            req_id = None
            try:
                req = json.loads(line)
                req_id = req.get("id")
                result = sim.submit(req["cmd"], req.get("args") or {}).result(timeout=300)
                resp = {"id": req_id, "ok": True, "result": result}
            except Exception as exc:
                resp = {"id": req_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            self.wfile.write((json.dumps(resp, default=_to_json) + "\n").encode())
            self.wfile.flush()


class _TCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_tcp_server(sim: Simulation, port: int = DEFAULT_PORT) -> _TCPServer:
    server = _TCPServer(("127.0.0.1", port), _Handler)
    server.sim = sim  # type: ignore[attr-defined]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true", help="no viewer window; physics runs as fast as possible")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    sim = Simulation()
    start_tcp_server(sim, args.port)
    print(f"robot sim listening on 127.0.0.1:{args.port} ({'headless' if args.headless else 'viewer'})", flush=True)
    if args.headless:
        sim.run_headless()
    else:
        sim.run_with_viewer()


if __name__ == "__main__":
    main()

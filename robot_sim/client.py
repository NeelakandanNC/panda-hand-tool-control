"""Client for the simulation process (robot_sim.server): newline-delimited JSON over TCP."""

import json
import socket
import threading

from robot_sim.server import DEFAULT_PORT


class SimError(RuntimeError):
    """The simulation rejected a command (bad arguments, unknown command, ...)."""


class SimClient:
    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, timeout: float = 300.0):
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._file = self._sock.makefile("rwb")
        self._lock = threading.Lock()
        self._next_id = 0

    def call(self, cmd: str, **args) -> dict:
        """Send one command and wait for its reply (motion commands reply once the robot has settled)."""
        with self._lock:
            self._next_id += 1
            request = {"id": self._next_id, "cmd": cmd, "args": args}
            self._file.write((json.dumps(request) + "\n").encode())
            self._file.flush()
            line = self._file.readline()
        if not line:
            raise ConnectionError("simulation closed the connection")
        reply = json.loads(line)
        if not reply["ok"]:
            raise SimError(reply["error"])
        return reply["result"]

    def close(self) -> None:
        self._file.close()
        self._sock.close()

    def __enter__(self) -> "SimClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

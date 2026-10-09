"""Ways for the agent to execute tools: through the real MCP server, or in-process (tests)."""

import json
import sys
from contextlib import ExitStack, asynccontextmanager

import anyio.from_thread
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent.config import ROOT
from robot_tools.catalog import build_catalog, compact_state
from robot_tools.mcp_server import run_tool


def _content_text(result) -> str:
    parts = []
    for item in result.content:
        parts.append(item.text if item.type == "text" else f"[{item.type}]")
    return "\n".join(parts)


class MCPExecutor:
    """Spawns `robot_tools.mcp_server` over stdio and calls its tools from synchronous code."""

    def __init__(self, sim_port: int):
        self._stack = ExitStack()
        self._portal = self._stack.enter_context(anyio.from_thread.start_blocking_portal())
        params = StdioServerParameters(command=sys.executable, cwd=str(ROOT),
                                       args=["-m", "robot_tools.mcp_server", "--sim-port", str(sim_port)])
        self._session = self._stack.enter_context(self._portal.wrap_async_context_manager(self._connect(params)))
        tools = self._portal.call(self._session.list_tools).tools
        self._schemas = {t.name: {"description": t.description, "input_schema": t.input_schema} for t in tools}

    @staticmethod
    @asynccontextmanager
    async def _connect(params):
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            yield session

    def tool_schemas(self) -> dict[str, dict]:
        return self._schemas

    def call(self, name: str, arguments: dict) -> tuple[str, bool]:
        result = self._portal.call(self._session.call_tool, name, arguments)
        return _content_text(result), bool(result.is_error)

    def state(self) -> dict:
        text, _ = self.call("get_robot_state", {})
        return compact_state(json.loads(text))

    def close(self) -> None:
        self._stack.close()


class InProcessExecutor:
    """Runs catalog tools directly against a Simulation in this thread (no MCP, no TCP)."""

    def __init__(self, sim):
        from robot_sim.client import SimError

        class _Client:
            def call(self, cmd, **args):
                try:
                    return sim.run_command(cmd, **args)
                except ValueError as exc:
                    raise SimError(str(exc)) from exc

        self.sim, self._client = sim, _Client()
        self._catalog = {t.name: t for t in build_catalog(sim.controller.describe())}

    def tool_schemas(self) -> dict[str, dict]:
        return {n: {"description": t.description, "input_schema": t.input_schema} for n, t in self._catalog.items()}

    def call(self, name: str, arguments: dict) -> tuple[str, bool]:
        result = run_tool(self._catalog[name], self._client, arguments)
        return _content_text(result), bool(result.is_error)

    def state(self) -> dict:
        return compact_state(self.sim.controller.get_state())

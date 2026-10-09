"""MCP server exposing every tool in the generated catalog; each call is forwarded to the running sim.

    scripts/mjpython.sh -m robot_sim.server                 # 1. the simulation (window)
    uv run python -m robot_tools.mcp_server                 # 2. this server, on stdio

The tool list is built at startup from the simulation's `describe` reply, so it always matches
the robot that is actually running.
"""

import argparse
import json

import anyio
import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from robot_sim.client import SimClient, SimError
from robot_sim.server import DEFAULT_PORT
from robot_tools.catalog import ToolSpec, build_catalog

INSTRUCTIONS = (
    "Tools to control a Franka Panda arm with a Shadow dexterous hand above a table with objects. "
    "Every motion tool returns the resulting state (palm pose, finger curl, objects: held / lifted / in zone)."
)


def _text(payload, is_error: bool = False) -> types.CallToolResult:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)], is_error=is_error)


def run_tool(spec: ToolSpec, sim: SimClient, arguments: dict) -> types.CallToolResult:
    """Execute one catalog tool synchronously and package the result for MCP."""
    missing = [k for k in spec.input_schema.get("required", []) if k not in arguments]
    unknown = [k for k in arguments if k not in spec.input_schema["properties"]]
    if missing or unknown:
        return _text(f"bad arguments for {spec.name}: missing {missing}, unknown {unknown}", is_error=True)
    try:
        result = spec.run(sim, arguments)
    except SimError as exc:
        return _text(f"simulation rejected the command: {exc}", is_error=True)
    if "png_base64" in result:
        return types.CallToolResult(content=[types.ImageContent(type="image", data=result["png_base64"],
                                                                mime_type="image/png")])
    return _text(result, is_error=result.get("moved") is False)  # e.g. unreachable palm target


def create_server(sim: SimClient) -> tuple[Server, list[ToolSpec]]:
    catalog = build_catalog(sim.call("describe"))
    by_name = {t.name: t for t in catalog}
    tool_list = [types.Tool(name=t.name, description=t.description, input_schema=t.input_schema) for t in catalog]

    async def list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tool_list)

    async def call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        spec = by_name.get(params.name)
        if spec is None:
            return _text(f"unknown tool {params.name!r}", is_error=True)
        # SimClient is blocking (motions take seconds): keep the event loop free.
        return await anyio.to_thread.run_sync(run_tool, spec, sim, params.arguments or {})

    server = Server("robot-hand", version="0.1.0", instructions=INSTRUCTIONS,
                    on_list_tools=list_tools, on_call_tool=call_tool)
    return server, catalog


async def serve_stdio(sim_port: int) -> None:
    sim = SimClient(port=sim_port)
    server, _ = create_server(sim)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim-port", type=int, default=DEFAULT_PORT, help="port of the running robot_sim.server")
    args = parser.parse_args()
    anyio.run(serve_stdio, args.sim_port)


if __name__ == "__main__":
    main()

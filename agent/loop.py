"""The agent loop: the LLM is given the robot's MCP tools, picks one -> execute via MCP -> repeat.

Every step is a fresh, self-contained request (task + action history + current state).
"""

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Protocol

from agent.config import Settings

FINISH_TOOL = {
    "type": "function",
    "name": "finish",
    "description": "End the task. Call with success=true once the state shows the task is done, "
                   "or success=false if it cannot be done.",
    "parameters": {
        "type": "object",
        "properties": {"success": {"type": "boolean"},
                       "summary": {"type": "string", "description": "One sentence on the outcome."}},
        "required": ["success", "summary"],
        "additionalProperties": False,
    },
    "strict": True,
}

INSTRUCTIONS = """\
You control a robot: a 7-joint arm with a five-fingered dexterous hand (thumb, index, middle, ring,
little) mounted above a table with objects on it. World frame in metres: +x away from the robot
base, +y the robot's left, +z up; the table top is at z=0.40.

Call exactly one tool per turn. You then see its result and the new state on the next turn.
- Work step by step: approach, act, then check the state (e.g. objects' held / lifted_cm /
  in_target_zone) before moving on.
- If an action fails (e.g. a position is unreachable), try a different approach.
- When the state shows the task is complete, call finish(success=true). Call finish(success=false)
  only if the task is clearly impossible.
"""


class Executor(Protocol):
    def tool_schemas(self) -> dict[str, dict]: ...           # name -> {"description", "input_schema"}
    def call(self, name: str, arguments: dict) -> tuple[str, bool]: ...  # (result text, is_error)
    def state(self) -> dict: ...                              # compact robot/object state


@dataclass
class StepRecord:
    step: int
    tool: str | None
    arguments: dict
    outcome: str
    is_error: bool
    llm_s: float
    tool_s: float
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class RunResult:
    task: str
    success: bool
    summary: str
    steps: list[StepRecord] = field(default_factory=list)
    seconds: float = 0.0
    model: str = ""

    def to_json(self) -> dict:
        return asdict(self)


def summarize_result(text: str, is_error: bool) -> str:
    """One line describing a tool result, for the action history."""
    if is_error:
        return "FAILED: " + text[:200]
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return text[:200]
    if "state" not in result:
        return "ok"
    state = result["state"]
    notes = [f"palm at {state['palm']['position']} facing {state['palm']['facing']}"]
    for name, obj in state["objects"].items():
        flags = [f for f in ("held", "in_target_zone") if obj[f]]
        if abs(obj["lifted_cm"]) >= 1:
            flags.append(f"lifted {obj['lifted_cm']} cm")
        if obj["touching_hand_parts"] and not obj["held"]:
            flags.append("touching " + "/".join(obj["touching_hand_parts"]))
        if flags:
            notes.append(f"{name}: {', '.join(flags)}")
    return "ok; " + "; ".join(notes)


class Agent:
    def __init__(self, llm, executor: Executor, settings: Settings | None = None):
        self.llm, self.executor = llm, executor
        self.settings = settings or Settings()
        self.schemas = executor.tool_schemas()
        self.tools = [self._function_tool(n) for n in self.schemas] + [FINISH_TOOL]

    def _function_tool(self, name: str) -> dict:
        schema = self.schemas[name]
        return {"type": "function", "name": name, "description": schema["description"],
                "parameters": schema["input_schema"], "strict": False}

    def _ask(self, task: str, step: int, max_steps: int, history: list[str], state: dict):
        prompt = "\n".join([
            f"Task: {task}",
            f"Turn {step} of at most {max_steps}.",
            "Actions so far:" if history else "No actions taken yet.",
            *history[-15:],
            "Current state: " + json.dumps(state),
        ])
        extra = {"reasoning": {"effort": self.settings.agent_reasoning_effort}} \
            if self.settings.agent_reasoning_effort else {}
        return self.llm.responses.create(
            model=self.settings.agent_model, instructions=INSTRUCTIONS, input=prompt,
            tools=self.tools,
            tool_choice="required", parallel_tool_calls=False, **extra,
        )

    def run(self, task: str, max_steps: int | None = None,
            on_step: Callable[[StepRecord], None] | None = None) -> RunResult:
        max_steps = max_steps or self.settings.max_steps
        result = RunResult(task, False, "step limit reached", model=self.settings.agent_model)
        history: list[str] = []
        start = time.time()
        for step in range(1, max_steps + 1):
            state = self.executor.state()
            t1 = time.time()
            response = self._ask(task, step, max_steps, history, state)
            t2 = time.time()
            call = next((item for item in response.output if item.type == "function_call"), None)
            usage = getattr(response, "usage", None)
            tokens = (getattr(usage, "input_tokens", 0) or 0, getattr(usage, "output_tokens", 0) or 0)

            finished = False
            if call is None:
                tool, args, outcome, is_error = None, {}, "no tool call: " + (response.output_text or "")[:200], True
            else:
                tool, args = call.name, json.loads(call.arguments or "{}")
                if tool == "finish":
                    finished, outcome, is_error = True, args.get("summary", ""), False
                    result.success, result.summary = bool(args.get("success")), outcome
                elif tool not in self.schemas:
                    outcome, is_error = f"FAILED: unknown tool {tool!r}", True
                else:
                    text, is_error = self.executor.call(tool, args)
                    outcome = summarize_result(text, is_error)
            record = StepRecord(step, tool, args, outcome, is_error, t2 - t1,
                                time.time() - t2 if tool not in (None, "finish") else 0.0, *tokens)
            result.steps.append(record)
            if on_step:
                on_step(record)
            if finished:
                break
            history.append(f"{step}. {tool or '(none)'}({json.dumps(args)}) -> {outcome}")
        result.seconds = time.time() - start
        return result

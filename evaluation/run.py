"""Benchmark: an LLM given the robot's MCP tools, on tasks checked against the simulator.

    uv run python -m evaluation.run                              # all tasks, 1 repeat
    uv run python -m evaluation.run --repeats 3 --workers 6
    uv run python -m evaluation.run --tasks peace_sign fist

Each run gets its own headless simulation + MCP server, so runs can go in parallel.
Success is checked against the final simulator state, not the LLM's own claim.
Results go to evaluation/results/<time>/.
"""

import argparse
import json
import statistics
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from agent.config import ROOT, Settings
from agent.executors import MCPExecutor
from agent.loop import Agent
from agent.run import start_headless_sim
from evaluation.tasks import TASKS, TASKS_BY_ID

RESULTS_DIR = ROOT / "evaluation" / "results"


def run_one(task, repeat: int, llm, settings: Settings, max_steps: int) -> dict:
    port, stop_sim = start_headless_sim()
    executor = MCPExecutor(port)
    record = {"task": task.id, "kind": task.kind, "repeat": repeat}
    try:
        result = Agent(llm, executor, settings).run(task.prompt, max_steps)
        steps = [s for s in result.steps if s.tool != "finish"]
        record |= {
            "success": bool(task.check(executor.state())),
            "agent_claimed_success": result.success,
            "steps": len(steps),
            "seconds": round(result.seconds, 1),
            "llm_seconds": round(sum(s.llm_s for s in result.steps), 1),
            "robot_seconds": round(sum(s.tool_s for s in result.steps), 1),
            "tokens_in": sum(s.tokens_in for s in result.steps),
            "tokens_out": sum(s.tokens_out for s in result.steps),
            "tool_errors": sum(s.is_error for s in steps),
            "summary": result.summary,
            "trace": [f"{s.tool}({json.dumps(s.arguments)}) -> {s.outcome}" for s in result.steps],
        }
    except Exception as exc:  # keep the benchmark going; record the failure
        record |= {"success": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    finally:
        executor.close()
        stop_sim()
    return record


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else float("nan")


def report(records: list[dict], settings: Settings) -> str:
    ok = sum(r["success"] for r in records)
    false_claims = sum(r.get("agent_claimed_success", False) and not r["success"] for r in records)
    lines = [
        "# Robot tool-use benchmark", "",
        f"Model `{settings.agent_model}`, all MCP tools offered every step.", "",
        f"- success: **{ok}/{len(records)} ({ok / max(len(records), 1):.0%})**",
        f"- LLM claimed success but the check failed: {false_claims}",
        f"- avg steps {_mean([r.get('steps') for r in records]):.1f}, "
        f"avg time {_mean([r.get('seconds') for r in records]):.0f}s "
        f"(LLM {_mean([r.get('llm_seconds') for r in records]):.0f}s), "
        f"avg tokens in {_mean([r.get('tokens_in') for r in records]):,.0f} / "
        f"out {_mean([r.get('tokens_out') for r in records]):,.0f}", "",
        "| task | kind | success | avg steps | avg time (s) | avg tokens in |",
        "|---|---|---|---|---|---|",
    ]
    for task in dict.fromkeys(r["task"] for r in records):
        rs = [r for r in records if r["task"] == task]
        lines.append(f"| {task} | {TASKS_BY_ID[task].kind} | {sum(r['success'] for r in rs)}/{len(rs)} | "
                     f"{_mean([r.get('steps') for r in rs]):.1f} | {_mean([r.get('seconds') for r in rs]):.0f} | "
                     f"{_mean([r.get('tokens_in') for r in rs]):,.0f} |")
    errors = [r for r in records if "error" in r]
    if errors:
        lines += ["", "## Crashed runs", ""] + [f"- {r['task']} #{r['repeat']}: {r['error']}" for r in errors]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="*", default=[t.id for t in TASKS], choices=[t.id for t in TASKS])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=20)
    args = parser.parse_args()

    settings = Settings()
    llm = OpenAI()
    jobs = [(TASKS_BY_ID[t], r) for r in range(args.repeats) for t in args.tasks]
    out_dir = RESULTS_DIR / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True)
    print(f"{len(jobs)} runs ({len(args.tasks)} tasks x {args.repeats}) with {settings.agent_model}, "
          f"{args.workers} in parallel -> {out_dir.relative_to(ROOT)}", flush=True)

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, task, rep, llm, settings, args.max_steps) for task, rep in jobs]
        for fut in as_completed(futures):
            r = fut.result()
            records.append(r)
            detail = r.get("error") or f"{r['steps']} steps, {r['seconds']}s, {r['tokens_in']:,} tokens in"
            print(f"  {'✓' if r['success'] else '✗'} {r['task']:18} #{r['repeat']}  {detail}", flush=True)
            (out_dir / "results.json").write_text(json.dumps(records, indent=1))

    text = report(records, settings)
    (out_dir / "report.md").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    main()

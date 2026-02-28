import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # Optional dependency; environment variables may already be present.
    pass

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run-task-batch",
        description="Run repo-assist agent on a task JSON/JSONL file and save outputs.",
    )
    parser.add_argument("--repo-path", required=True, help="Path to repository to analyze")
    parser.add_argument("--tasks", required=True, help="Path to task file (.jsonl or .json)")
    parser.add_argument("--output", required=True, help="Output JSONL path for model responses")

    parser.add_argument("--model", default="gemini-2.5-flash", help="Model name")
    parser.add_argument("--default-mode", choices=["explain", "locate", "suggest", "patch"], default="explain")
    parser.add_argument("--default-scope", choices=["files-only", "include-pr"], default="include-pr")
    parser.add_argument("--max-turns", type=int, default=10, help="Max orchestrator turns per task")

    parser.add_argument("--github-owner", default=None, help="GitHub owner/org")
    parser.add_argument("--github-repo", default=None, help="GitHub repo name")
    parser.add_argument("--session-id", default=None, help="Resume existing session ID")

    parser.add_argument("--limit", type=int, default=None, help="Run only first N tasks")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N tasks")
    parser.add_argument("--append", action="store_true", help="Append to output instead of overwrite")
    parser.add_argument("--verbose", action="store_true", help="Verbose tool/turn logging")
    return parser.parse_args()


def load_tasks(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows

    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("tasks"), list):
            return obj["tasks"]
        raise ValueError("JSON task file must be a list or {\"tasks\": [...]}.")

    raise ValueError("Task file must be .jsonl or .json")


def citation_to_dict(citation) -> dict:
    return {
        "file_path": citation.file_path,
        "start_line": citation.start_line,
        "end_line": citation.end_line,
        "snippet": citation.snippet,
        "source_type": citation.source_type,
        "ref_id": citation.ref_id,
    }


def ensure_output_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    from src.agent_orchestrator import AgentOrchestrator
    from src.session_manager import SessionManager
    from src.tool_gateway import ToolGateway

    repo_path = Path(args.repo_path)
    tasks_path = Path(args.tasks)
    output_path = Path(args.output)

    if not repo_path.exists():
        print(f"Error: repo path not found: {repo_path}", file=sys.stderr)
        sys.exit(1)

    try:
        tasks = load_tasks(tasks_path)
    except Exception as exc:
        print(f"Error loading tasks: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.offset:
        tasks = tasks[args.offset :]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    if not tasks:
        print("No tasks to run after applying offset/limit.")
        sys.exit(0)

    try:
        gateway = ToolGateway(
            repo_path=str(repo_path),
            github_owner=args.github_owner,
            github_repo=args.github_repo,
        )
    except Exception as exc:
        print(f"Error loading repository: {exc}", file=sys.stderr)
        sys.exit(1)

    session_mgr = SessionManager()
    if args.session_id and session_mgr.session_exists(args.session_id):
        session_mgr.load_session(args.session_id)
    else:
        session_mgr.create_session(
            repo_path=str(repo_path),
            github_owner=args.github_owner,
            github_repo=args.github_repo,
            mode=args.default_mode,
            scope=args.default_scope,
            verbose=args.verbose,
        )

    try:
        orchestrator = AgentOrchestrator(
            gateway=gateway,
            session=session_mgr,
            model=args.model,
        )
    except Exception as exc:
        print(f"Error creating orchestrator: {exc}", file=sys.stderr)
        sys.exit(1)

    ensure_output_path(output_path)
    write_mode = "a" if args.append else "w"

    total = len(tasks)
    print(f"Running {total} tasks | session={session_mgr.session_id} | output={output_path}")

    with output_path.open(write_mode, encoding="utf-8") as out:
        for idx, task in enumerate(tasks, start=1):
            task_id = str(task.get("task_id", f"task_{idx}"))
            question = task.get("question") or task.get("user_query") or task.get("prompt")
            mode = task.get("mode", args.default_mode)
            scope = task.get("scope", args.default_scope)

            if not question:
                row = {
                    "task_id": task_id,
                    "status": "error",
                    "error": "Task missing `question` (or `user_query`/`prompt`) field.",
                    "started_at": utc_now(),
                    "finished_at": utc_now(),
                }
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(f"[{idx}/{total}] {task_id} -> error (missing question)")
                continue

            started_at = utc_now()
            t0 = time.time()

            try:
                result = orchestrator.run(
                    question,
                    mode=mode,
                    scope=scope,
                    max_turns=args.max_turns,
                    verbose=args.verbose,
                )
                elapsed_ms = int((time.time() - t0) * 1000)

                row = {
                    "task_id": task_id,
                    "repo": str(repo_path),
                    "model": args.model,
                    "mode": mode,
                    "scope": scope,
                    "status": "ok",
                    "error": None,
                    "question": question,
                    "answer_text": result.final_response.answer_text,
                    "citations": [citation_to_dict(c) for c in result.final_response.citations],
                    "patch_diff": result.final_response.patch_diff,
                    "next_actions": result.final_response.next_actions,
                    "tool_call_count": len(result.executed_tool_calls),
                    "latency_ms": elapsed_ms,
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "session_id": session_mgr.session_id,
                }
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(f"[{idx}/{total}] {task_id} -> ok ({elapsed_ms} ms)")

            except Exception as exc:
                elapsed_ms = int((time.time() - t0) * 1000)
                row = {
                    "task_id": task_id,
                    "repo": str(repo_path),
                    "model": args.model,
                    "mode": mode,
                    "scope": scope,
                    "status": "error",
                    "error": str(exc),
                    "question": question,
                    "answer_text": "",
                    "citations": [],
                    "patch_diff": None,
                    "next_actions": [],
                    "tool_call_count": 0,
                    "latency_ms": elapsed_ms,
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "session_id": session_mgr.session_id,
                }
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(f"[{idx}/{total}] {task_id} -> error ({elapsed_ms} ms): {exc}")

    print("Batch run complete.")


if __name__ == "__main__":
    main()

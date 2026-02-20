import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.tool_gateway import ToolGateway
from src.session_manager import SessionManager
from src.agent_orchestrator import AgentOrchestrator, OrchestratorResult


def _print_text(result: OrchestratorResult, session_id: str, verbose: bool = False):
    resp = result.final_response

    print("\n" + "=" * 60)
    print("Answer:")
    print("  " + resp.answer_text.replace("\n", "\n  "))

    if resp.citations:
        print("\nCitations:")
        file_cits = [c for c in resp.citations if c.source_type == "file"]
        issue_cits = [c for c in resp.citations if c.source_type == "issue"]
        pr_cits = [c for c in resp.citations if c.source_type == "pr"]

        for i, c in enumerate(file_cits, 1):
            loc = c.file_path
            if c.start_line and c.end_line:
                loc += f":{c.start_line}-{c.end_line}"
            elif c.start_line:
                loc += f":{c.start_line}"
            print(f"  [{i}] {loc}")

        for c in issue_cits:
            print(f"  [Issue #{c.ref_id}] {c.snippet}  {c.file_path}")

        for c in pr_cits:
            print(f"  [PR #{c.ref_id}] {c.snippet}  {c.file_path}")

    if resp.patch_diff:
        print("\nPatch:")
        print("```diff")
        print(resp.patch_diff)
        print("```")

    if resp.next_actions:
        print("\nNext Actions:")
        for action in resp.next_actions:
            print(f"  - [ ] {action}")

    if verbose:
        print(f"\nTool calls executed: {len(result.executed_tool_calls)}")
        for call in result.executed_tool_calls:
            status = "OK" if not call.error else f"ERROR: {call.error}"
            print(f"  {call.tool_name}({call.args}) → {status}")

    print(f"\nSession: {session_id}")
    print("=" * 60)


def _print_json(result: OrchestratorResult, session_id: str):
    output = {
        "session_id": session_id,
        "tool_call_plan": [
            {"tool_name": t.tool_name, "args": t.args, "rationale": t.rationale}
            for t in result.tool_call_plan
        ],
        "executed_tool_calls": [
            {"tool_name": c.tool_name, "args": c.args, "error": c.error}
            for c in result.executed_tool_calls
        ],
        "consolidated_evidence": [
            {
                "file_path": c.file_path,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "snippet": c.snippet,
                "source_type": c.source_type,
                "ref_id": c.ref_id,
            }
            for c in result.consolidated_evidence
        ],
        "final_response": {
            "answer_text": result.final_response.answer_text,
            "citations": [
                {
                    "file_path": c.file_path,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "snippet": c.snippet,
                    "source_type": c.source_type,
                    "ref_id": c.ref_id,
                }
                for c in result.final_response.citations
            ],
            "patch_diff": result.final_response.patch_diff,
            "next_actions": result.final_response.next_actions,
        },
    }
    print(json.dumps(output, indent=2))


def _setup_session(args, session_mgr: SessionManager) -> str:
    if args.session:
        if session_mgr.session_exists(args.session):
            session_mgr.load_session(args.session)
            if args.verbose:
                print(f"[Session] Resumed session {args.session}")
        else:
            print(f"Warning: session '{args.session}' not found — starting a new session.")
            session_mgr.create_session(
                args.repo_path,
                github_owner=args.github_owner,
                github_repo=args.github_repo,
                mode=args.mode,
                scope=args.scope,
                verbose=args.verbose,
            )
    else:
        session_mgr.create_session(
            args.repo_path,
            github_owner=args.github_owner,
            github_repo=args.github_repo,
            mode=args.mode,
            scope=args.scope,
            verbose=args.verbose,
        )
        if args.verbose:
            print(f"[Session] Created session {session_mgr.session_id}")
    return session_mgr.session_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repo-assist",
        description="AI-powered repository assistant with citations and session history.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli ./my-repo "How does authentication work?" --mode explain
  python -m src.cli ./my-repo "Where is the login handler?" --mode locate --scope files-only
  python -m src.cli ./my-repo "What should I work on?" --mode suggest --github-owner myorg --github-repo myrepo
  python -m src.cli ./my-repo "Follow up question" --session abc123def456
  python -m src.cli ./my-repo "Explain the DB schema" --output json
  python -m src.cli ./my-repo --chat
        """,
    )

    parser.add_argument("repo_path", help="Path to the local repository")
    parser.add_argument("user_query", nargs="?", default=None,
                        help="Natural language question or request (omit when using --chat)")
    parser.add_argument("--mode", choices=["explain", "locate", "suggest", "patch"],
                        default="explain", help="Response mode (default: explain)")
    parser.add_argument("--scope", choices=["files-only", "include-pr"], default="include-pr",
                        help="Tool scope: include GitHub issues/PRs or only use code files (default: include-pr)")
    parser.add_argument("--github-owner", default=None, help="GitHub repository owner/org")
    parser.add_argument("--github-repo", default=None, help="GitHub repository name")
    parser.add_argument("--session", default=None, help="Resume an existing session by ID")
    parser.add_argument("--output", choices=["text", "json"], default="text",
                        help="Output format (default: text)")
    parser.add_argument("--verbose", action="store_true", help="Show tool calls and debug info")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat mode")
    parser.add_argument("--model", default="gemini-2.5-flash",
                        help="Gemini model name (default: gemini-2.5-flash)")

    return parser


def run_chat(orchestrator: AgentOrchestrator, session_mgr: SessionManager, args):
    print("Repo Assist — Interactive Chat")
    print(f"Repository : {args.repo_path}")
    print(f"Session    : {session_mgr.session_id}")
    print(f"Mode       : {args.mode}  |  Scope: {args.scope}")
    print("Type 'quit' or 'exit' to end. Type 'mode <name>' to switch modes.")
    print("-" * 60)

    mode = args.mode
    scope = args.scope

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print(f"Session saved: {session_mgr.session_id}")
            break

        if query.lower().startswith("mode "):
            new_mode = query.split(" ", 1)[1].strip()
            if new_mode in ("explain", "locate", "suggest", "patch"):
                mode = new_mode
                session_mgr.update_settings(mode=mode)
                print(f"Mode switched to: {mode}")
            else:
                print(f"Unknown mode '{new_mode}'. Choose: explain | locate | suggest | patch")
            continue

        if query.lower().startswith("scope "):
            new_scope = query.split(" ", 1)[1].strip()
            if new_scope in ("files-only", "include-pr"):
                scope = new_scope
                session_mgr.update_settings(scope=scope)
                print(f"Scope switched to: {scope}")
            else:
                print("Unknown scope. Choose: files-only | include-pr")
            continue

        print("\nAssistant: ", end="", flush=True)
        try:
            result = orchestrator.run(query, mode=mode, scope=scope, verbose=args.verbose)
        except Exception as e:
            print(f"[Error] {e}")
            continue

        if args.output == "json":
            _print_json(result, session_mgr.session_id)
        else:
            resp = result.final_response
            print(resp.answer_text)

            if resp.citations:
                print("\nCitations:")
                for i, c in enumerate([x for x in resp.citations if x.source_type == "file"], 1):
                    loc = c.file_path
                    if c.start_line and c.end_line:
                        loc += f":{c.start_line}-{c.end_line}"
                    print(f"  [{i}] {loc}")
                for c in [x for x in resp.citations if x.source_type in ("issue", "pr")]:
                    label = c.source_type.upper()
                    print(f"  [{label} #{c.ref_id}] {c.snippet}")

            if resp.next_actions:
                print("\nNext Actions:")
                for action in resp.next_actions:
                    print(f"  - [ ] {action}")

            if resp.patch_diff:
                print("\nPatch:")
                print("```diff")
                print(resp.patch_diff)
                print("```")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.chat and not args.user_query:
        parser.error("user_query is required unless --chat is specified")

    repo_path = Path(args.repo_path)
    if not repo_path.exists():
        print(f"Error: repository path not found: {repo_path}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"[Loading] repository at {repo_path} ...")

    try:
        gateway = ToolGateway(
            repo_path=str(repo_path),
            github_owner=args.github_owner,
            github_repo=args.github_repo,
        )
    except Exception as e:
        print(f"Error loading repository: {e}", file=sys.stderr)
        sys.exit(1)

    session_mgr = SessionManager()
    session_id = _setup_session(args, session_mgr)

    try:
        orchestrator = AgentOrchestrator(
            gateway=gateway,
            session=session_mgr,
            model=args.model,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.chat:
        run_chat(orchestrator, session_mgr, args)
        return

    if args.verbose:
        print(f"[Query] mode={args.mode}  scope={args.scope}")
        print(f"[Query] {args.user_query}\n")

    try:
        result = orchestrator.run(
            args.user_query,
            mode=args.mode,
            scope=args.scope,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"Error during query: {e}", file=sys.stderr)
        sys.exit(1)

    if args.output == "json":
        _print_json(result, session_id)
    else:
        _print_text(result, session_id, verbose=args.verbose)


if __name__ == "__main__":
    main()

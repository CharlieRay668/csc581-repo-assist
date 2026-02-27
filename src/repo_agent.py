import os
import time
from google import genai
from google.genai import types
from src.tool_gateway import ToolGateway
from src.agent_orchestrator import AgentOrchestrator
from src.session_manager import SessionManager


class RepoAgent:

    def __init__(self, gateway, api_key=None, model="gemini-2.5-flash", session=None):
        self.gateway = gateway
        self.model = model
        self.session = session

        api_key = api_key or os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable or api_key parameter required")

        self.client = genai.Client(api_key=api_key)
        self.tools = self._define_tools()
        self._orchestrator = AgentOrchestrator(
            gateway=gateway,
            session=session,
            api_key=api_key,
            model=model,
        )

    def _define_tools(self):
        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="search_repo",
                        description="Search the repository code for files and functions matching a query. Use this to find where specific functionality is implemented.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "What to search for"},
                                "top_k": {"type": "integer", "description": "Maximum number of results to return", "default": 5}
                            },
                            "required": ["query"]
                        }
                    ),
                    types.FunctionDeclaration(
                        name="open_file",
                        description="Read the contents of a specific file or file range.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "File path relative to repository root"},
                                "start_line": {"type": "integer", "description": "Starting line number (1-indexed, optional)"},
                                "end_line": {"type": "integer", "description": "Ending line number (1-indexed, inclusive, optional)"}
                            },
                            "required": ["path"]
                        }
                    ),
                    types.FunctionDeclaration(
                        name="get_issues",
                        description="Get GitHub issues for the repository.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search text in issue titles and bodies (optional)"},
                                "state": {"type": "string", "description": "Issue state: 'open', 'closed', or 'all'", "default": "open"},
                                "limit": {"type": "integer", "description": "Maximum number of issues to return", "default": 10}
                            }
                        }
                    ),
                    types.FunctionDeclaration(
                        name="get_pull_requests",
                        description="Get GitHub pull requests for the repository.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search text in PR titles and bodies (optional)"},
                                "state": {"type": "string", "description": "PR state: 'open', 'closed', or 'all'", "default": "open"},
                                "limit": {"type": "integer", "description": "Maximum number of PRs to return", "default": 10}
                            }
                        }
                    ),
                    types.FunctionDeclaration(
                        name="get_repo_stats",
                        description="Get statistics about the loaded repository including file count, chunks, issues, and PRs.",
                        parameters={"type": "object", "properties": {}}
                    )
                ]
            )
        ]

    def _execute_tool(self, tool_name, args):
        try:
            if tool_name == "search_repo":
                results = self.gateway.search_repo(args.get("query"), top_k=args.get("top_k", 5))
                return {"results": results, "count": len(results)}
            elif tool_name == "open_file":
                return self.gateway.open_file(args.get("path"), args.get("start_line"), args.get("end_line"))
            elif tool_name == "get_issues":
                results = self.gateway.get_issues(query=args.get("query"), state=args.get("state", "open"), limit=args.get("limit", 10))
                return {"issues": results, "count": len(results)}
            elif tool_name == "get_pull_requests":
                results = self.gateway.get_pull_requests(query=args.get("query"), state=args.get("state", "open"), limit=args.get("limit", 10))
                return {"pull_requests": results, "count": len(results)}
            elif tool_name == "get_repo_stats":
                return self.gateway.stats()
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {"error": str(e)}

    def ask(self, question, max_turns=10, verbose=False, mode="explain", scope="include-pr"):
        result = self._orchestrator.run(
            query=question,
            mode=mode,
            scope=scope,
            max_turns=max_turns,
            verbose=verbose,
        )
        return result.final_response.answer_text

    def chat(self, verbose=False):
        print("Repo Agent Chat (type 'quit' to exit)")
        print(f"Repository: {self.gateway.stats().get('repo_path', 'N/A')}")
        print("-" * 50)

        while True:
            question = input("\nYou: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not question:
                continue

            print("\nAgent:", end=" ", flush=True)
            answer = self.ask(question, verbose=verbose)
            print(answer)


if __name__ == '__main__':
    from src.cli import main
    main()

import os
import time
from google import genai
from google.genai import types
from src.tool_gateway import ToolGateway


class RepoAgent:
    """
    Agent that uses Gemini with tool calling to answer repository questions.
    """
    
    def __init__(self, gateway, api_key=None, model="gemini-2.0-flash"):
        """
        Initialize the repo agent.
        
        Args:
            gateway: ToolGateway instance
            api_key: Google API key (reads from GEMINI_API_KEY env if not provided)
            model: Gemini model to use
        """
        self.gateway = gateway
        self.model = model
        
        # Configure Gemini client
        api_key = api_key or os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable or api_key parameter required")
        
        self.client = genai.Client(api_key=api_key)
        
        # Define available tools
        self.tools = self._define_tools()
    
    def _define_tools(self):
        """Define tools in Gemini's function calling format."""
        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="search_repo",
                        description="Search the repository code for files and functions matching a query. Use this to find where specific functionality is implemented.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "What to search for (e.g., 'authentication', 'login function', 'database connection')"
                                },
                                "top_k": {
                                    "type": "integer",
                                    "description": "Maximum number of results to return",
                                    "default": 5
                                }
                            },
                            "required": ["query"]
                        }
                    ),
                    types.FunctionDeclaration(
                        name="open_file",
                        description="Read the contents of a specific file or file range. Use this after finding relevant files to get more context.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "File path relative to repository root"
                                },
                                "start_line": {
                                    "type": "integer",
                                    "description": "Starting line number (1-indexed, optional)"
                                },
                                "end_line": {
                                    "type": "integer",
                                    "description": "Ending line number (1-indexed, inclusive, optional)"
                                }
                            },
                            "required": ["path"]
                        }
                    ),
                    types.FunctionDeclaration(
                        name="get_issues",
                        description="Get GitHub issues for the repository. Use this to understand bugs, feature requests, or ongoing work.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search text in issue titles and bodies (optional)"
                                },
                                "state": {
                                    "type": "string",
                                    "description": "Issue state: 'open', 'closed', or 'all'",
                                    "default": "open"
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "Maximum number of issues to return",
                                    "default": 10
                                }
                            }
                        }
                    ),
                    types.FunctionDeclaration(
                        name="get_pull_requests",
                        description="Get GitHub pull requests for the repository. Use this to see recent changes or development activity.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search text in PR titles and bodies (optional)"
                                },
                                "state": {
                                    "type": "string",
                                    "description": "PR state: 'open', 'closed', or 'all'",
                                    "default": "open"
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "Maximum number of PRs to return",
                                    "default": 10
                                }
                            }
                        }
                    ),
                    types.FunctionDeclaration(
                        name="get_repo_stats",
                        description="Get statistics about the loaded repository including file count, chunks, issues, and PRs.",
                        parameters={
                            "type": "object",
                            "properties": {}
                        }
                    )
                ]
            )
        ]
    
    def _execute_tool(self, tool_name, args):
        """Execute a tool call and return results."""
        try:
            if tool_name == "search_repo":
                query = args.get("query")
                top_k = args.get("top_k", 5)
                results = self.gateway.search_repo(query, top_k=top_k)
                return {"results": results, "count": len(results)}
            
            elif tool_name == "open_file":
                path = args.get("path")
                start_line = args.get("start_line")
                end_line = args.get("end_line")
                result = self.gateway.open_file(path, start_line, end_line)
                return result
            
            elif tool_name == "get_issues":
                query = args.get("query")
                state = args.get("state", "open")
                limit = args.get("limit", 10)
                results = self.gateway.get_issues(query=query, state=state, limit=limit)
                return {"issues": results, "count": len(results)}
            
            elif tool_name == "get_pull_requests":
                query = args.get("query")
                state = args.get("state", "open")
                limit = args.get("limit", 10)
                results = self.gateway.get_pull_requests(query=query, state=state, limit=limit)
                return {"pull_requests": results, "count": len(results)}
            
            elif tool_name == "get_repo_stats":
                return self.gateway.stats()
            
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        
        except Exception as e:
            return {"error": str(e)}
    
    def ask(self, question, max_turns=10, verbose=False):
        """
        Ask the agent a question about the repository.
        
        Args:
            question: Natural language question
            max_turns: Maximum conversation turns
            verbose: Print intermediate steps
        
        Returns:
            Agent's answer as a string
        """
        # Build system prompt with repo context
        stats = self.gateway.stats()
        system_prompt = f"""You are a helpful repository assistant. You have access to tools to search code, read files, and query GitHub issues/PRs.

                            Repository Context:
                            - Path: {stats.get('repo_path', 'N/A')}
                            - Files: {stats.get('total_files', 0)}
                            - Code chunks: {stats.get('total_chunks', 0)}
                            - Issues: {stats.get('total_issues', 0)}
                            - Pull Requests: {stats.get('total_prs', 0)}

                            When answering questions:
                            1. Use tools to gather evidence from the repository
                            2. Cite specific files and line numbers when referencing code
                            3. Be concise but thorough
                            4. If you can't find something, say so clearly

                            Answer the user's question using the available tools."""

        # Initialize conversation
        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text=system_prompt + "\n\nQuestion: " + question)]
            )
        ]
        
        turn = 0
        
        while turn < max_turns:
            turn += 1
            
            if verbose:
                print(f"\n--- Turn {turn} ---")
            
            # Call Gemini
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=self.tools,
                    temperature=0.2
                )
            )
            
            # Sleep to avoid rate limits
            time.sleep(2)
            
            # Check if there are function calls
            if not response.candidates[0].content.parts:
                break
            
            # Collect all function calls and text responses
            function_calls = []
            text_response = None
            
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    function_calls.append(part.function_call)
                elif part.text:
                    text_response = part.text
            
            # If we have function calls, execute them all and respond
            if function_calls:
                if verbose:
                    print(f"Executing {len(function_calls)} tool call(s)")
                
                # Add the assistant's response with function calls
                contents.append(response.candidates[0].content)
                
                # Execute all function calls and collect responses
                function_response_parts = []
                for func_call in function_calls:
                    if verbose:
                        print(f"  Tool: {func_call.name}")
                        print(f"    Args: {dict(func_call.args)}")
                    
                    # Execute the tool
                    result = self._execute_tool(func_call.name, dict(func_call.args))
                    
                    if verbose:
                        print(f"    Result: {str(result)[:200]}...")
                    
                    # Add function response part
                    function_response_parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=func_call.name,
                                response=result
                            )
                        )
                    )
                
                # Add all function responses in a single content message
                contents.append(
                    types.Content(
                        role="user",
                        parts=function_response_parts
                    )
                )
            
            # If no function calls, we have the final answer
            elif text_response:
                if verbose:
                    print(f"\nFinal answer received")
                return text_response
        
        # Max turns reached
        if verbose:
            print(f"\nMax turns ({max_turns}) reached")
        
        # Try to extract any text response
        for part in response.candidates[0].content.parts:
            if part.text:
                return part.text
        
        return "I couldn't find a complete answer within the allowed conversation turns."
    
    def chat(self, verbose=False):
        """Interactive chat loop."""
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


# Example usage
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python repo_agent.py <repo_path> [github_owner] [github_repo]")
        sys.exit(1)
    
    repo = sys.argv[1]
    owner = sys.argv[2] if len(sys.argv) > 2 else None
    repo_name = sys.argv[3] if len(sys.argv) > 3 else None
    
    print("Loading repository...")
    gateway = ToolGateway(repo, owner, repo_name)
    
    print("Initializing agent...")
    agent = RepoAgent(gateway)
    
    # Interactive chat
    agent.chat(verbose=True)

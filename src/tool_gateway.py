import os
from pathlib import Path
from src.repo_ingestion import RepoIngestion


class ToolGateway:
    """
    Simple tool gateway that provides search, file access, and GitHub queries.
    """
    
    def __init__(self, repo_path=None, github_owner=None, github_repo=None, github_token=None):
        """
        Initialize the tool gateway.
        
        Args:
            repo_path: Path to local repository (optional, can set later)
            github_owner: GitHub repo owner (e.g., 'hack4impact-calpoly')
            github_repo: GitHub repo name (e.g., 'prfc-connect')
            github_token: GitHub token (reads from GITHUB_TOKEN env if not provided)
        """
        self.repo_path = Path(repo_path) if repo_path else None
        self.github_owner = github_owner
        self.github_repo = github_repo
        self.github_token = github_token or os.environ.get('GITHUB_TOKEN')
        self.ingestion = None
        
        # Auto-ingest if repo path provided
        if self.repo_path:
            self.load_repository(self.repo_path)
    
    def load_repository(self, repo_path):
        """Load and ingest a repository."""
        self.repo_path = Path(repo_path)
        self.ingestion = RepoIngestion()
        self.ingestion.ingest_repository(str(self.repo_path))
        
        # Also load issues/PRs if GitHub info available
        if self.github_owner and self.github_repo:
            try:
                self.ingestion.ingest_github_issues(
                    self.github_owner, 
                    self.github_repo,
                    github_token=self.github_token
                )
                self.ingestion.ingest_github_prs(
                    self.github_owner, 
                    self.github_repo,
                    github_token=self.github_token
                )
            except Exception as e:
                print(f"Warning: Could not load GitHub data: {e}")
    
    def search_repo(self, query, top_k=10, filters=None):
        """
        Search repository for relevant code chunks.
        
        Args:
            query: Search query string
            top_k: Maximum number of results
            filters: Optional dict with 'path_prefix' or 'extensions' filters
        
        Returns:
            List of evidence references with file paths, line ranges, and snippets
        """
        if self.ingestion is None:
            return []
        
        # Get all matching chunks
        results = self.ingestion.search_chunks(query, max_results=top_k * 2)
        
        # Apply filters if provided
        if filters:
            path_prefix = filters.get('path_prefix')
            extensions = filters.get('extensions', [])
            
            if path_prefix:
                results = [r for r in results if r['file_path'].startswith(path_prefix)]
            
            if extensions:
                results = [r for r in results 
                          if any(r['file_path'].endswith(ext) for ext in extensions)]
        
        # Format as evidence references
        evidence = []
        for r in results[:top_k]:
            evidence.append({
                'source': 'file',
                'chunk_id': r['chunk_id'],
                'file_path': r['file_path'],
                'start_line': r['start_line'],
                'end_line': r['end_line'],
                'snippet': r['text'][:200] + '...' if len(r['text']) > 200 else r['text'],
                'full_text': r['text']
            })
        
        return evidence
    
    def open_file(self, path, start_line=None, end_line=None):
        """
        Open and read a file or file range.
        
        Args:
            path: File path (relative to repo root)
            start_line: Optional start line (1-indexed)
            end_line: Optional end line (1-indexed, inclusive)
        
        Returns:
            Dict with file content and metadata
        """
        if self.repo_path is None:
            raise ValueError("No repository loaded")
        
        full_path = self.repo_path / path
        
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        # Read file
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            raise RuntimeError(f"Could not read file: {e}")
        
        # Extract range if specified
        if start_line is not None and end_line is not None:
            # Convert to 0-indexed
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)
            selected_lines = lines[start_idx:end_idx]
            text = ''.join(selected_lines)
            actual_start = start_line
            actual_end = end_idx
        else:
            text = ''.join(lines)
            actual_start = 1
            actual_end = len(lines)
        
        return {
            'file_path': path,
            'start_line': actual_start,
            'end_line': actual_end,
            'text': text,
            'total_lines': len(lines),
            'extension': full_path.suffix
        }
    
    def get_issues(self, query=None, state='open', label=None, limit=50):
        """
        Get GitHub issues.
        
        Args:
            query: Optional text search in title/body
            state: Issue state ('open', 'closed', 'all')
            label: Optional label filter
            limit: Maximum results
        
        Returns:
            List of issue records
        """
        if self.ingestion is None or not hasattr(self.ingestion.context, 'get'):
            return []
        
        issues = self.ingestion.get_issues(state=state if state != 'all' else None, label=label)
        
        # Text search if query provided
        if query:
            query_lower = query.lower()
            issues = [i for i in issues 
                     if query_lower in i['title'].lower() or query_lower in i.get('body', '').lower()]
        
        return issues[:limit]
    
    def get_pull_requests(self, query=None, state='open', label=None, limit=50):
        """
        Get GitHub pull requests.
        
        Args:
            query: Optional text search in title/body
            state: PR state ('open', 'closed', 'all')
            label: Optional label filter
            limit: Maximum results
        
        Returns:
            List of PR records
        """
        if self.ingestion is None or not hasattr(self.ingestion.context, 'get'):
            return []
        
        prs = self.ingestion.get_prs(state=state if state != 'all' else None, label=label)
        
        # Text search if query provided
        if query:
            query_lower = query.lower()
            prs = [p for p in prs 
                  if query_lower in p['title'].lower() or query_lower in p.get('body', '').lower()]
        
        return prs[:limit]
    
    def search_issues_and_prs(self, query, limit=20):
        """
        Combined search across issues and PRs.
        
        Args:
            query: Search query
            limit: Maximum total results
        
        Returns:
            Combined list of issues and PRs
        """
        issues = self.get_issues(query=query, state='all', limit=limit)
        prs = self.get_pull_requests(query=query, state='all', limit=limit)
        
        # Combine and sort by updated date
        combined = []
        for issue in issues:
            combined.append({'type': 'issue', **issue})
        for pr in prs:
            combined.append({'type': 'pr', **pr})
        
        # Sort by updated_at descending
        combined.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
        
        return combined[:limit]
    
    def list_files(self, path_prefix=None, extensions=None):
        """
        List all files in the indexed repository, optionally filtered.

        Args:
            path_prefix: Only return files under this path prefix (e.g. 'src/')
            extensions: List of extensions to filter by (e.g. ['.ts', '.tsx'])

        Returns:
            List of file path strings
        """
        if self.ingestion is None or self.ingestion.context is None:
            return []

        files = [f['path'] for f in self.ingestion.context.get('files', [])]

        if path_prefix:
            files = [f for f in files if f.startswith(path_prefix)]

        if extensions:
            files = [f for f in files if any(f.endswith(ext) for ext in extensions)]

        return sorted(files)

    def get_file_by_chunk_id(self, chunk_id):
        """Get the full file content for a given chunk ID."""
        if self.ingestion is None:
            return None
        
        chunk = self.ingestion.get_chunk_by_id(chunk_id)
        if not chunk:
            return None
        
        return self.open_file(chunk['file_path'])
    
    def stats(self):
        """Get statistics about loaded context."""
        if self.ingestion is None or self.ingestion.context is None:
            return {
                'loaded': False,
                'repo_path': str(self.repo_path) if self.repo_path else None
            }
        
        ctx = self.ingestion.context
        return {
            'loaded': True,
            'repo_path': str(self.repo_path),
            'total_files': ctx.get('total_files', 0),
            'total_chunks': ctx.get('total_chunks', 0),
            'total_issues': ctx.get('total_issues', 0),
            'total_prs': ctx.get('total_prs', 0),
            'github_owner': self.github_owner,
            'github_repo': self.github_repo
        }
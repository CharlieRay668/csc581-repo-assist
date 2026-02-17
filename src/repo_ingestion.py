import os
import json
import hashlib
import requests
from pathlib import Path


# Default configuration
DEFAULT_CHUNK_MAX_LINES = 40
DEFAULT_CHUNK_MIN_LINES = 10
DEFAULT_CHUNK_OVERLAP_LINES = 5

DEFAULT_IGNORE_DIRS = {
    'node_modules', '.git', '.venv', 'venv', '__pycache__', 
    'build', 'dist', '.next', 'coverage', '.pytest_cache'
}

DEFAULT_IGNORE_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.pdf',
    '.zip', '.tar', '.gz', '.exe', '.dll', '.so', '.dylib',
    '.lock', '.pyc', '.pyo', '.class', '.jar'
}


class RepoIngestion:
    """
    Repository ingestion class that walks a repo and generates JSON context.
    """
    
    def __init__(
        self,
        chunk_max_lines=DEFAULT_CHUNK_MAX_LINES,
        chunk_min_lines=DEFAULT_CHUNK_MIN_LINES,
        chunk_overlap_lines=DEFAULT_CHUNK_OVERLAP_LINES,
        ignore_dirs=None,
        ignore_extensions=None
    ):
        """
        Initialize the ingestion configuration.
        
        Args:
            chunk_max_lines: Maximum lines per chunk, I don't know if we really want to do chunks tbh
            chunk_min_lines: Minimum lines for chunking (smaller files = 1 chunk)
            chunk_overlap_lines: Number of overlapping lines between chunks
            ignore_dirs: Set of directory names to ignore
            ignore_extensions: Set of file extensions to ignore
        """
        self.chunk_max_lines = chunk_max_lines
        self.chunk_min_lines = chunk_min_lines
        self.chunk_overlap_lines = chunk_overlap_lines
        self.ignore_dirs = ignore_dirs if ignore_dirs is not None else DEFAULT_IGNORE_DIRS
        self.ignore_extensions = ignore_extensions if ignore_extensions is not None else DEFAULT_IGNORE_EXTENSIONS
        self.context = None
    
    def ingest_repository(
        self,
        repo_path,
        output_path=None
    ):
        """
        Main entry point: ingest a repository and return structured JSON context.
        
        Args:
            repo_path: Path to the repository to ingest
            output_path: Optional path to save JSON output (if None, only returns dict)
        
        Returns:
            Dict with 'files' and 'chunks' keys containing the repository context
        """
        repo_path = Path(repo_path).resolve()
        if not repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")
        
        files = []
        chunks = []
        chunk_counter = 0
        
        # Walk the repository
        for file_path in self._walk_repo(repo_path):
            try:
                # Create file record
                file_record = self._create_file_record(file_path, repo_path)
                
                # Read and chunk the file
                file_chunks = self._chunk_file(file_path, repo_path, chunk_counter)
                
                # Update file record with chunk IDs
                file_record['chunk_ids'] = [c['chunk_id'] for c in file_chunks]
                
                files.append(file_record)
                chunks.extend(file_chunks)
                chunk_counter += len(file_chunks)
                
            except Exception as e:
                # Skip files that can't be processed
                print(f"Warning: Skipping {file_path}: {e}")
                continue
        
        result = {
            'repository_path': str(repo_path),
            'total_files': len(files),
            'total_chunks': len(chunks),
            'files': files,
            'chunks': chunks
        }
        
        # Store context for later access
        self.context = result
        
        # Save to file if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
        
        return result
    
    def _walk_repo(self, repo_path):
        """Walk repository and return list of file paths to process."""
        files = []
        
        for root, dirs, filenames in os.walk(repo_path):
            # Remove ignored directories from traversal
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]
            
            for filename in filenames:
                file_path = Path(root) / filename
                
                # Skip ignored extensions
                if file_path.suffix in self.ignore_extensions:
                    continue
                
                # Skip hidden files
                if filename.startswith('.'):
                    continue
                
                files.append(file_path)
        
        return files
    
    def _create_file_record(self, file_path, repo_root):
        """Create metadata record for a file."""
        relative_path = file_path.relative_to(repo_root)
        
        # Read file content for hash
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception:
            # If can't read as text, read as binary for hash
            with open(file_path, 'rb') as f:
                content = f.read()
        
        content_hash = hashlib.sha256(
            content.encode('utf-8') if isinstance(content, str) else content
        ).hexdigest()[:16]
        
        # Count lines if text
        num_lines = 0
        if isinstance(content, str):
            num_lines = content.count('\n') + 1
        
        return {
            'path': str(relative_path),
            'full_path': str(file_path),
            'extension': file_path.suffix,
            'size_bytes': file_path.stat().st_size,
            'content_hash': content_hash,
            'num_lines': num_lines,
            'chunk_ids': []  # Will be filled in later
        }
    
    def _chunk_file(
        self,
        file_path,
        repo_root,
        chunk_id_start
    ):
        """
        Chunk a file into line-based segments.
        
        Returns list of chunk records with line ranges and text.
        """
        relative_path = file_path.relative_to(repo_root)
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            # Can't read as text, skip chunking
            return []
        
        num_lines = len(lines)
        
        # Small file: one chunk
        if num_lines <= self.chunk_min_lines:
            chunk_text = ''.join(lines)
            return [{
                'chunk_id': f"chunk_{chunk_id_start:05d}",
                'file_path': str(relative_path),
                'start_line': 1,
                'end_line': num_lines,
                'text': chunk_text,
                'num_lines': num_lines
            }]
        
        # Large file: chunk with overlap
        chunks = []
        chunk_num = chunk_id_start
        start_line = 0
        
        while start_line < num_lines:
            end_line = min(start_line + self.chunk_max_lines, num_lines)
            chunk_lines = lines[start_line:end_line]
            chunk_text = ''.join(chunk_lines)
            
            chunks.append({
                'chunk_id': f"chunk_{chunk_num:05d}",
                'file_path': str(relative_path),
                'start_line': start_line + 1,  # 1-indexed for display
                'end_line': end_line,
                'text': chunk_text,
                'num_lines': len(chunk_lines)
            })
            
            chunk_num += 1
            
            # Move forward with overlap
            if end_line >= num_lines:
                break
            start_line = end_line - self.chunk_overlap_lines
        
        return chunks
    
    def get_chunk_by_id(self, chunk_id):
        """Helper: retrieve a specific chunk by ID from ingested context."""
        if self.context is None:
            return None
        for chunk in self.context.get('chunks', []):
            if chunk['chunk_id'] == chunk_id:
                return chunk
        return None
    
    def get_file_by_path(self, file_path):
        """Helper: retrieve a specific file by path from ingested context."""
        if self.context is None:
            return None
        for file_record in self.context.get('files', []):
            if file_record['path'] == file_path:
                return file_record
        return None
    
    def search_chunks(self, query, max_results=10):
        """
        Simple text search over chunks.
        Returns chunks containing the query string (case-insensitive).
        """
        if self.context is None:
            return []
        
        query_lower = query.lower()
        results = []
        
        for chunk in self.context.get('chunks', []):
            if query_lower in chunk['text'].lower():
                results.append(chunk)
                if len(results) >= max_results:
                    break
        
        return results
    
    def ingest_github_issues(
        self,
        owner,
        repo,
        state='all',
        max_issues=100,
        github_token=None
    ):
        """
        Fetch and ingest GitHub issues for the repository.
        
        Args:
            owner: Repository owner (e.g., 'hack4impact-calpoly')
            repo: Repository name (e.g., 'prfc-connect')
            state: Issue state filter ('open', 'closed', 'all')
            max_issues: Maximum number of issues to fetch
            github_token: (Optional)
        Returns:
            List of issue records
        """
        if github_token is None:
            github_token = os.environ.get('GITHUB_TOKEN')
        
        headers = {}
        if github_token:
            headers['Authorization'] = f'token {github_token}'
        
        issues = []
        page = 1
        per_page = min(100, max_issues)
        
        while len(issues) < max_issues:
            url = f'https://api.github.com/repos/{owner}/{repo}/issues'
            params = {
                'state': state,
                'page': page,
                'per_page': per_page,
                'sort': 'updated',
                'direction': 'desc'
            }
            
            try:
                response = requests.get(url, headers=headers, params=params, timeout=10)
                response.raise_for_status()
                batch = response.json()
                
                if not batch:
                    break
                
                # Filter out pull requests (they show up in issues endpoint)
                for item in batch:
                    if 'pull_request' not in item:
                        issues.append({
                            'id': item['id'],
                            'number': item['number'],
                            'title': item['title'],
                            'body': item.get('body', ''),
                            'state': item['state'],
                            'labels': [label['name'] for label in item.get('labels', [])],
                            'created_at': item['created_at'],
                            'updated_at': item['updated_at'],
                            'url': item['html_url'],
                            'user': item['user']['login'] if item.get('user') else None
                        })
                        
                        if len(issues) >= max_issues:
                            break
                
                page += 1
                
            except Exception as e:
                print(f"Warning: Failed to fetch issues: {e}")
                break
        
        # Store in context
        if self.context is None:
            self.context = {}
        self.context['issues'] = issues
        self.context['total_issues'] = len(issues)
        
        return issues
    
    def ingest_github_prs(
        self,
        owner,
        repo,
        state='all',
        max_prs=100,
        github_token=None
    ):
        """
        Fetch and ingest GitHub pull requests for the repository.
        
        Args:
            owner: Repository owner (e.g., 'hack4impact-calpoly')
            repo: Repository name (e.g., 'prfc-connect')
            state: PR state filter ('open', 'closed', 'all')
            max_prs: Maximum number of PRs to fetch
            github_token: (Optional)
        Returns:
            List of PR records
        """
        if github_token is None:
            github_token = os.environ.get('GITHUB_TOKEN')
        
        headers = {}
        if github_token:
            headers['Authorization'] = f'token {github_token}'
        
        prs = []
        page = 1
        per_page = min(100, max_prs)
        
        while len(prs) < max_prs:
            url = f'https://api.github.com/repos/{owner}/{repo}/pulls'
            params = {
                'state': state,
                'page': page,
                'per_page': per_page,
                'sort': 'updated',
                'direction': 'desc'
            }
            
            try:
                response = requests.get(url, headers=headers, params=params, timeout=10)
                response.raise_for_status()
                batch = response.json()
                
                if not batch:
                    break
                
                for item in batch:
                    prs.append({
                        'id': item['id'],
                        'number': item['number'],
                        'title': item['title'],
                        'body': item.get('body', ''),
                        'state': item['state'],
                        'labels': [label['name'] for label in item.get('labels', [])],
                        'created_at': item['created_at'],
                        'updated_at': item['updated_at'],
                        'merged_at': item.get('merged_at'),
                        'url': item['html_url'],
                        'user': item['user']['login'] if item.get('user') else None,
                        'base_branch': item['base']['ref'],
                        'head_branch': item['head']['ref']
                    })
                    
                    if len(prs) >= max_prs:
                        break
                
                page += 1
                
            except Exception as e:
                print(f"Warning: Failed to fetch PRs: {e}")
                break
        
        # Store in context
        if self.context is None:
            self.context = {}
        self.context['pull_requests'] = prs
        self.context['total_prs'] = len(prs)
        
        return prs
    
    def get_issues(self, state=None, label=None):
        """Get filtered issues from context."""
        if self.context is None or 'issues' not in self.context:
            return []
        
        issues = self.context['issues']
        
        if state:
            issues = [i for i in issues if i['state'] == state]
        
        if label:
            issues = [i for i in issues if label in i['labels']]
        
        return issues
    
    def get_prs(self, state=None, label=None):
        """Get filtered PRs from context."""
        if self.context is None or 'pull_requests' not in self.context:
            return []
        
        prs = self.context['pull_requests']
        
        if state:
            prs = [p for p in prs if p['state'] == state]
        
        if label:
            prs = [p for p in prs if label in p['labels']]
        
        return prs

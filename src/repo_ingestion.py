"""
Repo Ingestion - Simple implementation
Walks a repository and generates JSON context (files + chunks).
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional


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
        chunk_max_lines: int = DEFAULT_CHUNK_MAX_LINES,
        chunk_min_lines: int = DEFAULT_CHUNK_MIN_LINES,
        chunk_overlap_lines: int = DEFAULT_CHUNK_OVERLAP_LINES,
        ignore_dirs: Optional[set] = None,
        ignore_extensions: Optional[set] = None
    ):
        """
        Initialize the ingestion configuration.
        
        Args:
            chunk_max_lines: Maximum lines per chunk
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
        repo_path: str,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
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
    
    def _walk_repo(self, repo_path: Path) -> List[Path]:
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
    
    def _create_file_record(self, file_path: Path, repo_root: Path) -> Dict[str, Any]:
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
        file_path: Path,
        repo_root: Path,
        chunk_id_start: int
    ) -> List[Dict[str, Any]]:
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
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Helper: retrieve a specific chunk by ID from ingested context."""
        if self.context is None:
            return None
        for chunk in self.context.get('chunks', []):
            if chunk['chunk_id'] == chunk_id:
                return chunk
        return None
    
    def get_file_by_path(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Helper: retrieve a specific file by path from ingested context."""
        if self.context is None:
            return None
        for file_record in self.context.get('files', []):
            if file_record['path'] == file_path:
                return file_record
        return None
    
    def search_chunks(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
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

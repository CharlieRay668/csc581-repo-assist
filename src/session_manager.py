import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path


SESSIONS_DIR = Path.home() / ".repo_assist" / "sessions"


class SessionManager:

    def __init__(self, sessions_dir=None):
        self.sessions_dir = Path(sessions_dir) if sessions_dir else SESSIONS_DIR
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._state = None

    def create_session(self, repo_path, github_owner=None, github_repo=None,
                       mode="explain", scope="include-pr", verbose=False):
        session_id = uuid.uuid4().hex[:12]
        now = self._now()
        self._state = {
            "session_id": session_id,
            "repo_identifier": str(Path(repo_path).resolve()),
            "github_owner": github_owner,
            "github_repo": github_repo,
            "recent_queries": [],
            "selected_evidence_refs": [],
            "user_settings": {
                "mode": mode,
                "scope": scope,
                "verbose": verbose,
            },
            "created_at": now,
            "updated_at": now,
        }
        self.save_session()
        return session_id

    def load_session(self, session_id):
        path = self._session_path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")
        with open(path, "r", encoding="utf-8") as f:
            self._state = json.load(f)
        return self._state

    def save_session(self):
        if self._state is None:
            raise RuntimeError("No active session to save")
        self._state["updated_at"] = self._now()
        path = self._session_path(self._state["session_id"])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2)

    def session_exists(self, session_id):
        return self._session_path(session_id).exists()

    def add_query(self, query, response_summary=""):
        self._require_state()
        entry = {
            "query": query,
            "summary": response_summary[:300],
            "timestamp": self._now(),
        }
        self._state["recent_queries"].append(entry)
        self._state["recent_queries"] = self._state["recent_queries"][-10:]
        self.save_session()

    def add_evidence(self, evidence_refs):
        self._require_state()
        existing = {
            (e["file_path"], e.get("start_line"), e.get("end_line"))
            for e in self._state["selected_evidence_refs"]
        }
        for ref in evidence_refs:
            key = (ref.get("file_path"), ref.get("start_line"), ref.get("end_line"))
            if key not in existing:
                self._state["selected_evidence_refs"].append(ref)
                existing.add(key)
        self._state["selected_evidence_refs"] = self._state["selected_evidence_refs"][-50:]
        self.save_session()

    def update_settings(self, **kwargs):
        self._require_state()
        for k, v in kwargs.items():
            if k in self._state["user_settings"]:
                self._state["user_settings"][k] = v
        self.save_session()

    def get_llm_context(self):
        self._require_state()
        return {
            "session_id": self._state["session_id"],
            "repo_identifier": self._state["repo_identifier"],
            "github_owner": self._state.get("github_owner"),
            "github_repo": self._state.get("github_repo"),
            "recent_queries": self._state["recent_queries"][-5:],
            "selected_evidence_refs": self._state["selected_evidence_refs"][-10:],
            "user_settings": self._state["user_settings"],
        }

    @property
    def session_id(self):
        self._require_state()
        return self._state["session_id"]

    @property
    def state(self):
        return self._state

    @property
    def settings(self):
        self._require_state()
        return self._state["user_settings"]

    def _session_path(self, session_id):
        return self.sessions_dir / f"{session_id}.json"

    def _require_state(self):
        if self._state is None:
            raise RuntimeError("No active session. Call create_session() or load_session() first.")

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()

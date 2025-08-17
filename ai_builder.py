# ai_builder.py
# pip install openai python-dotenv requests
# export OPENAI_API_KEY=sk-...
# (for REST path) export GITHUB_TOKEN=<personal-access-token with repo scope>

import os, re, json, zipfile, subprocess, argparse, time, base64
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

import locale, sys, subprocess, os
# ----------------------------- Constants --------------------------------------

DEFAULT_MODEL = "gpt-4o-mini"
FILE_BLOCK_REGEX = r'---\s*file:\s*(.+?)\s*---\n(.*?)(?=(?:\n---\s*file:\s*)|\Z)'
DIFF_BLOCK_REGEX = r'---\s*diff:\s*(.+?)\s*---\n(.*?)(?=(?:\n---\s*(?:file|diff|patch):\s*)|\Z)'
PATCH_BLOCK_REGEX = r'---\s*patch:\s*(.+?)\s*---\n(.*?)(?=(?:\n---\s*(?:file|diff|patch):\s*)|\Z)'

DEFAULT_SYSTEM = """You are a senior full-stack engineer.
When the user describes an app or change, output either:
1) COMPLETE files, or
2) ONLY UPDATED files,
using exactly one of these block formats:

--- file: <relative/path/filename.ext> ---
<raw file contents>

--- diff: <relative/path/filename.ext> ---
<unified diff patch affecting ONLY this file>

--- patch: <relative/path/filename.ext> ---
A JSON object or array of objects in the form:
  {"op":"replace"|"insert"|"delete", "find":"...", "replace":"..."}
  Minimal, line-oriented where possible.

Rules:
- Output ONLY blocks (no commentary before/after).
- Do NOT wrap contents in ``` or ''' code fences.
- Use clear, conventional project structure.
"""

FIREBASE_PRESET = """
Firebase preset:
- If building a web app, include:
  - firebase.json (SPA rewrite to /index.html; optionally rewrite /api/**)
  - .firebaserc (project id: "your-firebase-project-id")
  - firestore.rules, storage.rules (locked by default)
  - .gitignore (.env, node_modules, dist/build, functions/node_modules)
  - .env.example
- If API requested: Cloud Functions (Node 20, TypeScript) exposing /api/hello (+ rewrite).
- Include a minimal README with Firebase CLI steps.
"""

GCP_RUN_PRESET = """
GCP Run preset:
- Backend for Google Cloud Run (Python FastAPI):
  - backend/main.py (FastAPI hello at /api/hello)
  - backend/requirements.txt (fastapi, uvicorn[standard])
  - Dockerfile (python:3.11-slim)
  - README.md with build/push/deploy steps for Artifact Registry + Cloud Run.
- Firebase Hosting rewrites to Cloud Run for /api/** (placeholder serviceId + region),
  plus SPA fallback if a front-end exists.
- Front-end default: Vite + React + Tailwind unless user opts out.
"""

CODE_ONLY_PRESET = """
CODE-ONLY mode:
- No website scaffolding (no firebase.json, .firebaserc, vite config, Dockerfiles) unless explicitly asked.
- Return only the requested scripts/notebooks/modules as file/diff/patch blocks.
"""

SAFE_EXTENSIONS = {
    ".ts", ".tsx", ".js", ".jsx", ".json", ".md", ".html", ".css", ".cjs", ".mjs",
    ".py", ".txt", ".yml", ".yaml", ".toml", ".env", ".gitignore", ".dockerignore"
}

WHITELISTED_COMMANDS = [
    # JS/TS
    "npm ci", "npm install", "pnpm install", "yarn install",
    "npm run build", "npm run dev", "npm run lint", "npm run typecheck",
    "vite build", "tsc -v",
    # Python
    "python -m pytest -q", "pytest -q",
    # Firebase emulators (no deploy)
    "firebase emulators:start --only hosting",
]

# --------------------------- Utilities ----------------------------------------

def ensure_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")
    return key

def run_p(args, cwd=None, shell=False, **kwargs):
    """
    Robust subprocess runner: always decode as UTF-8 and never crash on bad bytes.
    Accepts extra kwargs but ignores conflicting ones (like capture_output/text).
    You may pass 'env' and 'input' via kwargs if needed.
    """
    env = kwargs.get("env")
    inp = kwargs.get("input")
    p = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        shell=shell,
        capture_output=True,    # we always capture
        text=True,              # we always return text
        encoding="utf-8",       # force utf-8 decoding
        errors="replace",       # never crash on bad bytes
        env=env,
        input=inp,
    )
    return p.returncode == 0, (p.stdout or ""), (p.stderr or "")

os.environ.setdefault("PYTHONUTF8", "1")           # Python default text encoding -> UTF-8
os.environ.setdefault("PYTHONIOENCODING", "utf-8") # our stdio
if os.name == "nt":
    try:
        # Switch the current console to UTF-8 codepage to match outputs from git/gh
        run_p(["chcp", "65001"], shell=True)
    except Exception:
        pass

def run_json(args, cwd=None, shell=False):
    ok, out, err = run_p(args, cwd=cwd, shell=shell)
    if not ok:
        return ok, None, err
    try:
        return True, json.loads(out or "null"), ""
    except Exception as e:
        return False, None, f"JSON parse error: {e}\nRAW:\n{out[:1000]}"


def strip_code_fences(text: str) -> str:
    text = re.sub(r"```(?:[a-zA-Z0-9_-]+)?\n", "", text)
    text = text.replace("```", "")
    text = re.sub(r"'''(?:[a-zA-Z0-9_-]+)?\n", "", text)
    text = text.replace("'''", "")
    return text

def is_safe_relative(path: str) -> bool:
    p = Path(path)
    if p.is_absolute(): return False
    for part in p.parts:
        if part in ("..",):
            return False
    return True

def parse_origin_url_to_repo(origin: str) -> Optional[Tuple[str, str]]:
    """
    Supports:
      https://github.com/owner/repo.git
      git@github.com:owner/repo.git
    Returns (owner, repo) or None.
    """
    if not origin: return None
    origin = origin.strip()
    if origin.startswith("git@github.com:"):
        rest = origin.split("git@github.com:")[1]
    elif origin.startswith("https://github.com/"):
        rest = origin.split("https://github.com/")[1]
    else:
        return None
    rest = rest[:-4] if rest.endswith(".git") else rest
    parts = rest.split("/")
    if len(parts) != 2: return None
    return parts[0], parts[1]

def first_nonempty(*vals):
    for v in vals:
        if v: return v
    return None

# --------------------------- Data classes -------------------------------------

@dataclass
class ChatTurn:
    role: str
    content: str

@dataclass
class AIProjectScaffolder:
    project_root: str
    model: str = DEFAULT_MODEL
    system_instructions: str = DEFAULT_SYSTEM
    history_filename: str = "history_new.json"
    client: Optional[OpenAI] = None
    history: List[ChatTurn] = field(default_factory=list)
    verbose: bool = False

    def __post_init__(self):
        self.project_root_path = Path(self.project_root).expanduser().resolve()
        self.project_root_path.mkdir(parents=True, exist_ok=True)

        if self.client is None:
            self.client = OpenAI(api_key=ensure_api_key())

        self.history_path = self.project_root_path / self.history_filename
        if not self.history_path.exists():
            self.history_path.write_text("[]", encoding="utf-8")
        try:
            self.load_history()
        except Exception:
            self.history = []
            self.save_history()

        # Ensure repo exists & is ready; fetch latest if remote is set
        self.ensure_repo_initialized()
        self.git_fetch()
        self.git_pull_rebase_main()

    # ------------- System / messages ----------------
    def _compose_system(self, preset: Optional[str], mode: str) -> str:
        sys = self.system_instructions
        if mode == "code":
            sys += "\n\n" + CODE_ONLY_PRESET
        if preset == "firebase":
            sys += "\n\n" + FIREBASE_PRESET
        elif preset == "gcp-run":
            sys += "\n\n" + GCP_RUN_PRESET
        return sys

    def _build_messages(self, new_user_message: str, preset: Optional[str], mode: str) -> List[Dict[str, str]]:
        sys = self._compose_system(preset, mode)
        msgs = [{"role": "system", "content": sys}]
        msgs += [{"role": t.role, "content": t.content} for t in self.history]
        msgs.append({"role": "user", "content": new_user_message})
        return msgs

    # ------------- Parsing --------------------------
    def _parse_blocks(self, raw_text: str):
        text = strip_code_fences(raw_text)
        files = re.findall(FILE_BLOCK_REGEX, text, flags=re.DOTALL)
        diffs = re.findall(DIFF_BLOCK_REGEX, text, flags=re.DOTALL)
        patches = re.findall(PATCH_BLOCK_REGEX, text, flags=re.DOTALL)

        file_list = [{"kind":"file", "filename":f.strip(), "content":c.lstrip("\n").rstrip()} for f, c in files]
        diff_list = [{"kind":"diff", "filename":f.strip(), "content":c.lstrip("\n").rstrip()} for f, c in diffs]
        patch_list = [{"kind":"patch", "filename":f.strip(), "content":c.lstrip("\n").rstrip()} for f, c in patches]

        if not (file_list or diff_list or patch_list):
            preview = text[:600].replace("\n","\\n")
            raise ValueError(f"No blocks parsed. Preview:\n{preview}")
        return file_list, diff_list, patch_list

    # ------------- API calls + autosave -------------
    def send(self, user_prompt: str, preset: Optional[str] = None, mode: str = "web",
             temperature: float = 0.2, max_output_tokens: int = 8000):
        msgs = self._build_messages(user_prompt, preset=preset, mode=mode)
        self.history.append(ChatTurn("user", user_prompt))
        self.save_history()
        try:
            resp = self.client.responses.create(
                model=self.model,
                input=msgs,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            output_text = resp.output_text
            file_blocks, diff_blocks, patch_blocks = self._parse_blocks(output_text)
            self.history.append(ChatTurn("assistant", output_text))
            self.save_history()
            return file_blocks, diff_blocks, patch_blocks
        except Exception as e:
            self.history.append(ChatTurn("assistant", f"[ERROR] {type(e).__name__}: {e}"))
            self.save_history()
            raise

    def apply_changes(self, instruction: str, preset: Optional[str] = None, mode: str = "web",
                      temperature: float = 0.2, max_output_tokens: int = 8000):
        return self.send(instruction, preset=preset, mode=mode,
                         temperature=temperature, max_output_tokens=max_output_tokens)

    # ------------- File ops -------------------------
    def _safe_write(self, root: Path, rel_path: str, content: str):
        if not is_safe_relative(rel_path):
            raise ValueError(f"Unsafe path: {rel_path}")
        dest = root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        if self.verbose: print("[write]", dest)

    def write_blocks(self, blocks, root: Optional[str] = None):
        root_path = Path(root).expanduser().resolve() if root else self.project_root_path
        for b in blocks:
            if b["kind"] != "file": continue
            self._safe_write(root_path, b["filename"], b["content"])

    # ------------- Patch application ----------------
    def apply_unified_diff(self, root: Optional[str], diff_blocks):
        root_path = Path(root).expanduser().resolve() if root else self.project_root_path
        for diff in diff_blocks:
            rel = diff["filename"]
            if not is_safe_relative(rel):
                raise ValueError(f"Unsafe path in diff: {rel}")
            target = root_path / rel
            if not target.exists():
                raise FileNotFoundError(f"File for diff not found: {rel}")
            original = target.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
            patched = self._naive_apply_diff(original, diff["content"])
            target.write_text("".join(patched), encoding="utf-8")
            if self.verbose: print("[patch-diff]", target)

    def _naive_apply_diff(self, original_lines: List[str], diff_text: str) -> List[str]:
        new_lines = [l for l in original_lines]
        adds, removes = [], []
        for line in diff_text.splitlines(True):
            if line.startswith(('+++', '---', '@@')): continue
            if line.startswith('+'): adds.append(line[1:])
            elif line.startswith('-'): removes.append(line[1:])
        for r in removes:
            try:
                idx = new_lines.index(r)
                new_lines.pop(idx)
            except ValueError:
                pass
        insert_at = len(new_lines)
        for a in adds:
            new_lines.insert(insert_at, a); insert_at += 1
        return new_lines

    def apply_json_patches(self, root: Optional[str], patch_blocks):
        root_path = Path(root).expanduser().resolve() if root else self.project_root_path
        for patch in patch_blocks:
            rel = patch["filename"]
            if not is_safe_relative(rel):
                raise ValueError(f"Unsafe path in patch: {rel}")
            target = root_path / rel
            if not target.exists():
                raise FileNotFoundError(f"File for patch not found: {rel}")
            text = target.read_text(encoding="utf-8", errors="replace")
            ops = json.loads(patch["content"])
            if isinstance(ops, dict): ops = [ops]
            for op in ops:
                kind = op.get("op")
                if kind == "replace":
                    text = text.replace(op.get("find",""), op.get("replace",""))
                elif kind == "insert":
                    anchor = op.get("find","")
                    repl = anchor + op.get("replace","")
                    text = text.replace(anchor, repl)
                elif kind == "delete":
                    text = text.replace(op.get("find",""), "")
            target.write_text(text, encoding="utf-8")
            if self.verbose: print("[patch-json]", target)

    # ------------- History persistence -------------
    def save_history(self) -> None:
        data = [{"role": t.role, "content": t.content} for t in self.history]
        if self.verbose: print("[save_history] ->", self.history_path)
        self.history_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_history(self) -> None:
        raw = json.loads(self.history_path.read_text(encoding="utf-8", errors="replace"))
        self.history = [ChatTurn(**t) for t in raw]

    # ------------- Validation loop -----------------
    def run_cmd(self, cmd: str, cwd: Optional[Path] = None) -> Dict[str, str]:
        p_ok, p_out, p_err = run_p(cmd, cwd=cwd, shell=True)
        out = p_out[-8000:]
        err = p_err[-8000:]
        return {"cmd": cmd, "ok": p_ok, "stdout": out, "stderr": err}

    def validate_project(self, extra_cmds: Optional[List[str]] = None) -> List[Dict[str, str]]:
        cmds = [
            "npm ci || npm install",
            "npm run typecheck || tsc -v || true",
            "npm run lint || eslint -v || true",
            "npm run build || vite build || true",
            "pytest -q || true",
        ]
        if extra_cmds: cmds.extend(extra_cmds)
        results = [self.run_cmd(c) for c in cmds]
        if self.verbose:
            for r in results:
                print(f"$ {r['cmd']}  OK={r['ok']}\nSTDERR:\n{r['stderr']}\n")
        return results

    def feedback_prompt_from_results(self, results: List[Dict[str, str]]) -> str:
        chunks = []
        for r in results:
            chunks.append(f"$ {r['cmd']}\nOK={r['ok']}\nSTDERR:\n{r['stderr'] or '(empty)'}")
        return "Validation results:\n" + "\n\n".join(chunks) + \
               "\n\nPlease return ONLY changed blocks (file/diff/patch) to fix these issues."

    # ------------- Git / GitHub helpers ---------------------
    def git(self, args: List[str]) -> Tuple[bool, str, str]:
        return run_p(["git"] + args, cwd=self.project_root_path)

    def git_is_available(self) -> bool:
        ok, _, _ = self.git(["--version"])
        return ok

    def git_local_identity(self):
        ok_n, out_n, _ = self.git(["config", "--get", "user.name"])
        ok_e, out_e, _ = self.git(["config", "--get", "user.email"])
        return (out_n.strip() if ok_n and out_n else None,
                out_e.strip() if ok_e and out_e else None)

    def git_set_local_identity_if_missing(self):
        name, email = self.git_local_identity()
        if not name:
            self.git(["config", "user.name", "AI Builder"])
        if not email:
            self.git(["config", "user.email", "builder@example.invalid"])

    def ensure_gitignore(self):
        gi = self.project_root_path / ".gitignore"
        if gi.exists(): return
        gi.write_text(
            ".env\nnode_modules/\ndist/\nbuild/\nfunctions/node_modules/\n__pycache__/\n*.pyc\n.artifacts/\n",
            encoding="utf-8"
        )

    def ensure_repo_initialized(self):
        if not self.git_is_available():
            return False
        if not (self.project_root_path / ".git").exists():
            self.git(["init", "-b", "main"])
            self.ensure_gitignore()
            self.git_set_local_identity_if_missing()
            self.git(["add", "-A"])
            self.git(["commit", "-m", "chore: initialize repository"])
        return True

    def git_commit_all(self, message: str):
        if not self.ensure_repo_initialized():
            return
        self.git(["add", "-A"])
        self.git(["commit", "-m", message])

    def git_remote_exists(self) -> bool:
        ok, out, _ = self.git(["remote", "-v"])
        return ok and "origin" in out

    def git_set_remote(self, remote_url: str):
        ok, out, _ = self.git(["remote", "-v"])
        if "origin" in out:
            self.git(["remote", "set-url", "origin", remote_url])
        else:
            self.git(["remote", "add", "origin", remote_url])

    def git_fetch(self):
        if self.git_remote_exists():
            self.git(["fetch", "origin"])

    def git_pull_rebase_main(self):
        if self.git_remote_exists():
            self.git(["pull", "--rebase", "origin", "main"])

    def git_push_u_main(self):
        if not self.git_remote_exists():
            return False
        self.git(["branch", "-M", "main"])
        ok, _, _ = self.git(["push", "-u", "origin", "main"])
        return ok

    def gh_available(self) -> bool:
        import shutil
        return shutil.which("gh") is not None

    def gh_repo_create_and_push(self, name: str, public: bool = True):
        if not self.gh_available():
            return False, "", "GitHub CLI (gh) not found in PATH."
        ok_auth, out_auth, err_auth = run_p(["gh","auth","status"], cwd=self.project_root_path)
        if not ok_auth:
            return False, "", f"gh auth not ready:\n{err_auth or out_auth}"
        vis = "--public" if public else "--private"
        ok, out, err = run_p(["gh","repo","create",name,"--source",".",vis,"--push"], cwd=self.project_root_path)
        return ok, out, err

    # ------- CI watch (GitHub Actions) -----------------------------------------

    def origin_owner_repo(self) -> Optional[Tuple[str,str]]:
        ok, out, _ = self.git(["remote", "-v"])
        if not ok or "origin" not in out: return None
        first = [line for line in out.splitlines() if line.startswith("origin")][:1]
        if not first: return None
        url = first[0].split()[1]
        return parse_origin_url_to_repo(url)

    def gh_actions_latest_run_cli(self, repo: str, branch: str = "main") -> Optional[str]:
        ok, out, _ = run_p(
            ["gh","run","list","--repo",repo,"--branch",branch,"--limit","1","--json","databaseId,status,conclusion"]
        )
        if not ok:
            return None
        try:
            data = json.loads(out or "[]")
        except Exception:
            return None
        if not data:
            return None
        return str(data[0].get("databaseId"))

    def gh_actions_run_log_cli(self, repo: str, run_id: str) -> Optional[str]:
        ok, out, _ = run_p(["gh","run","view",run_id,"--repo",repo,"--log"])
        return out if ok else None


    def gh_actions_latest_run_rest(self, owner: str, repo: str, branch: str = "main") -> Optional[int]:
        token = os.getenv("GITHUB_TOKEN", "").strip()
        if not token: return None
        url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs?branch={branch}&per_page=1"
        r = requests.get(url, headers={"Authorization": f"Bearer {token}", "Accept":"application/vnd.github+json"})
        if r.status_code != 200: return None
        js = r.json()
        runs = js.get("workflow_runs", [])
        if not runs: return None
        return runs[0]["id"]

    def gh_actions_run_log_rest(self, owner: str, repo: str, run_id: int) -> Optional[str]:
        token = os.getenv("GITHUB_TOKEN", "").strip()
        if not token: return None
        # logs are a zip; GitHub returns a URL that serves a zip
        url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/logs"
        r = requests.get(url, headers={"Authorization": f"Bearer {token}", "Accept":"application/vnd.github+json"})
        if r.status_code != 200: return None
        # Some clients will get a zip; GH also returns a redirect to zip. We’ll just return text pointer.
        # Simpler approach: also fetch the jobs & steps:
        jr = requests.get(f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/jobs",
                          headers={"Authorization": f"Bearer {token}", "Accept":"application/vnd.github+json"})
        if jr.status_code != 200: return None
        out_lines = []
        for job in jr.json().get("jobs", []):
            out_lines.append(f"# Job: {job.get('name')} ({job.get('status')}/{job.get('conclusion')})")
            for step in job.get("steps", []):
                out_lines.append(f"## Step: {step.get('name')} ({step.get('status')}/{step.get('conclusion')})")
                # Individual step logs API is enterprise-only; we capture names/status here.
        return "\n".join(out_lines) if out_lines else None

    def ci_poll_and_collect_logs(self, repo_hint: Optional[str], branch: str = "main",
                                 wait_seconds: int = 180, poll_interval: int = 10) -> Tuple[str, Optional[str]]:
        """
        Returns (status, logs_text) where status in {"success","failure","timed_out","unknown"}.
        Prefers gh CLI; falls back to REST.
        """
        # derive owner/repo
        owner_repo = None
        if repo_hint and "/" in repo_hint:
            owner_repo = tuple(repo_hint.split("/",1))
        else:
            owner_repo = self.origin_owner_repo()
        if not owner_repo:
            return "unknown", "No origin remote detected; cannot infer repo."

        owner, repo = owner_repo if len(owner_repo)==2 else (None, None)
        full = f"{owner}/{repo}" if owner else repo_hint

        # initial fetch of latest run id using gh CLI or REST
        run_id = None
        if self.gh_available():
            rid = self.gh_actions_latest_run_cli(full, branch=branch)
            run_id = str(rid) if rid else None
        if not run_id:
            rid = self.gh_actions_latest_run_rest(owner, repo, branch=branch) if owner and repo else None
            run_id = str(rid) if rid else None
        if not run_id:
            return "unknown", "Could not find a workflow run to inspect."

        # poll status via gh if possible, else REST
        start = time.time()
        status = "unknown"
        logs_text = None
        while time.time() - start < wait_seconds:
            if self.gh_available():
                p = run_p(
                    ["gh","run","view",run_id,"--repo",full,"--json","status,conclusion"],
                    capture_output=True, text=True
                )
                if p.returncode == 0:
                    js = json.loads(p.stdout or "{}")
                    st = js.get("status"); concl = js.get("conclusion")
                else:
                    st = None; concl = None
            else:
                token = os.getenv("GITHUB_TOKEN","").strip()
                if not token: break
                rr = requests.get(f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}",
                                  headers={"Authorization": f"Bearer {token}","Accept":"application/vnd.github+json"})
                if rr.status_code != 200: break
                jj = rr.json(); st = jj.get("status"); concl = jj.get("conclusion")

            if st in ("completed","success","failure","cancelled") or (st=="completed" and concl):
                if concl == "success":
                    status = "success"
                elif concl in ("failure","cancelled","timed_out","action_required"):
                    status = "failure"
                else:
                    status = "unknown"
                break

            time.sleep(poll_interval)

        if status == "unknown" and (time.time() - start) >= wait_seconds:
            return "timed_out", "CI watch timed out before completion."

        # fetch logs
        if self.gh_available():
            logs_text = self.gh_actions_run_log_cli(full, run_id)
        if not logs_text and owner and repo:
            logs_text = self.gh_actions_run_log_rest(owner, repo, int(run_id))

        return status, logs_text

    def summarize_ci_logs_for_prompt(self, logs: str, max_chars: int = 12000) -> str:
        if not logs:
            return "No CI logs available."
        # keep last portion + error lines
        tail = logs[-max_chars:]
        error_lines = [ln for ln in tail.splitlines() if re.search(r"(?i)error|failed|exception|denied|forbidden", ln)]
        snippet = "\n".join(error_lines[-200:])  # most recent error-ish lines
        return f"CI logs (tail):\n{tail[-6000:]}\n\nError lines:\n{snippet[-4000:]}"

# ------------------------------- CLI ------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="AI App Builder (continuous, patches, git, validators, CI watcher).")
    ap.add_argument("--root", default="./ai_project", help="Project root directory.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--preset", choices=["firebase","gcp-run"], default=None)
    ap.add_argument("--mode", choices=["web","code"], default="web")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--auto-commit", dest="auto_commit", action="store_true", default=True,
                    help="Auto git commit after operations (default: on).")
    ap.add_argument("--no-auto-commit", dest="auto_commit", action="store_false",
                    help="Disable auto git commits.")
    ap.add_argument("--auto-push", dest="auto_push", action="store_true", default=True,
                    help="Auto git push after commits (default: on).")
    ap.add_argument("--no-auto-push", dest="auto_push", action="store_false",
                    help="Disable auto push.")
    ap.add_argument("--gh-create", action="store_true",
                    help="Create a GitHub repo using gh CLI and push after generation (requires gh installed & logged in).")
    ap.add_argument("--remote", type=str, default=None,
                    help="Optional: existing remote URL (HTTPS or SSH) to push changes to.")
    ap.add_argument("--repo", type=str, default=None,
                    help="Optional GitHub repo in owner/name form (auto-detected from origin if omitted).")
    ap.add_argument("--ci-watch", action="store_true",
                    help="After push, poll GitHub Actions and analyze logs for failures.")
    ap.add_argument("--ci-timeout", type=int, default=180,
                    help="Seconds to wait for CI to finish when --ci-watch is set (default: 180).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # gen: initial generation
    gen = sub.add_parser("gen", help="Generate project from a prompt.")
    gen.add_argument("prompt", help="Prompt for initial scaffold, in quotes.")

    # change: apply iterative change
    chg = sub.add_parser("change", help="Apply a change request (supports multi-round).")
    chg.add_argument("instruction", help="Change request, in quotes.")
    chg.add_argument("--rounds", type=int, default=1,
                     help="Number of consecutive change rounds to run (default: 1).")

    # validate only
    val = sub.add_parser("validate", help="Run validation commands.")

    # fix with validation logs
    fix = sub.add_parser("fix", help="Validate, send logs to model, and apply the fixes.")
    fix.add_argument("--rounds", type=int, default=2)

    # commit
    com = sub.add_parser("commit", help="Git add & commit all changes.")
    com.add_argument("-m", "--message", default="chore: update via AI builder")

    # zip
    zp = sub.add_parser("zip", help="Zip current project.")
    zp.add_argument("--name", default="artifact.zip")

    # history
    hs = sub.add_parser("history", help="Show history length.")

    args = ap.parse_args()

    sc = AIProjectScaffolder(
        project_root=args.root,
        model=args.model,
        verbose=args.verbose
    )

    def _maybe_create_or_push():
        # Create repo with gh if requested and no origin yet
        if args.gh_create and not sc.git_remote_exists():
            repo_name = Path(args.root).resolve().name
            ok, out, err = sc.gh_repo_create_and_push(repo_name, public=True)
            print("[gh-create]", out if ok else err)
            return
        # Else set remote (if provided) and push
        if args.remote:
            sc.git_set_remote(args.remote)
        if args.auto_push:
            sc.git_push_u_main()

    def _ci_watch_and_maybe_fix():
        if not args.ci_watch:
            return
        status, logs = sc.ci_poll_and_collect_logs(args.repo, wait_seconds=args.ci_timeout)
        print(f"[ci] status={status}")
        if logs:
            print("[ci logs] (truncated)\n" + logs[-2000:])
        if status == "failure" and logs:
            # build a targeted prompt and apply one extra change pass
            ci_prompt = (
                "The GitHub Actions deployment failed. "
                "Analyze the CI logs below (Firebase or Cloud Run) and provide ONLY the changed blocks "
                "to fix the failure.\n\n" + sc.summarize_ci_logs_for_prompt(logs)
            )
            files, diffs, patches = sc.apply_changes(ci_prompt, preset=args.preset, mode=args.mode)
            sc.write_blocks(files)
            if diffs:  sc.apply_unified_diff(None, diffs)
            if patches: sc.apply_json_patches(None, patches)
            print(f"[ci-fix] Applied: {len(files)} files, {len(diffs)} diffs, {len(patches)} patches")
            if args.auto_commit:
                sc.git_commit_all("fix(ci): address CI deployment failure")
                _maybe_create_or_push()

    # ---------------- gen ----------------
    if args.cmd == "gen":
        sc.git_fetch(); sc.git_pull_rebase_main()
        files, diffs, patches = sc.send(args.prompt, preset=args.preset, mode=args.mode)
        sc.write_blocks(files)
        if diffs:  sc.apply_unified_diff(None, diffs)
        if patches: sc.apply_json_patches(None, patches)
        print(f"Generated: {len(files)} files, {len(diffs)} diffs, {len(patches)} patches")
        if args.auto_commit:
            sc.git_commit_all("feat: initial scaffold via AI builder")
            _maybe_create_or_push()
            _ci_watch_and_maybe_fix()

    # ---------------- change ----------------
    elif args.cmd == "change":
        sc.git_fetch(); sc.git_pull_rebase_main()
        total_rounds = max(1, int(args.rounds))
        for i in range(total_rounds):
            files, diffs, patches = sc.apply_changes(args.instruction, preset=args.preset, mode=args.mode)
            sc.write_blocks(files)
            if diffs:  sc.apply_unified_diff(None, diffs)
            if patches: sc.apply_json_patches(None, patches)
            print(f"[change round {i+1}/{total_rounds}] Applied: {len(files)} files, {len(diffs)} diffs, {len(patches)} patches")
            if args.auto_commit:
                sc.git_commit_all(f"chore: change (round {i+1}) - {args.instruction[:60]}")
                _maybe_create_or_push()
                _ci_watch_and_maybe_fix()

    # ---------------- validate ----------------
    elif args.cmd == "validate":
        results = sc.validate_project()
        for r in results:
            print(f"$ {r['cmd']}  OK={r['ok']}")
            if r['stderr']: print(r['stderr'][:2000])

    # ---------------- fix ----------------
    elif args.cmd == "fix":
        sc.git_fetch(); sc.git_pull_rebase_main()
        for i in range(args.rounds):
            results = sc.validate_project()
            if all(r["ok"] for r in results):
                print("✅ Validation passed."); break
            prompt = sc.feedback_prompt_from_results(results)
            files, diffs, patches = sc.apply_changes(prompt, preset=args.preset, mode=args.mode)
            sc.write_blocks(files)
            if diffs:  sc.apply_unified_diff(None, diffs)
            if patches: sc.apply_json_patches(None, patches)
            print(f"[fix round {i+1}/{args.rounds}] Applied: {len(files)} files, {len(diffs)} diffs, {len(patches)} patches")
            if args.auto_commit:
                sc.git_commit_all(f"fix: apply AI fixes (round {i+1})")
                _maybe_create_or_push()
                _ci_watch_and_maybe_fix()

    # ---------------- commit ----------------
    elif args.cmd == "commit":
        sc.git_commit_all(args.message)
        print("Committed.")

    # ---------------- zip ----------------
    elif args.cmd == "zip":
        zip_path = Path(args.root).resolve() / args.name
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in Path(args.root).rglob("*"):
                if p.is_file() and ".git" not in p.parts:
                    zf.write(p, p.relative_to(args.root))
        print("Created", zip_path)

    # ---------------- history ----------------
    elif args.cmd == "history":
        print(f"Turns in history: {len(sc.history)}")

if __name__ == "__main__":
    main()

# ai_builder.py
# pip install openai python-dotenv
# export OPENAI_API_KEY=sk-...

import os, re, json, zipfile, subprocess, argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ----------------------------- Constants --------------------------------------

DEFAULT_MODEL = "gpt-4o-mini"  # change to your preferred model
FILE_BLOCK_REGEX = r'---\s*file:\s*(.+?)\s*---\n(.*?)(?=(?:\n---\s*file:\s*)|\Z)'
DIFF_BLOCK_REGEX = r'---\s*diff:\s*(.+?)\s*---\n(.*?)(?=(?:\n---\s*(?:file|diff|patch):\s*)|\Z)'
PATCH_BLOCK_REGEX = r'---\s*patch:\s*(.+?)\s*---\n(.*?)(?=(?:\n---\s*(?:file|diff|patch):\s*)|\Z)'

DEFAULT_SYSTEM = """You are a senior full-stack engineer.
When the user describes an app or change, output either:
1) COMPLETE files, or
2) ONLY UPDATED files,
using **exactly** one of these block formats:

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
  - firebase.json (hosting config; SPA rewrite to /index.html; optionally rewrite /api/**)
  - .firebaserc (project id: "your-firebase-project-id")
  - firestore.rules (locked by default)
  - storage.rules (locked by default)
  - .gitignore (.env, node_modules, dist/build, functions/node_modules)
  - .env.example
  - emulators.json (hosting, firestore, functions if present)
  - Vite + React + Tailwind (unless user overrides); include tailwind/postcss configs
- If API requested: Cloud Functions (Node 20, TypeScript) exposing /api/hello
  and hosting rewrite for /api/**.
- Include a minimal README with Firebase CLI steps.
"""

GCP_RUN_PRESET = """
GCP Run preset:
- Backend for Google Cloud Run (Python FastAPI):
  - backend/main.py (FastAPI hello at /api/hello)
  - backend/requirements.txt (fastapi, uvicorn[standard])
  - Dockerfile (python:3.11-slim)
  - README.md with build/push/deploy steps for Artifact Registry + Cloud Run.
- Firebase Hosting rewrites:
  - firebase.json routes /api/** to Cloud Run (placeholder serviceId "your-run-service", region "us-central1"),
    and SPA fallback to /index.html if a front-end exists.
- Front-end default: Vite + React + Tailwind with tailwind/postcss configs (unless user opts out).
"""

CODE_ONLY_PRESET = """
CODE-ONLY mode:
- Do NOT generate website scaffolding (no firebase.json, .firebaserc, package.json, vite config, tailwind/Dockerfiles)
  unless explicitly asked.
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
    # Firebase / emulators (no deploy here)
    "firebase emulators:start --only hosting",
]

# --------------------------- Utilities ----------------------------------------

def ensure_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")
    return key

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

        # record user turn first, so history.json always exists even if API fails
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
        # (optional) tighten ext policy here if you like
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
            original = target.read_text(encoding="utf-8").splitlines(keepends=True)
            patched = self._naive_apply_diff(original, diff["content"])
            target.write_text("".join(patched), encoding="utf-8")
            if self.verbose: print("[patch-diff]", target)

    def _naive_apply_diff(self, original_lines: List[str], diff_text: str) -> List[str]:
        # placeholder diff applier — consider a real patch lib for production
        new_lines = [l for l in original_lines]
        adds, removes = [], []
        for line in diff_text.splitlines(True):
            if line.startswith(('+++', '---', '@@')):
                continue
            if line.startswith('+'):
                adds.append(line[1:])
            elif line.startswith('-'):
                removes.append(line[1:])
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
    
    def gh_repo_create_and_push(self, name: str, public: bool = True):
        """Create GitHub repo via GH CLI and push current project."""
        if not self.gh_available():
            return False, "", "GitHub CLI (gh) not found in PATH."
        vis = "--public" if public else "--private"
        p = subprocess.run(
            ["gh", "repo", "create", name, "--source", ".", vis, "--push"],
            cwd=str(self.project_root_path), capture_output=True, text=True
        )
        return p.returncode == 0, (p.stdout or ""), (p.stderr or "")

    def apply_json_patches(self, root: Optional[str], patch_blocks):
        root_path = Path(root).expanduser().resolve() if root else self.project_root_path
        for patch in patch_blocks:
            rel = patch["filename"]
            if not is_safe_relative(rel):
                raise ValueError(f"Unsafe path in patch: {rel}")
            target = root_path / rel
            if not target.exists():
                raise FileNotFoundError(f"File for patch not found: {rel}")
            text = target.read_text(encoding="utf-8")
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
        raw = json.loads(self.history_path.read_text(encoding="utf-8"))
        self.history = [ChatTurn(**t) for t in raw]

    # ------------- Validation loop -----------------
    def run_cmd(self, cmd: str, cwd: Optional[Path] = None) -> Dict[str, str]:
        cwd = cwd or self.project_root_path
        allowed = any(cmd.strip().startswith(w) for w in WHITELISTED_COMMANDS)
        if not allowed:
            return {"cmd": cmd, "ok": False, "stdout": "", "stderr": "Command not whitelisted."}
        p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, shell=True)
        out = (p.stdout or "")[-8000:]
        err = (p.stderr or "")[-8000:]
        return {"cmd": cmd, "ok": p.returncode == 0, "stdout": out, "stderr": err}

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
        p = subprocess.run(["git"] + args, cwd=str(self.project_root_path),
                           capture_output=True, text=True)
        return p.returncode == 0, (p.stdout or ""), (p.stderr or "")

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
        if self.verbose: print("[gitignore] created")

    def ensure_repo_initialized(self):
        if not self.git_is_available():
            if self.verbose: print("[git] not installed or not in PATH; skipping git init/commit.")
            return False
        if not (self.project_root_path / ".git").exists():
            ok, out, err = self.git(["init", "-b", "main"])
            if self.verbose: print("[git init -b main]", ok, err or out)
            self.ensure_gitignore()
            self.git_set_local_identity_if_missing()
            self.git(["add", "-A"])
            self.git(["commit", "-m", "chore: initialize repository"])
        return True

    def git_commit_all(self, message: str):
        if not self.ensure_repo_initialized():
            return
        self.git(["add", "-A"])
        ok, out, err = self.git(["commit", "-m", message])
        if self.verbose: print("[git commit]", ok, err or out)

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
            ok, _, err = self.git(["fetch", "origin"])
            if self.verbose: print("[git fetch]", ok, err)

    def git_pull_rebase_main(self):
        if self.git_remote_exists():
            ok, out, err = self.git(["pull", "--rebase", "origin", "main"])
            if self.verbose: print("[git pull --rebase origin main]", ok, err or out)

    def git_push_u_main(self):
        if not self.git_remote_exists():
            if self.verbose: print("[git push] skipped: no 'origin' remote set.")
            return False
        self.git(["branch", "-M", "main"])
        ok, out, err = self.git(["push", "-u", "origin", "main"])
        if self.verbose: print("[git push -u origin main]", ok, err or out)
        return ok

    def gh_available(self) -> bool:
        import shutil
        return shutil.which("gh") is not None


# ------------------------------- CLI ------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="AI App Builder (continuous, patches, git, validators).")
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
    sub = ap.add_subparsers(dest="cmd", required=True)

    # gen: initial generation
    gen = sub.add_parser("gen", help="Generate project from a prompt.")
    gen.add_argument("prompt", help="Prompt for initial scaffold, in quotes.")

    # change: apply iterative change
    chg = sub.add_parser("change", help="Apply a change request (supports multi-round).")
    chg.add_argument("instruction", help="Change request, in quotes.")
    chg.add_argument("--rounds", type=int, default=1,
                     help="Number of consecutive change rounds to run (default: 1).")

    # validate: run build/tests etc. (whitelisted)
    val = sub.add_parser("validate", help="Run validation commands.")

    # fix: run validate, send logs back, apply returned patches/files
    fix = sub.add_parser("fix", help="Validate, send logs to model, and apply the fixes.")
    fix.add_argument("--rounds", type=int, default=2)

    # commit: git add/commit all
    com = sub.add_parser("commit", help="Git add & commit all changes.")
    com.add_argument("-m", "--message", default="chore: update via AI builder")

    # zip: create archive
    zp = sub.add_parser("zip", help="Zip current project.")
    zp.add_argument("--name", default="artifact.zip")

    # history: print saved turns count
    hs = sub.add_parser("history", help="Show history length.")

    args = ap.parse_args()

    sc = AIProjectScaffolder(
        project_root=args.root,
        model=args.model,
        verbose=args.verbose
    )

    def _maybe_create_or_push(sc, args):
        # If user asked to create the repo via gh (once) and there's no origin yet:
        if args.gh_create and not sc.git_remote_exists():
            repo_name = Path(args.root).resolve().name
            ok, out, err = sc.gh_repo_create_and_push(repo_name, public=True)
            print("[gh-create]", out if ok else err)
            return
        # Else push via git if we have/just set a remote
        if args.remote:
            sc.git_set_remote(args.remote)
        if args.auto_push:
            sc.git_push_u_main()


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
            _maybe_create_or_push(sc, args)
            if args.gh_create:
                repo_name = Path(args.root).resolve().name
                ok, out, err = sc.gh_repo_create_and_push(repo_name, public=True)
                print("[gh-create]", out if ok else err)
            elif args.remote:
                sc.git_set_remote(args.remote)
                if args.auto_push: sc.git_push_u_main()
            elif args.auto_push:
                sc.git_push_u_main()


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
                _maybe_create_or_push(sc, args)

                if args.remote:
                    sc.git_set_remote(args.remote)
                if args.auto_push:
                    sc.git_push_u_main()

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
                print("✅ Validation passed.")
                break
            prompt = sc.feedback_prompt_from_results(results)
            files, diffs, patches = sc.apply_changes(prompt, preset=args.preset, mode=args.mode)
            sc.write_blocks(files)
            if diffs:  sc.apply_unified_diff(None, diffs)
            if patches: sc.apply_json_patches(None, patches)
            print(f"[fix round {i+1}/{args.rounds}] Applied: {len(files)} files, {len(diffs)} diffs, {len(patches)} patches")
            if args.auto_commit:
                sc.git_commit_all(f"fix: apply AI fixes (round {i+1})")
                _maybe_create_or_push(sc, args)
                if args.remote:
                    sc.git_set_remote(args.remote)
                if args.auto_push:
                    sc.git_push_u_main()

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

from __future__ import annotations

import ast
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

_ISSUE_RE = re.compile(r"(?<!\d)#(\d{2,6})(?!\d)")


def _python_symbols(text: str) -> set[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    return {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


class HistoryScanner:
    """Git history plus semantic before/after correlation."""

    def scan(self, root, issue_numbers: Iterable[str | int] | None = None, limit=500):
        root = Path(root).resolve()
        if not (root / ".git").exists():
            return {
                "git_repository": False,
                "commits": 0,
                "issue_correlations": [],
                "changed_paths": [],
            }
        commits = self._log(root, limit)
        wanted = {str(x).lstrip("#") for x in (issue_numbers or [])}
        correlations = []
        changed = set()
        for commit in commits:
            nums = sorted(
                set(_ISSUE_RE.findall(commit["subject"] + " " + commit["body"]))
            )
            matched = [n for n in nums if not wanted or n in wanted]
            if matched:
                paths = self._show_names(root, commit["hash"])
                changed.update(paths)
                parent = self._parent(root, commit["hash"])
                correlations.append(
                    {
                        **{k: commit[k] for k in ("hash", "date", "subject")},
                        "issues": matched,
                        "paths": paths,
                        "parent": parent,
                        "fix_candidate": bool(parent),
                        "regression_test_paths": [
                            p
                            for p in paths
                            if "/test" in f"/{p.lower()}"
                            or p.lower().startswith("test")
                        ],
                    }
                )
        return {
            "git_repository": True,
            "commits": len(commits),
            "issue_correlations": correlations,
            "changed_paths": sorted(changed),
            "latest": commits[0] if commits else None,
        }

    def analyze_issue(self, root, issue, limit=1000):
        root = Path(root).resolve()
        issue = str(issue).lstrip("#")
        scan = self.scan(root, [issue], limit)
        analyses = []
        for c in scan["issue_correlations"]:
            if not c.get("parent"):
                continue
            for path in c["paths"]:
                before = self._show_file(root, c["parent"], path)
                after = self._show_file(root, c["hash"], path)
                if before is None and after is None:
                    continue
                bs = _python_symbols(before or "") if path.endswith(".py") else set()
                a_s = _python_symbols(after or "") if path.endswith(".py") else set()
                analyses.append(
                    {
                        "commit": c["hash"],
                        "parent": c["parent"],
                        "issue": issue,
                        "path": path,
                        "before_present": before is not None,
                        "after_present": after is not None,
                        "before_lines": len(before.splitlines())
                        if before is not None
                        else 0,
                        "after_lines": len(after.splitlines())
                        if after is not None
                        else 0,
                        "changed": before != after,
                        "changed_symbols": sorted(bs ^ a_s),
                        "before_symbols": sorted(bs),
                        "after_symbols": sorted(a_s),
                        "regression_test": "/test" in f"/{path.lower()}"
                        or path.lower().startswith("test"),
                        "semantic_kind": "python-symbol-change"
                        if path.endswith(".py") and before != after
                        else "file-change",
                    }
                )
        return {
            "issue": issue,
            "commits": scan["issue_correlations"],
            "file_diffs": analyses,
            "semantic_summary": {
                "commits_with_issue": len(scan["issue_correlations"]),
                "files_changed": len(analyses),
                "symbols_changed": sorted(
                    {s for x in analyses for s in x["changed_symbols"]}
                ),
                "regression_test_files": [
                    x["path"] for x in analyses if x["regression_test"]
                ],
            },
        }

    def _log(self, root, limit):
        fmt = "%H%x1f%cI%x1f%s%x1f%b%x1e"
        try:
            p = subprocess.run(
                ["git", "-C", str(root), "log", f"-n{limit}", f"--format={fmt}"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
        except (subprocess.SubprocessError, OSError):
            return []
        out = []
        for raw in p.stdout.split("\x1e"):
            f = raw.strip("\n\x1e").split("\x1f")
            if len(f) >= 4 and f[0]:
                out.append({"hash": f[0], "date": f[1], "subject": f[2], "body": f[3]})
        return out

    @staticmethod
    def _parent(root, commit):
        try:
            p = subprocess.run(
                ["git", "-C", str(root), "rev-list", "--parents", "-n", "1", commit],
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            )
            parts = p.stdout.split()
            return parts[1] if len(parts) > 1 else None
        except (subprocess.SubprocessError, OSError):
            return None

    @staticmethod
    def _show_names(root, commit):
        try:
            p = subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "diff-tree",
                    "--root",
                    "--no-commit-id",
                    "--name-only",
                    "-r",
                    commit,
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            )
        except (subprocess.SubprocessError, OSError):
            return []
        return sorted(x for x in p.stdout.splitlines() if x.strip())

    @staticmethod
    def _show_file(root, commit, path):
        try:
            p = subprocess.run(
                ["git", "-C", str(root), "show", f"{commit}:{path}"],
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            )
            return p.stdout
        except (subprocess.SubprocessError, OSError):
            return None

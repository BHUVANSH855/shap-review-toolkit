from pathlib import Path


class TestScanner:
    def scan(self, root: str | Path) -> dict:
        root = Path(root)
        files = []
        for p in root.rglob("test*.py"):
            if ".git" not in p.parts:
                files.append(p.as_posix())
        for p in root.rglob("*_test.py"):
            if ".git" not in p.parts:
                files.append(p.as_posix())
        return {"count": len(set(files)), "files": sorted(set(files))}

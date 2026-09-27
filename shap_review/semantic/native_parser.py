from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class NativeSymbol:
    name: str
    kind: str
    line: int
    scope: str | None = None
    type_text: str | None = None

    def to_dict(self):
        return asdict(self)


class NativeParser:
    """Best-effort C/C++ structural parser with optional tree-sitter acceleration."""

    def parse(self, path: str | Path) -> dict:
        p = Path(path)
        text = p.read_text(encoding="utf-8", errors="replace")
        try:
            import tree_sitter_cpp
            from tree_sitter import Language, Parser

            parser = Parser(Language(tree_sitter_cpp.language()))
            tree = parser.parse(text.encode())
            return {
                "engine": "tree-sitter-cpp",
                "root_type": tree.root_node.type,
                "errors": tree.root_node.has_error,
                "symbols": self._tree_symbols(tree.root_node, text),
            }
        except Exception:
            return {
                "engine": "fallback",
                "root_type": "translation_unit",
                "errors": False,
                "symbols": [s.to_dict() for s in self._fallback_symbols(text)],
            }

    def _tree_symbols(self, node, text):
        lines = text.splitlines()
        out = []

        def walk(n):
            if n.type in {
                "function_definition",
                "function_declarator",
                "field_declaration",
                "declaration",
            }:
                snippet = text[n.start_byte : n.end_byte].splitlines()[0][:200]
                out.append(
                    NativeSymbol(snippet, n.type, n.start_point[0] + 1).to_dict()
                )
            for c in n.children:
                walk(c)

        walk(node)
        return out

    def _fallback_symbols(self, text):
        out = []
        pat = re.compile(
            r"(?:static\s+|inline\s+|const\s+|virtual\s+)*[\w:<>,~*&\s]+\s+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{"
        )
        for i, line in enumerate(text.splitlines(), 1):
            m = pat.search(line)
            if m:
                out.append(NativeSymbol(m.group(1), "function", i))
        return out

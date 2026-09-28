from __future__ import annotations

import ast
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class APIEraFinding:
    api: str
    era: str
    line: int
    symbol: str | None
    message: str
    provenance: str = "proven"
    confidence: str = "high"
    scope: str = "module"
    binding_state: str = "SHAP"

    def to_dict(self):
        return asdict(self)


class _Scope:
    """Lexical binding scope; SHAP aliases never leak between scopes."""

    def __init__(self, parent=None, name="module", conditional=False):
        self.parent = parent
        self.name = name
        self.conditional = conditional
        self.bindings: dict[str, str] = {}
        self.module_aliases: set[str] = set()
        self.constructor_aliases: dict[str, str] = {}

    def lookup(self, name):
        if name in self.bindings:
            return self.bindings[name]
        if self.parent:
            return self.parent.lookup(name)
        return "UNBOUND"

    def lookup_module_alias(self, name):
        if name in self.bindings and self.bindings[name] != "SHAP_MODULE":
            return False
        if name in self.module_aliases:
            return True
        return self.parent.lookup_module_alias(name) if self.parent else False

    def lookup_constructor(self, name):
        if name in self.bindings and self.bindings[name] not in {"SHAP", "MAYBE_SHAP"}:
            return "NON_SHAP"
        if name in self.constructor_aliases:
            return self.constructor_aliases[name]
        return self.parent.lookup_constructor(name) if self.parent else "UNBOUND"

    def set(self, name, state):
        self.bindings[name] = state
        if state != "SHAP_MODULE":
            self.module_aliases.discard(name)
            if state == "NON_SHAP":
                self.constructor_aliases.pop(name, None)

    def bind_shap_module(self, name):
        self.bindings[name] = "SHAP_MODULE"
        self.module_aliases.add(name)

    def bind_constructor(self, name, state="SHAP"):
        self.bindings[name] = state
        self.constructor_aliases[name] = state


def _binding_targets(node):
    if isinstance(node, ast.Name):
        return [node]
    if isinstance(node, (ast.Tuple, ast.List)):
        out = []
        for elt in node.elts:
            out.extend(_binding_targets(elt))
        return out
    return []


def scan_api_era(source: str) -> list[dict]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [
            APIEraFinding(
                "parser",
                "unknown",
                getattr(exc, "lineno", 1),
                None,
                f"syntax error: {exc}",
                "unknown",
                "low",
            ).to_dict()
        ]

    results = []

    def ctor_state(fn, scope: _Scope):
        if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
            if scope.lookup_module_alias(fn.value.id) and fn.attr in {
                "Explainer",
                "TreeExplainer",
            }:
                return "SHAP"
            return "NON_SHAP"
        if isinstance(fn, ast.Name):
            return scope.lookup_constructor(fn.id)
        return "NON_SHAP"

    def add_call_finding(node, state, scope, symbol, api="Explainer.__call__"):
        if state not in {"SHAP", "MAYBE_SHAP"}:
            return
        conditional = state == "MAYBE_SHAP"
        results.append(
            APIEraFinding(
                api,
                "modern",
                node.lineno,
                symbol,
                "SHAP Explainer/TreeExplainer invocation",
                "conditional" if conditional else "proven",
                "medium" if conditional else "high",
                scope.name,
                state,
            ).to_dict()
        )

    def visit_block(body, scope: _Scope):
        for node in body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "shap":
                        scope.bind_shap_module(alias.asname or "shap")
                    elif alias.asname:
                        scope.set(alias.asname, "NON_SHAP")
            elif isinstance(node, ast.ImportFrom):
                if node.module == "shap":
                    for alias in node.names:
                        if alias.name in {"Explainer", "TreeExplainer"}:
                            scope.bind_constructor(alias.asname or alias.name)
                        elif alias.asname:
                            scope.set(alias.asname, "NON_SHAP")
                elif node.module and node.module.startswith("shap"):
                    for alias in node.names:
                        if alias.asname:
                            scope.set(alias.asname, "NON_SHAP")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                child = _Scope(scope, node.name, scope.conditional)
                for arg in (
                    list(node.args.posonlyargs)
                    + list(node.args.args)
                    + list(node.args.kwonlyargs)
                ):
                    child.set(arg.arg, "NON_SHAP")
                if node.args.vararg:
                    child.set(node.args.vararg.arg, "NON_SHAP")
                if node.args.kwarg:
                    child.set(node.args.kwarg.arg, "NON_SHAP")
                visit_block(node.body, child)
                continue
            elif isinstance(node, ast.ClassDef):
                child = _Scope(scope, node.name, scope.conditional)
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        child.set(base.id, child.lookup(base.id))
                visit_block(node.body, child)
                continue
            elif isinstance(node, ast.If):
                left = _Scope(scope, scope.name, True)
                right = _Scope(scope, scope.name, True)
                visit_block(node.body, left)
                visit_block(node.orelse, right)
                names = set(left.bindings) | set(right.bindings)
                for name in names:
                    a = left.bindings.get(name, scope.lookup(name))
                    b = right.bindings.get(name, scope.lookup(name))
                    if a == b:
                        scope.set(name, a)
                    elif "SHAP" in {a, b} or "SHAP_MODULE" in {a, b}:
                        scope.set(name, "MAYBE_SHAP")
                    else:
                        scope.set(name, "NON_SHAP")
                continue
            elif isinstance(node, ast.Try):
                for block in [
                    node.body,
                    *[h.body for h in node.handlers],
                    node.orelse,
                    node.finalbody,
                ]:
                    visit_block(block, _Scope(scope, scope.name, True))
                continue
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                value = node.value
                targets = (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else (
                        [node.target]
                        if isinstance(node, ast.AnnAssign)
                        else [node.target]
                    )
                )
                state = "NON_SHAP"
                if isinstance(value, ast.Call):
                    state = ctor_state(value.func, scope)
                    if (
                        isinstance(value.func, ast.Attribute)
                        and value.func.attr == "shap_values"
                    ):
                        state = "SHAP"
                elif isinstance(value, ast.Name):
                    state = scope.lookup(value.id)
                    if state == "UNBOUND":
                        state = "NON_SHAP"
                elif isinstance(value, ast.Attribute) and isinstance(
                    value.value, ast.Name
                ):
                    state = (
                        "SHAP"
                        if scope.lookup_module_alias(value.value.id)
                        and value.attr in {"Explainer", "TreeExplainer"}
                        else "NON_SHAP"
                    )
                for target in targets:
                    for name_node in _binding_targets(target):
                        scope.set(name_node.id, state)
                        if isinstance(value, ast.Call) and state in {
                            "SHAP",
                            "MAYBE_SHAP",
                        }:
                            api = (
                                value.func.attr
                                if isinstance(value.func, ast.Attribute)
                                else (
                                    value.func.id
                                    if isinstance(value.func, ast.Name)
                                    else "Explainer"
                                )
                            )
                            results.append(
                                APIEraFinding(
                                    api,
                                    "modern",
                                    node.lineno,
                                    name_node.id,
                                    "SHAP Explainer/TreeExplainer construction",
                                    "conditional"
                                    if state == "MAYBE_SHAP"
                                    else "proven",
                                    "medium" if state == "MAYBE_SHAP" else "high",
                                    scope.name,
                                    state,
                                ).to_dict()
                            )

                # Direct callable: shap.TreeExplainer(model)(X)
                if isinstance(value, ast.Call) and isinstance(value.func, ast.Call):
                    direct = ctor_state(value.func.func, scope)
                    add_call_finding(value, direct, scope, ast.unparse(value.func.func))
                    for target in targets:
                        for name_node in _binding_targets(target):
                            scope.set(name_node.id, "NON_SHAP")
            elif isinstance(node, ast.AugAssign):
                for name_node in _binding_targets(node.target):
                    scope.set(name_node.id, "INVALIDATED")
            elif isinstance(node, ast.Delete):
                for target in node.targets:
                    for name_node in _binding_targets(target):
                        scope.set(name_node.id, "INVALIDATED")
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                fn = call.func
                if isinstance(fn, ast.Attribute) and fn.attr == "shap_values":
                    results.append(
                        APIEraFinding(
                            "shap_values",
                            "legacy",
                            call.lineno,
                            ast.unparse(fn.value),
                            "legacy TreeExplainer.shap_values invocation",
                            "possible",
                            "medium",
                            scope.name,
                            "SHAP",
                        ).to_dict()
                    )
                elif isinstance(fn, ast.Name):
                    add_call_finding(call, scope.lookup(fn.id), scope, fn.id)
                elif isinstance(fn, ast.Call):
                    add_call_finding(
                        call, ctor_state(fn.func, scope), scope, ast.unparse(fn.func)
                    )
            elif isinstance(node, ast.Return) and node.value is not None:
                # FIX: detect TreeExplainer / Explainer calls in return statements.
                # Previously only assignment statements (ast.Assign) were tracked
                # for binding — this missed the common pattern:
                #   def build(model): return shap.TreeExplainer(model, model_output="...")
                ret = node.value
                if isinstance(ret, ast.Call):
                    state = ctor_state(ret.func, scope)
                    if state in {"SHAP", "MAYBE_SHAP"}:
                        add_call_finding(
                            ret,
                            state,
                            scope,
                            ast.unparse(ret.func),
                            api="TreeExplainer.__call__",
                        )
            elif isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Attribute) and fn.attr == "shap_values":
                    results.append(
                        APIEraFinding(
                            "shap_values",
                            "legacy",
                            node.lineno,
                            ast.unparse(fn.value),
                            "legacy TreeExplainer.shap_values invocation",
                            "possible",
                            "medium",
                            scope.name,
                            "SHAP",
                        ).to_dict()
                    )
                if isinstance(fn, ast.Name):
                    add_call_finding(node, scope.lookup(fn.id), scope, fn.id)
                elif isinstance(fn, ast.Call):
                    add_call_finding(
                        node, ctor_state(fn.func, scope), scope, ast.unparse(fn.func)
                    )

    module_scope = _Scope()
    # Preserve the historical analyzer behavior of recognizing the canonical
    # ``shap`` module name in snippets that omit imports, while nested scopes
    # can still shadow it explicitly.
    module_scope.bind_shap_module("shap")
    visit_block(tree.body, module_scope)
    seen = set()
    out = []
    for result in sorted(
        results, key=lambda x: (x["line"], x["api"], x.get("symbol") or "", x["scope"])
    ):
        key = tuple(sorted(result.items()))
        if key not in seen:
            seen.add(key)
            out.append(result)
    return out

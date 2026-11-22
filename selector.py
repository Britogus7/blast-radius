#!/usr/bin/env python3
"""Blast Radius V2 selector:
  Layer 1: sys.monitoring coverage + TrackedDict per-key narrowing + AST/dialect-family maps
  Layer 2: AST-verified semantic equivalence skips (safe None & Enum `is <-> ==`)
  Layer 3: Lazy in-process unittest verifier with rollback-safe AST/bytecode/dict hot-patching
           & ROI-budgeted elimination + optimal 8-worker bin packing with local refinement.
"""

from __future__ import annotations

import ast
import builtins
import concurrent.futures
from concurrent.futures import Future
import contextlib
import gc
import importlib
import io
import json
import math
import os
import pickle
import re
import sys
import time
import types
import unittest
from collections import defaultdict
from pathlib import Path

DATA = Path("/data")
SQLGLOT_ROOT = Path("/app/sqlglot").resolve()
MAPS_PATH = Path("/app/maps/maps.pkl")
WORKER_COST_MS = 2000
MAX_WORKERS = 8
MAX_VERIFY_DUR_MS = 3500
HARD_WALL_BUDGET_S = 115.0

TRACKED_DICT_ATTRS = frozenset({
    "NUMERIC_ESCAPES",
    "FUNCTIONS",
    "FUNCTION_PARSERS",
    "SHOW_PARSERS",
    "PROPERTY_PARSERS",
    "STATEMENT_PARSERS",
    "NO_PAREN_FUNCTIONS",
    "TYPE_CONVERTERS",
    "TRANSFORMS",
    "TYPE_MAPPING",
    "PROPERTIES_LOCATION",
})

TRIE_BACKED_ATTRS = frozenset({
    "SHOW_PARSERS",
    "PROPERTY_PARSERS",
    "STATEMENT_PARSERS",
    "NO_PAREN_FUNCTIONS",
    "NUMERIC_ESCAPES",
    "PROPERTIES_LOCATION",
})

METACLASS_CONSUMED_ATTRS = frozenset({
    "KEYWORDS",
    "SINGLE_TOKENS",
    "QUOTES",
    "IDENTIFIERS",
    "COMMENTS",
    "FORMAT_STRINGS",
    "_QUOTES",
    "_IDENTIFIERS",
    "_COMMENTS",
    "_FORMAT_STRINGS",
    "UNESCAPED_SEQUENCES",
    "TIME_MAPPING",
    "FORMAT_MAPPING",
    "CREATABLE_KIND_MAPPING",
    "SUPPORTED_JSON_PATH_PARTS",
    "STRING_ESCAPES",
    "BYTE_STRING_ESCAPES",
    "IDENTIFIER_ESCAPES",
    "VAR_SINGLE_TOKENS",
    "SHOW_TRIE",
    "SET_TRIE",
})

EQUIV_EXCLUDED_FILES = frozenset({
    "sqlglot/executor/env.py",
    "sqlglot/executor/python.py",
    "sqlglot/executor/table.py",
})

STDLIB_SAFE_NONE_EQ_TYPES = frozenset({
    "None",
    "str",
    "int",
    "float",
    "bool",
    "bytes",
    "list",
    "dict",
    "set",
    "tuple",
    "frozenset",
    "Pattern",
    "Match",
    "Path",
    "datetime",
    "date",
    "timedelta",
    "Decimal",
    "UUID",
})

BASE_CLASSES = frozenset({
    "Dialect",
    "Tokenizer",
    "Parser",
    "Generator",
    "TypeAnnotator",
    "Optimizer",
    "Executor",
    "Expr",
    "Expression",
})

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
CODING_RE = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*[-\w.]+")


def duration_ms(seconds: float) -> int:
    return math.floor(seconds * 1000 + 0.5)


def pack_optimal_with_cost(
    test_nums: list[int] | set[int], dur_by_num: dict[int, int]
) -> tuple[list[list[int]], int]:
    if not test_nums:
        return [], 0
    ordered = sorted(test_nums, key=lambda n: (-dur_by_num[n], n))
    best_cost = None
    best_workers: list[list[int]] = []
    max_w = min(MAX_WORKERS, len(ordered))
    for w in range(1, max_w + 1):
        buckets: list[list[int]] = [[] for _ in range(w)]
        loads = [0] * w
        for n in ordered:
            idx = loads.index(min(loads))
            buckets[idx].append(n)
            loads[idx] += dur_by_num[n]

        if w >= 2 and len(ordered) > w:
            for _ in range(48):
                hi_idx = max(range(w), key=loads.__getitem__)
                hi = loads[hi_idx]
                improved = False
                best_delta = 0
                best_move = None  # ("move", b_pos, j) or ("swap", b_pos, j, j_pos)

                hi_bucket = buckets[hi_idx]
                for b_pos, x in enumerate(hi_bucket):
                    dx = dur_by_num[x]
                    for j in range(w):
                        if j == hi_idx:
                            continue
                        new_hi = hi - dx
                        new_j = loads[j] + dx
                        new_pair = new_hi if new_hi >= new_j else new_j
                        if new_pair < hi:
                            gain = hi - new_pair
                            if gain > best_delta:
                                best_delta = gain
                                best_move = ("move", b_pos, j)

                if best_move is None:
                    for b_pos, x in enumerate(hi_bucket):
                        dx = dur_by_num[x]
                        for j in range(w):
                            if j == hi_idx:
                                continue
                            j_bucket = buckets[j]
                            for j_pos, y in enumerate(j_bucket):
                                dy = dur_by_num[y]
                                if dx <= dy:
                                    continue
                                new_hi = hi - dx + dy
                                new_j = loads[j] + dx - dy
                                new_pair = new_hi if new_hi >= new_j else new_j
                                if new_pair < hi:
                                    gain = hi - new_pair
                                    if gain > best_delta:
                                        best_delta = gain
                                        best_move = ("swap", b_pos, j, j_pos)

                if best_move is None:
                    break
                if best_move[0] == "move":
                    _, b_pos, j = best_move
                    x = hi_bucket.pop(b_pos)
                    dx = dur_by_num[x]
                    loads[hi_idx] -= dx
                    buckets[j].append(x)
                    loads[j] += dx
                else:
                    _, b_pos, j, j_pos = best_move
                    x = hi_bucket[b_pos]
                    y = buckets[j][j_pos]
                    hi_bucket[b_pos] = y
                    buckets[j][j_pos] = x
                    diff = dur_by_num[x] - dur_by_num[y]
                    loads[hi_idx] -= diff
                    loads[j] += diff
                improved = True
                if not improved:
                    break

        cost = max(loads) + WORKER_COST_MS * w
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_workers = [list(b) for b in buckets]
    return [b for b in best_workers if b], (best_cost or 0)


def pack_cost(test_nums: list[int] | set[int], dur_by_num: dict[int, int]) -> int:
    if not test_nums:
        return 0
    _, cost = pack_optimal_with_cost(test_nums, dur_by_num)
    return cost


def parse_patch(patch_text: str) -> list[dict]:
    files: list[dict] = []
    cur = None
    old_line = 0
    new_line = 0
    in_hunk = False

    for raw in patch_text.splitlines():
        if raw.startswith("diff --git "):
            cur = None
            in_hunk = False
            continue
        if raw.startswith("--- "):
            in_hunk = False
            continue
        if raw.startswith("+++ "):
            path = raw[4:]
            if path.startswith("b/"):
                path = path[2:]
            cur = {"file": path, "hunks": []}
            files.append(cur)
            in_hunk = False
            continue
        if cur is None:
            continue
        m = HUNK_RE.match(raw)
        if m:
            old_s = int(m.group(1))
            old_c = int(m.group(2)) if m.group(2) is not None else 1
            new_s = int(m.group(3))
            new_c = int(m.group(4)) if m.group(4) is not None else 1
            hunk = {
                "old_start": old_s,
                "old_count": old_c,
                "new_start": new_s,
                "new_count": new_c,
                "raw_lines": [],
                "removed_lines": [],
                "added_lines": [],
                "del_nums": [],
                "add_anchors": [],
            }
            cur["hunks"].append(hunk)
            old_line = old_s
            new_line = new_s
            block_has_del = False
            in_hunk = True
            continue
        if not in_hunk or not cur["hunks"]:
            continue
        h = cur["hunks"][-1]
        h["raw_lines"].append(raw)
        if raw.startswith("-"):
            h["removed_lines"].append(raw[1:])
            h["del_nums"].append(old_line)
            old_line += 1
            block_has_del = True
        elif raw.startswith("+"):
            h["added_lines"].append(raw[1:])
            if not block_has_del:
                h["add_anchors"].append(max(1, old_line))
            new_line += 1
        elif raw.startswith(" ") or raw == "":
            old_line += 1
            new_line += 1
            block_has_del = False

    return files


def apply_hunks_in_memory(orig_text: str, hunks: list[dict]) -> str | None:
    lines = orig_text.splitlines(keepends=True)
    nl = "\n"
    out: list[str] = []
    pos = 0
    for h in hunks:
        old_s = max(0, h["old_start"] - 1)
        if h["old_count"] == 0:
            target = h["old_start"]
            if target < pos or target > len(lines):
                return None
            out.extend(lines[pos:target])
            pos = target
            for r in h["raw_lines"]:
                if r.startswith("+"):
                    out.append(r[1:] + nl)
        else:
            if old_s < pos or old_s > len(lines):
                return None
            out.extend(lines[pos:old_s])
            pos = old_s
            for r in h["raw_lines"]:
                if r.startswith("-"):
                    if pos >= len(lines):
                        return None
                    if lines[pos].rstrip("\r\n") != r[1:]:
                        return None
                    pos += 1
                elif r.startswith("+"):
                    out.append(r[1:] + nl)
                elif r.startswith(" ") or r == "":
                    if pos >= len(lines):
                        return None
                    out.append(lines[pos])
                    pos += 1
    out.extend(lines[pos:])
    return "".join(out)


def _coding_cookie_changed(orig_text: str, patched_text: str) -> bool:
    orig_top = orig_text.splitlines()[:2]
    pat_top = patched_text.splitlines()[:2]
    if orig_top == pat_top:
        return False
    return any(CODING_RE.match(ln) for ln in orig_top + pat_top)


def is_comment_or_ws_only(orig_text: str | None, patched_text: str | None) -> bool:
    if orig_text is None or patched_text is None:
        return False
    if _coding_cookie_changed(orig_text, patched_text):
        return False
    try:
        return ast.dump(ast.parse(orig_text)) == ast.dump(ast.parse(patched_text))
    except SyntaxError:
        return False


def is_dialect_file(rel: str) -> str | None:
    for prefix in ("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/"):
        if rel.startswith(prefix) and rel.endswith(".py"):
            stem = rel[len(prefix):-3]
            if stem not in ("__init__", "dialect", "parser", "generator", "typing"):
                return stem
    return None


def _build_file_import_map_mem(rel: str, tree: ast.Module) -> dict[str, str]:
    imp: dict[str, str] = {}
    pkg_parts = rel.split("/")[:-1]

    def _visit_top(stmts):
        for st in stmts:
            if isinstance(st, ast.ClassDef):
                imp[st.name] = f"{rel}::{st.name}"
            elif isinstance(st, ast.Import):
                for alias in st.names:
                    local = alias.asname or alias.name.split(".")[0]
                    imp[local] = "__MOD__:" + alias.name.replace(".", "/")
            elif isinstance(st, ast.ImportFrom):
                if st.level > 0:
                    up = st.level - 1
                    base = pkg_parts[: len(pkg_parts) - up] if up <= len(pkg_parts) else []
                    if st.module:
                        base = base + st.module.split(".")
                else:
                    base = st.module.split(".") if st.module else []
                prefix = "/".join(base)
                cand_py = f"{prefix}.py"
                cand_init = f"{prefix}/__init__.py"
                target_rel = None
                if prefix and (SQLGLOT_ROOT / cand_py).is_file():
                    target_rel = cand_py
                elif prefix and (SQLGLOT_ROOT / cand_init).is_file():
                    target_rel = cand_init
                if target_rel is not None:
                    for alias in st.names:
                        if alias.name == "*":
                            continue
                        local = alias.asname or alias.name
                        imp[local] = f"{target_rel}::{alias.name}"
            elif isinstance(st, ast.If):
                _visit_top(st.body)
                _visit_top(st.orelse)

    _visit_top(tree.body)
    return imp


def _qualify_name(name: str, imp_map: dict[str, str] | None) -> str:
    if imp_map:
        q = imp_map.get(name)
        if q and not q.startswith("__MOD__:"):
            return q
    return name


def _extract_type_tokens(ann: ast.AST | None, imp_map: dict[str, str] | None = None) -> set[str]:
    if ann is None:
        return {"__UNKNOWN__"}
    if isinstance(ann, ast.Constant):
        if ann.value is None:
            return {"None"}
        if isinstance(ann.value, str):
            try:
                return _extract_type_tokens(ast.parse(ann.value, mode="eval").body, imp_map)
            except SyntaxError:
                return {"__UNKNOWN__"}
        return {"__UNKNOWN__"}
    if isinstance(ann, ast.Name):
        return {"__UNKNOWN__"} if ann.id in ("Any", "object") else {_qualify_name(ann.id, imp_map)}
    if isinstance(ann, ast.Attribute):
        if ann.attr in ("Any", "object"):
            return {"__UNKNOWN__"}
        if isinstance(ann.value, ast.Name) and imp_map and ann.value.id in imp_map:
            m = imp_map[ann.value.id]
            if m.startswith("__MOD__:"):
                return {f"{m[8:]}::{ann.attr}"}
        return {ann.attr}
    if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
        return _extract_type_tokens(ann.left, imp_map) | _extract_type_tokens(ann.right, imp_map)
    if isinstance(ann, ast.Subscript):
        base_tokens = _extract_type_tokens(ann.value, imp_map)
        if any(b in ("Union", "Annotated", "ClassVar", "Final") for b in base_tokens):
            if "Union" in base_tokens and isinstance(ann.slice, ast.Tuple):
                out: set[str] = set()
                for elt in ann.slice.elts:
                    out |= _extract_type_tokens(elt, imp_map)
                return out
            if "Annotated" in base_tokens and isinstance(ann.slice, ast.Tuple) and ann.slice.elts:
                return _extract_type_tokens(ann.slice.elts[0], imp_map)
            return _extract_type_tokens(ann.slice, imp_map)
        if "Optional" in base_tokens:
            return _extract_type_tokens(ann.slice, imp_map) | {"None"}
        clean = base_tokens - {"t", "typing"}
        return clean if clean else {"__UNKNOWN__"}
    return {"__UNKNOWN__"}


def _infer_expr_types(
    node: ast.AST,
    local_types: dict[str, set[str]],
    cls_attrs: dict[str, set[str]],
    enum_classes: dict[str, list[str]],
    safe_none_eq_types: set[str],
    imp_map: dict[str, str] | None = None,
) -> set[str]:
    if isinstance(node, ast.Constant):
        if node.value is None:
            return {"None"}
        if isinstance(node.value, bool):
            return {"bool"}
        if isinstance(node.value, int):
            return {"int"}
        if isinstance(node.value, float):
            return {"float"}
        if isinstance(node.value, str):
            return {"str"}
        if isinstance(node.value, bytes):
            return {"bytes"}
        return {"__UNKNOWN__"}
    if isinstance(node, (ast.List, ast.ListComp)):
        return {"list"}
    if isinstance(node, (ast.Dict, ast.DictComp)):
        return {"dict"}
    if isinstance(node, (ast.Set, ast.SetComp)):
        return {"set"}
    if isinstance(node, ast.Tuple):
        return {"tuple"}
    if isinstance(node, ast.Name):
        return set(local_types.get(node.id, {"__UNKNOWN__"}))
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name):
            if node.value.id in ("self", "cls"):
                return set(cls_attrs.get(node.attr, {"__UNKNOWN__"}))
            e = node.value.id
            if e in enum_classes and node.attr in enum_classes[e]:
                return {_qualify_name(e, imp_map)}
        return {"__UNKNOWN__"}
    if isinstance(node, ast.IfExp):
        return _infer_expr_types(
            node.body, local_types, cls_attrs, enum_classes, safe_none_eq_types, imp_map
        ) | _infer_expr_types(
            node.orelse, local_types, cls_attrs, enum_classes, safe_none_eq_types, imp_map
        )
    if isinstance(node, ast.Call):
        fn = node.func
        if isinstance(fn, ast.Name):
            q = _qualify_name(fn.id, imp_map)
            if q in safe_none_eq_types and q != "None":
                return {q}
            if fn.id in safe_none_eq_types and fn.id != "None":
                return {fn.id}
        elif isinstance(fn, ast.Attribute) and fn.attr in enum_classes:
            return {_qualify_name(fn.attr, imp_map)}
    return {"__UNKNOWN__"}


class _EquivNormalizer(ast.NodeTransformer):
    def __init__(
        self,
        rel: str,
        imp_map: dict[str, str],
        safe_none_eq_types: set[str],
        enum_classes: dict[str, list[str]],
        class_attr_types: dict[str, dict[str, set[str]]],
    ):
        self.rel = rel
        self.imp_map = imp_map
        self.safe_none_eq_types = safe_none_eq_types
        self.enum_classes = enum_classes
        self.class_attr_types = class_attr_types
        self.cls_stack: list[str] = []
        self.local_stack: list[dict[str, set[str]]] = [{}]

    def _cur_cls_attrs(self) -> dict[str, set[str]]:
        if not self.cls_stack:
            return {}
        fq = f"{self.rel}::{self.cls_stack[-1]}"
        return self.class_attr_types.get(fq, self.class_attr_types.get(self.cls_stack[-1], {}))

    def visit_ClassDef(self, node: ast.ClassDef):
        self.cls_stack.append(node.name)
        res = self.generic_visit(node)
        self.cls_stack.pop()
        return res

    def _visit_func(self, node):
        cls_attrs = self._cur_cls_attrs()
        ltypes: dict[str, set[str]] = defaultdict(set)
        all_args = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
        for a in all_args:
            if a.arg in ("self", "cls"):
                continue
            if a.annotation is not None:
                ltypes[a.arg].update(_extract_type_tokens(a.annotation, self.imp_map))
            else:
                ltypes[a.arg].add("__UNKNOWN__")
        if node.args.vararg:
            ltypes[node.args.vararg.arg].add("tuple")
        if node.args.kwarg:
            ltypes[node.args.kwarg.arg].add("dict")

        # Flow-safe conservative scan of assignments within this function only
        def scan_stmts(stmts):
            for st in stmts:
                if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                    ltypes[st.target.id].update(_extract_type_tokens(st.annotation, self.imp_map))
                elif isinstance(st, ast.Assign):
                    rhs_t = _infer_expr_types(
                        st.value, ltypes, cls_attrs, self.enum_classes, self.safe_none_eq_types, self.imp_map
                    )
                    for t in st.targets:
                        if isinstance(t, ast.Name):
                            ltypes[t.id].update(rhs_t)
                for field in ("body", "orelse", "finalbody"):
                    sub = getattr(st, field, None)
                    if isinstance(sub, list):
                        scan_stmts(sub)
                handlers = getattr(st, "handlers", None)
                if isinstance(handlers, list):
                    for h in handlers:
                        if h.name:
                            ltypes[h.name].add("__UNKNOWN__")
                        scan_stmts(h.body)

        scan_stmts(node.body)
        scan_stmts(node.body)

        self.local_stack.append(dict(ltypes))
        res = self.generic_visit(node)
        self.local_stack.pop()
        return res

    def visit_FunctionDef(self, node: ast.FunctionDef):
        return self._visit_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        return self._visit_func(node)

    def _is_none_const(self, node: ast.AST) -> bool:
        return isinstance(node, ast.Constant) and node.value is None

    def _enum_name_of(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            ename = node.value.id
            if ename in self.enum_classes and node.attr in self.enum_classes[ename]:
                return ename
        return None

    def visit_Compare(self, node: ast.Compare):
        node = self.generic_visit(node)
        if len(node.ops) != 1 or len(node.comparators) != 1:
            return node
        op = node.ops[0]
        if not isinstance(op, (ast.Is, ast.IsNot, ast.Eq, ast.NotEq)):
            return node

        left = node.left
        right = node.comparators[0]
        ltypes = self.local_stack[-1]
        cls_attrs = self._cur_cls_attrs()

        # Rule E1: x is None <-> x == None
        target_expr = None
        if self._is_none_const(right) and not self._is_none_const(left):
            target_expr, const_node = left, right
        elif self._is_none_const(left) and not self._is_none_const(right):
            target_expr, const_node = right, left
        if target_expr is not None:
            tset = _infer_expr_types(
                target_expr, ltypes, cls_attrs, self.enum_classes, self.safe_none_eq_types, self.imp_map
            )
            if tset and "__UNKNOWN__" not in tset and tset.issubset(self.safe_none_eq_types):
                new_op = ast.Eq() if isinstance(op, (ast.Is, ast.Eq)) else ast.NotEq()
                return ast.Compare(left=target_expr, ops=[new_op], comparators=[const_node])

        # Rule E2: x is SafeEnum.MEMBER <-> x == SafeEnum.MEMBER
        enum_side = None
        enum_cls = None
        other_expr = None
        er = self._enum_name_of(right)
        el = self._enum_name_of(left)
        if er and not el:
            enum_cls, enum_side, other_expr = er, right, left
        elif el and not er:
            enum_cls, enum_side, other_expr = el, left, right
        if enum_cls is not None and other_expr is not None:
            tset = _infer_expr_types(
                other_expr, ltypes, cls_attrs, self.enum_classes, self.safe_none_eq_types, self.imp_map
            )
            enum_fq = _qualify_name(enum_cls, self.imp_map)
            allowed = {enum_cls, enum_fq, "None"}
            if tset and "__UNKNOWN__" not in tset and tset.issubset(allowed):
                new_op = ast.Eq() if isinstance(op, (ast.Is, ast.Eq)) else ast.NotEq()
                return ast.Compare(left=other_expr, ops=[new_op], comparators=[enum_side])

        return node


def is_semantically_equivalent_file(
    orig_text: str | None,
    patched_text: str | None,
    rel: str,
    safe_none_eq_types: set[str],
    enum_classes: dict[str, list[str]],
    class_attr_types: dict[str, dict[str, set[str]]],
) -> bool:
    if orig_text is None or patched_text is None:
        return False
    if rel.startswith("tests/") or rel in EQUIV_EXCLUDED_FILES:
        return False
    if _coding_cookie_changed(orig_text, patched_text):
        return False
    try:
        orig_t = ast.parse(orig_text)
        pat_t = ast.parse(patched_text)
    except SyntaxError:
        return False
    imp_map = _build_file_import_map_mem(rel, orig_t)
    n1 = _EquivNormalizer(rel, imp_map, safe_none_eq_types, enum_classes, class_attr_types).visit(orig_t)
    n2 = _EquivNormalizer(rel, imp_map, safe_none_eq_types, enum_classes, class_attr_types).visit(pat_t)
    return ast.dump(n1) == ast.dump(n2)


def _find_class_assign_dict(tree: ast.Module, cls_qual: str, attr_name: str) -> ast.Dict | None:
    parts = cls_qual.split(".") if cls_qual else []
    body = tree.body
    for cname in parts:
        found = None
        for st in body:
            if isinstance(st, ast.ClassDef) and st.name == cname:
                found = st
                break
        if found is None:
            return None
        body = found.body
    for st in body:
        if isinstance(st, ast.Assign) and len(st.targets) == 1:
            t = st.targets[0]
            if isinstance(t, ast.Name) and t.id == attr_name and isinstance(st.value, ast.Dict):
                return st.value
        elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.target.id == attr_name:
            if isinstance(st.value, ast.Dict):
                return st.value
    return None


def _norm_ast_dict_key(k: ast.AST, enum_classes: dict[str, list[str]]) -> str | None:
    if isinstance(k, ast.Constant) and type(k.value) is str:
        return "s:" + k.value
    if isinstance(k, ast.Name):
        return "t:" + k.id
    if isinstance(k, ast.Attribute):
        v = k.value
        v_name = v.id if isinstance(v, ast.Name) else (v.attr if isinstance(v, ast.Attribute) else None)
        if v_name and v_name in enum_classes:
            return f"e:{v_name}.{k.attr}"
        return "t:" + k.attr
    return None


def _parse_dict_ast(
    d_node: ast.Dict, enum_classes: dict[str, list[str]]
) -> tuple[list[str], dict[str, str]] | None:
    spreads: list[str] = []
    entries: dict[str, str] = {}
    for k, v in zip(d_node.keys, d_node.values):
        if k is None:
            spreads.append(ast.dump(v))
        else:
            nk = _norm_ast_dict_key(k, enum_classes)
            if nk is None or nk in entries:
                return None
            entries[nk] = ast.dump(v)
    return spreads, entries


def try_narrow_dict_attr(
    orig_tree: ast.Module | None,
    patched_tree: ast.Module | None,
    rel: str,
    cls_qual: str | None,
    attr: str,
    d_stem: str | None,
    enum_classes: dict[str, list[str]],
    dict_key_cov: dict[str, dict[str, list[int]]],
    tracked_dict_tags: set[str],
    class_descendants: dict[str, list[str]],
    dialect_family_children: dict[str, list[str]],
    dict_tags_by_stem_attr: dict[tuple[str, str], list[str]],
    idx_to_num: dict[int, int],
) -> set[int] | None:
    if orig_tree is None or patched_tree is None or not cls_qual or attr not in TRACKED_DICT_ATTRS:
        return None
    d_orig = _find_class_assign_dict(orig_tree, cls_qual, attr)
    d_pat = _find_class_assign_dict(patched_tree, cls_qual, attr)
    if d_orig is None or d_pat is None:
        return None
    p_orig = _parse_dict_ast(d_orig, enum_classes)
    p_pat = _parse_dict_ast(d_pat, enum_classes)
    if p_orig is None or p_pat is None:
        return None

    o_spreads, o_entries = p_orig
    n_spreads, n_entries = p_pat
    if o_spreads != n_spreads:
        return None

    added_or_removed = set(o_entries) ^ set(n_entries)
    modified = {k for k in o_entries if k in n_entries and o_entries[k] != n_entries[k]}
    if added_or_removed and (attr in TRIE_BACKED_ATTRS or not o_entries or not n_entries):
        return None

    changed_keys = added_or_removed | modified
    if not changed_keys:
        return set()

    cls_id = f"{rel}::{cls_qual}"
    cand_tags: set[str] = {f"{cls_id}::{attr}"}
    for desc in class_descendants.get(cls_id, ()):
        cand_tags.add(f"{desc}::{attr}")
    if d_stem:
        for child_stem in dialect_family_children.get(d_stem, (d_stem,)):
            cand_tags.update(dict_tags_by_stem_attr.get((child_stem, attr), ()))

    if not any(tag in tracked_dict_tags for tag in cand_tags):
        return None

    out: set[int] = set()
    for tag in cand_tags:
        kmap = dict_key_cov.get(tag)
        if not kmap:
            continue
        for idx in kmap.get("__ALL__", ()):
            num = idx_to_num.get(idx)
            if num is not None:
                out.add(num)
        for ck in changed_keys:
            for idx in kmap.get(ck, ()):
                num = idx_to_num.get(idx)
                if num is not None:
                    out.add(num)
    return out


class FastSyncExecutor:
    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        if initializer is not None:
            initializer(*initargs)

    def submit(self, fn, /, *args, **kwargs):
        fut = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            fut.set_exception(exc)
        return fut

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        return [self.submit(fn, *args).result() for args in zip(*iterables)]

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def _dump_list(nodes: list[ast.AST]) -> list[str]:
    return [ast.dump(n) for n in nodes]


def _check_decorator(dec: ast.AST) -> str | None:
    if isinstance(dec, ast.Name) and dec.id in ("classmethod", "staticmethod", "property"):
        return dec.id
    return None


def _unwrap_method_fn(attr_val):
    if isinstance(attr_val, types.FunctionType):
        return attr_val
    if isinstance(attr_val, (classmethod, staticmethod)):
        fn = attr_val.__func__
        return fn if isinstance(fn, types.FunctionType) else None
    if isinstance(attr_val, property):
        fn = attr_val.fget
        return fn if isinstance(fn, types.FunctionType) else None
    return None


def _find_child_code(parent_code: types.CodeType, name: str, ordinal: int) -> types.CodeType | None:
    seen = 0
    for c in parent_code.co_consts:
        if isinstance(c, types.CodeType) and c.co_name == name:
            if seen == ordinal:
                return c
            seen += 1
    return None


def _nested_fn_ast_compatible(o_node: ast.AST, p_node: ast.AST) -> bool:
    o_nested = [c for c in ast.iter_child_nodes(o_node) if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))]
    p_nested = [c for c in ast.iter_child_nodes(p_node) if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))]
    # Walk deeply within body stmts (excluding nested ClassDef)
    def collect_fns(root):
        res = []
        for ch in ast.walk(root):
            if ch is root:
                continue
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                res.append(ch)
        return res

    o_all = collect_fns(o_node)
    p_all = collect_fns(p_node)
    if len(o_all) != len(p_all):
        return False
    for oa, pa in zip(o_all, p_all):
        if oa.name != pa.name or ast.dump(oa.args) != ast.dump(pa.args) or _dump_list(oa.decorator_list) != _dump_list(pa.decorator_list):
            return False
    return True


def _collect_nested_code_swaps(
    old_code: types.CodeType,
    new_code: types.CodeType,
    swaps: list[tuple[types.FunctionType, types.CodeType, types.CodeType]],
) -> bool:
    compat_mask = (
        0x04  # CO_VARARGS
        | 0x08  # CO_VARKEYWORDS
        | 0x20  # CO_GENERATOR
        | 0x80  # CO_COROUTINE
        | 0x200  # CO_ASYNC_GENERATOR
    )
    if (
        old_code.co_freevars != new_code.co_freevars
        or old_code.co_cellvars != new_code.co_cellvars
        or old_code.co_argcount != new_code.co_argcount
        or old_code.co_posonlyargcount != new_code.co_posonlyargcount
        or old_code.co_kwonlyargcount != new_code.co_kwonlyargcount
        or (old_code.co_flags & compat_mask) != (new_code.co_flags & compat_mask)
    ):
        return False

    old_children = [c for c in old_code.co_consts if isinstance(c, types.CodeType)]
    new_children = [c for c in new_code.co_consts if isinstance(c, types.CodeType)]

    if len(old_children) != len(new_children):
        # Abort if any old child code object has live cached FunctionType referrers
        for oc in old_children:
            for ref in gc.get_referrers(oc):
                if isinstance(ref, types.FunctionType) and ref.__code__ is oc:
                    return False
        return True

    for oc, nc in zip(old_children, new_children):
        if oc.co_name != nc.co_name:
            return False
        if oc.co_code != nc.co_code or oc.co_consts != nc.co_consts or oc.co_names != nc.co_names:
            for ref in gc.get_referrers(oc):
                if isinstance(ref, types.FunctionType) and ref.__code__ is oc:
                    swaps.append((ref, oc, nc))
            if not _collect_nested_code_swaps(oc, nc, swaps):
                return False
    return True


class TestVerifier:
    def __init__(self, all_dialects_init: bool = True):
        self.initialized = False
        self._mods: dict[str, types.ModuleType] = {}
        self._setup_classes: set[type] = set()
        self._failed_setup_classes: set[type] = set()
        self._all_dialects_init = all_dialects_init
        self.dialect_base_cls = None
        self.dispatch_cache = None

    def ensure_init(self) -> None:
        if self.initialized:
            return
        os.chdir(SQLGLOT_ROOT)
        if str(SQLGLOT_ROOT) not in sys.path:
            sys.path.insert(0, str(SQLGLOT_ROOT))
        os.environ["SKIP_INTEGRATION"] = "0"
        concurrent.futures.ProcessPoolExecutor = FastSyncExecutor

        import sqlglot
        import sqlglot.generator
        from sqlglot.dialects.dialect import Dialect, Dialects

        if self._all_dialects_init:
            dialects_dir = SQLGLOT_ROOT / "sqlglot" / "dialects"
            if dialects_dir.is_dir():
                for dp in sorted(dialects_dir.glob("*.py")):
                    if dp.stem not in ("__init__", "dialect"):
                        try:
                            importlib.import_module(f"sqlglot.dialects.{dp.stem}")
                        except Exception:
                            pass
            for d in Dialects:
                try:
                    Dialect.get_or_raise(d)
                except Exception:
                    pass
        for mod_name in ("sqlglot.optimizer", "sqlglot.executor", "sqlglot.lineage", "sqlglot.schema", "sqlglot.diff", "sqlglot.planner"):
            try:
                importlib.import_module(mod_name)
            except Exception:
                pass

        self.dialect_base_cls = Dialect
        self.dispatch_cache = sqlglot.generator._DISPATCH_CACHE
        self.initialized = True

    def run_test(self, nodeid: str) -> str:
        """Returns 'pass', 'fail', or 'unverifiable'."""
        if "[" in nodeid or nodeid.count("::") != 2:
            return "unverifiable"
        self.ensure_init()
        fpath, cls_name, meth_name = nodeid.split("::")
        if not fpath.endswith(".py"):
            return "unverifiable"
        mod_name = fpath[:-3].replace("/", ".")
        try:
            mod = self._mods.get(mod_name)
            if mod is None:
                mod = importlib.import_module(mod_name)
                self._mods[mod_name] = mod
            cls = getattr(mod, cls_name, None)
            if cls is None or not (isinstance(cls, type) and issubclass(cls, unittest.TestCase)):
                return "unverifiable"
            if not hasattr(cls, meth_name):
                return "unverifiable"
            if cls in self._failed_setup_classes:
                return "unverifiable"
            if cls not in self._setup_classes:
                try:
                    cls.setUpClass()
                    self._setup_classes.add(cls)
                except Exception:
                    self._failed_setup_classes.add(cls)
                    return "unverifiable"

            inst = cls(meth_name)
            inst.subTest = lambda *a, **kw: contextlib.nullcontext()
            try:
                inst.setUp()
            except unittest.SkipTest:
                return "pass"
            except Exception:
                try:
                    inst.doCleanups()
                except Exception:
                    pass
                return "fail"
            try:
                getattr(inst, meth_name)()
            except unittest.SkipTest:
                return "pass"
            except Exception:
                return "fail"
            finally:
                try:
                    inst.tearDown()
                except Exception:
                    pass
                try:
                    inst.doCleanups()
                except Exception:
                    pass
            return "pass"
        except Exception:
            return "fail"


class CommitHotPatcher:
    def __init__(self, verifier: TestVerifier):
        self.verifier = verifier

    def try_apply(self, active_file_edits: list[dict], orig_texts: dict[str, str], patched_texts: dict[str, str]):
        """Returns a rollback callable on success, or None if unsafe/unsupported."""
        if not active_file_edits:
            return lambda: None
        self.verifier.ensure_init()

        code_swaps: list[tuple[types.FunctionType, types.CodeType, types.CodeType]] = []
        dict_updates: list[tuple[dict, dict, dict]] = []  # (target_dict, orig_snap, new_val)
        attr_updates: list[tuple[object, str, bool, object, object]] = []  # (obj, attr, existed, orig, new)
        fixture_overrides: dict[str, str] = {}

        for fe in active_file_edits:
            rel = fe["rel"]
            patched_str = patched_texts.get(rel)
            orig_str = orig_texts.get(rel)
            if patched_str is None or orig_str is None:
                return None

            if rel.startswith("tests/fixtures/") and not rel.endswith(".py"):
                abs_p = os.path.abspath(str(SQLGLOT_ROOT / rel))
                fixture_overrides[abs_p] = patched_str
                continue

            if not (rel.startswith("sqlglot/") and rel.endswith(".py")):
                return None
            if _coding_cookie_changed(orig_str, patched_str):
                return None

            try:
                orig_t = ast.parse(orig_str)
                pat_t = ast.parse(patched_str)
            except SyntaxError:
                return None

            if len(orig_t.body) != len(pat_t.body):
                return None

            mod_name = rel[:-3].replace("/", ".")
            if mod_name.endswith(".__init__"):
                mod_name = mod_name[:-9]
            try:
                mod = importlib.import_module(mod_name)
                patched_mod_code = compile(patched_str, str((SQLGLOT_ROOT / rel).resolve()), "exec")
            except Exception:
                return None

            ok = self._plan_body(
                mod,
                False,
                mod.__dict__,
                orig_t.body,
                pat_t.body,
                patched_mod_code,
                str((SQLGLOT_ROOT / rel).resolve()),
                code_swaps,
                dict_updates,
                attr_updates,
            )
            if not ok:
                return None

        # Execute all ops cleanly; build exact rollback
        orig_fn_codes: list[tuple[types.FunctionType, types.CodeType]] = []
        orig_dict_snaps: list[tuple[dict, dict]] = []
        orig_attrs: list[tuple[object, str, bool, object]] = []

        orig_open = builtins.open
        orig_io_open = io.open
        orig_path_open = Path.open

        def rollback():
            builtins.open = orig_open
            io.open = orig_io_open
            Path.open = orig_path_open
            for fn, oc in reversed(orig_fn_codes):
                fn.__code__ = oc
            for td, snap in reversed(orig_dict_snaps):
                td.clear()
                td.update(snap)
            for obj, attr, existed, old_v in reversed(orig_attrs):
                if existed:
                    setattr(obj, attr, old_v)
                elif hasattr(obj, attr):
                    delattr(obj, attr)
            if self.verifier.dispatch_cache is not None:
                self.verifier.dispatch_cache.clear()

        try:
            for fn, oc, nc in code_swaps:
                orig_fn_codes.append((fn, fn.__code__))
                fn.__code__ = nc
            for td, snap, nv in dict_updates:
                orig_dict_snaps.append((td, snap))
                td.clear()
                td.update(nv)
            for obj, attr, existed, old_v, new_v in attr_updates:
                orig_attrs.append((obj, attr, existed, old_v))
                setattr(obj, attr, new_v)

            if fixture_overrides:
                def patched_open(file, mode="r", *args, **kwargs):
                    if isinstance(file, (str, os.PathLike)):
                        try:
                            ap = os.path.abspath(os.fspath(file))
                            if ap in fixture_overrides:
                                s = fixture_overrides[ap]
                                if "b" in mode:
                                    return io.BytesIO(s.encode("utf-8"))
                                return io.StringIO(s)
                        except Exception:
                            pass
                    return orig_open(file, mode, *args, **kwargs)

                def patched_path_open(self_p, mode="r", *args, **kwargs):
                    try:
                        ap = os.path.abspath(os.fspath(self_p))
                        if ap in fixture_overrides:
                            s = fixture_overrides[ap]
                            if "b" in mode:
                                return io.BytesIO(s.encode("utf-8"))
                            return io.StringIO(s)
                    except Exception:
                        pass
                    return orig_path_open(self_p, mode, *args, **kwargs)

                builtins.open = patched_open
                io.open = patched_open
                Path.open = patched_path_open

            if self.verifier.dispatch_cache is not None:
                self.verifier.dispatch_cache.clear()
        except Exception:
            rollback()
            return None

        return rollback

    def _plan_body(
        self,
        container,
        is_class: bool,
        mod_globals: dict,
        body_o: list[ast.stmt],
        body_p: list[ast.stmt],
        parent_code: types.CodeType,
        file_abs: str,
        code_swaps: list,
        dict_updates: list,
        attr_updates: list,
    ) -> bool:
        # Track ordinal index among same-name definitions in body_p to match parent_code.co_consts safely
        name_ordinals: dict[str, int] = defaultdict(int)
        for o, p in zip(body_o, body_p):
            p_def_name = getattr(p, "name", None) if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else None
            ord_idx = 0
            if p_def_name is not None:
                ord_idx = name_ordinals[p_def_name]
                name_ordinals[p_def_name] += 1

            if ast.dump(o) == ast.dump(p):
                continue

            if isinstance(o, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not isinstance(p, type(o)) or o.name != p.name:
                    return False
                if ast.dump(o.args) != ast.dump(p.args) or _dump_list(o.decorator_list) != _dump_list(p.decorator_list):
                    return False
                if len(o.decorator_list) > 1:
                    return False
                if len(o.decorator_list) == 1 and _check_decorator(o.decorator_list[0]) is None:
                    return False

                if not is_class:
                    fn = mod_globals.get(o.name)
                    if not isinstance(fn, types.FunctionType):
                        return False
                else:
                    raw = container.__dict__.get(o.name)
                    fn = _unwrap_method_fn(raw)
                    if fn is None:
                        return False

                new_code = _find_child_code(parent_code, p.name, ord_idx)
                if new_code is None:
                    return False
                if not _nested_fn_ast_compatible(o, p):
                    return False

                nested_swaps: list = []
                if not _collect_nested_code_swaps(fn.__code__, new_code, nested_swaps):
                    return False
                # Abort if extra nested changes overlap incompatible state or already queued
                code_swaps.extend(nested_swaps)
                code_swaps.append((fn, fn.__code__, new_code))
                continue

            if isinstance(o, ast.ClassDef):
                if not isinstance(p, ast.ClassDef) or o.name != p.name:
                    return False
                if (
                    _dump_list(o.bases) != _dump_list(p.bases)
                    or _dump_list(o.keywords) != _dump_list(p.keywords)
                    or _dump_list(o.decorator_list) != _dump_list(p.decorator_list)
                ):
                    return False
                if len(o.body) != len(p.body):
                    return False
                sub_cls = container.__dict__.get(o.name) if is_class else mod_globals.get(o.name)
                if not isinstance(sub_cls, type):
                    return False
                cls_code = _find_child_code(parent_code, p.name, ord_idx)
                if cls_code is None:
                    return False
                if not self._plan_body(
                    sub_cls,
                    True,
                    mod_globals,
                    o.body,
                    p.body,
                    cls_code,
                    file_abs,
                    code_swaps,
                    dict_updates,
                    attr_updates,
                ):
                    return False
                continue

            if isinstance(o, (ast.Assign, ast.AnnAssign)):
                if isinstance(o, ast.Assign):
                    if not isinstance(p, ast.Assign) or len(o.targets) != 1 or len(p.targets) != 1:
                        return False
                    to, tp = o.targets[0], p.targets[0]
                    if not (isinstance(to, ast.Name) and isinstance(tp, ast.Name) and to.id == tp.id):
                        return False
                    attr_name = to.id
                    p_val_ast = p.value
                else:
                    if not isinstance(p, ast.AnnAssign):
                        return False
                    if not (isinstance(o.target, ast.Name) and isinstance(p.target, ast.Name) and o.target.id == p.target.id):
                        return False
                    if p.value is None or o.value is None:
                        return False
                    attr_name = o.target.id
                    p_val_ast = p.value

                if not is_class:
                    orig_val = mod_globals.get(attr_name)
                    if not isinstance(orig_val, dict):
                        return False
                    try:
                        expr_c = compile(ast.Expression(p_val_ast), file_abs, "eval")
                        new_val = eval(expr_c, mod_globals, mod_globals)
                    except Exception:
                        return False
                    if not isinstance(new_val, dict):
                        return False
                    dict_updates.append((orig_val, dict(orig_val), new_val))
                    continue

                if attr_name in METACLASS_CONSUMED_ATTRS:
                    return False
                if attr_name not in container.__dict__:
                    return False
                orig_val = container.__dict__[attr_name]
                if isinstance(orig_val, (classmethod, staticmethod, property, types.FunctionType)):
                    return False

                try:
                    expr_c = compile(ast.Expression(p_val_ast), file_abs, "eval")
                    new_val = eval(expr_c, mod_globals, dict(container.__dict__))
                except Exception:
                    return False

                if isinstance(orig_val, dict):
                    if not isinstance(new_val, dict):
                        return False
                    if attr_name in ("SHOW_PARSERS", "SET_PARSERS") and orig_val.keys() != new_val.keys():
                        return False
                    if attr_name == "NUMERIC_ESCAPES":
                        if (orig_val.keys() & {"x", "0", "u"}) != (new_val.keys() & {"x", "0", "u"}):
                            return False
                        if bool(orig_val) != bool(new_val):
                            return False
                    orig_snap = dict(orig_val)
                    dict_updates.append((orig_val, orig_snap, new_val))

                    # Propagate changed keys to MRO subclasses with spread copies
                    changed_k = [
                        k for k in new_val
                        if k in orig_snap and (new_val[k] is not orig_snap[k] and new_val[k] != orig_snap[k])
                    ]
                    removed_k = set(orig_snap) - set(new_val)
                    added_k = set(new_val) - set(orig_snap)

                    seen_subs: set[int] = set()
                    q = list(container.__subclasses__())
                    while q:
                        sub = q.pop()
                        sid = id(sub)
                        if sid in seen_subs:
                            continue
                        seen_subs.add(sid)
                        q.extend(sub.__subclasses__())

                        if attr_name == "arg_types" and hasattr(sub, "required_args") and "arg_types" not in sub.__dict__:
                            new_req = {k for k, v in new_val.items() if v}
                            attr_updates.append((
                                sub,
                                "required_args",
                                "required_args" in sub.__dict__,
                                sub.__dict__.get("required_args"),
                                new_req,
                            ))

                        if attr_name in sub.__dict__:
                            sub_d = sub.__dict__[attr_name]
                            if isinstance(sub_d, dict) and sub_d is not orig_val:
                                # Only propagate into spreads of orig_snap
                                sub_snap = dict(sub_d)
                                sub_new = dict(sub_d)
                                touched_sub = False
                                for k in removed_k:
                                    if k in sub_new and (sub_new[k] is orig_snap[k] or sub_new[k] == orig_snap[k]):
                                        del sub_new[k]
                                        touched_sub = True
                                for k in changed_k:
                                    if k in sub_new and (sub_new[k] is orig_snap[k] or sub_new[k] == orig_snap[k]):
                                        sub_new[k] = new_val[k]
                                        touched_sub = True
                                if added_k and orig_snap.keys() <= sub_d.keys():
                                    for k in added_k:
                                        if k not in sub_new:
                                            sub_new[k] = new_val[k]
                                            touched_sub = True
                                if touched_sub:
                                    dict_updates.append((sub_d, sub_snap, sub_new))

                    if attr_name == "arg_types" and hasattr(container, "required_args"):
                        new_req = {k for k, v in new_val.items() if v}
                        attr_updates.append((
                            container,
                            "required_args",
                            "required_args" in container.__dict__,
                            container.__dict__.get("required_args"),
                            new_req,
                        ))
                    continue

                # Immutable scalar/tuple class attribute (e.g. WEEK_OFFSET)
                if self.verifier.dialect_base_cls and issubclass(container, self.verifier.dialect_base_cls):
                    if attr_name in METACLASS_CONSUMED_ATTRS:
                        return False
                attr_updates.append((container, attr_name, True, orig_val, new_val))
                continue

            return False

        return True


TRIE_COMPANIONS = {
    "SHOW_PARSERS": ("SHOW_PARSERS", "SHOW_TRIE"),
    "SET_PARSERS": ("SET_PARSERS", "SET_TRIE"),
}


def _is_non_comment_fixture_only(active_edits: list[dict]) -> bool:
    if not active_edits:
        return False
    has_non_comment = False
    for fe in active_edits:
        rel = fe["rel"]
        if not (rel.startswith("tests/fixtures/") and not rel.endswith(".py")):
            return False
        for h in fe.get("hunks", ()):
            for ln in h.get("removed_lines", ()) + h.get("added_lines", ()):
                s = ln.strip()
                if s and not s.startswith(("#", "--")):
                    has_non_comment = True
    return has_non_comment


def _est_test_cpu_ms(nid: str, dur_ms: int, live_alpha: float) -> int:
    if dur_ms <= 50:
        return max(5, int(dur_ms * 0.85))
    return int(dur_ms * live_alpha)


def main() -> None:
    t0 = time.monotonic()
    with open(MAPS_PATH, "rb") as f:
        maps = pickle.load(f)

    tests_data = json.loads((DATA / "tests.json").read_text("utf-8"))["tests"]
    id_to_num = {t["id"]: t["number"] for t in tests_data}
    num_to_id = {t["number"]: t["id"] for t in tests_data}
    dur_by_num = {t["number"]: duration_ms(t["seconds"]) for t in tests_data}
    file_to_nums: dict[str, list[int]] = {}
    for t in tests_data:
        file_to_nums.setdefault(t["file"], []).append(t["number"])

    map_tests = maps["tests"]
    idx_to_num: dict[int, int] = {}
    for idx, tid in enumerate(map_tests):
        num = id_to_num.get(tid)
        if num is not None:
            idx_to_num[idx] = num

    line_cov: dict[str, dict[int, list[int]]] = {}
    for fpath, lmap in maps["line_cov"].items():
        line_cov[fpath] = {
            ln: [idx_to_num[i] for i in tids if i in idx_to_num]
            for ln, tids in lmap.items()
        }

    file_cov: dict[str, set[int]] = {
        fpath: {idx_to_num[i] for i in tids if i in idx_to_num}
        for fpath, tids in maps["file_cov"].items()
    }
    file_reads: dict[str, set[int]] = {
        fpath: {idx_to_num[i] for i in tids if i in idx_to_num}
        for fpath, tids in maps["file_reads"].items()
    }
    sym_tests: dict[str, set[int]] = {
        sym: {idx_to_num[i] for i in tids if i in idx_to_num}
        for sym, tids in maps["sym_tests"].items()
    }
    file_scopes = maps["file_scopes"]
    dict_key_cov: dict[str, dict[str, list[int]]] = maps.get("dict_key_cov", {})
    tracked_dict_tags: set[str] = set(maps.get("tracked_dict_tags", ()))
    class_descendants: dict[str, list[str]] = maps.get("class_descendants", {})
    dialect_family_children: dict[str, list[str]] = maps.get("dialect_family_children", {})
    enum_classes: dict[str, list[str]] = maps.get("enum_classes", {})
    enum_fq_tokens: set[str] = set()
    for c_key in maps.get("safe_classes_fq", ()):
        bare = c_key.split("::")[-1]
        if bare in enum_classes:
            enum_fq_tokens.add(c_key)
    safe_none_eq_types: set[str] = (
        STDLIB_SAFE_NONE_EQ_TYPES
        | set(maps.get("safe_classes_fq", ()))
        | set(enum_classes.keys())
        | enum_fq_tokens
    )
    class_attr_types: dict[str, dict[str, set[str]]] = {
        c: {a: set(ts) for a, ts in amap.items()}
        for c, amap in maps.get("class_attr_types", {}).items()
    }
    unverifiable_nums: set[int] = {
        idx_to_num[i] for i in maps.get("unverifiable_tests", ()) if i in idx_to_num
    }

    # Index tracked dict tags by (dialect_stem, attr) for O(1) lookup
    dict_tags_by_stem_attr: dict[tuple[str, str], list[str]] = defaultdict(list)
    for tag in tracked_dict_tags:
        parts = tag.split("::")
        if len(parts) == 3:
            d = is_dialect_file(parts[0])
            if d:
                dict_tags_by_stem_attr[(d, parts[2])].append(tag)

    # Dialect family coverage from file_cov + dialect_family_children
    dialect_cov: dict[str, set[int]] = {}
    for fpath, tids in file_cov.items():
        d = is_dialect_file(fpath)
        if d:
            dialect_cov.setdefault(d, set()).update(tids)
    dialect_family_cov: dict[str, set[int]] = {}
    all_dialects = set(dialect_cov) | set(dialect_family_children)
    for d in all_dialects:
        kids = dialect_family_children.get(d, [d])
        s: set[int] = set()
        for k in kids:
            s.update(dialect_cov.get(k, ()))
        dialect_family_cov[d] = s

    all_nums = [t["number"] for t in tests_data]
    patch_paths = sorted((DATA / "commits").glob("*.patch"))

    # Pass 1: Layer 1 + Layer 2 across all commits
    commit_records: list[dict] = []
    for patch_path in patch_paths:
        patch_text = patch_path.read_text("utf-8", errors="replace")
        file_edits = parse_patch(patch_text)
        selected: set[int] = set()
        active_edits: list[dict] = []
        orig_texts: dict[str, str] = {}
        patched_texts: dict[str, str] = {}

        for fe in file_edits:
            rel = fe["file"]
            hunks = fe["hunks"]
            if not hunks:
                continue

            if rel in file_reads:
                selected.update(file_reads[rel])
                if not rel.endswith(".py"):
                    orig_p = SQLGLOT_ROOT / rel
                    if orig_p.is_file():
                        ot = orig_p.read_text("utf-8", errors="replace")
                        pt = apply_hunks_in_memory(ot, hunks)
                        if pt is not None:
                            orig_texts[rel] = ot
                            patched_texts[rel] = pt
                    active_edits.append({"rel": rel, "hunks": hunks})
                    continue

            if not rel.endswith(".py"):
                if rel.startswith("tests/"):
                    selected.update(all_nums)
                active_edits.append({"rel": rel, "hunks": hunks})
                continue

            orig_p = SQLGLOT_ROOT / rel
            orig_text = orig_p.read_text("utf-8", errors="replace") if orig_p.is_file() else None
            patched_text = apply_hunks_in_memory(orig_text, hunks) if orig_text is not None else None

            if is_comment_or_ws_only(orig_text, patched_text):
                continue

            if is_semantically_equivalent_file(
                orig_text, patched_text, rel, safe_none_eq_types, enum_classes, class_attr_types
            ):
                continue

            if orig_text is not None and patched_text is not None:
                orig_texts[rel] = orig_text
                patched_texts[rel] = patched_text
            active_edits.append({"rel": rel, "hunks": hunks})

            if rel.startswith("tests/"):
                scopes = file_scopes.get(rel, [])
                fmap = line_cov.get(rel, {})
                for h in hunks:
                    touched_lines = set(h["del_nums"]) | set(h["add_anchors"])
                    hit_any_specific = False
                    for ln in touched_lines:
                        if ln in fmap:
                            selected.update(fmap[ln])
                            hit_any_specific = True
                        for sc in scopes:
                            if sc["kind"] == "func" and sc["name"].startswith("test_") and sc["start"] <= ln <= sc["end"]:
                                suffix = f"::{sc['cls']}::{sc['name']}" if sc["cls"] else f"::{sc['name']}"
                                for tnum in file_to_nums.get(rel, []):
                                    if num_to_id[tnum].endswith(suffix):
                                        selected.add(tnum)
                                        hit_any_specific = True
                    if not hit_any_specific:
                        if rel in file_to_nums:
                            selected.update(file_to_nums[rel])
                        elif rel in file_cov:
                            selected.update(file_cov[rel])
                        else:
                            selected.update(all_nums)
                continue

            if rel in ("sqlglot/__init__.py", "sqlglot/dialects/__init__.py", "sqlglot/dialects/dialect.py"):
                for tid_name in (
                    "tests/dialects/test_dialect.py::TestDialect::test_lazy_load",
                    "tests/test_dialect_imports.py::TestDialectImports::test_dialect_import_no_deadlock",
                    "tests/test_dialect_imports.py::TestDialectImports::test_dialect_attribute_access_is_cached",
                ):
                    if tid_name in id_to_num:
                        selected.add(id_to_num[tid_name])

            if rel not in file_scopes:
                if rel in file_cov:
                    selected.update(file_cov[rel])
                continue

            scopes = file_scopes[rel]
            fmap = line_cov.get(rel, {})
            d_stem = is_dialect_file(rel)

            orig_tree = None
            patched_tree = None
            if orig_text is not None and patched_text is not None:
                try:
                    orig_tree = ast.parse(orig_text)
                    patched_tree = ast.parse(patched_text)
                except SyntaxError:
                    pass

            narrowed_cache: dict[tuple[str | None, str], set[int] | None] = {}

            for h in hunks:
                touched_lines = sorted(set(h["del_nums"]) | set(h["add_anchors"]))
                for ln in touched_lines:
                    if ln in fmap:
                        selected.update(fmap[ln])

                    for sc in scopes:
                        if sc["kind"] == "stmt_hdr" and sc["start"] <= ln <= sc["end"]:
                            for k in range(sc["start"], sc["end"] + 1):
                                if k in fmap:
                                    selected.update(fmap[k])

                    enclosing_funcs = [
                        sc for sc in scopes
                        if sc["kind"] == "func" and sc["start"] <= ln <= sc["end"]
                    ]
                    if enclosing_funcs:
                        if ln not in fmap:
                            fn_sc = max(enclosing_funcs, key=lambda s: s["start"])
                            fn_lines = sorted(k for k in fmap if fn_sc["start"] <= k <= fn_sc["end"])
                            if fn_lines:
                                below = [k for k in fn_lines if k <= ln]
                                above = [k for k in fn_lines if k >= ln]
                                if below:
                                    selected.update(fmap[below[-1]])
                                if above:
                                    selected.update(fmap[above[0]])
                        continue

                    handled = False
                    for sc in scopes:
                        if sc["kind"] == "stmt_hdr":
                            continue
                        if sc["start"] <= ln <= sc["end"]:
                            if sc["kind"] == "def_header":
                                fn_lines = [k for k in fmap if sc["fn_start"] <= k <= sc["fn_end"]]
                                for k in fn_lines:
                                    selected.update(fmap[k])
                                if sc["name"] in sym_tests:
                                    t = sym_tests[sc["name"]]
                                    if d_stem and d_stem in dialect_family_cov:
                                        selected.update(t & dialect_family_cov[d_stem])
                                    else:
                                        selected.update(t)
                                handled = True
                                break
                            elif sc["kind"] in ("class_attr", "module"):
                                targets = sc.get("targets") or []
                                cls_name = sc.get("cls")
                                cls_qual = sc.get("cls_qual") or cls_name

                                if sc["kind"] == "class_attr" and len(targets) == 1 and targets[0] in TRACKED_DICT_ATTRS:
                                    tgt = targets[0]
                                    ck = (cls_qual, tgt)
                                    if ck not in narrowed_cache:
                                        narrowed_cache[ck] = try_narrow_dict_attr(
                                            orig_tree,
                                            patched_tree,
                                            rel,
                                            cls_qual,
                                            tgt,
                                            d_stem,
                                            enum_classes,
                                            dict_key_cov,
                                            tracked_dict_tags,
                                            class_descendants,
                                            dialect_family_children,
                                            dict_tags_by_stem_attr,
                                            idx_to_num,
                                        )
                                    narrowed = narrowed_cache[ck]
                                    if narrowed is not None:
                                        selected.update(narrowed)
                                        handled = True
                                        break

                                use_cls_sym = bool(cls_name and cls_name not in BASE_CLASSES and cls_name in sym_tests)
                                cand = set()
                                if targets:
                                    for tgt in targets:
                                        t_sym: set[int] = set()
                                        for sym_k in TRIE_COMPANIONS.get(tgt, (tgt,)):
                                            t_sym.update(sym_tests.get(sym_k, ()))
                                        if t_sym:
                                            if d_stem and d_stem in dialect_family_cov:
                                                inter = t_sym & dialect_family_cov[d_stem]
                                                cand.update(inter if inter else dialect_family_cov[d_stem])
                                            elif use_cls_sym:
                                                inter = t_sym & sym_tests[cls_name]
                                                cand.update(inter if inter else sym_tests[cls_name])
                                            else:
                                                cand.update(t_sym)
                                        else:
                                            if d_stem and d_stem in dialect_family_cov:
                                                cand.update(dialect_family_cov[d_stem])
                                            elif use_cls_sym:
                                                cand.update(sym_tests[cls_name])
                                            else:
                                                cand.update(file_cov.get(rel, set()))
                                elif d_stem and d_stem in dialect_family_cov:
                                    cand.update(dialect_family_cov[d_stem])
                                elif use_cls_sym:
                                    cand.update(sym_tests[cls_name])
                                else:
                                    cand.update(file_cov.get(rel, set()))
                                selected.update(cand)
                                handled = True
                                break
                    if not handled and ln not in fmap:
                        selected.update(file_cov.get(rel, set()))

        commit_records.append({
            "idx": len(commit_records),
            "commit": patch_path.stem,
            "selected": selected,
            "active_edits": active_edits,
            "orig_texts": orig_texts,
            "patched_texts": patched_texts,
        })

    # Pass 2: ROI-prioritized Layer 3 in-process verification (fast tests <= 3.5s only; strict EV gating)
    verifier = TestVerifier(all_dialects_init=True)
    patcher = CommitHotPatcher(verifier)
    live_alpha = 0.85
    v_pass = 0
    v_fail = 0
    v_unverif = 0
    hp_aborts = 0

    def _cpu_now() -> float:
        t = os.times()
        return t.user + t.system + t.children_user + t.children_system

    def _commit_eligible(rec: dict) -> tuple[bool, float]:
        sel = rec["selected"]
        if not sel or not rec["active_edits"]:
            return False, 0.0
        if _is_non_comment_fixture_only(rec["active_edits"]):
            return False, 0.0
        M0 = {u for u in sel if u in unverifiable_nums or dur_by_num[u] > MAX_VERIFY_DUR_MS}
        U0 = sel - M0
        if not U0:
            return False, 0.0
        cur_cost = pack_cost(sel, dur_by_num)
        min_cost = pack_cost(M0, dur_by_num)
        max_gain = cur_cost - min_cost
        if max_gain <= 0:
            return False, 0.0
        est_rem = sum(_est_test_cpu_ms(num_to_id[x], dur_by_num[x], live_alpha) for x in U0)
        if not M0:
            ok = max_gain > est_rem * 1.15
        else:
            ok = max_gain > est_rem * 1.45
        if not ok:
            return False, 0.0
        return True, (max_gain - est_rem)

    eligible_recs = []
    for r in commit_records:
        ok, net_val = _commit_eligible(r)
        if ok:
            eligible_recs.append((net_val, r))

    if eligible_recs:
        eligible_recs.sort(key=lambda x: (-x[0], x[1]["idx"]))
        for _, rec in eligible_recs:
            if time.monotonic() - t0 >= HARD_WALL_BUDGET_S:
                break
            sel = rec["selected"]
            ok_now, _ = _commit_eligible(rec)
            if not ok_now:
                continue

            rollback = patcher.try_apply(rec["active_edits"], rec["orig_texts"], rec["patched_texts"])
            if rollback is None:
                hp_aborts += 1
                continue

            try:
                M: set[int] = {u for u in sel if u in unverifiable_nums or dur_by_num[u] > MAX_VERIFY_DUR_MS}
                U_fast = sorted(sel - M, key=lambda u: (dur_by_num[u], u))
                U_rem: set[int] = set(U_fast)

                for u in U_fast:
                    if time.monotonic() - t0 >= HARD_WALL_BUDGET_S:
                        break
                    cur_cost = pack_cost(M | U_rem, dur_by_num)
                    min_cost = pack_cost(M, dur_by_num)
                    max_gain = cur_cost - min_cost
                    if max_gain <= 0:
                        break
                    est_rem = sum(_est_test_cpu_ms(num_to_id[x], dur_by_num[x], live_alpha) for x in U_rem)
                    if (not M and max_gain <= est_rem * 1.15) or (M and max_gain <= est_rem * 1.45):
                        break

                    c0 = _cpu_now()
                    res = verifier.run_test(num_to_id[u])
                    dc_ms = (_cpu_now() - c0) * 1000.0
                    if res == "pass" and 100 <= dur_by_num[u] <= MAX_VERIFY_DUR_MS and dc_ms > 5:
                        live_alpha = max(0.55, min(1.05, 0.7 * live_alpha + 0.3 * (dc_ms / dur_by_num[u])))

                    U_rem.discard(u)
                    if res == "pass":
                        v_pass += 1
                    elif res == "fail":
                        M.add(u)
                        v_fail += 1
                    else:
                        M.add(u)
                        v_unverif += 1

                rec["selected"] = M | U_rem
            finally:
                rollback()

    for rec in commit_records:
        workers, _ = pack_optimal_with_cost(sorted(rec["selected"]), dur_by_num)
        print(json.dumps({"commit": rec["commit"], "workers": workers}))

    cpu_s = _cpu_now()
    print(
        f"done wall={time.monotonic()-t0:.2f}s cpu={cpu_s:.2f}s "
        f"v_pass={v_pass} v_fail={v_fail} v_unverif={v_unverif} hp_aborts={hp_aborts}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

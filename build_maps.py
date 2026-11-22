#!/usr/bin/env python3
"""Build-time analyzer for sqlglot at pinned commit.

Runs the full test suite (across 4 build workers) with:
  1. sys.monitoring per-test LINE coverage on /app/sqlglot
  2. File open() tracking (to map fixtures/*.sql and data files to tests)
  3. Synchronous ProcessPoolExecutor override so multiprocess tests are traced in-process
  4. TrackedDict per-key runtime tracking on Dialect/Parser/Generator lookup dicts
Then builds static AST, symbol/attribute reference, type/eq-safety, and family maps.
"""

from __future__ import annotations

import ast
import builtins
import concurrent.futures
import json
import os
import pickle
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import Future
from enum import Enum
from pathlib import Path

SQLGLOT_ROOT = Path("/app/sqlglot").resolve()
MAPS_DIR = Path("/app/maps")

TRACKED_DICT_ATTRS = (
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
    "EXPRESSION_PARSERS",
    "UNARY_PARSERS",
    "BINARY_PARSERS",
    "BITWISE_PARSERS",
    "RANGE_PARSERS",
    "FACTOR_PARSERS",
    "TERM_PARSERS",
    "COLUMN_OPERATORS",
    "ALTER_PARSERS",
    "SET_PARSERS",
    "CONSTRAINT_PARSERS",
    "PLACEHOLDER_PARSERS",
    "PRIMARY_PARSERS",
)

UNSAFE_EQ_BASES = {"reverse_key", "Env", "ENV"}
SAFE_CUSTOM_EQ_FQ = frozenset({
    "sqlglot/expressions/core.py::Expr",
    "sqlglot/expressions/core.py::Expression",
    "sqlglot/dialects/dialect.py::Dialect",
    "sqlglot/dialects/dialect.py::_Dialect",
})


def norm_dict_key(key) -> str | None:
    if type(key) is str:
        return "s:" + key
    if isinstance(key, Enum):
        return f"e:{type(key).__name__}.{key.name}"
    if isinstance(key, type):
        return "t:" + key.__name__
    return None


class SyncExecutor:
    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        if initializer is not None:
            initializer(*initargs)

    def submit(self, fn, /, *args, **kwargs):
        fut = Future()
        try:
            try:
                fn_p, args_p, kwargs_p = pickle.loads(pickle.dumps((fn, args, kwargs)))
            except Exception:
                fn_p, args_p, kwargs_p = fn, args, kwargs
            res = fn_p(*args_p, **kwargs_p)
            try:
                res = pickle.loads(pickle.dumps(res))
            except Exception:
                pass
            fut.set_result(res)
        except BaseException as exc:
            fut.set_exception(exc)
        return fut

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        out = []
        for args in zip(*iterables):
            out.append(self.submit(fn, *args).result())
        return out

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def collect_all_nodeids() -> list[str]:
    env = dict(os.environ, SKIP_INTEGRATION="0", PYTHONPATH=str(SQLGLOT_ROOT))
    proc = subprocess.run(
        ["pytest", "--collect-only", "-q"],
        cwd=str(SQLGLOT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    nodeids = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if "::" in line and not line.startswith(("=", "-")):
            nodeids.append(line)
    return nodeids


def _rel_from_mod(cls: type, root_prefix: str) -> str | None:
    mod = sys.modules.get(getattr(cls, "__module__", ""))
    mfile = getattr(mod, "__file__", None) if mod is not None else None
    if not mfile:
        return None
    abs_p = os.path.abspath(mfile)
    if abs_p.startswith(root_prefix):
        return abs_p[len(root_prefix):]
    return None


def worker_trace_files(worker_id: int, files: list[str], out_path: str) -> None:
    os.chdir(SQLGLOT_ROOT)
    sys.path.insert(0, str(SQLGLOT_ROOT))
    os.environ["SKIP_INTEGRATION"] = "0"

    # Force in-process execution of ProcessPoolExecutor so sys.monitoring traces everything
    concurrent.futures.ProcessPoolExecutor = SyncExecutor

    import importlib
    import io
    import pytest
    from sqlglot.dialects.dialect import Dialect, Dialects
    from sqlglot.parser import Parser
    from sqlglot.generator import Generator

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

    root_prefix = str(SQLGLOT_ROOT) + "/"
    cur_lines: dict[str, set[int]] = defaultdict(set)
    cur_files: set[str] = set()
    setup_files: set[str] = set()
    active = False
    in_setup = False
    test_epoch = 0
    active_dicts: list[TrackedDict] = []

    class TrackedDict(dict):
        __slots__ = ("_tag", "_epoch", "_seen", "_all_epoch")

        def __init__(self, initial: dict, tag: str):
            super().__init__(initial)
            self._tag = tag
            self._epoch = -1
            self._seen: set = set()
            self._all_epoch = -1

        def _rec(self, key) -> None:
            if active:
                if self._epoch != test_epoch:
                    self._epoch = test_epoch
                    self._seen = set()
                    active_dicts.append(self)
                try:
                    self._seen.add(key)
                except TypeError:
                    self._all_epoch = test_epoch

        def _rec_all(self) -> None:
            if active:
                if self._epoch != test_epoch:
                    self._epoch = test_epoch
                    self._seen = set()
                    active_dicts.append(self)
                self._all_epoch = test_epoch

        def __getitem__(self, key):
            self._rec(key)
            return super().__getitem__(key)

        def get(self, key, default=None):
            self._rec(key)
            return super().get(key, default)

        def __contains__(self, key):
            self._rec(key)
            return super().__contains__(key)

        def setdefault(self, key, default=None):
            self._rec(key)
            return super().setdefault(key, default)

        def pop(self, key, *args):
            self._rec(key)
            return super().pop(key, *args)

        def __iter__(self):
            self._rec_all()
            return super().__iter__()

        def keys(self):
            self._rec_all()
            return super().keys()

        def values(self):
            self._rec_all()
            return super().values()

        def items(self):
            self._rec_all()
            return super().items()

        def copy(self):
            self._rec_all()
            return dict(self)

        def __reduce__(self):
            return (dict, (dict(self),))

    # Discover classes and wrap tracked dicts before running tests
    seen_cls_ids: set[int] = set()
    queue: list[type] = [Dialect, Parser, Generator]
    for d_cls in list(Dialect.classes.values()):
        queue.append(d_cls)
        for sub_attr in ("tokenizer_class", "parser_class", "generator_class"):
            sc = getattr(d_cls, sub_attr, None)
            if isinstance(sc, type):
                queue.append(sc)
    # Also scan already-imported sqlglot.* modules
    for mname, mod in list(sys.modules.items()):
        if mod is not None and (mname == "sqlglot" or mname.startswith("sqlglot.")):
            for obj in vars(mod).values():
                if isinstance(obj, type):
                    queue.append(obj)

    all_classes: list[type] = []
    while queue:
        cls = queue.pop()
        cid = id(cls)
        if cid in seen_cls_ids:
            continue
        seen_cls_ids.add(cid)
        all_classes.append(cls)
        try:
            for sub in cls.__subclasses__():
                if id(sub) not in seen_cls_ids:
                    queue.append(sub)
        except Exception:
            pass

    class_descendants: dict[str, set[str]] = defaultdict(set)
    dict_id_to_td: dict[int, TrackedDict] = {}
    dict_aliases: dict[str, str] = {}
    all_tracked_tags: list[str] = []

    for cls in all_classes:
        cls_rel = _rel_from_mod(cls, root_prefix)
        if not cls_rel:
            continue
        cls_id = f"{cls_rel}::{cls.__qualname__}"
        for base in cls.__mro__[1:]:
            b_rel = _rel_from_mod(base, root_prefix)
            if b_rel:
                b_id = f"{b_rel}::{base.__qualname__}"
                if b_id != cls_id:
                    class_descendants[b_id].add(cls_id)

        cls_dict = cls.__dict__
        for attr in TRACKED_DICT_ATTRS:
            if attr in cls_dict:
                val = cls_dict[attr]
                if type(val) is dict:
                    vid = id(val)
                    tag = f"{cls_id}::{attr}"
                    if vid in dict_id_to_td:
                        td = dict_id_to_td[vid]
                        dict_aliases[tag] = td._tag
                    else:
                        td = TrackedDict(val, tag)
                        dict_id_to_td[vid] = td
                    all_tracked_tags.append(tag)
                    setattr(cls, attr, td)

    orig_open = builtins.open
    orig_io_open = io.open

    def tracked_open(file, mode="r", *args, **kwargs):
        if (active or in_setup) and isinstance(file, (str, os.PathLike)):
            try:
                p = os.path.abspath(os.fspath(file))
                if p.startswith(root_prefix):
                    rel_p = p[len(root_prefix):]
                    if in_setup:
                        setup_files.add(rel_p)
                    else:
                        cur_files.add(rel_p)
            except Exception:
                pass
        return orig_open(file, mode, *args, **kwargs)

    builtins.open = tracked_open
    io.open = tracked_open
    orig_path_open = Path.open

    def tracked_path_open(self, mode="r", *args, **kwargs):
        if active or in_setup:
            try:
                p = os.path.abspath(os.fspath(self))
                if p.startswith(root_prefix):
                    rel_p = p[len(root_prefix):]
                    if in_setup:
                        setup_files.add(rel_p)
                    else:
                        cur_files.add(rel_p)
            except Exception:
                pass
        return orig_path_open(self, mode, *args, **kwargs)

    Path.open = tracked_path_open

    TOOL_ID = 0
    sys.monitoring.use_tool_id(TOOL_ID, "blast_cov")

    def on_line(code, line_no):
        if not active:
            return sys.monitoring.DISABLE
        fn = code.co_filename
        if fn.startswith(root_prefix):
            cur_lines[fn[len(root_prefix):]].add(line_no)
        return sys.monitoring.DISABLE

    sys.monitoring.register_callback(TOOL_ID, sys.monitoring.events.LINE, on_line)

    results: dict[str, dict] = {}
    setup_results: dict[str, set[str]] = defaultdict(set)
    dict_key_hits: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    class CovPlugin:
        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_setup(self, item):
            nonlocal in_setup
            setup_files.clear()
            in_setup = True
            try:
                yield
            finally:
                in_setup = False
                if setup_files:
                    parts = item.nodeid.split("::")
                    cls_key = "::".join(parts[:2]) if len(parts) >= 2 else parts[0]
                    setup_results[cls_key].update(setup_files)

        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_protocol(self, item, nextitem):
            nonlocal active, test_epoch
            cur_lines.clear()
            cur_files.clear()
            active_dicts.clear()
            test_epoch += 1
            sys.monitoring.restart_events()
            active = True
            sys.monitoring.set_events(TOOL_ID, sys.monitoring.events.LINE)
            try:
                yield
            finally:
                sys.monitoring.set_events(TOOL_ID, 0)
                active = False
                nid = item.nodeid
                results[nid] = {
                    "lines": {k: sorted(v) for k, v in cur_lines.items()},
                    "files": sorted(cur_files),
                }
                for td in active_dicts:
                    hits_for_tag = dict_key_hits[td._tag]
                    if td._all_epoch == test_epoch:
                        hits_for_tag["__ALL__"].append(nid)
                    for rk in td._seen:
                        nk = norm_dict_key(rk)
                        hits_for_tag[nk if nk is not None else "__ALL__"].append(nid)

    rc = pytest.main(["-q", "--no-header", "--tb=short", *files], plugins=[CovPlugin()])
    sys.monitoring.free_tool_id(TOOL_ID)
    if rc != 0:
        raise RuntimeError(f"worker {worker_id} pytest exit {rc}")

    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "tests": results,
                "setup_files": {k: sorted(v) for k, v in setup_results.items()},
                "dict_keys": {tag: dict(km) for tag, km in dict_key_hits.items()},
                "class_descendants": {k: sorted(v) for k, v in class_descendants.items()},
                "dict_aliases": dict_aliases,
                "tracked_dict_tags": sorted(all_tracked_tags),
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def _build_file_import_map(rel: str, tree: ast.Module, existing_rels: set[str]) -> dict[str, str]:
    """Map local imported/defined names in `rel` to `target_rel::Symbol` (or `__MOD__:pkg`)."""
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
                target_rel = cand_py if cand_py in existing_rels else (cand_init if cand_init in existing_rels else None)
                if target_rel is not None:
                    for alias in st.names:
                        if alias.name == "*":
                            continue
                        local = alias.asname or alias.name
                        imp[local] = f"{target_rel}::{alias.name}"
            elif isinstance(st, ast.If):
                # Include `if t.TYPE_CHECKING:` imports used in annotations
                _visit_top(st.body)
                _visit_top(st.orelse)

    _visit_top(tree.body)
    return imp


def _qualify_name(name: str, imp_map: dict[str, str]) -> str:
    q = imp_map.get(name)
    if q and not q.startswith("__MOD__:"):
        return q
    return name


def _extract_type_tokens(ann: ast.AST | None, imp_map: dict[str, str] | None = None) -> set[str]:
    """Extract set of base type names (module-qualified when resolvable) from a type annotation AST."""
    if ann is None:
        return {"__UNKNOWN__"}
    if isinstance(ann, ast.Constant):
        if ann.value is None:
            return {"None"}
        if isinstance(ann.value, str):
            try:
                sub = ast.parse(ann.value, mode="eval").body
                return _extract_type_tokens(sub, imp_map)
            except SyntaxError:
                return {"__UNKNOWN__"}
        return {"__UNKNOWN__"}
    if isinstance(ann, ast.Name):
        if ann.id in ("Any", "object"):
            return {"__UNKNOWN__"}
        return {_qualify_name(ann.id, imp_map) if imp_map else ann.id}
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


def _infer_rhs_tokens(
    rhs: ast.AST | None,
    enum_classes: dict[str, list[str]],
    safe_classes: set[str],
    cls_attrs: dict[str, set[str]],
    imp_map: dict[str, str] | None = None,
) -> set[str]:
    if rhs is None:
        return {"__UNKNOWN__"}
    if isinstance(rhs, ast.Constant):
        if rhs.value is None:
            return {"None"}
        if isinstance(rhs.value, bool):
            return {"bool"}
        if isinstance(rhs.value, int):
            return {"int"}
        if isinstance(rhs.value, float):
            return {"float"}
        if isinstance(rhs.value, str):
            return {"str"}
        if isinstance(rhs.value, bytes):
            return {"bytes"}
        return {"__UNKNOWN__"}
    if isinstance(rhs, (ast.List, ast.ListComp)):
        return {"list"}
    if isinstance(rhs, (ast.Dict, ast.DictComp)):
        return {"dict"}
    if isinstance(rhs, (ast.Set, ast.SetComp)):
        return {"set"}
    if isinstance(rhs, ast.Tuple):
        return {"tuple"}
    if isinstance(rhs, ast.IfExp):
        return (
            _infer_rhs_tokens(rhs.body, enum_classes, safe_classes, cls_attrs, imp_map)
            | _infer_rhs_tokens(rhs.orelse, enum_classes, safe_classes, cls_attrs, imp_map)
        )
    if isinstance(rhs, ast.Attribute):
        if isinstance(rhs.value, ast.Name):
            if rhs.value.id == "self":
                return set(cls_attrs.get(rhs.attr, {"__SELF_ATTR:" + rhs.attr}))
            e = rhs.value.id
            if e in enum_classes and rhs.attr in enum_classes[e]:
                return {_qualify_name(e, imp_map) if imp_map else e}
        return {"__UNKNOWN__"}
    if isinstance(rhs, ast.Call):
        fn = rhs.func
        if isinstance(fn, ast.Name):
            q = _qualify_name(fn.id, imp_map) if imp_map else fn.id
            if fn.id in enum_classes or fn.id in ("str", "int", "float", "bool", "bytes", "list", "dict", "set", "tuple", "frozenset"):
                return {fn.id}
            if q in safe_classes:
                return {q}
        elif isinstance(fn, ast.Attribute):
            if fn.attr in enum_classes:
                return {fn.attr}
    return {"__UNKNOWN__"}


def build_ast_maps(test_ids: list[str], line_cov: dict[str, dict[int, list[int]]]) -> dict:
    """Build static AST, symbol/attribute reference, type/eq-safety, and family maps."""
    attr_refs: dict[str, list[tuple[str, int]]] = defaultdict(list)
    file_scopes: dict[str, list[dict]] = {}
    parsed_trees: list[tuple[str, ast.Module]] = []

    py_files = sorted((SQLGLOT_ROOT / "sqlglot").rglob("*.py")) + sorted(
        (SQLGLOT_ROOT / "tests").rglob("*.py")
    )
    unverifiable_tests: set[int] = set()
    test_id_to_idx = {nid: i for i, nid in enumerate(test_ids)}
    for i, nid in enumerate(test_ids):
        if (
            nid.startswith("tests/test_sqlglotc.py")
            or nid.startswith("tests/test_dialect_imports.py")
            or nid.endswith("::test_lazy_load")
            or "[" in nid
            or nid.count("::") != 2
        ):
            unverifiable_tests.add(i)

    for path in py_files:
        rel = path.relative_to(SQLGLOT_ROOT).as_posix()
        src = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(src, filename=rel)
        except SyntaxError:
            continue

        if rel.startswith("sqlglot/"):
            parsed_trees.append((rel, tree))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                    attr_refs[node.attr].append((rel, node.lineno))
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    attr_refs[node.id].append((rel, node.lineno))

        scopes: list[dict] = []

        def visit_body(stmts, cls_name: str | None, cls_qual: str | None, depth: int):
            for st in stmts:
                s_start = st.lineno
                s_end = getattr(st, "end_lineno", st.lineno)
                if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    dec_start = st.decorator_list[0].lineno if st.decorator_list else st.lineno
                    body_start = st.body[0].lineno if st.body else st.lineno
                    scopes.append({
                        "kind": "def_header",
                        "cls": cls_name,
                        "cls_qual": cls_qual,
                        "name": st.name,
                        "start": dec_start,
                        "end": max(dec_start, body_start - 1),
                        "fn_start": st.lineno,
                        "fn_end": s_end,
                    })
                    scopes.append({
                        "kind": "func",
                        "cls": cls_name,
                        "cls_qual": cls_qual,
                        "name": st.name,
                        "start": body_start,
                        "end": s_end,
                    })
                    if rel.startswith("tests/") and st.name.startswith("test_"):
                        for sub_n in ast.walk(st):
                            if isinstance(sub_n, ast.Name) and sub_n.id in ("subprocess", "Popen"):
                                suffix = f"::{cls_name}::{st.name}" if cls_name else f"::{st.name}"
                                for nid, tidx in test_id_to_idx.items():
                                    if nid.startswith(rel + "::") and nid.endswith(suffix):
                                        unverifiable_tests.add(tidx)
                                break
                    visit_body(st.body, cls_name, cls_qual, depth + 1)
                elif isinstance(st, ast.ClassDef):
                    dec_start = st.decorator_list[0].lineno if st.decorator_list else st.lineno
                    body_start = st.body[0].lineno if st.body else st.lineno
                    sub_qual = f"{cls_qual}.{st.name}" if cls_qual else st.name
                    scopes.append({
                        "kind": "class_header",
                        "cls": st.name,
                        "cls_qual": sub_qual,
                        "name": st.name,
                        "start": dec_start,
                        "end": max(dec_start, body_start - 1),
                        "cls_start": st.lineno,
                        "cls_end": s_end,
                    })
                    visit_body(st.body, st.name, sub_qual, depth + 1)
                elif isinstance(st, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.Try)):
                    header_end = st.lineno
                    if hasattr(st, "test") and hasattr(st.test, "end_lineno"):
                        header_end = max(header_end, st.test.end_lineno)
                    if header_end > st.lineno:
                        scopes.append({
                            "kind": "stmt_hdr",
                            "cls": cls_name,
                            "cls_qual": cls_qual,
                            "name": None,
                            "start": st.lineno,
                            "end": header_end,
                        })
                    for child_list_name in ("body", "orelse", "handlers", "finalbody"):
                        sub = getattr(st, child_list_name, None)
                        if isinstance(sub, list):
                            if child_list_name == "handlers":
                                for h in sub:
                                    visit_body(h.body, cls_name, cls_qual, depth + 1)
                            else:
                                visit_body(sub, cls_name, cls_qual, depth + 1)
                elif depth == 0:
                    targets = []
                    if isinstance(st, ast.Assign):
                        for t in st.targets:
                            if isinstance(t, ast.Name):
                                targets.append(t.id)
                    elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                        targets.append(st.target.id)
                    scopes.append({
                        "kind": "module",
                        "cls": None,
                        "cls_qual": None,
                        "name": targets[0] if targets else None,
                        "targets": targets,
                        "start": s_start,
                        "end": s_end,
                    })
                elif cls_name is not None:
                    targets = []
                    if isinstance(st, ast.Assign):
                        for t in st.targets:
                            if isinstance(t, ast.Name):
                                targets.append(t.id)
                            elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                                targets.append(t.value.id)
                    elif isinstance(st, ast.AnnAssign):
                        if isinstance(st.target, ast.Name):
                            targets.append(st.target.id)
                    elif isinstance(st, ast.AugAssign):
                        if isinstance(st.target, ast.Name):
                            targets.append(st.target.id)
                    scopes.append({
                        "kind": "class_attr",
                        "cls": cls_name,
                        "cls_qual": cls_qual,
                        "name": targets[0] if targets else None,
                        "targets": targets,
                        "start": s_start,
                        "end": s_end,
                    })
                elif s_end > s_start:
                    scopes.append({
                        "kind": "stmt_hdr",
                        "cls": cls_name,
                        "cls_qual": cls_qual,
                        "name": None,
                        "start": s_start,
                        "end": s_end,
                    })

        visit_body(tree.body, None, None, 0)
        file_scopes[rel] = scopes

    # Static __eq__ safety + Enum collection + module-qualified class attribute type inference
    existing_rels = {rel for rel, _ in parsed_trees}
    file_imp_maps: dict[str, dict[str, str]] = {
        rel: _build_file_import_map(rel, tree, existing_rels)
        for rel, tree in parsed_trees
    }
    cls_fq_bases: dict[str, list[str]] = defaultdict(list)
    classes_with_eq_fq: set[str] = set()
    enum_classes: dict[str, list[str]] = {}
    raw_class_defs: list[tuple[str, ast.ClassDef, str]] = []

    for rel, tree in parsed_trees:
        imp = file_imp_maps[rel]
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                cls_fq = f"{rel}::{node.name}"
                raw_class_defs.append((rel, node, cls_fq))
                base_fqs: list[str] = []
                base_bare: list[str] = []
                for b in node.bases:
                    if isinstance(b, ast.Name):
                        base_bare.append(b.id)
                        bfq = imp.get(b.id)
                        base_fqs.append(bfq if (bfq and not bfq.startswith("__MOD__:")) else b.id)
                    elif isinstance(b, ast.Attribute):
                        base_bare.append(b.attr)
                        if isinstance(b.value, ast.Name) and b.value.id in imp:
                            m = imp[b.value.id]
                            if m.startswith("__MOD__:"):
                                base_fqs.append(f"{m[8:]}::{b.attr}")
                            else:
                                base_fqs.append(b.attr)
                        else:
                            base_fqs.append(b.attr)
                cls_fq_bases[cls_fq].extend(base_fqs)
                if node.name in UNSAFE_EQ_BASES:
                    classes_with_eq_fq.add(cls_fq)
                if any(bn in ("Enum", "AutoName", "IntEnum", "StrEnum") for bn in base_bare):
                    members: list[str] = []
                    for st in node.body:
                        if isinstance(st, ast.Assign):
                            for t in st.targets:
                                if isinstance(t, ast.Name) and t.id.isupper():
                                    members.append(t.id)
                        elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.target.id.isupper():
                            members.append(st.target.id)
                    enum_classes[node.name] = sorted(set(members))
                for st in node.body:
                    if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)) and st.name in ("__eq__", "__ne__"):
                        if cls_fq not in SAFE_CUSTOM_EQ_FQ:
                            classes_with_eq_fq.add(cls_fq)

    changed = True
    while changed:
        changed = False
        for cls_fq, bases in cls_fq_bases.items():
            if cls_fq in classes_with_eq_fq:
                continue
            for b in bases:
                bare_b = b.split("::")[-1]
                if b in classes_with_eq_fq or bare_b in UNSAFE_EQ_BASES:
                    classes_with_eq_fq.add(cls_fq)
                    changed = True
                    break

    safe_classes_fq: set[str] = {
        cls_fq for _, _, cls_fq in raw_class_defs if cls_fq not in classes_with_eq_fq
    }
    unsafe_bare_names = {c.split("::")[-1] for c in classes_with_eq_fq} | UNSAFE_EQ_BASES
    safe_classes_no_eq: set[str] = {
        node.name for _, node, cls_fq in raw_class_defs
        if cls_fq in safe_classes_fq and node.name not in unsafe_bare_names
    }

    # Collect class attribute annotations (explicit AnnAssign prioritized) & assignments
    class_attr_types: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    explicit_ann_attrs: dict[str, set[str]] = defaultdict(set)

    for rel, node, cls_fq in raw_class_defs:
        imp = file_imp_maps[rel]
        cattrs = class_attr_types[cls_fq]
        exp_set = explicit_ann_attrs[cls_fq]
        for st in node.body:
            if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                exp_set.add(st.target.id)
                cattrs[st.target.id].update(_extract_type_tokens(st.annotation, imp))
        for sub in ast.walk(node):
            if isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Attribute):
                if isinstance(sub.target.value, ast.Name) and sub.target.value.id in ("self", "cls"):
                    exp_set.add(sub.target.attr)
                    cattrs[sub.target.attr].update(_extract_type_tokens(sub.annotation, imp))

    for rel, node, cls_fq in raw_class_defs:
        imp = file_imp_maps[rel]
        cattrs = class_attr_types[cls_fq]
        exp_set = explicit_ann_attrs[cls_fq]
        for st in node.body:
            if isinstance(st, ast.Assign):
                inf = _infer_rhs_tokens(st.value, enum_classes, safe_classes_fq, cattrs, imp)
                for t in st.targets:
                    if isinstance(t, ast.Name) and t.id not in exp_set:
                        cattrs[t.id].update(inf)
        for sub in ast.walk(node):
            if isinstance(sub, ast.Assign):
                inf = _infer_rhs_tokens(sub.value, enum_classes, safe_classes_fq, cattrs, imp)
                for t in sub.targets:
                    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id in ("self", "cls"):
                        if t.attr not in exp_set:
                            cattrs[t.attr].update(inf)

    # Resolve __SELF_ATTR: references & propagate Dialect/subclass attributes in fixpoint
    changed = True
    while changed:
        changed = False
        for cls_fq, bases in cls_fq_bases.items():
            cattrs = class_attr_types[cls_fq]
            exp_set = explicit_ann_attrs[cls_fq]
            for b in bases:
                if b in class_attr_types:
                    for k, v in class_attr_types[b].items():
                        if k not in exp_set and not v.issubset(cattrs[k]):
                            cattrs[k].update(v)
                            changed = True
            for k, v in list(cattrs.items()):
                refs = [tok for tok in v if tok.startswith("__SELF_ATTR:")]
                for ref in refs:
                    ref_attr = ref.split(":", 1)[1]
                    ref_vals = cattrs.get(ref_attr)
                    if ref_vals:
                        clean_vals = {x for x in ref_vals if not x.startswith("__SELF_ATTR:")}
                        if not clean_vals.issubset(v):
                            v.update(clean_vals)
                            changed = True
                    elif "__UNKNOWN__" not in v:
                        v.add("__UNKNOWN__")
                        changed = True
        for cls_fq, cattrs in class_attr_types.items():
            for b in cls_fq_bases.get(cls_fq, ()):
                if b in class_attr_types:
                    b_exp = explicit_ann_attrs[b]
                    for k, v in cattrs.items():
                        if k in class_attr_types[b] and k not in b_exp:
                            if not v.issubset(class_attr_types[b][k]):
                                class_attr_types[b][k].update(v)
                                changed = True

    for cattrs in class_attr_types.values():
        for k, v in cattrs.items():
            for ref in [tok for tok in v if tok.startswith("__SELF_ATTR:")]:
                v.discard(ref)

    # Also store bare-class aliases for backward lookup when unambiguous
    all_cattrs_out = {c: {a: sorted(tset) for a, tset in amap.items()} for c, amap in class_attr_types.items()}

    # Dialect family children across dialects/, parsers/, generators/, typing/
    dialect_children: dict[str, set[str]] = {}
    import_re = re.compile(r"from\s+sqlglot\.(?:dialects|parsers|generators|typing)\.([a-z0-9_]+)\s+import")
    for sub in ("dialects", "parsers", "generators", "typing"):
        sub_dir = SQLGLOT_ROOT / "sqlglot" / sub
        if sub_dir.is_dir():
            for p in sub_dir.glob("*.py"):
                child = p.stem
                if child in ("__init__", "dialect", "parser", "generator", "typing"):
                    continue
                dialect_children.setdefault(child, {child}).add(child)
                txt = p.read_text("utf-8", errors="ignore")
                for m in import_re.finditer(txt):
                    parent = m.group(1)
                    if parent not in ("dialect", "parser", "generator", "typing"):
                        dialect_children.setdefault(parent, {parent}).add(child)
    changed = True
    while changed:
        changed = False
        for parent, kids in list(dialect_children.items()):
            extra = set()
            for k in kids:
                extra.update(dialect_children.get(k, ()))
            if not extra.issubset(kids):
                kids.update(extra)
                changed = True

    cov_attr_tests: dict[str, list[int]] = {}
    for sym, locs in attr_refs.items():
        tset: set[int] = set()
        for rel, ln in locs:
            fmap = line_cov.get(rel)
            if fmap:
                tids = fmap.get(ln)
                if tids:
                    tset.update(tids)
        if tset:
            cov_attr_tests[sym] = sorted(tset)

    return {
        "file_scopes": file_scopes,
        "sym_tests": cov_attr_tests,
        "safe_classes_no_eq": sorted(safe_classes_no_eq),
        "safe_classes_fq": sorted(safe_classes_fq),
        "enum_classes": enum_classes,
        "class_attr_types": all_cattrs_out,
        "dialect_family_children": {p: sorted(kids) for p, kids in dialect_children.items()},
        "unverifiable_tests": sorted(unverifiable_tests),
    }


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        worker_id = int(sys.argv[2])
        out_path = sys.argv[3]
        files = sys.argv[4:]
        worker_trace_files(worker_id, files, out_path)
        return

    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    nodeids = collect_all_nodeids()
    print(f"Collected {len(nodeids)} tests", flush=True)
    test_to_idx = {nid: i for i, nid in enumerate(nodeids)}

    by_file: dict[str, list[str]] = defaultdict(list)
    for nid in nodeids:
        by_file[nid.split("::", 1)[0]].append(nid)

    heavy_order = sorted(
        by_file.keys(),
        key=lambda f: (
            1000 if "test_executor" in f else 900 if "test_parser" in f else 700 if "test_optimizer" in f else len(by_file[f])
        ),
        reverse=True,
    )
    n_workers = min(4, os.cpu_count() or 2)
    buckets: list[list[str]] = [[] for _ in range(n_workers)]
    loads = [0] * n_workers
    for f in heavy_order:
        w = 1000 if "test_executor" in f else 900 if "test_parser" in f else 700 if "test_optimizer" in f else len(by_file[f])
        idx = loads.index(min(loads))
        buckets[idx].append(f)
        loads[idx] += w

    tmp_files = [f"/tmp/cov_w{i}.pkl" for i in range(n_workers)]
    procs = []
    for i, flist in enumerate(buckets):
        if not flist:
            continue
        p = subprocess.Popen(
            [sys.executable, __file__, "--worker", str(i), tmp_files[i], *flist]
        )
        procs.append((i, p))
    for i, p in procs:
        rc = p.wait()
        if rc != 0:
            raise SystemExit(f"Worker {i} failed with {rc}")

    line_cov: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    file_cov: dict[str, set[int]] = defaultdict(set)
    file_reads: dict[str, set[int]] = defaultdict(set)
    dict_key_cov: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    class_descendants: dict[str, set[str]] = defaultdict(set)
    dict_aliases: dict[str, str] = {}
    all_tracked_dict_tags: set[str] = set()

    cls_to_tids: dict[str, list[int]] = defaultdict(list)
    for nid, tidx in test_to_idx.items():
        parts = nid.split("::")
        cls_key = "::".join(parts[:2]) if len(parts) >= 2 else parts[0]
        cls_to_tids[cls_key].append(tidx)

    for i, _ in procs:
        with open(tmp_files[i], "rb") as f:
            payload_w = pickle.load(f)
        wres = payload_w["tests"]
        for nid, info in wres.items():
            tidx = test_to_idx[nid]
            for rel_f, lines in info["lines"].items():
                file_cov[rel_f].add(tidx)
                fmap = line_cov[rel_f]
                for ln in lines:
                    fmap[ln].append(tidx)
            for rf in info["files"]:
                file_reads[rf].add(tidx)
        for cls_key, files_opened in payload_w.get("setup_files", {}).items():
            for rf in files_opened:
                if "tpc-h/" in rf:
                    for tidx in cls_to_tids.get(cls_key, ()):
                        if "tpch" in nodeids[tidx].lower():
                            file_reads[rf].add(tidx)
                elif "tpc-ds/" in rf:
                    for tidx in cls_to_tids.get(cls_key, ()):
                        if "tpcds" in nodeids[tidx].lower():
                            file_reads[rf].add(tidx)
                else:
                    file_reads[rf].update(cls_to_tids.get(cls_key, ()))
        for tag, km in payload_w.get("dict_keys", {}).items():
            for nk, nids in km.items():
                dict_key_cov[tag][nk].update(test_to_idx[n] for n in nids if n in test_to_idx)
        for k, kids in payload_w.get("class_descendants", {}).items():
            class_descendants[k].update(kids)
        dict_aliases.update(payload_w.get("dict_aliases", {}))
        all_tracked_dict_tags.update(payload_w.get("tracked_dict_tags", ()))
        os.remove(tmp_files[i])

    # Expand dict aliases so alias tags point directly to canonical dict_key_cov
    for alias_tag, canon_tag in dict_aliases.items():
        if canon_tag in dict_key_cov and alias_tag not in dict_key_cov:
            dict_key_cov[alias_tag] = {nk: set(tids) for nk, tids in dict_key_cov[canon_tag].items()}

    for rel_f, fmap in line_cov.items():
        for ln in fmap:
            fmap[ln].sort()

    ast_maps = build_ast_maps(nodeids, line_cov)

    payload = {
        "tests": nodeids,
        "line_cov": {f: dict(fmap) for f, fmap in line_cov.items()},
        "file_cov": {f: sorted(tids) for f, tids in file_cov.items()},
        "file_reads": {f: sorted(tids) for f, tids in file_reads.items()},
        "file_scopes": ast_maps["file_scopes"],
        "sym_tests": ast_maps["sym_tests"],
        "dict_key_cov": {
            tag: {nk: sorted(tids) for nk, tids in km.items()}
            for tag, km in dict_key_cov.items()
        },
        "tracked_dict_tags": sorted(set(dict_key_cov) | set(dict_aliases) | all_tracked_dict_tags),
        "class_descendants": {k: sorted(v) for k, v in class_descendants.items()},
        "safe_classes_no_eq": ast_maps["safe_classes_no_eq"],
        "safe_classes_fq": ast_maps["safe_classes_fq"],
        "enum_classes": ast_maps["enum_classes"],
        "class_attr_types": ast_maps["class_attr_types"],
        "dialect_family_children": ast_maps["dialect_family_children"],
        "unverifiable_tests": ast_maps["unverifiable_tests"],
    }
    out_file = MAPS_DIR / "maps.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {out_file} ({out_file.stat().st_size / 1024:.1f} KB)", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build-time analyzer for sqlglot at pinned commit.

Runs the full test suite (across 4 build workers) with:
  1. sys.monitoring per-test LINE coverage on /app/sqlglot
  2. File open() tracking (to map fixtures/*.sql and data files to tests)
  3. Synchronous ProcessPoolExecutor override so multiprocess tests are traced in-process
Then builds static AST + symbol/attribute reference maps for class/module-level edits.
"""

from __future__ import annotations

import ast
import builtins
import concurrent.futures
import gzip
import io
import json
import os
import pickle
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import Future
from pathlib import Path

SQLGLOT_ROOT = Path("/app/sqlglot").resolve()
MAPS_DIR = Path("/app/maps")


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


def worker_trace_files(worker_id: int, files: list[str], out_path: str) -> None:
    os.chdir(SQLGLOT_ROOT)
    sys.path.insert(0, str(SQLGLOT_ROOT))
    os.environ["SKIP_INTEGRATION"] = "0"

    # Force in-process execution of ProcessPoolExecutor so sys.monitoring traces everything
    concurrent.futures.ProcessPoolExecutor = SyncExecutor

    import pytest

    root_prefix = str(SQLGLOT_ROOT) + "/"
    cur_lines: dict[str, set[int]] = defaultdict(set)
    cur_files: set[str] = set()
    active = False

    orig_open = builtins.open
    orig_io_open = io.open

    def tracked_open(file, mode="r", *args, **kwargs):
        if active and isinstance(file, (str, os.PathLike)):
            try:
                p = os.path.abspath(os.fspath(file))
                if p.startswith(root_prefix):
                    cur_files.add(p[len(root_prefix):])
            except Exception:
                pass
        return orig_open(file, mode, *args, **kwargs)

    builtins.open = tracked_open
    io.open = tracked_open
    orig_path_open = Path.open

    def tracked_path_open(self, mode="r", *args, **kwargs):
        if active:
            try:
                p = os.path.abspath(os.fspath(self))
                if p.startswith(root_prefix):
                    cur_files.add(p[len(root_prefix):])
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

    class CovPlugin:
        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_protocol(self, item, nextitem):
            nonlocal active
            cur_lines.clear()
            cur_files.clear()
            sys.monitoring.restart_events()
            active = True
            sys.monitoring.set_events(TOOL_ID, sys.monitoring.events.LINE)
            try:
                yield
            finally:
                sys.monitoring.set_events(TOOL_ID, 0)
                active = False
                results[item.nodeid] = {
                    "lines": {k: sorted(v) for k, v in cur_lines.items()},
                    "files": sorted(cur_files),
                }

    rc = pytest.main(["-q", "--no-header", "--tb=short", *files], plugins=[CovPlugin()])
    sys.monitoring.free_tool_id(TOOL_ID)
    if rc != 0:
        raise RuntimeError(f"worker {worker_id} pytest exit {rc}")

    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(results, f)


def build_ast_maps(test_ids: list[str], line_cov: dict[str, dict[int, list[int]]]) -> dict:
    """Build static AST + attribute/symbol reference maps across sqlglot/."""
    # Map attribute/name -> list of (rel_file, line) where it is referenced (loaded/used)
    attr_refs: dict[str, list[tuple[str, int]]] = defaultdict(list)
    # Map rel_file -> list of AST scopes (kind, name, cls_name, start, end, targets)
    file_scopes: dict[str, list[dict]] = {}

    py_files = sorted((SQLGLOT_ROOT / "sqlglot").rglob("*.py")) + sorted(
        (SQLGLOT_ROOT / "tests").rglob("*.py")
    )
    for path in py_files:
        rel = path.relative_to(SQLGLOT_ROOT).as_posix()
        src = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(src, filename=rel)
        except SyntaxError:
            continue

        # Record attribute/name references in sqlglot/
        if rel.startswith("sqlglot/"):
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                    attr_refs[node.attr].append((rel, node.lineno))
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    attr_refs[node.id].append((rel, node.lineno))

        scopes = []

        def visit_body(stmts, cls_name: str | None, depth: int):
            for st in stmts:
                s_start = st.lineno
                s_end = getattr(st, "end_lineno", st.lineno)
                if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Decorator / def signature lines
                    dec_start = st.decorator_list[0].lineno if st.decorator_list else st.lineno
                    body_start = st.body[0].lineno if st.body else st.lineno
                    scopes.append({
                        "kind": "def_header",
                        "cls": cls_name,
                        "name": st.name,
                        "start": dec_start,
                        "end": max(dec_start, body_start - 1),
                        "fn_start": st.lineno,
                        "fn_end": s_end,
                    })
                    scopes.append({
                        "kind": "func",
                        "cls": cls_name,
                        "name": st.name,
                        "start": body_start,
                        "end": s_end,
                    })
                    visit_body(st.body, cls_name, depth + 1)
                elif isinstance(st, ast.ClassDef):
                    dec_start = st.decorator_list[0].lineno if st.decorator_list else st.lineno
                    body_start = st.body[0].lineno if st.body else st.lineno
                    scopes.append({
                        "kind": "class_header",
                        "cls": st.name,
                        "name": st.name,
                        "start": dec_start,
                        "end": max(dec_start, body_start - 1),
                        "cls_start": st.lineno,
                        "cls_end": s_end,
                    })
                    visit_body(st.body, st.name, depth + 1)
                elif isinstance(st, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.Try)):
                    # Record multi-line header condition / loop header range so short-circuited
                    # multi-line boolean expressions include all tests entering the header!
                    header_end = st.lineno
                    if hasattr(st, "test") and hasattr(st.test, "end_lineno"):
                        header_end = max(header_end, st.test.end_lineno)
                    if header_end > st.lineno:
                        scopes.append({
                            "kind": "stmt_hdr",
                            "cls": cls_name,
                            "name": None,
                            "start": st.lineno,
                            "end": header_end,
                        })
                    for child_list_name in ("body", "orelse", "handlers", "finalbody"):
                        sub = getattr(st, child_list_name, None)
                        if isinstance(sub, list):
                            if child_list_name == "handlers":
                                for h in sub:
                                    visit_body(h.body, cls_name, depth + 1)
                            else:
                                visit_body(sub, cls_name, depth + 1)
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
                        "name": targets[0] if targets else None,
                        "targets": targets,
                        "start": s_start,
                        "end": s_end,
                    })
                elif cls_name is not None and depth == 1:
                    # Class-level attribute/statement
                    targets = []
                    if isinstance(st, ast.Assign):
                        for t in st.targets:
                            if isinstance(t, ast.Name):
                                targets.append(t.id)
                    elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                        targets.append(st.target.id)
                    scopes.append({
                        "kind": "class_attr",
                        "cls": cls_name,
                        "name": targets[0] if targets else None,
                        "targets": targets,
                        "start": s_start,
                        "end": s_end,
                    })
                elif s_end > s_start:
                    # Multi-line simple statement (return/assign/call) inside a function
                    scopes.append({
                        "kind": "stmt_hdr",
                        "cls": cls_name,
                        "name": None,
                        "start": s_start,
                        "end": s_end,
                    })

        visit_body(tree.body, None, 0)
        file_scopes[rel] = scopes

    # Compress attr_refs to unique (rel, line) that actually have dynamic test coverage!
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

    # Group by file and balance across 4 workers
    by_file: dict[str, list[str]] = defaultdict(list)
    for nid in nodeids:
        by_file[nid.split("::", 1)[0]].append(nid)

    # Heavy files separated onto distinct workers
    heavy_order = sorted(by_file.keys(), key=lambda f: (
        1000 if "test_executor" in f else 900 if "test_parser" in f else 700 if "test_optimizer" in f else len(by_file[f])
    ), reverse=True)
    n_workers = min(4, os.cpu_count() or 2)
    buckets: list[list[str]] = [[] for _ in range(n_workers)]
    loads = [0] * n_workers
    for f in heavy_order:
        w = 1000 if "test_executor" in f else 900 if "test_parser" in f else 700 if "test_optimizer" in f else len(by_file[f])
        idx = loads.index(min(loads))
        buckets[idx].append(f)
        loads[idx] += w

    tmp_files = [f"/tmp/cov_w{i}.json.gz" for i in range(n_workers)]
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

    # Merge worker results
    line_cov: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    file_cov: dict[str, set[int]] = defaultdict(set)
    file_reads: dict[str, set[int]] = defaultdict(set)

    for i, _ in procs:
        with gzip.open(tmp_files[i], "rt", encoding="utf-8") as f:
            wres = json.load(f)
        for nid, info in wres.items():
            tidx = test_to_idx[nid]
            for rel_f, lines in info["lines"].items():
                file_cov[rel_f].add(tidx)
                fmap = line_cov[rel_f]
                for ln in lines:
                    fmap[ln].append(tidx)
            for rf in info["files"]:
                file_reads[rf].add(tidx)
        os.remove(tmp_files[i])

    for rel_f, fmap in line_cov.items():
        for ln in fmap:
            fmap[ln].sort()

    ast_maps = build_ast_maps(nodeids, line_cov)

    payload = {
        "tests": nodeids,
        "line_cov": {f: {str(ln): tids for ln, tids in fmap.items()} for f, fmap in line_cov.items()},
        "file_cov": {f: sorted(tids) for f, tids in file_cov.items()},
        "file_reads": {f: sorted(tids) for f, tids in file_reads.items()},
        "file_scopes": ast_maps["file_scopes"],
        "sym_tests": ast_maps["sym_tests"],
    }
    out_file = MAPS_DIR / "maps.json.gz"
    with gzip.open(out_file, "wt", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"Wrote {out_file} ({out_file.stat().st_size / 1024:.1f} KB)", flush=True)


if __name__ == "__main__":
    main()

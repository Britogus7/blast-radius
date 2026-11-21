#!/usr/bin/env python3
"""Blast Radius selector: per-test sys.monitoring coverage + AST symbol/dialect maps
+ optimal 8-worker bin packing."""

from __future__ import annotations

import ast
import gzip
import json
import math
import os
import re
import sys
import time
from pathlib import Path

DATA = Path("/data")
SQLGLOT_ROOT = Path("/app/sqlglot")
MAPS_PATH = Path("/app/maps/maps.json.gz")
WORKER_COST_MS = 2000
MAX_WORKERS = 8


def duration_ms(seconds: float) -> int:
    return math.floor(seconds * 1000 + 0.5)


def pack_optimal(test_nums: list[int], dur_by_num: dict[int, int]) -> list[list[int]]:
    if not test_nums:
        return []
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
        cost = max(loads) + WORKER_COST_MS * w
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_workers = buckets
    return [b for b in best_workers if b]


HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_patch(patch_text: str) -> list[dict]:
    """Parse unified diff into per-file hunks and edited line info."""
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
        elif raw.startswith("+"):
            h["added_lines"].append(raw[1:])
            h["add_anchors"].append(old_line)
            new_line += 1
        elif raw.startswith(" ") or raw == "":
            old_line += 1
            new_line += 1

    return files


def apply_hunks_in_memory(orig_text: str, hunks: list[dict]) -> str | None:
    lines = orig_text.splitlines(keepends=True)
    # Detect newline style
    nl = "\n"
    out: list[str] = []
    pos = 0  # 0-based index into lines
    for h in hunks:
        old_s = max(0, h["old_start"] - 1)
        if h["old_count"] == 0:
            # Pure insertion after old_start (1-based; 0 means before line 1)
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


def is_comment_or_ws_only(orig_text: str | None, hunks: list[dict]) -> bool:
    """True only if Python AST is strictly identical after applying hunks (ignores comments/formatting,
    never ignores indentation shifts or docstrings)."""
    if orig_text is None:
        return False
    patched = apply_hunks_in_memory(orig_text, hunks)
    if patched is None:
        return False
    try:
        return ast.dump(ast.parse(orig_text)) == ast.dump(ast.parse(patched))
    except SyntaxError:
        return False


def is_dialect_file(rel: str) -> str | None:
    for prefix in ("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/"):
        if rel.startswith(prefix) and rel.endswith(".py"):
            stem = rel[len(prefix):-3]
            if stem not in ("__init__", "dialect", "parser", "generator", "typing"):
                return stem
    return None


def build_dialect_family_cov(dialect_cov: dict[str, set[int]]) -> dict[str, set[int]]:
    """Expand each dialect's coverage set to include child dialects importing it
    across dialects/, parsers/, generators/, and typing/."""
    children: dict[str, set[str]] = {d: {d} for d in dialect_cov}
    import_re = re.compile(r"from\s+sqlglot\.(?:dialects|parsers|generators|typing)\.([a-z0-9_]+)\s+import")
    for sub in ("dialects", "parsers", "generators", "typing"):
        sub_dir = SQLGLOT_ROOT / "sqlglot" / sub
        if sub_dir.is_dir():
            for p in sub_dir.glob("*.py"):
                child = p.stem
                if child in ("__init__", "dialect", "parser", "generator", "typing"):
                    continue
                txt = p.read_text("utf-8", errors="ignore")
                for m in import_re.finditer(txt):
                    parent = m.group(1)
                    if parent not in ("dialect", "parser", "generator", "typing"):
                        children.setdefault(parent, {parent}).add(child)
    # Transitive closure
    changed = True
    while changed:
        changed = False
        for parent, kids in list(children.items()):
            extra = set()
            for k in kids:
                extra.update(children.get(k, ()))
            if not extra.issubset(kids):
                kids.update(extra)
                changed = True
    family_cov: dict[str, set[int]] = {}
    for parent, kids in children.items():
        s = set()
        for k in kids:
            s.update(dialect_cov.get(k, ()))
        family_cov[parent] = s
    return family_cov


BASE_CLASSES = {
    "Dialect",
    "Tokenizer",
    "Parser",
    "Generator",
    "TypeAnnotator",
    "Optimizer",
    "Executor",
    "Expr",
    "Expression",
}


def main() -> None:
    t0 = time.monotonic()
    with gzip.open(MAPS_PATH, "rt", encoding="utf-8") as f:
        maps = json.load(f)

    tests_data = json.loads((DATA / "tests.json").read_text("utf-8"))["tests"]
    id_to_num = {t["id"]: t["number"] for t in tests_data}
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
            int(ln): [idx_to_num[i] for i in tids if i in idx_to_num]
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

    dialect_cov: dict[str, set[int]] = {}
    for fpath, tids in file_cov.items():
        d = is_dialect_file(fpath)
        if d:
            dialect_cov.setdefault(d, set()).update(tids)
    dialect_family_cov = build_dialect_family_cov(dialect_cov)

    all_nums = [t["number"] for t in tests_data]

    for patch_path in sorted((DATA / "commits").glob("*.patch")):
        patch_text = patch_path.read_text("utf-8", errors="replace")
        file_edits = parse_patch(patch_text)
        selected: set[int] = set()

        for fe in file_edits:
            rel = fe["file"]
            hunks = fe["hunks"]
            if not hunks:
                continue

            if rel in file_reads:
                selected.update(file_reads[rel])
                if not rel.endswith(".py"):
                    continue

            if not rel.endswith(".py"):
                if rel.startswith("tests/"):
                    selected.update(all_nums)
                continue

            orig_p = SQLGLOT_ROOT / rel
            orig_text = orig_p.read_text("utf-8", errors="replace") if orig_p.is_file() else None
            if is_comment_or_ws_only(orig_text, hunks):
                continue

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
                                    if tests_data[tnum]["id"].endswith(suffix):
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
                                use_cls_sym = bool(cls_name and cls_name not in BASE_CLASSES and cls_name in sym_tests)
                                cand = set()
                                if targets:
                                    for tgt in targets:
                                        t_sym = sym_tests.get(tgt, set())
                                        if t_sym:
                                            if d_stem and d_stem in dialect_family_cov:
                                                cand.update(t_sym & dialect_family_cov[d_stem])
                                            elif use_cls_sym:
                                                cand.update(t_sym & sym_tests[cls_name])
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

        # Pure static+dynamic selection (CPU < 1s; runtime verification disabled to avoid cold import overhead)
        workers = pack_optimal(sorted(selected), dur_by_num)
        print(json.dumps({"commit": patch_path.stem, "workers": workers}))

    t = os.times()
    cpu_s = t.user + t.system + t.children_user + t.children_system
    print(f"done wall={time.monotonic()-t0:.2f}s cpu={cpu_s:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()

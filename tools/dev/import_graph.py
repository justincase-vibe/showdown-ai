"""AST-based import reachability graph for the showdown AI-imitation repo.

Usage:
    python tools/dev/import_graph.py [--json out.json] [--root DIR]

Walks every .py file under the repo root (excluding vendor/dirs listed in
EXCLUDE_DIRS), resolves relative and absolute imports with `ast`, and reports
which modules are NOT reachable from the keep-entry-points defined below.

Entry points (keep-set, per plan Part C):
    - launcher/GUI: all of gui/
    - tools/gate_eval.py, tools/learnability_test.py, tools/bc_collect.py

Test files that import a reachable module are also kept (they are part of the
C4 suite). Node/pokemon-showdown code is outside this graph (not Python).
"""

from __future__ import annotations

import ast
import json
import os
import sys

EXCLUDE_DIRS = {
    "node_modules",
    "pokemon-showdown",
    ".git",
    "__pycache__",
    ".pytest_cache",
    "scratch",
}

# Default keep-entry-points (module-style names, file-relative to repo root).
# gui/__init__.py is empty, so list GUI entry modules explicitly (launcher/GUI is
# part of the keep-set per plan Part C).
ENTRY_POINTS = [
    "gui",
    "gui.launcher",
    "gui.m1_gate_main",
    "gui.trainer_main",
    "gui.bc_main",
    "gui.eval_main",
    "gui.desktop.app",
    # Subprocess entries launched by gui/launcher.py as strings (`python -m X`) —
    # invisible to the AST import graph, so they must be listed here explicitly.
    "net.human_play",
    "control.selfplay_main",
    "league.league_main",
    # Subprocess-based regression pin for the untracked pokemon-showdown fork
    # patch (no repo imports, so the test rule cannot see its target).
    "tools.tests.test_fork_safe_write",
    "tools.gate_eval",
    "tools.learnability_test",
    "tools.bc_collect",
    "tools.dev.import_graph",  # this tool (parked-area audits use it)
]


def iter_py_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def module_name_for(relpath: str) -> list[str]:
    """Map a repo-relative .py file to possible dotted module names."""
    rel = relpath.replace(os.sep, "/")
    parts = rel.split("/")
    names = []
    if parts[-1] == "__init__.py":
        names.append(".".join(parts[:-1]))
    else:
        stem = os.path.splitext(parts[-1])[0]
        names.append(".".join(parts[:-1] + [stem]))
    # also register parent package chain (import of `pkg.sub` resolves via __init__)
    if parts[-1] == "__init__.py":
        parent = parts[:-2]
        if parent:
            names.append(".".join(parent))
    return names


def resolve_imports(root: str, relpath: str, tree: ast.AST) -> list[str]:
    """Return module names imported by this file (absolute + relative resolved)."""
    rel = relpath.replace(os.sep, "/")
    parts = rel.split("/")
    if parts[-1] == "__init__.py":
        pkg_parts = parts[:-1]
        level_base = pkg_parts
    else:
        pkg_parts = parts[:-1]
        level_base = pkg_parts  # for level>0, module's package is its dir

    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                # relative: strip (level-1) trailing parts from package base
                base = level_base[: len(level_base) - (node.level - 1)]
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            if mod:
                out.add(mod)
                for alias in node.names:
                    # `from pkg import sub` may import pkg.sub
                    out.add(f"{mod}.{alias.name}" if mod else alias.name)
    return sorted(out)


def main() -> int:
    root = os.path.abspath(os.getcwd())
    json_out = None
    args = sys.argv[1:]
    if "--json" in args:
        json_out = args[args.index("--json") + 1]

    # 1) collect files + module names
    files: dict[str, str] = {}          # relpath -> abspath
    name_to_file: dict[str, str] = {}   # dotted name -> relpath
    for abspath in iter_py_files(root):
        rel = os.path.relpath(abspath, root).replace(os.sep, "/")
        files[rel] = abspath
        for name in module_name_for(rel):
            name_to_file.setdefault(name, rel)

    # 2) parse imports
    imports: dict[str, list[str]] = {}
    parse_errors = []
    for rel, abspath in files.items():
        try:
            with open(abspath, "r", encoding="utf-8", errors="replace") as fh:
                tree = ast.parse(fh.read())
            imports[rel] = resolve_imports(root, rel, tree)
        except SyntaxError as exc:
            parse_errors.append((rel, str(exc)))
            imports[rel] = []

    # 3) resolve name -> relpath (a name may be a package or module)
    def resolve(name: str) -> str | None:
        return name_to_file.get(name)

    # 4) closure from entry points
    reachable: set[str] = set()
    stack = []
    for ep in ENTRY_POINTS:
        rel = resolve(ep)
        if rel is None:
            # package dir without __init__? try as prefix
            continue
        stack.append(rel)
    while stack:
        rel = stack.pop()
        if rel in reachable:
            continue
        reachable.add(rel)
        for dep in imports.get(rel, []):
            tgt = resolve(dep)
            if tgt and tgt not in reachable:
                stack.append(tgt)
            # also handle `import pkg.sub` where pkg has __init__ importing sub
            for prefix in dep.split(".")[:-1] + [dep]:
                if prefix in name_to_file and prefix not in reachable:
                    # only auto-add if the prefix is a package (has __init__)
                    if name_to_file[prefix].endswith("__init__.py"):
                        stack.append(prefix)

    # 5) package structure + tests, as one fixed point:
    #    - __init__.py / conftest.py of any package that has a kept file are kept
    #      automatically (part of the package, not independently parkable)
    #    - __main__.py of a kept package is kept if all its repo deps are kept
    #    - a test file is kept only if it has at least one repo-internal import
    #      AND all its repo-internal imports are kept. A test importing a
    #      to-be-parked module must be parked with it, or C4 collection breaks.
    kept = set(reachable)

    def is_test(rel: str) -> bool:
        return rel.endswith("conftest.py") or "tests/" in rel or rel.startswith("test_") or "/test_" in rel

    def repo_deps(rel: str) -> set:
        return {resolve(d) for d in imports.get(rel, []) if resolve(d)}

    def pkg_has_kept(pkg: str) -> bool:
        return any(r.startswith(pkg + "/") for r in kept)

    changed = True
    while changed:
        changed = False
        for rel in files:
            if rel in kept:
                continue
            pkg = "/".join(rel.split("/")[:-1])
            if rel.endswith("/__init__.py") or rel.endswith("/conftest.py"):
                if pkg_has_kept(pkg):
                    kept.add(rel)
                    changed = True
            elif rel.endswith("/__main__.py"):
                if pkg_has_kept(pkg):
                    deps = repo_deps(rel)
                    if deps and deps <= kept:
                        kept.add(rel)
                        changed = True
            elif is_test(rel):
                deps = repo_deps(rel)
                if deps and (deps & kept) and (deps <= kept):
                    kept.add(rel)
                    changed = True

    keep = kept
    unreachable = sorted(set(files) - keep)

    # 7) report: who references each unreachable file (if anyone)
    referrers: dict[str, list[str]] = {u: [] for u in unreachable}
    for rel, deps in imports.items():
        for dep in deps:
            tgt = resolve(dep)
            if tgt in referrers:
                referrers[tgt].append(rel)

    result = {
        "root": root,
        "entry_points": ENTRY_POINTS,
        "total_py_files": len(files),
        "reachable": sorted(reachable),
        "kept_package_or_test": sorted(kept - reachable),
        "unreachable": unreachable,
        "unreachable_with_referrers": {
            k: v for k, v in referrers.items() if v
        },
        "parse_errors": parse_errors,
    }

    if json_out:
        with open(json_out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"wrote {json_out}")

    print(f"total .py files : {len(files)}")
    print(f"reachable       : {len(reachable)}")
    print(f"kept pkg/tests  : {len(kept - reachable)}")
    print(f"UNREACHABLE     : {len(unreachable)}")
    for u in unreachable:
        refs = referrers[u]
        tag = f"  (referenced by unreachable: {', '.join(refs)})" if refs else ""
        print(f"  {u}{tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

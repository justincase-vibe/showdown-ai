"""S4.0 corpus report — the S4.3 inputs (PLAN.md §3c).

Battle count, format mix, rating distribution, **human switch-rate** (the S4.3
gate target: "switch rate within a few pts of the human rate"), the legality
rate (the ">=95% of decision points yield a legal action" gate), and the
unknown-tag breakdown (the "0 unknown-tag errors" gate / the bug-catcher output).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .trajectories import BattleTrace


def _median(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    n = len(xs)
    mid = n // 2
    return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2.0


def _tally(values) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        if v is None:
            continue
        out[v] = out.get(v, 0) + 1
    return out


def corpus_report(traces: List[BattleTrace]) -> Dict:
    """Aggregate the per-battle traces into the S4.0 gate report dict."""
    replayed = [t for t in traces if t.skipped is None]
    skipped = [t for t in traces if t.skipped is not None]

    format_mix: Dict[str, int] = {}
    for t in replayed:
        format_mix[t.formatid] = format_mix.get(t.formatid, 0) + 1

    ratings = sorted(t.rating for t in replayed if t.rating and t.rating > 0)
    rating_dist = {
        "count": len(ratings),
        "min": ratings[0] if ratings else None,
        "median": _median(ratings),
        "max": ratings[-1] if ratings else None,
    }

    n_dec = sum(t.n_decisions for t in replayed)
    n_legal = sum(t.n_legal for t in replayed)
    n_switch = sum(1 for t in replayed for d in t.decisions if d.kind == "switch")
    n_move = n_dec - n_switch
    unknown_total = sum(t.unknown_count for t in traces)
    unknown_tags: Dict[str, int] = {}
    for t in traces:
        for tag, c in t.unknown_tags.items():
            unknown_tags[tag] = unknown_tags.get(tag, 0) + c
    both = sum(1 for t in replayed if t.has_both_povs)

    return {
        "n_battles_total": len(traces),
        "n_replayed": len(replayed),
        "n_skipped": len(skipped),
        "skipped_reasons": _tally(t.skipped for t in skipped),
        "format_mix": format_mix,
        "rating_dist": rating_dist,
        "n_decisions": n_dec,
        "n_legal": n_legal,
        "legal_rate": (n_legal / n_dec) if n_dec else 0.0,
        "n_switch_decisions": n_switch,
        "n_move_decisions": n_move,
        "human_switch_rate": (n_switch / n_dec) if n_dec else 0.0,
        "unknown_count": unknown_total,
        "unknown_tags": unknown_tags,
        "both_povs_rate": (both / len(replayed)) if replayed else 0.0,
    }


def format_report(rep: Dict) -> str:
    """Human-readable rendering of :func:`corpus_report` (for the console log)."""
    rd = rep["rating_dist"]
    lines = [
        "-- S4.0 corpus report " + "-" * 30,
        f"battles      : {rep['n_battles_total']} total | "
        f"{rep['n_replayed']} replayed | {rep['n_skipped']} skipped",
        f"format mix   : {rep['format_mix'] or '{}'}",
        f"rating       : min={rd['min']} median={rd['median']} max={rd['max']}"
        f" (n={rd['count']})",
        f"decisions    : {rep['n_decisions']}  "
        f"(moves {rep['n_move_decisions']}, switches {rep['n_switch_decisions']})",
        f"legal rate   : {rep['legal_rate']:.3f}  ({rep['n_legal']}/{rep['n_decisions']})",
        f"switch rate  : {rep['human_switch_rate']:.3f}  <- the S4.3 target",
        f"both-pov rate: {rep['both_povs_rate']:.3f}",
        f"unknown tags : {rep['unknown_count']}"
        + (f"  {rep['unknown_tags']}" if rep["unknown_tags"] else "  (0 - gate passes)"),
    ]
    if rep["n_skipped"]:
        lines.append(f"skipped why  : {rep['skipped_reasons']}")
    return "\n".join(lines)

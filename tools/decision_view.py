"""B2 — decision_view: SEE what the bot considers at one decision point.

The approved spec (M1-gate plan, SMALL ADDITIONS B2):

  > for any state print every legal action with policy %, expected damage,
  > KO flag and value estimate. Run it on the scenario states so we can SEE
  > what the bot is considering.

Input is a poke-env PROTOCOL LOG (the same ``_build_replay_log()`` format the
belief pipeline consumes — e.g. any league replay or a hand-written scenario
log) + the format + side + a checkpoint. The tool:

  1. rebuilds the battle surface the env reads (``build_ctx`` +
     ``build_action_list``) from the log's final state (team preview, active
     mons, fainted, revealed moves — the SAME information limit the BC
     pipeline has: only moves USED in the log are known);
  2. runs the checkpoint net (``PokeNet.load_file`` + ``PokeAgent.act``) for
     the policy distribution, value-head estimate and entropy;
  3. for every move action, runs the SHARED damage calc (``fitness.ko.move_ko``)
     against every known enemy mon (active + bench) for the expected damage
     and the KO flags;
  4. prints one readable table — policy %, damage band, KO targets — plus the
     state's value estimate.

Honest limitations (documented, not hidden):
  * action legality = revealed moves only (the protocol log does not carry
    full movesets — the same constraint as the BC/imitation pipeline);
  * bench-enemy KO flags assume FULL HP (the belief has no bench HP);
  * the value estimate is the state value (the net is single-step —
    per-action Q-values come with the M2b search, which is parked).

Usage (from the repo root, CPU-only):
    python -m tools.decision_view --log runs/replays/xxx.log --fmt gen9ou \
        --side p1 --ckpt checkpoints/some.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ── minimal battle-object surface (mirrors net.tests.test_env_obs) ─────
class _Mon:
    def __init__(self, species: str, moves: Optional[List[str]] = None) -> None:
        self.species = species
        self.base_species = species
        self.moves = {i: _Move(m) for i, m in enumerate(moves or [])}


class _Move:
    def __init__(self, mid: str) -> None:
        self.id = mid


class _LogBattle:
    """The battle surface ``build_ctx`` / ``build_action_list`` read, rebuilt
    from a saved protocol log (poke-env has no from_log — this is the same
    minimal surface ``_FakeBattle`` uses, filled from log state)."""

    def __init__(self, log_text: str, team: "OrderedDict[str, _Mon]",
                 active: Optional[_Mon], avail_moves: List[str],
                 switches: List[_Mon], can_tera: bool,
                 force_switch: bool = False) -> None:
        self._log_text = log_text
        self.team = team
        self.active_pokemon = active
        self.available_moves = [_Move(m) for m in avail_moves]
        self.available_switches = switches
        self.force_switch = force_switch
        self.trapped = False
        self.can_tera = can_tera
        self._wait = False

    def _build_replay_log(self) -> str:  # the belief pipeline's entry point
        return self._log_text


# ── log parsing (pure, testable) ───────────────────────────────────────
def _to_id(name: str) -> str:
    """Showdown id (lowercase alnum) — 'Knock Off' -> 'knockoff'."""
    return "".join(c for c in name.lower() if c.isalnum())


def parse_log_state(log_text: str, side: str = "p1") -> Dict[str, Any]:
    """Extract the decision-relevant state at the END of a protocol log.

    The log is the FORK's native Showdown protocol (the same lines the belief
    ``LogAdapter`` consumes; verified against league replays):

      * team preview : ``|poke|p1|Hawlucha, M|`` (side shape)
      * switch-in    : ``|switch|p1a: Samurott|Samurott-Hisui, F|361/361``
                       (``|drag|``/``|replace|`` = the forced variants)
      * faint        : ``|faint|p2a: Ogerpon``
      * move         : ``|move|p1a: Samurott|Knock Off|p2a: Ting-Lu``

    Returns ``{team: {p1: [...], p2: [...]}, active: {p1, p2},
    fainted: {p1, p2}, used_moves: {p1: [...], p2: [...]}}`` — team preview
    order preserved, active = the last switch-in to the ``pXa`` field slot,
    moves = the (de-duped, ordered) move ids that side USED (the only moves a
    log reveals — the documented information limit)."""
    team: Dict[str, List[str]] = {"p1": [], "p2": []}
    active: Dict[str, Optional[str]] = {"p1": None, "p2": None}
    fainted: Dict[str, set] = {"p1": set(), "p2": set()}
    used_moves: Dict[str, List[str]] = {"p1": [], "p2": []}
    for raw in log_text.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        # parts[0] == '' (before the first pipe); tag = parts[1]
        parts = line.split("|")
        if len(parts) < 2:
            continue
        tag = parts[1]
        if tag == "poke" and len(parts) >= 4:
            key = parts[2].strip()
            if key in ("p1", "p2"):
                # side shape: |poke|p1|Lokix, M|
                name = parts[3].split(",")[0].strip()
            elif len(key) >= 4 and key[:2] in ("p1", "p2") and key[2:3] in "abcdef":
                # slot shape: |poke| p1a 1 |Kyurem |
                key, name = key[:2], parts[3].split(",")[0].strip()
            else:
                continue
            if name:
                team[key].append(name)
        elif tag in ("switch", "drag", "replace") and len(parts) >= 4:
            # |switch|p1a: Samurott|Samurott-Hisui, F|361/361
            slot = parts[2].split(":", 1)[0].strip()
            species = parts[3].split(",")[0].strip()
            if len(slot) >= 3 and slot[:2] in ("p1", "p2") and slot[2] in "abcdef":
                side2 = slot[:2]
                if slot[2] == "a" and species:
                    active[side2] = species
        elif tag == "faint" and len(parts) >= 3:
            # |faint|p2a: Ogerpon
            p2 = parts[2].strip()
            side2 = p2[:2]
            name = p2.split(":", 1)[1].strip() if ":" in p2 else p2
            if side2 in fainted:
                fainted[side2].add(name)
        elif tag == "move" and len(parts) >= 4:
            # |move|p1a: Samurott|Knock Off|p2a: Ting-Lu
            p2 = parts[2].strip()
            side2 = p2[:2]
            mid = _to_id(parts[3])
            if side2 in used_moves and mid and mid not in used_moves[side2]:
                used_moves[side2].append(mid)
    return {
        "team": {k: list(v) for k, v in team.items()},
        "active": active,
        "fainted": fainted,
        "used_moves": used_moves,
    }


def build_log_battle(log_text: str, fmt: str, side: str = "p1",
                     force_switch: bool = False) -> _LogBattle:
    """A :class:`_LogBattle` for ``side`` at the log's final state."""
    st = parse_log_state(log_text)
    my_side = side if side in ("p1", "p2") else "p1"

    team_names: List[str] = st["team"].get(my_side) or []
    team: "OrderedDict[str, _Mon]" = OrderedDict()
    for sp in team_names:
        team[sp] = _Mon(sp)
    active_sp = st["active"].get(my_side)
    revealed = st["used_moves"].get(my_side) or []

    # the active's species may be a FORME (switch details: 'Samurott-Hisui')
    # while the preview carries the base ('Samurott') — match on base form.
    active_name: Optional[str] = None
    if active_sp:
        if active_sp in team:
            active_name = active_sp
        else:
            base = active_sp.split("-")[0]
            for sp in team:
                if sp.split("-")[0] == base:
                    active_name = sp
                    break

    active: Optional[_Mon] = None
    if active_name:
        active = _Mon(active_name, moves=list(revealed))
        team[active_name].moves = active.moves

    fainted = st["fainted"].get(my_side) or set()
    fainted_base = {f.split("-")[0] for f in fainted}
    switches = [_Mon(sp) for sp in team
                if sp != active_name and sp not in fainted
                and sp.split("-")[0] not in fainted_base]

    # tera is a gen-9 mechanic (gen9 + the champions engine, which reuses
    # gen-9 species data); absent everywhere else (OC-8: format-conditional).
    can_tera = fmt.startswith("gen9")
    return _LogBattle(log_text, team, active, list(revealed), switches,
                      can_tera, force_switch)


# ── the table (pure, testable) ─────────────────────────────────────────
def render_table(acts: List[Dict],
                 prob: Optional[List[float]] = None,
                 value: Optional[float] = None,
                 entropy: Optional[float] = None,
                 assessments: Optional[Dict[str, Dict]] = None) -> str:
    """Render the decision table. ``assessments`` maps a move id -> one
    ``fitness.ko.move_ko``-shaped dict (vs the ACTIVE enemy; bench KOs are
    listed in ``ko_bench`` as species names). Pure: fakes are fine."""
    assessments = assessments or {}
    head = f"{'#':>2} {'action':24} {'policy%':>8}"
    if value is not None:
        head += f" | state value (P win): {value:.3f}"
    if entropy is not None:
        head += f"  entropy: {entropy:.3f}"
    lines = [head, "-" * min(len(head), 100)]
    for i, a in enumerate(acts):
        name = a.get("move") or a.get("species") or "?"
        label = f"{a.get('kind', '?')}: {name}"
        p = (prob[i] if prob is not None and i < len(prob) else None)
        pcell = f"{100.0 * p:8.2f}" if p is not None else f"{'—':>8}"
        row = f"{i:>2} {label:24} {pcell}"
        asmt = assessments.get(a.get("move") or "")
        if asmt is not None:
            if asmt.get("unknown_defender") or asmt.get("unknown_attacker"):
                row += "  | dmg: n/a (unknown species)"
            else:
                row += (f"  | dmg: {asmt['min_pct']:.1f}-{asmt['max_pct']:.1f}% "
                        f"exp {asmt['expected_pct']:.1f}%  "
                        f"KO: {'GUARANTEED' if asmt.get('guaranteed_ko') else ('possible' if asmt.get('can_ko') else 'no')}")
            ko_bench = asmt.get("ko_bench") or []
            if ko_bench:
                row += f"  | bench-KO: {', '.join(ko_bench)}"
            if asmt.get("immune"):
                row += "  | IMMUNE"
        lines.append(row)
    return "\n".join(lines)


# ── the CLI ────────────────────────────────────────────────────────────
def cut_log_at_turn(log_text: str, turn: int) -> str:
    """Truncate a protocol log so the decision point is the START of ``turn``
    (all lines up to and including the ``|turn|N`` marker). The state then
    reflects everything BEFORE that turn's moves — the actual decision the
    players faced on that turn. Logs shorter than the marker are returned
    unchanged."""
    out: List[str] = []
    for ln in log_text.splitlines():
        out.append(ln)
        if ln.strip() == f"|turn|{turn}":
            break
    return "\n".join(out)


def _assessments_for(acts: List[Dict], ctx: Dict, fmt: str, gen: int) -> Dict:
    """move id -> move_ko assessment vs the ACTIVE enemy, + bench KO list.

    The bench-enemy calc uses each known bench species (full-HP assumption,
    documented: belief bench placeholders carry ``hp_frac = 0.0`` = UNKNOWN,
    so we substitute an explicit full-HP block — 0.0 would otherwise read as
    'already dead' and every move would 'KO' the bench). One move id is
    assessed ONCE (shared across its tera twin)."""
    from fitness import ko as _ko
    own_slot = (ctx.get("active") or {}).get("p1")
    opp_slot = (ctx.get("active") or {}).get("p2")
    out: Dict[str, Dict] = {}
    if own_slot is None or opp_slot is None:
        return out
    # bench ctx: full-HP assumption for unknown bench blocks (see above)
    opp = dict(ctx.get("opponent") or {})
    bench_ctx = ctx
    for slot, block in list(opp.items()):
        if slot == opp_slot or not block:
            continue
        if block.get("hp_frac") in (0.0, None):
            opp = {**opp, slot: {**block, "hp_frac": 1.0}}
            bench_ctx = {**ctx, "opponent": opp}
    for a in acts:
        mid = a.get("move")
        if not mid or mid in out:
            continue
        base = _ko.move_ko(ctx, own_slot, opp_slot, mid, fmt=fmt, gen=gen)
        bench_ko: List[str] = []
        if not base.get("unknown_defender") and not base.get("unknown_attacker"):
            for slot, block in (ctx.get("opponent") or {}).items():
                if slot == opp_slot or not block:
                    continue
                sp = block.get("name") or block.get("species") or ""
                if not sp:
                    continue
                b = _ko.move_ko(bench_ctx, own_slot, slot, mid, fmt=fmt, gen=gen)
                if b.get("can_ko"):
                    bench_ko.append(sp)
        base = dict(base)
        base["ko_bench"] = bench_ko
        out[mid] = base
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", required=True, help="poke-env protocol log path")
    ap.add_argument("--fmt", default="gen9ou")
    ap.add_argument("--side", default="p1", choices=["p1", "p2"])
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint .pt (policy % + value head); omit for calc-only")
    ap.add_argument("--force-switch", action="store_true",
                    help="treat this as a forced-switch decision point")
    ap.add_argument("--turn", type=int, default=None,
                    help="view the decision at the START of turn N (default: "
                         "the log's final state)")
    args = ap.parse_args(argv)

    log_text = open(args.log, encoding="utf-8").read()
    if args.turn is not None:
        log_text = cut_log_at_turn(log_text, args.turn)
    battle = build_log_battle(log_text, args.fmt, args.side,
                              force_switch=args.force_switch)

    from net.env import build_action_list, build_ctx
    ctx = build_ctx(battle, args.fmt, args.side)
    if ctx is None:
        print("ERROR: build_ctx failed (bad log / fmt?)")
        return 2
    acts = build_action_list(battle)
    if not acts:
        print("No legal actions at this decision point (wait turn or "
              "no revealed moves). Nothing to show.")
        return 1

    prob = value = entropy = None
    if args.ckpt:
        from net.vocab_db import build_vocabs
        from net.model import PokeNet
        from net.agent import PokeAgent
        from net.env import build_obs
        vocabs = build_vocabs()
        net = PokeNet.load_file(args.ckpt, vocabs)
        agent = PokeAgent(net, seed=0)
        obs = build_obs(battle, args.fmt, vocabs, args.side)
        d = agent.act(obs, acts, deterministic=False)
        prob = [float(d.prob[i]) for i in range(len(acts))]
        value = d.value
        entropy = d.entropy

    from legality.formats import engine_key_or_warn
    gen = int(engine_key_or_warn(args.fmt, default=9))  # LOUD on fallback (B1)
    assessments = _assessments_for(acts, ctx, args.fmt, gen)

    own_sp = (ctx.get("own") or {}).get((ctx.get("active") or {}).get("p1"), {})
    opp_sp = (ctx.get("opponent") or {}).get((ctx.get("active") or {}).get("p2"), {})
    print(f"=== decision view: {args.fmt} side={args.side}")
    print(f"own active : {own_sp.get('name') or own_sp.get('species') or '?'} "
          f"(hp {(own_sp.get('hp_frac') if own_sp.get('hp_frac') is not None else 1.0):.0%})")
    print(f"opp active : {opp_sp.get('name') or opp_sp.get('species') or '?'} "
          f"(hp {(opp_sp.get('hp_frac') if opp_sp.get('hp_frac') is not None else 1.0):.0%})")
    if args.ckpt:
        print(f"checkpoint : {args.ckpt}")
    else:
        print("(no checkpoint: policy/value columns empty)")
    print()
    print(render_table(acts, prob=prob, value=value, entropy=entropy,
                       assessments=assessments))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

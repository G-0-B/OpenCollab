#!/usr/bin/env python3
"""Assemble the Analyst cards of the handoff experiment from one shared body.

Every cell of that experiment is a claim that two cards differ in one named
place and nowhere else. While each card carried its own copy of the shared
body, that property was re-checked after the fact rather than guaranteed, and
it had already broken once -- the body split into two generations. Here there
is one copy, in ``configs/handoff-experiment/shared.md``, with two slots:

* ``{{CAPABILITIES}}`` -- the paragraph pair under ``## What you can do``,
  which moves only when the Analyst's tool bundle moves;
* ``{{BLOCK}}`` -- the closing section, which is what most cells vary.

Run this module to render every registered card. It refuses to write a card
whose content would change unless ``--write`` is given, so a drifting card is
reported rather than silently overwritten.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
CARDS_DIR = CONFIGS_DIR / "handoff-experiment"
CAPABILITIES_SLOT = "{{CAPABILITIES}}\n"
BLOCK_SLOT = "{{BLOCK}}\n"
#: Shared evidence rule, unchanged across experiment cells.
CLOSING_LINE = "Do not report a change as verified unless you have the evidence for it.\n"
#: Default analyst tools include the collaboration channel.
DEFAULT_TOOLS = [
    "apply_patch", "bash", "file_read", "file_write", "grep",
    "message_agent", "submit", "team_status",
]

@dataclass(frozen=True)
class CardSet:
    """One instrument's cards: where they live and which seat carries them.

    A card set is not a level of anything -- it is a different team. The
    handoff ladder seats an Analyst, a Coder and a Tester over a complete
    graph; the dual-candidate family seats an Adopter and two Coders over a
    partial one. Two cards from different sets share the closing evidence rule
    and nothing else, so each set has its own body, its own registry and its
    own test module, and no test in one set's module reads the other's files.
    """

    name: str
    directory: Path
    #: Entry seat and its card's filename stem.
    entry_role: str
    body: str = "shared.md"
    #: Additional slots, required in every cell and filled from
    #: ``<slot in lower case>/<value>.md`` using the cell's ``slots`` values.
    extra_slots: tuple[str, ...] = ()
    #: Varying peer cards render ``<peer>.md`` to ``<peer>.<cell>.md``.
    #: Their templates contain all extra slots and neither entry-card slot.
    peers: tuple[str, ...] = ()

HANDOFF_CARDS = CardSet(name="handoff", directory=CARDS_DIR, entry_role="analyst")
DUAL_CARDS = CardSet(
    name="dual-candidate",
    directory=CONFIGS_DIR / "dual-candidate",
    entry_role="adopter",
)
#: DUAL_CARDS roster and blocks appended to each role's Single2 system prompt.
#: The body adds coordination, termination and workspace rules.
S2DUAL_CARDS = CardSet(
    name="s2dual",
    directory=CONFIGS_DIR / "s2dual",
    entry_role="adopter",
)
#: S2DUAL with adopter tool variations reflected in its tool list,
#: capabilities and adoption instructions. The remaining body is shared.
S2TOOLS_CARDS = CardSet(
    name="s2tools",
    directory=CONFIGS_DIR / "s2tools",
    entry_role="adopter",
    extra_slots=("TOOLS", "ADOPTION"),
)
#: Handoff roles with Single2 profiles and varying communication topologies.
#: Every role's card carries TOPOLOGY; only the analyst carries the block.
S2SC_CARDS = CardSet(
    name="s2sc",
    directory=CONFIGS_DIR / "s2sc",
    entry_role="analyst",
    extra_slots=("TOPOLOGY",),
    peers=("coder", "tester"),
)
CARD_SETS: dict[str, CardSet] = {
    s.name: s
    for s in (HANDOFF_CARDS, DUAL_CARDS, S2DUAL_CARDS, S2TOOLS_CARDS, S2SC_CARDS)
}

def slot_marker(slot: str) -> str:
    """The line a slot occupies in a body, e.g. ``{{TOOLS}}``."""
    return "{{" + slot + "}}\n"

@dataclass(frozen=True)
class Variant:
    """One cell: which block it carries, and what bundle it is seated with."""

    name: str
    team_file: str
    block: str
    capabilities: str
    note: str
    tools: tuple[str, ...]
    card_set: CardSet = HANDOFF_CARDS
    #: ``(slot, value)`` for each of the card set's ``extra_slots``.
    slots: tuple[tuple[str, str], ...] = ()

    @property
    def card_name(self) -> str:
        return f"{self.card_set.entry_role}.{self.name}.md"

    @property
    def card_path(self) -> Path:
        return self.card_set.directory / self.card_name

    def peer_card_path(self, peer: str) -> Path:
        """Where the rendered card of one of the set's ``peers`` is filed."""
        return self.card_set.directory / f"{peer}.{self.name}.md"

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def shared_body(card_set: CardSet = HANDOFF_CARDS) -> str:
    """The one copy of everything no cell in this set may vary."""
    body = _read(card_set.directory / card_set.body)
    markers = [CAPABILITIES_SLOT, BLOCK_SLOT, *map(slot_marker, card_set.extra_slots)]
    if any(body.count(marker) != 1 for marker in markers):
        raise ValueError("shared.md must contain each slot exactly once")
    return body

def load_registry(
    card_set: CardSet = HANDOFF_CARDS,
) -> tuple[dict[str, Variant], dict[str, tuple[str, ...]]]:
    """The registered cells of one card set, and the ladders declared over them."""
    raw = yaml.safe_load(_read(card_set.directory / "variants.yaml"))
    variants: dict[str, Variant] = {}
    for name, entry in raw["variants"].items():
        declared = {str(k).upper(): str(v) for k, v in (entry.get("slots") or {}).items()}
        if set(declared) != set(card_set.extra_slots):
            raise ValueError(
                f"{card_set.name}/{name}: slots {sorted(declared)} do not fill "
                f"{sorted(card_set.extra_slots)}"
            )
        variants[name] = Variant(
            name=name,
            team_file=entry["team_file"],
            block=entry["block"],
            capabilities=entry.get("capabilities", "full"),
            note=entry["note"],
            tools=tuple(entry.get("tools", DEFAULT_TOOLS)),
            card_set=card_set,
            slots=tuple((slot, declared[slot]) for slot in card_set.extra_slots),
        )
    ladders = {name: tuple(rungs) for name, rungs in raw.get("ladders", {}).items()}
    return variants, ladders

def render(variant: Variant) -> str:
    """The card this cell's declaration assembles to."""
    directory = variant.card_set.directory
    capabilities = _read(directory / "capabilities" / f"{variant.capabilities}.md")
    block = _read(directory / "blocks" / f"{variant.block}.md")
    card = shared_body(variant.card_set)
    for slot, value in variant.slots:
        card = card.replace(slot_marker(slot), _read(directory / slot.lower() / f"{value}.md"))
    return card.replace(CAPABILITIES_SLOT, capabilities).replace(BLOCK_SLOT, block)

def peer_body(card_set: CardSet, peer: str) -> str:
    """A peer's card before its slots are filled: every extra slot once, and
    neither of the entry card's own two."""
    body = _read(card_set.directory / f"{peer}.md")
    if any(body.count(slot_marker(slot)) != 1 for slot in card_set.extra_slots):
        raise ValueError(f"{peer}.md must contain each of {card_set.extra_slots} exactly once")
    if CAPABILITIES_SLOT in body or BLOCK_SLOT in body:
        raise ValueError(f"{peer}.md carries a slot only the entry card has")
    return body

def render_peer(variant: Variant, peer: str) -> str:
    """The card of one of the set's ``peers`` for this cell."""
    card = peer_body(variant.card_set, peer)
    directory = variant.card_set.directory
    for slot, value in variant.slots:
        card = card.replace(slot_marker(slot), _read(directory / slot.lower() / f"{value}.md"))
    return card

def rendered_cards(variant: Variant) -> list[tuple[Path, str]]:
    """Every card this cell's declaration renders: the entry card, then each peer's."""
    cards = [(variant.card_path, render(variant))]
    for peer in variant.card_set.peers:
        cards.append((variant.peer_card_path(peer), render_peer(variant, peer)))
    return cards

def main(argv: list[str] | None = None) -> int:
    write = "--write" in (argv if argv is not None else sys.argv[1:])
    variants: dict[str, Variant] = {}
    for card_set in CARD_SETS.values():
        registered, _ = load_registry(card_set)
        for name, variant in registered.items():
            key = f"{card_set.name}/{name}"
            if key in variants:
                raise ValueError(f"two card sets register {key!r}")
            variants[key] = variant
    drifted = checked = 0
    for name in sorted(variants):
        for path, built in rendered_cards(variants[name]):
            checked += 1
            current = _read(path) if path.exists() else None
            if current == built:
                continue
            drifted += 1
            verb = "rewrote" if write else "DRIFTED"
            print(f"{verb}: {path.name}")
            if write:
                path.write_text(built, encoding="utf-8")
    if drifted and not write:
        print(f"\n{drifted} card(s) differ from their assembly. Re-run with --write to update.")
        return 1
    print(f"{checked} card(s) checked, {drifted} rewritten." if write
          else f"{checked} card(s) match their assembly.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

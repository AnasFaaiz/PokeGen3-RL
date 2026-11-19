"""Scripted catch rule. Pure logic: no sockets, no game access.

Goal: keep a pool of low-level Pokemon on hand for when the party has
outleveled the area it hunts in, so wild battles keep teaching something.

The env asks `decide()` before the agent's first move of a wild battle.
Keeping the rule behind this one function means a learned action can replace
it later without touching the env: the same context goes in, a throw/no-throw
comes out.

The rule: when every party member is at least GAP_TO_SWAP levels above the
highest wild level recently seen on this map, and the PC box holds fewer than
CANDIDATES_WANTED Pokemon at or just above that wild level, catch wild
Pokemon (skipping species already in the box), MAX_BALLS_PER_MON balls each.
"""
from collections import defaultdict, deque
from dataclasses import dataclass, field

GAP_TO_SWAP = 12            # party min level must be this far above the area's wild max
CANDIDATE_MARGIN = 5        # a box mon counts as a useful low-level if level <= area_max + this
CANDIDATES_WANTED = 6       # stop catching once the box holds this many useful low-levels
MAX_BALLS_PER_MON = 4
AREA_WINDOW = 20            # wild encounters remembered per map
AREA_MIN_OBS = 10           # encounters needed before the area max is trusted
SPECIES_FIRST_GEN4_SLOT = 277


def national_dex(species):
    """Gen 3 internal species id -> national dex number, or None if not a real species."""
    if 1 <= species <= 251:
        return species
    if SPECIES_FIRST_GEN4_SLOT <= species <= 411:
        return species - 25
    return None


def parse_owned(dex_line):
    """DEX:<104 hex chars> -> bytes of the owned flags, or None. Optional signal."""
    if not dex_line.startswith("DEX:") or dex_line.startswith("DEX:ERR"):
        return None
    try:
        return bytes.fromhex(dex_line[4:])
    except ValueError:
        return None


def is_owned(owned, species):
    n = national_dex(species)
    if owned is None or n is None:
        return None
    return bool(owned[(n - 1) // 8] >> ((n - 1) % 8) & 1)


@dataclass
class CatchContext:
    enemy_species: int
    enemy_level: int
    party_levels: list
    box_levels: list | None         # None = box could not be read
    box_species: set = field(default_factory=set)
    area_max: int | None = None     # highest wild level recently seen on this map
    trainer: bool = False
    balls: int = 0


@dataclass
class CatchPolicy:
    gap_to_swap: int = GAP_TO_SWAP
    candidate_margin: int = CANDIDATE_MARGIN
    candidates_wanted: int = CANDIDATES_WANTED
    max_balls_per_mon: int = MAX_BALLS_PER_MON
    enabled: bool = True
    thrown: int = 0
    blocked: bool = False           # a throw was refused or failed this battle
    last_reason: str = field(default="", repr=False)
    _areas: dict = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=AREA_WINDOW)),
                         repr=False)

    def start_battle(self):
        self.thrown = 0
        self.blocked = False
        self.last_reason = ""

    def record_throw(self, ok=True):
        if ok:
            self.thrown += 1
        else:
            self.blocked = True

    def observe_area(self, map_key, wild_level):
        self._areas[map_key].append(wild_level)

    def area_max(self, map_key):
        seen = self._areas.get(map_key)
        if not seen or len(seen) < AREA_MIN_OBS:
            return None
        # 90th percentile, so one rare high-level encounter doesn't inflate the area's level.
        return sorted(seen)[int(0.9 * (len(seen) - 1))]

    def overleveled(self, party_levels, area_max):
        return bool(party_levels) and area_max is not None and min(party_levels) >= area_max + self.gap_to_swap

    def decide(self, ctx):
        """True to throw a ball now. `last_reason` says why, for the log."""
        def no(why):
            self.last_reason = why
            return False

        if not self.enabled:
            return no("disabled")
        if self.blocked:
            return no("blocked")
        if ctx.trainer:
            return no("trainer")
        if ctx.balls <= 0:
            return no("no balls")
        if ctx.enemy_level <= 0 or national_dex(ctx.enemy_species) is None:
            return no("unknown enemy")
        if ctx.area_max is None:
            return no("area level unknown")
        if not self.overleveled(ctx.party_levels, ctx.area_max):
            return no("party not overleveled")
        if ctx.box_levels is None:
            return no("box unknown")
        ceiling = ctx.area_max + self.candidate_margin
        if sum(1 for lv in ctx.box_levels if lv <= ceiling) >= self.candidates_wanted:
            return no("box has enough low levels")
        if ctx.enemy_level > ceiling:
            return no("enemy too high")
        if ctx.enemy_species in ctx.box_species:
            return no("species already in box")
        if self.thrown >= self.max_balls_per_mon:
            return no("ball budget spent")
        self.last_reason = "low-level stock"
        return True


if __name__ == "__main__":
    p = CatchPolicy()
    for lv in [5, 6, 7, 5, 8, 6, 7, 5, 6, 7]:
        p.observe_area("0.16", lv)
    assert p.area_max("0.16") == 7 and p.area_max("9.9") is None
    base = dict(enemy_species=288, enemy_level=6, party_levels=[22, 21, 20], box_levels=[40, 35],
                box_species={25}, area_max=8, trainer=False, balls=10)
    assert p.decide(CatchContext(**base)) and p.last_reason == "low-level stock"
    assert not p.decide(CatchContext(**{**base, "party_levels": [22, 15, 20]}))      # 15 < 8 + 12
    assert p.last_reason == "party not overleveled"
    assert not p.decide(CatchContext(**{**base, "box_levels": [5, 6, 7, 8, 9, 10]}))  # enough stock
    assert p.last_reason == "box has enough low levels"
    assert not p.decide(CatchContext(**{**base, "enemy_level": 20}))
    assert not p.decide(CatchContext(**{**base, "enemy_species": 25}))
    assert p.last_reason == "species already in box"
    assert not p.decide(CatchContext(**{**base, "trainer": True}))
    assert not p.decide(CatchContext(**{**base, "balls": 0}))
    assert not p.decide(CatchContext(**{**base, "box_levels": None}))
    assert not p.decide(CatchContext(**{**base, "area_max": None}))
    for _ in range(MAX_BALLS_PER_MON):
        p.record_throw()
    assert not p.decide(CatchContext(**base)) and p.last_reason == "ball budget spent"
    p.start_battle()
    p.record_throw(ok=False)
    assert not p.decide(CatchContext(**base)) and p.last_reason == "blocked"
    assert national_dex(277) == 252 and national_dex(252) is None and national_dex(0) is None
    assert is_owned(bytes([1]) + bytes(51), 1) is True and is_owned(bytes(52), 1) is False
    print("ok")

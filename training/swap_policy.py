"""Scripted PC-box swap rule. Pure logic: no sockets, no game access.

Every party slot that has outleveled the wild area is exchanged for a box
Pokemon near the area's level, all in one pass (lead first, then highest
level). The whole party is checked, not just slot 0: in-battle switching and
faints reshuffle the order, so an overleveled bench mon would otherwise come
back to the front. The strong Pokemon goes to the box, nothing is lost.

The first pass after launch is a start-up fill: it uses the in-band ceiling
(area_max + ABOVE_MARGIN) instead of GAP_TO_SWAP, so the party starts the run
already near the area's level. Later passes use GAP_TO_SWAP, so a slot is only
replaced once it has grown well past the area.

Picks are weighted-random (new species and rarely-used species favoured) so the
party's species mix rotates over a long run.

Kept behind `plan()` so a learned action can replace it later.
"""
import random
from collections import Counter
from dataclasses import dataclass, field

GAP_TO_SWAP = 12        # a slot's level must be this far above the area's wild max
BELOW_MARGIN = 2        # a candidate may be this far below the area max
ABOVE_MARGIN = 5        # ...or this far above it
COOLDOWN_BATTLES = 3    # minimum battles between swap passes
RETRY_BATTLES = 5       # after finding nothing to swap in, wait this long before re-checking


@dataclass
class BoxEntry:
    key: str            # "box.slot", as the bridge reports it
    species: int
    level: int


@dataclass
class SwapPolicy:
    gap_to_swap: int = GAP_TO_SWAP
    below_margin: int = BELOW_MARGIN
    above_margin: int = ABOVE_MARGIN
    cooldown: int = COOLDOWN_BATTLES
    retry: int = RETRY_BATTLES
    enabled: bool = True
    last_swap_episode: int = -10**9
    next_check_episode: int = -10**9
    last_reason: str = field(default="", repr=False)
    seed: int = None                                   # set for reproducible picks
    species_uses: Counter = field(default_factory=Counter, repr=False)   # times each species was swapped in
    rng: random.Random = field(default=None, repr=False)

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    def weight(self, entry, in_party):
        """Sampling weight: species new to the party x3, and less-used species favoured, so the
        party's species mix rotates instead of settling on the same few box mons."""
        w = 3.0 if entry.species not in in_party else 1.0
        return w / (1 + self.species_uses[entry.species])

    def due(self, episode):
        """Cheap gate: False while on cooldown or backing off after a fruitless check."""
        return (self.enabled and episode - self.last_swap_episode >= self.cooldown
                and episode >= self.next_check_episode)

    def overleveled_slots(self, party_levels, area_max, startup=False):
        if area_max is None:
            return []
        limit = area_max + (self.above_margin + 1 if startup else self.gap_to_swap)
        return [i for i, lv in enumerate(party_levels) if lv >= limit]

    def candidates(self, area_max, box):
        lo, hi = area_max - self.below_margin, area_max + self.above_margin
        return [e for e in box if lo <= e.level <= hi]

    def plan(self, episode, party_levels, area_max, box, party_species=(), startup=False):
        """Return [(party_slot, BoxEntry), ...] for one pass (possibly empty).
        `last_reason` says why when empty. Each box entry is used at most once."""
        def no(why, back_off=False):
            self.last_reason = why
            if back_off:
                self.next_check_episode = episode + self.retry
            return []

        if not self.enabled:
            return no("disabled")
        if episode - self.last_swap_episode < self.cooldown:
            return no("cooldown")
        if area_max is None:
            return no("area level unknown")
        over = self.overleveled_slots(party_levels, area_max, startup)
        if not over:
            return no("party not overleveled")
        if not box:
            return no("box unknown or empty", back_off=True)
        pool = self.candidates(area_max, box)
        if not pool:
            return no("no box mon near area level", back_off=True)
        over.sort(key=lambda i: (i != 0, -party_levels[i]))      # lead first, then highest level
        in_party = set(party_species)
        out = []
        for slot in over:
            if not pool:
                break
            # Weighted random: favour species the party lacks and ones swapped in least so far.
            best = self.rng.choices(pool, weights=[self.weight(e, in_party) for e in pool])[0]
            pool.remove(best)
            in_party.add(best.species)
            self.species_uses[best.species] += 1
            out.append((slot, best))
        self.last_swap_episode = episode
        self.last_reason = "startup fill" if startup else "overleveled"
        return out


if __name__ == "__main__":
    p = SwapPolicy()
    box = [BoxEntry("0.0", 25, 40), BoxEntry("0.1", 288, 7), BoxEntry("0.2", 263, 9), BoxEntry("0.3", 300, 20),
           BoxEntry("0.4", 288, 8), BoxEntry("0.5", 300, 10)]
    assert p.due(100)
    r = p.plan(100, [40, 30, 28, 10], 8, box)                                  # slots 0,1,2 overleveled
    assert [s for s, _ in r] == [0, 1, 2] and len({e.key for _, e in r}) == 3 and p.last_reason == "overleveled"
    assert all(6 <= e.level <= 13 for _, e in r)
    assert p.plan(101, [40], 8, box) == [] and p.last_reason == "cooldown" and not p.due(101)
    p.last_swap_episode = -10**9
    r = p.plan(1, [10, 12, 15], 8, box)                                        # 15 < 8+12
    assert r == [] and p.last_reason == "party not overleveled"
    r = p.plan(1, [10, 12, 20], 8, box)                                        # slot 2 at exactly area+12
    assert [s for s, _ in r] == [2]
    p.last_swap_episode = -10**9
    r = p.plan(1, [10, 14, 20], 8, box, startup=True)                          # start-up: ceiling 13 -> slots 1 and 2
    assert [s for s, _ in r] == [2, 1] and p.last_reason == "startup fill"     # highest level first after the lead
    p.last_swap_episode = -10**9
    assert p.plan(1, [40], None, box) == [] and p.last_reason == "area level unknown"
    assert p.plan(1, [40], 8, []) == [] and p.last_reason == "box unknown or empty"
    assert not p.due(2) and p.due(1 + RETRY_BATTLES)                           # backed off, then due again
    p.next_check_episode = -10**9
    assert p.plan(1, [40], 8, [BoxEntry("0.0", 25, 40)]) == [] and p.last_reason == "no box mon near area level"
    p.next_check_episode = -10**9
    r = p.plan(1, [40, 40, 40, 40], 8, box)                                    # more slots than candidates
    assert len(r) == 4 and len({e.key for _, e in r}) == 4
    p.last_swap_episode = -10**9
    r = p.plan(1, [40], 8, box, party_species=[263, 288, 300])                 # all in-band species in party
    assert len(r) == 1
    p.last_swap_episode = -10**9
    for seed in range(20):                                                     # new species is favoured, not forced
        q = SwapPolicy(seed=seed)
        assert q.plan(1, [40], 8, box, party_species=[263])[0][1].level in range(6, 14)
    seen = Counter()                                                           # long run: no species starves or dominates
    q = SwapPolicy(seed=0)
    for ep in range(300):
        q.last_swap_episode = -10**9
        seen[q.plan(ep, [40], 8, box, party_species=[263])[0][1].species] += 1
    assert set(seen) == {288, 263, 300} and min(seen.values()) > 60, seen
    print("ok")

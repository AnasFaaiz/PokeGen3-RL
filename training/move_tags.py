"""Hand-curated effect tags for moves: heal / status-inflicting / setup (stat boost).

emerald_moves.json has power, type, accuracy and priority but not what a status move DOES, so the
agent couldn't tell Recover from Growl. Names are matched after lower-casing and dropping
punctuation ("Will-O-Wisp", "Will O Wisp"); a name missing from the move list is ignored.
"""
import json
import re
from pathlib import Path

_MOVES = Path(__file__).resolve().parent.parent / "games" / "emerald" / "game_data" / "emerald_moves.json"

_HEAL = ["Recover", "Softboiled", "Rest", "Moonlight", "Synthesis", "Morning Sun", "Milk Drink", "Slack Off",
         "Dream Eater", "Giga Drain", "Mega Drain", "Absorb", "Leech Life", "Wish", "Swallow",
         "Ingrain"]
_STATUS = ["Thunder Wave", "Toxic", "Sleep Powder", "Spore", "Hypnosis", "Will-O-Wisp", "Confuse Ray",
           "Stun Spore", "Poison Powder", "Glare", "Lovely Kiss", "Sing", "Supersonic", "Swagger",
           "Sweet Kiss", "Poison Gas", "Leech Seed", "Yawn", "Teeter Dance", "Flatter", "Grass Whistle",
           "Attract", "Mean Look", "Spider Web", "Block"]
_SETUP = ["Swords Dance", "Calm Mind", "Dragon Dance", "Bulk Up", "Agility", "Growth", "Howl", "Meditate",
          "Harden", "Defense Curl", "Barrier", "Acid Armor", "Amnesia", "Iron Defense", "Cosmic Power",
          "Belly Drum", "Double Team", "Minimize", "Sharpen", "Charge", "Withdraw",
          "Tail Glow", "Curse", "Focus Energy"]


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _ids(names):
    wanted = {_norm(n) for n in names}
    with open(_MOVES) as f:
        moves = json.load(f)
    return frozenset(int(k) for k, v in moves.items() if _norm(v.get("name", "")) in wanted)


HEAL_MOVES, STATUS_MOVES, SETUP_MOVES = _ids(_HEAL), _ids(_STATUS), _ids(_SETUP)

# Moves a mon cannot forget in the game (the teaching rule must not overwrite them either).
_HM = ["Cut", "Fly", "Surf", "Strength", "Flash", "Rock Smash", "Waterfall", "Dive"]
HM_MOVES = _ids(_HM)

# How much a move's raw power is worth in a real fight (1.0 = takes effect normally). Two-turn and recharge moves give
# the enemy a free turn, conditional ones are dead weight most of the time, and 0.10 means never worth teaching
# (self-KO, no effect, copy moves).
_QUIRKS = {
    0.55: ["Solar Beam", "Sky Attack", "Razor Wind", "Skull Bash", "Fly", "Dig", "Dive", "Bounce", "Hyper Beam"],
    0.30: ["Focus Punch", "Dream Eater", "Snore", "Sleep Talk", "Spit Up", "Swallow", "Last Resort", "Endeavor",
           "Counter", "Mirror Coat", "Bide", "Future Sight", "Doom Desire"],
    0.70: ["Superpower", "Overheat", "Psycho Boost", "Leaf Storm", "Draco Meteor", "Revenge"],   # strong, with a drawback
    0.10: ["Splash", "Struggle", "Celebrate", "Teleport", "Transform", "Mimic", "Sketch", "Nightmare", "Spite",
           "Explosion", "Selfdestruct", "Memento"],      # the mon would faint itself: never worth teaching
}
QUIRK_MULT = {}
for _mult, _names in _QUIRKS.items():
    for _mid in _ids(_names):
        QUIRK_MULT[_mid] = _mult


def quirk(move):
    """Multiplier for a move's usefulness (1.0 for an ordinary move)."""
    return QUIRK_MULT.get(move, 1.0)


def tags(move):
    """(heal, status, setup) as 0.0/1.0 flags for a move id (0 / unknown -> all zero)."""
    return (float(move in HEAL_MOVES), float(move in STATUS_MOVES), float(move in SETUP_MOVES))


if __name__ == "__main__":
    by_name = {}
    with open(_MOVES) as f:
        for k, v in json.load(f).items():
            by_name[_norm(v["name"])] = int(k)
    assert tags(by_name["recover"]) == (1.0, 0.0, 0.0)
    assert tags(by_name["softboiled"]) == (1.0, 0.0, 0.0)               # "Soft Boiled" in the move list
    assert tags(by_name["thunderwave"]) == (0.0, 1.0, 0.0)
    assert tags(by_name["willowisp"]) == (0.0, 1.0, 0.0)                # "Will O Wisp" in the move list
    assert tags(by_name["swordsdance"]) == (0.0, 0.0, 1.0)
    assert tags(by_name["tackle"]) == (0.0, 0.0, 0.0) and tags(0) == (0.0, 0.0, 0.0) and tags(99999) == (0.0, 0.0, 0.0)
    assert tags(by_name["growl"]) == (0.0, 0.0, 0.0)                    # a debuff, not tagged
    assert HEAL_MOVES and STATUS_MOVES and SETUP_MOVES
    assert len(HM_MOVES) == 8 and by_name["surf"] in HM_MOVES and by_name["tackle"] not in HM_MOVES
    assert quirk(by_name["hyperbeam"]) == 0.55 and quirk(by_name["explosion"]) == 0.10 and quirk(by_name["tackle"]) == 1.0
    assert quirk(by_name["focuspunch"]) == 0.30 and quirk(0) == 1.0
    # every curated name should exist in this ROM's move list; report any that don't
    known = set(by_name)
    missing = [n for n in _HEAL + _STATUS + _SETUP if _norm(n) not in known]
    print("not in move list (ignored):", missing)
    print("ok")

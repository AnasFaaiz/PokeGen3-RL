"""Trainer lookups from the decomp-derived tables (games/emerald/game_data/trainers.json and
trainer_objects.json, built by games/emerald/data_gen/generate_trainers.py), plus the bridge reads
that need a live address. Everything degrades to "unknown" instead of raising: a missing table,
an unknown id or a bridge error (TRAINER:ERR:no_address until the RAM address is confirmed live) just
gives no information.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

_DATA = Path(__file__).resolve().parent.parent / "games" / "emerald" / "game_data"


@lru_cache(maxsize=1)
def _trainers():
    try:
        return json.loads((_DATA / "trainers.json").read_text())
    except (OSError, ValueError):
        return {}


@lru_cache(maxsize=1)
def _objects():
    try:
        return json.loads((_DATA / "trainer_objects.json").read_text())
    except (OSError, ValueError):
        return {}


def get(trainer_id):
    """The trainer's table entry, or None for an unknown / invalid id."""
    if trainer_id is None or trainer_id < 1:
        return None
    return _trainers().get(str(trainer_id))


@lru_cache(maxsize=2048)
def party_size(trainer_id):
    t = get(trainer_id)
    return len(t["party"]) if t else 0


@lru_cache(maxsize=2048)
def max_level(trainer_id):
    t = get(trainer_id)
    return max((m["level"] for m in t["party"]), default=0) if t else 0


@lru_cache(maxsize=2048)
def is_double(trainer_id):
    t = get(trainer_id)
    return bool(t and t["double"])


@lru_cache(maxsize=2048)
def is_rematch(trainer_id):
    """True for the 2nd and later versions of a trainer (TRAINER_TONY_2, ...)."""
    t = get(trainer_id)
    m = re.search(r"_(\d+)$", t["const"]) if t else None
    return bool(m and int(m.group(1)) > 1)


def parse_trainer_reply(line):
    """TRAINER:<id> -> int, or None on TRAINER:ERR:... / garbage."""
    if not line.startswith("TRAINER:") or line.startswith("TRAINER:ERR"):
        return None
    try:
        return int(line.split(":")[1])
    except (IndexError, ValueError):
        return None


def read_trainer_id(request):
    """Opponent trainer id from the bridge, or -1 when unavailable."""
    try:
        tid = parse_trainer_reply(request("TRAINER"))
    except OSError:
        return -1
    return tid if tid is not None and tid > 0 else -1


def trainer_defeated(request, trainer_id):
    """True/False from the TRAINERFLAGS bridge command, or None when it can't tell."""
    try:
        line = request(f"TRAINERFLAGS:{int(trainer_id)}")
    except (OSError, ValueError):
        return None
    if not line.startswith("TRAINERFLAGS:") or line.startswith("TRAINERFLAGS:ERR"):
        return None
    try:
        return line.split(":")[2] == "1"
    except IndexError:
        return None


def trainers_on_map(map_key):
    """Trainer objects on a map ("group.num"), only those whose id is known."""
    return [o for o in _objects().get(map_key, []) if o.get("id")]


def undefeated_on_map(request, map_key):
    """Trainers on the map that have not been beaten yet (a trainer whose flag can't be read counts as
    undefeated, so a navigation hook errs towards trying). Navigation does not use this yet."""
    return [o for o in trainers_on_map(map_key) if trainer_defeated(request, o["id"]) is not True]


if __name__ == "__main__":
    t = get(265)                                              # ROXANNE_1 in this decomp
    assert t and t["name"] == "ROXANNE" and len(t["party"]) == 3 and max_level(265) == 15
    assert party_size(0) == 0 and max_level(-1) == 0 and get(None) is None and get(99999) is None
    assert not is_double(265) and not is_rematch(265)
    assert any(is_rematch(int(i)) for i in list(_trainers())[:900]), "expected some _2 rematch trainers"
    assert parse_trainer_reply("TRAINER:318") == 318
    assert parse_trainer_reply("TRAINER:ERR:no_address") is None and parse_trainer_reply("junk") is None
    assert read_trainer_id(lambda l: "TRAINER:ERR:no_address") == -1 and read_trainer_id(lambda l: "TRAINER:615") == 615
    def boom(line):
        raise OSError("socket closed")
    assert read_trainer_id(boom) == -1 and trainer_defeated(boom, 5) is None
    assert trainer_defeated(lambda l: "TRAINERFLAGS:5:1", 5) is True
    assert trainer_defeated(lambda l: "TRAINERFLAGS:5:0", 5) is False
    assert trainer_defeated(lambda l: "TRAINERFLAGS:ERR:bad_id", 5) is None
    route102 = trainers_on_map("0.16")
    assert isinstance(route102, list)
    assert len(undefeated_on_map(lambda l: "TRAINERFLAGS:ERR:x", "0.17")) == len(trainers_on_map("0.17"))
    assert undefeated_on_map(lambda l: "TRAINERFLAGS:1:1", "0.17") == []
    print("ok")

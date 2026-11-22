"""Offline checks for the Gen 3 mon structure mirror: python training/test_mon_struct.py"""
import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mon_struct as ms

# Every one of the 24 substruct orderings round-trips, keeps a valid checksum and reads back the same moves.
for pid in range(24):
    raw = ms.build_mon(0x10000 * 7 + pid, 0xA1B2C3D4, 277, [33, 45, 71, 0], [35, 40, 25, 0], exp=1000, pp_bonuses=0b00000001)
    assert ms.checksum_ok(raw), pid
    p, species, moves, pp, bonus, ok = ms.read_info(raw)
    assert (p, species, moves, pp, bonus, ok) == (0x70000 + pid, 277, [33, 45, 71, 0], [35, 40, 25, 0], 1, True), pid
assert len(set(ms.ORDERS)) == 24 and all(sorted(o) == [0, 1, 2, 3] for o in ms.ORDERS)

# Editing moves changes only the attacks block: the other substructs, pid, ot id, header bytes and species are untouched.
rng = random.Random(1)
for _ in range(200):
    pid, otid = rng.getrandbits(32), rng.getrandbits(32)
    raw = ms.build_mon(pid, otid, rng.randrange(1, 412), [rng.randrange(1, 355) for _ in range(4)],
                       [rng.randrange(1, 40) for _ in range(4)], exp=rng.randrange(0, 10 ** 6), pp_bonuses=rng.randrange(256))
    new_moves = [rng.randrange(0, 355) for _ in range(4)]
    new_pp = [rng.randrange(0, 60) for _ in range(4)]
    out = ms.set_moves(raw, new_moves, new_pp)
    assert ms.checksum_ok(out) and len(out) == 80
    p0, sp0, _, _, b0, _ = ms.read_info(raw)
    p1, sp1, moves1, pp1, b1, ok1 = ms.read_info(out)
    assert (p1, sp1, moves1, pp1, b1, ok1) == (p0, sp0, new_moves, new_pp, b0, True)
    before, after = ms.decrypt(raw), ms.decrypt(out)
    assert all(before[t] == after[t] for t in (0, 2, 3))                       # growth / evs / misc blocks unchanged
    assert raw[:0x1C] == out[:0x1C]                                            # header (pid, otid, nickname...) unchanged

# PP-Up bits: set_moves can replace the Growth byte (checksum stays valid, nothing else changes); clear_bonus clears one slot.
raw = ms.build_mon(0x00BADA55, 0x1234ABCD, 77, [1, 2, 3, 4], [10, 10, 10, 10], exp=999, pp_bonuses=0b11100100)
out = ms.set_moves(raw, [9, 2, 3, 4], [20, 10, 10, 10], pp_bonuses=ms.clear_bonus(0b11100100, 0))
assert ms.read_info(out)[4] == 0b11100100 & ~3 and ms.checksum_ok(out)
assert ms.read_info(ms.set_moves(raw, [9, 2, 3, 4], [20, 10, 10, 10]))[4] == 0b11100100          # None leaves it alone
assert ms.clear_bonus(0b11100100, 2) == 0b11000100 and ms.clear_bonus(0b11111111, 3) == 0b00111111
b0, b1 = ms.decrypt(raw), ms.decrypt(out)
assert b0[0][:2] == b1[0][:2] and b0[2] == b1[2] and b0[3] == b1[3]                                # species/item/exp untouched

# A corrupted struct is detected (this is what makes the Lua refuse to write).
raw = ms.build_mon(12345, 67890, 100, [1, 2, 3, 4], [10, 10, 10, 10])
bad = bytearray(raw)
bad[0x25] ^= 0xFF
assert ms.checksum_ok(raw) and not ms.checksum_ok(bytes(bad))
assert ms.read_info(bytes(bad))[5] is False

# Max PP: base plus 20% per PP Up in that slot (2 bits per slot).
assert ms.max_pp(35, 0, 0) == 35 and ms.max_pp(35, 0b11, 0) == 35 + 35 * 3 // 5 and ms.max_pp(20, 0b01000000, 3) == 24
assert ms.max_pp(10, 0b00000100, 0) == 10 and ms.max_pp(10, 0b00000100, 1) == 12
print("ok")

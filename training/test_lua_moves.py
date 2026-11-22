"""Runs the real games/emerald/inventory.lua MONMOVES/SETMOVES code under Lua 5.4 against a mocked emulator and
checks it against the Python mirror (mon_struct.py): python training/test_lua_moves.py

Not a substitute for the live check (addresses and the real game's structs), but it proves the Lua logic itself:
decrypt, checksum, pid guard, in-battle refusal, dry run, write + verify, and restore on a failed verify.
"""
import os
import random
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mon_struct as ms

LUA = shutil.which("lua5.4") or shutil.which("lua")
if not LUA:
    print("skipped: no lua interpreter")
    sys.exit(0)
INV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "games", "emerald", "inventory.lua")

PARTY_BASE, STRIDE, COUNT_ADDR, LEVEL_OFF = 0x1000, 0x64, 0x0F00, 0x54

HARNESS = r'''
local mem = {}
local function rd(a) return mem[a] or 0 end
local corrupt_left = ARG_CORRUPT and 3 or 0       -- corrupt only the first three 32-bit writes (the new data), not the restore
local emu_mt = {}
emu = setmetatable({}, {__index = {
  read8 = function(_, a) return rd(a) end,
  read16 = function(_, a) return rd(a) | (rd(a + 1) << 8) end,
  read32 = function(_, a) return rd(a) | (rd(a + 1) << 8) | (rd(a + 2) << 16) | (rd(a + 3) << 24) end,
  write8 = function(_, a, v) mem[a] = v & 0xFF end,
  write16 = function(_, a, v) mem[a] = v & 0xFF; mem[a + 1] = (v >> 8) & 0xFF end,
  write32 = function(_, a, v)
    if corrupt_left > 0 then corrupt_left = corrupt_left - 1; v = v ~ 1 end
    mem[a] = v & 0xFF; mem[a + 1] = (v >> 8) & 0xFF; mem[a + 2] = (v >> 16) & 0xFF; mem[a + 3] = (v >> 24) & 0xFF
  end,
}})
local hex = ARG_HEX
for i = 1, #hex // 2 do mem[ARG_BASE + i - 1] = tonumber(hex:sub(2 * i - 1, 2 * i), 16) end
local block_orders = {
  {0,1,2,3},{0,1,3,2},{0,2,1,3},{0,2,3,1},{0,3,1,2},{0,3,2,1},{1,0,2,3},{1,0,3,2},{1,2,0,3},{1,2,3,0},{1,3,0,2},{1,3,2,0},
  {2,0,1,3},{2,0,3,1},{2,1,0,3},{2,1,3,0},{2,3,0,1},{2,3,1,0},{3,0,1,2},{3,0,2,1},{3,1,0,2},{3,1,2,0},{3,2,0,1},{3,2,1,0}}
local in_battle = ARG_INBATTLE
local D = {
  CONFIG = {INV = {PARTY_COUNT_ADDR = ARG_COUNT_ADDR, PARTY_LEVEL_OFFSET = ARG_LEVEL_OFF}, OW = {}},
  partySlotBase = function(slot) return ARG_BASE + slot * ARG_STRIDE end,
  block_orders = block_orders,
  isInBattle = function() return in_battle end,
}
local Inv = dofile("ARG_INV")(D)
emu:write8(ARG_COUNT_ADDR, ARG_NMON)
for _, cmd in ipairs(ARG_CMDS) do
  if cmd:sub(1, 9) == "MONMOVES:" then print(Inv.monMovesLine(tonumber(cmd:sub(10))))
  else print(Inv.setMovesLine(cmd)) end
end
local out = {}
for i = 0, ARG_NMON * ARG_STRIDE - 1 do out[#out + 1] = string.format("%02x", rd(ARG_BASE + i)) end
print("MEM:" .. table.concat(out))
'''


def run(raws, cmds, in_battle=False, corrupt=False):
    body = b"".join(r + bytes(ms_pad) for r in raws)
    code = (HARNESS.replace("ARG_CORRUPT", "true" if corrupt else "false")
            .replace("ARG_HEX", '"%s"' % body.hex()).replace("ARG_BASE", str(PARTY_BASE))
            .replace("ARG_INBATTLE", "true" if in_battle else "false").replace("ARG_COUNT_ADDR", str(COUNT_ADDR))
            .replace("ARG_LEVEL_OFF", str(LEVEL_OFF)).replace("ARG_STRIDE", str(STRIDE)).replace("ARG_NMON", str(len(raws)))
            .replace("ARG_INV", INV.replace("\\", "/"))
            .replace("ARG_CMDS", "{" + ",".join('"%s"' % c for c in cmds) + "}"))
    with tempfile.NamedTemporaryFile("w", suffix=".lua", delete=False) as f:
        f.write(code)
    try:
        res = subprocess.run([LUA, f.name], capture_output=True, text=True, timeout=30)
    finally:
        os.unlink(f.name)
    assert res.returncode == 0, res.stderr
    lines = res.stdout.strip().splitlines()
    mem = bytes.fromhex(lines[-1][4:])
    return lines[:-1], [mem[i * STRIDE:i * STRIDE + 80] for i in range(len(raws))]


ms_pad = bytes(STRIDE - 80)             # the party struct is 100 bytes: 80 of box data + the party tail
rng = random.Random(7)
raws = [ms.build_mon(rng.getrandbits(32), rng.getrandbits(32), rng.randrange(1, 412), [33, 45, 0, 0], [35, 40, 0, 0],
                     exp=rng.randrange(10 ** 5), pp_bonuses=0b00000100) for _ in range(6)]
# make sure several different substruct orderings are exercised
for i, r in enumerate(raws):
    pid = (rng.getrandbits(32) // 24) * 24 + [0, 5, 6, 11, 17, 23][i]
    raws[i] = ms.build_mon(pid, rng.getrandbits(32), rng.randrange(1, 412), [33, 45, 0, 0], [35, 40, 0, 0], exp=1234, pp_bonuses=4)

# MONMOVES agrees with the Python decoder for every slot.
out, _ = run(raws, [f"MONMOVES:{i}" for i in range(6)] + ["MONMOVES:6"])
for i, line in enumerate(out[:6]):
    p, sp, moves, pp, bonus, ok = ms.read_info(raws[i])
    f = line.split(":")
    assert f[0] == "MONMOVES" and int(f[1]) == p and int(f[2]) == sp and [int(x) for x in f[4].split(",")] == moves
    assert [int(x) for x in f[5].split(",")] == pp and int(f[6]) == bonus and f[7] == "1", (i, line)
assert out[6] == "MONMOVES:ERR:bad_slot"

# SETMOVES writes exactly what the Python mirror writes, and leaves the other mons alone.
pid2 = ms.read_info(raws[2])[0]
cmd = f"SETMOVES:2:{pid2}:55:45:71:98:25:40:20:30"
out, mem = run(raws, [cmd])
assert out == ["SETMOVES:OK"], out
expect = ms.set_moves(raws[2], [55, 45, 71, 98], [25, 40, 20, 30])
assert mem[2] == expect and ms.checksum_ok(mem[2]), "lua write differs from the python mirror"
assert all(mem[i] == raws[i] for i in (0, 1, 3, 4, 5))
assert ms.read_info(mem[2])[2:5] == ([55, 45, 71, 98], [25, 40, 20, 30], 4)

# Dry run changes nothing; refusals change nothing.
out, mem = run(raws, [cmd + ":dry"])
assert out == ["SETMOVES:DRY:ok"] and mem == raws
out, mem = run(raws, [f"SETMOVES:2:{pid2 ^ 1}:55:45:71:98:25:40:20:30"])
assert out == ["SETMOVES:ERR:pid_mismatch"] and mem == raws                     # a reordered party must not be written
out, mem = run(raws, [cmd], in_battle=True)
assert out == ["SETMOVES:ERR:in_battle"] and mem == raws
for bad in ("SETMOVES:2:%d:55:45:71:98:25:40:20" % pid2, "SETMOVES:2:%d:999:45:71:98:25:40:20:30" % pid2,
            "SETMOVES:2:%d:55:45:71:98:25:40:20:999" % pid2, "SETMOVES:9:%d:55:45:71:98:25:40:20:30" % pid2):
    out, mem = run(raws, [bad])
    assert out[0].startswith("SETMOVES:ERR:") and mem == raws, (bad, out)
broken = list(raws)
bad_raw = bytearray(broken[2])
bad_raw[0x25] ^= 0xFF                                                                # corrupt a data byte: checksum no longer matches
broken[2] = bytes(bad_raw)
out, mem = run(broken, [cmd])
assert out == ["SETMOVES:ERR:bad_checksum"] and mem == broken
out, _ = run(broken, ["MONMOVES:2"])
assert out[0].endswith(":0")                                                         # checksum_ok reported as 0

# Optional PP-Up byte: written only when given, and only the Growth byte changes; dry run and bad values change nothing.
bon_cmd = f"SETMOVES:2:{pid2}:55:45:71:98:25:40:20:30:0"
out, mem = run(raws, [bon_cmd])
assert out == ["SETMOVES:OK"], out
expect = ms.set_moves(raws[2], [55, 45, 71, 98], [25, 40, 20, 30], pp_bonuses=0)
assert mem[2] == expect and ms.checksum_ok(mem[2]) and ms.read_info(mem[2])[4] == 0
assert ms.read_info(raws[2])[4] == 4 and all(mem[i] == raws[i] for i in (0, 1, 3, 4, 5))
out, mem = run(raws, [bon_cmd + ":dry"])
assert out == ["SETMOVES:DRY:ok"] and mem == raws
out, mem = run(raws, [f"SETMOVES:2:{pid2}:55:45:71:98:25:40:20:30:300"])
assert out == ["SETMOVES:ERR:bad_args"] and mem == raws
# Without the byte the existing PP Ups are left exactly as they were (checked above: ppBonuses stayed 4).
# A failed read-back puts the original bytes back (the mock corrupts the first three writes to force it).
out, mem = run(raws, [cmd], corrupt=True)
assert out == ["SETMOVES:ERR:verify_failed"] and mem == raws, out
out, mem = run(raws, [bon_cmd], corrupt=True)                               # the PP-Up byte is rolled back too
assert out == ["SETMOVES:ERR:verify_failed"] and mem == raws and ms.read_info(mem[2])[4] == 4, out
print("ok")

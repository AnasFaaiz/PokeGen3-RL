"""Gen 3 Pokemon data structure (first 80 bytes): decrypt, edit moves, re-encrypt, fix the checksum.

This is the algorithm games/emerald/inventory.lua implements for MONMOVES / SETMOVES, kept here in Python so it can
be unit-tested without the emulator (test_mon_struct.py) and so the Lua can be compared against it.

Layout (little endian):  0 pid u32 | 4 otId u32 | 8 nickname[10] | 18 language | 19 flags | 20 otName[7] | 27 markings |
28 checksum u16 | 30 u16 | 32..79 four 12-byte substructs in an order set by pid % 24, each u32 word XORed with
key = pid ^ otId.  Substructs: 0 Growth (species u16, item u16, exp u32, ppBonuses u8, friendship u8, u16),
1 Attacks (moves u16 x4, pp u8 x4), 2 EVs/condition, 3 Misc.  checksum = sum of the 24 u16 words of the decrypted 48 bytes.
"""
import struct

# ORDERS[pid % 24][position] = substruct type (0 growth, 1 attacks, 2 evs, 3 misc): the same table as battle_bridge.lua.
ORDERS = [
    (0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (0, 2, 3, 1), (0, 3, 1, 2), (0, 3, 2, 1),
    (1, 0, 2, 3), (1, 0, 3, 2), (1, 2, 0, 3), (1, 2, 3, 0), (1, 3, 0, 2), (1, 3, 2, 0),
    (2, 0, 1, 3), (2, 0, 3, 1), (2, 1, 0, 3), (2, 1, 3, 0), (2, 3, 0, 1), (2, 3, 1, 0),
    (3, 0, 1, 2), (3, 0, 2, 1), (3, 1, 0, 2), (3, 1, 2, 0), (3, 2, 0, 1), (3, 2, 1, 0),
]
DATA_OFFSET, DATA_SIZE, CHECKSUM_OFFSET = 0x20, 48, 0x1C


def _key(raw):
    pid, otid = struct.unpack_from("<II", raw, 0)
    return pid ^ otid


def decrypt(raw):
    """-> {type: [3 u32 words]} for the four substructs of an 80-byte mon."""
    key = _key(raw)
    order = ORDERS[struct.unpack_from("<I", raw, 0)[0] % 24]
    words = struct.unpack_from("<12I", raw, DATA_OFFSET)
    return {order[pos]: [w ^ key for w in words[pos * 3:pos * 3 + 3]] for pos in range(4)}


def checksum(blocks):
    """Sum of the 24 u16 halves of the decrypted substructs, modulo 2**16."""
    total = 0
    for t in range(4):
        for w in blocks[t]:
            total += (w & 0xFFFF) + (w >> 16)
    return total & 0xFFFF


def checksum_ok(raw):
    return struct.unpack_from("<H", raw, CHECKSUM_OFFSET)[0] == checksum(decrypt(raw))


def encrypt(raw, blocks):
    """New 80-byte mon with `blocks` re-encrypted into place and the checksum rewritten."""
    key = _key(raw)
    order = ORDERS[struct.unpack_from("<I", raw, 0)[0] % 24]
    out = bytearray(raw)
    words = []
    for pos in range(4):
        words += [w ^ key for w in blocks[order[pos]]]
    struct.pack_into("<12I", out, DATA_OFFSET, *words)
    struct.pack_into("<H", out, CHECKSUM_OFFSET, checksum(blocks))
    return bytes(out)


def read_info(raw):
    """(pid, species, moves[4], pp[4], ppBonuses, checksum_ok)."""
    b = decrypt(raw)
    g, a = b[0], b[1]
    moves = [a[0] & 0xFFFF, a[0] >> 16, a[1] & 0xFFFF, a[1] >> 16]
    pp = [a[2] & 255, (a[2] >> 8) & 255, (a[2] >> 16) & 255, a[2] >> 24]
    return struct.unpack_from("<I", raw, 0)[0], g[0] & 0xFFFF, moves, pp, g[2] & 255, checksum_ok(raw)


def clear_bonus(bonuses, slot):
    """PP-Up byte with the two bits of `slot` cleared (what the game does when a move in that slot is replaced)."""
    return bonuses & ~(3 << (2 * slot)) & 0xFF


def max_pp(base_pp, bonuses, slot):
    """Full PP of a move: base PP plus 20% per PP Up applied to that slot (2 bits each)."""
    return base_pp + base_pp * ((bonuses >> (2 * slot)) & 3) // 5


def set_moves(raw, moves, pp, pp_bonuses=None):
    """New raw struct with the four moves and their PP replaced (everything else, including the checksum rule, intact).
    `pp_bonuses` (0-255) also replaces the Growth substruct's PP-Up byte; None leaves it alone."""
    blocks = decrypt(raw)
    if pp_bonuses is not None:
        blocks[0] = [blocks[0][0], blocks[0][1], (blocks[0][2] & ~0xFF) | (pp_bonuses & 0xFF)]
    a = list(blocks[1])
    a[0] = (moves[0] & 0xFFFF) | ((moves[1] & 0xFFFF) << 16)
    a[1] = (moves[2] & 0xFFFF) | ((moves[3] & 0xFFFF) << 16)
    a[2] = (pp[0] & 255) | ((pp[1] & 255) << 8) | ((pp[2] & 255) << 16) | ((pp[3] & 255) << 24)
    blocks[1] = a
    return encrypt(raw, blocks)


def build_mon(pid, otid, species, moves, pp, exp=0, pp_bonuses=0):
    """A synthetic, valid 80-byte mon (used by the tests)."""
    raw = bytearray(80)
    struct.pack_into("<II", raw, 0, pid, otid)
    blocks = {
        0: [species | (0 << 16), exp, pp_bonuses],
        1: [(moves[0] & 0xFFFF) | ((moves[1] & 0xFFFF) << 16), (moves[2] & 0xFFFF) | ((moves[3] & 0xFFFF) << 16),
            pp[0] | (pp[1] << 8) | (pp[2] << 16) | (pp[3] << 24)],
        2: [0x01020304, 0x0506, 0],
        3: [0xDEADBEEF, 0x12345678, 0x0BADF00D],
    }
    return encrypt(bytes(raw), blocks)

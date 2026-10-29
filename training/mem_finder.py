"""
mem_finder.py -- automatic memory-address finder for GBA games.

Runs next to training (or while you play) and needs no manual searching.
battle_bridge.lua serves raw RAM and a set of labelled signals on port 8889;
this tool records RAM snapshots whenever the game enters a new state, then
works out which addresses track which signals and which addresses tell the
game's screens/states apart.

    python tools/mem_finder.py watch                   # record + re-analyze every 10 min
    python tools/mem_finder.py record                  # record only
    python tools/mem_finder.py analyze analysis/mem_sessions/<session>

Optional, for readable reports:
    --sym  path/to/pokeemerald.sym     names each address (gBattleMons+0x28)
    --pret path/to/pokeemerald         names battle-script opcodes

What it finds, per run (report.md + report.json in the session folder):
  1. Signal mirrors: every address whose value always equals a known signal
     (HP, cursors, outcome...). Finds backup copies and confirms addresses.
  2. State discriminators: addresses whose value is fixed within each game
     state and differs between states (screen, in-battle, phase, prompt).
     This is how flags like gPartyMenu.action or gMain.inBattle show up.
  3. Unidentified waits: battle-script opcodes the game sat on for a long
     time with nothing handling them -- prompts the bridge doesn't know yet.
  4. Config check: each emerald_config.lua address tested against labels
     that do NOT come from that same address.

Limits: results are candidates ranked by evidence, not guarantees. More
varied play means stronger results. Data behind pointers (heap structs)
shows up as the pointer's target area, which moves between sessions.
Encrypted party data won't mirror plain values.
"""
import argparse
import bisect
import json
import os
import re
import socket
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

DEFAULT_PORT = 8889
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "..", "analysis", "mem_sessions")

CATEGORICAL = ["cb2", "inBattle", "phase", "ctrl0", "opcode", "partyAction"]
NUMERIC = {  # signal -> width in bytes
    "pHP": 2, "eHP": 2, "actionCursor": 1, "moveCursor": 1,
    "partySlot": 1, "outcome": 1, "weather": 2,
}
# Which config address each signal is read from, so a mirror result can say
# "this is the source" instead of pretending it found something new.
SIGNAL_SOURCE = {
    "pHP": 0x020240AC, "eHP": 0x02024104, "actionCursor": 0x020244AC,
    "moveCursor": 0x020244B0, "partySlot": 0x0203CED1, "outcome": 0x0202433A,
    "weather": 0x020243CC,
}


# ============================================================ probe client

class Probe:
    def __init__(self, host="127.0.0.1", port=DEFAULT_PORT, timeout=10.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.f = self.sock.makefile("rb")
        self.sock.sendall(b"REGIONS\n")
        line = self.f.readline().decode().strip()
        self.regions = []
        for part in line.split(","):
            name, start, size = part.split(":")
            self.regions.append({"name": name, "start": int(start, 16), "size": int(size)})

    def signals(self):
        self.sock.sendall(b"SIGNALS\n")
        return parse_signals(self.f.readline().decode().strip())

    def dump(self):
        self.sock.sendall(b"DUMP\n")
        header = self.f.readline().decode().strip()
        _, total, sig = header.split(" ", 2)
        total = int(total)
        data = self.f.read(total)
        if len(data) != total:
            raise ConnectionError(f"short dump: {len(data)}/{total}")
        return np.frombuffer(data, dtype=np.uint8).copy(), parse_signals(sig)

    def close(self):
        self.sock.close()


def parse_signals(line):
    out = {}
    for kv in line.split(";"):
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        try:
            out[k] = int(v, 0)
        except ValueError:
            out[k] = v
    return out


# ================================================================ recorder

def state_key(s):
    """Coarse game state. A new key means a new screen/situation worth a snapshot."""
    in_battle = s.get("inBattle", 0)
    return (s.get("cb2"), in_battle, s.get("phase"),
            s.get("ctrl0") if in_battle else 0,
            s.get("opcode") if in_battle else -1)


class Recorder:
    def __init__(self, out_dir, port=DEFAULT_PORT, per_state=25, max_dumps=1500,
                 periodic_s=20.0, poll_s=0.1, chunk=50):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(out_dir, stamp)
        os.makedirs(self.dir, exist_ok=True)
        self.port = port
        self.per_state, self.max_dumps = per_state, max_dumps
        self.periodic_s, self.poll_s, self.chunk = periodic_s, poll_s, chunk
        self.buf, self.buf_labels = [], []
        self.n_dumps, self.n_chunks = 0, 0
        self.per_key = Counter()
        self.signal_log = open(os.path.join(self.dir, "signals.jsonl"), "a")
        self.probe = None

    def connect(self):
        while self.probe is None:
            try:
                self.probe = Probe(port=self.port)
                with open(os.path.join(self.dir, "regions.json"), "w") as f:
                    json.dump(self.probe.regions, f)
                print(f"[finder] connected, recording to {self.dir}")
            except OSError:
                print("[finder] waiting for battle_bridge.lua on port", self.port)
                time.sleep(3)

    def _flush(self):
        if not self.buf:
            return
        path = os.path.join(self.dir, f"chunk_{self.n_chunks:04d}.npz")
        np.savez_compressed(path, dumps=np.stack(self.buf))
        with open(os.path.join(self.dir, f"chunk_{self.n_chunks:04d}.json"), "w") as f:
            json.dump(self.buf_labels, f)
        self.n_chunks += 1
        self.buf, self.buf_labels = [], []

    def _take(self, reason):
        dump, sig = self.probe.dump()
        sig["reason"] = reason
        sig["t"] = time.time()
        self.buf.append(dump)
        self.buf_labels.append(sig)
        self.n_dumps += 1
        self.per_key[state_key(sig)] += 1
        if len(self.buf) >= self.chunk:
            self._flush()

    def run(self, duration_s=None, on_tick=None):
        self.connect()
        last_key, stable, last_periodic = None, 0, 0.0
        t0 = time.time()
        try:
            while self.n_dumps < self.max_dumps:
                if duration_s and time.time() - t0 > duration_s:
                    break
                try:
                    s = self.probe.signals()
                except (OSError, ValueError, ConnectionError):
                    self.probe = None
                    self.connect()
                    continue
                s["t"] = time.time()
                self.signal_log.write(json.dumps(s) + "\n")
                key = state_key(s)
                stable = stable + 1 if key == last_key else 0
                last_key = key
                try:
                    # New state, settled for 2 polls, and not over-sampled yet.
                    if stable == 2 and self.per_key[key] < self.per_state:
                        self._take("new_state")
                    elif time.time() - last_periodic > self.periodic_s:
                        self._take("periodic")
                        last_periodic = time.time()
                except (OSError, ValueError, ConnectionError) as e:
                    print("[finder] dump failed:", e)
                    self.probe = None
                    self.connect()
                if on_tick:
                    on_tick(self)
                time.sleep(self.poll_s)
        finally:
            self._flush()
            self.signal_log.close()
            print(f"[finder] stopped: {self.n_dumps} snapshots in {self.dir}")
        return self.dir


# ================================================================ analysis

def load_session(session_dir):
    with open(os.path.join(session_dir, "regions.json")) as f:
        regions = json.load(f)
    chunks = sorted(p for p in os.listdir(session_dir) if p.startswith("chunk_") and p.endswith(".npz"))
    for c in chunks:
        dumps = np.load(os.path.join(session_dir, c))["dumps"]
        with open(os.path.join(session_dir, c.replace(".npz", ".json"))) as f:
            labels = json.load(f)
        yield regions, dumps, labels


def offset_to_addr(regions, off):
    base = 0
    for r in regions:
        if off < base + r["size"]:
            return r["start"] + (off - base)
        base += r["size"]
    raise ValueError(off)


def addr_to_offset(regions, addr):
    base = 0
    for r in regions:
        if r["start"] <= addr < r["start"] + r["size"]:
            return base + addr - r["start"]
        base += r["size"]
    return None


class SymbolTable:
    def __init__(self, path=None):
        self.addrs, self.entries = [], []
        if not path or not os.path.exists(path):
            return
        rows = []
        for line in open(path):
            p = line.split()
            if len(p) < 4:
                continue
            try:
                a, size = int(p[0], 16), int(p[2], 16)
            except ValueError:
                continue
            if 0x02000000 <= a < 0x04000000:
                rows.append((a, size, p[3]))
        rows.sort()
        self.addrs = [r[0] for r in rows]
        self.entries = rows

    def name(self, addr):
        if not self.addrs:
            return ""
        i = bisect.bisect_right(self.addrs, addr) - 1
        if i < 0:
            return ""
        a, size, name = self.entries[i]
        if addr < a + max(size, 1):
            return name if addr == a else f"{name}+0x{addr - a:X}"
        return ""


def opcode_names(pret_dir):
    if not pret_dir:
        return {}
    path = os.path.join(pret_dir, "src", "battle_script_commands.c")
    if not os.path.exists(path):
        return {}
    names = {}
    for line in open(path):
        m = re.search(r"=\s*Cmd_(\w+),?\s*//\s*0x([0-9A-Fa-f]+)", line)
        if m:
            names[int(m.group(2), 16)] = m.group(1)
    return names


def analyze(session_dir, sym_path=None, pret_dir=None, min_support=3, top=15):
    syms = SymbolTable(sym_path)
    opnames = opcode_names(pret_dir)

    regions, n = None, 0
    disc = {k: {} for k in CATEGORICAL}        # label -> value -> [first, consistent, count]
    mirror_hits = {k: None for k in NUMERIC}   # matches per address
    mirror_n = Counter()
    mirror_values = defaultdict(set)
    all_labels = []

    for regions, dumps, labels in load_session(session_dir):
        for dump, lab in zip(dumps, labels):
            n += 1
            all_labels.append(lab)
            for k in CATEGORICAL:
                v = lab.get(k)
                if v is None:
                    continue
                slot = disc[k].get(v)
                if slot is None:
                    disc[k][v] = [dump.copy(), np.ones(dump.shape, bool), 1]
                else:
                    slot[1] &= (dump == slot[0])
                    slot[2] += 1
            u16 = dump[: len(dump) // 2 * 2].view("<u2")
            for k, width in NUMERIC.items():
                v = lab.get(k)
                if v is None or v < 0:
                    continue
                mirror_values[k].add(v)
                mirror_n[k] += 1
                hits = (dump == (v & 0xFF)) if width == 1 else np.repeat(u16 == (v & 0xFFFF), 2)[: len(dump)]
                if mirror_hits[k] is None:
                    mirror_hits[k] = hits.astype(np.int32)
                else:
                    mirror_hits[k] += hits

    if n == 0:
        raise SystemExit(f"No snapshots in {session_dir} yet.")

    def where(off):
        a = offset_to_addr(regions, off)
        return a, syms.name(a)

    report = {"session": session_dir, "snapshots": n, "mirrors": {}, "discriminators": {},
              "unidentified_waits": [], "config_check": []}

    # 1. mirrors
    for k, hits in mirror_hits.items():
        if hits is None or len(mirror_values[k]) < 3:
            report["mirrors"][k] = {"note": f"signal took {len(mirror_values[k])} distinct values; need 3+"}
            continue
        width = NUMERIC[k]
        need = int(0.98 * mirror_n[k])
        offs = np.flatnonzero(hits >= need)
        if width == 2:
            offs = offs[offs % 2 == 0]
        cands = []
        for off in offs[:200]:
            a, name = where(int(off))
            cands.append({"addr": f"0x{a:08X}", "symbol": name, "match": round(hits[off] / mirror_n[k], 3),
                          "source": a == SIGNAL_SOURCE.get(k)})
        report["mirrors"][k] = {"samples": mirror_n[k], "distinct_values": len(mirror_values[k]),
                                "candidates": cands[:top], "total_candidates": int(len(offs))}

    # 2. discriminators
    for k, classes in disc.items():
        usable = {v: c for v, c in classes.items() if c[2] >= min_support}
        if len(usable) < 2:
            report["discriminators"][k] = {"note": f"{len(usable)} states with {min_support}+ snapshots; need 2+"}
            continue
        consistent = np.logical_and.reduce([c[1] for c in usable.values()])
        idx = np.flatnonzero(consistent)
        firsts = np.stack([c[0] for c in usable.values()])
        if len(idx):
            cols = np.sort(firsts[:, idx], axis=0)          # distinct values per address, vectorized
            n_distinct = 1 + np.count_nonzero(np.diff(cols, axis=0), axis=0)
        if len(idx) == 0:
            report["discriminators"][k] = {"states": len(usable), "candidates": []}
            continue
        order = np.argsort(-n_distinct, kind="stable")
        cands = []
        for j in order:
            if n_distinct[j] < 2:
                break
            off = int(idx[j])
            a, name = where(off)
            mapping = {(f"0x{v:08X}" if isinstance(v, int) and v > 0xFFFF else str(v)): int(c[0][off])
                       for v, c in usable.items()}
            cands.append({"addr": f"0x{a:08X}", "symbol": name, "states_separated": int(n_distinct[j]),
                          "of_states": len(usable), "values": mapping})
            if len(cands) >= top:
                break
        report["discriminators"][k] = {"states": len(usable),
                                       "perfect": int(np.sum(n_distinct == len(usable))),
                                       "candidates": cands}

    # 3. unidentified waits: long stretches in WAITING on one opcode
    sig_path = os.path.join(session_dir, "signals.jsonl")
    if os.path.exists(sig_path):
        stretches = defaultdict(list)
        cur, start, last_t = None, None, None
        for line in open(sig_path):
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (s.get("opcode"), s.get("ctrl0")) if s.get("phase") == "WAITING" else None
            if key != cur:
                if cur is not None and last_t is not None:
                    stretches[cur].append(last_t - start)
                cur, start = key, s.get("t")
            last_t = s.get("t")
        for (op, ctrl), lens in stretches.items():
            longest = max(lens)
            if longest >= 5.0:
                report["unidentified_waits"].append({
                    "opcode": f"0x{op:02X}" if isinstance(op, int) and op >= 0 else str(op),
                    "command": opnames.get(op, ""), "ctrl0": f"0x{ctrl:08X}" if isinstance(ctrl, int) else ctrl,
                    "times_seen": len(lens), "longest_s": round(longest, 1)})
        report["unidentified_waits"].sort(key=lambda r: -r["longest_s"])

    # 4. config check, each against labels from a different address
    report["config_check"] = config_check(regions, session_dir, all_labels)

    with open(os.path.join(session_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    write_markdown(report, os.path.join(session_dir, "report.md"))
    return report


def config_check(regions, session_dir, labels):
    """Independent checks: the label used never comes from the address tested."""
    checks = []
    BATTLE_MAIN = 0x08038420

    def same_fn(a, b):
        return isinstance(a, int) and (a | 1) == (b | 1)

    # Load the bytes we need from every snapshot.
    wanted = {"inBattle": 0x030026F9, "partyAction": 0x0203CED3, "outcome": 0x0202433A}
    vals = {k: [] for k in wanted}
    for _, dumps, labs in load_session(session_dir):
        for dump, lab in zip(dumps, labs):
            for k, a in wanted.items():
                off = addr_to_offset(regions, a)
                vals[k].append((int(dump[off]) if off is not None else None, lab))

    # gMain.inBattle bit must be set whenever the screen is BattleMainCB2.
    rows = [(v, l) for v, l in vals["inBattle"] if same_fn(l.get("cb2"), BATTLE_MAIN)]
    if rows:
        ok = sum(1 for v, _ in rows if v is not None and v & 0x02)
        checks.append({"name": "gMain.inBattle (0x030026F9 bit 1) set on battle screen",
                       "result": "CONFIRMED" if ok == len(rows) else f"CONTRADICTED ({ok}/{len(rows)})",
                       "samples": len(rows)})
    else:
        checks.append({"name": "gMain.inBattle", "result": "NO DATA (no battle snapshots yet)"})

    # gPartyMenu.action must be SEND_OUT(1) or SWITCH(8) on the in-battle party screen.
    rows = [(v, l) for v, l in vals["partyAction"]
            if same_fn(l.get("cb2"), 0x081B01B0) and l.get("inBattle") == 1]
    if rows:
        ok = sum(1 for v, _ in rows if v in (1, 8))
        checks.append({"name": "gPartyMenu.action (0x0203CED3) is 1 or 8 in battle party menu",
                       "result": "CONFIRMED" if ok == len(rows) else f"CONTRADICTED ({ok}/{len(rows)})",
                       "samples": len(rows)})
    else:
        checks.append({"name": "gPartyMenu.action", "result": "NO DATA (no in-battle party menu yet)"})

    # gBattleOutcome must be 0 whenever the game is asking for a move.
    rows = [(v, l) for v, l in vals["outcome"] if l.get("phase") == "CHOOSE_ACTION"]
    if rows:
        ok = sum(1 for v, _ in rows if v == 0)
        checks.append({"name": "gBattleOutcome (0x0202433A) is 0 while choosing actions",
                       "result": "CONFIRMED" if ok == len(rows) else f"CONTRADICTED ({ok}/{len(rows)})",
                       "samples": len(rows)})
    else:
        checks.append({"name": "gBattleOutcome", "result": "NO DATA"})

    # Overworld callback value actually observed.
    ow = Counter(l.get("cb2") for l in labels if l.get("inBattle") == 0)
    if ow:
        top_cb2, cnt = ow.most_common(1)[0]
        checks.append({"name": "most common callback2 outside battle (expect 0x08085E5D)",
                       "result": f"0x{top_cb2:08X} ({cnt} snapshots)",
                       "matches_config": same_fn(top_cb2, 0x08085E5C)})
    return checks


def write_markdown(r, path):
    L = [f"# Memory finder report", "", f"Session: `{r['session']}` — {r['snapshots']} snapshots", ""]
    L += ["## Config check", ""]
    for c in r["config_check"]:
        L.append(f"- {c['name']}: **{c['result']}**")
    L += ["", "## Unidentified waits (prompts the bridge doesn't handle yet)", ""]
    if r["unidentified_waits"]:
        for w in r["unidentified_waits"]:
            L.append(f"- opcode {w['opcode']} {w['command']} (ctrl {w['ctrl0']}): seen {w['times_seen']}x, "
                     f"longest {w['longest_s']}s")
    else:
        L.append("- none")
    L += ["", "## Signal mirrors", ""]
    for k, m in r["mirrors"].items():
        if "note" in m:
            L.append(f"- **{k}**: {m['note']}")
            continue
        L.append(f"- **{k}** ({m['samples']} samples, {m['distinct_values']} values, "
                 f"{m['total_candidates']} candidates):")
        for c in m["candidates"]:
            tag = " ← signal source" if c["source"] else ""
            L.append(f"  - `{c['addr']}` {c['symbol']} match {c['match']}{tag}")
    L += ["", "## State discriminators", ""]
    for k, d in r["discriminators"].items():
        if "note" in d:
            L.append(f"- **{k}**: {d['note']}")
            continue
        L.append(f"- **{k}** ({d['states']} states, {d.get('perfect', 0)} addresses separate all of them):")
        for c in d["candidates"]:
            vals = ", ".join(f"{s}→{v}" for s, v in list(c["values"].items())[:6])
            L.append(f"  - `{c['addr']}` {c['symbol']} separates {c['states_separated']}/{c['of_states']}: {vals}")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


# ===================================================================== CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("watch", "record"):
        p = sub.add_parser(name)
        p.add_argument("--out", default=DEFAULT_OUT)
        p.add_argument("--port", type=int, default=DEFAULT_PORT)
        p.add_argument("--max-dumps", type=int, default=1500)
        p.add_argument("--per-state", type=int, default=25)
        p.add_argument("--sym")
        p.add_argument("--pret")
        p.add_argument("--every-min", type=float, default=10.0, help="watch: re-analyze interval")
    p = sub.add_parser("analyze")
    p.add_argument("session")
    p.add_argument("--sym")
    p.add_argument("--pret")
    args = ap.parse_args(argv)

    if args.cmd == "analyze":
        r = analyze(args.session, args.sym, args.pret)
        print(f"report: {os.path.join(args.session, 'report.md')} ({r['snapshots']} snapshots)")
        return

    rec = Recorder(args.out, port=args.port, per_state=args.per_state, max_dumps=args.max_dumps)
    if args.cmd == "record":
        rec.run()
        return

    state = {"last": time.time()}

    def tick(r):
        if time.time() - state["last"] >= args.every_min * 60 and r.n_dumps > 0:
            state["last"] = time.time()
            r._flush()
            try:
                rep = analyze(r.dir, args.sym, args.pret)
                print(f"[finder] report refreshed ({rep['snapshots']} snapshots): {r.dir}/report.md")
            except SystemExit:
                pass

    out = rec.run(on_tick=tick)
    analyze(out, args.sym, args.pret)
    print(f"[finder] final report: {out}/report.md")


if __name__ == "__main__":
    main()

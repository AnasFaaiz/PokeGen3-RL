#!/usr/bin/env python3
"""Talk to battle_bridge.lua by hand.

The bridge accepts one client at a time, so close this before starting
training -- otherwise env.py can't connect.

    python tools/bridge_cli.py              # interactive
    python tools/bridge_cli.py MAP_DUMP     # one command and exit
    python tools/bridge_cli.py NAV:HEAL --watch
"""
import argparse
import socket
import sys
import time

from exp_tables import box_level, species_names

BALL_NAMES = {1: "Master", 2: "Ultra", 3: "Great", 4: "Poke", 5: "Safari", 6: "Net",
              7: "Dive", 8: "Nest", 9: "Repeat", 10: "Timer", 11: "Luxury", 12: "Premier"}


def pretty(line):
    """Decode INVENTORY / BOX replies; anything else passes through."""
    if line.startswith("INV:") and not line.startswith("INV:ERR"):
        f = dict(kv.split("=", 1) for kv in line[4:].split(";"))
        balls = ", ".join(
            f"{BALL_NAMES.get(int(i), i)} x{q}"
            for i, q in (b.split(":") for b in f["balls"].split(",") if b)
        ) or "none"
        return (f"money={f['money']}  balls: {balls}\n"
                f"party size={f['party']} levels=[{f['lv']}]  "
                f"outcome={f['outcome']} trainer_battle={f['trainer']}")
    if line.startswith("BOX:") and not line.startswith("BOX:ERR"):
        names = species_names()
        rows = []
        for ent in filter(None, line[4:].split(",")):
            pos, sp, exp = ent.split(":")
            sp, exp = int(sp), int(exp)
            rows.append(f"  box {pos:>5}  {names.get(sp, sp)!s:<12} Lv{box_level(sp, exp)}  (exp {exp})")
        return f"{len(rows)} PC mons\n" + "\n".join(rows)
    return line

BANNER = """commands:
  MAP_DUMP            ASCII map of where you are
  NAV_DEBUG           why the planner can or cannot find grass
  PARTY_HP            PARTY_HP:<current>:<max>:<conscious>
  NAV:HUNT            walk to grass/cave floor and keep stepping until a battle starts (NAV:GRASS is an alias)
  NAV:HEAL            walk to the Pokemon Center, heal, then return to where you were hunting
  NAV:STATUS          what navigation is doing right now
  NAV:CANCEL          stop navigating
  AUTO:OFF            hands off the controls entirely (same as pressing L)
  AUTO:ON             give the controls back
  AUTO:STATUS         is automation on?
  STATUS              battle phase
  GET_STATE           the raw observation vector
  INVENTORY           money, ball counts, party levels, last battle outcome (read-only)
  THROW[:itemId]      throw a ball (default Poke Ball) in a wild single battle; poll STATUS after
  BAG                 bag pocket/cursor, current callback and controller fn (read-only)
  BOX                 every occupied PC slot with species and derived level (read-only)
  BOXMON:<b.s>        raw fields of one PC mon (species, exp, EVs, IVs); PARTYMON:<n> for a party slot
  SWAP:...            exchange a party slot with a PC mon (see swap_test.py for the arguments)
  watch               poll NAV:STATUS until it settles
  quit
"""


class Bridge:
    def __init__(self, host, port):
        self.sock = socket.create_connection((host, port), timeout=10)
        self.f = self.sock.makefile("r")

    def send(self, line):
        self.sock.sendall((line + "\n").encode())
        # MAP_DUMP answers with many lines; everything else with one.
        if line in ("MAP_DUMP", "NAV_DEBUG"):
            time.sleep(0.6)
            self.sock.settimeout(1.0)
            chunks = []
            try:
                while True:
                    b = self.sock.recv(65536)
                    if not b:
                        break
                    chunks.append(b)
                    if len(b) < 65536:
                        break
            except socket.timeout:
                pass
            self.sock.settimeout(10)
            return b"".join(chunks).decode(errors="replace")
        return self.f.readline().rstrip("\n")

    def watch(self, timeout=240):
        """Poll NAV:STATUS until it stops moving, printing each change."""
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            s = self.send("NAV:STATUS")
            if s != last:
                print(f"  {time.strftime('%H:%M:%S')}  {s}")
                last = s
            if s.startswith(("NAV:ARRIVED", "NAV:BATTLE", "NAV:FAILED", "NAV:IDLE")):
                return s
            time.sleep(0.5)
        print("  (timed out watching)")
        return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--watch", action="store_true",
                    help="after the command, poll NAV:STATUS until it settles")
    args = ap.parse_args()

    try:
        b = Bridge(args.host, args.port)
    except OSError as e:
        sys.exit(f"can't reach the bridge on {args.host}:{args.port} ({e})\n"
                 "Is mGBA running with battle_bridge.lua loaded, and is "
                 "training stopped? The bridge takes one client at a time.")

    if args.command:
        print(pretty(b.send(args.command)))
        if args.watch:
            b.watch()
        return

    print(BANNER)
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in ("quit", "exit"):
            return
        if line == "watch":
            b.watch()
            continue
        try:
            print(pretty(b.send(line)))
        except OSError as e:
            sys.exit(f"connection lost: {e}")


if __name__ == "__main__":
    main()

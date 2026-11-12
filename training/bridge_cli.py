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
        print(b.send(args.command))
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
            print(b.send(line))
        except OSError as e:
            sys.exit(f"connection lost: {e}")


if __name__ == "__main__":
    main()

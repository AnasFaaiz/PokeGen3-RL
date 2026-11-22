"""Pre-flight for parallel training: which mGBA windows answer on which bridge port?

    python training/check_ports.py            # the ports training would use (POKE_N_ENVS, POKE_PORT_BASE, ...)
    python training/check_ports.py 4          # the first 4 ports (8888, 8890, 8892, 8894)

Read-only: it sends STATUS and INVENTORY and disconnects at once (the bridge serves one client at a time, so a
connection that stays open would block training).
"""
import os
import socket
import sys

DEFAULT_BASE, DEFAULT_STEP = 8888, 2


def ports_from_env(n_envs, environ=None):
    """Ports for env 0..n-1: POKE_PORTS (comma list) if set, else POKE_PORT_BASE + POKE_PORT_STEP * i."""
    env = os.environ if environ is None else environ
    explicit = env.get("POKE_PORTS", "").strip()
    if explicit:
        ports = [int(p) for p in explicit.split(",") if p.strip()]
        if len(ports) != n_envs:
            raise ValueError(f"POKE_PORTS lists {len(ports)} ports but POKE_N_ENVS is {n_envs}")
        return ports
    base, step = int(env.get("POKE_PORT_BASE", DEFAULT_BASE)), int(env.get("POKE_PORT_STEP", DEFAULT_STEP))
    return [base + step * i for i in range(n_envs)]


def probe_port(port, host="127.0.0.1", timeout=3.0):
    """-> (ok, info). info is a dict (phase, map, party, has_px) when ok, else a short reason."""
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.settimeout(timeout)
            f = s.makefile("rw")
            f.write("STATUS\n")
            f.flush()
            status = f.readline().strip()
            f.write("INVENTORY\n")
            f.flush()
            inv = f.readline().strip()
    except OSError as e:
        return False, f"no answer ({type(e).__name__}: {e})"
    parts = status.split(":")
    if len(parts) != 3 or parts[0] != "STATUS":
        return False, f"unexpected reply {status!r} (is battle_bridge.lua loaded in that window?)"
    info = {"phase": parts[1], "map": "?", "party": "?", "has_px": False}
    if inv.startswith("INV:") and not inv.startswith("INV:ERR"):
        fields = dict(kv.split("=", 1) for kv in inv[4:].split(";") if "=" in kv)
        info.update(map=fields.get("map", "?"), party=fields.get("party", "?"), has_px="px" in fields)
    return True, info


def check_ports(ports, host="127.0.0.1", timeout=3.0):
    """[(port, ok, info)] for every port, in order."""
    return [(p, *probe_port(p, host, timeout)) for p in ports]


def format_report(results):
    lines, bad = [], 0
    for i, (port, ok, info) in enumerate(results):
        if ok:
            lines.append(f"  env {i}  port {port}  OK   phase={info['phase']}  map={info['map']}  party={info['party']}  "
                         f"new-lua={'yes' if info['has_px'] else 'NO (reload the script)'}")
        else:
            bad += 1
            lines.append(f"  env {i}  port {port}  FAIL {info}")
    maps = {i['map'] for _, ok, i in results if ok}
    if len(maps) > 1:
        lines.append(f"  note: the windows are on different maps {sorted(maps)}; that is fine, each env learns its own area.")
    return "\n".join(lines), bad


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("POKE_N_ENVS", 1))
    text, bad = format_report(check_ports(ports_from_env(n)))
    print(text)
    if bad:
        print(f"{bad} of {n} bridges did not answer. Each window needs games/battle_bridge.lua loaded "
              "(Tools -> Scripting -> Load script), first window first.")
        sys.exit(1)


if __name__ == "__main__":
    main()

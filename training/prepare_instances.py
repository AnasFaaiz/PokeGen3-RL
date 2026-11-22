"""Make one ROM + save + start-state copy per mGBA window for parallel training, and print the launch commands.

    python training/prepare_instances.py --rom <Emerald .gba> --n 4 [--state <.ss1>] [--out instances]

Every window needs its OWN save file (two windows writing one .sav would corrupt it), so the ROM and its .sav (the file
next to the ROM with the same name, if it exists) are copied into <out>/env<i>/. This script only copies files and prints
commands; it never starts anything. <out> is git-ignored (it holds copies of the ROM).
"""
import argparse
import os
import shutil
import sys

LAUNCH = "flatpak run io.mgba.mGBA"


def out_root(out):
    return os.path.abspath(out)


def make_plan(rom, n, state, out):
    """[{index, dir, rom, sav, state, command}] (paths only; nothing is copied here)."""
    rom_name = os.path.basename(rom)
    stem = os.path.splitext(rom_name)[0]
    plan = []
    for i in range(n):
        d = os.path.join(out_root(out), f"env{i}")
        inst = {"index": i, "dir": d, "rom": os.path.join(d, rom_name), "sav": os.path.join(d, stem + ".sav"),
                "state": os.path.join(d, "start.ss1") if state else None}
        inst["command"] = f"{LAUNCH} " + (f"-t {inst['state']} " if state else "") + inst["rom"]
        plan.append(inst)
    return plan


def copy_instances(rom, plan, state):
    """Copies the ROM, its .sav (if any) and the state into each instance dir. Existing files are not overwritten
    unless they differ in size, so a half-played save in an instance directory is never silently replaced."""
    src_sav = os.path.splitext(rom)[0] + ".sav"
    notes = []
    for inst in plan:
        os.makedirs(inst["dir"], exist_ok=True)
        if not os.path.exists(inst["rom"]):
            shutil.copyfile(rom, inst["rom"])
        if os.path.exists(src_sav):
            if not os.path.exists(inst["sav"]):
                shutil.copyfile(src_sav, inst["sav"])
            else:
                notes.append(f"kept the existing save in {inst['dir']}")
        if state and not os.path.exists(inst["state"]):
            shutil.copyfile(state, inst["state"])
    if not os.path.exists(src_sav):
        notes.append(f"no {os.path.basename(src_sav)} next to the ROM: each window will start from the state only")
    return notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--state", default=None, help="a save state (.ss1) taken while standing on the grass route")
    ap.add_argument("--out", default="instances")
    args = ap.parse_args()
    if not os.path.isfile(args.rom):
        sys.exit(f"ROM not found: {args.rom}")
    if args.state and not os.path.isfile(args.state):
        sys.exit(f"state not found: {args.state}")
    if not 1 <= args.n <= 16:
        sys.exit("--n must be between 1 and 16 (the bridge tries 16 ports)")
    plan = make_plan(args.rom, args.n, args.state, args.out)
    for note in copy_instances(args.rom, plan, args.state):
        print("note:", note)
    print(f"\nPrepared {args.n} instance folders under {out_root(args.out)}.\n"
          "Start one window per command below (keep them open), then load games/battle_bridge.lua in each one,\n"
          "first window first (Tools -> Scripting -> Load script). The console prints the port each window got.\n")
    for inst in plan:
        print(f"  env {inst['index']} (port {8888 + 2 * inst['index']}):  {inst['command']}")
    print("\nIf mGBA cannot read or save in that folder (Flatpak sandbox), allow it once with:\n"
          f"  flatpak override --user --filesystem={out_root(args.out)} io.mgba.mGBA\n"
          "Then check the windows with: python training/check_ports.py " + str(args.n))


if __name__ == "__main__":
    main()

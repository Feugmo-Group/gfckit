"""Pull the reduced MSD tiers of every COMPLETED fracMD run down from Nibi.

Trajectories (~1.3 GB per run) stay on the cluster; only msd_tier{A,B,C}.dat
comes back, about 850 kB per run. A run is only pulled once its log carries the
FRACMD_DONE marker, so a job still writing tier C cannot be mistaken for a
finished one -- tier C grows throughout the production phase and a partial copy
looks superficially valid.

Usage:
    python examples/fracmd_sync.py            # pull anything new
    python examples/fracmd_sync.py --list     # just report what is ready
"""
import os
import subprocess
import sys

HOST = "nibi"
REMOTE_ROOT = "scratch/fracMD/runs"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_ROOT = os.path.join(HERE, "data", "fracmd")
TIERS = ("A", "B", "C")
EXPECTED_LINES = {"A": 10003, "B": 9903, "C": 1903}   # a full 20 ns run


def ssh(cmd):
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25",
                        HOST, cmd], capture_output=True, text=True)
    return r.stdout.strip()


def completed_runs():
    """Runs whose log carries the completion marker, newest last."""
    out = ssh(f"grep -l FRACMD_DONE {REMOTE_ROOT}/*/*/*/log.lammps 2>/dev/null")
    return [p.rsplit("/", 1)[0] for p in out.splitlines() if p.strip()]


def main(list_only=False):
    runs = completed_runs()
    if not runs:
        print("no completed runs found (is the campaign still in production?)")
        return
    print(f"{len(runs)} completed run(s) on {HOST}\n")
    pulled = skipped = 0
    for remote in sorted(runs):
        rel = remote[len(REMOTE_ROOT) + 1:]           # comp/T###/r#
        local = os.path.join(LOCAL_ROOT, rel)
        have = all(os.path.exists(os.path.join(local, f"msd_tier{t}.dat"))
                   for t in TIERS)
        if have:
            print(f"  have    {rel}")
            skipped += 1
            continue
        if list_only:
            print(f"  READY   {rel}")
            continue
        os.makedirs(local, exist_ok=True)
        ok = True
        for t in TIERS:
            r = subprocess.run(
                ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25",
                 f"{HOST}:{remote}/msd_tier{t}.dat", local],
                capture_output=True, text=True)
            if r.returncode != 0:
                ok = False
                print(f"  FAILED  {rel} tier {t}: {r.stderr.strip()[:60]}")
                break
        if not ok:
            continue
        # a truncated transfer is worse than none: it looks like real data
        bad = []
        for t in TIERS:
            n = sum(1 for _ in open(os.path.join(local, f"msd_tier{t}.dat")))
            if n != EXPECTED_LINES[t]:
                bad.append(f"tier {t}: {n} lines, expected {EXPECTED_LINES[t]}")
        print(f"  pulled  {rel}" + ("  [" + "; ".join(bad) + "]" if bad else ""))
        pulled += 1
    if not list_only:
        print(f"\n{pulled} pulled, {skipped} already local")


if __name__ == "__main__":
    main(list_only="--list" in sys.argv)

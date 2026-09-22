"""Convert a class2 LiPS data file to the atomic form the MACE model expects.

The production glasses were built for a bonded force field: atom_style full,
eight atom types, and explicit bonds, angles and dihedrals. Types encode
chemical environment as well as element -- three distinct phosphorus types and
four sulfur types, distinguishing bridging from terminal sulfur and PS4 from
P2S7 phosphorus -- because the class2 terms need that distinction.

A machine-learned potential does not. The LiPS-25 MACE models take element
identity only and infer environment from the neighbourhood, so the conversion
collapses the eight types onto the three elements and drops the topology.

Elements are assigned from the Masses block rather than from a hardcoded table,
so a data file whose type ordering differs is either converted correctly or
rejected, never silently mislabelled. Getting this wrong would be invisible in
the output and fatal in the physics: phosphorus read as sulfur still runs.

Type order in the output is Li, P, S, matching the pair_coeff line used by the
LiPS-25 benchmark inputs.

Usage:
    python lammps_to_mace_data.py in.data out.data
"""
import re
import sys

# LAMMPS `real`/`metal` masses; matched to tolerance, not equality.
ELEMENTS = [("Li", 6.941), ("P", 30.9738), ("S", 32.066)]
OUT_ORDER = ["Li", "P", "S"]


def parse(path):
    text = open(path).read()
    lines = text.splitlines()

    natoms = int(re.search(r"^\s*(\d+)\s+atoms", text, re.M).group(1))
    ntypes = int(re.search(r"^\s*(\d+)\s+atom types", text, re.M).group(1))
    box = []
    for lo, hi in (("xlo", "xhi"), ("ylo", "yhi"), ("zlo", "zhi")):
        m = re.search(rf"^\s*(\S+)\s+(\S+)\s+{lo}\s+{hi}", text, re.M)
        box.append((float(m.group(1)), float(m.group(2))))

    def section(name):
        for i, ln in enumerate(lines):
            if ln.strip().split("#")[0].strip() == name:
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                out = []
                while j < len(lines) and lines[j].strip():
                    out.append(lines[j])
                    j += 1
                return out
        raise SystemExit(f"{path}: no '{name}' section")

    masses = {}
    for ln in section("Masses"):
        f = ln.split()
        masses[int(f[0])] = float(f[1])
    if len(masses) != ntypes:
        raise SystemExit(f"{path}: {len(masses)} masses for {ntypes} types")

    # type -> element, by mass
    t2e = {}
    for t, m in masses.items():
        hit = [e for e, mm in ELEMENTS if abs(m - mm) < 0.05]
        if len(hit) != 1:
            raise SystemExit(f"{path}: mass {m} of type {t} matches {hit}")
        t2e[t] = hit[0]

    # Atoms # full -> id mol type q x y z [ix iy iz]
    atoms = []
    for ln in section("Atoms"):
        f = ln.split()
        atoms.append((int(f[0]), t2e[int(f[2])], f[4], f[5], f[6]))
    if len(atoms) != natoms:
        raise SystemExit(f"{path}: {len(atoms)} atoms, header says {natoms}")
    return box, t2e, atoms


def main(src, dst):
    box, t2e, atoms = parse(src)
    idx = {e: i + 1 for i, e in enumerate(OUT_ORDER)}
    counts = {e: sum(1 for a in atoms if a[1] == e) for e in OUT_ORDER}

    with open(dst, "w") as fh:
        fh.write(f"# from {src}: {len(t2e)} class2 types -> Li,P,S\n\n")
        fh.write(f"{len(atoms)} atoms\n{len(OUT_ORDER)} atom types\n\n")
        for (lo, hi), d in zip(box, "xyz"):
            fh.write(f"{lo:.10e} {hi:.10e} {d}lo {d}hi\n")
        fh.write("\nMasses\n\n")
        for e in OUT_ORDER:
            fh.write(f"{idx[e]} {dict(ELEMENTS)[e]}\n")
        fh.write("\nAtoms # atomic\n\n")
        # Original atom IDs are preserved, not renumbered. The class2 file does
        # not store atoms in ID order, so renumbering by position would silently
        # permute the mapping and every `rerun` against a trajectory dumped from
        # the original file would assign each frame's coordinates to the wrong
        # atom -- and still run to completion.
        for aid, e, x, y, z in sorted(atoms, key=lambda a: a[0]):
            fh.write(f"{aid} {idx[e]} {x} {y} {z}\n")

    n = len(atoms)
    frac = {e: counts[e] / n for e in OUT_ORDER}
    # x in xLi2S-(100-x)P2S5 implied by the Li:P ratio, as a sanity check that
    # the type collapse preserved the composition the file claims to be.
    x = 100 * (counts["Li"] / 2) / (counts["Li"] / 2 + counts["P"] / 2) \
        if counts["P"] else float("nan")
    print(f"  {src} -> {dst}")
    print(f"    {n} atoms  " + "  ".join(
        f"{e} {counts[e]} ({frac[e]:.1%})" for e in OUT_ORDER))
    print(f"    implied x(Li2S) = {x:.1f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

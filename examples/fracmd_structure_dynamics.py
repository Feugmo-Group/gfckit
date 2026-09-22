"""Do mobile Li sit at different local environments from immobile Li?

The composition-level trend -- F_IS rising with beta across three glasses -- has
three points, so every monotone descriptor correlates with beta and none of it
means much. This is the within-structure version: thousands of Li in a single
run, each with its own local inversion symmetry and its own time-averaged MSD.

If the subdiffusion comes from a distribution of local mobilities, as the TAMSD
result implies, then mobility should be predictable from the local environment.
If F_IS carries no information about which Li move, then it is a composition-level
coincidence and should not be offered as an explanation.

Both quantities are per-atom and joined on LAMMPS id, verified rather than
assumed: ASE returns lammps-data sorted by id, so ASE index + 1 equals the id in
the Li-only trajectory dump.

Requires the torchdisorder environment:
    cd ~/Code/Python/torchdisorder
    PYTHONPATH=. .venv-analysis/bin/python <this file>
"""
import os
import sys

import ase.io
import numpy as np

sys.path.insert(0, os.path.expanduser("~/Code/Python/torchdisorder"))
from torchdisorder.analysis.descriptors import StructureDescriptors  # noqa: E402

ROOT = os.path.expanduser(
    "~/Code/Python/Fractional_Calculus/gfckit/data/fracmd")
STRUCT = os.path.join(ROOT, "structures")
COMPS = ["67Li2S-33P2S5", "70Li2S-30P2S5", "75Li2S-25P2S5"]


def spearman(x, y):
    """Rank correlation, so a monotone but nonlinear relation still registers."""
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def main(which="initial"):
    print(f"  Per-atom, within a single structure ({which} configuration).")
    print("  n is the number of Li.\n")
    print("  composition        n     Spearman rho(descriptor, TAMSD)")
    print("                          F_IS      q4       q6      tet     CN")
    for comp in COMPS:
        atoms = ase.io.read(os.path.join(STRUCT, f"{comp}_250K_{which}.data"),
                            format="lammps-data", atom_style="full",
                            units="real")
        m = atoms.get_masses()
        atoms.set_chemical_symbols(
            list(np.where(m < 10, "Li", np.where(m < 31.5, "P", "S"))))
        li_idx = np.where(m < 10)[0]

        z = np.load(os.path.join(ROOT, comp, "T250", "r1",
                                 "per_atom_mobility.npz"))
        if not np.array_equal(li_idx + 1, z["ids"]):
            raise ValueError(f"{comp}: id mapping does not hold")
        tam = z["tamsd"]

        sd = StructureDescriptors(atoms, cutoff=5.0)
        op = sd.order_params_per_atom(central_z=3, neighbor_z=16, cutoff=3.0)
        cn = np.asarray(sd.coordination_numbers("Li", "S", 3.0)).astype(float)

        rho = {k: spearman(np.asarray(op[k]).astype(float), tam)
               for k in ("fis", "q4", "q6", "tet")}
        rho["cn"] = spearman(cn, tam)
        print(f"  {comp:16s} {len(tam):5d}   " +
              "  ".join(f"{rho[k]:+.3f}" for k in ("fis", "q4", "q6", "tet", "cn")))

        # is the effect big enough to matter? compare the slowest and fastest
        # deciles of Li by their environment
        f = np.asarray(op["fis"]).astype(float)
        lo, hi = np.percentile(tam, [10, 90])
        slow, fast = tam <= lo, tam >= hi
        print(f"                     F_IS: slowest decile {f[slow].mean():.4f}, "
              f"fastest decile {f[fast].mean():.4f}, "
              f"difference {f[fast].mean()-f[slow].mean():+.4f} "
              f"(sd of F_IS {f.std():.4f})")

    print("\n  rho near zero means the local environment does not predict which")
    print("  Li move, and the composition-level F_IS trend is then a coincidence")
    print("  rather than a mechanism.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "initial")

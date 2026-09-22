"""Structural disorder of the fracMD glasses, via torchdisorder.

The transport results split the three compositions: alpha = k_B T / E_0 holds for
75Li2S-25P2S5 and drifts systematically for 67 and 70, and the TAMSD test says
none of them is a trap-limited CTRW. Both point at the structure rather than at a
waiting-time distribution, so this quantifies the local order around Li -- the
mobile species -- and asks whether it tracks the transport.

Computed with torchdisorder.analysis.descriptors.StructureDescriptors on the
final configurations of the actual production runs, not on generated structures:

  * F_IS, local inversion symmetry: 1 for a centrosymmetric environment, falling
    towards 0 as the neighbour shell loses inversion symmetry. A broad F_IS
    distribution means sites are inequivalent, which is the structural statement
    that a distribution of local mobilities would need.
  * q4, q6 Steinhardt bond-orientational order, and the tetrahedral parameter.
  * coordination numbers.

Requires the torchdisorder analysis environment, not the gfckit one:
    cd ~/Code/Python/torchdisorder
    PYTHONPATH=. .venv-analysis/bin/python <this file>
"""
import os
import sys

import ase.io
import numpy as np

sys.path.insert(0, os.path.expanduser("~/Code/Python/torchdisorder"))
from torchdisorder.analysis.descriptors import StructureDescriptors  # noqa: E402

DATA = os.path.expanduser(
    "~/Code/Python/Fractional_Calculus/gfckit/data/fracmd/structures")
COMPS = ["67Li2S-33P2S5", "70Li2S-30P2S5", "75Li2S-25P2S5"]

# transport summary, for the correlation at the end
BETA_250 = {"67Li2S-33P2S5": 0.5414, "70Li2S-30P2S5": 0.5791,
            "75Li2S-25P2S5": 0.7412}
E0_HOLDS = {"67Li2S-33P2S5": False, "70Li2S-30P2S5": False,
            "75Li2S-25P2S5": True}


def load(comp):
    """Final production configuration, with LAMMPS types mapped to elements."""
    path = os.path.join(DATA, f"{comp}_250K_final.data")
    atoms = ase.io.read(path, format="lammps-data", atom_style="full",
                        units="real")
    # ASE assigns species from the mass table; make the symbols explicit so the
    # descriptors group by element rather than by LAMMPS type
    m = atoms.get_masses()
    sym = np.where(m < 10, "Li", np.where(m < 31.5, "P", "S"))
    atoms.set_chemical_symbols(list(sym))
    return atoms


def main():
    stats = {}
    for comp in COMPS:
        atoms = load(comp)
        sym = np.array(atoms.get_chemical_symbols())
        is_li = sym == "Li"
        sd = StructureDescriptors(atoms, cutoff=5.0)

        # Li centred, S neighbours: the Li-S shell is what a mobile Li sees.
        # 3.0 A sits past the Li-S first peak (~2.4-2.6 A) and before the second.
        op = sd.order_params_per_atom(central_z=3, neighbor_z=16, cutoff=3.0)
        print(f"{comp}: {len(atoms)} atoms, {is_li.sum()} Li "
              f"({100*is_li.mean():.1f}%)")

        row = {"n_li": int(is_li.sum())}
        for k, v in op.items():
            v = np.asarray(v).astype(float)          # already Li-only
            row[f"{k}_mean"] = float(np.nanmean(v))
            row[f"{k}_std"] = float(np.nanstd(v))
        cn = np.asarray(sd.coordination_numbers("Li", "S", 3.0)).astype(float)
        row["cnLiS_mean"] = float(np.nanmean(cn))
        row["cnLiS_std"] = float(np.nanstd(cn))
        stats[comp] = row
        print()

    keys = [k for k in stats[COMPS[0]] if k.endswith(("_mean", "_std"))]
    print("  quantity                67Li2S     70Li2S     75Li2S    trend")
    for k in sorted(keys):
        vals = np.array([stats[c][k] for c in COMPS])
        mono = ("up" if np.all(np.diff(vals) > 0) else
                "down" if np.all(np.diff(vals) < 0) else "--")
        print(f"  {k:22s} {vals[0]:9.4f} {vals[1]:10.4f} {vals[2]:10.4f}    {mono}")

    print("\n  transport, for comparison:")
    for c in COMPS:
        print(f"    {c}: beta(250 K) = {BETA_250[c]:.4f}, "
              f"alpha=kT/E0 {'holds' if E0_HOLDS[c] else 'fails'}")
    print("\n  A quantity that moves monotonically with beta is a candidate")
    print("  structural explanation; one that does not, is not.")

    np.savez_compressed(os.path.join(DATA, "descriptors.npz"), **{
        f"{c}|{k}": v for c, r in stats.items() for k, v in r.items()})


if __name__ == "__main__":
    main()

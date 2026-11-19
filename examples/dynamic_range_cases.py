"""Dynamic-range requirement beyond the best case of dynamic_range_study.py.

dynamic_range_study.py uses the most favourable pair, {0.2, 0.9} with equal
weights.  This script repeats the same measurement (fraction of the order range
recovered by the local slope at the two ends of a window of D decades centred
on the crossover) for closer orders and unequal weights, including the paper's
own benchmark {0.3, 0.8} with weights {0.6, 0.4}.

The crossover of  w_lo s^a_lo + w_hi s^a_hi  is where the two terms are equal,
s* = (w_lo / w_hi)^(1 / (a_hi - a_lo)); the window is centred on t* = 1 / s*.
Noise-free, exact (fixed-Talbot) responses, so these are lower bounds.
"""
import jax
import jax.numpy as jnp
import numpy as np

from gfckit.laplace import distributed_order_response

jax.config.update("jax_enable_x64", True)

CASES = [((0.2, 0.9), (0.5, 0.5)),
         ((0.3, 0.8), (0.5, 0.5)),
         ((0.3, 0.8), (0.6, 0.4)),
         ((0.4, 0.7), (0.5, 0.5)),
         ((0.2, 0.9), (0.8, 0.2)),
         ((0.2, 0.9), (0.2, 0.8))]


def required(alphas, weights, target=0.9):
    (a_lo, a_hi), (w_lo, w_hi) = alphas, weights
    tc = 1.0 / (w_lo / w_hi) ** (1.0 / (a_hi - a_lo))
    t = tc * jnp.logspace(-10.0, 10.0, 800)
    u = distributed_order_response(alphas, weights, t, lam=0.0, S=1.0)
    lt = np.log(np.asarray(t))
    slope = np.gradient(np.log(np.asarray(u)), lt)
    D = np.arange(0.25, 16.01, 0.25)
    frac = []
    for d in D:
        hi = np.interp(np.log(tc * 10 ** (-d / 2)), lt, slope)
        lo = np.interp(np.log(tc * 10 ** (d / 2)), lt, slope)
        frac.append((hi - lo) / (a_hi - a_lo))
    frac = np.clip(np.array(frac), 0, 1)
    at = lambda d: float(np.interp(d, D, frac))
    need = float(np.interp(target, frac, D)) if frac.max() >= target else np.inf
    return need, at(2.0), at(3.0)


def main():
    print(f"{'orders':<12}{'weights':<12}{'D for 90%':>10}{'frac@2':>8}{'frac@3':>8}")
    for a, w in CASES:
        need, f2, f3 = required(a, w)
        print(f"{str(a):<12}{str(w):<12}{need:>10.2f}{f2:>8.2f}{f3:>8.2f}")


if __name__ == "__main__":
    main()

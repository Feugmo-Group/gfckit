"""de Levie's porous-electrode transmission line (pure JAX).

Every other mechanistic model in this package is one of our own generators, so
recovering a fractional order from it only proves internal consistency.  This
module is different: de Levie (1963) solved a cylindrical pore *in closed form*
and the answer is known analytically -- the high-frequency exponent is exactly
1/2.  Recovering alpha = 0.500 from it is therefore validation against external
textbook truth, not against ourselves.

A pore of length L is an RC transmission line: electrolyte resistance R per unit
length down the pore, double-layer capacitance C per unit length across the
wall.  Solving the line gives

    Z(w) = sqrt( R / (j w C) ) * coth( L * sqrt(j w R C) ).

Two limits matter, and they are the whole story:

  * high frequency, |L sqrt(jwRC)| >> 1:   coth -> 1, so
        Z -> sqrt(R/C) * (j w)^(-1/2)                        [exponent -1/2]
    The signal penetrates only a short way into the pore and never learns how
    long it is.  This is a constant-phase element with alpha = 1/2 exactly.

  * low frequency, |L sqrt(jwRC)| << 1:    coth(x) -> 1/x + x/3, so
        Z -> 1 / (j w C L)                                   [exponent -1]
    The whole pore charges as one capacitor.

The crossover sits at w_c = 1 / (R C L^2).

The non-identifiability is analytic, not numerical.  In the high-frequency
branch the response depends on R and C *only through the ratio* sqrt(R/C), and
on L not at all.  So pore length is strictly invisible there, and the absolute
scale of R and C is too -- any (R, C, L) with the same R/C produces a bit-for-bit
identical response.  That is the same conclusion as Section 3.5's spectral
degeneracy, reached from textbook electrochemistry with a closed-form proof.
"""
import jax.numpy as jnp


def pore_impedance(omega, R, C, L):
    """de Levie cylindrical pore: Z = sqrt(R/(jwC)) * coth(L sqrt(jwRC)).

    omega : angular frequency [rad/s]
    R     : electrolyte resistance per unit pore length [ohm/m]
    C     : double-layer capacitance per unit pore length [F/m]
    L     : pore length [m]
    """
    s = 1j * jnp.asarray(omega)
    lam = jnp.sqrt(s * R * C)                 # inverse penetration depth [1/m]
    return jnp.sqrt(R / (s * C)) / jnp.tanh(L * lam)


def semi_infinite_pore_impedance(omega, R, C):
    """The L -> infinity limit, Z = sqrt(R/C) (jw)^(-1/2): a CPE with alpha=1/2
    exactly.  Depends only on the ratio R/C, which is the whole point."""
    s = 1j * jnp.asarray(omega)
    return jnp.sqrt(R / C) * s ** (-0.5)


def crossover_frequency(R, C, L):
    """w_c = 1/(R C L^2), where the -1/2 branch turns over into the -1 branch.
    Above it the pore looks semi-infinite; below it, like a single capacitor."""
    return 1.0 / (R * C * L ** 2)


def cpe_impedance(omega, A, alpha):
    """Constant-phase element Z = A (jw)^(-alpha) -- the single-fractional-order
    model we fit to the pore.  alpha = 1/2 recovers a Warburg element."""
    return A * (1j * jnp.asarray(omega)) ** (-alpha)


def fit_cpe_exponent(omega, Zmag):
    """Fit |Z| = A w^(-alpha) by least squares on log|Z| vs log w.

    This is the frequency-domain twin of the log-log slope estimator the rest of
    the paper uses in time domain.  Returns (alpha, A, rms) where rms is the
    relative residual of the fit in linear (not log) units.
    """
    lw = jnp.log(jnp.asarray(omega))
    lz = jnp.log(jnp.asarray(Zmag))
    M = jnp.stack([lw, jnp.ones_like(lw)], axis=1)
    coef = jnp.linalg.lstsq(M, lz, rcond=None)[0]
    alpha = -coef[0]
    A = jnp.exp(coef[1])
    resid = jnp.asarray(Zmag) / jnp.exp(M @ coef) - 1.0
    return alpha, A, jnp.sqrt(jnp.mean(resid ** 2))


def local_exponent(omega, Zmag):
    """Pointwise -d log|Z| / d log w, so the -1/2 -> -1 crossover is visible
    rather than averaged away."""
    lw = jnp.log(jnp.asarray(omega))
    lz = jnp.log(jnp.asarray(Zmag))
    return -jnp.gradient(lz, lw)

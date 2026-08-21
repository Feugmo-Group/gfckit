"""Identification methods (pure JAX).

Order recovery (single trajectory):
  * recover_order_strong  -- min coefficient-of-variation of ^C D^a u (noise-sensitive)
  * recover_order_weak    -- fit the integrated power-law solution (noise-robust)

Tempered-kernel (alpha, lambda) recovery (multi-condition, shared kernel):
  * recover_params_strong -- match the operator residual (differentiates data)
  * recover_params_weak   -- match a forward simulation to data (noise-robust)

Free nonparametric kernel:
  * recover_free_kernel   -- non-negative exponential-sum kernel (ill-posed; needs C>1)
  * kernel_shape_error    -- L1-weight relative RMSE vs a reference kernel
"""
import jax
import jax.numpy as jnp
from .operators import (caputo_L1, apply_gfd, expsum_L1_weights,
                        tempered_powerlaw_L1_weights)
from .generate import solve_gfode_constant


# -------------------- order recovery (single trajectory) ------------------
def _coeff_of_variation(u, ds, alpha, trim=0.35):
    d = caputo_L1(u, ds, alpha)
    seg = d[int(trim * u.shape[0]):]
    return jnp.std(seg) / (jnp.abs(jnp.mean(seg)) + 1e-12)


def recover_order_strong(u, ds, grid):
    """Recover alpha as the order making ^C D^a u most constant (strong form)."""
    cv = jax.vmap(lambda a: _coeff_of_variation(u, ds, a))(grid)
    return grid[jnp.argmin(cv)]


def recover_order_weak(u, t, grid):
    """Recover alpha by fitting u ~ A t^alpha + B (weak/integral form)."""
    def resid(alpha):
        Phi = jnp.stack([t ** alpha, jnp.ones_like(t)], axis=1)
        coef, *_ = jnp.linalg.lstsq(Phi, u, rcond=None)
        return jnp.mean((Phi @ coef - u) ** 2)
    r = jax.vmap(resid)(grid)
    return grid[jnp.argmin(r)]


# -------------------- shared 2-D grid search over (alpha, lam) -------------
def _grid_search(lp, a_lo=0.1, a_hi=0.9, l_lo=0.05, l_hi=1.5, n=21):
    def search(ag, lg):
        best, ba, bl = jnp.inf, ag[0], lg[0]
        for a in ag:
            for l in lg:
                val = float(lp(a, l))
                if val < best:
                    best, ba, bl = val, a, l
        return ba, bl
    a0, l0 = search(jnp.linspace(a_lo, a_hi, n), jnp.linspace(l_lo, l_hi, n))
    a1, l1 = search(jnp.linspace(max(float(a0) - 0.06, 0.02), float(a0) + 0.06, 13),
                    jnp.linspace(max(float(l0) - 0.15, 0.02), float(l0) + 0.15, 13))
    return float(a1), float(l1)


def _grid_search_fine(lp, a_lo=0.1, a_hi=0.9, l_lo=0.05, l_hi=1.5,
                      n=41, zooms=3, m=17):
    """Grid search for objectives whose minimum is NARROW.

    `_grid_search` above spaces its coarse lambda nodes 0.0725 apart and then
    refines inside a fixed box.  That is adequate for a broad objective, but it
    steps over a sharp minimum: the refinement box is then centred on the wrong
    node and the reported alpha lands on the box EDGE, which is the same
    "pinned to a search bound" pathology this package rejects elsewhere.

    Here the coarse grid is halved in spacing and the refinement is iterated,
    each stage re-centred on the current best and shrunk by the factor the
    previous stage resolved, so the returned point is interior.  Cost is a few
    thousand objective evaluations, which is affordable for a forward-simulation
    misfit and is not affordable for anything that re-differentiates the data."""
    def search(ag, lg):
        best, ba, bl = jnp.inf, float(ag[0]), float(lg[0])
        for a in ag:
            for l in lg:
                val = float(lp(a, l))
                if val < best:
                    best, ba, bl = val, float(a), float(l)
        return ba, bl

    ha, hl = (a_hi - a_lo) / (n - 1), (l_hi - l_lo) / (n - 1)
    a, l = search(jnp.linspace(a_lo, a_hi, n), jnp.linspace(l_lo, l_hi, n))
    for _ in range(zooms):
        a, l = search(jnp.linspace(max(a - ha, 0.02), min(a + ha, 0.98), m),
                      jnp.linspace(max(l - hl, 0.01), l + hl, m))
        ha, hl = 2.0 * ha / (m - 1), 2.0 * hl / (m - 1)
    return float(a), float(l)


# -------------------- tempered-kernel (alpha, lam) recovery ---------------
def recover_params_strong(u_list, ds_list, c_sink, n_lag):
    """Strong form: (alpha,lam) minimizing the operator residual over conditions.
    Differentiates the data -> accurate on clean data, sensitive to noise.

    SEARCH RESOLUTION.  At C=1 this objective is not sharp: alpha and lambda
    trade off along a ridge, so a coarse `_grid_search` can land on a node that
    looks like a clean single-parameter estimate but is really an arbitrary
    point on that ridge, one whose reported precision the objective does not
    support.  A dense scan (alpha step 0.005 over [0.1,0.9], lambda step 0.005
    over [0.005,0.6]) shows the ridge spans nearly the full search box at C=1 --
    objective values within 2x of the global minimum occur all the way from
    lambda=0.005 (best-pairing alpha=0.78) to lambda=0.555 (best-pairing
    alpha=0.205) -- and only collapses to a well-localized minimum as more
    Arrhenius conditions are added.  `_grid_search_fine` resolves that ridge
    properly at every C; it agrees with the old coarse search to within 0.005
    in alpha and lambda for C=2..4, so only the C=1 point changes."""
    u_arr = jnp.stack(u_list)
    ds_arr = jnp.array(ds_list)

    def one(u, ds, alpha, lam):
        w = tempered_powerlaw_L1_weights(alpha, lam, n_lag, ds)
        d = apply_gfd(u, w)
        return jnp.mean((d[n_lag:] - c_sink) ** 2)

    lp = jax.jit(lambda a, l: jnp.mean(
        jax.vmap(lambda u, ds: one(u, ds, a, l))(u_arr, ds_arr)))
    return _grid_search_fine(lp)


def recover_params_weak(u_list, rates, c_sink, N, dt):
    """Weak/integral form: (alpha,lam) whose FORWARD simulation best matches the
    data across conditions.  Never differentiates the data -> noise-robust.

    SEARCH RESOLUTION.  This misfit is sharp: the forward model and the
    generator are the same solver, so on clean data the objective is exactly
    zero at the true (alpha,lambda) and rises steeply away from it.  The coarse
    `_grid_search` used by the strong form has no lambda node at 0.5 and misses
    that minimum entirely, reporting alpha = 0.56 at zero noise -- pinned to the
    edge of its own refinement box, not a property of the estimator.  A dense
    scan (alpha step 0.005, lambda step 0.01) puts the true minimum at exactly
    (0.500, 0.500) up to 5% noise, so the fine search below is what the weak
    form actually recovers."""
    data = jnp.stack(u_list)
    rates_a = jnp.array(rates)
    t = jnp.arange(N) * dt
    kmax = float(jnp.max(rates_a))
    s_grid = jnp.arange(N) * (dt * kmax)

    def predict(alpha, lam):
        v = solve_gfode_constant(alpha, lam, c_sink, s_grid)
        return jax.vmap(lambda kc: jnp.interp(kc * t, s_grid, v))(rates_a)

    lp = jax.jit(lambda a, l: jnp.mean((predict(a, l) - data) ** 2))
    return _grid_search_fine(lp)


# -------------------- free nonparametric kernel ---------------------------
def recover_free_kernel(u_list, ds_list, c_sink, r, n_lag,
                        steps=3000, lr=3e-2, lam_smooth=1e-5):
    """Recover a non-negative exponential-sum kernel shared across conditions
    (Adam, pure jax).  Ill-posed from few conditions; improves as C grows."""
    M = r.shape[0]

    def kernel(p):
        return jax.nn.softplus(p["raw"])

    def loss(p):
        a = kernel(p)
        total = 0.0
        for u, ds in zip(u_list, ds_list):
            w = expsum_L1_weights(a, r, n_lag, ds)
            d = apply_gfd(u, w)
            total = total + jnp.mean((d[n_lag:] - c_sink) ** 2)
        Ksamp = expsum_L1_weights(a, r, n_lag, ds_list[0])
        return total / len(u_list) + lam_smooth * jnp.mean(jnp.diff(Ksamp, 2) ** 2)

    grad = jax.jit(jax.grad(loss))
    b1, b2, eps = 0.9, 0.999, 1e-8
    p = {"raw": jnp.full((M,), -2.0)}
    m = jax.tree_util.tree_map(jnp.zeros_like, p)
    v = jax.tree_util.tree_map(jnp.zeros_like, p)
    for i in range(1, steps + 1):
        g = grad(p)
        m = jax.tree_util.tree_map(lambda mm, gg: b1 * mm + (1 - b1) * gg, m, g)
        v = jax.tree_util.tree_map(lambda vv, gg: b2 * vv + (1 - b2) * gg * gg, v, g)
        mh = jax.tree_util.tree_map(lambda mm: mm / (1 - b1 ** i), m)
        vh = jax.tree_util.tree_map(lambda vv: vv / (1 - b2 ** i), v)
        p = jax.tree_util.tree_map(
            lambda pp, mm, vv: pp - lr * mm / (jnp.sqrt(vv) + eps), p, mh, vh)
    return kernel(p)


def _r2(u_data, u_model):
    ss_res = jnp.sum((u_data - u_model) ** 2)
    ss_tot = jnp.sum((u_data - jnp.mean(u_data)) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def powerlaw_reconstruction(t, u):
    """Best SINGLE power-law model  u ~ A t^alpha + B.  Returns (alpha, R^2)."""
    a = float(recover_order_weak(u, t, jnp.linspace(0.05, 1.6, 201)))
    Phi = jnp.stack([t ** a, jnp.ones_like(t)], axis=1)
    coef, *_ = jnp.linalg.lstsq(Phi, u, rcond=None)
    return a, _r2(u, Phi @ coef)


def recover_order_spectrum(t, u, order_grid, steps=50000, lr=0.06, ridge=1e-5):
    """DISTRIBUTED-ORDER identification (general fractional calculus).

    Fit a non-negative order spectrum w(alpha) (sum to 1) and (lambda, S) so that
    the forward solution of  (sum_j w_j ^C D^{alpha_j}) u + lambda u = S  best
    RECONSTRUCTS the data.  A single power law is the special case w = one spike.
    Returns (spectrum w over order_grid, R^2 of reconstruction, u_model).

    CONVERGENCE.  This objective converges slowly and the default was previously
    5000 steps, which stops far short of the optimum: on the two-order benchmark
    it returns R^2 = 0.84, *worse* than the nested single-power-law special case
    (0.9989).  A richer model fitting worse than a model it contains diagnoses
    optimizer failure, not ill-posedness, so any claim about what the spectrum
    can or cannot resolve must be made at convergence.  With steps >= 50000 the
    fit reaches R^2 = 0.99997 and is stable thereafter (0.99998 at 80000).  The
    `ridge` smoothness penalty is not the limiting factor: results are identical
    for ridge in {1e-5, 1e-7, 0}."""
    from .complex import spectrum_design, forward_multiterm_spectrum
    N = u.shape[0]
    dt = float(t[1] - t[0])
    u0 = u[0]
    G, tri, lag_idx = spectrum_design(order_grid, N, dt)

    def unpack(p):
        return jax.nn.softmax(p["theta"]), jax.nn.softplus(p["lam"]), p["S"]

    def model(p):
        w, lam, S = unpack(p)
        return w, forward_multiterm_spectrum(w, G, tri, lag_idx, lam, S, u0)

    def loss(p):
        w, um = model(p)
        scale = jnp.mean(u ** 2) + 1e-12
        return jnp.mean((um - u) ** 2) / scale + ridge * jnp.mean(jnp.diff(w) ** 2)

    grad = jax.jit(jax.grad(loss))
    b1, b2, eps = 0.9, 0.999, 1e-8
    p = {"theta": jnp.zeros(order_grid.shape[0]),
         "lam": jnp.array(-2.0), "S": jnp.array(1.0)}
    m = jax.tree_util.tree_map(jnp.zeros_like, p)
    v = jax.tree_util.tree_map(jnp.zeros_like, p)
    for i in range(1, steps + 1):
        lr_i = lr * (1.0 - 0.9 * i / steps)                # linear decay for stability
        g = grad(p)
        m = jax.tree_util.tree_map(lambda mm, gg: b1 * mm + (1 - b1) * gg, m, g)
        v = jax.tree_util.tree_map(lambda vv, gg: b2 * vv + (1 - b2) * gg * gg, v, g)
        mh = jax.tree_util.tree_map(lambda mm: mm / (1 - b1 ** i), m)
        vh = jax.tree_util.tree_map(lambda vv: vv / (1 - b2 ** i), v)
        p = jax.tree_util.tree_map(
            lambda pp, mm, vv: pp - lr_i * mm / (jnp.sqrt(vv) + eps), p, mh, vh)
    w, um = model(p)
    return w, _r2(u, um), um


def recover_spectrum_multicondition(t, u_list, rates, order_grid,
                                    steps=4000, lr=0.05, ridge=1e-5, trim=0.12):
    """Recover ONE shared order spectrum from several Arrhenius-clock conditions,
    via a CONVEX linear fit.

    By the Caputo scaling law  ^C D^a_t [v(k t)] = k^a (^C D^a v)(k t),  the
    balance  sum_j w_j ^C D^{a_j} v = -lambda v + S  becomes, in physical time
    for condition c,
        sum_j (w_j k_c^{-a_j}) ^C D^{a_j}_t u_c  +  lambda u_c  =  S.
    The per-order factors k_c^{-a_j} differ across conditions, so ONE shared w
    must satisfy all of them at once -- which a single curve cannot pin down.
    Linear in (w, lambda, S); we fix scale with sum(w)=1 (softmax).  Returns
    (spectrum w, residual)."""
    ds = float(t[1] - t[0])
    n0 = int(trim * t.shape[0])
    cols = []              # (C*M, J) design for the spectrum
    u_stack, one_stack = [], []
    for u, kc in zip(u_list, rates):
        Dj = jnp.stack([caputo_L1(u, ds, float(a)) for a in order_grid], axis=1)
        scale_j = jnp.asarray([float(kc) ** (-float(a)) for a in order_grid])
        cols.append((Dj * scale_j)[n0:])                  # (M, J)
        u_stack.append(u[n0:])
        one_stack.append(jnp.ones(u.shape[0] - n0))
    A = jnp.concatenate(cols, axis=0)                      # (C*M, J)
    us = jnp.concatenate(u_stack)
    ones = jnp.concatenate(one_stack)
    nrm = jnp.mean(us ** 2) + 1e-12

    def unpack(p):
        return jax.nn.softmax(p["theta"]), p["lam"], p["S"]

    def loss(p):
        w, lam, S = unpack(p)
        r = A @ w + lam * us - S * ones
        return jnp.mean(r ** 2) / nrm + ridge * jnp.mean(jnp.diff(w) ** 2)

    grad = jax.jit(jax.grad(loss))
    lj = jax.jit(loss)
    b1, b2, eps = 0.9, 0.999, 1e-8
    p = {"theta": jnp.zeros(order_grid.shape[0]),
         "lam": jnp.array(0.0), "S": jnp.array(0.0)}
    m = jax.tree_util.tree_map(jnp.zeros_like, p)
    v = jax.tree_util.tree_map(jnp.zeros_like, p)
    for i in range(1, steps + 1):
        g = grad(p)
        m = jax.tree_util.tree_map(lambda mm, gg: b1 * mm + (1 - b1) * gg, m, g)
        v = jax.tree_util.tree_map(lambda vv, gg: b2 * vv + (1 - b2) * gg * gg, v, g)
        mh = jax.tree_util.tree_map(lambda mm: mm / (1 - b1 ** i), m)
        vh = jax.tree_util.tree_map(lambda vv: vv / (1 - b2 ** i), v)
        p = jax.tree_util.tree_map(
            lambda pp, mm, vv: pp - lr * mm / (jnp.sqrt(vv) + eps), p, mh, vh)
    w, lam, S = unpack(p)
    return w, float(lj(p))


def kernel_shape_error(a, r, alpha, lam, n_lag, ds):
    """Relative RMSE between a recovered exp-sum kernel and a tempered-PL kernel,
    compared via cell-integrated L1 weights (finite even for singular kernels)."""
    wh = expsum_L1_weights(a, r, n_lag, ds)
    wh = wh / jnp.sum(wh)
    wt = tempered_powerlaw_L1_weights(alpha, lam, n_lag, ds)
    wt = wt / jnp.sum(wt)
    return float(jnp.sqrt(jnp.mean((wh - wt) ** 2)) / jnp.mean(wt))

"""Plotting helpers (matplotlib).

Computation stays in JAX; arrays are converted to numpy only here, at the
rendering boundary.

PAPER FIGURES.  Every figure that appears in manuscripts/gfc_identifiability is
drawn at the manuscript's own text width (TEXTWIDTH_IN) with the shared rcParams
of PAPER_RC, and saved by `save_paper_figure` as a vector PDF plus a raster
preview.  Drawing at final size is what makes the type legible: a 13-inch figure
scaled into a 6.48-inch column shrinks 9 pt labels to 4.5 pt, which is roughly
half the caption size.  At final size the numbers in the rcParams below are the
printed point sizes, and the floor is 7 pt.

Figure titles are deliberately absent.  A journal figure is titled by its
caption; a baked-in suptitle duplicates it, cannot be copy-edited, and cannot
carry the caption's typography (en dashes, math, references).
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless-safe
import matplotlib.pyplot as plt
import matplotlib.ticker


# \textwidth of the cas-sc article class is 468.33 pt = 6.48 in.
TEXTWIDTH_IN = 6.48

PAPER_RC = {
    "font.size": 8.0,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.titlesize": 9.0,
    "axes.linewidth": 0.6,
    "axes.titlepad": 4.0,
    "axes.labelpad": 2.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.45,
    "ytick.minor.width": 0.45,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "xtick.minor.size": 1.6,
    "ytick.minor.size": 1.6,
    "lines.linewidth": 1.3,
    "lines.markersize": 3.4,
    "legend.frameon": False,
    "legend.handlelength": 1.7,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.32,
    "legend.borderaxespad": 0.35,
    "legend.columnspacing": 1.1,
    "grid.linewidth": 0.45,
    "grid.alpha": 0.3,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,           # embed TrueType, not Type 3 (Elsevier)
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavusans",
}


def use_paper_style():
    """Apply the manuscript rcParams. Call once at the top of a figure function."""
    plt.rcParams.update(PAPER_RC)


def panel_label(ax, letter, x=-0.155, y=1.02, size=9.0, **kw):
    """Bold '(a)' outside the top-left corner of an axes, in axes coordinates.

    Every panel a caption refers to gets one; a caption pointing at a letter the
    figure does not print is a dead cross-reference the reader has to guess."""
    return ax.text(x, y, letter, transform=ax.transAxes, fontsize=size,
                   fontweight="bold", ha="left", va="bottom", **kw)


def save_paper_figure(fig, out_path, png_dpi=600):
    """Save a vector PDF next to `out_path` and a high-resolution raster preview.

    The manuscript includes the PDF: line art has no native resolution, so it
    cannot fall below a production floor the way a 160-dpi PNG does.  The PNG is
    kept because the slides, the tutorial and the package README still point at
    it.  Returns the PDF path."""
    base, _ = os.path.splitext(out_path)
    pdf_path = base + ".pdf"
    fig.savefig(pdf_path)
    fig.savefig(base + ".png", dpi=png_dpi)
    return pdf_path


def _np(x):
    return np.asarray(x)


# Benchmark identifiers are code names with underscores.  A figure is read
# against its caption, and the caption says "two-order case", "passivating
# curve", "variable-order data" -- so the figures say that too.
DISPLAY_NAME = {
    "powerlaw_alpha": "power law",
    "fick_uptake": "Fickian uptake",
    "sei_reduced": "SEI film",
    "multiterm": "two-order (distributed)",
    "variable_order": "variable order",
    "knee_battery": "capacity knee",
    "pdm_passivation": "passivating (PDM)",
    "powerlaw_0.5": r"power law, $\alpha=0.5$",
    "powerlaw_0.75": r"power law, $\alpha=0.75$",
}


def _sci_tex(x, sig=1):
    """'6.6e-08' is computer notation; a figure should print 6.6 x 10^-8."""
    if x == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10.0 ** e
    return rf"${m:.{sig}f}\times10^{{{e}}}$"


def display_name(name):
    """Caption vocabulary for a benchmark identifier; falls back to the raw
    name with underscores turned into spaces."""
    return DISPLAY_NAME.get(name, str(name).replace("_", " "))


def plot_multicondition_summary(
        t, u_clean_list, u_noisy, rates,
        n_conditions, param_err_alpha, param_err_lambda,
        noise_levels, alpha_strong, alpha_weak,
        tau, K_true, K_hat, alpha_true, lam_true, out_path,
        noisy_rate=None):
    """Four-panel summary figure for the multi-condition harness.

    Everything here is a nondimensional model problem, so the axes carry
    "model units" rather than a fabricated second or a fabricated ampere-hour."""
    use_paper_style()
    fig, ax = plt.subplots(2, 2, figsize=(TEXTWIDTH_IN, 5.0))

    # (a) the data: same kernel, different Arrhenius clocks -----------------
    # The noisy overlay belongs to ONE condition (the fastest clock).  Draw it
    # underneath every clean curve, or it buries the curve it is drawn from.
    a0 = ax[0, 0]
    k_noisy = rates[-1] if noisy_rate is None else noisy_rate
    a0.plot(_np(t), _np(u_noisy), ".", ms=1.4, alpha=0.30, color="0.45",
            zorder=1, label=f"5% noise, $k={k_noisy}$")
    styles = ["-", "--", "-.", ":"]
    for i, (kc, u) in enumerate(zip(rates, u_clean_list)):
        a0.plot(_np(t), _np(u), styles[i % len(styles)], lw=1.3, zorder=3,
                label=f"clock $k={kc}$")
    a0.set(xlabel="time $t$ (model units)",
           ylabel="degradation $u(t)$ (model units)")
    # headroom above the fan of curves so the two-column key sits clear of them
    a0.set_ylim(top=float(np.max([np.max(_np(u)) for u in u_clean_list])) * 1.42)
    a0.legend(ncol=2, loc="upper left")
    panel_label(a0, "(a)")

    # (b) multi-condition identifiability: param error vs #conditions -------
    a1 = ax[0, 1]
    # at C=1 the objective is a ridge, not a bowl: alpha and lambda trade off
    # against each other over most of the search box, so the C=1 point is a
    # best-fitting COMBINATION rather than a localized estimate of either
    # parameter.  Flag it rather than let it read as just the worst point of
    # an otherwise-ordinary convergence curve.
    a1.axvspan(0.8, 1.2, color="0.88", zorder=0)
    a1.plot(n_conditions, param_err_alpha, "o-", color="C0",
            label=r"$|\Delta\alpha|$  (dimensionless)")
    a1.plot(n_conditions, param_err_lambda, "s--", color="C1",
            label=r"$|\Delta\lambda|$  (time$^{-1}$)")
    a1.set(xlabel="number of aging conditions $C$",
           ylabel="absolute parameter error")
    a1.set_xticks(list(n_conditions))
    a1.set_ylim(0, None)
    a1.legend(loc="upper right")
    a1.grid(True)
    # the two errors have different dimensions; they share an axis only because
    # both true values are 0.5, which puts them on a common relative scale
    a1.text(0.03, 0.06, r"true $\alpha=\lambda=0.5$", transform=a1.transAxes,
            fontsize=7, color="0.35")
    a1.annotate(r"$C{=}1$: $\alpha$-$\lambda$ ridge" + "\n(not individually\nidentified)",
                xy=(1, 0.5 * (param_err_alpha[0] + param_err_lambda[0])),
                xytext=(1.55, 0.30), fontsize=6.7, color="0.25", ha="left",
                arrowprops=dict(arrowstyle="-", color="0.45", lw=0.7))
    panel_label(a1, "(b)")

    # (c) strong vs weak form under noise -----------------------------------
    a2 = ax[1, 0]
    nz = _np(noise_levels) * 100
    a2.axhline(alpha_true, ls=":", color="0.25", lw=1.0)
    a2.text(nz.max(), alpha_true + 0.012, r"true $\alpha$", fontsize=7,
            color="0.25", ha="right", va="bottom")
    a2.plot(nz, alpha_strong, "s--", color="C3", label="strong (derivative) form")
    a2.plot(nz, alpha_weak, "o-", color="C0", label="weak (integral) form")
    a2.set(xlabel="measurement noise (%)", ylabel=r"recovered $\alpha$")
    a2.set_ylim(0, 0.68)
    a2.legend(loc="center right")
    a2.grid(True)
    panel_label(a2, "(c)")

    # (d) recovered vs true kernel ------------------------------------------
    a3 = ax[1, 1]
    a3.plot(_np(tau), _np(K_true), "k-", lw=2.2, alpha=0.45,
            label=fr"true ($\alpha={alpha_true:g}$, $\lambda={lam_true:g}$)")
    a3.plot(_np(tau), _np(K_hat), "C1--", lw=1.3,
            label="recovered (weak form, 5% noise)")
    a3.set(xlabel=r"lag $\tau$ (model units)",
           ylabel=r"kernel $K(\tau)$ (model units$^{-1}$)")
    a3.set_yscale("log")
    a3.legend(loc="upper right")
    a3.grid(True, which="both")
    panel_label(a3, "(d)")

    fig.tight_layout(w_pad=1.8, h_pad=1.4)
    out = save_paper_figure(fig, out_path)
    plt.close(fig)
    return out


def plot_physical_models(t_sei, L_sei, sei_fit, alpha_sei,
                         t_pdm, L_pdm, L_ss, out_path, t_fit_lo=1.0):
    """Mechanistic (Jacobian-free Newton-Krylov) data + gfckit exponent fits.

    Panel (c) plots the LOCAL log-log slope, not the curves again.  The paper
    cites this panel for the statement that the SEI exponent trends to the
    Fickian 1/2, and the previous version could not show it: it drew a slope-1/2
    reference anchored at the very first sample, inside the sub-Fickian
    nucleation transient (local slope 0.32 at t=0.06), so the data climbed away
    from the reference and the panel appeared to say the opposite of the text.
    The local slope says it directly -- it rises to 0.79 near t=0.5 and then
    falls monotonically, 0.77 -> 0.64 across the fit window, heading for 1/2."""
    use_paper_style()
    fig, ax = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.35))

    ts, Ls = _np(t_sei), _np(L_sei)
    tp, Lp = _np(t_pdm), _np(L_pdm)
    fit = _np(sei_fit)
    m_fit = ts >= t_fit_lo

    # (a) SEI reaction-diffusion: capacity fade ~ SEI thickness -------------
    ax[0].plot(ts, Ls, "C0", lw=1.6, label="SEI film")
    # the fit is only drawn where it was identified: extrapolated back to t=0 it
    # runs to L = -0.3, i.e. outside the data and outside its own window
    ax[0].plot(ts[m_fit], fit[m_fit], "k--", lw=1.1,
               label=fr"fit on $t\geq{t_fit_lo:g}$, $\alpha={alpha_sei:.2f}$")
    ax[0].axvspan(t_fit_lo, ts[-1], color="0.92", zorder=0)
    ax[0].set(xlabel="time $t$ (model units)",
              ylabel="SEI thickness $L(t)$ (model units)")
    ax[0].legend(loc="lower right", fontsize=7.0)
    panel_label(ax[0], "(a)")

    # (b) PDM film growth: passivation --------------------------------------
    ax[1].plot(tp, Lp, "C2", lw=1.6)
    ax[1].axhline(L_ss, ls=":", color="0.35", lw=1.0)
    # labelled on the curve rather than in a legend: at this width a two-entry
    # legend is wider than the axes and its handles hang past the left spine
    ax[1].text(tp[-1], Lp[-1] * 0.93, "PDM oxide film ", color="C2",
               fontsize=7.0, ha="right", va="top")
    ax[1].text(tp[1], L_ss * 0.978, fr" steady thickness $L_{{ss}}={L_ss:.2f}$",
               color="0.35", fontsize=7.0, ha="left", va="top")
    ax[1].set(xlabel="time $t$ (model units)",
              ylabel="film thickness $L(t)$ (model units)")
    ax[1].set_ylim(top=float(L_ss) * 1.07)
    panel_label(ax[1], "(b)")

    # (c) local growth-law slope: SEI descends toward 1/2, PDM toward 0 ------
    def local_slope(t, L):
        m = (t > 0) & (L > 0)
        lt, lL = np.log(t[m]), np.log(L[m])
        return t[m], np.gradient(lL, lt)

    t_s, s_s = local_slope(ts, Ls)
    t_p, s_p = local_slope(tp, Lp)
    ax[2].axvspan(t_fit_lo, t_s[-1], color="0.92", zorder=0)
    ax[2].axhline(1.0, ls="--", color="0.35", lw=1.0)
    ax[2].axhline(0.5, ls=":", color="0.35", lw=1.0)
    ax[2].semilogx(t_s, s_s, "C0", lw=1.6)
    ax[2].semilogx(t_p, s_p, "C2", lw=1.6)
    ax[2].set_xlim(t_s[0], t_s[-1])
    ax[2].set_ylim(0, 1.15)
    # references labelled at the right, where neither curve runs through them
    ax[2].text(t_s[-1], 1.015, "reaction-limited, 1 ", fontsize=7.0,
               color="0.35", ha="right", va="bottom")
    ax[2].text(t_s[-1], 0.515, r"Fickian, $\frac{1}{2}$ ", fontsize=7.0,
               color="0.35", ha="right", va="bottom")
    # the curves are labelled on themselves: a two-entry legend in a 2 in panel
    # sits on whichever curve it is placed over
    i_s, i_p = int(s_s.argmax()), int(s_p.argmax())
    ax[2].text(t_s[i_s], s_s[i_s] + 0.045, "SEI film", color="C0",
               fontsize=7.0, ha="center", va="bottom")
    ax[2].text(t_p[i_p], s_p[i_p] + 0.045, "PDM oxide film", color="C2",
               fontsize=7.0, ha="center", va="bottom")
    ax[2].set(xlabel="time $t$ (model units)",
              ylabel=r"local slope $\mathrm{d}\ln L/\mathrm{d}\ln t$")
    panel_label(ax[2], "(c)")

    fig.tight_layout(w_pad=1.9)
    out = save_paper_figure(fig, out_path)
    plt.close(fig)
    return out


def plot_fick(x_semi, x_gauss, profiles, t_snapshots, t, uptake, uptake_fit,
              alpha_uptake, out_path):
    """Fick's-law concentration profiles + parabolic diffusion-limited uptake."""
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    fig.suptitle("Fick's-law diffusion systems (baseline: normal / alpha=1)",
                 fontsize=12, fontweight="bold")

    # (a) semi-infinite erfc profiles at several times
    for ts, c in zip(t_snapshots, profiles["semi_infinite"]):
        ax[0].plot(_np(x_semi), _np(c), lw=1.8, label=f"t={ts:g}")
    ax[0].set(title="(a) Semi-infinite, fixed surface (erfc)",
              xlabel="depth x", ylabel="concentration c")
    ax[0].legend(fontsize=8, frameon=False)

    # (b) Gaussian point source spreading
    for ts, c in zip(t_snapshots, profiles["gaussian"]):
        ax[1].plot(_np(x_gauss), _np(c), lw=1.8, label=f"t={ts:g}")
    ax[1].set(title="(b) Instantaneous source (Gaussian)",
              xlabel="position x", ylabel="concentration c")
    ax[1].legend(fontsize=8, frameon=False)

    # (c) diffusion-limited uptake ~ sqrt(t): identified alpha ~ 0.5
    ax[2].plot(_np(t), _np(uptake), "C0", lw=2, label="uptake (Fick)")
    ax[2].plot(_np(t), _np(uptake_fit), "k--", lw=1.3,
               label=fr"fit $\alpha$={alpha_uptake:.2f} ($\to\frac{{1}}{{2}}$)")
    ax[2].set(title="(c) Diffusion-limited uptake = parabolic law",
              xlabel="time t", ylabel="cumulative uptake")
    ax[2].legend(fontsize=8, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# One meaning per colour, fixed across every figure of the battery manuscript:
# the Okabe-Ito colour-blind-safe hues, so the four mechanisms stay separable
# under deuteranopia. Where two series still overlap, the plotting code adds a
# redundant marker or dash pattern rather than leaning on hue alone.
_MECH_COLORS = {
    "SEI": "#0072B2",            # blue
    "plating": "#D55E00",        # vermillion
    "cracks": "#009E73",         # bluish green
    "LAM": "#E69F00",            # orange
}
NEUTRAL = "0.20"                 # schematic strokes: carry no meaning
DEG = "°"                   # "$^\\circ$C" renders as "25 ° C"; this does not

# The ensemble is keyed by directory names, which are fine in a file system and
# wrong on an axis: "plating_pr", "SEI_ec" and "cold_10C_fast" were printed as
# tick labels with nothing defining "pr", "ec" or -- worst in a battery paper --
# "10C", which reads as a 10C rate rather than 10 degrees Celsius.
CONFIG_LABELS = {
    "SEI_ec": "SEI (EC-limited)",
    "SEI_solvent": "SEI (solvent-diffusion)",
    "plating_irr": "plating (irreversible)",
    "plating_pr": "plating (partly reversible)",
    "LAM_strong": "LAM (strong)",
    "LAM_mild": "LAM (mild)",
    "cracks_strong": "cracking (strong)",
    "cracks_mild": "cracking (mild)",
}
CONDITION_LABELS = {
    "cold_10C_fast": "10 " + DEG + "C\nfast charge",
    "hot_45C": "45 " + DEG + "C",
    "mild_25C": "25 " + DEG + "C",
}
MECH_LABELS = {"SEI": "SEI", "plating": "plating", "cracks": "cracking",
               "LAM": "LAM"}


def _config_label(key):
    return CONFIG_LABELS.get(key, key.replace("_", " "))


def _condition_label(key):
    return CONDITION_LABELS.get(key, key.replace("_", " "))


def mirror_figure(paths, *dirs):
    """Copy generated figure files into each manuscript figure directory.

    Those directories used to be refreshed by hand, which is how two figures
    drifted away from their generators and ended up with no generator at all.
    """
    import shutil
    if isinstance(paths, str):
        paths = [paths]
    out = []
    for p in paths:
        for ext in (".pdf", ".png"):
            src = os.path.splitext(p)[0] + ext
            if not os.path.exists(src):
                continue
            out.append(src)
            for d in dirs:
                if d:
                    os.makedirs(d, exist_ok=True)
                    shutil.copy2(src, os.path.join(d, os.path.basename(src)))
    return out


def plot_oxford(cells, out_path):
    """Oxford dataset: capacity retention (8 cells) + clean low-rate dV/dQ (DVA)
    evolving over aging for one cell.

    Eight tightly bundled series cannot be told apart by hue alone, so the cells
    are drawn as four colours times two dash patterns. Panel (b) gets a colour
    bar: it draws about seven curves and the legend named only the first and the
    last, leaving the ones in between unassignable to any cycle number.
    """
    from matplotlib.ticker import FuncFormatter
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    use_paper_style()
    fig, ax = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.65))

    cyc_fmt = FuncFormatter(lambda v, _: f"{int(v):,}")
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for i, (name, d) in enumerate(sorted(cells.items())):
        cap = _np(d["capacity_mAh"])
        ax[0].plot(_np(d["cycle"]), cap / cap[0], lw=1.0, alpha=0.9,
                   color=palette[i % len(palette)],
                   ls="-" if i < len(palette) else (0, (4, 1.6)),
                   label=name)
    # Cell5's final checkpoint drops from 0.834 to 0.699 in one 200-cycle step,
    # against under 0.02 for every other cell over the same interval. It is the
    # last record the dataset holds for that cell and is plotted as recorded;
    # flagging it is the alternative to silently deleting a measured point.
    d5 = cells.get("Cell5")
    if d5 is not None:
        cap5 = _np(d5["capacity_mAh"]) / _np(d5["capacity_mAh"])[0]
        c5 = _np(d5["cycle"])
        ax[0].plot([c5[-1]], [cap5[-1]], "o", mfc="none", mec="0.2", mew=0.8,
                   ms=5, zorder=6)
        ax[0].annotate("Cell5, final checkpoint:\nplotted as recorded",
                       xy=(c5[-1], cap5[-1]), xytext=(c5[-1] + 450, 0.723),
                       fontsize=6.3, color="0.25", ha="left", va="center",
                       linespacing=1.25,
                       arrowprops=dict(arrowstyle="->", color="0.45", lw=0.7,
                                       shrinkB=3))
    ax[0].set(xlabel="cycle", ylabel="capacity retention")
    ax[0].set_title("measured capacity retention (8 cells)")
    ax[0].xaxis.set_major_formatter(cyc_fmt)
    ax[0].grid(alpha=0.25)
    ax[0].legend(fontsize=6.3, ncol=2, loc="lower left", labelspacing=0.25)

    key = sorted(cells)[0]
    dva = cells[key]["dva"]
    cycles = [float(c) for c, *_ in dva]
    norm = Normalize(vmin=min(cycles), vmax=max(cycles))
    cmap = plt.cm.viridis
    for (cyc, Q, V, dvdq) in dva:
        ax[1].plot(_np(Q), np.clip(_np(dvdq), -0.02, 0.02) * 1e3,
                   color=cmap(norm(float(cyc))), lw=1.0)
    ax[1].set(xlabel="discharge capacity [mA h]",
              ylabel=r"d$V$/d$Q$ [mV (mA h)$^{-1}$]")
    ax[1].set_title(f"low-rate d$V$/d$Q$ over aging ({key})")
    ax[1].set_ylim(-20, 5)
    ax[1].grid(alpha=0.25)

    for a, letter in zip(ax, ["(a)", "(b)"]):
        panel_label(a, letter, x=-0.20)

    fig.subplots_adjust(left=0.105, right=0.855, top=0.88, bottom=0.19,
                        wspace=0.36)
    cax = fig.add_axes([0.885, 0.19, 0.020, 0.69])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label("cycle", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)
    cbar.ax.yaxis.set_major_formatter(cyc_fmt)
    cbar.outline.set_linewidth(0.6)
    return save_paper_figure(fig, out_path)


def plot_real_clustering(pc, temps, clusters, Z, labels, out_path):
    """Unsupervised structure of real cells: PCA embedding (colour = temperature,
    marker = discovered cluster) + a hierarchical-clustering dendrogram."""
    from scipy.cluster.hierarchy import dendrogram
    tcol = {25: "C0", 35: "C1", 45: "C3"}
    markers = ["o", "s", "^", "D", "v"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.suptitle("Unsupervised structure of real cells "
                 "(capacity + EIS features)", fontsize=12, fontweight="bold")

    pc = _np(pc)
    for i in range(len(temps)):
        ax[0].scatter(pc[i, 0], pc[i, 1], color=tcol.get(temps[i], "0.5"),
                      marker=markers[int(clusters[i]) % len(markers)],
                      s=70, edgecolor="0.2", zorder=3)
    ax[0].set(title="(a) PCA embedding", xlabel="PC1", ylabel="PC2")
    ax[0].grid(alpha=0.3)
    h_t = [plt.Line2D([], [], marker="o", ls="", color=c, label=f"{t}$^\\circ$C")
           for t, c in tcol.items()]
    h_c = [plt.Line2D([], [], marker=markers[k], ls="", color="0.4",
                      label=f"cluster {k}") for k in sorted(set(int(c) for c in clusters))]
    ax[0].legend(handles=h_t + h_c, fontsize=7.5, frameon=False, ncol=2)

    dendrogram(Z, labels=labels, ax=ax[1], leaf_font_size=8,
               color_threshold=0.6 * max(Z[:, 2]))
    ax[1].set(title="(b) hierarchical clustering (Ward)", ylabel="distance")
    ax[1].tick_params(axis="x", labelrotation=90)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_real_features(rows, r_eis, r_shape, out_path):
    """Quantitative real-data analysis: (a) EIS arc growth vs measured capacity
    fade (with correlation), (b) fade-shape exponent by temperature.
    rows: list of dicts with keys temp, fade, dR_ct, exponent.

    Temperature is an ordered variable, so it is drawn on an ordered
    light-to-dark ramp with a distinct marker per level rather than on a
    categorical palette. Panel (a) now carries its least-squares line and the
    95% band around it: the correlation used to be asserted in the panel title
    with no caveat and nothing on the axes a reader could check it against,
    while the caption, the body and the abstract all record that it does not
    survive correction for the six features examined.
    """
    use_paper_style()
    tcol = {25: "#9ecae1", 35: "#4292c6", 45: "#08519c"}
    tmark = {25: "o", 35: "s", 45: "^"}
    fig, ax = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.55))

    x = np.array([r["fade"] * 100 for r in rows], float)
    y = np.array([r["eis_feat"] for r in rows], float)
    for r in rows:
        ax[0].scatter(r["fade"] * 100, r["eis_feat"],
                      color=tcol.get(r["temp"], "0.5"),
                      marker=tmark.get(r["temp"], "o"), s=26,
                      edgecolor="0.25", linewidth=0.5, zorder=3)

    # ordinary least squares plus the 95% band on the mean response; both come
    # straight from the twelve plotted points
    n = len(x)
    if n > 2 and x.std() > 0:
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        yhat = a + b * xs
        resid = y - (a + b * x)
        s = np.sqrt((resid ** 2).sum() / (n - 2))
        sxx = ((x - x.mean()) ** 2).sum()
        se = s * np.sqrt(1.0 / n + (xs - x.mean()) ** 2 / sxx)
        from scipy.stats import t as student_t
        tcrit = float(student_t.ppf(0.975, n - 2))
        ax[0].fill_between(xs, yhat - tcrit * se, yhat + tcrit * se,
                           color="0.75", alpha=0.35, lw=0, zorder=1)
        ax[0].plot(xs, yhat, "--", color="0.35", lw=1.0, zorder=2)
        ax[0].text(0.97, 0.95,
                   f"least squares, $r={r_eis:+.2f}$ ($n={n}$)\n"
                   "not significant after correcting\nfor the six features examined",
                   transform=ax[0].transAxes, ha="right", va="top",
                   fontsize=6.5, color="0.25", linespacing=1.35)
    ax[0].set(xlabel="capacity fade [%]",
              ylabel=r"$R_{\mathrm{ct}}$ at end of life [$\Omega$]")
    ax[0].set_title("end-of-life charge-transfer resistance vs fade")
    ax[0].grid(alpha=0.25)
    ax[0].legend(handles=[plt.Line2D([], [], marker=tmark[t], ls="", color=c,
                                     mec="0.25", mew=0.5, ms=4,
                                     label=f"{t} {DEG}C")
                          for t, c in tcol.items()],
                 fontsize=7, loc="lower left", title="temperature",
                 title_fontsize=7)

    temps = sorted({r["temp"] for r in rows})
    for t in temps:
        vals = [r["exponent"] for r in rows if r["temp"] == t]
        ax[1].scatter([t] * len(vals), vals, color=tcol.get(t, "0.5"),
                      marker=tmark.get(t, "o"), s=26, edgecolor="0.25",
                      linewidth=0.5, zorder=3)
    ax[1].axhline(0.5, ls="--", color="0.15", lw=1.0,
                  label=r"$\sqrt{n}$ (diffusion-limited)")
    ax[1].axhline(1.0, ls=":", color="0.4", lw=1.0, label="linear")
    ax[1].set(xlabel=f"temperature [{DEG}C]", ylabel=r"log-log exponent $\gamma$")
    ax[1].set_title("capacity-loss shape exponent by temperature")
    ax[1].set_xticks(temps)
    ax[1].set_xlim(min(temps) - 5, max(temps) + 5)
    ax[1].set_ylim(0, 1.18)          # the gamma=1.0 reference used to sit half
    ax[1].legend(fontsize=7, loc="lower right")  # clipped by the top spine
    ax[1].grid(alpha=0.25)

    for a, letter in zip(ax, ["(a)", "(b)"]):
        panel_label(a, letter, x=-0.19)

    fig.subplots_adjust(left=0.10, right=0.985, top=0.87, bottom=0.19,
                        wspace=0.30)
    return save_paper_figure(fig, out_path)


def plot_real_overview(cap_dict, eis_cell_states, out_path):
    """Real-cell observables (Zhang 2020): capacity fade + EIS evolution over aging.
    cap_dict: {(T,cell):(cycles,cap)}; eis_cell_states: list of (state, Zre, negZim)."""
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    fig.suptitle("Real cells (Zhang et al.\\ 2020): capacity fade and EIS over aging",
                 fontsize=12, fontweight="bold")
    tcol = {25: "C0", 35: "C1", 45: "C3"}
    seen = set()
    for (T, cell), (cyc, cap) in sorted(cap_dict.items()):
        lbl = f"{T}$^\\circ$C" if T not in seen else None
        seen.add(T)
        ax[0].plot(_np(cyc), _np(cap) / _np(cap)[0], color=tcol.get(T, "0.5"),
                   lw=1.2, alpha=0.8, label=lbl)
    ax[0].set(title="(a) measured capacity fade (12 cells)",
              xlabel="cycle", ylabel="capacity retention")
    ax[0].legend(fontsize=8, frameon=False, title="temperature")

    cmap = plt.cm.viridis
    n = len(eis_cell_states)
    xmax = 0.0
    for i, (st, freq, Zre, negZim) in enumerate(eis_cell_states):
        f, zr, zi = _np(freq), _np(Zre), _np(negZim)
        uf, idx = np.unique(f, return_index=True)     # one sweep (dedupe by freq)
        order = idx[np.argsort(f[idx])[::-1]]         # traverse high -> low freq
        f, zr, zi = f[order], zr[order], zi[order]
        m = f >= 0.2                                  # drop the LF diffusion tail
        ax[1].plot(zr[m], zi[m], "-", lw=1.4, color=cmap(i / max(n - 1, 1)),
                   label=("fresh" if i == 0 else "most aged" if i == n - 1 else None))
        xmax = max(xmax, float(np.percentile(zr[m], 97)))
    ax[1].set(title="(b) measured EIS Nyquist, one cell over aging",
              xlabel=r"$Z'$ [$\Omega$]", ylabel=r"$-Z''$ [$\Omega$]")
    ax[1].set_xlim(0, xmax * 1.05)
    ax[1].set_ylim(-0.05, xmax * 0.75)
    ax[1].legend(fontsize=8, frameon=False)
    ax[1].grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_nyquist(scenarios, out_path, freqs=None):
    """Nyquist plot of mechanism-representative EIS scenarios.

    scenarios: list of (label, Z_complex_array); freqs: the sweep, in Hz, used
    to place decade markers.

    A Nyquist plot is only readable at equal scaling -- arc size is the whole
    claim here -- so the axes are locked to 1:1 rather than stretched to fill
    the box. The LLI and SEI traces run close together over much of their
    length, so each series carries its own marker and dash pattern; hue alone
    would separate them for no reader with deuteranopia and for nobody at all
    in greyscale.
    """
    use_paper_style()
    styles = [("o", "-"), ("s", "--"), ("^", "-."), ("D", (0, (1.5, 1.5)))]
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 3.15))

    xs, ys = [], []
    for i, (label, Z) in enumerate(scenarios):
        Z = _np(Z)
        m, ls = styles[i % len(styles)]
        x, y = Z.real * 1e3, -Z.imag * 1e3
        xs.append(x); ys.append(y)
        # the baseline is a wide pale halo so that any scenario coinciding with
        # it stays visible instead of being buried under the trace on top
        if i == 0:
            ax.plot(x, y, "-", lw=3.4, color="0.72", zorder=1)
        ax.plot(x, y, marker=m, ls=ls, ms=2.6, lw=1.1, markevery=2,
                label=label, zorder=2 + i)

    xall = np.concatenate(xs); yall = np.concatenate(ys)
    ymax = float(yall.max())
    # the box used to run to -10 mOhm against data spanning 0 to 15.4, so 40% of
    # it held nothing and negative -Z'' had no physical content here; the extra
    # headroom above the data is for the legend, which otherwise lands on an arc
    ax.set_ylim(-0.8, ymax * 1.62)
    ax.set_xlim(0.0, float(xall.max()) * 1.04)
    ax.set_aspect("equal", adjustable="box")

    # frequency is the axis a Nyquist plot hides; mark a few decades on the
    # baseline so the reader can tell which end of the arc is which
    if freqs is not None and len(scenarios):
        f = _np(freqs)
        Z0 = _np(scenarios[0][1])
        for target, (dx, dy, ha, va) in [(1e3, (13, 1, "left", "center")),
                                         (1e1, (0, 9, "center", "bottom")),
                                         (1e-1, (11, -2, "left", "center"))]:
            if not (f.min() <= target <= f.max()):
                continue
            k = int(np.argmin(np.abs(np.log10(f) - np.log10(target))))
            px, py = Z0.real[k] * 1e3, -Z0.imag[k] * 1e3
            ax.plot([px], [py], "|", color="0.25", ms=5, mew=1.0, zorder=8)
            ax.annotate(f"{target:g} Hz", xy=(px, py), xytext=(dx, dy),
                        textcoords="offset points", fontsize=6.3, color="0.3",
                        ha=ha, va=va,
                        arrowprops=dict(arrowstyle="-", color="0.55", lw=0.6,
                                        shrinkB=1))

    ax.set(xlabel=r"$Z'$ [m$\Omega$]", ylabel=r"$-Z''$ [m$\Omega$]")
    ax.legend(fontsize=7, loc="upper left", ncol=2, columnspacing=1.4)
    ax.grid(alpha=0.25)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.97, bottom=0.13)
    return save_paper_figure(fig, out_path)


def plot_fraction_r2(mechs, r2_cap, r2_dvdq, out_path):
    """Per-mechanism R² of fraction recovery: capacity vs capacity+dV/dQ."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    y = np.arange(len(mechs))
    ax.barh(y - 0.2, np.clip(_np(r2_cap), -0.1, 1), 0.4, color="0.6",
            label="capacity only")
    ax.barh(y + 0.2, np.clip(_np(r2_dvdq), -0.1, 1), 0.4, color="C0",
            label="capacity + dV/dQ")
    ax.set_yticks(y); ax.set_yticklabels(mechs)
    ax.axvline(0, color="k", lw=0.8)
    ax.set(title="Mechanism-fraction recovery (leave-one-out $R^2$)",
           xlabel=r"$R^2$")
    ax.set_xlim(-0.1, 1.0)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pybamm_curves(cells, out_path):
    """Charge/discharge (cycler-like) curves and clean dV/dQ (DVA) signatures."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    fig.suptitle("Voltage curves and differential-voltage (dV/dQ) signatures",
                 fontsize=12, fontweight="bold")

    # (a) raw discharge V(Q) at BOL/mid/EOL for a representative degrading cell
    cell = max(cells, key=lambda c: 1.0 - float(c["retention"][-1]))
    for nm, style, lab in [("vq_first", "C0", "begin-of-life"),
                           ("vq_mid", "C1", "mid-life"), ("vq_last", "C3", "end-of-life")]:
        if nm in cell:
            Q, V = cell[nm]
            ax[0].plot(_np(Q), _np(V), style, lw=1.6, label=lab)
    ax[0].set(title=f"(a) discharge curves over life\n({cell['config']})",
              xlabel="discharge capacity [A.h]", ylabel="voltage [V]")
    ax[0].legend(fontsize=8, frameon=False)

    # (b) clean low-rate dV/dQ (DVA), one representative cell per mechanism
    seen = set()
    for c in sorted(cells, key=lambda z: z["dominant"]):
        m = c["dominant"]
        if m in seen or m not in _MECH_COLORS:
            continue
        seen.add(m)
        Q, y = _np(c["dvdq_Q"]), _np(c["dvdq"])
        ax[1].plot(Q, np.clip(y, -8, 1), color=_MECH_COLORS[m], lw=1.6, label=m)
    ax[1].set(title="(b) dV/dQ (DVA) by mechanism", xlabel="discharge capacity [A.h]",
              ylabel="dV/dQ [V/A.h]")
    ax[1].set_ylim(-6, 0.5)
    ax[1].legend(fontsize=8, frameon=False)

    # (c) all cells' dV/dQ on normalized capacity, colored by mechanism (clustering?)
    for c in cells:
        Q, y = _np(c["dvdq_Q"]), _np(c["dvdq"])
        if Q.size < 5:
            continue
        x = (Q - Q.min()) / (Q.max() - Q.min() + 1e-9)
        o = np.argsort(x)
        ax[2].plot(x[o], np.clip(y[o], -8, 1), color=_MECH_COLORS.get(c["dominant"], "0.6"),
                   alpha=0.5, lw=1.0)
    ax[2].set(title="(c) dV/dQ, all cells (color = mechanism)",
              xlabel="normalized capacity", ylabel="dV/dQ [V/A.h]")
    ax[2].set_ylim(-6, 0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_confusion_pair(C1, acc1, title1, C2, acc2, title2, labels, out_path,
                        protocol="leave-one-configuration-out"):
    """Two confusion matrices side by side (e.g. magnitude-only vs +shape).

    `protocol` names the validation scheme. It is a parameter rather than a
    constant because it used to be hardcoded to "leave-one-configuration-out":
    a caller scoring by plain leave-one-out then produced a figure asserting a
    protocol it had not used. It is now recorded on the axes rather than in a
    baked-in title, which additionally said "from capacity observable" over a
    right-hand panel scored on capacity AND dV/dQ.

    Accuracies print to one decimal. Rounded to whole percent the two panels
    read 38% and 21% while the abstract, Section 3.2, the caption and the
    conclusion all quote 9/24 = 37.5% and 5/24 = 20.8%; a reader comparing the
    figure with the text found four places where they disagreed.
    """
    use_paper_style()
    fig, ax = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.85))
    n_total = int(C1.sum())
    for a, C, acc, ttl, letter in [(ax[0], C1, acc1, title1, "(a)"),
                                   (ax[1], C2, acc2, title2, "(b)")]:
        Cn = C / np.clip(C.sum(1, keepdims=True), 1, None)
        im = a.imshow(Cn, cmap="Blues", vmin=0, vmax=1)
        a.set_xticks(range(len(labels)))
        a.set_xticklabels([MECH_LABELS.get(l, l) for l in labels],
                          fontsize=7.5, rotation=30, ha="right",
                          rotation_mode="anchor")
        a.set_yticks(range(len(labels)))
        a.set_yticklabels([MECH_LABELS.get(l, l) for l in labels], fontsize=7.5)
        a.set_xlabel("predicted mechanism")
        a.set_ylabel("true mechanism")
        n_correct = int(np.trace(C))
        a.set_title(f"{ttl}\naccuracy = {n_correct}/{n_total} = {acc:.1%}",
                    fontsize=8)
        a.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
        a.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        a.grid(which="minor", color="white", lw=0.9)
        a.tick_params(which="minor", length=0)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if C[i].sum():
                    a.text(j, i, f"{C[i, j]}", ha="center", va="center",
                           color="white" if Cn[i, j] > 0.5 else "black",
                           fontsize=8)
        panel_label(a, letter, x=-0.26)

    fig.subplots_adjust(left=0.115, right=0.79, top=0.80, bottom=0.26,
                        wspace=0.42)
    # the shading was never explained anywhere: it is the row fraction, i.e.
    # how many of that true mechanism's cells landed in that column
    cax = fig.add_axes([0.835, 0.26, 0.022, 0.54])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("share of that true\nmechanism's cells", fontsize=7,
                   linespacing=1.3)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.6)
    fig.text(0.45, 0.035, f"scoring: {protocol}", ha="center", va="bottom",
             fontsize=7, color="0.3")
    return save_paper_figure(fig, out_path)


def plot_pybamm_overview(cells, out_path):
    """Four-panel overview of the PyBaMM aging ensemble.
    cells: list of dicts with keys cycle, retention, dominant, contrib(dict),
    dvdq_Q, dvdq, config, condition.

    One mechanism legend for the whole figure, below the panels: it used to be
    repeated three times, and in (b) it was drawn on top of the LAM (mild) bar
    with an orange swatch over an orange bar.
    """
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap

    use_paper_style()
    fig, ax = plt.subplots(2, 2, figsize=(TEXTWIDTH_IN, 4.55))

    mechs = ["SEI", "plating", "cracks", "LAM"]
    curve_alpha = 0.8

    # (a) capacity fade, coloured by ground-truth dominant mechanism
    for c in cells:
        ax[0, 0].plot(_np(c["cycle"]), _np(c["retention"]),
                      color=_MECH_COLORS.get(c["dominant"], "0.6"),
                      alpha=curve_alpha, lw=1.0)
    ax[0, 0].set_title("(a) capacity retention\n"
                       "(colour = true dominant mechanism)",
                       fontsize=7.8, loc="left")
    ax[0, 0].set(xlabel="cycle", ylabel="capacity retention")
    ax[0, 0].set_ylim(0, 1.03)
    ax[0, 0].grid(alpha=0.25)
    # One LAM cell fades to 5.2% by cycle 98 and is then flat to 200: the solver
    # pins it there. It is a real record in the ensemble and is kept, but a
    # collapsing curve running dead flat needs saying, not leaving to be guessed.
    pinned = [c for c in cells
              if c["config"] == "LAM_strong" and c["condition"] == "cold_10C_fast"]
    if pinned:
        p = pinned[0]
        r, cyc = _np(p["retention"]), _np(p["cycle"])
        j = int(np.argmax(r <= r[-1] + 1e-9))
        ax[0, 0].plot([cyc[j]], [r[j]], "v", color=_MECH_COLORS["LAM"], ms=4,
                      mec="0.2", mew=0.5, zorder=5)
        ax[0, 0].annotate(f"pinned at {r[-1]:.0%}\nfrom cycle {int(cyc[j])}",
                          xy=(cyc[j], r[j]), xytext=(cyc[j] + 22, 0.20),
                          fontsize=6.3, color="0.25", va="center",
                          linespacing=1.25,
                          arrowprops=dict(arrowstyle="->", color="0.45",
                                          lw=0.7, shrinkB=2))

    # (b) mechanism contribution by configuration (averaged over conditions)
    configs = sorted({c["config"] for c in cells})
    frac = {m: [] for m in mechs}
    for cfg in configs:
        sub = [c for c in cells if c["config"] == cfg]
        tot = np.zeros(len(mechs))
        for c in sub:
            tot += np.array([c["contrib"][m] for m in mechs])
        tot = tot / max(tot.sum(), 1e-12)
        for i, m in enumerate(mechs):
            frac[m].append(tot[i])
    labels = [_config_label(cfg) for cfg in configs]
    y = np.arange(len(configs))
    bottom = np.zeros(len(configs))
    for m in mechs:
        ax[0, 1].barh(y, frac[m], left=bottom, height=0.72,
                      color=_MECH_COLORS[m])
        bottom += np.array(frac[m])
    ax[0, 1].set_yticks(y)
    ax[0, 1].set_yticklabels(labels, fontsize=6.5)
    ax[0, 1].set_title("(b) ground-truth mechanism split\nby configuration",
                       fontsize=7.8, loc="left")
    ax[0, 1].set(xlabel="fraction of capacity loss")
    ax[0, 1].set_xlim(0, 1)

    # (c) normalised fade shape: x = cycle/max, y = capacity loss / final loss
    for c in cells:
        loss = 1.0 - _np(c["retention"])
        if loss[-1] < 1e-4:
            continue                        # essentially non-degrading cells
        x = _np(c["cycle"]) / _np(c["cycle"])[-1]
        yy = loss / loss[-1]
        ax[1, 0].plot(x, yy, color=_MECH_COLORS.get(c["dominant"], "0.6"),
                      alpha=curve_alpha, lw=1.0)
    ax[1, 0].plot([0, 1], [0, 1], ":", color="0.15", lw=1.0, label="linear")
    ax[1, 0].plot(np.linspace(0, 1, 50), np.sqrt(np.linspace(0, 1, 50)),
                  "--", color="0.15", lw=1.0, label=r"$\sqrt{n}$ (diffusion)")
    ax[1, 0].set_title("(c) normalised fade shape\n"
                       "(colour = true dominant mechanism)",
                       fontsize=7.8, loc="left")
    ax[1, 0].set(xlabel="cycle / final cycle",
                 ylabel="capacity loss / final loss")
    ax[1, 0].set_xlim(0, 1); ax[1, 0].set_ylim(0, 1.02)
    ax[1, 0].grid(alpha=0.25)
    ax[1, 0].legend(loc="lower right", fontsize=7, frameon=True,
                    framealpha=0.85, edgecolor="none")

    # (d) dominance map: configuration x condition -> dominant mechanism
    # ordered by temperature, not alphabetically: an ordered variable should
    # read left to right in its own order
    cond_order = ["cold_10C_fast", "mild_25C", "hot_45C"]
    conds = sorted({c["condition"] for c in cells},
                   key=lambda k: (cond_order.index(k) if k in cond_order
                                  else len(cond_order), k))
    midx = {m: i for i, m in enumerate(mechs)}
    grid = np.full((len(configs), len(conds)), np.nan)
    for c in cells:
        grid[configs.index(c["config"]),
             conds.index(c["condition"])] = midx.get(c["dominant"], np.nan)
    cmap = ListedColormap([_MECH_COLORS[m] for m in mechs])
    ax[1, 1].imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=len(mechs) - 1)
    ax[1, 1].set_xticks(range(len(conds)))
    ax[1, 1].set_xticklabels([_condition_label(c) for c in conds], fontsize=6.5)
    ax[1, 1].set_yticks(range(len(configs)))
    ax[1, 1].set_yticklabels(labels, fontsize=6.5)
    # cell boundaries: without them the configuration x condition grid is
    # invisible and neighbouring cells of one colour read as a single block
    ax[1, 1].set_xticks(np.arange(-0.5, len(conds), 1), minor=True)
    ax[1, 1].set_yticks(np.arange(-0.5, len(configs), 1), minor=True)
    ax[1, 1].grid(which="minor", color="white", lw=1.1)
    ax[1, 1].tick_params(which="minor", length=0)
    ax[1, 1].set_title("(d) dominant mechanism\nby configuration and condition",
                       fontsize=7.8, loc="left")

    handles = [mpatches.Patch(color=_MECH_COLORS[m], alpha=curve_alpha,
                              label=MECH_LABELS[m]) for m in mechs]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5,
               title="mechanism", title_fontsize=7.5,
               bbox_to_anchor=(0.5, -0.005))

    fig.subplots_adjust(left=0.115, right=0.985, top=0.90, bottom=0.15,
                        hspace=0.75, wspace=0.62)
    return save_paper_figure(fig, out_path)


def plot_dynamic_range(t, slope, alphas, D_grid, resolved_frac,
                       exp_window=(1.0, 3.0), out_path=None,
                       t_hi_end=None, t_lo_start=None, d_required=None):
    """Identifiability vs dynamic range for a two-order memory.
    (a) local slope over many decades approaches the true orders only slowly;
    (b) fraction of the order range resolved vs observation-window width.

    `t_hi_end`/`t_lo_start` bracket the transition whose width is the section's
    headline number; `d_required` is the window width that resolves 90%.  Both
    were quoted in the text and marked nowhere in the figure.

    The "typical experiment" band is drawn with `axvspan`, in DATA coordinates.
    It used to be an `axhspan` whose xmin/xmax are axes FRACTIONS, which placed
    a band advertised as 1 to 3 decades at 1.61 to 3.54 -- putting the 3-decade
    requirement inside the typical range instead of at its edge."""
    a_lo, a_hi = min(alphas), max(alphas)
    use_paper_style()
    fig, ax = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.6))

    tn, sn = _np(t), _np(slope)
    ax[0].axhline(a_hi, ls="--", color="0.35", lw=0.9)
    ax[0].axhline(a_lo, ls="--", color="0.35", lw=0.9)
    ax[0].semilogx(tn, sn, "C3", lw=1.5)
    ax[0].set_xlim(tn[0], tn[-1])
    ax[0].set_ylim(a_lo - 0.17, a_hi + 0.16)
    # labels on the right, clear of the curve's own plateau at the left edge
    # each reference is labelled at the end of the axis where the curve is on
    # the OTHER plateau, so neither label lands on the curve or on the other
    ax[0].text(tn[-1], a_hi + 0.012, f"true order {a_hi} ", fontsize=7.0,
               color="0.3", ha="right", va="bottom")
    ax[0].text(tn[0], a_lo + 0.012, f" true order {a_lo}", fontsize=7.0,
               color="0.3", ha="left", va="bottom")

    # the transition width, which is the number the section rests on
    if t_hi_end is not None and t_lo_start is not None:
        y_ann = a_lo - 0.075
        ax[0].annotate("", xy=(t_hi_end, y_ann), xytext=(t_lo_start, y_ann),
                       arrowprops=dict(arrowstyle="<->", color="0.25", lw=0.9))
        ax[0].axvline(t_hi_end, color="0.6", ls=":", lw=0.8)
        ax[0].axvline(t_lo_start, color="0.6", ls=":", lw=0.8)
        width = float(np.log10(t_lo_start / t_hi_end))
        ax[0].text(np.sqrt(t_hi_end * t_lo_start), y_ann - 0.012,
                   f"{width:.2f} decades of transition", fontsize=7.0,
                   color="0.25", ha="center", va="top")
    ax[0].set(xlabel="time $t$ (model units, log scale)",
              ylabel=r"local slope $\mathrm{d}\ln u/\mathrm{d}\ln t$")
    panel_label(ax[0], "(a)")

    Dg, rf = _np(D_grid), _np(resolved_frac)
    ax[1].axvspan(exp_window[0], exp_window[1], color="0.88", zorder=0,
                  label=f"typical experiment, {exp_window[0]:g} to "
                        f"{exp_window[1]:g} decades")
    ax[1].axhline(0.9, ls="--", color="0.35", lw=0.9)
    ax[1].plot(Dg, rf, "o-", color="C0", ms=3.6)
    if d_required is not None:
        ax[1].plot([d_required], [0.9], "*", color="C3", ms=8, zorder=5)
        ax[1].annotate(f"$D\\approx{d_required:.1f}$ decades\nresolves 90%",
                       xy=(d_required, 0.9), xytext=(d_required + 0.7, 0.60),
                       fontsize=7.0, color="C3",
                       arrowprops=dict(arrowstyle="->", color="C3", lw=0.8))
    ax[1].set(xlabel="observation window width\n(decades, centered on crossover)",
              ylabel="fraction of the order range resolved")
    ax[1].set_xlim(Dg.min() - 0.2, Dg.max() + 0.2)
    ax[1].set_ylim(0.3, 1.06)
    ax[1].legend(loc="lower right")
    panel_label(ax[1], "(b)", x=-0.09)

    fig.tight_layout(w_pad=2.2)
    out = save_paper_figure(fig, out_path)
    plt.close(fig)
    return out


def plot_keystone(orders_ref, w_ref, twin, t, u_ref, u_twin, gap, floor,
                  D_window, D_grid, gap_curves, out_path):
    """Constructive non-identifiability: two spectra a spectral gap apart whose
    data are indistinguishable below the noise floor.
    (a) the two spectra; (b) their responses and residual; (c) the gap that
    survives as a function of window width and noise floor."""
    a_lo, a_hi, w_lo = twin[0], twin[1], twin[2]
    use_paper_style()
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 5.9))
    # (a)/(b) and (c)/(d) are stacked pairs sharing an abscissa, so they are set
    # tight; only the block of four is separated from (e)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.15, 2.05], hspace=0.30,
                          left=0.105, right=0.985, top=0.965, bottom=0.075)
    top = gs[0].subgridspec(2, 2, height_ratios=[2.0, 1.05], hspace=0.24,
                            wspace=0.30)
    ax_s = fig.add_subplot(top[0, 0])
    ax_c = fig.add_subplot(top[1, 0], sharex=ax_s)
    ax_u = fig.add_subplot(top[0, 1])
    ax_r = fig.add_subplot(top[1, 1], sharex=ax_u)
    ax_g = fig.add_subplot(gs[1])

    # (a) the two spectra ---------------------------------------------------
    ax_s.stem(list(orders_ref), list(w_ref), linefmt="C0-", markerfmt="C0o",
              basefmt=" ", label="reference  $w_A$")
    ax_s.stem([a_lo, a_hi], [w_lo, 1 - w_lo], linefmt="C3--", markerfmt="C3s",
              basefmt=" ", label="twin  $w_B$")
    mean_ref = float(np.dot(orders_ref, w_ref))
    mean_twin = float(a_lo * w_lo + a_hi * (1 - w_lo))
    ax_s.axvline(mean_ref, color="0.4", ls=":", lw=1.0)
    # the body compares 0.500 against 0.494; printing only "shared mean 0.50"
    # asserted an equality the text does not claim
    ax_s.text(0.985, 0.97,
              f"mean order {mean_ref:.3f} (ref),\n{mean_twin:.3f} (twin)",
              transform=ax_s.transAxes, fontsize=7.0, color="0.3",
              ha="right", va="top")
    ax_s.set(ylabel=r"weight $w(\alpha)$", xlim=(0, 1), ylim=(0, 1.12))
    ax_s.tick_params(labelbottom=False)
    ax_s.legend(loc="upper left", fontsize=7)
    panel_label(ax_s, "(a)")

    # (b) the W1 gap IS the area between the two CDFs -- draw it, don't assert it
    knots = np.linspace(0, 1, 501)
    cdf_a = np.array([np.sum(np.asarray(w_ref)[np.asarray(orders_ref) <= k])
                      for k in knots])
    cdf_b = np.array([np.sum(np.array([w_lo, 1 - w_lo])[np.array([a_lo, a_hi]) <= k])
                      for k in knots])
    ax_c.fill_between(knots, cdf_a, cdf_b, color="0.6", alpha=0.55,
                      label=f"$W_1$ = {gap:.1%} of the axis")
    ax_c.plot(knots, cdf_a, "C0", lw=1.2)
    ax_c.plot(knots, cdf_b, "C3--", lw=1.2)
    ax_c.set(xlabel=r"order $\alpha$", ylabel="CDF", ylim=(-0.02, 1.16))
    ax_c.legend(loc="upper left", fontsize=7)
    panel_label(ax_c, "(b)")

    # (c) responses ----------------------------------------------------------
    tn, uA, uB = _np(t), _np(u_ref), _np(u_twin)
    Phi = np.stack([uB, np.ones_like(uB)], axis=1)
    coef, *_ = np.linalg.lstsq(Phi, uA, rcond=None)
    uB_fit = Phi @ coef
    ax_u.loglog(tn, uA, "C0", lw=2.8, alpha=0.45, label="data from $w_A$")
    ax_u.loglog(tn, uB_fit, "C3--", lw=1.1, label="best fit of $w_B$")
    ax_u.set(ylabel="$u(t)$")
    ax_u.text(0.5, 1.03, f"{D_window:.0f}-decade window", transform=ax_u.transAxes,
              fontsize=7, color="0.35", ha="center", va="bottom")
    ax_u.legend(loc="lower right", fontsize=7)
    ax_u.tick_params(labelbottom=False)
    panel_label(ax_u, "(c)")

    # (d) residual -----------------------------------------------------------
    rel = (uA - uB_fit) / np.sqrt(np.mean(uA ** 2))
    rms = float(np.sqrt(np.mean(rel ** 2)))
    peak = float(np.abs(rel).max())
    ax_r.axhspan(-floor, floor, color="0.85", zorder=0)
    ax_r.semilogx(tn, rel, "C3", lw=1.0, zorder=3)
    ax_r.axhline(0, color="k", lw=0.5)
    lim = max(1.6 * floor, 1.95 * peak)
    ax_r.set(xlabel="time $t$ (model units)", ylabel="relative residual",
             ylim=(-lim, lim))
    ax_r.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    # the criterion is an RMS, not a pointwise bound: the curve leaves the band
    # on both ends by construction, and saying only "RMS" invited the reader to
    # read the band as an envelope it never was
    ax_r.text(0.5, 0.98,
              f"RMS {rms:.2%} is the criterion; the $\\pm${floor:.0%} band\n"
              f"is not a pointwise bound (peak {peak:.1%})",
              transform=ax_r.transAxes, fontsize=7.0, color="0.25",
              ha="center", va="top")
    panel_label(ax_r, "(d)")

    # (e) surviving gap vs window width, per noise floor --------------------
    styles = [("C0", "o", "-"), ("C1", "s", "--"), ("C2", "^", "-."),
              ("C3", "D", ":")]
    for (fl, curve), (c, mk, ls) in zip(gap_curves, styles):
        ax_g.plot(_np(D_grid), _np(curve), marker=mk, ls=ls, ms=3.6, lw=1.2,
                  color=c, label=f"noise floor {fl:.1%}")
    ax_g.axvspan(1.0, 3.0, color="0.88", zorder=0)
    ax_g.text(2.0, 0.012, "typical experiment", ha="center", fontsize=7.0,
              color="0.35")
    ax_g.set(xlabel="observation window (decades)",
             ylabel=r"max admissible $W_1$ gap")
    ax_g.set_ylim(bottom=0)
    # every gap quoted in the text is a percentage of the order axis; so is this
    ax_g.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_g.legend(loc="upper right", ncol=2, fontsize=7)
    panel_label(ax_g, "(e)", x=-0.078)

    out = save_paper_figure(fig, out_path)
    plt.close(fig)
    return out


def plot_delevie(w, Zmag, wc, alpha_hf, rms_hf, w_ex, ex, geoms, floor,
                 dec_grid, dev_curve, out_path):
    """External analytical validation on de Levie's porous electrode.
    (a) |Z| with its two closed-form asymptotes; (b) the local exponent sweeping
    1/2 -> 1, with the recovered high-frequency order; (c) pores of different
    geometry that are indistinguishable until the window reaches the crossover."""
    use_paper_style()
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 5.6))
    # (a) and (b) share an axis, so they are set close enough to read as one
    # stacked pair; the tick numbers under (b) then belong to both, instead of
    # sitting an inch of blank figure away from the panel they also label
    gs = fig.add_gridspec(3, 1, height_ratios=[2.1, 1.5, 2.3],
                          hspace=0.14, top=0.965, bottom=0.075,
                          left=0.135, right=0.985)
    ax_z = fig.add_subplot(gs[0])
    ax_e = fig.add_subplot(gs[1], sharex=ax_z)
    ax_g = fig.add_subplot(gs[2])
    # the third panel is a different abscissa; give it its own breathing room
    pos = ax_g.get_position()
    ax_g.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.80])

    wn, Zn = _np(w), _np(Zmag)
    # (a) magnitude with both analytic asymptotes ---------------------------
    ax_z.loglog(wn, Zn, "C0", lw=2.6, alpha=0.5, label="de Levie pore")
    hf = Zn[-1] * (wn / wn[-1]) ** -0.5
    lf = Zn[0] * (wn / wn[0]) ** -1.0
    ax_z.loglog(wn, hf, "C3--", lw=1.1, label=r"$(j\omega)^{-1/2}$  (exact)")
    ax_z.loglog(wn, lf, "C2:", lw=1.1, label=r"$(j\omega)^{-1}$  (capacitive)")
    ax_z.axvline(wc, color="0.4", ls=":", lw=1.0)
    ax_z.text(wc * 1.5, Zn.max() * 0.35, r"$\omega_c=1/RCL^2$", fontsize=7.0,
              color="0.3")
    ax_z.set(ylabel=r"$|Z(\omega)|$  [$\Omega$]")
    ax_z.set_ylim(Zn.min() * 0.4, Zn.max() * 2.5)
    ax_z.tick_params(labelbottom=False)
    ax_z.legend(loc="lower left", fontsize=7)
    panel_label(ax_z, "(a)", x=-0.125)

    # (b) local exponent -----------------------------------------------------
    exn, wxn = _np(ex), _np(w_ex)
    ax_e.axhline(0.5, color="C3", ls="--", lw=1.0)
    ax_e.axhline(1.0, color="C2", ls=":", lw=1.0)
    ax_e.axvline(wc, color="0.4", ls=":", lw=1.0)
    ax_e.semilogx(wxn, exn, "C0", lw=1.4, zorder=3)
    # the transition undershoots 1/2 -- a narrow window parked here reports the
    # wrong order, so say so rather than smoothing it away
    j = int(np.argmin(exn))
    ax_e.plot(wxn[j], exn[j], "v", color="0.25", ms=4.5, zorder=4)
    # placed to the RIGHT of the dip: to its left the label collided with the
    # descending branch and crossed the omega_c guide
    ax_e.annotate(f"undershoot to {exn[j]:.2f}\n"
                  f"at $\\omega\\approx{wxn[j]/wc:.0f}\\,\\omega_c$",
                  xy=(wxn[j], exn[j]), xytext=(wxn[j] * 60, 0.313),
                  fontsize=7.0, color="0.25", ha="left", va="bottom",
                  arrowprops=dict(arrowstyle="->", color="0.45", lw=0.8))
    ax_e.text(0.985, 0.78, "recovered high-frequency order  "
              rf"$\alpha = {float(alpha_hf):.3f}$" "\n"
              f"(fit RMS {_sci_tex(float(rms_hf))})",
              transform=ax_e.transAxes, fontsize=7.0, color="C3", ha="right",
              va="top")
    ax_e.set(xlabel=r"angular frequency $\omega$  [rad s$^{-1}$]",
             ylabel=r"$-\,\mathrm{d}\log|Z|/\mathrm{d}\log\omega$",
             ylim=(0.30, 1.15))
    ax_e.set_yticks([0.5, 0.75, 1.0])
    panel_label(ax_e, "(b)", x=-0.125)

    # (c) geometries collapse until the window reaches the crossover ---------
    # The first two pores deviate IDENTICALLY: their curves coincide to machine
    # precision over the whole panel.  A halo alone leaves the legend claiming
    # three visible curves where there are two, so each also carries its own
    # marker and the coincidence is stated.
    style = [("C0", "o", "-", 3.6, 1.0, 0.45),
             ("C1", "x", "--", 1.3, 4.0, 1.0),
             ("C3", "^", "-.", 1.3, 3.4, 1.0)]
    for (lbl, curve), (c, mk, ls, lw, ms, al) in zip(dev_curve, style):
        ax_g.semilogy(_np(dec_grid), _np(curve), marker=mk, ls=ls, ms=ms, lw=lw,
                      color=c, alpha=al, label=lbl)
    ax_g.axhline(floor, color="0.35", ls="--", lw=1.1)
    ax_g.axvline(0.0, color="0.4", ls=":", lw=1.0)
    ax_g.axvspan(dec_grid[0], 0.0, color="0.9", zorder=0)
    # "decades below w_c" over an axis running -3 to +1 is a double negative;
    # both ends are now said in words as well
    ax_g.set(xlabel="position of the window's lower edge "
                    r"(decades below $\omega_c$; negative $=$ still above it)",
             ylabel="pore-to-pore deviation")
    ax_g.set_xlim(dec_grid[0], dec_grid[-1])
    lo = min(c.min() for _, c in dev_curve)
    ax_g.set_ylim(max(lo * 0.3, 1e-17), 8.0)
    ax_g.text(dec_grid[-1] - 0.05, floor * 2.2, f"{floor:.0%} noise floor",
              fontsize=7.0, color="0.35", ha="right")
    ax_g.text(dec_grid[0] * 0.5, 4.5, r"window entirely above $\omega_c$",
              fontsize=7.0, color="0.35", ha="center", va="top")
    ax_g.legend(loc="lower right", fontsize=7.0,
                title=f"{len(geoms)} pores, equal $R/C$\n"
                      "(the first two coincide exactly)", title_fontsize=7.0)
    panel_label(ax_g, "(c)", x=-0.085)

    out = save_paper_figure(fig, out_path)
    plt.close(fig)
    return out


def plot_real_window(rows, d_required, out_path):
    """The identifiability boundary measured on real cells.
    (a) the observation window each published cell actually spans, against the
    requirement derived synthetically; (b) the effective order recovered from
    those same cells."""
    ds = [r[0] for r in rows]
    names = [r[1] for r in rows]
    dec = np.array([r[2] for r in rows])
    al = np.array([r[3] for r in rows])
    r2 = np.array([r[4] for r in rows])
    DCOL = {"Oxford": "C0", "Zhang": "C1"}
    col = [DCOL.get(d, "0.5") for d in ds]

    use_paper_style()
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 3.5))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.36,
                             left=0.10, right=0.985, top=0.90, bottom=0.135)
    ax_d = fig.add_subplot(outer[0])
    # (b) is split: eight of nine good fits crowd into R^2 in [0.996, 0.999]
    # while one outlier at 0.955 sets the range, so on one linear axis the
    # spread the panel exists to show is a single unreadable line at the top
    inner = outer[1].subgridspec(2, 1, height_ratios=[2.5, 1.0], hspace=0.11)
    ax_hi = fig.add_subplot(inner[0])
    ax_lo = fig.add_subplot(inner[1], sharex=ax_hi)

    # (a) how far each real experiment actually reaches ---------------------
    order = np.argsort(dec)
    y = np.arange(len(rows))
    ax_d.axvspan(d_required, d_required + 1.45, color="#eaf6ea", zorder=0)
    ax_d.barh(y, dec[order], color=[col[i] for i in order], alpha=0.9,
              height=0.72, zorder=2)
    ax_d.axvline(d_required, color="C3", lw=1.4, ls="--", zorder=3)
    ax_d.text(d_required + 0.12, len(rows) - 0.4,
              f"{d_required} decades\nrequired", color="C3", fontsize=7.0,
              va="top")
    ax_d.set_yticks(y)
    ax_d.set_yticklabels([names[i] for i in order], fontsize=7.0)
    ax_d.set_xlim(0, d_required + 1.45)
    ax_d.set_ylim(-1.0, len(rows) - 0.3)
    ax_d.set_xlabel("observation window actually spanned (decades)")
    # inside the axes: the note used to start left of the y spine, in the
    # tick-label gutter
    # inside the green band: right-aligned across the panel it ran through the
    # requirement line, and left of the y spine it sat in the tick-label gutter
    ax_d.text(d_required + 0.12, len(rows) * 0.45,
              f"none of {len(rows)}\ncells reaches\nthe threshold;\n"
              f"best is {dec.max():.2f}",
              fontsize=7.0, color="0.3", va="center")
    import matplotlib.patches as mpatches
    ax_d.legend(handles=[mpatches.Patch(color=DCOL["Oxford"], alpha=0.9,
                                        label="Oxford"),
                         mpatches.Patch(color=DCOL["Zhang"], alpha=0.9,
                                        label="Zhang et al.")],
                loc="lower right", bbox_to_anchor=(1.0, 0.06), fontsize=7.0)
    panel_label(ax_d, "(a)", x=-0.135)

    # (b) what those same cells do give up: the effective order -------------
    usable = np.array([bool(r[5]) for r in rows])
    n_use = int(usable.sum())
    for a in (ax_hi, ax_lo):
        a.axvline(0.5, color="0.35", ls=":", lw=1.0)
        a.axvline(1.0, color="0.35", ls=":", lw=1.0)
        for lbl, c in DCOL.items():
            m = np.array([d == lbl for d in ds]) & usable
            a.scatter(al[m], r2[m], s=26, facecolor="none", edgecolor=c,
                      linewidth=1.2, label=lbl if a is ax_hi else None,
                      zorder=3)
        a.grid(True, axis="x")

    hi_vals = r2[usable & (r2 > 0.99)]
    lo_vals = r2[usable & (r2 <= 0.99)]
    pad_hi = max((hi_vals.max() - hi_vals.min()) * 0.35, 3e-4)
    ax_hi.set_ylim(hi_vals.min() - pad_hi, hi_vals.max() + pad_hi)
    if lo_vals.size:
        ax_lo.set_ylim(min(0.949, lo_vals.min() - 0.004), lo_vals.max() + 0.004)
    else:
        ax_lo.set_ylim(0.945, 0.96)
    ax_lo.axhline(0.95, color="C3", ls="--", lw=1.0)
    ax_lo.text(0.988, 0.95, r"$R^2=0.95$ cut ",
               transform=ax_lo.get_yaxis_transform(), fontsize=7.0, color="C3",
               va="bottom", ha="right")

    # break marks
    ax_hi.spines["bottom"].set_visible(False)
    ax_lo.spines["top"].set_visible(False)
    ax_hi.tick_params(labelbottom=False, bottom=False)
    kw = dict(marker=[(-1, -0.6), (1, 0.6)], ms=5, ls="none", color="k",
              mec="k", mew=0.8, clip_on=False)
    ax_hi.plot([0, 1], [0, 0], transform=ax_hi.transAxes, **kw)
    ax_lo.plot([0, 1], [1, 1], transform=ax_lo.transAxes, **kw)

    # the two vertical guides are the mechanistic limits; label them INSIDE the
    # axes, where they cannot overrun the top spine
    ax_hi.text(0.5, 0.97, " diffusion-limited", transform=ax_hi.get_xaxis_transform(),
               fontsize=7.0, color="0.35", rotation=90, va="top", ha="left")
    ax_hi.text(1.0, 0.97, " reaction-limited", transform=ax_hi.get_xaxis_transform(),
               fontsize=7.0, color="0.35", rotation=90, va="top", ha="left")
    ax_hi.text(0.985, 0.115, f"{n_use} usable fits of {len(rows)}",
               transform=ax_hi.transAxes, fontsize=7.0, color="0.35",
               ha="right")
    ax_hi.legend(loc="upper right", fontsize=7.0)

    ax_lo.set_xlabel(r"effective order $\alpha$ of the capacity loss")
    ax_hi.set_ylabel(r"single-power-law $R^2$")
    ax_hi.yaxis.set_label_coords(-0.235, 0.30)
    ax_hi.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax_lo.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax_lo.set_yticks([0.95, 0.955])
    panel_label(ax_hi, "(b)", x=-0.265)

    out = save_paper_figure(fig, out_path)
    plt.close(fig)
    return out


def plot_order_spectra(results, out_path):
    """Recovered distributed-order spectra w(alpha) for each dataset.
    Power-law => single spike; non-power-law => spread / multi-modal.

    All panels share one weight axis.  With a per-panel y-scale the caption's
    comparison -- a single spike here against a collapse there -- could not be
    made across panels, because a 0.40 spike and a 0.75 spike were drawn the
    same height.  The last cell of the grid carries the key rather than being
    left blank."""
    import matplotlib.patches as mpatches
    use_paper_style()
    n = len(results)
    ncol = 2 if n <= 4 else 3
    nrow = -(-n // ncol)                 # no spare cells: the key goes below
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(TEXTWIDTH_IN, 1.85 * nrow + 0.72),
                             squeeze=False)
    letters = "abcdefgh"
    wmax = max(float(_np(r["w"]).max()) for r in results)
    for i, res in enumerate(results):
        ax = axes[i // ncol][i % ncol]
        og = _np(res["order_grid"])
        ax.bar(og, _np(res["w"]), width=(og[1] - og[0]) * 0.9,
               color="C0" if res["family"] == "power-law" else "C3", alpha=0.85)
        for gt in res.get("ground_truth", []):
            ax.axvline(gt, color="k", ls="--", lw=1.0)
        r2_pl = 1.0 - res["res_pl"]
        r2_sp = 1.0 - res["res_dist"]
        ax.set_title(f"{display_name(res['name'])}\n"
                     f"$R^2$: power law {r2_pl:.3f}, spectrum {r2_sp:.3f}",
                     fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, wmax * 1.08)
        ax.set_xlabel(r"order $\alpha$")
        if i % ncol == 0:
            ax.set_ylabel(r"weight $w(\alpha)$")
        else:
            ax.tick_params(labelleft=False)
        panel_label(ax, f"({letters[i]})", x=-0.18, y=1.22)

    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    fig.tight_layout(w_pad=2.0, h_pad=1.6, rect=[0, 0.115, 1, 1])
    # the key sits under the grid: the dashed lines were never defined in the
    # caption, and one panel has none at all
    handles = [mpatches.Patch(color="C0", alpha=0.85,
                              label="recovered $w(\\alpha)$, power-law data"),
               mpatches.Patch(color="C3", alpha=0.85,
                              label="recovered $w(\\alpha)$, non-power-law data"),
               plt.Line2D([], [], color="k", ls="--", lw=1.0,
                          label="true order of the generator")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.0,
               bbox_to_anchor=(0.5, 0.048))
    fig.text(0.5, 0.012, "All panels share one weight axis. The PDM generator "
             "contains no fractional order, so its panel carries no true-order "
             "line.", ha="center", va="bottom", fontsize=7.0, color="0.3")
    out = save_paper_figure(fig, out_path)
    plt.close(fig)
    return out


def plot_benchmark_gallery(datasets, out_path):
    """Gallery: every benchmark curve with its best single-power-law fit."""
    ncol = 4
    nrow = 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 7))
    fig.suptitle("gfckit benchmark suite -- power-law (top) vs non-power-law (bottom)",
                 fontsize=13, fontweight="bold")
    pl = [d for d in datasets if d["family"] == "power-law"]
    npl = [d for d in datasets if d["family"] == "non-power-law"]

    def draw(ax, d):
        t = _np(d["t"])
        ax.plot(t, _np(d["y_clean"]), "C0", lw=1.6, label="clean")
        ax.plot(t, _np(d["y_meas"]), ".", ms=4, color="0.35", label="measured")
        a = d["pl_alpha"]
        Phi = np.stack([_np(d["t"]) ** a, np.ones_like(t)], 1)
        coef, *_ = np.linalg.lstsq(Phi, _np(d["y_clean"]), rcond=None)
        ax.plot(t, Phi @ coef, "r--", lw=1.3, label=fr"$t^{{{a:.2f}}}$ fit")
        good = d["pl_R2"] > 0.999 and d["slope_var"] < 0.05
        tag = "power-law OK" if good else "NOT power-law"
        ax.set_title(f"{d['name']}\n$R^2$={d['pl_R2']:.4f}, slope-var={d['slope_var']:.2f}  [{tag}]",
                     fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, frameon=False)

    for j in range(ncol):
        axes[0, j].axis("off")
        axes[1, j].axis("off")
    for j, d in enumerate(pl):
        axes[0, j].axis("on"); draw(axes[0, j], d)
    for j, d in enumerate(npl):
        axes[1, j].axis("on"); draw(axes[1, j], d)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_benchmark_diagnostic(datasets, out_path):
    """Local log-log slope for each curve: flat => power law, varying => not.

    Every curve is named and styled individually.  The previous version drew all
    four non-power-law curves in one red and listed all four in the legend with
    identical swatches, so no name mapped to any curve, and it left the two
    power-law curves out of the legend entirely."""
    from .complex import local_loglog_slope
    use_paper_style()
    fig, ax = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.6),
                           gridspec_kw=dict(width_ratios=[1.0, 0.92]))

    # distinct hue AND dash pattern for every series: the two families are
    # distinguished by warm/cool, the members within a family by dash
    pl_style = [("C0", "-"), ("C0", "--"), ("C0", "-.")]
    npl_style = [("C3", "-"), ("C1", "--"), ("#7a1fa2", "-."), ("#8c564b", ":")]

    ymax = 0.0
    i_pl = i_npl = 0
    for d in datasets:
        ts, sl = local_loglog_slope(d["t"], d["y_clean"])   # clips internally
        if d["family"] == "power-law":
            c, ls = pl_style[i_pl % len(pl_style)]
            i_pl += 1
        else:
            c, ls = npl_style[i_npl % len(npl_style)]
            i_npl += 1
        sl_n = _np(sl)
        ymax = max(ymax, float(sl_n.max()))
        ax[0].plot(_np(ts) / _np(ts)[-1], sl_n, color=c, ls=ls, lw=1.2,
                   label=display_name(d["name"]))
    ax[0].set(xlabel="normalized time $t/t_{\\max}$",
              ylabel=r"local exponent $\mathrm{d}\ln y/\mathrm{d}\ln t$")
    # no clipping: the two-order curve peaked at 1.6 and left the axes
    # headroom for the legend: at 1.08x the peak it sat on the curves
    ax[0].set_ylim(0, max(1.7, ymax * 1.42))
    ax[0].legend(loc="upper left", ncol=2, fontsize=7.0)
    panel_label(ax[0], "(a)")

    names = [display_name(d["name"]) for d in datasets]
    svar = [float(d["slope_var"]) for d in datasets]
    cols = ["C0" if d["family"] == "power-law" else "C3" for d in datasets]
    y = np.arange(len(names))
    ax[1].barh(y, svar, color=cols, height=0.68)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels(names)
    ax[1].axvline(0.05, ls="--", color="k", lw=0.9)
    span = max(svar)
    ax[1].set_xlim(0, span * 1.30)
    ax[1].set_ylim(len(names) - 0.45, -0.85)
    # above the top bar, where it clears both the printed values and the key
    ax[1].text(0.05 + 0.016 * span, -0.62, "threshold 0.05",
               fontsize=7.0, color="0.3", va="center")
    # a zero-length bar reads as missing data; print every value instead, set
    # clear of the threshold line so the small ones stay readable
    for yi, v in zip(y, svar):
        ax[1].text(max(v, 0.05) + 0.022 * span, yi, f"{v:.3f}", va="center",
                   fontsize=7.0, color="0.25")
    ax[1].set(xlabel="standard deviation of the local exponent\n"
                     r"(0 $\rightarrow$ power law)")
    import matplotlib.patches as mpatches
    ax[1].legend(handles=[mpatches.Patch(color="C0", label="power-law family"),
                          mpatches.Patch(color="C3", label="non-power-law family")],
                 loc="upper right", bbox_to_anchor=(1.005, 1.005), fontsize=7.0)
    panel_label(ax[1], "(b)", x=-0.42)

    fig.tight_layout(w_pad=2.4)
    out = save_paper_figure(fig, out_path)
    plt.close(fig)
    return out


def plot_order_identification(noise_levels, alpha_strong, alpha_weak,
                              alpha_true, out_path):
    """Strong vs weak order recovery vs noise (single-trajectory demo).

    The weak-form estimate is exactly alpha_true at every noise level, so it
    lies on top of the reference line.  Drawn as a plain solid line it hides the
    reference completely and the legend then points at an invisible element;
    drawn as open markers on a dashed line, both are readable at once."""
    use_paper_style()
    fig, ax = plt.subplots(figsize=(0.62 * TEXTWIDTH_IN, 2.65))
    nz = _np(noise_levels) * 100
    ax.axhline(alpha_true, ls="-", color="0.75", lw=2.6, zorder=1,
               label=fr"true $\alpha={alpha_true:g}$")
    ax.plot(nz, alpha_strong, "s--", color="C3", ms=3.6, zorder=3,
            label="strong (derivative) form")
    ax.plot(nz, alpha_weak, "o-", color="C0", ms=4.2, mfc="white", mew=1.1,
            zorder=4, label="weak (integral) form")
    ax.set(xlabel="measurement noise (%)", ylabel=r"recovered $\alpha$")
    # the sampled noise levels are unevenly spaced; mark each one so the reader
    # is not invited to read the straight joining segments as interpolation
    ax.set_xticks(list(nz))
    ax.set_ylim(0, max(0.62, float(np.max(alpha_weak)) * 1.15))
    ax.legend(loc="lower left")
    ax.grid(True)
    fig.tight_layout()
    out = save_paper_figure(fig, out_path)
    plt.close(fig)
    return out

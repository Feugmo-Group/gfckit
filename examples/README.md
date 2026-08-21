# gfckit examples: the equations being solved

Each example writes a figure into `../figures/`. Run from the package root, e.g.

```bash
python examples/physical_data.py
```

All numerics are pure JAX; matplotlib is used only for the figures.

---

## 1. `order_identification.py`: anomalous order from a single curve

**Model.** A degradation observable obeying a Caputo fractional relaxation with
a constant drive. The analytic test signal is the exact solution

$$
u(t) = t^{\alpha}, \qquad {}^{C}\!D^{\alpha}_t\, u = \Gamma(\alpha+1) \;=\; \text{const}.
$$

The Caputo derivative is the memory convolution
$\big({}^{C}\!D^{\alpha}_t u\big)(t)=\frac{1}{\Gamma(1-\alpha)}\int_0^t (t-\tau)^{-\alpha}u'(\tau)\,\mathrm d\tau$.

**What we solve.** Recover $\alpha$ from noisy $u$ two ways:
- **strong form** — the order making ${}^{C}\!D^{\alpha}u$ most constant (differentiates the data → noise-sensitive);
- **weak/integral form** — least-squares fit of $u\approx A\,t^{\alpha}+B$ (never differentiates the data → noise-robust).

**Result.** Weak form holds $\alpha=0.500$ up to 5 % noise; strong form collapses.
$\alpha=\tfrac12$ ⇒ classical Fickian (√t) growth; $\alpha\neq\tfrac12$ ⇒ anomalous/memory aging.

---

## 2. `multicondition.py`: shared kernel across Arrhenius clocks

**Model.** A **general fractional** relaxation in an internal (warped) time
$s=g_c(t)$, with the same **tempered-power-law** memory kernel for every condition:

$$
\big(\mathbb D^{(K),*}_{g_c} u_c\big)(t) = c_{\text{sink}},
\qquad
K(\tau)=\frac{\tau^{-\alpha}\,e^{-\lambda \tau}}{\Gamma(1-\alpha)} .
$$

Here $g_c(t)=k_c\,t$ is the Arrhenius clock of condition $c$ (temperature enters
through $k_c\propto e^{-E_a/k_BT_c}$), and Tarasov's substitution operator makes
each condition *the same operator in warped time*:
$\mathbb D^{(K),*}_{g}=Q_g\,\mathbb D^{(K),*}\,Q_g^{-1}$, with $(Q_g f)(t)=f(g(t))$.

- $\alpha$ — anomalous exponent ($\tfrac12$ = Fickian).
- $\lambda$ — inverse memory horizon (finite‑memory / self‑limiting aging).

**What we solve.** Recover the shared $(\alpha,\lambda)$ from several conditions at
once. **Strong form** matches the operator residual (clean data); **weak form**
matches a forward simulation to data (noise-robust).

**Result.** At a single condition, $\alpha$ and $\lambda$ are coupled along a
ridge in the objective and neither is individually identified: the best fit
sits at $(\hat\alpha,\hat\lambda)=(0.75,0.08)$ against true $(0.5,0.5)$, an error of
$(0.25,0.43)$. Fitting four conditions jointly breaks the ridge and the error
falls to $(0.04,0.07)$; the weak form stays accurate under noise throughout.

---

## 3. `physical_data.py`: mechanistic data via Newton-Krylov

Data are generated from **independent physics** (not from the fractional kernel),
then handed to gfckit: a genuine cross-model test. Both models are stepped
implicitly (backward Euler) and each nonlinear step is solved by **Jacobian-free
Newton-Krylov** (GMRES with autodiff Jacobian-vector products).

### 3a. SEI growth / capacity fade: moving-boundary reaction-diffusion

Solvent diffuses through the growing SEI film $[0,L(t)]$ and reacts at the
electrode. With Fick's second law inside the film and the Landau change of
variable $\xi=x/L(t)$ mapping to a fixed grid:

$$
\frac{\partial u}{\partial t}
= \frac{D}{L^2}\frac{\partial^2 u}{\partial \xi^2}
+ \xi\,\frac{\dot L}{L}\,\frac{\partial u}{\partial \xi},
\qquad
\underbrace{\frac{D}{L}\frac{\partial u}{\partial \xi}\Big|_{0}=k\,u(0)}_{\text{electrode reaction}},
\quad
u(1,t)=c_b,
\quad
\dot L = \Omega\,k\,u(0).
$$

$L(t)$ (∝ lost capacity) is **reaction-limited early** ($L\sim t$) and
**diffusion-limited late** ($L\sim\sqrt t$). Identified effective exponent
$\alpha\approx0.59$ (between $1$ and $\tfrac12$), trending to the Fickian $\tfrac12$.

### 3b. Corrosion film growth: Point Defect Model (high-field)

Oxide film thickness under a high interfacial field (Cabrera–Mott / PDM), with
field-assisted growth competing with chemical dissolution:

$$
\frac{\mathrm dL}{\mathrm dt}
= k_g\,\exp\!\Big(\frac{b\,\Delta V}{L}\Big) - k_d .
$$

Growth ($\propto$ field $\sim 1/L$) weakens as $L$ grows; with $k_g<k_d$ the film
**passivates** to $L_{ss}=b\,\Delta V/\ln(k_d/k_g)$. Identified $\alpha\approx0.18$
and a poor single-power-law fit: the **saturation is the finite memory horizon**
$\lambda$, i.e. this is where the *tempered* kernel of example 2 is needed.

---

## 4. `fick_diffusion.py`: Fick's law in different systems

Baseline **normal** diffusion (mean-squared displacement $\langle x^2\rangle=2Dt$,
$\alpha=1$). Fick's laws: flux $J=-D\,\partial_x c$; transport
$\partial_t c=D\,\partial_x^2 c$. Canonical closed-form solutions:

| System | Solution |
|---|---|
| Semi-infinite, fixed surface $c_s$ (diffusion couple, carburizing) | $c(x,t)=c_s\,\mathrm{erfc}\!\big(x/2\sqrt{Dt}\big)$ |
| Surface flux (Fick 1st law) | $J(0,t)=c_s\sqrt{D/\pi t}\ \sim t^{-1/2}$ |
| Instantaneous planar source $M$ (thin-film couple) | $c(x,t)=\dfrac{M}{\sqrt{4\pi Dt}}\,e^{-x^2/4Dt}$ |
| Finite slab $[0,L]$, faces at $c_s$ | $c=c_s\big[1-\tfrac{4}{\pi}\sum_{m\,\text{odd}}\tfrac{1}{m}\sin\tfrac{m\pi x}{L}\,e^{-D(m\pi/L)^2 t}\big]$ |

**Key link.** Diffusion-limited **cumulative uptake** is
$\int_0^t J(0,\tau)\,\mathrm d\tau = 2c_s\sqrt{Dt/\pi}\ \sim t^{1/2}$, the
**parabolic growth law**. gfckit identifies its exponent as $\alpha=0.500$:
*Fickian diffusion is exactly the $\alpha=\tfrac12$ special case that anomalous
(memory) degradation deviates from.*

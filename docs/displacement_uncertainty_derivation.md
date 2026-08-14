# Phase-unwrap uncertainty: derivation

This walks through the statistics behind propagating per-sample phase noise
through phase unwrapping, without reference to any particular
implementation. Throughout, a pair of adjacent indices is always written as
$(i, i-1)$ — never $(i, i+1)$ — to match the direction the recursion below
actually runs in (built up from the past).

## 0. Setup

At each sample $i$ we observe a noisy phase $\theta_i \in (-\pi,\pi]$, modeled
as independent across $i$ with known variance $\mathrm{Var}(\theta_i)$. Define
the successive difference

$$
\Delta\theta_i = \theta_i - \theta_{i-1}
$$

Because $\theta_i$ and $\theta_{i-1}$ are independent,

$$
\mathrm{Var}(\Delta\theta_i) = \mathrm{Var}(\theta_i) + \mathrm{Var}(\theta_{i-1})
$$

Two consecutive differences share one $\theta$ term, so they're correlated:

$$
\mathrm{Cov}(\Delta\theta_i, \Delta\theta_{i-1})
= \mathrm{Cov}\big(\theta_i-\theta_{i-1},\ \theta_{i-1}-\theta_{i-2}\big)
= -\mathrm{Var}(\theta_{i-1})
$$

(every other cross term is zero by independence). Two apart, there is no
shared term:

$$
\mathrm{Cov}(\Delta\theta_i, \Delta\theta_{i-2})
= \mathrm{Cov}\big(\theta_i-\theta_{i-1},\ \theta_{i-2}-\theta_{i-3}\big) = 0
$$

so $\Delta\theta_i \perp \Delta\theta_{i-2}$ (uncorrelated jointly-Gaussian
$\Rightarrow$ independent). **This one fact — correlation only between
immediate neighbors — is the load-bearing assumption for everything below.**

## 1. The unwrap correction

Phase unwrapping corrects a $2\pi$ ambiguity whenever a raw phase difference
exceeds $\pm\pi$. Define the correction at step $i$:

$$
f_i =
\begin{cases}
+1 & \Delta\theta_i < -\pi \\
-1 & \Delta\theta_i > \ \ \pi \\
0 & \text{otherwise}
\end{cases}
$$

and the cumulative correction count $S_i = f_0 + f_1 + \cdots + f_i$. The
unwrapped phase is $\theta_i + 2\pi S_i$, so its variance is (dropping the
cross term between $\theta_i$ and $S_i$, a separate simplifying assumption)

$$
\mathrm{Var}(\text{unwrapped phase}_i) = \mathrm{Var}(\theta_i) + 4\pi^2\,\mathrm{Var}(S_i)
$$

The rest of this document is entirely about computing $\mathrm{Var}(S_i)$.

## 2. Marginal variance of one correction

$f_i$ is a three-way threshold of the single Gaussian
$\Delta\theta_i \sim \mathcal N\big(\mu_i,\ \mathrm{Var}(\Delta\theta_i)\big)$.
Write

$$
p_i^+ = P(\Delta\theta_i > \pi), \qquad p_i^- = P(\Delta\theta_i < -\pi)
$$

Then $f_i$ takes value $-1$ w.p. $p_i^+$, $+1$ w.p. $p_i^-$, and $0$
otherwise, giving

$$
E[f_i] = p_i^- - p_i^+, \qquad E[f_i^2] = p_i^- + p_i^+
$$

$$
D_i \;:=\; \mathrm{Var}(f_i) \;=\; p_i^+(1-p_i^+) + p_i^-(1-p_i^-) + 2p_i^+p_i^-
$$

## 3. Joint distribution of a neighboring pair

$f_i$ and $f_{i-1}$ are each a threshold of $\Delta\theta_i$ and
$\Delta\theta_{i-1}$ respectively, and — from §0 — these two are *jointly
Gaussian and correlated*. Lay the $(\Delta\theta_i,\Delta\theta_{i-1})$-plane
out on two axes and cut it with the four lines $\Delta\theta_i=\pm\pi$,
$\Delta\theta_{i-1}=\pm\pi$. This produces a $3\times 3$ grid of regions, one
per possible pair of values $(f_i, f_{i-1}) \in \{-1,0,1\}^2$. Each region's
probability is an inclusion–exclusion of the bivariate normal CDF evaluated
at the four corners $(\pm\pi,\pm\pi)$ together with the two marginal CDFs at
$\pm\pi$ — standard orthant-probability bookkeeping for a truncated bivariate
normal. Label the four joint-CDF corners $F_{--}, F_{-+}, F_{+-}, F_{++}$
(sign order = $(\Delta\theta_i,\ \Delta\theta_{i-1})$) and the marginal CDFs
$F_i^{\pm}, F_{i-1}^{\pm}$. The nine region probabilities work out to:

$$
\begin{array}{lll}
R_1 = F_{--} & \;\to\; (f_i,f_{i-1})=(+1,+1) \\
R_2 = F_{+-}-F_{--} & \;\to\; (0,+1) \\
R_3 = F_{i-1}^{-}-F_{+-} & \;\to\; (-1,+1) \\
R_4 = F_{-+}-F_{--} & \;\to\; (+1,0) \\
R_5 = F_{++}-F_{+-}-F_{-+}+F_{--} & \;\to\; (0,0) \\
R_6 = F_{i-1}^{+}-F_{i-1}^{-}-F_{++}+F_{+-} & \;\to\; (-1,0) \\
R_7 = F_i^{-}-F_{-+} & \;\to\; (+1,-1) \\
R_8 = F_i^{+}-F_i^{-}-F_{++}+F_{-+} & \;\to\; (0,-1) \\
R_9 = 1-F_i^{+}-F_{i-1}^{+}+F_{++} & \;\to\; (-1,-1)
\end{array}
$$

These nine numbers are a full probability mass function for the pair
$(f_i, f_{i-1})$ — everything below is just moments of it.

## 4. Variance of a neighboring sum

Group the nine regions by the value of $f_i+f_{i-1} \in \{-2,-1,0,1,2\}$:

$$
P(\text{sum}=2)=R_1,\quad P(\text{sum}=1)=R_2+R_4,\quad P(\text{sum}=0)=R_3+R_5+R_7,
$$
$$
P(\text{sum}=-1)=R_6+R_8,\quad P(\text{sum}=-2)=R_9
$$

giving

$$
V_i \;:=\; \mathrm{Var}(f_i+f_{i-1}) \;=\; E[\text{sum}^2] - E[\text{sum}]^2
$$

directly from that distribution.

## 5. Covariance of a neighboring pair

The same nine regions give the cross moment for free: of the nine
$(f_i,f_{i-1})$ outcomes, only the four **corners** have a nonzero product —
$(+1,+1)\!\to\!+1$ ($R_1$), $(-1,+1)\!\to\!-1$ ($R_3$), $(+1,-1)\!\to\!-1$
($R_7$), $(-1,-1)\!\to\!+1$ ($R_9$):

$$
E[f_i f_{i-1}] = R_1 - R_3 - R_7 + R_9
$$

and, from §2, $E[f_i] = p_i^- - p_i^+ = F_i^- - (1-F_i^+) = F_i^- + F_i^+ - 1$
(same for $E[f_{i-1}]$, using the $i-1$ marginals):

$$
C_i \;:=\; \mathrm{Cov}(f_i, f_{i-1}) \;=\; E[f_i f_{i-1}] - E[f_i]\,E[f_{i-1}]
$$

Note the identity linking §2, §4, §5 — it's just the definition of variance
of a sum:

$$
V_i = D_i + D_{i-1} + 2C_i
$$

## 6. Independence beyond nearest neighbors

From §0, $\Delta\theta_i \perp \Delta\theta_{i-2}$, and $f_i$, $f_{i-2}$ are
deterministic functions of one input apiece, so independence of the inputs
carries straight through:

$$
\mathrm{Cov}(f_i, f_j) = 0 \qquad \text{for all } |i-j|\ge 2
$$

## 7. The covariance matrix is tridiagonal

Collect $f_0, f_1, \dots, f_i$ into a vector. By §6, every entry more than
one step off the diagonal is exactly zero; the diagonal is $D_j$ (§2) and
the one-off-diagonal entries are $C_j$ (§5). Writing it out from index $i$
down to $0$:

$$
\Sigma =
\begin{pmatrix}
D_i    & C_i    & 0      & 0      & \cdots & 0 \\
C_i    & D_{i-1}& C_{i-1}& 0      & \cdots & 0 \\
0      & C_{i-1}& D_{i-2}& C_{i-2}& \cdots & 0 \\
0      & 0      & C_{i-2}& D_{i-3}& \ddots & \vdots \\
\vdots & \vdots & \vdots & \ddots & \ddots & C_1 \\
0      & 0      & 0      & \cdots & C_1    & D_0
\end{pmatrix}
$$

This is the exact joint second-moment structure of the correction sequence —
no approximation beyond the independence established in §0/§6, which is
itself exact for a Gaussian $\theta$ model.

## 8. Variance of the running sum — proof and pattern

$S_i = \mathbf 1^\top \mathbf f$ where $\mathbf 1$ is the all-ones vector and
$\mathbf f = (f_0,\dots,f_i)^\top$, so

$$
\mathrm{Var}(S_i) = \mathbf 1^\top \Sigma\, \mathbf 1
= \sum_{j=0}^{i} D_j \;+\; 2\sum_{j=1}^{i} C_j
$$

— every diagonal entry once, every off-diagonal entry twice (symmetric
matrix). Substituting $C_j = \tfrac12(V_j - D_j - D_{j-1})$ from §5 turns
this into a telescoping sum: each $D_j$ for $1 \le j \le i-1$ gets counted
once by $\sum D_j$ and then cancelled once by the substitution, leaving only
the two endpoints and the $V_j$'s:

$$
\mathrm{Var}(S_i) = \sum_{j=1}^{i} V_j \;-\; \sum_{j=1}^{i-1} D_j
$$

Writing this out low-to-high makes the pattern the matrix implies completely
explicit — each additional term in the running sum contributes one more $V$
and, except at the very first step, cancels one more single-index variance:

$$
\begin{aligned}
\mathrm{Var}(f_i) &= D_i \\[2pt]
\mathrm{Var}(f_i+f_{i-1}) &= V_i \\[2pt]
\mathrm{Var}(f_i+f_{i-1}+f_{i-2}) &= V_i + V_{i-1} \;-\; D_{i-1} \\[2pt]
\mathrm{Var}(f_i+f_{i-1}+f_{i-2}+f_{i-3}) &= V_i + V_{i-1} + V_{i-2} \;-\; D_{i-1} - D_{i-2} \\[2pt]
\mathrm{Var}(f_i+\cdots+f_{i-4}) &= V_i + V_{i-1} + V_{i-2} + V_{i-3} \;-\; D_{i-1}-D_{i-2}-D_{i-3} \\[2pt]
&\ \ \vdots
\end{aligned}
$$

Each row keeps every term from the row above and appends exactly one new
$V_{i-n}$ and one new $-D_{i-n}$ as the window grows one index further back
— the double-counted interior variance is peeled off one at a time, while
the two endpoints ($f_i$ and the oldest term in the sum) are never
subtracted, since each appears in only one adjacent pair.

## 9. Recursive form

The same result, read as a one-step update instead of a closed sum (useful
if $S_i$ is wanted for every $i$, not just one): comparing consecutive rows
of §8's pattern,

$$
\mathrm{Var}(S_i) = \mathrm{Var}(S_{i-1}) + V_i - D_{i-1}, \qquad \mathrm{Var}(S_0) = D_0
$$

Unrolling this recursion one step at a time regenerates exactly the pattern
in §8 — it's the same statement either way, closed-form sum or recursion.
The only thing either form leans on is §6: correlation dies after one step,
so extending the sum by one more term never needs to reach back further than
the term it's adjacent to.

## 10. Back to physical uncertainty

$\mathrm{Var}(S_i)$ feeds back into §1's
$\mathrm{Var}(\text{unwrapped phase}_i) = \mathrm{Var}(\theta_i) + 4\pi^2\mathrm{Var}(S_i)$,
and from there into any downstream physical quantity (e.g. displacement)
that is a linear function of phase — a linear map $x = a\cdot\phi$ propagates
standard deviation as $\sigma_x = |a|\,\sigma_\phi$. Whatever coefficient $a$
is used going forward from phase to the physical quantity must be the exact
same coefficient used when propagating the uncertainty backward — that's
just the one-dimensional delta method, but it's an easy place to introduce a
silent factor mismatch if the two are computed in different places.

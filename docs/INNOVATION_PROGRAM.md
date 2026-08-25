# Causal LSSO: SolveDelta Canonical Contract

`causallsso/reference.py` owns the executable mathematics and FP64 numerical
truth. This document explains that one **SolveDelta** contract; it does not
define a second implementation. All simpler Delta operators are exact
restrictions of this contract, not maintained variants.

The LSSO inheritance is the solved-operator principle. At every token,
fixed-size prefix statistics instantiate a compact contextual system; applying
its solve action defines the frame used by the causal Delta recurrence. The
bidirectional LSSO chart `I + F F^T + Omega` is one realization of that
principle, not the definition of Causal LSSO.

## 1. Shapes and causal order

Use the resolved key-side width per head,

\[
\boxed{r:=d_k.}
\]

`d_k` is resolved by the containing model configuration, not chosen again by
SolveDelta. A frontend may specify it directly as `head_k_dim`, or derive a total
key width from `hidden_size` and a key expansion and then require exact
divisibility by the number of key heads. In either case,

\[
d_k=\frac{d_{\mathrm{key,total}}}{H_k},
\qquad d_{\mathrm{key,total}}=H_kd_k.
\]

There is no independent public geometry rank. The first native specialization
and principal benchmark profile use `d_k=r=128`, but 128 is an implementation
target, not the mathematical default. The reference contract accepts every
positive resolved `d_k`; a native backend may initially support an explicit
finite specialization set and must reject other widths rather than silently
changing `r`.

Let `num_edits = K` be a positive integer fixed for a layer. The recommended
default is `K = 1`; edit count is a capacity hyperparameter, not the defining
contribution. Token projections produce one geometry pair, `K` edit pairs, and
one query:

\[
u_t,h_t,q_t,k_{t,1},\ldots,k_{t,K}\in\mathbb R^r,
\qquad
v_{t,1},\ldots,v_{t,K}\in\mathbb R^{d_v}.
\]

`u`, all edit keys, and `q` are L2-normalized. `u` is independent of the edit
keys: geometry answers what the prefix space is, while edit keys answer how the
current token modifies content.

The containing layer applies fixed GDN2-style depthwise causal `conv4` plus
SiLU independently to the projected query, packed edit keys, and packed edit
values before head reshape and normalization. This frontend is enabled by
default and has one structural off switch. Geometry features, drives, decay,
erase/write logits, and output gates are not convolved. Recurrent layer
state therefore owns the operator state plus three minimal convolution
caches; convolution does not alter the four-tensor operator recurrence below.
For a batch of size `B`, the minimal conv4 continuation caches are
`C_q:[B,Hr,3]`, `C_k:[B,HKr,3]`, and `C_v:[B,HKd_v,3]`, with
the last axis ordered oldest to newest. They store raw projected inputs in
the projection activation dtype,
are zero-initialized, hold on invalid tokens, and reset immediately before a
valid reset token is shifted in. Their gradient contract includes dependence
through both convolution outputs and returned final caches. When the structural
switch is off, Q/K/V receive SiLU without convolution and all three cache fields
are `None`.

The recurrent state is

\[
m_t\in\mathbb R,
\qquad J_t,D_t\in\mathbb R^{r\times r},
\qquad S_t\in\mathbb R^{r\times d_v},
\]

initialized to zero. The equations are real-valued operator semantics and the
executable oracle is FP64. In the first native training contract,
projection/conv caches and raw vector operands are BF16, analytically bounded
private panels are written directly from FP32 producers to FP16, and all
contractions accumulate in FP32. The `m,J,D,S` continuation states remain FP32
and are never rounded at chunk or recurrent-call boundaries. The private FP16
representation does not change the BF16-observable operator contract.

Every valid token executes exactly this order:

1. update geometry once;
2. construct one current transpose-dual solve adapter;
3. decay associative memory once;
4. apply edits `1, ..., K` in order;
5. read the `K`-times-edited memory.

Invalid tokens leave every operator and convolution state unchanged. Sequence
boundaries reset all states. `K` is public configuration, static across tokens, and shared by the
reference and optimized execution contracts. Changing `K` changes capacity
within this one operator family; it does not select a parallel architecture.

## 2. Normalized prefix geometry

Let

\[
0<\lambda_t^{(g)}\le1.
\]

Reuse the mature log-space Gated Delta parameterization in FP32,

\[
g_t^{(g)}
=-\exp(A_g)\operatorname{softplus}(\ell_t^{(g)}+\delta_g),
\qquad
\lambda_t^{(g)}=\exp(g_t^{(g)}),
\]

with one geometry-forgetting channel per head. Exact no-forgetting
`lambda_g=1` is a forced algebraic intervention for provenance tests. Prefix
products are accumulated in log space inside chunks.

Update

\[
m_t=\lambda_t^{(g)}m_{t-1}+1,
\]

\[
J_t=\lambda_t^{(g)}J_{t-1}+u_tu_t^T,
\qquad
D_t=\lambda_t^{(g)}D_{t-1}+u_th_t^T.
\]

Then

\[
H_t=J_t/m_t,
\qquad R_t=D_t/m_t,
\]

Because normalization maps a zero projected feature to zero, reachable states
satisfy

\[
0\le\operatorname{tr}(H_t)\le1,
\qquad 0\preceq H_t\preceq I,
\]

with trace one exactly when every positively weighted remembered normalized
geometry feature has unit norm. Under the executable
`F.normalize(..., eps=1e-12)` contract, this includes raw projected features
whose norm is at least `eps`; a nonzero sub-`eps` feature has norm strictly
between zero and one after normalization and therefore contributes less than
unit trace. A zero feature is a deterministic no-direction observation, not an
invalid state. Forgetting changes the remembered geometry but not these
normalized Gram facts. Let

\[
\gamma_g=\sigma(\widehat\gamma_g)\in(0,1).
\]

`gamma_g` is one learned static strength per head. Forced `gamma_g=0` is the
exact identity-geometry reduction point; it is a reduction intervention rather
than an additional maintained model. Initialize `gamma_hat_g = -2` and the
direct `h` projection to zero so training starts from a weak prefix solve.

## 3. Property-preserving transpose-dual solve adapter

The Gram and driven moments generate two distinct causal system coordinates,

\[
\boxed{
X_t^{(H)}=\gamma_g\left(H_t-\frac1rI\right),
\qquad
X_t^{(R)}=\gamma_gR_t.
}
\]

The centered Gram term contributes prefix occupancy geometry without making an
isotropic prefix a preferred direction. The general cross moment `R_t` supplies
all signed and asymmetric coordinates. They must remain separate through the
nonlinear bounded chart: summing them first would permit the exact collapse
`D+J=sum u(h+u)^T` and make one moment redundant. If an earlier construction writes
`C_t = sum u_t c_t^T` followed by a fixed `W_drive`, then

\[
C_tW_{\mathrm{drive}}
=\sum_i u_i(W_{\mathrm{drive}}^Tc_i)^T.
\]

Thus projecting `h_t = W_drive^T c_t` directly removes a redundant
`r x r` matrix product exactly. No full-matrix whitening or post-moment drive
GEMM is part of the causal chart.

Define the fixed smooth radial map

\[
\mathcal B_c(Y)=c\frac{Y}{\sqrt{c^2+\|Y\|_F^2}},
\]

with separate allocations

\[
c_H=c_R=\frac18,
\qquad c=c_H+c_R=\frac14.
\]

Construct the strict-triangular factors only after separately bounding the two
moment coordinates:

\[
N_t^-=\mathcal B_{c_H}(\operatorname{tril}(X_t^{(H)},-1))
+\mathcal B_{c_R}(\operatorname{tril}(X_t^{(R)},-1)),
\]

\[
N_t^+=\mathcal B_{c_H}(\operatorname{triu}(X_t^{(H)},1))
+\mathcal B_{c_R}(\operatorname{triu}(X_t^{(R)},1)).
\]

Both are strictly triangular, have Frobenius and operator norm below `c`, and
are identity-to-first-order coordinates at zero. Bound the diagonal log scale
with separate allocations `s_H=s_R=1/8` and total `s_max=1/4`,

\[
\delta_t=s_H\tanh\!\left(
\frac{\operatorname{diag}(X_t^{(H)})}{s_H}
\right)
+s_R\tanh\!\left(
\frac{\operatorname{diag}(X_t^{(R)})}{s_R}
\right),
\qquad
\Sigma_t=\operatorname{Diag}(\exp(\delta_t)).
\]

In general

\[
\mathcal B_{c_H}(X^{(H)})+\mathcal B_{c_R}(X^{(R)})
\ne \mathcal B_c(X^{(H)}+X^{(R)}),
\]

and the same is true of the separate diagonal `tanh` maps. Thus `(J,D)` cannot
be replaced by one summed moment without changing the operator.

The prefix-generated causal solve system is

\[
\boxed{
M_t=(I+N_t^-)\Sigma_t(I+N_t^+),
\qquad P_t=M_t^{-1}.
}
\]

This bounded LDU chart, rather than the bidirectional
`I + F F^T + Omega` chart, is the canonical Causal LSSO system. It is dense and
full-rank but owns its factors directly, so no token-local Cholesky, LU, QR, or
full-matrix factorization is required.

The solve and the Delta recurrence are separate operators. Geometry exposes
one primal action and its dual:

\[
\boxed{
P_t=M_t^{-1},
\qquad
P_t^{-T}=M_t^T.
}
\]

The implementation never forms either matrix. Write
`L_t^B=I+N_t^-` and `U_t^B=I+N_t^+`. It applies `P_t a` by solving
`L_t^B y=a`, scaling `z=Sigma_t^-1 y`, and solving `U_t^B d=z`. It applies
`P_t^-T b` as the direct product `(U_t^B)^T Sigma_t (L_t^B)^T b`. Multiple
edit slots are packed as right-hand sides. These are unit-triangular solves
with factors already generated by the prefix, not factorizations of a dense
matrix.

For arbitrary solve-domain vectors `a,b`, the primal/dual interface guarantees

\[
\boxed{
(P_t^{-T}b)^T(P_ta)=b^Ta.
}
\]

Consequently a local edit is transported by similarity:

\[
I-(P_ta)(P_t^{-T}b)^T
=P_t(I-ab^T)P_t^{-1}.
\]

This preserves edit rank, eigenvalues, and pairing without coupling the Delta
engine to a particular factorization. At `gamma_g=0`, both coordinates vanish, every factor is
identity, and therefore `M_t=P_t=I`. The radial and diagonal maps are identity
to first order, so at the identity

\[
\mathrm D M[\Delta X^{(H)},\Delta X^{(R)}]
=\Delta X^{(H)}+\Delta X^{(R)}.
\]

As a map from unconstrained ambient `(X^(H),X^(R))` coordinates, the chart
consequently has full `r^2` local differential rank at `X^(H)=X^(R)=0` and
contains an open neighborhood of the identity rather than a symmetric,
orthogonal-only, diagonal, or low-rank submanifold. This derivative is with
respect to chart inputs, not with respect to `R` after the structural
`gamma_g=0` reduction, where `dX^(R)/dR=0`. Prefix reachability at nonzero
`gamma_g` is separate: at short lengths the moment construction restricts
which chart coordinates are attainable, as stated in Section 7.

The factor bounds are explicit:

\[
\|I+N_t^\pm\|_2\le1+c,
\qquad
\|(I+N_t^\pm)^{-1}\|_2\le\frac1{1-c},
\]

\[
\|\Sigma_t\|_2,\|\Sigma_t^{-1}\|_2\le e^{s_{\max}}.
\]

Hence

\[
\boxed{
\|M_t\|_2\le(1+c)^2e^{s_{\max}},
\qquad
\|M_t^{-1}\|_2\le\frac{e^{s_{\max}}}{(1-c)^2},
}
\]

and

\[
\boxed{
\kappa_2(P_t)=\kappa_2(M_t)
\le e^{2s_{\max}}\left(\frac{1+c}{1-c}\right)^2
\approx4.58.
}
\]

Thus primal, dual, and similarity amplification are controlled by construction,
without an unbounded transpose action or a posterior repair guard.

For later bounds define

\[
B_P:=\frac{e^{s_{\max}}}{(1-c)^2}\approx2.283,
\qquad
B_D:=(1+c)^2e^{s_{\max}}\approx2.006.
\]

## 4. K bounded asymmetric edit factors

For `j in {1,...,K}`, project

\[
b_{t,j}\in(0,2)^r,
\quad w_{t,j}\in(0,2)^{d_v},
\]

Construct the local edit entirely in the fixed solve domain:

\[
b_{t,j}=2\sigma(\widehat b_{t,j}),
\qquad
w_{t,j}=2\sigma(\widehat w_{t,j}).
\]

\[
a_{t,j}=k_{t,j},
\qquad
\bar b_{t,j}=b_{t,j}\odot k_{t,j}.
\]

There is no separate write-direction gate. Adding a bounded sigmoid gate here
would make exact GDN2 and DeltaProduct reductions require an unattainable
finite-logit endpoint while duplicating direction control already owned by the
projected normalized key. The baseline write direction is therefore structural
rather than a closure limit, and `b` remains the independent coordinatewise
erase control.

The base nonnegative pairing is

\[
\tau_{t,j}=\bar b_{t,j}^Ta_{t,j}
=\sum_i b_{t,j,i}k_{t,j,i}^2.
\]

For a nonzero normalized edit key, finite sigmoid logits give `0 < tau < 2`.
A zero projected key remains zero under normalization and gives the identity
edit with `tau=0`. Thus the finite contract is `0 <= tau < 2`; including the
continuous gate closure and explicit boundary interventions gives

\[
0\le\tau_{t,j}\le2.
\]

The coordinatewise base pair guarantees only the scalar pairing above. It does
not make the rank-one generator $a\bar b^T$ positive semidefinite. For `r >= 2`,
the smallest eigenvalue of its symmetric part is

\[
\lambda_{\min}\!\left(
\frac{a\bar b^T+\bar b a^T}{2}\right)
=\frac{a^T\bar b-\|a\|_2\|\bar b\|_2}{2}\le0,
\]

with equality only when the nonzero factors are positively collinear. For
`r=1` the symmetric part is the nonnegative scalar `a bar_b`; there is no
non-normal direction. Calling the general higher-dimensional asymmetric edit
dissipative would therefore be incorrect.

Define the canonical write vector, erase covector, and value target:

\[
\boxed{
d_{t,j}=P_ta_{t,j},
\qquad
e_{t,j}=P_t^{-T}\bar b_{t,j},
\qquad
z_{t,j}=w_{t,j}\odot v_{t,j}.
}
\]

Each pairing remains exactly

\[
\boxed{e_{t,j}^Td_{t,j}=\tau_{t,j}\in[0,2].}
\]

The declared gates give

\[
\|a_{t,j}\|_2\le1,
\qquad
\|\bar b_{t,j}\|_2\le2,
\]

and therefore

\[
\|d_{t,j}\|_2\le B_P,
\qquad
\|e_{t,j}\|_2\le2B_D.
\]

These are finiteness bounds, not tight trainability guarantees. Realized norms
must be much smaller than their conservative worst cases.

## 5. Ordered block update and read

Project one channel-wise associative decay

\[
\alpha_t\in(0,1]^r
\]

using the GDN2/KDA log-space form

\[
g_t^{(s)}
=-\exp(A_s)\odot\operatorname{softplus}(\ell_t^{(s)}+\delta_s),
\qquad
\alpha_t=\exp(g_t^{(s)}),
\]

evaluated and retained in FP32 for the FP32 associative-state update.

and apply it once:

\[
S_{t,0}=\operatorname{Diag}(\alpha_t)S_{t-1}.
\]

Perform `K` ordered prediction-error edits:

\[
\boxed{
S_{t,j}
=S_{t,j-1}
+d_{t,j}(z_{t,j}-S_{t,j-1}^Te_{t,j})^T,
\qquad j=1,\ldots,K.
}
\]

Set `S_t = S_{t,K}`. Read is a covector action, so the solve-conditioned query
uses the same dual interface as erase:

\[
\boxed{
\chi_t=P_t^{-T}q_t
=M_t^Tq_t.
}
\]

Normalized `q_t` and the bounded dual action give

\[
\|\chi_t\|_2\le B_D.
\]

and read-after-write output is

\[
\boxed{o_t=S_t^T\chi_t.}
\]

All edits share the same prefix adapter. There is no dense transition branch
or separately maintained architecture for each value of `K`.

The associative state `S_t` always remains in one fixed ambient memory basis.
It is never re-expressed in the changing `P_t` coordinate system. Storing it in
the current solve domain would require a dense cross-frame transport involving
`P_t^{-1}P_{t-1}` at every token, reintroducing inverse conditioning and
destroying the clean Delta/WY recurrence.

This fixes the original-versus-dualized update decision. An untransformed
factor `I-a_t b_t^T` depends only on current local controls and leaves prefix
geometry outside the memory edit. Storing the whole memory in the current
dualized domain makes its coordinates change every token. The canonical middle
choice keeps `S_t` fixed but applies

\[
\boxed{
I-d_te_t^T=P_t(I-a_t\bar b_t^T)P_t^{-1}.
}
\]

Thus the current token determines which local edit to request, while all
history represented in `(m_t,J_t,D_t)` determines how that edit acts on the
fixed memory. Original-token information and dualized-prefix information are
composed rather than treated as competing inputs.

## 6. Exact containment

Here exact means equality at finite shared gate parameters, together with the
explicit structural identity-geometry switch. It never means convergence as a
sigmoid logit tends to infinity. The open
`(0,2)` ranges of `b` and `w` match the corresponding finite-logit baseline
gates; `[0,2]` below denotes their continuous stability closure, not a step
needed by the reductions.

Reduction equality is defined on the common observable state: token outputs,
associative state `S`, and any shared convolution caches, together with their
Jacobians on shared inputs and parameters. SolveDelta's extra `(m,J,D)` cache
continues to update under identity geometry but is inert with respect to this
projection. A loss placed directly on that auxiliary cache has no GDN2 state
counterpart and is not part of the reduction claim.

### GDN2 and ordinary Delta family

Set `K=1` and identity geometry. Then

\[
d_{t,1}=k_{t,1},
\qquad e_{t,1}=b_{t,1}\odot k_{t,1},
\qquad\chi_t=q_t,
\]

which is exactly GDN2 in this state orientation. Published gate ties recover
KDA, GDN, and DeltaNet.

### DeltaProduct-K

Set identity geometry. For all `K` edits, tie the erase scalar `b=beta`, set the
value target `z=beta v`, and structurally disable associative decay for the
ungated baseline. Each edit becomes

\[
S\leftarrow S+ k\,\beta(v-S^Tk)^T,
\]

so their ordered composition is DeltaProduct-`K`. Finite gates in `(0,2)`
include the negative-eigenvalue regime `beta>1`; the exact Householder value
`beta=2` belongs only to the declared stability closure.

All reductions require deterministic output, projected-state, and shared
gradient equality. Ordinary DeltaNet uses the same structural no-decay
intervention; gated reductions retain their matched finite decay.

## 7. Guaranteed expression gains

One finite nonzero rank-one factor

\[
I-de^T
\]

fixes an `r-1` dimensional subspace and has only one non-unit eigenvalue. The
zero edit is the identity and fixes the full space. `K` ordered factors can
produce an identity-plus-rank-at-most-`K` transition. For `K >= 2` they can act
on a two-dimensional subspace, produce nonsymmetric rank-two changes, and at
finite gates realize planar rotation-contractions with a nonreal conjugate
eigenvalue pair. Exact orthogonal rotations arise only at the `beta=2`
Householder closure endpoint and are not a finite-parameter claim.

Within each factor, the transpose-dual frame maps the solve-domain pair into
generally non-collinear ambient write and erase factors while preserving their
exact pairing. Prefix-dependent `P_t` makes `(d,e,chi)` depend on all tokens
represented in the causal geometry state. Thus two prefixes with the same
current edit token but different geometry summaries can induce different
memory-edit coefficients.

The solve-frame chart itself has no low-rank or orthogonal-only local ceiling.
As a map on its ambient matrix coordinates, its joint identity differential is
`D M[Delta X_H,Delta X_R]=Delta X_H+Delta X_R`, so an unconstrained
`X^(R)` coordinate supplies all `r^2` first-order chart directions. For any
fixed nonzero `gamma_g`, `R -> X^(R)=gamma_g R` preserves those local
directions. This is a property of the chart and a conditional reachability
statement, not a claim that every causal prefix can instantly reach every
coordinate. At prefix length `t`,

\[
\operatorname{rank}(D_t),\operatorname{rank}(R_t)\le \min(t,r),
\qquad
\operatorname{col}(D_t)\subseteq
\operatorname{span}\{u_1,\ldots,u_t\}.
\]

Therefore a short prefix `t < r` accesses only the directions induced by its
history span. Once `t >= r` and the remembered `u` vectors span
\(\mathbb R^r\), variations of the drives `h` can locally generate arbitrary `R`
directions; this is a reachability condition, not an automatic guarantee for
every prefix. The two moment channels are not separately identifiable to first
order; their additional capacity appears through their independent nonlinear
responses away from identity. At the chart level this is a strict local
capacity advantage over diagonal preconditioners, rank-`p` Woodbury charts,
one-sided orthogonal-scale charts, and a single butterfly sweep. It does not
remove the separate rank-`K` ceiling of one token's associative edit. That
ceiling is inherited from DeltaProduct and is handled by the static
`num_edits` capacity/compute hyperparameter; it is not a new solve-frame
limitation.

This is full-prefix conditioning, not lossless full-context storage. With no
forgetting, `J_t` and `D_t` contain contributions from every prefix token, but
two histories with the same `(m_t,J_t,D_t)` are indistinguishable to the solve
adapter. These are explicit weighted second-order collisions: `J` is an
auto-moment and `D` is a cross moment. `U^TU` also discards unrepresented order
information; exponential forgetting adds recency weights but does not make the
moments injective.

History collision itself is not new. A Delta-family recurrence also maps an
unbounded history into a fixed associative state `S_t`, so distinct prefixes
can induce the same future state. Its local erase factor already contains the
second-order product `k_t k_t^T`, or `d_t e_t^T` in an asymmetric form; its
recursive erase/write expansion additionally contains ordered products and
higher-order cross-time interactions, and is not generally reducible to one
second-order moment.
SolveDelta instead maintains two simultaneous equivalence relations: equality of
`(m,J,D)` gives the same solve adapter, while equality of `S` gives the same
associative content state. A geometry collision is therefore not a collision
of the complete SolveDelta state unless both summaries agree. Conversely, geometry
can distinguish some prefixes that collide in `S`, and `S` can distinguish
some prefixes that collide in geometry.

These facts establish strict fixed-width operator separations. They do not
guarantee better task loss or imply containment of arbitrary dense transitions,
unbounded-edit DeltaProduct, or deeper/wider networks.

## 8. Known limitations and open problems

The single route retains the following explicit boundaries.

1. **Non-normal transients.** `e^T d in [0,2]` controls the nontrivial
   eigenvalue of each edit, not its largest singular value. Channel decay and
   the bounded frame do not prove global stability across changing
   adapters and `K`-factor products. GDN2 already has mild non-normality from
   asymmetric erase factors; SolveDelta adds the similarity bound

   \[
   \|P_tTP_t^{-1}\|_2\le\kappa_2(P_t)\|T\|_2,
   \]

   so frame conditioning is the project-specific amplification channel. The
   fixed LDU bound makes it finite but does not turn it into a contraction.
2. **Triangular solve execution.** The canonical adapter has no dense
   factorization and costs `O(r^2)` per right-hand-side bundle, but each
   triangular solve has dependencies along rank. Training must batch token/head
   systems and use block-level TRSM or an equivalent fused kernel. The
   remaining acceptance metrics are triangular solve residual, realized
   `cond(M)`, action norms, pairing drift, and complete-layer throughput.
3. **Deliberate geometry capacity.** At `r=d_v=128`, full FP32 `J+D+S` costs
   approximately 192 KiB per head before transient workspace. Packing the
   symmetric `J` triangle reduces the recurrent representation to
   approximately 160.25 KiB per head; `D` and `S` are generally dense. This is
   an accepted fixed-size capacity cost: `J` preserves sign-robust occupancy
   geometry while `D` preserves directional driven geometry. It is not a
   compression blocker, but both moments must show utilization and task value.
4. **Identifiability.** Solve-adapter orientation, dual-factor scaling, and `K`
   edit slots introduce gauges. The contract fixes update order and gate roles,
   but utilization must be measured.
5. **Attribution.** Gains must survive interventions that remove prefix
   geometry, reduce `K`, or remove the independent geometry feature under
   matched budgets.
6. **Inheritance scope.** The original LSSO reference remains a provenance
   diagnostic for normalized prefix moments and solved contextual adaptation,
   not a coordinate oracle. The bounded LDU system, transpose-dual action, and
   normalized forgetting model are causal constructions rather than exact
   reformulations of the bidirectional mixer.

## 9. Falsification criteria

SolveDelta should not remain the main operator unless it:

1. matches its FP64 token recurrence in every optimized path;
2. recovers GDN2 at `K=1` and DeltaProduct-`K` exactly;
3. uses prefix geometry on controlled history-collision tasks;
4. shows that the chosen `K` passes ordinary DeltaProduct reduction and
   throughput checks;
5. remains finite and trainable over the declared long-sequence envelope; and
6. reaches acceptable complete-layer throughput at `r=128` and justifies the
   fixed two-moment cache by intervention evidence.

Failure is evidence against this operator, not a reason to preserve another
candidate inside the repository.

# SolveDelta ChunkSolve--WY Execution Contract

This document fixes the only optimized training path for SolveDelta. It is an
exact chunkwise execution of the token recurrence in
`causallsso/reference.py`; it is not a second definition of the operator. If
this document and an optimized kernel disagree with the FP64 token oracle, the
kernel is wrong. `docs/VALIDATION_PLAN.md` remains the owner of numerical
ceilings and release gates.

The production graph is

\[
\boxed{
\text{projection/conv}
\rightarrow \text{geometry boundary scan}
\rightarrow \text{fused ChunkSolve--WY prepare}
\rightarrow \text{state scan}
\rightarrow \text{output}
}
\]

and its reverse is

\[
\boxed{
\text{output/state transpose}
\rightarrow \text{WY transpose solve}
\rightarrow \text{frame transpose action}
\rightarrow \text{chart transpose}
\rightarrow \text{geometry transpose}
\rightarrow \text{boundary-scan adjoint}.
}
\]

The first native specialization is

\[
r=d_k=128,\qquad K=1,\qquad C=32,
\]

with BF16 activation and matrix operands, FP32 accumulation and scalar
evaluation, and FP32 continuation states. The formulas below are written for
one batch element, one head, and one chunk. Batch and head dimensions are
independent. Column-vector orientation is used in the frame, while chunk
factors are stored by row.

The following are forbidden production interfaces:

- a public or cross-component forward `d/e/chi` staging ABI; Section 8's
  explicitly benchmarked private backward cache is not an operator interface;
- aliases such as `qg/kg/ag` constructed only to enter a generic DPLR API;
- a synthetic `3C=96` action dimension;
- `compact_pair -> coefficients -> leaf` or another chain of entrywise VJPs;
- tokenwise materialization of `H`, `R`, `L`, `U`, or their dense gradients;
- literal materialization of the inverse-decay gauge `exp(-G_i) * d_i`;
- a token-local factorization, approximate inverse, or data-dependent fallback.

## 1. Shapes and state order

Let `d_v` be the per-value-head width. For token `t` and edit slot `j`,

\[
u_t,h_t,q_t,k_{t,j}\in\mathbb R^r,\qquad
v_{t,j}\in\mathbb R^{d_v}.
\]

The operator continuation state is

\[
m_t\in\mathbb R,\qquad
J_t,D_t\in\mathbb R^{r\times r},\qquad
S_t\in\mathbb R^{r\times d_v}.
\]

Every valid token performs exactly this order:

1. update `(m,J,D)` once;
2. construct one shared bounded LDU frame;
3. decay `S` once;
4. execute edits `j=1,...,K` in order;
5. read after edit `K`.

A valid reset clears the operator and convolution states immediately before
the token. An invalid token emits zero and leaves every state unchanged. A
native packed implementation must express resets as exact segment boundaries;
padding lanes have the identity transition and zero output.

## 2. Frontend forward

For hidden state `x_t`, one linear projection produces

\[
\begin{aligned}
(&u_t^0,h_t,q_t^0,k_{t,1}^0,\ldots,k_{t,K}^0,
v_{t,1}^0,\ldots,v_{t,K}^0,\\
&\widehat b_{t,1},\ldots,\widehat b_{t,K},
\widehat w_{t,1},\ldots,\widehat w_{t,K},
\ell_t^{(g)},\ell_t^{(s)})
=W_{\rm in}x_t+b_{\rm in}.
\end{aligned}
\]

The geometry strength is not token-projected. It is one static parameter per
head:

\[
\gamma_g=\operatorname{sigmoid}(\widehat\gamma_g).
\]

Query, packed keys, and packed values independently use a bias-free depthwise
causal convolution of width four. For any one of these branches,

\[
c_t=\sum_{j=0}^{3}\theta_j x_{t-3+j},\qquad
\operatorname{SiLU}(c_t)=c_t\operatorname{sigmoid}(c_t).
\]

The four raw projected inputs, ordered oldest to newest, form the recurrent
convolution cache. Geometry, drive, gates, and decay logits bypass convolution.
With the structural convolution switch off, Q/K/V still receive SiLU.

Using `epsilon=1e-12`, define

\[
\operatorname{norm}(x)=\frac{x}{\max(\|x\|_2,\epsilon)}.
\]

Then

\[
u_t=\operatorname{norm}(u_t^0),\qquad
q_t=\operatorname{norm}(q_t^{\rm conv}),\qquad
k_{t,j}=\operatorname{norm}(k_{t,j}^{\rm conv}),
\]

and `v` is the convolved SiLU value without normalization. Gate and decay
nonlinearities are evaluated in FP32:

\[
b_{t,j}=2\operatorname{sigmoid}(\widehat b_{t,j})\in(0,2)^r,
\]

\[
w_{t,j}=2\operatorname{sigmoid}(\widehat w_{t,j})\in(0,2)^{d_v},
\]

\[
g_t^{(g)}
=-\exp(A_g)\operatorname{softplus}(\ell_t^{(g)}+\delta_g)\le0,
\qquad \lambda_t=\exp(g_t^{(g)}),
\]

\[
g_t^{(s)}
=-\exp(A_s)\odot\operatorname{softplus}(\ell_t^{(s)}+\delta_s)
\in\mathbb R^r_{\le0}.
\]

The structural no-associative-decay intervention replaces `g^(s)` by exact
zeros. The structural identity-geometry intervention replaces `gamma_g` by
exact zero. Neither reduction is represented by a limiting logit.

After the operator returns all head outputs `o_t`, the containing layer emits

\[
y_t^{\rm layer}
=W_{\rm out}\operatorname{concat}_{h=1}^{H}(o_{t,h})+b_{\rm out}.
\]

## 3. Geometry boundary scan

The token recurrence is

\[
m_i=\lambda_i m_{i-1}+1,
\]

\[
J_i=\lambda_iJ_{i-1}+u_iu_i^T,
\qquad
D_i=\lambda_iD_{i-1}+u_ih_i^T,
\]

\[
H_i=J_i/m_i,\qquad R_i=D_i/m_i.
\]

For one chunk, define

\[
a_i=\prod_{p=1}^{i}\lambda_p,
\qquad
w_{is}=\prod_{p=s+1}^{i}\lambda_p,
\quad 1\le s\le i\le C,
\]

with the empty product `w_ss=1`. Then

\[
m_i=a_im_0+\sum_{s\le i}w_{is},
\]

\[
J_i=a_iJ_0+\sum_{s\le i}w_{is}u_su_s^T,
\qquad
D_i=a_iD_0+\sum_{s\le i}w_{is}u_sh_s^T.
\]

Let

\[
\eta_i=\frac{a_i}{m_i},\qquad
\omega_{is}=\frac{w_{is}}{m_i}.
\]

The exact local normalized moments are therefore

\[
\boxed{
H_i=\eta_iJ_0+\sum_{s\le i}\omega_{is}u_su_s^T,
\qquad
R_i=\eta_iD_0+\sum_{s\le i}\omega_{is}u_sh_s^T.
}
\]

Only `(m_0,J_0,D_0)` crosses the chunk boundary. The local expression has no
cross-chunk recurrence, but the boundary remains the exact summary of all
earlier valid tokens.

For a complete chunk, define its affine summary

\[
\mathcal G_c=(a_C,p_C,Q_C,V_C),
\]

\[
p_C=\sum_{s\le C}w_{Cs},\quad
Q_C=\sum_{s\le C}w_{Cs}u_su_s^T,\quad
V_C=\sum_{s\le C}w_{Cs}u_sh_s^T.
\]

For consecutive summaries `2 after 1`, association is

\[
(a_2,p_2,Q_2,V_2)\circ(a_1,p_1,Q_1,V_1)
\]

\[
=
(a_2a_1, p_2+a_2p_1, Q_2+a_2Q_1, V_2+a_2V_1).
\]

The Triton scan computes all chunk boundaries and the final geometry state
with this exact affine monoid.

## 4. Bounded LDU chart

Construct two separate coordinates

\[
X_i^{(H)}=\gamma_g(H_i-I/r),
\qquad
X_i^{(R)}=\gamma_gR_i.
\]

For `c>0`, the radial map is

\[
\mathcal B_c(X)=\frac{cX}{\sqrt{c^2+\|X\|_F^2}}.
\]

The fixed constants are

\[
c_H=c_R=s_H=s_R=\frac18.
\]

Strict factors use explicit masks:

\[
N_i^-
=\mathcal B_{c_H}(\operatorname{tril}(X_i^{(H)},-1))
+\mathcal B_{c_R}(\operatorname{tril}(X_i^{(R)},-1)),
\]

\[
N_i^+
=\mathcal B_{c_H}(\operatorname{triu}(X_i^{(H)},1))
+\mathcal B_{c_R}(\operatorname{triu}(X_i^{(R)},1)).
\]

The diagonal coordinate is

\[
\delta_i
=s_H\tanh\!\left(\frac{\operatorname{diag}X_i^{(H)}}{s_H}\right)
+s_R\tanh\!\left(\frac{\operatorname{diag}X_i^{(R)}}{s_R}\right),
\]

\[
\sigma_i=\exp(\delta_i),\qquad
\Sigma_i=\operatorname{Diag}(\sigma_i).
\]

Finally,

\[
L_i=I+N_i^-,\qquad U_i=I+N_i^+,
\]

\[
M_i=L_i\Sigma_iU_i,qquad P_i=M_i^{-1}.
\]

States reached from zero have symmetric `J_i` and `H_i`, but the executable
oracle accepts a full `r x r` initial `J_0` and defines its full ambient VJP.
The production ABI therefore retains independent strict-lower and strict-upper
H coordinates. On an internally known symmetric boundary, forward may share
their norm value and transpose factor values, but it must preserve the two
ambient dependencies in reverse. R is asymmetric in both forward and reverse.
No packed-symmetric state or lower-only cotangent is introduced by this path.

For later implementation, if `A` is one strict projection of `H-I/r` or `R`,
then

\[
\mathcal B_c(\gamma_g A)=\mu A,
\qquad
\mu=\frac{c\gamma_g}{\sqrt{c^2+\gamma_g^2\|A\|_F^2}}.
\]

Thus every strict action is a scalar radial coefficient multiplying an exact
boundary-plus-local moment action.

## 5. ChunkSolve forward

For edit slot `j`, define the solve-domain write direction, erase covector,
and value target

\[
a_{i,j}=k_{i,j},\qquad
x_{e,i,j}=b_{i,j}\odot k_{i,j},\qquad
z_{i,j}=w_{i,j}\odot v_{i,j}.
\]

The primal action is evaluated by two unit-triangular solves:

\[
L_i y_{i,j}=k_{i,j},
\]

\[
p_{i,j}=\sigma_i^{-1}\odot y_{i,j},
\]

\[
U_i d_{i,j}=p_{i,j}.
\]

The erase and query dual actions are direct products. For
`x in {x_e,q}`,

\[
\ell_{x,i}=L_i^Tx_i,
\qquad
s_{x,i}=\sigma_i\odot\ell_{x,i},
\qquad
r_{x,i}=U_i^Ts_{x,i}.
\]

Set

\[
e_{i,j}=r_{x_e,i,j},\qquad \chi_i=r_{q,i}.
\]

This is the exact primal/dual pair

\[
d_{i,j}=P_ik_{i,j},\qquad
e_{i,j}=P_i^{-T}(b_{i,j}\odot k_{i,j}),\qquad
\chi_i=P_i^{-T}q_i,
\]

and therefore

\[
\boxed{
e_{i,j}^Td_{i,j}
=(b_{i,j}\odot k_{i,j})^Tk_{i,j}.
}
\]

The production RHS panel is `r x C`, namely `128 x 32`. The two identical
dual schedules may be temporarily batched as `128 x 64` if profiling shows a
gain. Neither choice changes the ABI or introduces a `3C` dimension.

## 6. Boundary GEMM and local semiseparable action

For vectors `a,b,x in R^r`, strict-lower and strict-upper masked outer actions
are

\[
[\operatorname{tril}(ab^T,-1)x]_j
=a_j\sum_{\ell<j}b_\ell x_\ell,
\]

\[
[\operatorname{triu}(ab^T,1)x]_j
=a_j\sum_{\ell>j}b_\ell x_\ell.
\]

Each generator action costs `O(r)` through a coordinate prefix or suffix.
Using the local geometry expansion, for example,

\[
\operatorname{tril}(H_i,-1)x_i
=\eta_i\operatorname{tril}(J_0,-1)x_i
+\sum_{s\le i}\omega_{is}
\operatorname{tril}(u_su_s^T,-1)x_i,
\]

\[
\operatorname{tril}(R_i,-1)x_i
=\eta_i\operatorname{tril}(D_0,-1)x_i
+\sum_{s\le i}\omega_{is}
\operatorname{tril}(u_sh_s^T,-1)x_i.
\]

Across all chunk RHS, the first terms are broad products of the form

\[
\operatorname{tril}(J_0,-1)
[c_1x_1,\ldots,c_Cx_C]\in\mathbb R^{r\times C},
\]

and similarly for `D_0`, upper masks, and transposes. They are
`[128,128] x [128,32]` Tensor Core GEMMs. The local sums cost
`O(C^2r)` and use the exact masked-outer prefix/suffix identities.

Partition the coordinate axis into eight blocks of 16. A lower solve executes

\[
y^{(b)}=k^{(b)}-N^-_{b,<b}y^{(<b)}
\]

followed by the unit-lower `16 x 16` diagonal micro-solve. An upper solve uses
the reverse block order:

\[
d^{(b)}=p^{(b)}-N^+_{b,>b}d^{(>b)}.
\]

Off-diagonal updates are MMA products with 32 RHS. Only the strict dependency
inside one diagonal 16-coordinate block requires a warp prefix/suffix solve.
Any appearance of 96 rows here means `128-2*16` remaining coordinates, never
`3*32` action streams.

Radial norms are also computed without dense token moments. For a strict
unnormalized moment recurrence

\[
A_i=\lambda_iA_{i-1}+B_i,
\]

the exact norm recurrence is

\[
\boxed{
n_i=\|A_i\|_F^2
=\lambda_i^2n_{i-1}
+2\lambda_i\langle A_{i-1},B_i\rangle
+\|B_i\|_F^2.
}
\]

The cross term is another boundary-plus-local masked-outer contraction. Norms,
cross terms, and radial coefficients are accumulated in deterministic FP32.

The generic recurrence above tracks unnormalized strict parts of `J_i` or
`D_i`. The chart normalization and strength must be applied explicitly. Define

\[
A_{H,i}^-=\operatorname{tril}(J_i,-1),
\qquad
A_{H,i}^+=\operatorname{triu}(J_i,1),
\]

\[
n_{H,i}^-=\|A_{H,i}^-\|_F^2,
\qquad
n_{H,i}^+=\|A_{H,i}^+\|_F^2.
\]

Since centering changes only the diagonal,

\[
\operatorname{tril}(X_i^{(H)},-1)
=\frac{\gamma_g}{m_i}A_{H,i}^-,
\qquad
\operatorname{triu}(X_i^{(H)},1)
=\frac{\gamma_g}{m_i}A_{H,i}^+.
\]

Consequently

\[
\boxed{
N_{H,i}^{\pm}=\mu_{H,i}^{\pm}A_{H,i}^{\pm},
\qquad
\mu_{H,i}^{\pm}
=\frac{c_H\gamma_g/m_i}
{\sqrt{c_H^2+\gamma_g^2n_{H,i}^{\pm}/m_i^2}}.
}
\]

For the driven moment, define

\[
A_{R,i}^-=\operatorname{tril}(D_i,-1),
\qquad
A_{R,i}^+=\operatorname{triu}(D_i,1),
\]

\[
n_{R,i}^{\pm}=\|A_{R,i}^{\pm}\|_F^2.
\]

Then

\[
\boxed{
N_{R,i}^{\pm}=\mu_{R,i}^{\pm}A_{R,i}^{\pm},
\qquad
\mu_{R,i}^{\pm}
=\frac{c_R\gamma_g/m_i}
{\sqrt{c_R^2+\gamma_g^2n_{R,i}^{\pm}/m_i^2}}.
}
\]

The diagonal inputs are likewise explicit:

\[
x_{H,i}^{\rm diag}
=\gamma_g\left(\frac{\operatorname{diag}J_i}{m_i}-\frac1r\right),
\qquad
x_{R,i}^{\rm diag}
=\gamma_g\frac{\operatorname{diag}D_i}{m_i}.
\]

They are the two arguments of the diagonal `tanh` maps in Section 4. Kernel
code must not infer or reassociate these normalization factors independently.

## 7. Exact stable WY prepare for K=1

The token memory recurrence is

\[
S_{i,0}=\operatorname{Diag}(\alpha_i)S_{i-1},
\qquad \alpha_i=\exp(g_i^{(s)}),
\]

\[
S_i=S_{i,0}+d_i(z_i-S_{i,0}^Te_i)^T,
\qquad o_i=S_i^T\chi_i.
\]

Inside a chunk, define the inclusive channel-wise log decay

\[
G_i=\sum_{p=1}^{i}g_p^{(s)}\in\mathbb R^r.
\]

For `i >= j`, define the elementwise stable ratio

\[
\Delta_{ij}=\exp(G_i-G_j)\in(0,1]^r.
\]

This is the mathematical contract, not an instruction to evaluate one
transcendental per `(i,j,channel)`. Compute only

\[
\alpha_i=\exp(g_i^{(s)})
\]

once per token and channel. Ratios inside a causal tile are generated by

\[
\boxed{
\Delta_{jj}=1,
\qquad
\Delta_{ij}=\alpha_i\odot\Delta_{i-1,j}\quad(i>j),
}
\]

or the algebraically identical reverse-column recurrence. No full
`C x C x r` ratio tensor is written to HBM.

For an off-diagonal tile whose row indices `I` are strictly later than all
column indices `J`, choose a boundary index `b` between the tiles. Monotone
nonpositive decay gives, componentwise,

\[
G_i\le G_b\le G_j,
\qquad i\in I,\ j\in J.
\]

Define tile-local stable gauges

\[
e_i^\star=\exp(G_i-G_b)\odot e_i,
\qquad
q_i^\star=\exp(G_i-G_b)\odot\chi_i,
\]

\[
d_j^\star=\exp(G_b-G_j)\odot d_j.
\]

Both exponents are nonpositive and

\[
\boxed{
T_{IJ}=E_I^\star(D_J^\star)^T,
\qquad
(A_{qd})_{IJ}=Q_I^\star(D_J^\star)^T.
}
\]

Thus fully causal tiles are ordinary Tensor Core GEMMs. A diagonal tile uses
the masked recurrence because one common reference cannot cover both sides of
its causal diagonal without positive exponents. Backward reuses the same tile
boundaries and gauge values; it must not recreate pairwise exponentials.

The strict edit interaction and inclusive query interaction are

\[
T_{ij}=
\begin{cases}
\langle e_i,\Delta_{ij}\odot d_j\rangle,&i>j,\\
0,&i\le j,
\end{cases}
\]

\[
(A_{qd})_{ij}=
\begin{cases}
\langle\chi_i,\Delta_{ij}\odot d_j\rangle,&i\ge j,\\
0,&i<j.
\end{cases}
\]

Define the stable scaled factors

\[
E_{\gamma,i}=\exp(G_i)\odot e_i,
\qquad
Q_{\gamma,i}=\exp(G_i)\odot\chi_i,
\]

\[
D_{{\rm tail},i}=\exp(G_C-G_i)\odot d_i.
\]

Every exponent in `Delta_ij` and `D_tail` is nonpositive. The implementation
must not construct

\[
\exp(-G_i)\odot d_i,
\]

because it can overflow even though all pair interactions are finite.

Let

\[
W=I+T.
\]

Compute `Y in R^(C x r)` and `U_z in R^(C x d_v)` from one unit-lower solve:

\[
\boxed{
W[Y\ U_z]=[E_\gamma\ Z].
}
\]

`W^-1` is mathematical notation only. The production ABI is a solve. A
specialized kernel may use an explicit small inverse internally only if a
matched benchmark and the same numerical gates select it.

Given the chunk input state `S_0`, define

\[
R_z=U_z-YS_0.
\]

All token outputs and the chunk final state are

\[
\boxed{
O=Q_\gamma S_0+A_{qd}R_z,
}
\]

\[
\boxed{
S_C=\operatorname{Diag}(\exp(G_C))S_0+D_{\rm tail}^TR_z.
}
\]

Equivalently, the factorized affine transition is

\[
\boxed{
S_C=
[\operatorname{Diag}(\exp(G_C))-D_{\rm tail}^TY]S_0
+D_{\rm tail}^TU_z.
}
\]

This identity is the boundary consumed by the mature chunk-state scan.

For `K>1`, flatten each token into ordered microsteps `(i,1),...,(i,K)`. Apply
`g_i^(s)` only to microstep `(i,1)`, use zero decay for the remaining edits,
set the microstep query to exact zero for `j<K`, place `chi_i` at `j=K`, and
select only the `j=K` output row. The same formulas then apply to a micro-chunk
of length `KC`. The first native specialization remains K=1.

## 8. Forward ownership, ABI, and private cache

The forward launch graph is:

1. projection, three conv4 branches, normalization, gates, and decays;
2. Triton affine geometry boundary scan;
3. one chunk-owned CUDA `SolveDelta--WY prepare`;
4. FLA-style factorized chunk state scan;
5. a SolveDelta output kernel implementing the two displayed output terms;
6. output projection.

The fused prepare performs, in one ownership boundary:

\[
\text{local geometry}
\rightarrow\text{chart scalars/tiles}
\rightarrow\text{primal/dual actions}
\rightarrow\text{stable pair interactions}
\rightarrow\text{unit-lower WY solve}.
\]

`L`, `U`, `d`, `e`, `chi`, and raw gauge factors are not public forward staging
tensors. Forward generates and consumes them inside this ownership boundary;
training may retain only the explicitly selected private backward cache below.
If state scan and output remain separate kernels, the natural cross-kernel
interface is

\[
\boxed{
(D_{\rm tail},Y,U_z,Q_\gamma,A_{qd},G_C)
}
\]

plus segment metadata. When `d_v=r`, this contains four `C x r`-scale panels:
`D_tail`, `Y`, `Q_gamma`, and `U_z`; workspace accounting must include all
four. `A_qd` is `C x C`.

Training may save a private cache without turning it into a public ABI. The
first performance-oriented cache is

\[
\boxed{W,\ y,\ d,\ s_e,\ s_q,\ e,\ \chi}
\]

with `W` in FP32 and the vector panels in BF16 after their FP32 reductions.
Here `s_e=sigma*ell_e` and `s_q=sigma*ell_q`; the lower dual values required
by reverse are recovered elementwise as `ell=s/sigma` only if a consumer
actually needs them. This cache avoids replaying the primal solves and both
upper dual actions. At
`B=1,T=1024,H=8,r=128`, the six BF16 vector panels occupy about 12 MiB and
`W` occupies about 1 MiB, for about 13 MiB per layer before allocator padding.

The first cache A/B compares:

- the 13 MiB schedule above;
- a roughly 9 MiB schedule omitting `e/chi` and replaying two upper actions;
- a larger schedule retaining selected vector panels in FP32.

Cache variants are scheduling choices and must report complete forward plus
backward wall time and peak memory. Reduced-precision private caches are
numerical approximations of the native VJP subject to the frozen VJP gates;
they do not redefine the FP64 operator. When cached `e/chi` are the exact BF16
bits consumed by forward WY, saving them introduces no additional mismatch.
No cache variant may change the forward formulas or continuation-state
precision.

FLA code may be reused for state scan and, only when its native inputs match
the interface above, output. Reintroducing a generic DPLR staging ABI to reuse
an output kernel is forbidden; a small dedicated output kernel is preferred.

## 9. Backward entry and chunk-state propagation

Let upstream cotangents for one chunk be

\[
\bar O\in\mathbb R^{C\times d_v},\qquad
\bar S_C\in\mathbb R^{r\times d_v}.
\]

Chunks are visited in reverse. The `bar S_0` produced below is added to the
cotangent entering the preceding chunk. This is the factorized reverse of the
state scan; no dense `r x r` transition is required.

From

\[
O=Q_\gamma S_0+A_{qd}R_z,
\qquad
S_C=\Gamma_CS_0+D_{\rm tail}^TR_z,
\]

where `Gamma_C=Diag(exp(G_C))`, obtain

\[
\bar A_{qd}=\operatorname{tril}(\bar O R_z^T,0),
\]

\[
\bar R_z=A_{qd}^T\bar O+D_{\rm tail}\bar S_C,
\]

\[
\bar Q_\gamma=\bar O S_0^T,
\qquad
\bar D_{\rm tail}=R_z\bar S_C^T,
\]

\[
\boxed{
\bar S_0
=Q_\gamma^T\bar O+\Gamma_C\bar S_C-Y^T\bar R_z.
}
\]

The direct contribution to the last cumulative log-decay row is

\[
\bar G_C\mathrel{+}=
\operatorname{rowsum}[\bar S_C\odot(\Gamma_CS_0)].
\]

Finally, from `R_z=U_z-YS_0`,

\[
\bar U_z=\bar R_z,
\qquad
\bar Y=-\bar R_zS_0^T.
\]

All products in this section are broad GEMMs with FP32 accumulation.

## 10. WY transpose solve

Forward solves

\[
WY=E_\gamma,\qquad WU_z=Z.
\]

The exact reverse uses the corresponding transpose solves

\[
\boxed{
W^T\widetilde Y=\bar Y,
\qquad
W^T\widetilde U=\bar U_z.
}
\]

Then

\[
\bar E_\gamma=\widetilde Y,
\qquad
\bar Z=\widetilde U,
\]

\[
\boxed{
\bar T
=\operatorname{tril}
(-\widetilde YY^T-\widetilde U U_z^T,-1).
}
\]

This is the complete VJP of the unit-lower solve. There is no inverse-valued
autograd node and no per-entry solve VJP.

## 11. Stable pair-interaction reverse

Initialize `bar d_i`, `bar e_i`, `bar chi_i`, and every `bar G_i` to zero.
The scaled factors contribute

\[
\bar e_i\mathrel{+}=\exp(G_i)\odot\bar E_{\gamma,i},
\qquad
\bar G_i\mathrel{+}=\bar E_{\gamma,i}\odot E_{\gamma,i},
\]

\[
\bar\chi_i\mathrel{+}=\exp(G_i)\odot\bar Q_{\gamma,i},
\qquad
\bar G_i\mathrel{+}=\bar Q_{\gamma,i}\odot Q_{\gamma,i}.
\]

For every strict interaction `i>j`, define the vector

\[
c^{(T)}_{ij}
=\bar T_{ij}\,\Delta_{ij}\odot e_i\odot d_j.
\]

Its complete reverse is

\[
\bar e_i\mathrel{+}=\bar T_{ij}\,\Delta_{ij}\odot d_j,
\]

\[
\bar d_j\mathrel{+}=\bar T_{ij}\,\Delta_{ij}\odot e_i,
\]

\[
\bar G_i\mathrel{+}=c^{(T)}_{ij},
\qquad
\bar G_j\mathrel{-}=c^{(T)}_{ij}.
\]

For every inclusive query interaction `i>=j`, define

\[
c^{(q)}_{ij}
=\bar A_{qd,ij}\,\Delta_{ij}\odot\chi_i\odot d_j.
\]

Its complete reverse is

\[
\bar\chi_i\mathrel{+}=\bar A_{qd,ij}\,\Delta_{ij}\odot d_j,
\]

\[
\bar d_j\mathrel{+}=\bar A_{qd,ij}\,\Delta_{ij}\odot\chi_i,
\]

\[
\bar G_i\mathrel{+}=c^{(q)}_{ij},
\qquad
\bar G_j\mathrel{-}=c^{(q)}_{ij}.
\]

For the tail factor, let

\[
\Theta_i=\exp(G_C-G_i).
\]

Then

\[
\bar d_i\mathrel{+}=\Theta_i\odot\bar D_{{\rm tail},i},
\]

\[
c_i^{({\rm tail})}
=\bar D_{{\rm tail},i}\odot D_{{\rm tail},i},
\]

\[
\bar G_C\mathrel{+}=c_i^{({\rm tail})},
\qquad
\bar G_i\mathrel{-}=c_i^{({\rm tail})}.
\]

For `i=C`, these two contributions cancel exactly, as required by
`Theta_C=1`.

Because

\[
G_i=\sum_{p\le i}g_p^{(s)},
\]

the token log-decay cotangent is the suffix sum

\[
\boxed{
\bar g_p^{(s)}=\sum_{i\ge p}\bar G_i.
}
\]

Every quantity in this reverse is channel-wise. `Delta_ij` is an r-vector
inside the inner product, not a scalar multiplying an unweighted dot product.

The value target reverse is

\[
\bar w_i\mathrel{+}=\bar z_i\odot v_i,
\qquad
\bar v_i\mathrel{+}=\bar z_i\odot w_i.
\]

## 12. Frame transpose action

The WY reverse supplies `bar d`, `bar e`, and `bar chi`. For one primal action,

\[
U_id_i=p_i,\qquad p_i=\sigma_i^{-1}\odot y_i,\qquad L_iy_i=k_i.
\]

Reverse the upper solve:

\[
\bar p_i=U_i^{-T}\bar d_i,
\qquad
G_{U,i}^{(p)}=-\bar p_i d_i^T.
\]

Reverse the diagonal:

\[
\bar y_i=\sigma_i^{-1}\odot\bar p_i,
\]

\[
\bar\sigma_i^{(p)}
=-\bar p_i\odot y_i\odot\sigma_i^{-2}.
\]

Reverse the lower solve:

\[
\bar k_i^{(p)}=L_i^{-T}\bar y_i,
\qquad
G_{L,i}^{(p)}=-\bar k_i^{(p)}y_i^T.
\]

For either dual action

\[
r_x=U_i^Ts_x,
\qquad s_x=\sigma_i\odot\ell_x,
\qquad \ell_x=L_i^Tx,
\]

the reverse is

\[
\bar s_x=U_i\bar r_x,
\qquad
G_{U,i}^{(x)}=s_x\bar r_x^T,
\]

\[
\bar\ell_x=\sigma_i\odot\bar s_x,
\qquad
\bar\sigma_i^{(x)}=\bar s_x\odot\ell_x,
\]

\[
\bar x=L_i\bar\ell_x,
\qquad
G_{L,i}^{(x)}=x\bar\ell_x^T.
\]

For `K=1`, the complete factor descriptors are therefore

\[
\boxed{
G_{L,i}
=-\bar k_i^{(p)}y_i^T
+(b_i\odot k_i)\bar\ell_{e,i}^T
+q_i\bar\ell_{q,i}^T,
}
\]

\[
\boxed{
G_{U,i}
=-\bar p_i d_i^T
+s_{e,i}\bar e_i^T
+s_{q,i}\bar\chi_i^T.
}
\]

The diagonal cotangent is

\[
\bar\sigma_i
=\bar\sigma_i^{(p)}
+\bar\sigma_i^{(e)}
+\bar\sigma_i^{(q)}.
\]

Only

\[
\bar N_i^-=\operatorname{tril}(G_{L,i},-1),
\qquad
\bar N_i^+=\operatorname{triu}(G_{U,i},1)
\]

enters the chart. These are rank-three masked descriptors; they are never
stored as dense tokenwise matrices.

The erase input reverse is

\[
\bar k_i
=\bar k_i^{(p)}+b_i\odot\bar x_{e,i},
\qquad
\bar b_i=k_i\odot\bar x_{e,i},
\]

and the query reverse is `bar q_i=bar x_q,i`.

The transpose semiseparable primitive is the exact adjoint of the forward
primitive. For

\[
p_j=\sum_{\ell<j}b_\ell x_\ell,\qquad y_j=a_jp_j,
\]

it computes

\[
\bar a_j\mathrel{+}=\bar y_jp_j,
\qquad
\bar p_j=a_j\bar y_j,
\]

\[
\bar b_\ell\mathrel{+}=x_\ell\sum_{j>\ell}\bar p_j,
\qquad
\bar x_\ell\mathrel{+}=b_\ell\sum_{j>\ell}\bar p_j.
\]

The upper case exchanges prefix and suffix. Forward block actions and backward
transpose block actions therefore have the same asymptotic work.

## 13. Chart and radial reverse

For

\[
Y=\mathcal B_c(X)=\frac{cX}{s},\qquad
s=\sqrt{c^2+\|X\|_F^2},
\]

the exact VJP is

\[
\boxed{
\bar X
=\frac{c}{s}\bar Y
-\frac{c}{s^3}X\langle X,\bar Y\rangle.
}
\]

Because `N_i^-` and `N_i^+` are sums of their H and R contributions, their
cotangents enter both radial branches:

\[
\bar N_{H,i}^-=\bar N_{R,i}^-=\bar N_i^-,
\qquad
\bar N_{H,i}^+=\bar N_{R,i}^+=\bar N_i^+.
\]

Define the four strict chart inputs

\[
A_{H,i}^-=\operatorname{tril}(X_i^{(H)},-1),
\qquad
A_{H,i}^+=\operatorname{triu}(X_i^{(H)},1),
\]

\[
A_{R,i}^-=\operatorname{tril}(X_i^{(R)},-1),
\qquad
A_{R,i}^+=\operatorname{triu}(X_i^{(R)},1).
\]

With `R_c(A,bar N)` denoting the displayed radial VJP, reverse all four maps
independently:

\[
\bar A_{H,i}^-=\mathcal R_{c_H}(A_{H,i}^-,\bar N_{H,i}^-),
\qquad
\bar A_{H,i}^+=\mathcal R_{c_H}(A_{H,i}^+,\bar N_{H,i}^+),
\]

\[
\bar A_{R,i}^-=\mathcal R_{c_R}(A_{R,i}^-,\bar N_{R,i}^-),
\qquad
\bar A_{R,i}^+=\mathcal R_{c_R}(A_{R,i}^+,\bar N_{R,i}^+).
\]

The strict ambient cotangents are

\[
\boxed{
\bar X_{H,i}^{\rm strict}
=\operatorname{tril}(\bar A_{H,i}^-,-1)
+\operatorname{triu}(\bar A_{H,i}^+,1),
}
\]

\[
\boxed{
\bar X_{R,i}^{\rm strict}
=\operatorname{tril}(\bar A_{R,i}^-,-1)
+\operatorname{triu}(\bar A_{R,i}^+,1).
}
\]

Even when `J_i` is symmetric at a reached state, the FP64 oracle exposes a
full ambient `J_0`. A symmetric-value forward specialization may share radial
norm metadata and transpose factor values, but reverse must preserve the two
strict H cotangent RHS above. It must not combine them into a lower-only
representative or copy one triangle's cotangent into the other.

For the diagonal,

\[
\bar\delta_i=\sigma_i\odot\bar\sigma_i,
\]

\[
\operatorname{diag}\bar X_i^{(H)}
=\operatorname{sech}^2\!\left(
\frac{\operatorname{diag}X_i^{(H)}}{s_H}
\right)\odot\bar\delta_i,
\]

\[
\operatorname{diag}\bar X_i^{(R)}
=\operatorname{sech}^2\!\left(
\frac{\operatorname{diag}X_i^{(R)}}{s_R}
\right)\odot\bar\delta_i.
\]

Combine each strict cotangent with its corresponding diagonal cotangent into
full ambient `bar X_H` and `bar X_R`. Then

\[
\bar H_i=\gamma_g\bar X_i^{(H)},
\qquad
\bar R_i=\gamma_g\bar X_i^{(R)},
\]

\[
\bar\gamma_g\mathrel{+}=
\langle\bar X_i^{(H)},H_i-I/r\rangle
+\langle\bar X_i^{(R)},R_i\rangle.
\]

After summing tokens and batches for the shared head parameter,

\[
\boxed{
\bar{\widehat\gamma}_g
=\bar\gamma_g\,\gamma_g(1-\gamma_g).
}
\]

`bar H_i` and `bar R_i` remain implicit combinations of masked rank
descriptors and radial-state terms. A `C x r x r` chart cotangent is forbidden.
The geometry transpose nevertheless returns the full ambient boundary
cotangent

\[
\bar J_0=\sum_i\eta_i\bar H_i,
\]

using exactly the `J0` tangent/cotangent convention of `reference.py` and the
full-initial-state VJP tests in `VALIDATION_PLAN.md`.

## 14. Chunk-local geometry transpose

Given all implicit `bar H_i` and `bar R_i`, the boundary cotangents are

\[
\boxed{
\bar J_0=\sum_i\eta_i\bar H_i,
\qquad
\bar D_0=\sum_i\eta_i\bar R_i.
}
\]

For each local source `s`,

\[
\boxed{
\bar u_s
=\sum_{i\ge s}\omega_{is}
[(\bar H_i+\bar H_i^T)u_s+\bar R_i h_s],
}
\]

\[
\boxed{
\bar h_s
=\sum_{i\ge s}\omega_{is}\bar R_i^Tu_s.
}
\]

The scalar weight cotangents are

\[
\bar\eta_i
=\langle\bar H_i,J_0\rangle
+\langle\bar R_i,D_0\rangle,
\]

\[
\bar\omega_{is}
=\langle\bar H_i,u_su_s^T\rangle
+\langle\bar R_i,u_sh_s^T\rangle.
\]

Reverse `eta_i=a_i/m_i` and `omega_is=w_is/m_i`:

\[
\bar a_i\mathrel{+}=\bar\eta_i/m_i,
\qquad
\bar w_{is}\mathrel{+}=\bar\omega_{is}/m_i,
\]

\[
\bar m_i\mathrel{+}=
-\frac{\bar\eta_i a_i+\sum_{s\le i}\bar\omega_{is}w_{is}}{m_i^2}.
\]

Reverse

\[
m_i=a_im_0+\sum_{s\le i}w_{is}:
\]

\[
\bar a_i\mathrel{+}=\bar m_i m_0,
\qquad
\bar w_{is}\mathrel{+}=\bar m_i,
\qquad
\bar m_0\mathrel{+}=\bar m_i a_i.
\]

Never divide by a potentially underflowed `lambda`. Since

\[
a_i=\exp\!\left(\sum_{p\le i}g_p^{(g)}\right),
\qquad
w_{is}=\exp\!\left(\sum_{s<p\le i}g_p^{(g)}\right),
\]

the exact local log-decay cotangent is

\[
\boxed{
\bar g_p^{(g)}
=\sum_{i\ge p}\bar a_i a_i
+\sum_{\substack{s<p\\i\ge p}}\bar w_{is}w_{is}.
}
\]

All displayed contractions with implicit chart cotangents are evaluated by the
same boundary broad actions and local semiseparable actions as forward. This
gives the frame and geometry reverse complexity `O(Cr^2+C^2r)` rather than a
coordinate-tile replay.

The boundary scan adjoint adds any loss on final `(m,J,D)`, consumes the local
boundary cotangents above, and propagates them through the affine monoid in
reverse chunk order. It also adds its contribution to every
`bar g_p^(g)`, `bar u_p`, and `bar h_p`.

The production reverse owns exactly one FP32 `bar u/bar h/bar g^(g)` panel and
one full-ambient FP32 `bar J_0/bar D_0` boundary pair. The strict transpose
action initializes these buffers; radial and diagonal reverse add to the same
storage in a fixed deterministic order before the boundary-scan adjoint
consumes it. Separate strict/radial vector panels followed by a combine kernel,
or an extra boundary cotangent that is merged only in Python, are forbidden.

For completeness, one chunk boundary map is

\[
m_1=am_0+p,\qquad J_1=aJ_0+Q,\qquad D_1=aD_0+V.
\]

Its exact reverse is

\[
\bar m_0\mathrel{+}=a\bar m_1,
\quad
\bar J_0\mathrel{+}=a\bar J_1,
\quad
\bar D_0\mathrel{+}=a\bar D_1,
\]

\[
\bar a\mathrel{+}=m_0\bar m_1
+\langle\bar J_1,J_0\rangle
+\langle\bar D_1,D_0\rangle,
\]

\[
\bar p\mathrel{+}=\bar m_1,
\qquad
\bar Q\mathrel{+}=\bar J_1,
\qquad
\bar V\mathrel{+}=\bar D_1.
\]

Equivalently, a tokenwise statement of the same affine adjoint, useful as the
scan oracle, reverses

\[
m_i=\lambda_i m_{i-1}+1,
\quad J_i=\lambda_iJ_{i-1}+u_iu_i^T,
\quad D_i=\lambda_iD_{i-1}+u_ih_i^T
\]

as

\[
\bar m_{i-1}\mathrel{+}=\lambda_i\bar m_i,
\quad
\bar J_{i-1}\mathrel{+}=\lambda_i\bar J_i,
\quad
\bar D_{i-1}\mathrel{+}=\lambda_i\bar D_i,
\]

\[
\bar\lambda_i\mathrel{+}=
m_{i-1}\bar m_i
+\langle\bar J_i,J_{i-1}\rangle
+\langle\bar D_i,D_{i-1}\rangle,
\]

\[
\bar u_i\mathrel{+}=(\bar J_i+\bar J_i^T)u_i+\bar D_i h_i,
\qquad
\bar h_i\mathrel{+}=\bar D_i^Tu_i,
\]

\[
\bar g_i^{(g)}\mathrel{+}=\lambda_i\bar\lambda_i.
\]

The Triton adjoint may parallelize this recurrence through the transpose of
the affine scan, but it must return exactly these cotangents.

## 15. Gate, normalization, convolution, and projection reverse

For a gate `c=2 sigmoid(a)`,

\[
\bar a=2\bar c\,\operatorname{sigmoid}(a)
[1-\operatorname{sigmoid}(a)].
\]

For either log-decay parameterization

\[
g=-\exp(A)\operatorname{softplus}(x),
\]

the reverse is

\[
\bar x=-\bar g\exp(A)\operatorname{sigmoid}(x),
\qquad
\bar A\mathrel{+}=\bar g\,g.
\]

The bias cotangent is the reduction of `bar x` over tokens and batches. These
reductions remain FP32.

For normalization `y=x/max(||x||,epsilon)`, if `n=||x||>epsilon`,

\[
\boxed{
\bar x=\frac{\bar y}{n}-\frac{x(x^T\bar y)}{n^3}.
}
\]

If `n<epsilon`,

\[
\bar x=\bar y/\epsilon.
\]

The implementation uses the declared deterministic branch at `n=epsilon`.

For `s=SiLU(c)=c sigmoid(c)`,

\[
\frac{\partial s}{\partial c}
=\operatorname{sigmoid}(c)
+c\operatorname{sigmoid}(c)[1-\operatorname{sigmoid}(c)].
\]

Let `bar c_t=bar s_t * ds/dc_t`. The depthwise conv4 reverse is

\[
\bar x_s\mathrel{+}=
\sum_{\substack{t\\0\le s-t+3\le3}}
\theta_{s-t+3}\bar c_t,
\]

\[
\bar\theta_j\mathrel{+}=\sum_t x_{t-3+j}\bar c_t.
\]

Returned final-cache cotangents are added to the corresponding four raw input
positions or initial-cache positions. Invalid tokens do not shift the cache;
valid resets cut all earlier cache dependence.

Finally, concatenate all projected cotangents as `bar p_t`. The linear
projection reverse is

\[
\bar x_t\mathrel{+}=W_{\rm in}^T\bar p_t,
\qquad
\bar W_{\rm in}\mathrel{+}=\sum_t\bar p_tx_t^T,
\qquad
\bar b_{\rm in}\mathrel{+}=\sum_t\bar p_t.
\]

For `c_t=concat_h(o_t)` and
`y_t^(layer)=W_out c_t+b_out`, the output-projection VJP is

\[
\bar c_t=W_{\rm out}^T\bar y_t^{\rm layer},
\qquad
\bar W_{\rm out}\mathrel{+}=\sum_t\bar y_t^{\rm layer}c_t^T,
\qquad
\bar b_{\rm out}\mathrel{+}=\sum_t\bar y_t^{\rm layer}.
\]

Split `bar c_t` by head to obtain the operator-output cotangents used in
Section 9.

## 16. Precision contract

The native path uses the following fixed precision map.

| Quantity | Native representation and arithmetic |
|---|---|
| projected activations, conv inputs/outputs, normalized `u/q/k`, values, erase/write operands | BF16, rounded once at the declared boundary |
| continuation and chunk-boundary `m/J/D/S` | FP32; never rounded between chunks or recurrent calls |
| geometry and associative log-decay; `exp`, `softplus`, `sigmoid` | FP32 |
| normalization norms; radial norms/scales; diagonal `tanh/exp`; sensitive inner products and scalar reverse | FP32 with deterministic reductions |
| generated LDU factor coordinates | compute in FP32, pack once to BF16 when entering an MMA action |
| primal and dual action operands | the same packed BF16 factor bits and BF16 RHS |
| GEMM, dot, block update, action, and backward partials | FP32 accumulation; reduced-precision accumulation is forbidden |
| `T`, `A_qd`, `W`, and the C32 unit-triangular solve | FP32 |
| `D_tail/Y/U_z/Q_gamma` storage | BF16 only after FP32 accumulation and validation; any panel-type upgrade is static for the complete hardware specialization |
| public activation and activation-gradient outputs | BF16 after the complete FP32 reduction |
| state and scalar cotangents | FP32 |

Tensor Core operand conversion is frozen per named contraction and hardware
specialization, not selected from runtime values. The initial implementation
uses direct BF16 packing and one MMA wherever the table below says `direct`.
Only a contraction that fails the unchanged oracle, cancellation, or VJP gates
is promoted. Once promoted, that contraction uses the promoted schedule for
every legal input of the specialization.

| Contraction | Initial static schedule |
|---|---|
| strict `J0` boundary action and transpose | direct BF16 packing, one MMA, FP32 accumulation |
| strict `D0` boundary action and transpose | direct BF16 packing, one MMA, FP32 accumulation; first high/low candidate if the frozen cancellation class fails |
| broad `Y S0`, `Q_gamma S0`, `D_tail^T R_z`, and their transpose products | direct BF16 packing of the FP32 `S0` or `R_z`, one MMA, FP32 accumulation |
| `Gamma_C S0` | FP32 elementwise multiplication; no Tensor Core packing |
| `A_qd R_z` and its transpose contributions | dedicated FP32 C32 kernel |

For a promoted contraction with one FP32 operand `X` and one already-BF16
operand, use the fixed twofold representation

\[
X_{\rm hi}=\operatorname{RN}_{\rm BF16}(X),
\qquad
X_{\rm lo}=\operatorname{RN}_{\rm BF16}(X-X_{\rm hi}),
\]

and two BF16 MMA products accumulated into the same FP32 result. The pair is
one representation of one FP32 state, not two parameters, and its VJP returns
one FP32 boundary cotangent. Persistent `J/D/S` and returned state remain FP32;
temporary direct or high/low packing is not a BF16 state round trip.

Data-dependent precision switching, threshold-selected fallback, iterative
correction semantics, and silently applying direct BF16 packing after that
contraction has failed a frozen gate are forbidden. High/low is therefore a
targeted static repair, not an unconditional two-MMA tax on every FP32 state
product.

The stable WY implementation computes `Delta_ij=exp(G_i-G_j)` directly or via
the equivalent FLA log2/exp2 schedule. It never computes `exp(-G_j)` by itself.
Log-decays `-110` and `-1000` must remain finite and semantically correct even
when an individual ordinary exponential underflows.

## 17. Complexity and workspace

Per chunk and head at fixed `K=1`, the intended work is

\[
\boxed{
O(Cr^2)\ \text{dense boundary action}
+O(C^2r)\ \text{local semiseparable/chart/WY interaction}
+O(C^3)\ \text{small WY solve}.
}
\]

For `r=128,C=32`, the first term is served by `128 x 128` times
`128 x 32` MMA panels. The local and pairwise terms use `C x C` interaction
tiles with r-wide reductions. There is no `O(Cr^3)` factorization and no
`O(Cr^2)` token-state workspace.

Report at least:

- persistent `(m,J,D,S)` and convolution-cache bytes;
- cross-kernel factor bytes;
- private backward-cache bytes;
- peak forward and forward-plus-backward allocation;
- bytes written and read for each retained panel;
- launch count and kernel attribution.

Absence from a public ABI does not make a private training cache free. The
selected cache schedule must beat recomputation end to end, including its HBM
traffic.

## 18. Numerical acceptance

The comparison point is the continuous FP64 oracle evaluated at the exact
runtime inputs. Generate deterministic master tensors, round each BF16 operand
once, promote those bits to FP64, and run `causallsso/reference.py`. FP32 gate,
decay, and state values are promoted without another BF16 rounding. Apply the
same rule to upstream cotangents.

For reference `x` and optimized `x_hat`, report

\[
\rho(x,\widehat x)
=\frac{\operatorname{RMS}(\widehat x-x)}
{\operatorname{RMS}(x)+10^{-8}},
\qquad
a_\infty(x,\widehat x)=\|\widehat x-x\|_\infty.
\]

An FP32 internal tensor passes when `a_inf <= 1e-6` or its declared relative
ceiling passes. A BF16 public or packed tensor uses `a_inf <= 2e-4`. The
absolute branch is not added to the relative allowance. NaN or infinity is an
unconditional failure.

For one contraction with identical packed operands,

\[
\tau(A,B,\widehat C)
=\frac{\|\widehat C-AB\|_F}
{\|A\|_F\|B\|_F+10^{-12}}.
\]

For a triangular solve `AX=B`,

\[
\eta(A,X,B)
=\frac{\|AX-B\|_F}
{\|A\|_2\|X\|_F+\|B\|_F+10^{-12}}.
\]

For every primal/dual pair,

\[
\pi
=\frac{|e^Td-(b\odot k)^Tk|}
{\|e\|_2\|d\|_2+\|b\odot k\|_2\|k\|_2+10^{-12}}.
\]

Pure FP64 algebraic equivalence uses `rtol=1e-10, atol=1e-12`. The native path
must pass these internal ceilings before end-to-end acceptance:

| Boundary | Forward ceiling | Backward ceiling | Additional gate |
|---|---:|---:|---|
| one BF16-operand/FP32-accumulated resident tile | `tau <= 2e-4` | `tau <= 2e-4` | measure before BF16 output storage |
| deterministic FP32 long triangular reduction | `tau <= 5e-4` | `tau <= 5e-4` | no data-dependent order |
| chunk-boundary `m/J/D/H/R` | `rho <= 5e-3` | `rho <= 5e-3` | initial-state VJP `rho <= 5e-4` |
| radial scalars and pre-pack chart from the same FP32 moments | `rho <= 1e-3` | `rho <= 3e-3` | reconstructed norm nonnegative |
| BF16 packed factors against the exact chart | `rho <= 6e-3` | `rho <= 1e-2` | strict masks, unit diagonal, identity exact |
| FP32-accumulated primal/dual action using the same packed bits | `rho <= 5e-3` | `rho <= 1e-2` | every triangular `eta <= 2e-5` |
| composed frame factors at the quantized point | `rho <= 6e-3` | end-to-end VJP class | pre-cast `pi <= 5e-4`, stored-BF16 `pi <= 8e-3` |

Stable WY has no newly invented tolerance row. Each `T/A_qd` interaction tile
must pass the existing `tau <= 2e-4` contraction gate, each C32 forward and
transpose solve must pass `eta <= 2e-5`, and the composed
`Y/U_z/D_tail/Q_gamma`, output, state, and VJP use the existing end-to-end
classes below. The diagnostic `d/e/chi` rows in `VALIDATION_PLAN.md` remain
legal internal checkpoints even though those tensors are not a production
forward interface; a private saved-tensor cache remains governed by Section 8.

Direct-BF16, promoted high/low, 9 MiB cache, 13 MiB cache, and FP32-cache
variants all face these identical cancellation, state, pairing, and VJP gates.
A variant that fails a gate is not a performance candidate; tolerance changes
or a looser cache-specific reference are forbidden.

The complete native row is

| Advertised path | BF16 outputs and FP32 `S` | `q/k/v/S0/Cq0/Ck0/Cv0` gradients | `u/h/J0/D0` and gate/geometry gradients |
|---|---:|---:|---:|
| `r=128,K=1,C=32`, BF16 operands and FP32 accumulation/state/scalars | `rho <= 6e-3` | `rho <= 1.5e-2` | `rho <= 2.5e-2` |

Every returned geometry boundary and final state must also pass its stricter
internal row. End-to-end tolerance cannot hide a failed scan, chart, action,
or WY statistic.

## 19. Semantic acceptance

Acceptance requires all of the following:

1. token outputs, every chunk boundary, final `(m,J,D,S)`, and all input/state/
   parameter VJPs match the quantized-input FP64 oracle;
2. whole-sequence execution equals arbitrary recurrent splits;
3. invalid tokens hold all states and emit zero; reset and packed segments
   equal independent executions;
4. identity geometry gives exact GDN2 observables and gradients at finite
   shared parameters or the structural switch;
5. structural no-decay and ordered `K=1,2,4` references give the declared
   Delta/DeltaProduct reductions;
6. strict masks, unit diagonals, primal/dual pairing, chart bounds, and the
   declared condition-number bound hold;
7. zero keys give an exact identity edit; weak, saturated, repeated,
   orthogonal, asymmetric, and alternating-sign cases remain finite;
8. lengths `1,31,32,33,1024,8192`, irregular tails, `t<r`, `t=r`, and `t>r`
   pass without a sequence-length tolerance multiplier;
9. geometry and associative log-decays `-110` and `-1000` pass without an
   inverse-decay overflow;
10. legal J and D cancellation fixtures, including the frozen `2^12` class,
    pass the unchanged `VALIDATION_PLAN.md` radial and tying-map gates;
11. repeat runs are bitwise deterministic for fixed inputs and launch config;
12. unsupported widths, edit counts, dtypes, masks, resets, packed layouts, or
    architectures fail explicitly until their exact native path is accepted.

## 20. Performance acceptance

The primary operator profile is

```text
B=1, T=1024, H=8, r=d_v=128, K=1, C=32,
BF16 activation/matrix operands, FP32 accumulation and recurrent state.
```

Report forward, backward alone, forward plus backward, peak memory, and launch
attribution for:

- normalization;
- geometry boundary scan and adjoint;
- fused ChunkSolve--WY prepare and transpose;
- state scan and reverse;
- output and output VJP;
- the complete operator;
- the complete layer including projection and conv4;
- matched FLA GDN2 under the same loss and synchronization protocol.

Benchmark `C in {16,32,64}` only as a scheduling choice after each candidate
passes the same formulas and numerical gates. The first selected specialization
remains C32 unless measured complete-operator evidence selects another value.

For training, benchmark the roughly 13 MiB
`W/y/d/s_e/s_q/e/chi` cache first because it removes both upper-action
replays. Compare it directly with the roughly 9 MiB cache without `e/chi` and
with selected FP32-panel retention. Report complete forward plus backward time,
backward alone, peak allocation, and HBM traffic for each; choose by end-to-end
training latency after numerical acceptance, not by saved bytes or an isolated
kernel row.

The initial performance objective is to make backward a small constant multiple
of forward and to bring the complete operator close to matched GDN2. In
particular, acceptance of the rewrite requires the paired transpose path to
remove the old frame-backward launch/VJP chain and to reduce the target-profile
backward below 5 ms before secondary scalar tuning is considered. Performance
never overrides a failed semantic or numerical gate.

## 21. Provenance

The chunk WY derivation follows Yang et al., *Parallelizing Linear Transformers
with the Delta Rule over Sequence Length*, arXiv:2406.06484. GDN2 supplies the
matched asymmetric erase/write and mixed-precision baseline, arXiv:2605.22791.
Reusable state-scan and WY implementation patterns come from MIT-licensed FLA
v0.5.2. The official NVlabs GDN2 repository has a non-MIT license and is a
paper/baseline reference only; its implementation is not copied. Material
research decisions and source links remain recorded in `docs/PRIOR_ART.md`.

# SolveDelta From-Scratch Native Rebuild

Status: audited implementation blueprint, not an implementation.

This document deliberately assumes that the current native kernels and the
older execution documents are wrong. They are useful only as failed
experiments and benchmark evidence. They do not constrain names, layouts,
kernel boundaries, caches, or source reuse in the rebuild. The only semantic
authority is `causallsso/reference.py`.

The one hypothesis retained for revalidation is a chunked geometry execution
feeding an exact chunk-WY associative exterior. Section 4 proves that the WY
part is exact. Every other implementation choice below must earn its place by
an oracle check and an end-to-end benchmark.

This is not a compatibility plan. The old native path, Python glue, tests, and
obsolete execution documents may be deleted before construction begins;
version control is their recovery mechanism. Section 16 governs when the new
path may be accepted as production, not how long obsolete source must remain in
the working tree. No legacy ABI, backend selector, or fallback survives.

## 1. Conclusions first

The rebuild should not try to make one giant SolveDelta kernel. The correct
reuse unit is usually a complete, mature upstream kernel with one algebraic
specialization, not a device-level GEMM collective inserted into an unrelated
custom CTA.

The production split is:

1. PyTorch/cuBLASLt owns the two dense projections.
2. `causal-conv1d` owns the packed q/k/v conv4 and its full state VJP.
3. FLA owns normalization schedules, gate/cumsum schedules, the selected
   chunk-width pair/WY exterior, the FP32 associative state scan, output
   composition, and their transpose kernels.
4. A dtype-specialized MESA resident `Hkk/Hkv` program owns the paired FP32
   `(m,J,D)` geometry boundary scan and its resident transpose. The scalar
   mass route is owned by one matrix-tile CTA in that same resident program;
   it is not a separate sequence scan.
5. MESA's two-dot action and pair-reduction schedules, instantiated with CuTe
   warp-level atoms and CUB reductions, own the local structured work.
6. One thin SolveDelta-specific orchestration owns the bounded-LDU composition:
   it wires three matrix-free routes into blocked primal actions, paired direct
   dual actions, and their transpose order. Radial normalization, diagonal
   soft-capping, matrix products, reductions, and scans remain standard
   primitives with upstream owners.

The internal frame/exterior boundary is selected by the fusion A/B in Section
12.4. Private `d/e/chi` HBM panels are the reference split candidate, not a
required ABI; at the C32 reference shape all three FP16 panels total 6 MiB.

Chunk width and launch shape are implementation parameters. The first target
must compare `C in {16,32,64}` and mature `num_warps in {4,8}` schedules,
together with their supported stage/tile choices. Every candidate first passes
the same oracle and VJP gates; warmed complete forward and F+B then select one
static winner for the target architecture/profile. There is no runtime
autotune, data-dependent selection, or semantic distinction between these
chunkings.
That traffic is cheaper than extending chart factors, pair statistics, WY
state, and output accumulators through one low-occupancy mega-kernel. Their
cotangents are also valid backward kernel boundaries. The forbidden objects
are synthetic descriptor bundles and per-entry VJP workspaces, not useful
vector panels.

The claim that less than one percent of the source must be original is not a
sound engineering metric. The bounded-LDU orchestration is operator-specific
and will be more than one percent of source lines. The realistic objective is
that essentially no low-level mechanism is invented: global/shared copies,
swizzles, MMA atoms, reductions, scans, convolutions, chunk solves, state
passing, and their reverse schedules all come from mature implementations.
Original code is restricted to composing those mechanisms around the chart
formulas in Sections 2, 7, and 9; it does not redefine their scalar norm or
activation primitives.

"Hardware-exact" in this document means that every load/store boundary,
precision conversion, collective owner, synchronization point, and arithmetic
instruction family is frozen. Literal SASS instruction counts are compiler
artifacts, not mathematical promises. They are accepted only from the compiled
cubin as required by Section 12; a source-level estimate cannot be used to
excuse an extra barrier, spill, conversion, or global temporary.

## 2. Mathematical source of truth

All vectors below are columns. A panel stores one vector per row. The main
derivation uses `K=1`; Section 5 gives the exact general-`K` mapping.

For token `t`, geometry is updated before the frame is constructed:

\[
\lambda_t=\exp(a_t),\qquad a_t\le 0,
\]

\[
m_t=\lambda_t m_{t-1}+1,
\]

\[
J_t=\lambda_t J_{t-1}+u_tu_t^T,
\qquad
D_t=\lambda_t D_{t-1}+u_th_t^T.
\]

Production `J_0` is exactly symmetric. Therefore every reachable `J_t` is
symmetric. `D_t` is unconstrained. Define

\[
H_t=J_t/m_t,\qquad R_t=D_t/m_t,
\qquad \widetilde H_t=H_t-I/r.
\]

On the reachable state domain this normalized coordinate has an exact
matrix-valued delta-rule form. For `m_t>0`, set

\[
\alpha_t=\frac1{m_t}.
\]

Since `lambda_t*m_{t-1}/m_t=1-alpha_t`,

\[
\boxed{
H_t=H_{t-1}+\alpha_t(u_tu_t^T-H_{t-1}),}
\]

\[
\boxed{
R_t=R_{t-1}+\alpha_t(u_th_t^T-R_{t-1}).}
\]

At the zero initial state the first valid token has `m_1=1`; the equations use
the canonical base case `H_1=u_1u_1^T`, `R_1=u_1h_1^T` without dividing the
zero boundary. Thus H is a decayed online second moment and R is a decayed
online cross moment. This is an exact residual interpretation, not another
operator variant.

The same statement holds at a chunk merge. Let a reset-free chunk summarize
its decay, added mass, and two weighted observations as

\[
(\Lambda,b,C_J,C_D),
\]

so that

\[
m'=\Lambda m+b,\qquad J'=\Lambda mH+C_J,
\qquad D'=\Lambda mR+C_D.
\]

For `b>0`, define `H_c=C_J/b`, `R_c=C_D/b`, and

\[
\theta=\frac{b}{\Lambda m+b}.
\]

Then

\[
\boxed{H'=H+\theta(H_c-H),\qquad R'=R+\theta(R_c-R).}
\]

This identity explains the natural variables without changing the execution
state. `(H,R)` alone is not an associative summary because the merge also
needs mass. Production therefore keeps `(m,J,D)` as the FP32 scan coordinate,
where MESA/FLA already supplies the exact affine accumulator, and treats
`(H,R)` as the interpretation and action coordinate. Recurrently rounding
normalized states would define a different split-dependent floating
recurrence and is forbidden.

Let `P_-` and `P_+` be strict-lower and strict-upper projection. For
`c_H=c_R=s_H=s_R=1/8`, define

\[
x_{H,t}^-=\gamma P_-(\widetilde H_t),
\qquad x_{H,t}^+=(x_{H,t}^-)^T,
\]

\[
x_{R,t}^-=\gamma P_-(R_t),
\qquad x_{R,t}^+=\gamma P_+(R_t),
\]

and the radial map

\[
\mathcal B_c(x)=\frac{c x}{\sqrt{c^2+\lVert x\rVert_F^2}}.
\]

This map is exactly a scaled epsilon-L2 normalization. Flatten the selected
strict route, set

\[
r_c=(c^2+\lVert x\rVert_F^2)^{-1/2},\qquad
y=xr_c.
\]

Then

\[
\boxed{\mathcal B_c(x)=c\,\operatorname{L2Norm}_{\epsilon=c^2}(x)=c y.}
\]

Given `bar Y` for `Y=mathcal B_c(x)`, its transpose is

\[
\boxed{
\bar x=c r_c\bar Y-c r_c^3x\langle\bar Y,x\rangle_F.}
\]

FLA `l2norm_bwd` writes the same expression as
`dy*rstd - <dy,y>*y*rstd`, with `dy=c*bar Y`, `y=r_c*x`, and
`rstd=r_c`. This is a literal primitive equivalence, not an analogy.
Production does not materialize the flattened route merely to call the public
wrapper: MESA Gram/Hadamard supplies `||x||_F^2`, while the FLA rsqrt, scale,
inner-product, and transpose epilogue are specialized at their existing
matrix-free consumer.

The factors are

\[
L_t=I+\mathcal B_{c_H}(x_{H,t}^-)
       +\mathcal B_{c_R}(x_{R,t}^-),
\]

\[
U_t=I+\mathcal B_{c_H}(x_{H,t}^+)
       +\mathcal B_{c_R}(x_{R,t}^+),
\]

\[
\ell_t=s_H\tanh\!\left(
  \frac{\gamma\,\operatorname{diag}(\widetilde H_t)}{s_H}
\right)
+s_R\tanh\!\left(
  \frac{\gamma\,\operatorname{diag}(R_t)}{s_R}
\right),
\]

\[
\sigma_t=\exp(\ell_t),\qquad
M_t=L_t\operatorname{diag}(\sigma_t)U_t.
\]

Likewise, with `phi_s(x)=s*tanh(x/s)`, the diagonal chart is only

\[
\ell=\phi_{s_H}(x_H)+\phi_{s_R}(x_R),\qquad \sigma=\exp(\ell),
\]

and

\[
\boxed{\phi_s'(x)=1-\tanh^2(x/s).}
\]

It reuses the same FP32 tanh soft-cap forward/reverse primitive already used
by FLA fused cross entropy, followed by the ordinary exp epilogue. Only its
placement inside the bounded-LDU composition is SolveDelta-specific.

The LDU factors themselves are residual operators:

\[
L=I+N^-,\qquad U=I+N^+.
\]

Their direct actions are ordinary residual matrix actions, for example
`L^T x=x+(N^-)^T x`. Their inverse actions are the standard unit-triangular
primitive

\[
(I+N)y=b,
\]

or, algebraically because a strict triangular `N` is nilpotent,

\[
(I+N)^{-1}=\sum_{k=0}^{r-1}(-N)^k.
\]

The production inverse is an exact coordinate-axis generalized-Delta solve.
For one chunk-frame row `i`, let `b_i^H,b_i^R` denote its boundary
coefficients and `omega_i^H,omega_i^R in R^C` its local coefficients. Each
strict-factor entry before masking is

\[
N_{i,pq}=b_i^H J_{0,pq}+b_i^R D_{0,pq}
+\sum_s\omega^H_{is}u_{s,p}u_{s,q}
+\sum_s\omega^R_{is}u_{s,p}h_{s,q}.
\]

Introduce notation-only feature rows

\[
\mathcal Q_{i,p}=\left[
b_i^H J_{0,p,:}+b_i^R D_{0,p,:},\;
\omega_i^H\odot u_{:,p},\;
\omega_i^R\odot u_{:,p}
\right],
\]

\[
\mathcal K_q=\left[e_q,\;u_{:,q},\;h_{:,q}\right],
\]

where `e_q` is the `q`th coordinate basis vector. Then, exactly,

\[
\boxed{N_{i,pq}=\mathcal Q_{i,p}^T\mathcal K_q.}
\]

For a lower factor, `(I+N_i)y_i=b_i` is therefore

\[
s_{i,p-1}=\sum_{q<p}\mathcal K_qy_{i,q},\qquad
y_{i,p}=b_{i,p}-\mathcal Q_{i,p}^Ts_{i,p-1},
\]

\[
s_{i,p}=s_{i,p-1}+\mathcal K_py_{i,p}.
\]

This is the generalized-Delta recurrence along the coordinate axis. An upper
factor is the same recurrence in reversed coordinate order. Production ports
FLA GDN2/DPLR's blocked substitution: 16-coordinate diagonal blocks use its
exact ordered solve, while complete off-diagonal blocks use Tensor-Core pair
dots. Boundary and the two local pair contributions are generated directly at
the consuming block; the formal `r+2C` features, a dense factor, its inverse,
and iterative panels never reach HBM.

The reverse is the corresponding exact transpose recurrence. For
`y=(I+N)^{-1}b`,

\[
z=(I+N)^{-T}\bar y,\qquad \bar b=z,\qquad \bar N=-zy^T.
\]

FLA's pair-dot transpose schedule consumes each strict `bar N` block directly
into boundary and local operands. It does not materialize `bar N` or replay a
coordinate descriptor chain. MathDx remains an independent exact TRSM oracle.
The only native error relative to the FP64 solve is the declared operand
rounding and FP32 reduction order under the BF16-observable contract; there is
no authorized solve approximation.

With normalized key `k_t`, erase covector source
`b_t=erase_t\odot k_t`, normalized query `q_t`, and write target
`z_t=write_t\odot v_t`, the frame actions are

\[
d_t=M_t^{-1}k_t,
\qquad e_t=M_t^Tb_t,
\qquad \chi_t=M_t^Tq_t.
\]

The associative state remains in the ambient basis:

\[
S_t^-=\operatorname{diag}(\exp g_t)S_{t-1},
\]

\[
r_t=z_t-e_t^TS_t^-,
\qquad S_t=S_t^-+d_tr_t^T,
\qquad o_t=\chi_t^TS_t.
\]

These equations, masks/resets, and returned states must remain exactly those
of `reference.py`. An implementation layout is never allowed to redefine
them.

The FP64 mathematical operator receives activated `erase/write` gates. The
layer owns

\[
erase=2\sigma(erase_{raw}),\qquad
write=2\sigma(write_{raw}),
\]

\[
a=-\exp(\rho_g)\operatorname{softplus}(a_{raw}+b_g),
\qquad
g=-\exp(\rho_s)\operatorname{softplus}(g_{raw}+b_s),
\]

and `gamma=sigmoid(gamma_raw)` in FP32. Their reverse uses the matching fused
FLA gate schedule and reduces static parameter gradients in FP32. In the
native production graph, BF16 `erase_raw/write_raw` cross the model-to-kernel
boundary and their activations are composed into their unique consumer
epilogues: erase in the normalized dual-source store and write in the
FLA-native `z=write*v` pack. No activated full-tensor gate is written to HBM.
The same epilogues apply the sigmoid transpose from FP32 register values, so
this changes only the private boundary and not the layer-owned expression or
its composed VJP. For
`y=x/max(||x||_2,epsilon)`, the normalization transpose is

\[
\bar x=
\begin{cases}
(\bar y-y(y^T\bar y))/\lVert x\rVert_2,&\lVert x\rVert_2>\epsilon,\\
\bar y/\epsilon,&\lVert x\rVert_2<\epsilon,
\end{cases}
\]

with the equality convention matched bit-for-bit to the reference primitive.
This is one fused three-panel FLA/CUB reduction, not three PyTorch graphs.

## 3. Exact chunk geometry

For a geometry chunk beginning at boundary `(m_0,J_0,D_0)`, number local
tokens `i=0,...,C-1` and define

\[
G_i^{(g)}=\sum_{p=0}^{i} a_p,
\qquad \Lambda_i=\exp(G_i^{(g)}),
\qquad w_{ij}=\mathbf1_{j\le i}\exp(G_i^{(g)}-G_j^{(g)}).
\]

Then the exact closed form is

\[
m_i=\Lambda_i m_0+\sum_{j\le i}w_{ij},
\]

\[
J_i=\Lambda_iJ_0+\sum_{j\le i}w_{ij}u_ju_j^T,
\]

\[
D_i=\Lambda_iD_0+\sum_{j\le i}w_{ij}u_jh_j^T.
\]

The chunk continuation is the same expression at `i=C-1`. This gives an
affine chunk state transition; it does not make the token frame independent
across chunks. The boundary remains FP32 and is read by every frame panel.

This boundary computation is not a new SolveDelta kernel design. In MESA's
`chunk_mesa_net_fwd_kernel_h`, set

\[
k=u,\qquad k_2=u,\qquad v=h,\qquad \beta=1,
\]

map `h_kk -> J` and `h_kv -> D`, use the selected `C`, and set
`states_in_fp32=True`. MESA's resident update then is exactly

\[
J_C=e^{G_C}J_0+\sum_j e^{G_C-G_j}u_ju_j^T,
\qquad
D_C=e^{G_C}D_0+\sum_j e^{G_C-G_j}u_jh_j^T.
\]

The adopted specialization retains MESA's loads, log2/exp2 gauge, resident
FP32 accumulators, dots, stores, and grid. It only separates the operand dtype:
the J dot consumes direct-FP16 normalized `u`, while the D dot consumes BF16
`u` and unbounded BF16 `h`. This avoids MESA's model-specific cast of `v` to
the key dtype without splitting the paired resident loop. The normalization
producer writes both declared `u` consumer bit patterns; neither is obtained
by casting an already rounded public tensor to pretend it has more precision.

Those formulas describe one all-valid segment. Production masks and resets are
not an afterthought. Define

\[
\alpha_i=
\begin{cases}
1,&valid_i=0,\\
0,&valid_i=1\ \text{and}\ reset_i=1,\\
\lambda_i,&valid_i=1\ \text{and}\ reset_i=0,
\end{cases}
\qquad \beta_i=\mathbf1_{valid_i}.
\]

Then the actual geometry recurrence is

\[
m_i=\alpha_i m_{i-1}+\beta_i,
\quad J_i=\alpha_iJ_{i-1}+\beta_i u_iu_i^T,
\quad D_i=\alpha_iD_{i-1}+\beta_i u_ih_i^T.
\]

For any `i>=j`, replace the log-only weights by

\[
\Lambda_i=\prod_{p=0}^i\alpha_p,
\qquad
w_{ij}=\beta_j\prod_{p=j+1}^i\alpha_p.
\]

This is an ordinary affine monoid scan and exactly represents reset by a zero
multiplier and invalid input by the identity element. On a reset-free segment,
it reduces to the exponential formulas above. The implementation reuses FLA's
packed chunk-index schedule to start a logical segment at each reset without
copying panels; arbitrary invalid rows remain predicated identity rows. No
logarithm of the zero reset multiplier is ever taken.

For a strict route `P` and local outer products `O_j=A_jB_j^T`, write

\[
X_i=\Lambda_iX_0+\sum_{j\le i}w_{ij}O_j.
\]

The unnormalized strict moment norm is computed without materializing `X_i`:

\[
n_0=\lVert P(X_0)\rVert_F^2,
\]

\[
c_j=\langle P(X_0),P(O_j)\rangle_F,
\qquad
G_{jk}=\langle P(O_j),P(O_k)\rangle_F,
\]

\[
\boxed{
n_i=\Lambda_i^2n_0
  +2\Lambda_i\sum_jw_{ij}c_j
  +\sum_{j,k}w_{ij}G_{jk}w_{ik}.}
\]

There are exactly three production routes: one symmetric H strict route and
separate R lower and upper routes. For each route with radius `c`, the
coefficient multiplying unscaled moment entries is

\[
\boxed{
\kappa_i=\frac{c\gamma}
{\sqrt{c^2m_i^2+\gamma^2n_i}}.}
\]

Thus the strict factor is `kappa_i P(X_i)`. This avoids constructing
`X_i/m_i` before normalization.

For the symmetric H route,

\[
G^{H}_{jk}=\frac12\left[
  (u_j^Tu_k)^2
  -(u_j\odot u_j)^T(u_k\odot u_k)
\right].
\]

For R, lower and upper statistics remain distinct. Partition the coordinate
axis into ordered tiles `p`. Let `U_p,H_p in R^{C x b_p}` be the token panels
restricted to tile `p`, and define

\[
K^u_p=U_pU_p^T,\qquad K^h_p=H_pH_p^T,
\]

\[
K^{u,<p}=\sum_{q<p}K^u_q,\quad K^{u,>p}=\sum_{q>p}K^u_q,
\qquad
K^{h,<p}=\sum_{q<p}K^h_q,\quad K^{h,>p}=\sum_{q>p}K^h_q.
\]

For token `j`, the within-tile strict representatives are

\[
T^-_{p,j}=P_-^{(p)}(u_{j,p}h_{j,p}^T),\qquad
T^+_{p,j}=P_+^{(p)}(u_{j,p}h_{j,p}^T),
\]

with pair Grams

\[
G^{\pm}_{p,jk}=\langle T^{\pm}_{p,j},T^{\pm}_{p,k}\rangle_F.
\]

The complete strict-coordinate Grams are therefore

\[
\boxed{
G^- = \sum_p K^u_p\odot K^{h,<p}+\sum_pG^-_p,\qquad
G^+ = \sum_p K^u_p\odot K^{h,>p}+\sum_pG^+_p.}
\]

The first terms cover complete coordinate tiles below or above the diagonal;
`G_p^-` and `G_p^+` cover the corresponding strict part inside a diagonal
tile. This is the exact Gram/Hadamard form consumed by the radial norm. No
coordinate-pair workspace reaches HBM.

## 4. Exact associative chunk-WY algebra

Within one associative chunk define

\[
G_i=\sum_{p=0}^{i}g_p.
\]

The globally gauged panels below are notation only:

\[
\widetilde D_j=d_j\odot\exp(-G_j),
\qquad E_i=e_i\odot\exp(G_i),
\qquad Q_i=\chi_i\odot\exp(G_i).
\]

The implementation uses FLA's tile-local stable gauge and never materializes
`exp(-G_j)` globally. Define

\[
W_{ij}=\delta_{ij}+\mathbf1_{j<i}E_i^T\widetilde D_j,
\]

\[
A_{ij}=\mathbf1_{j\le i}Q_i^T\widetilde D_j,
\qquad
D^{tail}_j=d_j\odot\exp(G_{C-1}-G_j).
\]

Let `Z` contain `z_i` by row. The residual equations are

\[
Wr=Z-ES_0.
\]

Using two wide right-hand sides,

\[
Y=W^{-1}E,
\qquad U=W^{-1}Z,
\qquad R_z=U-YS_0.
\]

The exact output and continuation are

\[
\boxed{O=QS_0+AR_z,}
\]

\[
\boxed{S_C=\operatorname{diag}(\exp G_{C-1})S_0
       +(D^{tail})^TR_z.}
\]

Substitution into the token recurrence proves these identities. An independent
FP64 check produced output error `1.56e-13`, final-state error `6.04e-14`, and
leaf/state VJP errors near `1e-12`.

As in Section 3, these equations apply independently inside each valid
reset-free associative segment. A valid reset gives the incoming `S_0`
coefficient zero; an invalid token has `d=e=chi=z=0`, an identity WY row, zero
output, and an identity continuation. For `K>1`, reset is attached only to the
first micro-edit and invalidity predicates all micro-edits. FLA's variable-
length/chunk-index machinery supplies the segment grid; the algebra never
allows a pair interaction to cross a reset.

## 5. More than one edit

`K>1` does not require another associative algorithm. Flatten `n=tK+j` and
set

\[
g'_{t,0}=g_t,\qquad g'_{t,j}=0\quad(j>0).
\]

Construct `d/e/z` for every edit from the same token frame and place the read
after edit `K-1`. This is FLA's gated DeltaProduct recurrence. Geometry and
micro-edit chunk boundaries need not align because the frame panels are an
intentional private interface. The first specialization remains `K=1,r=128`;
`C` is selected from `{16,32,64}` by the complete-path procedure above.

## 6. Complete exterior reverse

Given `bar O` and `bar S_C`,

\[
\bar Q\mathrel{+}=\bar O S_0^T,
\quad \bar S_0\mathrel{+}=Q^T\bar O,
\quad \bar A\mathrel{+}=\bar O R_z^T,
\quad \bar R_z\mathrel{+}=A^T\bar O,
\]

\[
\bar S_0\mathrel{+}=\exp(G_{C-1})\odot\bar S_C,
\quad
\bar G_{C-1}\mathrel{+}=\sum_v
 \bar S_C[:,v]\odot\left[\exp(G_{C-1})\odot S_0[:,v]\right],
\quad
\overline{D^{tail}}\mathrel{+}=R_z\bar S_C^T,
\quad
\bar R_z\mathrel{+}=D^{tail}\bar S_C.
\]

For `R_z=U-YS_0`,

\[
\bar U\mathrel{+}=\bar R_z,
\quad \bar Y\mathrel{-}=\bar R_zS_0^T,
\quad \bar S_0\mathrel{-}=Y^T\bar R_z.
\]

Concatenate `B=[Z,E]`, `X=[U,Y]`, and `bar X=[bar U,bar Y]`. Since `WX=B`,

\[
\boxed{\bar B=W^{-T}\bar X,}
\qquad
\boxed{\bar W=-\bar B X^T.}
\]

Let `L_W=tril(bar W,-1)` and `L_A=tril(bar A,0)`. The pair transpose is

\[
\bar E\mathrel{+}=L_W\widetilde D,
\quad \bar Q\mathrel{+}=L_A\widetilde D,
\quad
\overline{\widetilde D}\mathrel{+}=L_W^TE+L_A^TQ.
\]

The two column blocks of `bar B` are `bar Z` and an additional `bar E`. In the
global-gauge notation, the exact remaining reverse is

\[
\bar e_i\mathrel{+}=\bar E_i\odot\exp(G_i),
\qquad
\bar G_i\mathrel{+}=\bar E_i\odot E_i,
\]

\[
\bar\chi_i\mathrel{+}=\bar Q_i\odot\exp(G_i),
\qquad
\bar G_i\mathrel{+}=\bar Q_i\odot Q_i,
\]

\[
\bar d_i\mathrel{+}=\overline{\widetilde D_i}\odot\exp(-G_i),
\qquad
\bar G_i\mathrel{-}=\overline{\widetilde D_i}
 \odot\widetilde D_i.
\]

These equations define the transpose. The production kernel evaluates their
tile-local stable-gauge equivalent and never forms `exp(-G_i)`. `Dtail` adds

\[
\bar d_j\mathrel{+}=\overline{D^{tail}_j}
 \odot\exp(G_{C-1}-G_j),
\]

\[
\bar G_{C-1}\mathrel{+}=
 \sum_j\overline{D^{tail}_j}\odot D^{tail}_j,
\qquad
\bar G_j\mathrel{-}=
 \overline{D^{tail}_j}\odot D^{tail}_j.
\]

A reverse chunk cumsum maps `bar G` to `bar g`. FLA's resident state reverse
carries `bar S` in FP32 across chunks. The stage boundary is exactly three
vector cotangent panels plus `bar z`, `bar g`, and `bar S_0`; there is no
descriptor bundle.

### 6.1 Frozen tile-local gauge transpose

The global panels above define the operator, but the production pair kernels
factor each token tile around a coordinate-wise reference
`rho_tau in R^r`. For row set `I_tau`, column set `J_tau`, and the already
applied causal mask, define

\[
\widehat E_i=e_i\odot\exp(G_i-\rho_\tau),\qquad
\widehat Q_i=\chi_i\odot\exp(G_i-\rho_\tau),
\quad i\in I_\tau,
\]

\[
\widehat D_j=d_j\odot\exp(\rho_\tau-G_j),
\quad j\in J_\tau.
\]

Then every retained pair is unchanged:

\[
\widehat E_i^T\widehat D_j
=E_i^T\widetilde D_j,
\qquad
\widehat Q_i^T\widehat D_j
=Q_i^T\widetilde D_j.
\]

For a fully off-diagonal causal tile, monotone nonpositive gates permit the
coordinate-wise choice

\[
\max_{i\in I_\tau}G_i\ \le\ \rho_\tau\ \le\
\min_{j\in J_\tau}G_j.
\]

Thus both exponent vectors are nonpositive. For a diagonal token tile, the
frozen FLA midpoint reference is used under the declared static gate bound.
This choice affects range, not the mathematical pair value.

Let `L_W^tau` and `L_A^tau` be the masked cotangent tiles of `W` and `A`.
The pair GEMM transpose first produces

\[
\overline{\widehat E}=L_W^\tau\widehat D,
\qquad
\overline{\widehat Q}=L_A^\tau\widehat D,
\]

\[
\overline{\widehat D}=(L_W^\tau)^T\widehat E
 +(L_A^\tau)^T\widehat Q.
\]

The local ungauging transpose is

\[
\bar e_i\mathrel{+}=\overline{\widehat E_i}
 \odot\exp(G_i-\rho_\tau),\qquad
\bar\chi_i\mathrel{+}=\overline{\widehat Q_i}
 \odot\exp(G_i-\rho_\tau),
\]

\[
\bar G_i\mathrel{+}=\overline{\widehat E_i}\odot\widehat E_i
 +\overline{\widehat Q_i}\odot\widehat Q_i,
\quad i\in I_\tau,
\]

\[
\bar d_j\mathrel{+}=\overline{\widehat D_j}
 \odot\exp(\rho_\tau-G_j),\qquad
\bar G_j\mathrel{-}=\overline{\widehat D_j}\odot\widehat D_j,
\quad j\in J_\tau.
\]

There is deliberately no `bar rho_tau` output. Componentwise, the formal
cotangent is

\[
\bar\rho_\tau=
-\sum_{i\in I_\tau}
 \left(\overline{\widehat E_i}\odot\widehat E_i
      +\overline{\widehat Q_i}\odot\widehat Q_i\right)
+\sum_{j\in J_\tau}
 \overline{\widehat D_j}\odot\widehat D_j=0.
\]

The equality follows by expanding both sums into the same masked pair terms,
once from the row operands and once from the column operand. Therefore ignoring
the derivative of the rule that selects `rho_tau` is exact gauge elimination,
not a stop-gradient approximation. Tiles that share a token accumulate both
its row and column contributions before the reverse cumsum

\[
\boxed{\bar g_p=\sum_{i\ge p}\bar G_i.}
\]

These equations use natural-log cumulative gates. An `exp2` implementation
stores `G/log(2)`; its `log(2)` exponential derivative and the cumsum's
`1/log(2)` scale cancel at the raw-`g` interface. Forward and reverse use the
same reference and packed operand bits.

## 7. Frame forward and transpose

Do not differentiate a materialized M or build a coordinatewise VJP chain.
Forward and reverse use the paired exact blocked substitutions derived in
Section 2. The same structured pair generator serves the primal solve, its
transpose solve, the direct dual actions, and their pair-dot transposes.

### 7.1 Primal

\[
y=L^{-1}k,\qquad v=\sigma^{-1}\odot y,
\qquad d=U^{-1}v.
\]

For each lower or upper factor, production executes the exact coordinate-axis
generalized-Delta recurrence in 16-coordinate blocks. The diagonal block uses
FLA's ordered unit-triangular solve. Complete off-diagonal blocks use broad
Tensor-Core pair products. Full boundary blocks are direct J/D tiles; local
outer products use MESA's identity

\[
\boxed{((XB^T)\odot\Omega)A.}
\]

For H, `(A,B)=(u,u)`; for D, `(A,B)=(u,h)`. Each action forms the two resident
score panels

\[
C_H=Xu^T,\qquad C_R=Xh^T,
\]

then applies the strict mask and route coefficients before the second dot. The
lower action uses `kappa_R^-`; the upper action uses `kappa_R^+` and reverses
the coordinate order. Each diagonal tile preserves the required ordered
dependency; all earlier complete tiles are consumed by block products. No full
L/U/H/R tensor or formal generalized-Delta feature panel exists.

### 7.2 Paired dual

For `x` equal to b and q, batch both routes:

\[
t=L^Tx,\qquad s=\sigma\odot t,
\qquad [e,\chi]=U^Ts.
\]

For transposed D actions, H and R source scores share u; the target panel is a
row-scaled combination of u and h. The native action does not require a
synthetic `3C=96` axis, but an internal packed consumer layout remains eligible
under the fusion-boundary A/B in Section 12.4.

### 7.3 Primal reverse

For one factor solve `y=(I+N)^{-1}b`, the mathematical implicit VJP is

\[
z=(I+N)^{-T}\bar y,\qquad
\bar b=z,\qquad \bar N=-zy^T.
\]

Production evaluates `z` with the reversed blocked substitution and streams
the strict blocks of `-zy^T` through FLA's pair-dot transpose schedule. Applied
to the staged primal action, this gives exactly, up to declared operand
rounding and FP32 reduction order,

\[
\bar v=U^{-T}\bar d,
\qquad \bar U\mathrel{+}=-\bar v d^T,
\]

\[
\bar y=\bar v\oslash\sigma,
\qquad \bar\ell\mathrel{-}=\bar v\odot v,
\]

\[
\bar k=L^{-T}\bar y,
\qquad \bar L\mathrel{+}=-\bar k y^T.
\]

The forward `y` panel used by the rank-one cotangent is a declared private
training cache or is recomputed by the same exact blocked recurrence. Cache
versus recompute is a static complete-path A/B choice; neither choice creates
per-coordinate HBM traffic.

Each rank-one factor cotangent is generated in registers and consumed by the
chart transpose before its coordinate tile is released.

### 7.4 Paired dual reverse

Recompute `t=L^Tx,s=sigma*t` for `x in {b,q}`, then

\[
\bar s=U\bar e,
\quad \bar U\mathrel{+}=s\bar e^T,
\quad \bar t=\sigma\odot\bar s,
\quad \bar\ell\mathrel{+}=\bar s\odot s,
\]

\[
\bar b=L\bar t,
\qquad \bar L\mathrel{+}=b\bar t^T.
\]

The same equations hold for chi/q. These are transpose block actions, not
entrywise VJP chains.

The reverse applies the same validity predicates. Cotangents pass unchanged
through an invalid row, stop at a reset boundary, and never enter a masked
frame. This rule is implemented by the affine scan transpose, not by clearing
a large gradient workspace.

## 8. Chart reverse without descriptors

For `Y=B_c(x)=a*x`,

\[
\boxed{
\bar x=a\left[
\bar Y-x\frac{\langle\bar Y,x\rangle}
{c^2+\lVert x\rVert^2}
\right].}
\]

This is the scaled FLA L2Norm transpose from Section 2 with
`a=c/sqrt(c^2+||x||^2)`. The formulas below only propagate that standard
primitive through the matrix-free representation
`x=gamma*P(X)/m`; they do not define another radial backward.

For `Y=kappa P(X)`, let

\[
R_c=c^2m^2+\gamma^2n,
\qquad \kappa=c\gamma R_c^{-1/2},
\qquad s=\langle\bar Y,P(X)\rangle.
\]

Then

\[
\bar n\mathrel{+}=-s\frac{c\gamma^3}{2R_c^{3/2}},
\quad
\bar m\mathrel{+}=-s\frac{c^3\gamma m}{R_c^{3/2}},
\]

\[
\bar\gamma\mathrel{+}=s\frac{c^3m^2}{R_c^{3/2}},
\qquad
\overline{P(X)}\mathrel{+}=\kappa\bar Y.
\]

For H, combine factor cotangents before radial reverse:

\[
\bar Y_H^-=P_-(\bar L)+P_-((\bar U)^T).
\]

There is one H radial VJP. R keeps lower and upper routes.

For diagonal coordinates,

\[
\bar x_H=\bar\ell\odot[1-\tanh^2(x_H/s_H)],
\quad
\bar x_R=\bar\ell\odot[1-\tanh^2(x_R/s_R)],
\]

\[
\overline{\operatorname{diag}J_i}\mathrel{+}
=\gamma\bar x_H/m_i,
\qquad
\overline{\operatorname{diag}D_i}\mathrel{+}
=\gamma\bar x_R/m_i,
\]

\[
\bar m_i\mathrel{-}=\frac{\gamma}{m_i^2}
\left[\bar x_H^T\operatorname{diag}J_i
+\bar x_R^T\operatorname{diag}D_i\right],
\]

\[
\bar\gamma\mathrel{+}=
\bar x_H^T(\operatorname{diag}J_i/m_i-1/r)
+\bar x_R^T(\operatorname{diag}D_i/m_i).
\]

Frame reverse splits only at the radial global scalar dependency. Primal and
paired-dual kernels apply the fixed-scale `kappa*barY` transpose immediately
and emit one `bar n_i` scalar per route. A radial pair-statistics transpose
then consumes those scalars and saved `G,c,n`. This writes about 96 KiB of
FP32 scalars at the target, not a 24 MiB descriptor tensor.

For

\[
n_i=\Lambda_i^2n_0+2\Lambda_i c^Tw_i+w_i^TGw_i,
\]

the reverse is

\[
\bar n_0\mathrel{+}=\bar n_i\Lambda_i^2,
\quad \bar c\mathrel{+}=2\bar n_i\Lambda_iw_i,
\quad \bar G\mathrel{+}=\bar n_iw_iw_i^T,
\]

\[
\bar\Lambda_i\mathrel{+}=2\bar n_i(\Lambda_i n_0+c^Tw_i),
\quad
\bar w_i\mathrel{+}=2\bar n_i(\Lambda_i c+Gw_i).
\]

The scale path also has a direct cotangent. Define
`C_i=kappa_i*bar Y_i`. Since `Y_i=kappa_i P(X_i)`,

\[
\overline{P(X_0)}\mathrel{+}=\sum_i\Lambda_iP(C_i),
\qquad
\overline{P(O_j)}\mathrel{+}=\sum_{i\ge j}w_{ij}P(C_i).
\]

The norm-statistics transpose is

\[
\overline{P(X_0)}\mathrel{+}=
2\bar n_0P(X_0)+\sum_j\bar c_jP(O_j),
\]

\[
\overline{P(O_j)}\mathrel{+}=
\bar c_jP(X_0)+
\sum_k(\bar G_{jk}+\bar G_{kj})P(O_k).
\]

The following formulas freeze the Gram/Hadamard transpose that implements the
last equation. They are algebraic interfaces; MESA remains the owner of the
tile schedule.

For the symmetric H route, stack the token rows into `U in R^{C x r}` and set

\[
K=UU^T,\qquad U^{(2)}=U\odot U,
\qquad G^H=\tfrac12\left(K\odot K-U^{(2)}(U^{(2)})^T\right).
\]

Given `Z^H=bar G^H` and `Z_s^H=Z^H+(Z^H)^T`, its complete transpose is

\[
\boxed{
\bar U\mathrel{+}=
 \left(Z_s^H\odot K\right)U
 -U\odot\left[Z_s^H(U\odot U)\right].}
\]

The production `Z^H` is symmetric because it is a sum of
`bar n_i w_iw_i^T`; the unsimplified expression above also defines the
cotangent convention for a general test input.

For the R routes, let `Z^-=bar G^-` and `Z^+=bar G^+`. The complete-coordinate
tiles in Section 3 have cotangents

\[
\bar K^u_p\mathrel{+}=
 Z^-\odot K^{h,<p}+Z^+\odot K^{h,>p},
\]

\[
\bar K^h_p\mathrel{+}=
 Z^-\odot K^{u,>p}+Z^+\odot K^{u,<p}.
\]

Their Gram transposes are

\[
\boxed{
\bar U_p\mathrel{+}=(\bar K^u_p+(\bar K^u_p)^T)U_p,
\qquad
\bar H_p\mathrel{+}=(\bar K^h_p+(\bar K^h_p)^T)H_p.}
\]

For each within-tile strict Gram, the diagonal-tile transpose is

\[
\overline{T^{\pm}_{p,j}}\mathrel{+}=
 \sum_k(Z^{\pm}_{jk}+Z^{\pm}_{kj})T^{\pm}_{p,k},
\]

\[
\boxed{
\bar u_{j,p}\mathrel{+}=
 P_\pm^{(p)}(\overline{T^{\pm}_{p,j}})h_{j,p},
\qquad
\bar h_{j,p}\mathrel{+}=
 P_\pm^{(p)}(\overline{T^{\pm}_{p,j}})^Tu_{j,p}.}
\]

The lower and upper contributions are added. The same strict mask is applied
when forming `T`, loading `Z`, and applying its transpose; diagonal entries
never acquire a cotangent. These `bar U/bar H` contributions are added to the
boundary-correlation and direct scale-path contributions already given above.
No projected outer product or coordinate-pair descriptor is an external
backward interface.

For `O_j=A_jB_j^T`, the last transpose is simply

\[
\bar A_j\mathrel{+}=P(\bar O_j)B_j,
\qquad
\bar B_j\mathrel{+}=P(\bar O_j)^TA_j.
\]

Thus H with `A=B=u` receives
`[P(bar O)+P(bar O)^T]u`; R receives `P(bar O)h` in `bar u` and
`P(bar O)^T u` in `bar h`. On a reset-free segment, the weight transpose is

\[
\bar G_p^{(g)}\mathrel{+}=\bar\Lambda_p\Lambda_p
+\sum_{j\le p}\bar w_{pj}w_{pj}
-\sum_{i\ge p}\bar w_{ip}w_{ip},
\qquad
\bar a_p=\sum_{i\ge p}\bar G_i^{(g)}.
\]

The segmented case is the ordinary transpose of the affine monoid in Section
3; a reset has zero derivative into the preceding segment. MESA-style
Gram/Hadamard transpose maps these formulas directly to boundary J/D, u/h,
and decay gradients. FP32 atomics are accepted only if repeated-run drift
passes the VJP gate. The deterministic alternative follows Mamba's fixed
workspace plus one reduction. There is no runtime numeric fallback.

## 9. Geometry reverse

The optimized chunk reverse must equal this tokenwise oracle:

\[
\bar\lambda_i\mathrel{+}=\bar m_i m_{i-1}
+\langle\bar J_i,J_{i-1}\rangle_F
+\langle\bar D_i,D_{i-1}\rangle_F,
\]

\[
\bar m_{i-1}\mathrel{+}=\lambda_i\bar m_i,
\quad \bar J_{i-1}\mathrel{+}=\lambda_i\bar J_i,
\quad \bar D_{i-1}\mathrel{+}=\lambda_i\bar D_i,
\]

\[
\bar u_i\mathrel{+}=(\bar J_i+\bar J_i^T)u_i+\bar D_i h_i,
\quad
\bar h_i\mathrel{+}=\bar D_i^Tu_i,
\quad
\bar a_i\mathrel{+}=\lambda_i\bar\lambda_i.
\]

The chunk implementation decomposes this linear reverse into chart-local
transpose, a MESA/FLA resident reverse scan over chunk boundaries, and the
MESA Hkk/Hkv transition transpose. The reverse resident loop is the exact
specialization

\[
\bar J_c=\bar J_c^{local}+\Lambda_c\bar J_{c+1},
\qquad
\bar D_c=\bar D_c^{local}+\Lambda_c\bar D_{c+1}.
\]

It reuses FLA common `chunk_bwd_kernel_dh` residency, replacing only that
kernel's model-specific `q*do` source by the already formed chart cotangent.
The Hkk/Hkv Gram transpose supplies transition gradients. MESA's high-level CG
backward wrapper is not called because its `q_star/do/lambda` ABI is unrelated
to SolveDelta. No load, scan, matrix multiply, reduction, or state-store
mechanism is newly designed. Geometry scan remains separate; serializing all
chunks in a frame kernel would expose only `B*H=8` CTAs at the target.

For full symmetric J storage the cotangent representative is

\[
\boxed{\bar J_{sym}=\tfrac12(\bar J+\bar J^T).}
\]

## 10. Upstream ownership

Audited sources:

- FLA `bc3b101dcb713ddc5bd8924b66754eb68b5ccf89`, MIT;
- MESA kernels in that FLA tree, MIT;
- Mamba `e9594ce1c732d97440f0332fdc43170a2294dbfa`, Apache-2.0;
- causal-conv1d `cd81f0413cad2fc1e6f17e785ac39f59aae690cd`, BSD-3-Clause;
- CUTLASS `7107b05535f8977f5ecb9d01ee203205b1fd9bc4`, BSD-3-Clause;
- installed cuBLASDx/cuSolverDx 26.06.

| Stage | Reuse owner | Upstream block | SolveDelta change |
|---|---|---|---|
| packed conv4 | causal-conv1d | channel-last width-4 fwd/bwd | pack q/k/v channels and expose existing state ABI |
| q/k L2 norm | FLA `modules/l2norm.py` | launch/reduction schedule | use reference `max(norm,eps)` and direct FP16 store |
| gate/cumsum | FLA KDA/GDN2 utils | scalar/vector schedules | pointer/shape specialization |
| J/D boundaries | MESA `chunk_h_fwd.py` plus FLA common state reverse | paired resident Hkk/Hkv | map to `(u_fp16,u_bf16,h)`, add m, retain FP32 |
| radial norm/VJP | FLA `modules/l2norm.py` plus MESA Gram/Hadamard | scaled epsilon-L2Norm and matrix-free norm statistics | set `eps=c^2`, retain masks/routes, never materialize x |
| diagonal chart | FLA tanh soft-cap primitive plus exp | `s*tanh(x/s)` forward/reverse | add H/R routes in the frame FP32 epilogue |
| L/U direct action | MESA `chunk_update_once` two-dot | resident state plus local rank-k residual action | coordinate strict mask and route coefficients |
| L/U inverse action | FLA GDN2/KDA blocked substitution plus MathDx TRSM oracle | exact ordered diagonal solve and off-diagonal pair-dot composition | specialize the strict H/R pair producer and its transpose |
| W/A and chunk solve | FLA GDN2/KDA/DPLR | safe diagonal, gauged off-diagonal, candidate C16/C32/C64 solve blocks | independent e/chi row panels and d columns |
| WY RHS | FLA GDN2 `wy_fast.py` | wide RHS schedule | direct E and Z operands |
| S state/output | FLA common state and DPLR/GLA output | complete fwd/bwd programs | direct-e naming and FP32 continuation |
| frame MMA | CUTLASS CuTe plus MESA | SM80 MMA/cp.async atoms and two-dot | coordinate orchestration only |
| reductions | CUB | BlockReduce/BlockScan/warp primitives | scalar functors only |
| diagonal oracle | MathDx | thread TRSM | A/B oracle, never block collective by default |

The exact source symbols audited for first adoption are:

| Purpose | Upstream symbol |
|---|---|
| radial scalar forward/reverse | FLA `l2norm_fwd_kernel` / `l2norm_bwd_kernel` |
| diagonal soft-cap forward/reverse | FLA `ops.utils.op.tanh` and `cross_entropy_fwd_kernel` / `cross_entropy_bwd_kernel` soft-cap epilogues |
| matrix-free residual state action | MESA `chunk_update_once` |
| exact coordinate-axis inverse action | FLA GDN2/KDA fused inter-solve schedule and `solve_tril` ordered diagonal block |
| chunk unit-lower inverse candidates | FLA C16 solve, C32 `merge_16x16_to_32x32_inverse_kernel`, and a C64 upstream schedule to audit |
| wide WY right-hand sides | FLA DPLR `wu_fwd_kernel` |
| WY matrix reverse | FLA DPLR `prepare_wy_repr_bwd_kernel` |
| pair Tensor Core forward/reverse | FLA DPLR `chunk_dplr_fwd_A_kernel_intra_tensorcore` / `chunk_dplr_bwd_kernel_intra_tensorcore` |
| output and output reverse | FLA DPLR `chunk_dplr_fwd_kernel_o` / `chunk_dplr_bwd_o_kernel` |
| state and state reverse | FLA DPLR `chunk_dplr_fwd_kernel_h` / `chunk_dplr_bwd_kernel_dhu` |
| paired J/D resident state | MESA `chunk_mesa_net_fwd_kernel_h`, FLA common `chunk_bwd_kernel_dh`, and MESA Hkk/Hkv transpose dot schedules |
| Gram/Hadamard transpose | MESA `chunk_mesa_net_h_kk_bwd_intra_kernel` / `chunk_mesa_net_h_kv_bwd_intra_kernel` |
| copy and MMA atoms | CuTe `SM80_CP_ASYNC_*`, `SM80_16x8x16_F32F16F16F32_TN`, and `SM80_16x8x16_F32BF16BF16F32_TN` |

Names are pinned to the commits above. Adoption means copying or specializing
the complete program around that symbol, including its producer/consumer
layout. It does not mean calling one of its collectives inside a foreign CTA.
FLA sites that currently permit TF32, including the intermediate dot in
`wu_fwd_kernel`, are specialized to the declared FP16/BF16 operand bits with
FP32 accumulation. The generic kernel's unchecked TF32 comment is not part of
the SolveDelta precision contract.

Copied or materially adapted code retains source headers and is listed in
`THIRD_PARTY_NOTICES.md`.

Rejected reuse forms are a complete device-level GEMM collective embedded in
the frame CTA and a generic DPLR wrapper that materializes fake gates or 3C
panels. The former duplicates staging/barriers/register lifetimes; the latter
restores the glue this rebuild removes.

## 11. Forward launch graph

| ID | Owner | Grid at target | Persistent output |
|---|---|---:|---|
| F0 | cuBLASLt/PyTorch | library | packed projection |
| F1 | causal-conv1d | `(B,ceil(T/128),channels)` | packed q/k/v and optional cache |
| F2 | FLA-derived preprocess | token rows | FP16 u/q/k/b, BF16 z, FP32 logs; raw sigmoid is consumer-owned |
| F3 | FLA cumsum | chunk/head/channel tiles | geometry and associative cumulative logs |
| F4 | MESA resident m/J/D specialization | `(B*H,ceil(r/32),ceil(r/32))` | FP32 m/J/D boundary history |
| F5 | radial specializations | `(3,B*H*Nchunk)` | FP32 G/c/n and coefficients |
| F6 | chart diagonal | `(B*H*Nchunk,ceil(r/128))` | FP16 sigma |
| F7 | bounded-LDU primal | `B*H*Nchunk` | FP16 d |
| F8 | bounded-LDU dual2 | `B*H*Nchunk` | FP16 e/chi |
| F9 | FLA pair/inverse at selected C | diagonal and off-diagonal grids | packed chunk inverse and A |
| F10 | FLA wide RHS | `B*H*Nchunk` | U/Y |
| F11 | FLA state forward | state tiles | FP32 final S |
| F12 | FLA output | panel/value tiles | BF16 output |
| F13 | cuBLASLt/PyTorch | library | model-width output |

F0 places q/k/v contiguously. F1 consumes the strided channel-last slice as
Mamba does; there is no `cat/permute/contiguous`. Conv weights are one packed
parameter, not concatenated on each call.

That zero-copy statement is the all-valid, reset-free performance path. For an
arbitrary mask/reset batch, a CUB prefix/select compacts valid rows into
reset-delimited sequences, upstream causal-conv1d consumes the packed buffer
with `seq_idx`, and one indexed scatter restores token positions. Its current
Python ABI intentionally forbids `seq_idx` together with `return_final_states`.
The exact width-4 cache is therefore a three-element gather from the final
packed segment, zero-filled before its reset boundary; its VJP is the matching
three-element scatter-add before causal-conv1d backward. Invalid rows never
shift the cache. This small mask-path adapter is operator glue, not a second
convolution implementation, and it is tested against the token recurrence.

F2 retains the exact reference normalization

\[
y=x/\max(\lVert x\rVert_2,\epsilon).
\]

FLA's current `sqrt(sum+eps)` formula is not adopted unchanged because its
zero-input VJP differs.

F10 uses FLA's safe diagonal path. Fully off-diagonal 16-by-16 token tiles use
a reference gauge rho:

\[
\exp(G_i-G_j)=\exp(G_i-\rho)\exp(\rho-G_j),
\]

with both exponents nonpositive. Those factors feed MMA; no off-diagonal pair
performs its own exponential and no unsafe global inverse gauge exists.

## 12. SM120 hardware contract

The RTX 5070 Ti target (`sm_120`) has 70 SMs, 65,536 registers/SM, and 102,400
bytes shared memory/SM. Local FLA PTX confirms this FP16/BF16 pipeline:

1. `cp.async.{ca,cg}.shared.global` stages 16-byte segments;
2. `cp.async.commit_group/wait_group` manages the pipeline;
3. swizzled tiles use `ldmatrix.sync.aligned`;
4. products use
   `mma.sync.aligned.m16n8k16.row.col.f32.{f16|bf16}.{f16|bf16}.f32`;
5. accumulators remain FP32 until declared stores.

There is no `tcgen05`/TMEM requirement for FP16/BF16 on this SM120 path.
CUTLASS SM120-specific atoms target narrow 4/6/8-bit operations; CuTe's SM80
FP16/BF16 warp atoms are the correct primitives.

### 12.1 Frame CTA

The initial schedule search includes four- and eight-warp programs for each
`C in {16,32,64}`. Tile width, stages, shared allocation, register lifetime,
spill traffic, and achieved CTAs/SM follow the complete mature schedule being
adapted; none is a semantic constant or an isolated pass/fail threshold. The
forward program keeps the source, solved blocks, and generalized-Delta prefix
state resident across coordinate blocks. Reverse uses the matching suffix
state and FP32 cotangents because their magnitude is not statically bounded.

The following is the C32/four-warp resource estimate used to seed one candidate,
not a contract for the winner:

| Shared buffer | Type/shape | Bytes |
|---|---|---:|
| double-buffered u/h tiles | `2 x 2 x [C,16] x 2` | 4,096 |
| source and solved panel | `2 x [C,r] FP16` | 16,384 |
| H/R prefix state or pair scores | `2 x [C,C] FP16` | 4,096 |
| double-buffered J/D tile | `2 x 2 x [16,16] BF16` | 2,048 |
| coefficient/RHS scratch | fixed ceiling | 8,192 |
| alignment/CuTe reserve | fixed ceiling | 6,144 |
| total ceiling | | 40,960 |

Scores accumulate in FP32 fragments and cross one FP16 shared boundary before
their consumer dot. FP32 J/D entries convert RN to BF16 during staging. There
is no high/low expansion or data-dependent precision branch.

A `32x16x16` product is exactly four `m16n8k16` MMA instructions: two token
halves by two coordinate halves. A `32x32x16` score update is eight MMA
instructions. Each off-diagonal coordinate block uses these pair products;
each 16-coordinate diagonal block uses the ordered FLA solve because that is
where the true dependency lives. Solved blocks update the resident prefix once
and are never replayed as polynomial iterates.

The only project-owned arithmetic is the strict H/R specialization and its
transpose around MESA's resident two-dot action. MathDx exact TRSM remains an
out-of-line oracle; no device-level TRSM collective is embedded in the frame
CTA.

CI records registers/thread, shared bytes, spills, barriers, MMA/cp.async
counts, and achieved CTAs/SM from the compiled cubin. Source estimates are not
accepted as hardware facts.

CuTe MMA/copy atoms are compile-time instruction mappings and add no hidden
launch or collective barrier. CUTLASS `CollectiveMma` and cuBLASDx block GEMM
do own staging and synchronization, so they are not embedded.

The radial epilogue is scaled FLA L2Norm, and the diagonal epilogue is FLA's
tanh soft-cap derivative followed by exp. Their placement, route masks, and
ordering are SolveDelta-specific; their scalar mathematics, reductions,
copies, matrix products, and fixed resident loop skeleton are not.

### 12.2 Frame forward microprogram

For each `(batch,head,chunk)` CTA, the primal lower pass executes this fixed
sequence:

1. Stage `u/h`, FP32 boundary `J/D`, validity predicates, radial statistics,
   and the source panel once; invalid rows are zero-filled.
2. Generate the bounded lower coefficients and initialize the resident
   generalized-Delta prefix state.
3. Traverse 16-coordinate blocks in increasing order. Generate each diagonal
   pair tile, run the exact FLA ordered solve, update the resident prefix, and
   apply the solved block to later complete blocks with the upstream pair-dot
   schedule. Generate-use-discard every pair tile.
4. Apply `exp(-ell)` in FP32 to the exact lower result and directly pack the
   upper-solve source.
5. Traverse the same program in decreasing coordinate order with
   `kappa_R^+`, then emit the exact primal result at the selected private
   frame-to-pair boundary.

The paired dual CTA loads `b` and `q` together, applies exact direct actions
`L^T`, `exp(ell)`, and `U^T` with the same matrix-free atoms, and emits the two
row panels at that same private boundary. It performs no inverse iteration and
does not require a synthetic `3C` tile; such packing is a downstream schedule
choice rather than an operator property.

The exact factor bounds give `||L^-1||,||U^-1|| <= 4/3`, so the complete primal
panel is bounded by `e^(1/4)(4/3)^2 < 2.284` times its normalized source and the
declared direct FP16 panels are analytically safe. FP32 produces each ordered
update before its direct store. Casting an already rounded BF16 result to FP16
is not permitted.

Every CTA barrier in this list protects an actual shared producer/consumer.
Adding a barrier, a global score store, or a second shared-memory owner is a
design change that must win a complete-path A/B.

### 12.3 Frame reverse microprogram

The primal transpose CTA runs the exact reversed block substitution for
`U^-T bar_d`, produces `-bar_v d^T` as FP32 fragments, and immediately feeds
those fragments to the upper H/R pair transpose. It then forms
`bar_y/bar_ell`, runs the exact reversed block substitution for `L^-T bar_y`,
and consumes `-bar_k y^T` in the lower pair transpose. Forward `y` is consumed
from the declared cache or recomputed with the same exact recurrence; no
coordinate descriptor or iterative panel is written to HBM.

Reverse iterates and base additions remain FP32 because upstream cotangents
have no static FP16 magnitude bound. Their Tensor Core action operands use one
statically chosen BF16 conversion with FP32 accumulation. There is no FP16
pseudo-promotion, magnitude test, convergence branch, or correction pass. The
reverse overlays buffers whose lifetimes do not overlap; its separate compiled
resource report is compared with the other C/warp candidates rather than
inheriting the C32/four-warp estimate by assertion.

The paired-dual transpose CTA batches `e` and `chi`: one `U` action produces
both `bar_s` panels and the two `s bar_e^T` contributions, then elementwise
FP32 scaling produces `bar_t/bar_ell`, and one `L` action produces `bar_b` and
`bar_q` plus `b bar_t^T` and `q bar_t^T`. For each coordinate tile, fixed-scale
chart contributions are consumed before register reuse. Only three FP32
`bar_n` scalar panels leave these CTAs. The later MESA Gram/Hadamard transpose
owns the norm-dependent global reduction and writes final FP32 geometry
partials directly.

The four factor contributions use output-owned accumulation. A boundary or
vector tile has one CTA owner; the primal-upper, primal-lower, dual-upper, and
dual-lower launches update the same FP32 `bar J/bar D/bar u/bar h` tile in
stream order and no additional combined matrix panel is written. This is
cross-kernel accumulation with disjoint tile ownership, not an atomic
reduction or a cross-chunk mega-kernel. Only the small route-scalar and sigma
cotangents retain separate reduction buffers.

Thus forward and reverse have paired ownership: solve versus transpose solve,
action versus transpose action, and Gram reduction versus Gram transpose. No
autograd-generated entrywise VJP or descriptor checkpoint is permitted.

### 12.4 Fusion boundary rule

The model-facing SolveDelta API does not expose frame intermediates, but the
internal frame-to-pair ABI is an implementation choice. The selected
implementation compares at least these legal schedules:

1. a direct generate-use-discard epilogue when frame and pair/WY share a
   compatible CTA count, tile layout, and resident lifetime;
2. separate mature kernels connected by one compact private staging layout,
   using `d/e/chi`, transformed `D_bar/E/Q`, or an upstream-native packed
   layout according to the consumer schedule.

The first schedule is adopted only when complete forward and F+B median and
p95 improve. The A/B must report registers, local-memory spills, shared bytes,
active CTAs/SM, barriers, producer/consumer wait time, launch count, HBM bytes,
and backward recomputation or saved-cache cost. Lower HBM traffic alone is not
an adoption result. A fusion that lengthens unrelated lifetimes, introduces a
CTA-wide synchronization owner, drops occupancy, or serializes work that the
split form executes across more CTAs is rejected.

This follows the audited GDN2 organization: it selectively fuses compatible
intra/WY gradient work but leaves cumsum, state, output, `dAv`, and other
transpose stages in separate kernels. SolveDelta likewise fuses by measured
schedule compatibility, not by the aesthetic goal of one kernel or one CTA.
No internal tensor name, packed axis, Python/CUDA boundary, or zero-HBM target
is part of the mathematical contract. An internal `d/e/chi` ABI is retained
when it lets mature kernels keep a better layout or CTA schedule, and removed
only when a measured fused or transformed-panel alternative wins end to end.
Only model-visible outputs and continuation states remain fixed public API.

The locally compiled upstream DPLR output kernel uses four warps, three stages,
28,672 bytes shared, `cp.async`, `ldmatrix`, FP16 `mma.sync`, and zero TMEM.
This is one source schedule and comparison point, not a required warp/stage
count or isolated-latency ratio for the selected exterior.

## 13. Backward launch graph

| ID | Owner | Output/action |
|---|---|---|
| B0 | output projection | BF16 bar O |
| B1 | FLA output/state reverse | bar Rz/Q/A and FP32 state cotangents |
| B2 | FLA transpose state scan | FP32 bar S0 and WY cotangents |
| B3 | FLA selected-C transpose/pair | FP16 bar d/e/chi, BF16 bar z, FP32 bar g |
| B4 | primal frame transpose | bar k, direct chart contributions, radial scalars |
| B5 | paired dual transpose | bar b/q, direct chart contributions, radial scalars |
| B6 | MESA radial pair transpose | norm-dependent boundary/local contributions |
| B7 | MESA/FLA resident boundary reverse | FP32 symmetric bar J, dense bar D, scalar bar m |
| B8 | MESA Hkk/Hkv transpose | remaining bar u/h/a |
| B9 | norm/gate reverse | raw and parameter cotangents |
| B10 | causal-conv1d reverse | q/k/v and conv state/weights |
| B11 | cuBLASLt/PyTorch | projection/input/parameter gradients |

Default FP32 atomics are frozen only after drift and complete-latency gates.
Deterministic mode is a compile-time schedule, not a numeric fallback.

Forbidden allocations:

- `[route,panel,C,3,r]` descriptors;
- tokenwise bar H/bar R matrices;
- coordinate-pair correlation or projection partials;
- both full W and its inverse unless a complete A/B proves they are needed.

## 14. Precision contract

| Quantity | Storage/arithmetic |
|---|---|
| public/raw and unbounded z/value | BF16 |
| normalized q/k and bounded b | FP32 producer, direct FP16 store |
| normalized u | FP32 producer, direct FP16 J/chart bits and BF16 D bits |
| bounded coefficients/factors/d/e/chi | FP32 producer, direct FP16 store |
| log rates, exp/softplus/tanh, m, norms, coefficients | FP32 |
| J/D/S continuation and boundaries | FP32 |
| dense boundary MMA operands | FP32 load converted to BF16 |
| H local normalized operands | FP16 |
| D local operand involving unbounded h | BF16 |
| Tensor Core accumulators | FP32 only |
| W/A and diagonal solve | FP32 accumulation/scalars, saved packed consumer bits |
| backward partials/parameter reductions | FP32 |

FP16 is allowed only for a statically bounded direct FP32 producer.
BF16-to-FP16 pseudo-promotion is forbidden. Forward and reverse consume the
same packed bits. There is no threshold, fallback, clipping, or high/low path.

The bounds are explicit. Each H or R radial strict contribution has Frobenius
norm below `1/8`, hence the summed strict part of either triangular factor has
spectral norm below `1/4`. Therefore

\[
\lVert L\rVert_2,\lVert U\rVert_2\le 5/4,
\qquad
\lVert L^{-1}\rVert_2,\lVert U^{-1}\rVert_2\le 4/3,
\]

and `exp(ell)` lies in `[exp(-1/4),exp(1/4)]`. With normalized `k/q` and
`erase in [0,2]`,

\[
\lVert d\rVert_2\le (4/3)^2e^{1/4}<2.29,
\quad
\lVert\chi\rVert_2\le (5/4)^2e^{1/4}<2.01,
\quad
\lVert e\rVert_2<4.02.
\]

`sigma`, factor entries, `d/e/chi`, normalized panels, and bounded `b` are
therefore safe direct-FP16 producers. Unbounded `h`, values, `z`, and
continuation states are never admitted to that class.

The oracle quantizes public operands to BF16 once, promotes those bits to FP64,
and reproduces each private FP16 rounding point. Structural identities remain
bitwise exact. Nonstructural cancellation is judged at BF16-observable actions
and VJPs.

Old ceilings are not inherited. Before native acceptance, new same-packed
ceilings must be generated over the declared envelope and frozen before
performance tuning. They may not depend on architecture, sequence length, or
a runtime condition estimate.

## 15. Target memory ledger

The following ledger is the retained C32 reference for
`B=1,T=1024,H=8,r=d_v=128,K=1`; C16 and C64 candidates must report their own
panel counts, boundary histories, saved cache, and allocator peak.

| Saved object | Size |
|---|---:|
| one `[B,T,H,r]` FP16 panel | 2.00 MiB |
| d/e/chi | 6.00 MiB |
| sigma | 2.00 MiB |
| three route G caches | 3.00 MiB |
| route c/n/coefficients | about 0.20 MiB |
| packed inverse plus A at C32 | 1.00 MiB |
| one FP32 J or D 33-boundary history | 16.50 MiB |
| J+D histories | 33.00 MiB |
| radial scalar cotangents | about 0.09 MiB |

Core saved training cache is about 45 MiB/layer before ordinary projection
inputs. Dense canonical geometry boundaries dominate, not d/e/chi.

FLA S boundaries are recomputed in B1/B2. Their temporary 16.5 MiB lifetime
ends before the 33 MiB geometry adjoint buffer is allocated. Actual allocator
peak, not a sum of disjoint lifetimes, is reported.

## 16. Performance and falsification gates

Compare against a matched complete GDN2 layer: same model/head widths, conv4,
projections, output projection, dtype, gradients, and state policy.

1. Compare every C/warp candidate with the corresponding mature FLA exterior
   at the same chunk width and workload; select by complete forward and F+B,
   not a fixed isolated-kernel percentage.
2. J/D boundary forward/reverse must stay within 15 percent of corresponding
   two-state MESA after accounting for m.
3. Report frame registers, shared bytes, spills, barriers, stages, achieved
   CTAs/SM, and occupancy. A spill or lower occupancy is rejected only when its
   complete-path cost loses to another passing schedule.
4. Frame backward consists of transpose block actions plus radial scalar/pair
   reverse, never descriptor replay.
5. Complete F+B must improve the archived `6.539 ms` median by at least 25
   percent before the replacement is accepted. The archived implementation may
   be reproduced in a separate worktree; it need not coexist with the rebuild.
6. First acceptance is at most `1.5x` matched complete GDN2 F+B; research
   target is `1.25x` or better.
7. Report saved cache and allocator peak with latency.

Timing uses alternating A/B order, warmup, CUDA events, at least seven medians,
and identical process/clocks. A candidate is deleted if complete F+B regresses
even when an isolated kernel wins.

## 17. Replacement order

1. Delete obsolete native sources, Python glue, tests, validation documents,
   execution documents, and private ABIs. Retain `causallsso/reference.py`, this
   blueprint, the current model-visible contract, and provenance records.
2. Add standalone FP64 chunk identities for Sections 3-9.
3. Build direct-e FLA exterior without chart code.
4. Specialize MESA/FLA FP32 J/D resident forward/reverse without changing its
   low-level schedule.
5. Build radial statistics and diagonal producers.
6. Build primal forward and strict transpose together.
7. Build paired dual forward and strict transpose together.
8. Compose the full operator and rebuild/freeze the minimal semantic,
   production-observable, and same-packed diagnostic gates from the oracle.
9. Profile/tune only after the entire forward/backward closes.
10. Update notices and publish measured artifacts.

Old native behavior is never a compatibility requirement. Only the FP64
operator, declared mixed-precision schedule, complete VJP, and measured target
workload survive.

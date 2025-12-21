# 计算 ∂P/∂F 或 δP

> 原文：[`phys-sim-book.github.io/lec14.3-compute_stress_deriv.html`](https://phys-sim-book.github.io/lec14.3-compute_stress_deriv.html)



要计算 P 对 F 的导数，我们利用之前讨论的 P 的旋转不变性属性。考虑两个任意的旋转矩阵 R 和 Q。从 P 的旋转属性中，我们有：

P(F)=P(RRTFQQT)=RP(RTFQ)QT.

定义 K=RTFQ，然后：

P(F)=RP(K)QT.

在将 R 和 Q 视为常数的同时对 P 求微分，得到：

δP=R[∂F∂P​(K):δ(K)]QT=R[∂F∂P​(K):(RTδFQ)]QT.

通过设置 R=U 和 Q=V，其中 K=Σ，微分表达式简化为：

δP=U[∂F∂P​(Σ):(UTδFV)]VT.

张量导数 ∂P/∂F 然后用指标符号表示为：

(δP)ij​=Uik​(∂F∂P​(Σ))klmn​Urm​δFrs​Vsn​Vjl​,and(δP)ij​=(∂F∂P​(F))ijrs​δFrs​.

这些表达式必须对任何 δF 成立，从而得到以下关系：

(∂F∂P​(F))ijrs​=(∂F∂P​(Σ))klmn​Uik​Urm​Vsn​Vjl​.

因此，剩余的任务是计算 ∂F∂P​(Σ)。我们将在 3D 中展示如何进行计算。

首先，让我们介绍罗德里格斯旋转公式，该公式提供了一种用单位向量 k 和旋转角 θ 来表示任何旋转矩阵的方法。公式如下：R=I+sin(θ)K+(1−cos(θ))K2,(14.3.1) 其中 K 是与 k 相关的反对称交叉乘积矩阵。这个公式表明，任何旋转矩阵只由三个自由度来表征，分别表示为 r1​,r2​,r3​。这些分量用于定义旋转向量 r，从 r 中可以导出 k 和 θ，如下所示：

k=∣r∣r​,θ=∣r∣.

使用这种参数化，旋转矩阵 U 和 V 每个都可以用三个参数来描述。

现在我们有以下代码，用于根据 s1, s2, s3, u1, u2, u3, v1, v2, v3 定义 F，其中 U 和 V 由 ui​ 和 vi​ 通过罗德里格斯旋转公式定义，si​ 是 Σ 中的奇异值。

```py
id=IdentityMatrix[3];
var={s1,s2,s3,u1,u2,u3,v1,v2,v3};
Sigma=DiagonalMatrix[{s1,s2,s3}];
cp[k1_,k2_,k3_]={{0,-k3,k2},{k3,0,-k1},{-k2,k1,0}};
vV={v1,v2,v3};
vU={u1,u2,u3};
nv=Sqrt[Dot[vV,vV]];
nu=Sqrt[Dot[vU,vU]];
UU=cp[u1,u2,u3]/nu;
VV=cp[v1,v2,v3]/nv;
U=id+Sin[nu]*UU+(1-Cos[nu])*UU.UU;
V=id+Sin[nv]*VV+(1-Cos[nv])*VV.VV;
F=U.Sigma.Transpose[V]; 
```

其中 cp 是一个生成交叉乘积矩阵的函数（对应于计算方程 (14.3.1) 中的 K）。

从现在开始，我们将 3×3×3×3 张量 ∂F∂P​(Σ) 和任何其他此类张量写成 9×9 矩阵。这意味着每个 3×3 矩阵现在是一个大小为 9 的向量。很容易看出旧的 ∂Fkl​∂Pij​​ 现在是 ∂F3(k−1)+l​∂P3(i−1)+j​​。我们进一步将向量 S={s1,s2,s3,u1,u2,u3,v1,v2,v3} 定义为 F 的参数化。然后我们可以应用链式法则 ∂F∂P​(Σ)=∂S∂P​(Σ)∂F∂S​(Σ)

这里是计算它们的 Mathematica 代码。注意，我们通过取极限 {u1,u2,u3,v1,v2,v3}=+ϵ 来实现 F=Σ，这对应于几乎为零的旋转。

```py
dFdS=D[Flatten[F],{var}];
dFdS0=dFdS/.{u1->e,u2->e,u3->e,v1->e,v2->e,v3->e};
dFdS1=Limit[dFdS0,e->0,Direction->-1];
dSdF0=Inverse[dFdS1];
Phat=DiagonalMatrix[{t1[s1,s2,s3],t2[s1,s2,s3],t3[s1,s2,s3]}];
P=U.Phat.Transpose[V];
dPdS=D[Flatten[P],{var}];
dPdS0=dPdS/.{u1->e,u2->e,u3->e,v1->e,v2->e,v3->e};
dPdS1=Limit[dPdS0,e->0,Direction->-1];
dPdF=Simplify[dPdS1.dSdF0]; 
```

注意，Mathematica 中的 'Direction->-1' 表示从大值到小极限值的极限。Mathematica 的计算结果将以奇异值和 P^ 的形式给出。然后可以取公式在代码中实现它们。[[Stomakhin 等人 2012]](bibliography.html#stomakhin2012energetically) 给出了 ∂F∂P(Σ)（9×9 矩阵大小）被排列为块对角矩阵的结果，其中对角块为 A3×3,B122×2,B132×2,B232×2，其中 A=Ψ^,σ1σ1Ψ^,σ2σ1Ψ^,σ3σ1Ψ^,σ1σ2Ψ^,σ2σ2Ψ^,σ3σ2Ψ^,σ1σ3Ψ^,σ2σ3Ψ^,σ3σ3Ψ^,σi2−σj2(σiΨ^,σi−σjΨ^,σjσjΨ^,σi−σiΨ^,σjσjΨ^,σi−σiΨ^,σjσiΨ^,σi−σjΨ^,σj)。对于可能引入除以零的 B 项，需要分母限制。在这里，我们用 Ψ^,σi 和 Ψ^,σiσj 分别表示 ∂σi∂Ψ^ 和 ∂σi∂σj∂2Ψ^。当两个奇异值几乎相等或两个奇异值几乎相加为零时，除以 σi2−σj2 是有问题的。后者可以通过允许负奇异值的约定来实现（如可逆弹性 [[Irving 等人 2004]](bibliography.html#irving2004invertible) [[Stomakhin 等人 2012]](bibliography.html#stomakhin2012energetically))。

将 Bij 展开成部分分式得到有用的分解 Bij=21σi−σjΨ^,σi−Ψ^,σj(11−11)+21σi+σjΨ^,σi+Ψ^,σj(1−1−11). 注意，如果 Ψ^ 在奇异值排列下是不变的，那么当 σi→σj 时，Ψ^,σi→Ψ^,σj. 因此，如果实现得小心，第一个项通常可以稳健地计算一个各向同性模型。如果 Ψ^,σi+Ψ^,σj→0 当 σi+σj→0 时，其他分数也可以稳健地计算。但这种情况通常不成立，因为这意味着本构模型将难以从退化或倒置配置中恢复。因此，在某些情况下，这个项将是无界的。我们通过在除法之前将分母的幅度限制为不小于 10−6 来解决这个问题，以限制导数。

对于二维，旋转矩阵现在仅用单个 θ 参数化，其重建为

R=(cosθsinθ−sinθcosθ). 整个 Mathematica 代码的二维版本是

```py
id=IdentityMatrix[2];
var={s1,s2,u1,v1};
S=DiagonalMatrix[{s1,s2}];
U={{Cos[u1],-Sin[u1]

},{Sin[u1],Cos[u1]}};
V={{Cos[v1],-Sin[v1]},{Sin[v1],Cos[v1]}};
F=U.S.Transpose[V];
dFdS=D[Flatten[F],{var}];
dFdS0=dFdS/.{u1->e,v1->e};
dFdS1=Limit[dFdS0,e->0,Direction->-1};
dSdF0=Inverse[dFdS1];
Phat=DiagonalMatrix[{t1[s1,s2],t2[s1,s2]}];
P=U.Phat.Transpose[V];
dPdS=D[Flatten[P],{var}];
dPdS0=dPdS/.{u1->e,v1->e};
dPdS1=Limit[dPdS0,e->0,Direction->-1];
dPdF=Simplify[dPdS1.dSdF0]; 
```

其中 A 现在也是 2×2，只有一个 B。

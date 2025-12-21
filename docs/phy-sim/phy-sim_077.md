# 计算能量、梯度和 Hessian

> 原文：[`phys-sim-book.github.io/lec15.2-energy_grad_hess.html`](https://phys-sim-book.github.io/lec15.2-energy_grad_hess.html)



我们首先遵循应变能和应力和其导数这两节，实现计算Ψ(F)，P=∂F/∂Ψ，以及 SPD 投影的∂F/∂P：

**实现 15.2.1（关于 F 的能量导数，NeoHookeanEnergy.py）**

```py
import utils
import numpy as np
import math

def polar_svd(F):
    [U, s, VT] = np.linalg.svd(F)
    if np.linalg.det(U) < 0:
        U[:, 1] = -U[:, 1]
        s[1] = -s[1]
    if np.linalg.det(VT) < 0:
        VT[1, :] = -VT[1, :]
        s[1] = -s[1]
    return [U, s, VT]

def dPsi_div_dsigma(s, mu, lam):
    ln_sigma_prod = math.log(s[0] * s[1])
    inv0 = 1.0 / s[0]
    dPsi_dsigma_0 = mu * (s[0] - inv0) + lam * inv0 * ln_sigma_prod
    inv1 = 1.0 / s[1]
    dPsi_dsigma_1 = mu * (s[1] - inv1) + lam * inv1 * ln_sigma_prod
    return [dPsi_dsigma_0, dPsi_dsigma_1]

def d2Psi_div_dsigma2(s, mu, lam):
    ln_sigma_prod = math.log(s[0] * s[1])
    inv2_0 = 1 / (s[0] * s[0])
    d2Psi_dsigma2_00 = mu * (1 + inv2_0) - lam * inv2_0 * (ln_sigma_prod - 1)
    inv2_1 = 1 / (s[1] * s[1])
    d2Psi_dsigma2_11 = mu * (1 + inv2_1) - lam * inv2_1 * (ln_sigma_prod - 1)
    d2Psi_dsigma2_01 = lam / (s[0] * s[1])
    return [[d2Psi_dsigma2_00, d2Psi_dsigma2_01], [d2Psi_dsigma2_01, d2Psi_dsigma2_11]]

def B_left_coef(s, mu, lam):
    sigma_prod = s[0] * s[1]
    return (mu + (mu - lam * math.log(sigma_prod)) / sigma_prod) / 2

def Psi(F, mu, lam):
    J = np.linalg.det(F)
    lnJ = math.log(J)
    return mu / 2 * (np.trace(np.transpose(F).dot(F)) - 2) - mu * lnJ + lam / 2 * lnJ * lnJ

def dPsi_div_dF(F, mu, lam):
    FinvT = np.transpose(np.linalg.inv(F))
    return mu * (F - FinvT) + lam * math.log(np.linalg.det(F)) * FinvT

def d2Psi_div_dF2(F, mu, lam):
    [U, sigma, VT] = polar_svd(F)

    Psi_sigma_sigma = utils.make_PSD(d2Psi_div_dsigma2(sigma, mu, lam))

    B_left = B_left_coef(sigma, mu, lam)
    Psi_sigma = dPsi_div_dsigma(sigma, mu, lam)
    B_right = (Psi_sigma[0] + Psi_sigma[1]) / (2 * max(sigma[0] + sigma[1], 1e-6))
    B = utils.make_PSD([[B_left + B_right, B_left - B_right], [B_left - B_right, B_left + B_right]])

    M = np.array([[0, 0, 0, 0]] * 4)
    M[0, 0] = Psi_sigma_sigma[0, 0]
    M[0, 3] = Psi_sigma_sigma[0, 1]
    M[1, 1] = B[0, 0]
    M[1, 2] = B[0, 1]
    M[2, 1] = B[1, 0]
    M[2, 2] = B[1, 1]
    M[3, 0] = Psi_sigma_sigma[1, 0]
    M[3, 3] = Psi_sigma_sigma[1, 1]

    dP_div_dF = np.array([[0, 0, 0, 0]] * 4)
    for j in range(0, 2):
        for i in range(0, 2):
            ij = j * 2 + i
            for s in range(0, 2):
                for r in range(0, 2):
                    rs = s * 2 + r
                    dP_div_dF[ij, rs] = M[0, 0] * U[i, 0] * VT[0, j] * U[r, 0] * VT[0, s] \
                        + M[0, 3] * U[i, 0] * VT[0, j] * U[r, 1] * VT[1, s] \
                        + M[1, 1] * U[i, 1] * VT[0, j] * U[r, 1] * VT[0, s] \
                        + M[1, 2] * U[i, 1] * VT[0, j] * U[r, 0] * VT[1, s] \
                        + M[2, 1] * U[i, 0] * VT[1, j] * U[r, 1] * VT[0, s] \
                        + M[2, 2] * U[i, 0] * VT[1, j] * U[r, 0] * VT[1, s] \
                        + M[3, 0] * U[i, 1] * VT[1, j] * U[r, 0] * VT[0, s] \
                        + M[3, 3] * U[i, 1] * VT[1, j] * U[r, 1] * VT[1, s]
    return dP_div_dF 
```

接下来，我们实现计算 F(x)，以及与∂x∂F 相关的张量积，用于基于链式法则计算弹性能量梯度和 Hessian：

**实现 15.2.2（关于 x 的能量导数，NeoHookeanEnergy.py）**

```py
def deformation_grad(x, elemVInd, IB):
    F = [x[elemVInd[1]] - x[elemVInd[0]], x[elemVInd[2]] - x[elemVInd[0]]]
    return np.transpose(F).dot(IB)

def dPsi_div_dx(P, IB):  # applying chain-rule, dPsi_div_dx = dPsi_div_dF * dF_div_dx
    dPsi_dx_2 = P[0, 0] * IB[0, 0] + P[0, 1] * IB[0, 1]
    dPsi_dx_3 = P[1, 0] * IB[0, 0] + P[1, 1] * IB[0, 1]
    dPsi_dx_4 = P[0, 0] * IB[1, 0] + P[0, 1] * IB[1, 1]
    dPsi_dx_5 = P[1, 0] * IB[1, 0] + P[1, 1] * IB[1, 1]
    return [np.array([-dPsi_dx_2 - dPsi_dx_4, -dPsi_dx_3 - dPsi_dx_5]), np.array([dPsi_dx_2, dPsi_dx_3]), np.array([dPsi_dx_4, dPsi_dx_5])]

def d2Psi_div_dx2(dP_div_dF, IB):  # applying chain-rule, d2Psi_div_dx2 = dF_div_dx^T * d2Psi_div_dF2 * dF_div_dx (note that d2F_div_dx2 = 0)
    intermediate = np.array([[0.0, 0.0, 0.0, 0.0]] * 6)
    for colI in range(0, 4):
        _000 = dP_div_dF[0, colI] * IB[0, 0]
        _010 = dP_div_dF[0, colI] * IB[1, 0]
        _101 = dP_div_dF[2, colI] * IB[0, 1]
        _111 = dP_div_dF[2, colI] * IB[1, 1]
        _200 = dP_div_dF[1, colI] * IB[0, 0]
        _210 = dP_div_dF[1, colI] * IB[1, 0]
        _301 = dP_div_dF[3, colI] * IB[0, 1]
        _311 = dP_div_dF[3, colI] * IB[1, 1]
        intermediate[2, colI] = _000 + _101
        intermediate[3, colI] = _200 + _301
        intermediate[4, colI] = _010 + _111
        intermediate[5, colI] = _210 + _311
        intermediate[0, colI] = -intermediate[2, colI] - intermediate[4, colI]
        intermediate[1, colI] = -intermediate[3, colI] - intermediate[5, colI]
    result = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 6)
    for colI in range(0, 6):
        _000 = intermediate[colI, 0] * IB[0, 0]
        _010 = intermediate[colI, 0] * IB[1, 0]
        _101 = intermediate[colI, 2] * IB[0, 1]
        _111 = intermediate[colI, 2] * IB[1, 1]
        _200 = intermediate[colI, 1] * IB[0, 0]
        _210 = intermediate[colI, 1] * IB[1, 0]
        _301 = intermediate[colI, 3] * IB[0, 1]
        _311 = intermediate[colI, 3] * IB[1, 1]
        result[2, colI] = _000 + _101
        result[3, colI] = _200 + _301
        result[4, colI] = _010 + _111
        result[5, colI] = _210 + _311
        result[0, colI] = -_000 - _101 - _010 - _111
        result[1, colI] = -_200 - _301 - _210 - _311
    return result 
```

最后，整个网格上的 Neo-Hookean 能量值、梯度和 Hessian 可以按以下方式计算：

**实现 15.2.3（能量值、梯度和 Hessian，NeoHookeanEnergy.py）**

```py
def val(x, e, vol, IB, mu, lam):
    sum = 0.0
    for i in range(0, len(e)):
        F = deformation_grad(x, e[i], IB[i])
        sum += vol[i] * Psi(F, mu[i], lam[i])
    return sum

def grad(x, e, vol, IB, mu, lam):
    g = np.array([[0.0, 0.0]] * len(x))
    for i in range(0, len(e)):
        F = deformation_grad(x, e[i], IB[i])
        P = vol[i] * dPsi_div_dF(F, mu[i], lam[i])
        g_local = dPsi_div_dx(P, IB[i])
        for j in range(0, 3):
            g[e[i][j]] += g_local[j]
    return g

def hess(x, e, vol, IB, mu, lam):
    IJV = [[0] * (len(e) * 36), [0] * (len(e) * 36), np.array([0.0] * (len(e) * 36))]
    for i in range(0, len(e)):
        F = deformation_grad(x, e[i], IB[i])
        dP_div_dF = vol[i] * d2Psi_div_dF2(F, mu[i], lam[i])
        local_hess = d2Psi_div_dx2(dP_div_dF, IB[i])
        for xI in range(0, 3):
            for xJ in range(0, 3):
                for dI in range(0, 2):
                    for dJ in range(0, 2):
                        ind = i * 36 + (xI * 3 + xJ) * 4 + dI * 2 + dJ
                        IJV[0][ind] = e[i][xI] * 2 + dI
                        IJV[1][ind] = e[i][xJ] * 2 + dJ
                        IJV[2][ind] = local_hess[xI * 2 + dI, xJ * 2 + dJ]
    return IJV 
```

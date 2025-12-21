# 弹性

> 原文：[`phys-sim-book.github.io/lec23.3-elasticity.html`](https://phys-sim-book.github.io/lec23.3-elasticity.html)



对于弹性，类似于二维情况，变形梯度 F 在每个四面体内也是常数，我们可以将其计算为 F=≈=∂(β,γ,τ)/∂x(∂(β,γ,τ)/∂X)−1∂(β,γ,τ)/∂x^(∂(β,γ,τ)/∂X)−1[x2−x1,x3−x1,x4−x1][X2−X1,X3−X1,X4−X1]−1. 对于力和 Hessian 的计算，所需的∂F/∂x 可以使用∇XN1(X)=∂X/∂(1−β−γ−τ)=(∂(β,γ,τ)/∂(1−β−γ−τ)(∂(β,γ,τ)/∂X)−1)^T=([−1,−1,−1][X2−X1,X3−X1,X4−X1]−1)^T 和类似地∇XN2(X)∇XN3(X)∇XN3(X)=∂X/∂β=([1,0,0][X2−X1,X3−X1,X4−X1]−1)^T,=∂X/∂γ=([0,1,0][X2−X1,X3−X1,X4−X1]−1)^T,=∂X/∂τ=([0,0,1][X2−X1,X3−X1,X4−X1]−1)^T. 使用 F，应变能Ψ、应力 P 和应力导数∂P/∂F 的计算都可以在应变能和应力和其导数中找到，力的计算和 Hessian 矩阵的计算遵循与二维相同的精神。

为了保证在模拟过程中四面体元素的非反转，可以通过求解每个四面体的 1D 方程 V(xi+αIpi)=0，首先将任何四面体的体积降至 0 的临界步长αI 可以找到，然后取解的步长中的最小值。这里 pi 是节点 i 的搜索方向，在 3D 中，这相当于 det([x21α,x31α,x41α])≡(x21α×x31α)⋅x41α=0(23.3.1) with xijα=xij+αIpij and xij=xi−xj, pij=pi−pj. 展开方程(23.3.1)，我们得到以下关于αI 的三次方程：

((p21×p31)⋅p41)αI3+((x21×p31+p21×x31)⋅p41+(p21×p31)⋅x41)αI2+((x21×p31+p21×x31)⋅x41+(x21×x31)⋅p41)αI+(x21×x31)⋅x41=0,​

这个三次方程有时会退化成二次或线性方程，尤其是在节点移动没有显著改变四面体体积的情况下。为了解决潜在的数值不稳定性，我们根据常数项系数缩放方程项：

(x21×x31)⋅x41(p21×p31)⋅p41αI3+(x21×x31)⋅x41(x21×p31+p21×x31)⋅p41+(p21×p31)⋅x41αI2+(x21×x31)⋅x41(x21×p31+p21×x31)⋅x41+(x21×x31)⋅p41αI+1=0,​(23.3.2)

确保可以使用标准阈值（例如，10−6）安全地进行量级检查。

实际上，我们通过求解αI 来确保一些安全余量，这可以减少任何四面体的体积 80%，将方程(23.3.2)中的常数项系数从 1 修改为 0.8。如果没有找到正实根，步长可以被认为是安全的，并且不会发生反转。以下是解决此缩放三次方程的 C++代码片段：

**实现 23.3.1（三次方程求解器）。**

```py
double getSmallestPositiveRealRoot_cubic(double a, double b, double c, double d,
    double tol)
{
    // return negative value if no positive real root is found
    double t = -1;

    if (abs(a) <= tol)
        t = getSmallestPositiveRealRoot_quad(b, c, d, tol); // covered in the 2D case
    else {
        complex<double> i(0, 1);
        complex<double> delta0(b * b - 3 * a * c, 0);
        complex<double> delta1(2 * b * b * b - 9 * a * b * c + 27 * a * a * d, 0);
        complex<double> C = pow((delta1 + sqrt(delta1 * delta1 - 4.0 * delta0 * delta0 * delta0)) / 2.0, 1.0 / 3.0);
        if (std::abs(C) == 0.0) // a corner case
            C = pow((delta1 - sqrt(delta1 * delta1 - 4.0 * delta0 * delta0 * delta0)) / 2.0, 1.0 / 3.0);

        complex<double> u2 = (-1.0 + sqrt(3.0) * i) / 2.0;
        complex<double> u3 = (-1.0 - sqrt(3.0) * i) / 2.0;

        complex<double> t1 = (b + C + delta0 / C) / (-3.0 * a);
        complex<double> t2 = (b + u2 * C + delta0 / (u2 * C)) / (-3.0 * a);
        complex<double> t3 = (b + u3 * C + delta0 / (u3 * C)) / (-3.0 * a);

        if ((abs(imag(t1)) < tol) && (real(t1) > 0))
            t = real(t1);
        if ((abs(imag(t2)) < tol) && (real(t2) > 0) && ((real(t2) < t) || (t < 0)))
            t = real(t2);
        if ((abs(imag(t3)) < tol) && (real(t3) > 0) && ((real(t3) < t) || (t < 0)))
            t = real(t3);
    }
    return t;
} 
```

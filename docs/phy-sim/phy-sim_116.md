# 3D 弹性动力学

> 原文：[`phys-sim-book.github.io/lec23-3d_elastodynamics.html`](https://phys-sim-book.github.io/lec23-3d_elastodynamics.html)

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css">

为了将我们的二维固体模拟器（2D 摩擦自接触）扩展到三维，我们可以使用 3-简单四面体单元来离散化三维固体域。在这种方法中，固体的表面被表示为三角形网格，这是计算机图形学中用于表示三维几何形状的常用方法。此外，我们还需要在固体的内部采样顶点，以形成离散化惯性和弹性能量的四面体元素。

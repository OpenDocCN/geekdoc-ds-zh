# 3D 摩擦自接触

> 原文：[`phys-sim-book.github.io/lec24-3d_fric_self_contact.html`](https://phys-sim-book.github.io/lec24-3d_fric_self_contact.html)

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css">

在三维空间中，表示为三角形网格的固体域边界之间的接触可以简化为点-三角形和边-边接触。直观上，二维空间中的点-边接触对直接扩展到三维空间成为点-三角形对。然而，即使我们防止了三维空间中所有点-三角形之间的穿透，三角形网格仍然可以相互穿透。这需要考虑边-边对，尤其是在网格分辨率不是很高的情况下。

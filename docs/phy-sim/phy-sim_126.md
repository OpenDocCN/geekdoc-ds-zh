# 刚体模拟*

> 原文：[`phys-sim-book.github.io/lec25-rigid_body_sim.html`](https://phys-sim-book.github.io/lec25-rigid_body_sim.html)



**本讲座作者：[杜文欣](https://dwxrycb123.github.io/), 加州大学洛杉矶分校**

为了将 IPC 方法扩展到刚体接触，我们可以采用子空间方法来有效地减少系统的自由度（DOFs）。由于刚体是极硬的、实际物体的一种理想化模型，其变形可以忽略不计，因此它不需要大量的自由度。这一观察结果促使了仿射体动力学（ABD）[[Lan et al. 2022]](bibliography.html#lan2022affine)方法的出现。根据子空间的选择，这种方法也可以适应为适合不同类别软体的各种算法。

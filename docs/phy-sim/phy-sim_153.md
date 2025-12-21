# 模拟设置

> 原文：[`phys-sim-book.github.io/lec29.1-simulation_setup.html`](https://phys-sim-book.github.io/lec29.1-simulation_setup.html)

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css">

在本节中，我们定义了实现二维**两个碰撞弹性块的最小 MPM 模拟**所需的物理和数值设置。我们介绍了模拟属性的定义、粒子位置和速度的初始化以及整个模拟中使用的数据结构。

### 物理和数值参数

我们首先设置模拟域的离散化和块的材料参数：

**实现 30.1.1（物理和数值参数，simulator.py）**。

```py
# simulation setup
grid_size = 128 # background Eulerian grid's resolution, in 2D is [128, 128]
dx = 1.0 / grid_size # the domain size is [1m, 1m] in 2D, so dx for each cell is (1/128)m
dt = 2e-4 # time step size in second
ppc = 8 # average particles per cell

density = 1000 # mass density, unit: kg / m³
E, nu = 1e4, 0.3 # block's Young's modulus and Poisson's ratio
mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu)) # Lame parameters 
```

这些参数定义了一个均匀的密集背景网格、粒子分辨率和时间积分步长。整个模拟域从 `[0, 0]` 到 `[1, 1]` 米，我们希望平均每个网格单元有大约 8 个粒子。块被设置为质量密度为 1000kg/m3，杨氏模量为 104Pa，泊松比为 0.3。

### 初始粒子采样和场景设置

我们使用**统一网格采样**从两个矩形区域采样粒子。这两个盒子被对称地放置在域的左右两侧，并初始化为相反的速度，以模拟正面碰撞。

与泊松盘采样相比，由于其结构化和简单的参数化，统一采样更容易实现分析形状，如盒子和平面，但由于这种规律性，可能会导致**混叠伪影**，如模拟中的可见图案或条纹，这可能会将不自然的结构噪声引入结果中。

在这里，我们为了简洁和清晰，采用了**统一采样**，将重点放在 MPM 管道本身。

**实现 30.1.2（初始粒子采样和场景设置，simulator.py）**。

```py
# uniformly sampling material particles
def uniform_grid(x0, y0, x1, y1, dx):
    xx, yy = np.meshgrid(np.arange(x0, x1 + dx, dx), np.arange(y0, y1 + dx, dx))
    return np.column_stack((xx.ravel(), yy.ravel()))

box1_samples = uniform_grid(0.2, 0.4, 0.4, 0.6, dx / np.sqrt(ppc))
box1_velocities = np.tile(np.array([10.0, 0]), (len(box1_samples), 1))
box2_samples = uniform_grid(0.6, 0.4, 0.8, 0.6, dx / np.sqrt(ppc))
box2_velocities = np.tile(np.array([-10.0, 0]), (len(box1_samples), 1))
all_samples = np.concatenate([box1_samples, box2_samples], axis=0)
all_velocities = np.concatenate([box1_velocities, box2_velocities], axis=0) 
```

每个块由均匀分布的材料点组成，代表一个均匀的弹性体。左侧的块被赋予初始速度([+10, 0]) m/s，右侧的块为([-10, 0]) m/s，设置了一个具有**零净线性动量**的对称、正面碰撞场景。这种配置模拟了一个受控冲击实验。

### 粒子和网格数据字段

我们定义数据字段来表示每个材料点（粒子）和背景网格节点的状态。对于粒子，这包括位置、速度、体积、质量和变形梯度，遵循材料粒子。对于网格，我们使用密集数组定义节点质量和速度字段，这对于小规模模拟是足够的。这些可以通过稀疏网格结构进一步优化——我们将这一方向留给感兴趣的读者作为未来的工作。

**实现 30.1.3 (粒子与网格数据字段，simulator.py)**.

```py
# material particles data
N_particles = len(all_samples)
x = ti.Vector.field(2, float, N_particles) # the position of particles
x.from_numpy(all_samples)
v = ti.Vector.field(2, float, N_particles) # the velocity of particles
v.from_numpy(all_velocities)
vol = ti.field(float, N_particles)         # the volume of particle
vol.fill(0.2 * 0.4 / N_particles) # get the volume of each particle as V_rest / N_particles
m = ti.field(float, N_particles)           # the mass of particle
m.fill(vol[0] * density)
F = ti.Matrix.field(2, 2, float, N_particles)  # the deformation gradient of particles
F.from_numpy(np.tile(np.eye(2), (N_particles, 1, 1)))

# grid data
grid_m = ti.field(float, (grid_size, grid_size))
grid_v = ti.Vector.field(2, float, (grid_size, grid_size)) 
```

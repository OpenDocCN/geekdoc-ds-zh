# GPU 加速模拟

> 原文：[`phys-sim-book.github.io/lec4.6-gpu_accel.html`](https://phys-sim-book.github.io/lec4.6-gpu_accel.html)



**本节作者：[罗肇锋](https://roushelfy.github.io/)，卡内基梅隆大学**

我们现在重写 2D 质量-弹簧模拟器以利用 GPU 加速。我们不是直接编写[CUDA](https://developer.nvidia.com/cuda-toolkit)，而是求助于[MUDA](https://github.com/MuGdxy/muda)，这是一个轻量级的库，它提供了一个简单的接口用于 GPU 加速计算。

GPU 加速模拟器的架构与 Python 版本类似。所有函数和变量名都与 Numpy 版本一致。然而，由于 GPU 架构和编程模型的不同，实现细节也不同。在深入细节之前，让我们先从以下 gif（图 4.6.1）中感受 GPU 能为我们带来的加速效果。

![](img/fbd53762482cd3796eb946c75f5c6562.png) ![](img/ac44bb5d43380bbdb6eebffdc5e83ef8.png)

**图 4.6.1**。Numpy CPU（左）和 MUDA GPU（右）版本模拟速度的示意图。

### GPU 编程的关键考虑因素

为了最大化 GPU 上的资源利用率，有两个重要的方面需要考虑：

+   **最小化数据传输**。在大多数现代架构中，CPU 和 GPU 有独立的内存空间。在这些空间之间传输数据可能很昂贵。因此，最小化 CPU 和 GPU 之间的数据传输至关重要。

+   **利用并行性**。GPU 在并行计算方面表现出色。然而，当多个线程同时尝试访问相同的内存位置时，必须小心避免读写冲突。

### 最小化数据传输

为了减少 CPU 和 GPU 之间的数据传输，我们将主要能量值及其导数存储在 GPU 上。然后直接在 GPU 上执行计算，仅将必要的位置信息传输回 CPU 进行控制和渲染。更高效的实现可以在 GPU 上直接渲染，从而消除这种数据传输，但为了简单和可读性，我们在这里没有实现这一点。

为了使代码更易读，以`device_`开头的变量存储在 GPU 内存中，以`host_`开头的变量存储在 CPU 内存中。

**实现 4.6.1（数据结构，MassSpringEnergy.cu）。**

```py
template <typename T, int dim>
struct MassSpringEnergy<T, dim>::Impl
{
	DeviceBuffer<T> device_x;
	DeviceBuffer<T> device_l2, device_k;
	DeviceBuffer<int> device_e;
	int N;
	DeviceBuffer<T> device_grad;
	DeviceTripletMatrix<T, 1> device_hess;
}; 
```

如上代码所示，能量值及其导数以及所有必要的参数都存储在一个`DeviceBuffer`对象中，这是由 MUDA 库实现的 CUDA 设备内存的包装器。这使得我们能够在 GPU 上直接进行计算，而无需在 CPU 和 GPU 之间进行数据传输。

### 牛顿法

牛顿法的迭代是一个串行过程，不能并行化。因此，我们将这部分实现在了 CPU 上：

**实现 4.6.2（牛顿法，simulator.cu）。**

```py
template <typename T, int dim>
void MassSpringSimulator<T, dim>::Impl::step_forward()
{
    update_x_tilde(add_vector<T>(device_x, device_v, 1, h));
    DeviceBuffer<T> device_x_n = device_x; // Copy current positions to device_x_n
    int iter = 0;
    T E_last = IP_val();
    DeviceBuffer<T> device_p = search_direction();
    T residual = max_vector(device_p) / h;
    while (residual > tol)
    {
        std::cout << "Iteration " << iter << " residual " << residual << "E_last" << E_last << "\n";
        // Line search
        T alpha = 1;
        DeviceBuffer<T> device_x0 = device_x;
        update_x(add_vector<T>(device_x0, device_p, 1.0, alpha));
        while (IP_val() > E_last)
        {
            alpha /= 2;
            update_x(add_vector<T>(device_x0, device_p, 1.0, alpha));
        }
        std::cout << "step size = " << alpha << "\n";
        E_last = IP_val();
        device_p = search_direction();
        residual = max_vector(device_p) / h;
        iter += 1;
    }
    update_v(add_vector<T>(device_x, device_x_n, 1 / h, -1 / h));
} 
```

在这个函数 `step_forward` 中，实现了带有线搜索的投影牛顿法，在 GPU 上执行必要的计算，同时在 CPU 上控制过程。这里以 `device_` 开头的任何变量都是 GPU 上的 `DeviceBuffer` 对象。为了调试目的打印 `DeviceBuffer` 中的值，常见的做法是将数据传输回 CPU，或者调用 `uti.cu` 中实现的 `display_vec` 函数（该函数在 GPU 上并行调用 `printf`）。

`update_x` 函数更新所有能量类别的节点位置，并将更新后的位置传输回 CPU 进行渲染：

**实现 4.6.3（更新位置，simulator.cu）。**

```py
template <typename T, int dim>
void MassSpringSimulator<T, dim>::Impl::update_x(const DeviceBuffer<T> &new_x)
{
    inertialenergy.update_x(new_x);
    massspringenergy.update_x(new_x);
    device_x = new_x;
} 
```

由于能量类别已经更新了其位置，`IP_val` 函数不再需要传递任何参数，从而避免了不必要的资料传输。实际上，它只调用所有能量类别的 `val` 函数，然后将结果相加：

**实现 4.6.4（计算 IP，simulator.cu）。**

```py
template <typename T, int dim>
T MassSpringSimulator<T, dim>::Impl::IP_val()
{

    return inertialenergy.val() + massspringenergy.val() * h * h;
} 
```

对于 `IP_grad` 和 `IP_hess` 函数也是类似的：

**实现 4.6.5（计算 IP 梯度和海森矩阵，simulator.cu）。**

```py
template <typename T, int dim>
DeviceBuffer<T> MassSpringSimulator<T, dim>::Impl::IP_grad()
{
    return add_vector<T>(inertialenergy.grad(), massspringenergy.grad(), 1.0, h * h);
}

template <typename T, int dim>
DeviceTripletMatrix<T, 1> MassSpringSimulator<T, dim>::Impl::IP_hess()
{
    DeviceTripletMatrix<T, 1> inertial_hess = inertialenergy.hess();
    DeviceTripletMatrix<T, 1> massspring_hess = massspringenergy.hess();
    DeviceTripletMatrix<T, 1> hess = add_triplet<T>(inertial_hess, massspring_hess, 1.0, h * h);
    return hess;
} 
```

注意，它们利用了 GPU 上的并行操作（`add_vector` 和 `add_triplet`，这些操作在 `uti.cu` 中实现）来执行梯度和海森矩阵的求和。

### 并行计算

在我们的实现中，并行计算主要用于能量及其导数的计算，以及向量的加法和减法。以质量弹簧能量计算为例。

#### 能量计算

**实现 4.6.6（计算能量，MassSpringEnergy.cu）。**

```py
template <typename T, int dim>
T MassSpringEnergy<T, dim>::val()
{
	auto &device_x = pimpl_->device_x;
	auto &device_e = pimpl_->device_e;
	auto &device_l2 = pimpl_->device_l2;
	auto &device_k = pimpl_->device_k;
	int N = device_e.size() / 2;
	DeviceBuffer<T> device_val(N);
	ParallelFor(256).apply(N, [device_val = device_val.viewer(), device_x = device_x.cviewer(), device_e = device_e.cviewer(), device_l2 = device_l2.cviewer(), device_k = device_k.cviewer()] __device__(int i) mutable
						   {
		int idx1= device_e(2 * i); // First node index
		int idx2 = device_e(2 * i + 1); // Second node index
		T diff = 0;
		for (int d = 0; d < dim;d++){
			T diffi = device_x(dim * idx1 + d) - device_x(dim * idx2 + d);
			diff += diffi * diffi;
		}
		device_val(i) = 0.5 * device_l2(i) * device_k(i) * (diff / device_l2(i) - 1) * (diff / device_l2(i) - 1); })
		.wait();

	return devicesum(device_val);
} // Calculate the energy 
```

`ParallelFor` 函数将计算分配到多个 GPU 线程。lambda 函数中捕获的变量允许在每个线程中访问必要的数据结构。

#### 梯度计算

**实现 4.6.7（计算梯度，MassSpringEnergy.cu）。**

```py
template <typename T, int dim>
const DeviceBuffer<T> &MassSpringEnergy<T, dim>::grad()
{
	auto &device_x = pimpl_->device_x;
	auto &device_e = pimpl_->device_e;
	auto &device_l2 = pimpl_->device_l2;
	auto &device_k = pimpl_->device_k;
	auto N = pimpl_->device_e.size() / 2;
	auto &device_grad = pimpl_->device_grad;
	device_grad.fill(0);
	ParallelFor(256).apply(N, [device_x = device_x.cviewer(), device_e = device_e.cviewer(), device_l2 = device_l2.cviewer(), device_k = device_k.cviewer(), device_grad = device_grad.viewer()] __device__(int i) mutable
						   {
		int idx1= device_e(2 * i); // First node index
		int idx2 = device_e(2 * i + 1); // Second node index
		T diff = 0;
		T diffi[dim];
		for (int d = 0; d < dim;d++){
			diffi[d] = device_x(dim * idx1 + d) - device_x(dim * idx2 + d);
			diff += diffi[d] * diffi[d];
		}
		T factor = 2 * device_k(i) * (diff / device_l2(i) -1);
		for(int d=0;d<dim;d++){
		   atomicAdd(&device_grad(dim * idx1 + d), factor * diffi[d]);
		   atomicAdd(&device_grad(dim * idx2 + d), -factor * diffi[d]);	  
		} })
		.wait();
	// display_vec(device_grad);
	return device_grad;
} 
```

在梯度计算中，`atomicAdd` 函数至关重要，它确保了对共享数据的并发更新是安全的（不同的边可以更新相同节点的梯度），从而防止了竞态条件。

#### 海森矩阵计算

我们使用了稀疏矩阵数据结构来存储海森矩阵。计算在多个线程上并行化，每个线程更新海森矩阵的特定元素。稀疏矩阵的实际大小在模拟开始时计算，只为非零项分配足够的内存。这里的主要考虑是在模拟期间计算每个元素的正确索引：

**实现 4.6.8（计算海森矩阵，MassSpringEnergy.cu）。**

```py
template <typename T, int dim>
const DeviceTripletMatrix<T, 1> &MassSpringEnergy<T, dim>::hess()
{
	auto &device_x = pimpl_->device_x;
	auto &device_e = pimpl_->device_e;
	auto &device_l2 = pimpl_->device_l2;
	auto &device_k = pimpl_->device_k;
	auto N = device_e.size() / 2;
	auto &device_hess = pimpl_->device_hess;
	auto device_hess_row_idx = device_hess.row_indices();
	auto device_hess_col_idx = device_hess.col_indices();
	auto device_hess_val = device_hess.values();
	device_hess_val.fill(0);
	ParallelFor(256).apply(N, [device_x = device_x.cviewer(), device_e = device_e.cviewer(), device_l2 = device_l2.cviewer(), device_k = device_k.cviewer(), device_hess_val = device_hess_val.viewer(), device_hess_row_idx = device_hess_row_idx.viewer(), device_hess_col_idx = device_hess_col_idx.viewer(), N] __device__(int i) mutable
						   {
		int idx[2] = {device_e(2 * i), device_e(2 * i + 1)}; // First node index
		T diff = 0;
		T diffi[dim];
		for (int d = 0; d < dim; d++)
		{
			diffi[d] = device_x(dim * idx[0] + d) - device_x(dim * idx[1] + d);
			diff += diffi[d] * diffi[d];
		}
		Eigen::Matrix<T, dim, 1> diff_vec(diffi);
		Eigen::Matrix<T, dim, dim> diff_outer = diff_vec * diff_vec.transpose();
		T scalar = 2 * device_k(i) / device_l2(i);
		Eigen::Matrix<T, dim, dim> H_diff = scalar * (2 * diff_outer + (diff_vec.dot(diff_vec) - device_l2(i)) * Eigen::Matrix<T, dim, dim>::Identity());
		Eigen::Matrix<T, dim * 2, dim * 2> H_block, H_local;
		H_block << H_diff, -H_diff,
			-H_diff, H_diff;
		make_PSD(H_block, H_local);
		// add to global matrix
		for (int ni = 0; ni < 2; ni++)
			for (int nj = 0; nj < 2; nj++)
			{
				int indStart = i * 4*dim*dim + (ni * 2 + nj) * dim*dim;
				for (int d1 = 0; d1 < dim; d1++)
					for (int d2 = 0; d2 < dim; d2++){
						device_hess_row_idx(indStart + d1 * dim + d2)= idx[ni] * dim + d1;
						device_hess_col_idx(indStart + d1 * dim + d2)= idx[nj] * dim + d2;
						device_hess_val(indStart + d1 * dim + d2) = H_local(ni * dim + d1, nj * dim + d2);
					}
			} })
		.wait();
	return device_hess;
} // Calculate the Hessian of the energy 
```

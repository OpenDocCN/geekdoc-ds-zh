# GPU 编程模型简介

> [原文链接](https://enccs.github.io/gpu-programming/5-intro-to-gpu-prog-models/)

*GPU 编程：为什么、何时以及如何？* **   GPU 编程模型简介

+   [在 GitHub 上编辑](https://github.com/ENCCS/gpu-programming/blob/main/content/5-intro-to-gpu-prog-models.rst)

* * *

问题

+   不同 GPU 编程方法之间的关键区别是什么？

+   我应该如何选择适合我项目的框架？

目标

+   理解不同 GPU 编程框架的基本思想

+   在自己的代码项目中进行快速的成本效益分析

教师备注

+   20 分钟教学

+   10 分钟讨论

使用 GPU 进行计算有不同的方法。在最佳情况下，当代码已经编写完成时，只需要设置参数和初始配置就可以开始。在某些其他情况下，问题被提出的方式使得可以使用第三方库来解决代码中最密集的部分（例如，这在 Python 中的机器学习工作流程中越来越常见）。然而，这些情况仍然相当有限；一般来说，可能需要额外的编程。有许多 GPU 编程软件环境和 API 可用，可以大致分为**指令集模型**、**不可移植内核模型**和**可移植内核模型**，以及高级框架和库（包括语言级别支持的尝试）。

## 标准 C++/Fortran

使用标准 C++和 Fortran 语言编写的程序现在可以利用 NVIDIA GPU，而不需要依赖任何外部库。这要归功于[NVIDIA SDK](https://developer.nvidia.com/hpc-sdk)套件的编译器，它可以将代码翻译并优化以在 GPU 上运行。

+   [关于使用标准语言并行性加速的文章系列](https://developer.nvidia.com/blog/developing-accelerated-code-with-standard-language-parallelism/)

+   C++代码编写的指南可以在[这里](https://developer.nvidia.com/blog/accelerating-standard-c-with-gpus-using-stdpar/)找到，

+   Fortran 代码的相关信息可以在[这里](https://developer.nvidia.com/blog/accelerating-fortran-do-concurrent-with-gpus-and-the-nvidia-hpc-sdk/)找到。

这两种方法的表现很有希望，正如那些指南中提供的示例所示。

## 指令集编程

一种快速且经济的方法是使用**基于指令**的方法。在这种情况下，现有的*串行*代码被添加了*提示*，这些提示指示编译器哪些循环和区域应该在 GPU 上执行。如果没有 API，这些指令被视为注释，代码将像通常的串行代码一样执行。这种方法侧重于生产力和易用性（但可能损害性能），通过向现有代码添加并行性，无需编写特定于加速器的代码，就可以以最小的编程努力使用加速器。使用指令编程有两种常见方法，即**OpenACC**和**OpenMP**。

### OpenACC

[OpenACC](https://www.openacc.org/)是由 2010 年成立的一个联盟开发的，旨在为加速器（包括 GPU）开发一个标准、可移植和可扩展的编程模型。OpenACC 联盟的成员包括 GPU 供应商，如 NVIDIA 和 AMD，以及领先的超级计算中心、大学和软件公司。直到最近，它只支持 NVIDIA GPU，但现在有努力支持更多设备和架构。

### OpenMP

[OpenMP](https://www.openmp.org/)最初是一个多平台、共享内存并行编程 API，用于多核 CPU，最近也增加了对 GPU 卸载的支持。OpenMP 旨在支持各种类型的 GPU，这是通过父编译器实现的。

基于指令的方法与 C/C++和 FORTRAN 代码一起工作，同时为其他语言提供了第三方扩展。

## 不可移植的基于内核的模型（原生编程模型）

在进行直接 GPU 编程时，开发者可以通过编写直接与 GPU 及其硬件通信的低级代码来获得很高的控制水平。理论上，直接 GPU 编程方法提供了编写低级、GPU 加速的代码的能力，这可以在 CPU-only 代码上提供显著的性能提升。然而，它们也要求对 GPU 架构及其功能有更深入的了解，以及正在使用的特定编程方法。

### CUDA

[CUDA](https://developer.nvidia.com/cuda-toolkit)是由 NVIDIA 开发的一个并行计算平台和 API。它历史上是第一个主流 GPU 编程框架。它允许开发者编写在 GPU 上执行的类似 C 的代码。CUDA 提供了一套用于低级 GPU 编程的库和工具，并为计算密集型应用程序提供了性能提升。虽然有一个广泛的生态系统，但 CUDA 仅限于 NVIDIA 硬件。

### HIP

[HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/what_is_hip.html)（异构接口可移植性）是由 AMD 开发的一个 API，它为 GPU 编程提供了一个低级接口。HIP 旨在提供一个可以在 NVIDIA 和 AMD GPU 上使用的单一源代码。它基于 CUDA 编程模型，并提供与 CUDA 几乎相同的编程接口。

在此存储库的[content/examples/cuda-hip](https://github.com/ENCCS/gpu-programming/tree/main/content/examples/cuda-hip)目录中提供了多个 CUDA/HIP 代码示例。

## 基于可移植内核的模型（跨平台可移植性生态系统）

跨平台可移植性生态系统通常提供更高层次的抽象层，这使得 GPU 编程具有方便且可移植的编程模型。它们可以帮助减少维护和部署 GPU 加速应用程序所需的时间和精力。这些生态系统的目标是实现单源应用程序的性能可移植性。在 C++中，最著名的跨平台可移植性生态系统包括[SYCL](https://www.khronos.org/sycl/)、[OpenCL](https://www.khronos.org/opencl/)（C 和 C++ API）和[Kokkos](https://github.com/kokkos/kokkos)；其他还包括[alpaka](https://alpaka.readthedocs.io/)和[RAJA](https://github.com/LLNL/RAJA)。

### OpenCL

[OpenCL](https://www.khronos.org/opencl/)（开放计算语言）是一个跨平台的开放标准 API，用于在 CPU、GPU 和 FPGA 上执行通用并行计算。它支持来自多个供应商的广泛硬件。OpenCL 为 GPU 编程提供了一个低级编程接口，并允许开发者编写可在各种平台上执行的程序。与 CUDA、HIP、Kokkos 和 SYCL 等编程模型不同，OpenCL 使用单独的源模型。OpenCL 标准的最新版本增加了对 API 和内核代码的 C++支持，但基于 C 的接口仍然更广泛地被使用。OpenCL 工作组不提供任何自己的框架。相反，生产 OpenCL 兼容设备的供应商将其框架作为其软件开发工具包（SDK）的一部分发布。最受欢迎的两个 OpenCL SDK 分别由 NVIDIA 和 AMD 发布。在两种情况下，开发套件都是免费的，并包含构建 OpenCL 应用程序所需的库和工具。

### Kokkos

[Kokkos](https://github.com/kokkos/kokkos) 是一个开源的性能可移植并行计算编程模型，主要在桑迪亚国家实验室开发。它是一个基于 C++ 的生态系统，为在多核架构（如 CPU、GPU 和 FPGA）上运行的高效和可扩展的并行应用程序提供编程模型。Kokkos 生态系统包括几个组件，例如提供并行执行和内存抽象的 Kokkos 核心库，提供线性代数和图算法数学内核的 Kokkos 内核库，以及提供性能分析和调试工具的 Kokkos 工具库。Kokkos 组件与其他软件库和技术（如 MPI 和 OpenMP）集成良好。此外，该项目与其他项目合作，以提供可移植 C++ 编程的互操作性和标准化。

### alpaka

[alpaka](https://alpaka.readthedocs.io/)（并行内核加速抽象库）是一个开源的仅包含头文件的 C++ 库，旨在通过抽象底层并行级别，在异构加速器架构之间提供性能可移植性。该库是平台无关的，并支持包括主机 CPU（x86、ARM、RISC-V）和来自不同供应商的 GPU（NVIDIA、AMD 和 Intel）在内的多个设备的并发和协作使用。

alpaka 的一个关键优势是它只需要一个用户内核的单个实现，该实现以具有标准化接口的函数对象的形式表达。这消除了为不同后端编写专用代码的需要。该库提供各种加速器后端，包括 CUDA、HIP、SYCL、OpenMP 和串行执行，可以根据目标设备进行选择。此外，甚至可以将多个加速器后端组合起来，以在单个应用程序中针对不同的供应商硬件。

### SYCL

[SYCL](https://www.khronos.org/sycl/) 是一个免费的、开放标准的 C++ 多设备编程编程模型。它为包括 GPU 在内的异构系统提供了一种高级、单源编程模型。最初，SYCL 是在 OpenCL 的基础上开发的；然而，它不再仅限于这一点。它可以在其他低级异构计算 API（如 CUDA 或 HIP）之上实现，使开发者能够编写可以在各种平台上执行的程序。请注意，虽然 SYCL 是一个相对高级的模型，但开发者仍然需要明确编写 GPU 内核。

虽然 alpaka、Kokkos 和 RAJA 指的是特定的项目，但 SYCL 本身只是一个标准，存在多个实现。对于 GPU 编程，[Intel oneAPI DPC++](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html)（原生支持 Intel GPU，以及通过 [Codeplay oneAPI 插件](https://codeplay.com/solutions/oneapi/) 支持的 NVIDIA 和 AMD GPU）和 [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp/)（之前也称为 hipSYCL 或 Open SYCL，支持 NVIDIA 和 AMD GPU，与 Intel oneAPI DPC++ 结合使用时提供实验性的 Intel GPU 支持）是最广泛使用的。其他值得注意的实现包括 [triSYCL](https://github.com/triSYCL/triSYCL) 和 [ComputeCPP](https://developer.codeplay.com/products/computecpp/ce/home/)。

## 高级语言支持

### Python

Python 通过多个抽象级别提供对 GPU 编程的支持。

**CUDA Python, HIP Python 和 PyCUDA**

这些项目分别是 [NVIDIA-](https://developer.nvidia.com/cuda-python)、[AMD-](https://rocm.docs.amd.com/projects/hip-python/en/latest/) 和 [社区支持](https://documen.tician.de/pycuda/) 的包装器，提供了对低级 CUDA 和 HIP API 的 Python 绑定。要直接使用这些方法，在大多数情况下需要了解 CUDA 或 HIP 编程。

CUDA Python 也旨在支持更高级的工具包和库，例如 **CuPy** 和 **Numba**。

**CuPy**

[CuPy](https://cupy.dev/) 是一个基于 GPU 的数据数组库，与 NumPy/SciPy 兼容。它提供了一个与 NumPy 和 SciPy 非常相似的接口，使得开发者可以轻松过渡到 GPU 计算。使用 NumPy 编写的代码通常可以经过最小修改后适应 CuPy；在大多数直接情况下，人们可能只需在他们的 Python 代码中将‘numpy’和‘scipy’替换为‘cupy’和‘cupyx.scipy’即可。

**Numba**

[Numba](https://numba.pydata.org/) 是一个开源的 JIT 编译器，它将 Python 和 NumPy 代码的一个子集转换为优化的机器代码。Numba 支持 CUDA 兼容的 GPU，并能够使用几种不同的语法变体为它们生成代码。2021 年，对 [AMD (ROCm) 支持](https://numba.readthedocs.io/en/stable/release-notes.html#version-0-54-0-19-august-2021)的上游支持已被终止。然而，截至 2025 年，AMD 通过 [Numba HIP 包](https://github.com/ROCm/numba-hip)添加了对 Numba API 的下游支持。

### Julia

Julia 通过以下针对所有三个主要供应商的 GPU 的包，提供了对 GPU 编程的一流支持：

+   [CUDA.jl](https://cuda.juliagpu.org/stable/) 用于 NVIDIA GPU

+   [AMDGPU.jl](https://amdgpu.juliagpu.org/stable/) 用于 AMD GPU

+   [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl) 用于 Intel GPU

+   [Metal.jl](https://github.com/JuliaGPU/Metal.jl) 用于 Apple M 系列 GPU

`CUDA.jl`是最成熟的，`AMDGPU.jl`稍微落后但仍然适用于通用用途，而`oneAPI.jl`和`Metal.jl`功能齐全但可能存在错误，缺少一些功能，并提供次优性能。然而，它们的相应 API 是完全相似的，库之间的转换简单直接。

所有包都提供了高级抽象，这需要很少的编程工作，同时也提供了一种低级方法来编写内核，以实现细粒度的控制。

简而言之

+   **基于指令的编程：**

    +   现有的串行代码通过指令标注了哪些部分应该在 GPU 上执行。

    +   OpenACC 和 OpenMP 是常见的基于指令的编程模型。

    +   优先考虑生产力和易用性，而不是性能。

    +   在现有代码中添加并行性需要最少的编程工作。

+   **不可移植的基于内核的模型：**

    +   低级代码被编写来直接与 GPU 通信。

    +   CUDA 是 NVIDIA 的并行计算平台和 GPU 编程的 API。

    +   HIP 是由 AMD 开发的一个 API，为 NVIDIA 和 AMD GPU 提供了与 CUDA 相似的编程接口。

    +   需要更深入地了解 GPU 架构和编程方法。

+   **可移植的基于内核的模型：**

    +   为 GPU 编程提供可移植性的高级抽象。

    +   例子包括 OpenCL、Kokkos、alpaka、RAJA 和 SYCL。

    +   力求通过单源应用程序实现性能可移植性。

    +   可以在各种 GPU 和平台上运行，减少了维护和部署 GPU 加速应用程序所需的努力。

+   **高级语言支持：**

    +   C++和 Fortran 有支持 GPU 的语言标准并行的计划。

    +   Python 库如 PyCUDA、CuPy 和 Numba 提供了 GPU 编程能力。

    +   Julia 有 CUDA.jl、AMDGPU.jl、oneAPI.jl 和 Metal.jl 等包用于 GPU 编程。

    +   这些方法为各自语言中的 GPU 编程提供了高级抽象和接口。

## 摘要

这些 GPU 编程环境各自都有其优势和劣势，对于特定项目而言，最佳选择将取决于一系列因素，包括：

+   针对的目标硬件平台，

+   正在执行的计算类型，以及

+   开发者的经验和偏好。

**高级和生产力导向的 API**提供了一个简化的编程模型，并最大化代码的可移植性，而**低级和性能导向的 API**提供了对 GPU 硬件的高级别控制，但也需要更多的编码工作和专业知识。

## 练习

讨论

+   你以前是否使用过任何 GPU 编程框架？

+   你认为最具有挑战性的是什么？什么最有用？

请在主房间或通过 HackMD 文档中的评论告诉我们。

重点

+   GPU 编程方法可以分为 1）基于指令的，2）不可移植的基于内核的，3）可移植的基于内核的，和 4）高级语言支持。

+   每种方法都有多个框架/语言可供选择，每个都有其优缺点。上一页 下一页

* * *

© 版权所有 2023-2024，贡献者。

使用[Read the Docs](https://readthedocs.org)提供的[主题](https://github.com/readthedocs/sphinx_rtd_theme)和[Sphinx](https://www.sphinx-doc.org/)构建

+   不同的 GPU 编程方法之间有哪些关键区别？

+   我应该如何选择适合我项目的框架？

目标

+   理解不同 GPU 编程框架的基本思想

+   在自己的代码项目背景下进行快速的成本效益分析

讲师备注

+   20 分钟教学

+   10 分钟讨论

使用 GPU 进行计算有不同的方法。在最佳情况下，当代码已经编写完成时，只需设置参数和初始配置即可开始。在某些其他情况下，问题被提出的方式使得可以使用第三方库来解决代码中最密集的部分（例如，这在 Python 中的机器学习工作流程中越来越常见）。然而，这些情况仍然相当有限；通常，可能需要额外的编程。有许多 GPU 编程软件环境和 API 可用，可以大致分为**指令集模型**、**非可移植内核模型**和**可移植内核模型**，以及高级框架和库（包括语言级别的支持尝试）。

## 标准 C++/Fortran

使用标准 C++和 Fortran 语言编写的程序现在可以利用 NVIDIA GPU，而无需依赖任何外部库。这要归功于[NVIDIA SDK](https://developer.nvidia.com/hpc-sdk)编译器套件，它可以将代码翻译并优化以在 GPU 上运行。

+   [这里](https://developer.nvidia.com/blog/developing-accelerated-code-with-standard-language-parallelism/)是关于使用标准语言并行性加速的文章系列。

+   可以在这里找到编写 C++代码的指南[这里](https://developer.nvidia.com/blog/accelerating-standard-c-with-gpus-using-stdpar/),

+   而 Fortran 代码的指南可以在[这里](https://developer.nvidia.com/blog/accelerating-fortran-do-concurrent-with-gpus-and-the-nvidia-hpc-sdk/)找到。

这两种方法的表现很有希望，正如那些指南中提供的示例所示。

## 指令集编程

一种快速且经济的方法是使用**基于指令**的方法。在这种情况下，现有的*串行*代码被添加了*提示*，这些提示指示编译器哪些循环和区域应该在 GPU 上执行。如果没有 API，指令将被视为注释，代码将像通常的串行代码一样执行。这种方法侧重于生产力和易用性（但可能损害性能），并且通过在不编写特定加速器代码的情况下将并行性添加到现有代码中，以最小的编程努力使用加速器。使用指令编程有两种常见方法，即**OpenACC**和**OpenMP**。

### OpenACC

[OpenACC](https://www.openacc.org/) 是由 2010 年成立的一个联盟开发的，该联盟的目的是开发一个标准、可移植和可扩展的加速器编程模型，包括 GPU。OpenACC 联盟的成员包括 GPU 供应商，如 NVIDIA 和 AMD，以及领先的超级计算中心、大学和软件公司。直到最近，它只支持 NVIDIA GPU，但现在有努力支持更多设备和架构。

### OpenMP

[OpenMP](https://www.openmp.org/) 最初是一个多平台、共享内存并行编程 API，用于多核 CPU，并且相对较近地增加了对 GPU 卸载的支持。OpenMP 旨在支持各种类型的 GPU，这是通过父编译器实现的。

基于指令的方法与 C/C++和 FORTRAN 代码一起工作，同时还有一些第三方扩展可用于其他语言。

## 非可移植的基于内核的模型（原生编程模型）

当进行直接 GPU 编程时，开发者通过编写与 GPU 及其硬件直接通信的低级代码，拥有很高的控制水平。理论上，直接 GPU 编程方法能够提供编写低级、GPU 加速的代码的能力，这可以在性能上显著优于仅使用 CPU 的代码。然而，这也要求开发者对 GPU 架构及其功能有更深入的理解，以及所使用的特定编程方法。

### CUDA

[CUDA](https://developer.nvidia.com/cuda-toolkit) 是由 NVIDIA 开发的一个并行计算平台和 API。它历史上是第一个主流 GPU 编程框架。它允许开发者编写在 GPU 上执行的类似 C 的代码。CUDA 提供了一套用于低级 GPU 编程的库和工具，并为计算密集型应用提供了性能提升。尽管存在一个广泛的生态系统，CUDA 仍然仅限于 NVIDIA 硬件。

### HIP

[HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/what_is_hip.html)（异构接口可移植性）是由 AMD 开发的一个 API，它为 GPU 编程提供了一个低级接口。HIP 旨在提供一个可以在 NVIDIA 和 AMD GPU 上使用的单一源代码。它基于 CUDA 编程模型，并提供与 CUDA 几乎相同的编程接口。

在此存储库的[content/examples/cuda-hip](https://github.com/ENCCS/gpu-programming/tree/main/content/examples/cuda-hip)目录中提供了多个 CUDA/HIP 代码示例。

## 可移植的基于内核的模型（跨平台可移植性生态系统）

跨平台可移植性生态系统通常提供更高层次的抽象层，这使得 GPU 编程的编程模型既方便又可移植。它们可以帮助减少维护和部署 GPU 加速应用程序所需的时间和精力。这些生态系统的目标是实现单源应用程序的性能可移植性。在 C++中，最著名的跨平台可移植性生态系统是[SYCL](https://www.khronos.org/sycl/)，[OpenCL](https://www.khronos.org/opencl/)（C 和 C++ API）和[Kokkos](https://github.com/kokkos/kokkos)；其他还包括[alpaka](https://alpaka.readthedocs.io/)和[RAJA](https://github.com/LLNL/RAJA)。

### OpenCL

[OpenCL](https://www.khronos.org/opencl/)（开放计算语言）是一个跨平台的开放标准 API，用于在 CPU、GPU 和 FPGA 上执行通用并行计算。它支持来自多个供应商的广泛硬件。OpenCL 为 GPU 编程提供了一个低级编程接口，并允许开发者编写可以在各种平台上执行的程序。与 CUDA、HIP、Kokkos 和 SYCL 等编程模型不同，OpenCL 使用单独的源模型。OpenCL 标准的最新版本增加了对 API 和内核代码的 C++支持，但基于 C 的接口仍然更广泛地使用。OpenCL 工作组不提供自己的框架。相反，生产 OpenCL 兼容设备的供应商将其作为软件开发套件（SDK）的一部分发布框架。最受欢迎的两个 OpenCL SDK 是由 NVIDIA 和 AMD 发布的。在这两种情况下，开发套件都是免费的，并包含构建 OpenCL 应用程序所需的库和工具。

### Kokkos

[Kokkos](https://github.com/kokkos/kokkos) 是一个开源的性能可移植并行计算编程模型，主要在桑迪亚国家实验室开发。它是一个基于 C++ 的生态系统，为在多核架构（如 CPU、GPU 和 FPGA）上运行的高效和可扩展的并行应用程序提供编程模型。Kokkos 生态系统包括几个组件，例如提供并行执行和内存抽象的 Kokkos 核心库，提供线性代数和图算法数学内核的 Kokkos 内核库，以及提供性能分析和调试工具的 Kokkos 工具库。Kokkos 组件与 MPI 和 OpenMP 等其他软件库和技术很好地集成。此外，该项目与其他项目合作，以提供可移植 C++ 编程的互操作性和标准化。

### alpaka

[alpaka](https://alpaka.readthedocs.io/)（并行内核加速抽象库）是一个开源的仅包含头文件的 C++ 库，旨在通过抽象底层并行级别，在异构加速器架构之间提供性能可移植性。该库是平台无关的，并支持包括主机 CPU（x86、ARM、RISC-V）和来自不同供应商的 GPU（NVIDIA、AMD 和 Intel）在内的多个设备的并发和协作使用。

alpaka 的一个关键优势是它只需要用户内核的单个实现，该实现以具有标准化接口的函数对象的形式表达。这消除了为不同后端编写专用代码的需要。该库提供各种加速器后端，包括 CUDA、HIP、SYCL、OpenMP 和串行执行，可以根据目标设备进行选择。此外，甚至可以将多个加速器后端组合起来，以在单个应用程序中针对不同的供应商硬件。

### SYCL

[SYCL](https://www.khronos.org/sycl/) 是一个免费的、开放的 C++ 多设备编程标准编程模型。它为包括 GPU 在内的异构系统提供高级、单源编程模型。最初 SYCL 是在 OpenCL 的基础上开发的；然而，它不再仅限于这一点。它可以在其他低级异构计算 API（如 CUDA 或 HIP）之上实现，使开发者能够编写可在各种平台上执行的程序。请注意，虽然 SYCL 是一个相对高级的模型，但开发者仍然需要明确编写 GPU 内核。

虽然 alpaka、Kokkos 和 RAJA 指的是特定的项目，但 SYCL 本身只是一个标准，目前存在多个实现。对于 GPU 编程，[Intel oneAPI DPC++](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html)（原生支持 Intel GPU，以及通过 [Codeplay oneAPI 插件](https://codeplay.com/solutions/oneapi/) 支持的 NVIDIA 和 AMD GPU）和 [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp/)（之前也称为 hipSYCL 或 Open SYCL，支持 NVIDIA 和 AMD GPU，与 Intel oneAPI DPC++ 结合使用时提供实验性的 Intel GPU 支持）是最广泛使用的。其他值得注意的实现包括 [triSYCL](https://github.com/triSYCL/triSYCL) 和 [ComputeCPP](https://developer.codeplay.com/products/computecpp/ce/home/)。

## 高级语言支持

### Python

Python 通过多个抽象级别提供对 GPU 编程的支持。

**CUDA Python、HIP Python 和 PyCUDA**

这些项目分别是 [NVIDIA-](https://developer.nvidia.com/cuda-python)、[AMD-](https://rocm.docs.amd.com/projects/hip-python/en/latest/) 和 [社区支持的](https://documen.tician.de/pycuda/)包装器，为低级 CUDA 和 HIP API 提供了 Python 绑定。要直接使用这些方法，在大多数情况下需要了解 CUDA 或 HIP 编程。

CUDA Python 也旨在支持高级工具包和库，例如 **CuPy** 和 **Numba**。

**CuPy**

[CuPy](https://cupy.dev/) 是一个基于 GPU 的数据数组库，与 NumPy/SciPy 兼容。它提供了一个与 NumPy 和 SciPy 非常相似的接口，使得开发者可以轻松过渡到 GPU 计算。使用 NumPy 编写的代码通常可以经过最小修改后适应 CuPy；在大多数直接情况下，人们可能只需在他们的 Python 代码中将 'numpy' 和 'scipy' 替换为 'cupy' 和 'cupyx.scipy'。

**Numba**

[Numba](https://numba.pydata.org/) 是一个开源的 JIT 编译器，它将 Python 和 NumPy 代码的一个子集转换为优化的机器代码。Numba 支持 CUDA 兼容的 GPU，并能够使用几种不同的语法变体为它们生成代码。2021 年，对 [AMD (ROCm) 支持](https://numba.readthedocs.io/en/stable/release-notes.html#version-0-54-0-19-august-2021)的上游支持已被终止。然而，截至 2025 年，AMD 通过 [Numba HIP 包](https://github.com/ROCm/numba-hip)为 Numba API 添加了下游支持。

### Julia

Julia 通过以下针对所有三个主要供应商的 GPU 的包提供了一等 GPU 编程支持：

+   [CUDA.jl](https://cuda.juliagpu.org/stable/) 用于 NVIDIA GPU

+   [AMDGPU.jl](https://amdgpu.juliagpu.org/stable/) 用于 AMD GPU

+   [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl) 用于 Intel GPU

+   [Metal.jl](https://github.com/JuliaGPU/Metal.jl) 用于 Apple M 系列 GPU

`CUDA.jl`是最成熟的，`AMDGPU.jl`稍微落后但仍然可以通用，而`oneAPI.jl`和`Metal.jl`功能齐全但可能存在错误，缺少一些功能，并提供次优性能。然而，它们的相应 API 是完全类似的，库之间的转换简单直接。

所有包都提供了高级抽象，需要极少的编程努力，以及用于编写内核的较低级别方法，以实现细粒度控制。

简而言之

+   **基于指令的编程：**

    +   现有的串行代码通过指令进行注释，以指示应在 GPU 上执行的部分。

    +   OpenACC 和 OpenMP 是常见的基于指令的编程模型。

    +   优先考虑生产力和易用性，而不是性能。

    +   将并行性添加到现有代码中所需的编程工作量最小。

+   **不可移植的基于内核的模型：**

    +   低级代码是直接与 GPU 通信编写的。

    +   CUDA 是 NVIDIA 的并行计算平台和 GPU 编程 API。

    +   HIP 是 AMD 开发的一个 API，为 NVIDIA 和 AMD 的 GPU 提供了与 CUDA 类似的编程接口。

    +   需要更深入地了解 GPU 架构和编程方法。

+   **可移植的基于内核的模型：**

    +   为 GPU 编程提供可移植性的高级抽象。

    +   包括 OpenCL、Kokkos、alpaka、RAJA 和 SYCL。

    +   旨在通过单个源应用程序实现性能的可移植性。

    +   可在多种 GPU 和平台上运行，从而减少维护和部署 GPU 加速应用程序所需的努力。

+   **高级语言支持：**

    +   C++和 Fortran 有支持 GPU 的语言标准并行的计划。

    +   Python 库如 PyCUDA、CuPy 和 Numba 提供了 GPU 编程能力。

    +   Julia 有 CUDA.jl、AMDGPU.jl、oneAPI.jl 和 Metal.jl 等用于 GPU 编程的包。

    +   这些方法为各自语言中的 GPU 编程提供了高级抽象和接口。

## 摘要

每个 GPU 编程环境都有其自身的优点和缺点，对于特定项目而言，最佳选择将取决于一系列因素，包括：

+   目标硬件平台，

+   执行的计算类型，以及

+   开发者的经验和偏好。

**高级且注重生产力的 API**提供了一个简化的编程模型并最大化代码的可移植性，而**低级且注重性能的 API**则提供了对 GPU 硬件的高级别控制，但也需要更多的编码努力和专业知识。

## 练习

讨论

+   您之前是否已经使用过任何 GPU 编程框架？

+   您认为哪部分最具挑战性？哪部分最有用？

请在主房间或通过 HackMD 文档中的评论告知我们。

重点

+   GPU 编程方法可以分为 1）基于指令的，2）不可移植的基于内核的，3）可移植的基于内核的，和 4）高级语言支持。

+   每种方法都有多个框架/语言可供选择，各有优缺点。

## 标准 C++/Fortran

使用标准 C++和 Fortran 语言编写的程序现在可以利用 NVIDIA GPU，而不需要依赖任何外部库。这是由于[NVIDIA SDK](https://developer.nvidia.com/hpc-sdk)编译器套件，它可以将代码翻译并优化以在 GPU 上运行。

+   [这里](https://developer.nvidia.com/blog/developing-accelerated-code-with-standard-language-parallelism/)是关于使用标准语言并行化加速的文章系列。

+   可以在[这里](https://developer.nvidia.com/blog/accelerating-standard-c-with-gpus-using-stdpar/)找到编写 C++代码的指南，

+   而 Fortran 代码的指南可以在[这里](https://developer.nvidia.com/blog/accelerating-fortran-do-concurrent-with-gpus-and-the-nvidia-hpc-sdk/)找到。

这两种方法的表现很有希望，如那些指南中提供的示例所示。

## 指令基于编程

一种快速且经济的方法是使用**指令基于**的方法。在这种情况下，现有的**串行**代码被添加了**提示**，指示编译器哪些循环和区域应该在 GPU 上执行。如果没有 API，指令将被视为注释，代码将像通常的串行代码一样执行。这种方法侧重于生产力和易用性（但可能损害性能），通过向现有代码添加并行性而不需要编写特定于加速器的代码，可以以最小的编程工作量使用加速器。使用指令编程有两种常见方法，即**OpenACC**和**OpenMP**。

### OpenACC

[OpenACC](https://www.openacc.org/)是由 2010 年成立的一个联盟开发的，旨在为加速器（包括 GPU）开发一个标准、可移植和可扩展的编程模型。OpenACC 联盟的成员包括 GPU 供应商，如 NVIDIA 和 AMD，以及领先的超级计算中心、大学和软件公司。直到最近，它只支持 NVIDIA GPU，但现在有努力支持更多设备和架构。

### OpenMP

[OpenMP](https://www.openmp.org/)最初是一个多平台、共享内存并行编程 API，用于多核 CPU，最近也增加了对 GPU 卸载的支持。OpenMP 旨在支持各种类型的 GPU，这是通过父编译器实现的。

指令基于的方法与 C/C++和 FORTRAN 代码一起工作，而一些第三方扩展可用于其他语言。

### OpenACC

[OpenACC](https://www.openacc.org/)是由 2010 年成立的一个联盟开发的，该联盟的目标是开发一个标准、可移植和可扩展的编程模型，用于加速器，包括 GPU。OpenACC 联盟的成员包括 GPU 供应商，如 NVIDIA 和 AMD，以及领先的超级计算中心、大学和软件公司。直到最近，它只支持 NVIDIA GPU，但现在正在努力支持更多设备和架构。

### OpenMP

[OpenMP](https://www.openmp.org/)最初是一个针对多平台、共享内存并行编程 API，用于多核 CPU，并且最近增加了对 GPU 卸载的支持。OpenMP 旨在支持各种类型的 GPU，这是通过父编译器实现的。

基于指令的方法适用于 C/C++和 FORTRAN 代码，同时还有一些第三方扩展适用于其他语言。

## 不便携的基于内核的模型（原生编程模型）

在进行直接 GPU 编程时，开发者可以通过编写直接与 GPU 及其硬件通信的低级代码来获得很高的控制权。理论上，直接 GPU 编程方法提供了编写低级、GPU 加速代码的能力，这可以在性能上显著优于仅使用 CPU 的代码。然而，这也需要更深入地了解 GPU 架构及其功能，以及所使用的特定编程方法。

### CUDA

[CUDA](https://developer.nvidia.com/cuda-toolkit)是由 NVIDIA 开发的一个并行计算平台和 API。它历史上是第一个主流 GPU 编程框架。它允许开发者编写在 GPU 上执行的类似 C 的代码。CUDA 提供了一套用于低级 GPU 编程的库和工具，并为计算密集型应用提供了性能提升。虽然有一个广泛的生态系统，但 CUDA 仅限于 NVIDIA 硬件。

### HIP

[HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/what_is_hip.html)（异构接口用于可移植性）是由 AMD 开发的一个 API，它为 GPU 编程提供了一个低级接口。HIP 旨在提供可以在 NVIDIA 和 AMD GPU 上使用的单一源代码。它基于 CUDA 编程模型，并提供与 CUDA 几乎相同的编程接口。

在此存储库的[content/examples/cuda-hip](https://github.com/ENCCS/gpu-programming/tree/main/content/examples/cuda-hip)目录中提供了多个 CUDA/HIP 代码示例。

### CUDA

[CUDA](https://developer.nvidia.com/cuda-toolkit) 是由 NVIDIA 开发的并行计算平台和 API。它在历史上是第一个主流的 GPU 编程框架。它允许开发者编写类似于 C 的代码，这些代码在 GPU 上执行。CUDA 为低级 GPU 编程提供了一套库和工具，并为计算密集型应用程序提供了性能提升。虽然有一个广泛的生态系统，但 CUDA 仅限于 NVIDIA 硬件。

### HIP

[HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/what_is_hip.html)（异构接口可移植性）是由 AMD 开发的一个 API，它为 GPU 编程提供了一个低级接口。HIP 设计用于提供可以在 NVIDIA 和 AMD GPU 上使用的单一源代码。它基于 CUDA 编程模型，并提供与 CUDA 几乎相同的编程接口。

在此存储库的 [content/examples/cuda-hip](https://github.com/ENCCS/gpu-programming/tree/main/content/examples/cuda-hip) 目录中提供了多个 CUDA/HIP 代码示例。

## 可移植的基于内核的模型（跨平台可移植性生态系统）

跨平台可移植性生态系统通常提供更高层次的抽象层，这使得 GPU 编程具有方便且可移植的编程模型。它们可以帮助减少维护和部署 GPU 加速应用程序所需的时间和精力。这些生态系统的目标是实现单源应用程序的性能可移植性。在 C++ 中，最著名的跨平台可移植性生态系统是 [SYCL](https://www.khronos.org/sycl/)、[OpenCL](https://www.khronos.org/opencl/)（C 和 C++ API）和 [Kokkos](https://github.com/kokkos/kokkos)；其他还包括 [alpaka](https://alpaka.readthedocs.io/) 和 [RAJA](https://github.com/LLNL/RAJA)。

### OpenCL

[OpenCL](https://www.khronos.org/opencl/)（开放计算语言）是一个跨平台的开放标准 API，用于在 CPU、GPU 和 FPGA 上进行通用并行计算。它支持来自多个供应商的广泛硬件。OpenCL 为 GPU 编程提供了一个低级编程接口，并允许开发者编写可以在各种平台上执行的程序。与 CUDA、HIP、Kokkos 和 SYCL 等编程模型不同，OpenCL 使用单独的源模型。OpenCL 标准的最新版本增加了对 API 和内核代码的 C++ 支持，但基于 C 的接口仍然更广泛地使用。OpenCL 工作组不提供任何自己的框架。相反，生产 OpenCL 兼容设备的供应商作为其软件开发套件（SDK）的一部分发布框架。最受欢迎的两个 OpenCL SDK 分别由 NVIDIA 和 AMD 发布。在两种情况下，开发套件都是免费的，并包含构建 OpenCL 应用程序所需的库和工具。

### Kokkos

[Kokkos](https://github.com/kokkos/kokkos) 是一个开源的性能可移植编程模型，主要用于异构并行计算，主要在桑迪亚国家实验室开发。它是一个基于 C++的生态系统，为在多核架构（如 CPU、GPU 和 FPGA）上运行的高效和可扩展的并行应用程序提供编程模型。Kokkos 生态系统包括几个组件，如 Kokkos 核心库，它提供并行执行和内存抽象；Kokkos 内核库，它提供线性代数和图算法的数学内核；以及 Kokkos 工具库，它提供分析和调试工具。Kokkos 组件与其他软件库和技术（如 MPI 和 OpenMP）集成良好。此外，该项目与其他项目合作，以提供可移植 C++编程的互操作性和标准化。

### alpaka

[alpaka](https://alpaka.readthedocs.io/)（并行内核加速抽象库）是一个开源的仅包含头文件的 C++库，旨在通过抽象并行性的底层级别，实现跨异构加速器架构的性能可移植性。该库是平台无关的，并支持多个设备的并发和协作使用，包括主机 CPU（x86、ARM、RISC-V）和来自不同供应商的 GPU（NVIDIA、AMD 和 Intel）。

alpaka 的一个关键优势是它只需要一个用户内核的单个实现，该实现以具有标准化接口的函数对象的形式表达。这消除了为不同后端编写专用代码的需要。该库提供各种加速器后端，包括 CUDA、HIP、SYCL、OpenMP 和串行执行，可以根据目标设备进行选择。此外，甚至可以将多个加速器后端组合起来，以在单个应用程序中针对不同供应商的硬件。

### SYCL

[SYCL](https://www.khronos.org/sycl/) 是一个免版税的开放标准 C++编程模型，用于多设备编程。它为包括 GPU 在内的异构系统提供了一种高级、单源编程模型。最初，SYCL 是在 OpenCL 之上开发的；然而，它不再仅限于这一点。它可以在其他低级异构计算 API 之上实现，例如 CUDA 或 HIP，使开发者能够编写可在各种平台上执行的程序。请注意，虽然 SYCL 是一个相对高级的模型，但开发者仍然需要明确编写 GPU 内核。

虽然 alpaka、Kokkos 和 RAJA 指的是特定的项目，但 SYCL 本身只是一个标准，存在多个实现。对于 GPU 编程，[Intel oneAPI DPC++](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html)（原生支持 Intel GPU，以及通过[Codeplay oneAPI 插件](https://codeplay.com/solutions/oneapi/)支持 NVIDIA 和 AMD GPU）和[AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp/)（之前也称为 hipSYCL 或 Open SYCL，支持 NVIDIA 和 AMD GPU，与 Intel oneAPI DPC++结合使用时提供实验性的 Intel GPU 支持）是最广泛使用的。其他值得注意的实现包括[triSYCL](https://github.com/triSYCL/triSYCL)和[ComputeCPP](https://developer.codeplay.com/products/computecpp/ce/home/)。

### OpenCL

[OpenCL](https://www.khronos.org/opencl/)（开放计算语言）是一个跨平台的、开放标准的 API，用于在 CPU、GPU 和 FPGA 上执行通用并行计算。它支持来自多个供应商的广泛硬件。OpenCL 为 GPU 编程提供了一个低级编程接口，并允许开发者编写可在各种平台上执行的程序。与 CUDA、HIP、Kokkos 和 SYCL 等编程模型不同，OpenCL 使用单独的源模型。OpenCL 标准的最新版本增加了对 API 和内核代码的 C++支持，但基于 C 的接口仍然更广泛地被使用。OpenCL 工作组不提供任何自己的框架。相反，生产 OpenCL 兼容设备的供应商将其框架作为其软件开发工具包（SDK）的一部分发布。最受欢迎的两个 OpenCL SDK 是由 NVIDIA 和 AMD 发布的。在两种情况下，开发套件都是免费的，并包含构建 OpenCL 应用程序所需的库和工具。

### Kokkos

[Kokkos](https://github.com/kokkos/kokkos)是一个开源的性能可移植编程模型，用于异构并行计算，主要在桑迪亚国家实验室开发。它是一个基于 C++的生态系统，为在 CPU、GPU 和 FPGA 等多核架构上运行的高效和可扩展的并行应用程序提供编程模型。Kokkos 生态系统包括几个组件，例如 Kokkos 核心库，它提供并行执行和内存抽象；Kokkos 内核库，它提供线性代数和图算法的数学内核；以及 Kokkos 工具库，它提供分析和调试工具。Kokkos 组件与其他软件库和技术（如 MPI 和 OpenMP）集成良好。此外，该项目与其他项目合作，以提供可移植 C++编程的互操作性和标准化。

### alpaka

[alpaka](https://alpaka.readthedocs.io/)（并行内核加速抽象库）是一个开源的仅包含头文件的 C++ 库，旨在通过抽象并行性的底层级别，实现跨异构加速器架构的性能可移植性。该库是平台无关的，并支持包括主机 CPU（x86、ARM、RISC-V）和来自不同供应商的 GPU（NVIDIA、AMD 和 Intel）在内的多个设备的并发和协作使用。

alpaka 的一个关键优势是它只需要一个用户内核的单个实现，该实现以具有标准化接口的函数对象的形式表达。这消除了为不同后端编写专用代码的需要。该库提供各种加速器后端，包括 CUDA、HIP、SYCL、OpenMP 和串行执行，可以根据目标设备进行选择。此外，甚至可以将多个加速器后端组合起来，以在单个应用程序中针对不同供应商的硬件。

### SYCL

[SYCL](https://www.khronos.org/sycl/) 是一个免费的、开放的 C++ 多设备编程编程模型标准。它为包括 GPU 在内的异构系统提供了一种高级、单源编程模型。最初，SYCL 是在 OpenCL 的基础上开发的；然而，它不再仅限于这一点。它可以在其他低级异构计算 API 上实现，如 CUDA 或 HIP，使开发者能够编写可在各种平台上执行的程序。请注意，虽然 SYCL 是一个相对高级的模型，但开发者仍然需要显式编写 GPU 内核。

虽然 alpaka、Kokkos 和 RAJA 指的是特定的项目，但 SYCL 本身只是一个标准，目前存在多个实现。对于 GPU 编程，[Intel oneAPI DPC++](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html)（原生支持 Intel GPU，以及通过 [Codeplay oneAPI 插件](https://codeplay.com/solutions/oneapi/) 支持的 NVIDIA 和 AMD GPU）和 [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp/)（之前也称为 hipSYCL 或 Open SYCL，支持 NVIDIA 和 AMD GPU，与 Intel oneAPI DPC++ 结合使用时提供实验性的 Intel GPU 支持）是最广泛使用的。其他值得注意的实现包括 [triSYCL](https://github.com/triSYCL/triSYCL) 和 [ComputeCPP](https://developer.codeplay.com/products/computecpp/ce/home/)。

## 高级语言支持

### Python

Python 通过多个抽象级别提供对 GPU 编程的支持。

**CUDA Python，HIP Python 和 PyCUDA**

这些项目分别是分别为 [NVIDIA-](https://developer.nvidia.com/cuda-python)、[AMD-](https://rocm.docs.amd.com/projects/hip-python/en/latest/) 和 [社区支持的](https://documen.tician.de/pycuda/)包装器，提供对低级 CUDA 和 HIP API 的 Python 绑定。要直接使用这些方法，在大多数情况下需要了解 CUDA 或 HIP 编程。

CUDA Python 也旨在支持更高层次的工具包和库，例如 **CuPy** 和 **Numba**。

**CuPy**

[CuPy](https://cupy.dev/) 是一个与 NumPy/SciPy 兼容的 GPU 数据数组库。它提供了与 NumPy 和 SciPy 高度相似的用户界面，使得开发者可以轻松过渡到 GPU 计算。使用 NumPy 编写的代码通常可以经过最小修改后适应 CuPy；在大多数直接的情况下，人们可能只需在 Python 代码中将‘numpy’和‘scipy’替换为‘cupy’和‘cupyx.scipy’。

**Numba**

[Numba](https://numba.pydata.org/) 是一个开源的 JIT 编译器，它将 Python 和 NumPy 代码的一个子集转换为优化的机器代码。Numba 支持具有 CUDA 功能的 GPU，并能够使用几种不同的语法变体为它们生成代码。2021 年，对 [AMD (ROCm) 支持](https://numba.readthedocs.io/en/stable/release-notes.html#version-0-54-0-19-august-2021) 的上游支持已被终止。然而，截至 2025 年，AMD 已通过 [Numba HIP 包](https://github.com/ROCm/numba-hip) 为 Numba API 添加了下游支持。

### Julia

Julia 通过以下针对所有三个主要供应商的 GPU 的包，为 GPU 编程提供了一级支持：

+   [CUDA.jl](https://cuda.juliagpu.org/stable/) 用于 NVIDIA GPU

+   [AMDGPU.jl](https://amdgpu.juliagpu.org/stable/) 用于 AMD GPU

+   [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl) 用于 Intel GPU

+   [Metal.jl](https://github.com/JuliaGPU/Metal.jl) 用于 Apple M 系列 GPU

`CUDA.jl` 是最成熟的，`AMDGPU.jl` 稍微落后但仍然适用于通用用途，而 `oneAPI.jl` 和 `Metal.jl` 虽然功能齐全但可能包含错误，缺少一些功能，并提供次优性能。然而，它们的相应 API 是完全类似的，库之间的转换简单直接。

所有包都提供高级抽象，需要非常少的编程工作，以及一个较低级别的编写内核的方法，以实现细粒度控制。

简而言之

+   **基于指令的编程：**

    +   现有的串行代码通过指令进行注释，以指示哪些部分应该在 GPU 上执行。

    +   OpenACC 和 OpenMP 是常见的基于指令的编程模型。

    +   优先考虑生产力和易用性，而不是性能。

    +   在现有代码中添加并行性需要最少的编程工作。

+   **不可移植的基于内核的模型：**

    +   低级代码编写用于直接与 GPU 通信。

    +   CUDA 是 NVIDIA 的并行计算平台和 GPU 编程的 API。

    +   HIP 是 AMD 开发的一个 API，为 NVIDIA 和 AMD GPU 提供与 CUDA 类似的编程接口。

    +   需要更深入地理解 GPU 架构和编程方法。

+   **可移植的基于内核的模型：**

    +   提供更高层次抽象的 GPU 编程，以实现可移植性。

    +   包括 OpenCL、Kokkos、alpaka、RAJA 和 SYCL 等示例。

    +   力求通过单个源应用程序实现性能可移植性。

    +   可在多种 GPU 和平台上运行，减少了维护和部署 GPU 加速应用程序所需的努力。

+   **高级语言支持：**

    +   C++ 和 Fortran 通过语言标准并行性支持 GPU。

    +   Python 库如 PyCUDA、CuPy 和 Numba 提供了 GPU 编程能力。

    +   Julia 有 CUDA.jl、AMDGPU.jl、oneAPI.jl 和 Metal.jl 等用于 GPU 编程的包。

    +   这些方法为各自语言中的 GPU 编程提供了高级抽象和接口。

### Python

Python 通过多个抽象级别提供对 GPU 编程的支持。

**CUDA Python，HIP Python 和 PyCUDA**

这些项目分别是 [NVIDIA-](https://developer.nvidia.com/cuda-python)、[AMD-](https://rocm.docs.amd.com/projects/hip-python/en/latest/) 和 [社区支持](https://documen.tician.de/pycuda/) 的包装器，提供对低级 CUDA 和 HIP API 的 Python 绑定。要直接使用这些方法，在大多数情况下需要了解 CUDA 或 HIP 编程。

CUDA Python 还旨在支持更高级的工具包和库，如 **CuPy** 和 **Numba**。

**CuPy**

[CuPy](https://cupy.dev/) 是一个基于 GPU 的数据数组库，与 NumPy/SciPy 兼容。它提供了与 NumPy 和 SciPy 非常相似的接口，使得开发者可以轻松过渡到 GPU 计算。使用 NumPy 编写的代码通常可以经过最小修改后适应 CuPy；在大多数直接情况下，人们可能只需在他们的 Python 代码中将‘numpy’和‘scipy’替换为‘cupy’和‘cupyx.scipy’。

**Numba**

[Numba](https://numba.pydata.org/) 是一个开源的 JIT 编译器，可以将 Python 和 NumPy 代码的子集转换为优化的机器代码。Numba 支持 CUDA 兼容的 GPU，并能够使用几种不同的语法变体为它们生成代码。到 2021 年，对 [AMD (ROCm) 支持](https://numba.readthedocs.io/en/stable/release-notes.html#version-0-54-0-19-august-2021)的上游支持已被终止。然而，截至 2025 年，AMD 通过 [Numba HIP 包](https://github.com/ROCm/numba-hip)为 Numba API 添加了下游支持。

### Julia

Julia 通过以下针对所有三个主要供应商的 GPU 的包，提供了对 GPU 编程的一流支持：

+   [CUDA.jl](https://cuda.juliagpu.org/stable/) 用于 NVIDIA GPU

+   [AMDGPU.jl](https://amdgpu.juliagpu.org/stable/) 用于 AMD GPU

+   [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl) 用于 Intel GPU

+   [Metal.jl](https://github.com/JuliaGPU/Metal.jl) 用于 Apple M 系列 GPU

`CUDA.jl` 是最成熟的，`AMDGPU.jl` 稍微落后但仍然适用于通用用途，而 `oneAPI.jl` 和 `Metal.jl` 虽然功能齐全但可能存在错误，缺少一些功能，并提供次优性能。然而，它们的相应 API 完全类似，库之间的转换简单直接。

所有包都提供了高级抽象，需要极少的编程努力，以及一个较低级别的编写内核的方法，以实现细粒度控制。

简而言之

+   **基于指令的编程：**

    +   现有的串行代码通过指令进行注释，以指示应在 GPU 上执行哪些部分。

    +   OpenACC 和 OpenMP 是常见的基于指令的编程模型。

    +   优先考虑生产力和易用性，而不是性能。

    +   将并行性添加到现有代码中所需的编程工作量最小。

+   **不可移植的基于内核的模型：**

    +   低级代码是直接与 GPU 通信编写的。

    +   CUDA 是 NVIDIA 的并行计算平台和 GPU 编程 API。

    +   HIP 是 AMD 开发的一个 API，为 NVIDIA 和 AMD GPU 提供了与 CUDA 类似的编程接口。

    +   需要更深入理解 GPU 架构和编程方法。

+   **可移植的基于内核的模型：**

    +   提供可移植性的 GPU 编程的高级抽象。

    +   例如，OpenCL、Kokkos、alpaka、RAJA 和 SYCL。

    +   力求通过单源应用程序实现性能可移植性。

    +   可在多种 GPU 和平台上运行，从而减少了维护和部署 GPU 加速应用程序所需的工作量。

+   **高级语言支持：**

    +   C++和 Fortran 都有支持 GPU 的语言标准并行的计划。

    +   Python 库如 PyCUDA、CuPy 和 Numba 提供了 GPU 编程功能。

    +   Julia 有 CUDA.jl、AMDGPU.jl、oneAPI.jl 和 Metal.jl 等用于 GPU 编程的包。

    +   这些方法为各自语言中的 GPU 编程提供了高级抽象和接口。

## 概述

这些 GPU 编程环境各有其优势和劣势，对于特定项目最佳选择将取决于一系列因素，包括：

+   目标硬件平台，

+   正在执行的计算类型，以及

+   开发者的经验和偏好。

**高级且以生产力为导向的 API**提供了一个简化的编程模型，并最大化代码的可移植性，而**低级且以性能为导向的 API**提供了对 GPU 硬件的高级别控制，但也需要更多的编码努力和专业知识。

## 练习

讨论

+   你之前是否已经使用过任何 GPU 编程框架？

+   你认为最具有挑战性的是什么？最有用的是什么？

请在主房间或通过 HackMD 文档中的评论告诉我们。

重点

+   GPU 编程方法可以分为 1)基于指令的，2)不可移植的基于内核的，3)可移植的基于内核的，和 4)高级语言支持。

+   每种方法都有多个框架/语言可供选择，各有优缺点。

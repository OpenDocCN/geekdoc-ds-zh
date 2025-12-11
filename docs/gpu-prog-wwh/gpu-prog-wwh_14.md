# 准备代码以进行 GPU 移植

> 原文：[`enccs.github.io/gpu-programming/11-gpu-porting/`](https://enccs.github.io/gpu-programming/11-gpu-porting/)

*GPU 编程：为什么、何时以及如何？](../)* **   准备代码以进行 GPU 移植

+   [在 GitHub 上编辑](https://github.com/ENCCS/gpu-programming/blob/main/content/11-gpu-porting.rst)

* * *

问题

+   识别利用 GPU 并行处理能力的代码移植的关键步骤是什么？

+   我如何识别代码中可以受益于 GPU 加速的计算密集部分？

+   重构循环以适应 GPU 架构并改进内存访问模式时需要考虑哪些因素？

+   有没有工具可以在不同的框架之间自动翻译？

目标

+   熟悉将代码移植到 GPU 以利用并行处理能力的步骤。

+   提供一些关于重构循环和修改操作以适应 GPU 架构并改进内存访问模式的想法。

+   学习使用自动翻译工具将 CUDA 转换为 HIP 以及将 OpenACC 转换为 OpenMP

讲师备注

+   30 分钟教学

+   20 分钟练习

## 从 CPU 到 GPU 的移植

当将代码移植以利用 GPU 的并行处理能力时，需要遵循几个步骤，并在编写实际在 GPU 上执行的并行代码之前做一些额外的工作：

+   **识别目标部分**：首先识别对执行时间贡献显著的代码部分。这些通常是计算密集的部分，如循环或矩阵运算。帕累托原则表明，大约 10-20%的代码占用了 80-90%的执行时间。

+   **等效 GPU 库**：如果原始代码使用 CPU 库如 BLAS、FFT 等，识别等效的 GPU 库至关重要。例如，cuBLAS 或 hipBLAS 可以替换基于 CPU 的 BLAS 库。利用 GPU 特定的库确保高效的 GPU 利用。

+   **重构循环**：当直接将循环移植到 GPU 时，需要进行一些重构以适应 GPU 架构。这通常涉及将循环拆分为多个步骤或修改操作以利用迭代之间的独立性并改进内存访问模式。原始循环的每一步都可以映射到一个内核，由多个 GPU 线程执行，每个线程对应一个迭代。

+   **内存访问优化**：考虑代码中的内存访问模式。当内存访问是归一化和对齐时，GPU 表现最佳。最小化全局内存访问并最大化共享内存或寄存器的利用率可以显著提高性能。审查代码以确保 GPU 执行的最优内存访问。

### 讨论

> 这将如何移植？（n_soap ≈ 100，n_sites ⩾ 10000，k_max ≈ 20*n_sites）
> 
> > 检查以下 Fortran 代码（如果你不读 Fortran：do-loops == for-loops）
> > 
> > ```
> > k2  =  0
> > do i  =  1,  n_sites
> >   do j  =  1,  n_neigh(i)
> >   k2  =  k2  +  1
> >   counter  =  0
> >   counter2  =  0
> >   do n  =  1,  n_max
> >   do np  =  n,  n_max
> >   do l  =  0,  l_max
> >   if(  skip_soap_component(l,  np,  n)  )cycle
> > 
> >   counter  =  counter+1
> >   do m  =  0,  l
> >   k  =  1  +  l*(l+1)/2  +  m
> >   counter2  =  counter2  +  1
> >   multiplicity  =  multiplicity_array(counter2)
> >   soap_rad_der(counter,  k2)  =  soap_rad_der(counter,  k2)  +  multiplicity  *  real(  cnk_rad_der(k,  n,  k2)  *  conjg(cnk(k,  np,  i))  +  cnk(k,  n,  i)  *  conjg(cnk_rad_der(k,  np,  k2))  )
> >   soap_azi_der(counter,  k2)  =  soap_azi_der(counter,  k2)  +  multiplicity  *  real(  cnk_azi_der(k,  n,  k2)  *  conjg(cnk(k,  np,  i))  +  cnk(k,  n,  i)  *  conjg(cnk_azi_der(k,  np,  k2))  )
> >   soap_pol_der(counter,  k2)  =  soap_pol_der(counter,  k2)  +  multiplicity  *  real(  cnk_pol_der(k,  n,  k2)  *  conjg(cnk(k,  np,  i))  +  cnk(k,  n,  i)  *  conjg(cnk_pol_der(k,  np,  k2))  )
> >   end do
> >  end do
> >  end do
> >  end do
> > 
> >   soap_rad_der(1:n_soap,  k2)  =  soap_rad_der(1:n_soap,  k2)  /  sqrt_dot_p(i)  -  soap(1:n_soap,  i)  /  sqrt_dot_p(i)**3  *  dot_product(  soap(1:n_soap,  i),  soap_rad_der(1:n_soap,  k2)  )
> >   soap_azi_der(1:n_soap,  k2)  =  soap_azi_der(1:n_soap,  k2)  /  sqrt_dot_p(i)  -  soap(1:n_soap,  i)  /  sqrt_dot_p(i)**3  *  dot_product(  soap(1:n_soap,  i),  soap_azi_der(1:n_soap,  k2)  )
> >   soap_pol_der(1:n_soap,  k2)  =  soap_pol_der(1:n_soap,  k2)  /  sqrt_dot_p(i)  -  soap(1:n_soap,  i)  /  sqrt_dot_p(i)**3  *  dot_product(  soap(1:n_soap,  i),  soap_pol_der(1:n_soap,  k2)  )
> > 
> >   if(  j  ==  1  )then
> >   k3  =  k2
> >   else
> >   soap_cart_der(1,  1:n_soap,  k2)  =  dsin(thetas(k2))  *  dcos(phis(k2))  *  soap_rad_der(1:n_soap,  k2)  -  dcos(thetas(k2))  *  dcos(phis(k2))  /  rjs(k2)  *  soap_pol_der(1:n_soap,  k2)  -  dsin(phis(k2))  /  rjs(k2)  *  soap_azi_der(1:n_soap,  k2)
> >   soap_cart_der(2,  1:n_soap,  k2)  =  dsin(thetas(k2))  *  dsin(phis(k2))  *  soap_rad_der(1:n_soap,  k2)  -  dcos(thetas(k2))  *  dsin(phis(k2))  /  rjs(k2)  *  soap_pol_der(1:n_soap,  k2)  +  dcos(phis(k2))  /  rjs(k2)  *  soap_azi_der(1:n_soap,  k2)
> >   soap_cart_der(3,  1:n_soap,  k2)  =  dcos(thetas(k2))  *  soap_rad_der(1:n_soap,  k2)  +  dsin(thetas(k2))  /  rjs(k2)  *  soap_pol_der(1:n_soap,  k2)
> >   soap_cart_der(1,  1:n_soap,  k3)  =  soap_cart_der(1,  1:n_soap,  k3)  -  soap_cart_der(1,  1:n_soap,  k2)
> >   soap_cart_der(2,  1:n_soap,  k3)  =  soap_cart_der(2,  1:n_soap,  k3)  -  soap_cart_der(2,  1:n_soap,  k2)
> >   soap_cart_der(3,  1:n_soap,  k3)  =  soap_cart_der(3,  1:n_soap,  k3)  -  soap_cart_der(3,  1:n_soap,  k2)
> >   end if
> >  end do
> > end do 
> > ```
> > 
> 一些初步步骤：
> 
> > +   代码可能（必须）拆分为 3-4 个内核。为什么？
> > +   
> > +   检查是否存在可能导致迭代之间产生虚假依赖的变量，例如索引 k2
> > +   
> > +   对于 GPU 来说，将工作分配到索引 i 是否高效？关于内存访问呢？注意 Fortran 中的数组是二维的。
> > +   
> > +   是否可以合并一些循环？合并嵌套循环可以减少开销并改善内存访问模式，从而提高 GPU 性能。
> > +   
> > +   在 GPU 中最佳的内存访问方式是什么？审查代码中的内存访问模式。通过在适当的地方使用共享内存或寄存器来最小化全局内存访问，确保内存访问是归约和齐的，以最大化 GPU 内存吞吐量。

代码重构完成！

+   寄存器数量有限，内核使用越多寄存器，导致活跃线程减少（占用率低）。

+   为了计算 soap_rad_der(is,k2)，CUDA 线程需要访问所有之前的值 soap_rad_der(1:nsoap,k2)。

+   为了计算 soap_cart_der(1, 1:n_soap, k3)，需要访问所有值(k3+1:k2+n_neigh(i))。

+   注意第一部分的索引。矩阵被转置以获得更好的访问模式。

> ```
>  !omp target teams distribute parallel do private (i)
>   do k2  =  1,  k2_max
>   i=list_of_i(k2)
>   counter  =  0
>   counter2  =  0
>   do n  =  1,  n_max
>   do np  =  n,  n_max
>   do l  =  0,  l_max
>   if(  skip_soap_component(l,  np,  n)  )  then
>  cycle
>  endif
>   counter  =  counter+1
>   do m  =  0,  l
>   k  =  1  +  l*(l+1)/2  +  m
>   counter2  =  counter2  +  1
>   multiplicity  =  multiplicity_array(counter2)
>   tsoap_rad_der(k2,counter)  =  tsoap_rad_der(k2,counter)  +  multiplicity  *  real(  tcnk_rad_der(k2,k,n)  *  conjg(tcnk(i,k,np))  +  tcnk(i,k,n)  *  conjg(tcnk_rad_der(k2,k,np))  )
>   tsoap_azi_der(k2,counter)  =  tsoap_azi_der(k2,counter)  +  multiplicity  *  real(  tcnk_azi_der(k2,k,n)  *  conjg(tcnk(i,k,np))  +  tcnk(i,k,n)  *  conjg(tcnk_azi_der(k2,k,np))  )
>   tsoap_pol_der(k2,counter)  =  tsoap_pol_der(k2,counter)  +  multiplicity  *  real(  tcnk_pol_der(k2,k,n)  *  conjg(tcnk(i,k,np))  +  tcnk(i,k,n)  *  conjg(tcnk_pol_der(k2,k,np))  )
>   end do
>  end do
>  end do
>  end do
>  end do
> 
> ! Before the next part the variables are transposed again to their original layout.
> 
>   !omp target teams  distribute private(i)
>   do k2  =  1,  k2_max
>   i=list_of_i(k2)
>   locdot=0.d0
> 
>   !omp parallel do reduction(+:locdot_rad_der,locdot_azi_der,locdot_pol_der)
>   do is=1,nsoap
>   locdot_rad_der=locdot_rad_der+soap(is,  i)  *  soap_rad_der(is,  k2)
>   locdot_azi_der=locdot_azi_der+soap(is,  i)  *  soap_azi_der(is,  k2)
>   locdot_pol_der=locdot_pol_der+soap(is,  i)  *  soap_pol_der(is,  k2)
>   enddo
>   dot_soap_rad_der(k2)=  locdot_rad_der
>   dot_soap_azi_der(k2)=  locdot_azi_der
>   dot_soap_pol_der(k2)=  locdot_pol_der
>   end do
> 
>   !omp target teams distribute
>   do k2  =  1,  k2_max
>   i=list_of_i(k2)
> 
>   !omp parallel do
>   do is=1,nsoap
>   soap_rad_der(is,  k2)  =  soap_rad_der(is,  k2)  /  sqrt_dot_p(i)  -  soap(is,  i)  /  sqrt_dot_p(i)**3  *  dot_soap_rad_der(k2)
>   soap_azi_der(is,  k2)  =  soap_azi_der(is,  k2)  /  sqrt_dot_p(i)  -  soap(is,  i)  /  sqrt_dot_p(i)**3  *  dot_soap_azi_der(k2)
>   soap_pol_der(is,  k2)  =  soap_pol_der(is,  k2)  /  sqrt_dot_p(i)  -  soap(is,  i)  /  sqrt_dot_p(i)**3  *  dot_soap_pol_der(k2)
>   end do
>  end do
> 
>   !omp teams distribute private(k3)
>   do k2  =  1,  k2_max
>   k3=list_k2k3(k2)
> 
>   !omp parallel do private (is)
>   do is=1,n_soap
>   if(  k3  /=  k2)then
>   soap_cart_der(1,  is,  k2)  =  dsin(thetas(k2))  *  dcos(phis(k2))  *  soap_rad_der(1:n_soap,  k2)  -  dcos(thetas(k2))  *  dcos(phis(k2))  /  rjs(k2)  *  soap_pol_der(1:n_soap,  k2)  -  dsin(phis(k2))  /  rjs(k2)  *  soap_azi_der(is,  k2)
>   soap_cart_der(2,  is,  k2)  =  dsin(thetas(k2))  *  dsin(phis(k2))  *  soap_rad_der(1:n_soap,  k2)  -  dcos(thetas(k2))  *  dsin(phis(k2))  /  rjs(k2)  *  soap_pol_der(1:n_soap,  k2)  +  dcos(phis(k2))  /  rjs(k2)  *  soap_azi_der(is,  k2)
>   soap_cart_der(3,  is,  k2)  =  dcos(thetas(k2))  *  soap_rad_der(is,  k2)  +  dsin(thetas(k2))  /  rjs(k2)  *  soap_pol_der(is,  k2)
>   end if
>  end do
>  end do
> 
>   !omp teams distribute private(k3)
>   do i  =  1,  n_sites
>   k3=list_k3(i)
> 
>   !omp parallel do private(is, k2)
>   do is=1,n_soap
>   do k2=k3+1,k3+n_neigh(i)
>   soap_cart_der(1,  is,  k3)  =  soap_cart_der(1,  is,  k3)  -  soap_cart_der(1,  is,  k2)
>   soap_cart_der(2,  is,  k3)  =  soap_cart_der(2,  is,  k3)  -  soap_cart_der(2,  is,  k2)
>   soap_cart_der(3,  is,  k3)  =  soap_cart_der(3,  is,  k3)  -  soap_cart_der(3,  is,  k2)
>   end do
>  end do
>  end do 
> ```

重点

+   确定与基于 CPU 的库等效的 GPU 库，并利用它们以确保高效的 GPU 利用率。

+   识别代码中计算密集部分的重要性，这些部分对执行时间有显著贡献。

+   需要重构循环以适应 GPU 架构。

+   内存访问优化对高效 GPU 执行的重要性，包括归约和齐内存访问模式。

## 不同 GPU 框架之间的移植

你可能还会遇到需要将代码从一个特定的 GPU 框架移植到另一个框架的情况。本节概述了不同的工具，这些工具可以将 CUDA 和 OpenACC 代码分别转换为 HIP 和 OpenMP。此转换过程使应用程序能够针对各种 GPU 架构进行优化，特别是 NVIDIA 和 AMD GPU。在此，我们关注[hipify](https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html)和[clacc](https://csmd.ornl.gov/project/clacc)工具。本指南改编自[NRIS 文档](https://documentation.sigma2.no/code_development/guides/cuda_translating-tools.html)。

### 使用 Hipify 将 CUDA 转换为 HIP

在本节中，我们介绍了使用`hipify-perl`和`hipify-clang`工具将 CUDA 代码翻译为 HIP 的方法。

#### Hipify-perl

`hipify-perl` 工具是一个基于 perl 的脚本，它将 CUDA 语法转换为 HIP 语法（例如，有关更多详细信息，请参阅[此处](https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl)）。例如，在一个包含 CUDA 函数 `cudaMalloc` 和 `cudaDeviceSynchronize` 的 CUDA 代码中，该工具会将 `cudaMalloc` 替换为 HIP 函数 `hipMalloc`。同样，CUDA 函数 `cudaDeviceSynchronize` 也会被替换为 HIP 函数 `hipDeviceSynchronize`。以下是在 LUMI-G 上运行 `hipify-perl` 的基本步骤。

+   **步骤 1**：生成 `hipify-perl` 脚本

    ```
    $ module  load  rocm/6.0.3
    $ hipify-clang  --perl 
    ```

+   **步骤 2**：运行生成的 `hipify-perl`

    ```
    $ hipify-perl  program.cu  >  program.cu.hip 
    ```

+   **步骤 3**：使用 `hipcc` 编译生成的 HIP 代码

    ```
    $ hipcc  --offload-arch=gfx90a  -o  program.hip.exe  program.cu.hip 
    ```

尽管使用 `hipify-perl` 的操作很简单，但该工具可能不适合大型应用程序，因为它严重依赖于将 CUDA 字符串替换为 HIP 字符串（例如，它将 `*cuda*` 替换为 `*hip*`）。此外，`hipify-perl` 缺乏区分设备/主机函数调用的能力。[区分设备/主机函数调用](https://docs.amd.com/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl) 的替代方案是使用下一节将要描述的 `hipify-clang` 工具。

#### Hipify-clang

如 [HIPIFY 文档](https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl) 所述，`hipify-clang` 工具基于 clang，用于将 CUDA 源代码转换为 HIP 源代码。与 `hipify-perl` 工具相比，该工具在转换 CUDA 代码方面更为稳健。此外，它通过提供辅助功能来促进代码分析。

简而言之，`hipify-clang` 需要 `LLVM+CLANG` 和 `CUDA`。有关构建 `hipify-clang` 的详细信息，请参阅[此处](https://github.com/ROCm/HIPIFY)。请注意，`hipify-clang` 在 LUMI-G 上可用。然而，问题可能与 CUDA-toolkit 的安装有关。为了避免安装过程中的任何潜在问题，我们选择使用 CUDA singularity 容器。在此，我们提供了一个逐步指南，用于运行 `hipify-clang`：

+   **步骤 1**：拉取 CUDA singularity 容器，例如

    ```
    $ singularity  pull  docker://nvcr.io/nvidia/cuda:11.4.3-devel-ubuntu20.04 
    ```

+   **步骤 2**：加载 rocm 模块并启动 CUDA singularity

    ```
    $ module  load  rocm/6.0.3
    $ singularity  shell  -B  $PWD,/opt:/opt  cuda_11.4.0-devel-ubuntu20.04.sif 
    ```

    其中，主机当前目录 `$PWD` 被挂载到容器中的相应目录，主机中的 `/opt` 目录被挂载到容器内的相应目录。

+   **步骤 3**：设置环境变量 `$PATH`。为了在容器内运行 `hipify-clang`，可以设置环境变量 `$PATH`，该变量定义了查找二进制文件 `hipify-clang` 的路径。

    ```
    $ export  PATH=/opt/rocm-6.0.3/bin:$PATH 
    ```

    注意，我们使用的 rocm 版本是 `rocm-6.0.3`。

+   **步骤 4**：在 singularity 容器内运行 `hipify-clang`

    ```
    $ hipify-clang  program.cu  -o  hip_program.cu.hip  --cuda-path=/usr/local/cuda-11.4  -I  /usr/local/cuda-11.4/include 
    ```

    在此处，应指定 cuda 路径以及 `*includes*` 和 `*defines*` 文件的路经。CUDA 源代码和生成的输出代码分别是 program.cu 和 hip_program.cu.hip。

    生成的 hip 代码的编译过程语法与上一节中描述的类似（参见 hipify-perl 部分的 **步骤 3**）。

可以通过克隆此仓库来访问 `Hipify` 练习的代码示例，该示例位于 `content/examples/exercise_hipify` 子目录中：

> ```
> $ git  clone  https://github.com/ENCCS/gpu-programming.git
> $ cd  gpu-programming/content/examples/exercise_hipify
> $ ls 
> ```

练习 I：使用 `hipify-perl` 将 CUDA 代码翻译为 HIP

1.1 生成 `hipify-perl` 工具。

1.2 使用 `Hipify-perl` 工具将位于 `/exercise_hipify/Hipify_perl` 的 `vec_add_cuda.cu` CUDA 代码转换为 HIP。

1.3 使用 `hipcc` 编译器包装器编译生成的 HIP 代码并运行它。

练习 II：使用 `hipify-clang` 将 CUDA 代码翻译为 HIP

2.1 使用 `Hipify-clang` 工具将位于 `/exercise_hipify/Hipify_clang` 的 `vec_add_cuda.cu` CUDA 代码转换为 HIP。

2.2 使用 `hipcc` 编译器包装器编译生成的 HIP 代码并运行它。

### 使用 Clacc 将 OpenACC 转换为 OpenMP

[Clacc](https://github.com/llvm-doe-org/llvm-project/tree/clacc/main) 是一个工具，可以将 OpenACC 应用程序转换为使用 Clang/LLVM 编译器环境的 OpenMP 转发。请注意，该工具特定于 OpenACC C，而 OpenACC Fortran 已在 AMD GPU 上得到支持。如 [GitHub 仓库](https://github.com/llvm-doe-org/llvm-project/tree/clacc/main) 中所示，编译器 `Clacc` 是位于 `\install` 目录 `\bin` 子目录中的 `Clang` 可执行文件，如下所述。

在以下内容中，我们提供了一个逐步指南，用于构建和使用 Clacc：

+   **步骤 1**：构建和安装 [Clacc](https://github.com/llvm-doe-org/llvm-project/tree/clacc/main)。

    ```
    $ git  clone  -b  clacc/main  https://github.com/llvm-doe-org/llvm-project.git
    $ cd  llvm-project
    $ mkdir  build  &&  cd  build
    $ cmake  -DCMAKE_INSTALL_PREFIX=../install  \
      -DCMAKE_BUILD_TYPE=Release  \
      -DLLVM_ENABLE_PROJECTS="clang;lld"  \
      -DLLVM_ENABLE_RUNTIMES=openmp  \
      -DLLVM_TARGETS_TO_BUILD="host;AMDGPU"  \
      -DCMAKE_C_COMPILER=gcc  \
      -DCMAKE_CXX_COMPILER=g++  \
      ../llvm
    $ make
    $ make  install 
    ```

+   **步骤 2**：设置环境变量，以便能够从 `/install` 目录工作，这是最简单的方法。我们假设 `/install` 目录位于路径 `/project/project_xxxxxx/Clacc/llvm-project`。

对于更高级的使用，例如修改 `Clacc`，我们建议读者参考 [“从构建目录使用”](https://github.com/llvm-doe-org/llvm-project/blob/clacc/main/README.md)。

> ```
> $ export  PATH=/scratch/project_465002387/clacc/llvm-project/install/bin:$PATH
> $ export  LD_LIBRARY_PATH=/scratch/project_465002387/clacc/llvm-project/install/lib:$LD_LIBRARY_PATH 
> ```

+   **步骤 3**：将 openACC_code.c 代码从源代码转换为要打印到文件 openMP_code.c 的代码：

    ```
    $ clang  -fopenacc-print=omp  -fopenacc-structured-ref-count-omp=no-ompx-hold  openACC_code.c  >  openMP_code.c 
    ```

    这里引入了标志 `-fopenacc-structured-ref-count-omp=no-ompx-hold` 来禁用 `ompx_hold` 映射类型修饰符，该修饰符用于 OpenACC `copy` 子句的翻译。`ompx_hold` 是一个 OpenMP 扩展，可能尚未被其他编译器支持。

+   **步骤 4** 使用 [cc 编译器包装器](https://docs.lumi-supercomputer.eu/development/compiling/prgenv/) 编译代码

    ```
    module load CrayEnv
    module load PrgEnv-cray
    module load craype-accel-amd-gfx90a
    module load rocm/6.0.3

    cc -fopenmp -o executable openMP_code.c 
    ```

访问练习材料

可以通过克隆此仓库来访问 `Clacc` 练习的代码示例，该示例位于 `content/examples/exercise_clacc` 子目录中：

```
$ git  clone  https://github.com/ENCCS/gpu-programming.git
$ cd  gpu-programming/content/examples/exercise_clacc
$ ls 
```

练习：将 OpenACC 代码转换为 OpenMP

1.  使用 `Clacc` 编译器将位于 `/exercise_clacc` 的 `openACC_code.c` 代码转换为 OpenACC 代码。

1.  使用 `cc` 编译器包装器编译生成的 OpenMP 代码并运行它。

### 使用 SYCLomatic 将 CUDA 转换为 SYCL/DPC++

英特尔提供了一种 CUDA 到 SYCL 代码迁移的工具，包含在 Intel oneAPI Basekit 中。

它没有安装在 LUMI 上，但一般工作流程与 HIPify Clang 类似，也要求现有的 CUDA 安装：

> ```
> $ dpct  program.cu
> $ cd  dpct_output/
> $ icpx  -fsycl  program.dp.cpp 
> ```

SYCLomatic 可以通过使用`-in-root`和`-out-root`标志递归地处理目录来迁移较大的项目。它还可以使用编译数据库（由 CMake 和其他构建系统支持）来处理更复杂的项目布局。

请注意，由 SYCLomatic 生成的代码依赖于 oneAPI 特定的扩展，因此不能直接与其他 SYCL 实现（如 AdaptiveCpp（hipSYCL））一起使用。可以通过将`--no-incremental-migration`标志添加到`dpct`命令中来最小化，但无法完全避免使用此兼容层。这需要手动操作，因为某些 CUDA 概念不能直接映射到 SYCL。

此外，CUDA 应用程序可能假设某些硬件行为，例如 32 宽的 warp。如果目标硬件不同（例如，LUMI 中使用的 AMD MI250 GPU 具有 64 个 warp 大小），则算法可能需要手动调整。

### 结论

这简要概述了使用现有工具将 CUDA 代码转换为 HIP 和 SYCL，以及将 OpenACC 代码转换为 OpenMP 卸载的用法。一般来说，大型应用程序的翻译过程可能不完整，因此需要手动修改以完成移植过程。然而，值得注意的是，翻译过程的准确性要求应用程序根据 CUDA 和 OpenACC 语法正确编写。

## 参考阅读

+   [Hipify GitHub](https://github.com/ROCm/HIPIFY)

+   [HIPify 参考指南 v5.1](https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html)

+   [HIP 示例](https://github.com/olcf-tutorials/simple_HIP_examples/tree/master/vector_addition)

+   [将 CUDA 移植到 HIP](https://www.admin-magazine.com/HPC/Articles/Porting-CUDA-to-HIP)

+   [Clacc 主仓库 README](https://github.com/llvm-doe-org/llvm-project/blob/clacc/main/README.md)

+   [SYCLomatic 主页面](https://www.intel.com/content/www/us/en/developer/articles/technical/syclomatic-new-cuda-to-sycl-code-migration-tool.html)

+   [SYCLomatic 文档](https://oneapi-src.github.io/SYCLomatic/get_started/index.html)

重点

+   存在着一些有用的工具，可以自动将工具从 CUDA 转换为 HIP 和 SYCL，以及从 OpenACC 转换为 OpenMP，但它们可能需要手动修改。上一页 下一页

* * *

© 版权所有 2023-2024，贡献者。

使用[Sphinx](https://www.sphinx-doc.org/)和由[Read the Docs](https://readthedocs.org)提供的[主题](https://github.com/readthedocs/sphinx_rtd_theme)构建。问题

+   将代码移植以利用 GPU 并行处理能力的关键步骤有哪些？

+   我如何识别代码中可以受益于 GPU 加速的计算密集型部分？

+   重构循环以适应 GPU 架构并改进内存访问模式时需要考虑哪些因素？

+   有没有工具可以自动在不同框架之间进行翻译？

目标

+   熟悉将代码移植到 GPU 以利用并行处理能力的步骤。

+   提供一些关于重构循环和修改操作以适应 GPU 架构并改进内存访问模式的想法。

+   学习使用自动翻译工具将 CUDA 移植到 HIP 以及将 OpenACC 移植到 OpenMP

教师备注

+   30 分钟教学

+   20 分钟练习

## 从 CPU 移植到 GPU

在将代码移植以利用 GPU 的并行处理能力时，需要遵循几个步骤，并在实际编写要在 GPU 上执行的并行代码之前做一些额外的工作：

+   **识别目标部分**：首先识别对执行时间贡献显著的代码部分。这些通常是计算密集型部分，如循环或矩阵运算。帕累托原则表明，大约 10-20% 的代码占用了 80-90% 的执行时间。

+   **等效 GPU 库**：如果原始代码使用 CPU 库如 BLAS、FFT 等，识别等效的 GPU 库至关重要。例如，cuBLAS 或 hipBLAS 可以替换基于 CPU 的 BLAS 库。利用 GPU 特定的库可以确保高效的 GPU 利用。

+   **重构循环**：在直接将循环移植到 GPU 时，需要进行一些重构以适应 GPU 架构。这通常涉及将循环拆分成多个步骤或修改操作以利用迭代之间的独立性并改进内存访问模式。原始循环的每个步骤都可以映射到一个内核，由多个 GPU 线程执行，每个线程对应一个迭代。

+   **内存访问优化**：考虑代码中的内存访问模式。GPU 在内存访问合并和对齐时表现最佳。最小化全局内存访问并最大化共享内存或寄存器的利用率可以显著提高性能。审查代码以确保 GPU 执行的内存访问最优。

### 讨论

> 这将如何移植？（n_soap ≈ 100，n_sites ⩾ 10000，k_max ≈ 20*n_sites）
> 
> > 检查以下 Fortran 代码（如果你不读 Fortran：do-loops == for-loops）
> > 
> > ```
> > k2  =  0
> > do i  =  1,  n_sites
> >   do j  =  1,  n_neigh(i)
> >   k2  =  k2  +  1
> >   counter  =  0
> >   counter2  =  0
> >   do n  =  1,  n_max
> >   do np  =  n,  n_max
> >   do l  =  0,  l_max
> >   if(  skip_soap_component(l,  np,  n)  )cycle
> > 
> >   counter  =  counter+1
> >   do m  =  0,  l
> >   k  =  1  +  l*(l+1)/2  +  m
> >   counter2  =  counter2  +  1
> >   multiplicity  =  multiplicity_array(counter2)
> >   soap_rad_der(counter,  k2)  =  soap_rad_der(counter,  k2)  +  multiplicity  *  real(  cnk_rad_der(k,  n,  k2)  *  conjg(cnk(k,  np,  i))  +  cnk(k,  n,  i)  *  conjg(cnk_rad_der(k,  np,  k2))  )
> >   soap_azi_der(counter,  k2)  =  soap_azi_der(counter,  k2)  +  multiplicity  *  real(  cnk_azi_der(k,  n,  k2)  *  conjg(cnk(k,  np,  i))  +  cnk(k,  n,  i)  *  conjg(cnk_azi_der(k,  np,  k2))  )
> >   soap_pol_der(counter,  k2)  =  soap_pol_der(counter,  k2)  +  multiplicity  *  real(  cnk_pol_der(k,  n,  k2)  *  conjg(cnk(k,  np,  i))  +  cnk(k,  n,  i)  *  conjg(cnk_pol_der(k,  np,  k2))  )
> >   end do
> >  end do
> >  end do
> >  end do
> > 
> >   soap_rad_der(1:n_soap,  k2)  =  soap_rad_der(1:n_soap,  k2)  /  sqrt_dot_p(i)  -  soap(1:n_soap,  i)  /  sqrt_dot_p(i)**3  *  dot_product(  soap(1:n_soap,  i),  soap_rad_der(1:n_soap,  k2)  )
> >   soap_azi_der(1:n_soap,  k2)  =  soap_azi_der(1:n_soap,  k2)  /  sqrt_dot_p(i)  -  soap(1:n_soap,  i)  /  sqrt_dot_p(i)**3  *  dot_product(  soap(1:n_soap,  i),  soap_azi_der(1:n_soap,  k2)  )
> >   soap_pol_der(1:n_soap,  k2)  =  soap_pol_der(1:n_soap,  k2)  /  sqrt_dot_p(i)  -  soap(1:n_soap,  i)  /  sqrt_dot_p(i)**3  *  dot_product(  soap(1:n_soap,  i),  soap_pol_der(1:n_soap,  k2)  )
> > 
> >   if(  j  ==  1  )then
> >   k3  =  k2
> >   else
> >   soap_cart_der(1,  1:n_soap,  k2)  =  dsin(thetas(k2))  *  dcos(phis(k2))  *  soap_rad_der(1:n_soap,  k2)  -  dcos(thetas(k2))  *  dcos(phis(k2))  /  rjs(k2)  *  soap_pol_der(1:n_soap,  k2)  -  dsin(phis(k2))  /  rjs(k2)  *  soap_azi_der(1:n_soap,  k2)
> >   soap_cart_der(2,  1:n_soap,  k2)  =  dsin(thetas(k2))  *  dsin(phis(k2))  *  soap_rad_der(1:n_soap,  k2)  -  dcos(thetas(k2))  *  dsin(phis(k2))  /  rjs(k2)  *  soap_pol_der(1:n_soap,  k2)  +  dcos(phis(k2))  /  rjs(k2)  *  soap_azi_der(1:n_soap,  k2)
> >   soap_cart_der(3,  1:n_soap,  k2)  =  dcos(thetas(k2))  *  soap_rad_der(1:n_soap,  k2)  +  dsin(thetas(k2))  /  rjs(k2)  *  soap_pol_der(1:n_soap,  k2)
> >   soap_cart_der(1,  1:n_soap,  k3)  =  soap_cart_der(1,  1:n_soap,  k3)  -  soap_cart_der(1,  1:n_soap,  k2)
> >   soap_cart_der(2,  1:n_soap,  k3)  =  soap_cart_der(2,  1:n_soap,  k3)  -  soap_cart_der(2,  1:n_soap,  k2)
> >   soap_cart_der(3,  1:n_soap,  k3)  =  soap_cart_der(3,  1:n_soap,  k3)  -  soap_cart_der(3,  1:n_soap,  k2)
> >   end if
> >  end do
> > end do 
> > ```
> > 
> 一些初步步骤：
> 
> > +   代码可以（必须）分成 3-4 个内核。为什么？
> > +   
> > +   检查是否存在可能导致迭代之间产生虚假依赖的变量，如索引 k2
> > +   
> > +   对于 GPU 来说，将工作分配到索引 i 上是否高效？关于内存访问呢？注意 Fortran 中的数组是二维的。
> > +   
> > +   是否可以合并一些循环？合并嵌套循环可以减少开销并改进内存访问模式，从而提高 GPU 性能。
> > +   
> > +   在 GPU 中最佳的内存访问方式是什么？审查代码中的内存访问模式。通过在适当的地方使用共享内存或寄存器来最小化全局内存访问。确保内存访问是归一化和对齐的，以最大化 GPU 内存吞吐量

重构后的代码！

+   寄存器数量有限，内核使用越多寄存器，就会导致更少的活跃线程（低占用率）。

+   为了计算`soap_rad_der(is,k2)`，CUDA 线程需要访问所有之前计算的值`soap_rad_der(1:nsoap,k2)`。

+   为了计算`soap_cart_der(1, 1:n_soap, k3)`，需要访问所有值（`k3+1:k2+n_neigh(i)`）。

+   注意第一部分中的索引。矩阵被转置以获得更好的访问模式。

> ```
>  !omp target teams distribute parallel do private (i)
>   do k2  =  1,  k2_max
>   i=list_of_i(k2)
>   counter  =  0
>   counter2  =  0
>   do n  =  1,  n_max
>   do np  =  n,  n_max
>   do l  =  0,  l_max
>   if(  skip_soap_component(l,  np,  n)  )  then
>  cycle
>  endif
>   counter  =  counter+1
>   do m  =  0,  l
>   k  =  1  +  l*(l+1)/2  +  m
>   counter2  =  counter2  +  1
>   multiplicity  =  multiplicity_array(counter2)
>   tsoap_rad_der(k2,counter)  =  tsoap_rad_der(k2,counter)  +  multiplicity  *  real(  tcnk_rad_der(k2,k,n)  *  conjg(tcnk(i,k,np))  +  tcnk(i,k,n)  *  conjg(tcnk_rad_der(k2,k,np))  )
>   tsoap_azi_der(k2,counter)  =  tsoap_azi_der(k2,counter)  +  multiplicity  *  real(  tcnk_azi_der(k2,k,n)  *  conjg(tcnk(i,k,np))  +  tcnk(i,k,n)  *  conjg(tcnk_azi_der(k2,k,np))  )
>   tsoap_pol_der(k2,counter)  =  tsoap_pol_der(k2,counter)  +  multiplicity  *  real(  tcnk_pol_der(k2,k,n)  *  conjg(tcnk(i,k,np))  +  tcnk(i,k,n)  *  conjg(tcnk_pol_der(k2,k,np))  )
>   end do
>  end do
>  end do
>  end do
>  end do
> 
> ! Before the next part the variables are transposed again to their original layout.
> 
>   !omp target teams  distribute private(i)
>   do k2  =  1,  k2_max
>   i=list_of_i(k2)
>   locdot=0.d0
> 
>   !omp parallel do reduction(+:locdot_rad_der,locdot_azi_der,locdot_pol_der)
>   do is=1,nsoap
>   locdot_rad_der=locdot_rad_der+soap(is,  i)  *  soap_rad_der(is,  k2)
>   locdot_azi_der=locdot_azi_der+soap(is,  i)  *  soap_azi_der(is,  k2)
>   locdot_pol_der=locdot_pol_der+soap(is,  i)  *  soap_pol_der(is,  k2)
>   enddo
>   dot_soap_rad_der(k2)=  locdot_rad_der
>   dot_soap_azi_der(k2)=  locdot_azi_der
>   dot_soap_pol_der(k2)=  locdot_pol_der
>   end do
> 
>   !omp target teams distribute
>   do k2  =  1,  k2_max
>   i=list_of_i(k2)
> 
>   !omp parallel do
>   do is=1,nsoap
>   soap_rad_der(is,  k2)  =  soap_rad_der(is,  k2)  /  sqrt_dot_p(i)  -  soap(is,  i)  /  sqrt_dot_p(i)**3  *  dot_soap_rad_der(k2)
>   soap_azi_der(is,  k2)  =  soap_azi_der(is,  k2)  /  sqrt_dot_p(i)  -  soap(is,  i)  /  sqrt_dot_p(i)**3  *  dot_soap_azi_der(k2)
>   soap_pol_der(is,  k2)  =  soap_pol_der(is,  k2)  /  sqrt_dot_p(i)  -  soap(is,  i)  /  sqrt_dot_p(i)**3  *  dot_soap_pol_der(k2)
>   end do
>  end do
> 
>   !omp teams distribute private(k3)
>   do k2  =  1,  k2_max
>   k3=list_k2k3(k2)
> 
>   !omp parallel do private (is)
>   do is=1,n_soap
>   if(  k3  /=  k2)then
>   soap_cart_der(1,  is,  k2)  =  dsin(thetas(k2))  *  dcos(phis(k2))  *  soap_rad_der(1:n_soap,  k2)  -  dcos(thetas(k2))  *  dcos(phis(k2))  /  rjs(k2)  *  soap_pol_der(1:n_soap,  k2)  -  dsin(phis(k2))  /  rjs(k2)  *  soap_azi_der(is,  k2)
>   soap_cart_der(2,  is,  k2)  =  dsin(thetas(k2))  *  dsin(phis(k2))  *  soap_rad_der(1:n_soap,  k2)  -  dcos(thetas(k2))  *  dsin(phis(k2))  /  rjs(k2)  *  soap_pol_der(1:n_soap,  k2)  +  dcos(phis(k2))  /  rjs(k2)  *  soap_azi_der(is,  k2)
>   soap_cart_der(3,  is,  k2)  =  dcos(thetas(k2))  *  soap_rad_der(is,  k2)  +  dsin(thetas(k2))  /  rjs(k2)  *  soap_pol_der(is,  k2)
>   end if
>  end do
>  end do
> 
>   !omp teams distribute private(k3)
>   do i  =  1,  n_sites
>   k3=list_k3(i)
> 
>   !omp parallel do private(is, k2)
>   do is=1,n_soap
>   do k2=k3+1,k3+n_neigh(i)
>   soap_cart_der(1,  is,  k3)  =  soap_cart_der(1,  is,  k3)  -  soap_cart_der(1,  is,  k2)
>   soap_cart_der(2,  is,  k3)  =  soap_cart_der(2,  is,  k3)  -  soap_cart_der(2,  is,  k2)
>   soap_cart_der(3,  is,  k3)  =  soap_cart_der(3,  is,  k3)  -  soap_cart_der(3,  is,  k2)
>   end do
>  end do
>  end do 
> ```

关键点

+   识别与基于 CPU 的库等效的 GPU 库，并利用它们以确保高效的 GPU 利用。

+   识别代码中计算密集部分的重要性，这些部分对执行时间有显著贡献。

+   需要重构循环以适应 GPU 架构。

+   内存访问优化对高效 GPU 执行的重要性，包括归一化和对齐的内存访问模式。

## 在不同的 GPU 框架之间移植

你可能也会遇到需要将代码从一个特定的 GPU 框架移植到另一个框架的情况。本节概述了不同的工具，这些工具可以将 CUDA 和 OpenACC 代码分别转换为 HIP 和 OpenMP。此转换过程使应用程序能够针对不同的 GPU 架构进行优化，特别是 NVIDIA 和 AMD GPU。在此，我们重点关注[hipify](https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html)和[clacc](https://csmd.ornl.gov/project/clacc)工具。本指南改编自[NRIS 文档](https://documentation.sigma2.no/code_development/guides/cuda_translating-tools.html)。

### 使用 Hipify 将 CUDA 转换为 HIP

在本节中，我们将介绍使用`hipify-perl`和`hipify-clang`工具将 CUDA 代码转换为 HIP。

#### Hipify-perl

`hipify-perl`工具是基于 perl 的脚本，它将 CUDA 语法转换为 HIP 语法（例如，请参阅[此处](https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl)以获取更多详细信息）。例如，在一个包含 CUDA 函数`cudaMalloc`和`cudaDeviceSynchronize`的 CUDA 代码中，该工具将`cudaMalloc`替换为 HIP 函数`hipMalloc`。同样，CUDA 函数`cudaDeviceSynchronize`将被替换为 HIP 函数`hipDeviceSynchronize`。以下是在 LUMI-G 上运行`hipify-perl`的基本步骤。

+   **步骤 1**：生成`hipify-perl`脚本

    ```
    $ module  load  rocm/6.0.3
    $ hipify-clang  --perl 
    ```

+   **步骤 2**：运行生成的`hipify-perl`

    ```
    $ hipify-perl  program.cu  >  program.cu.hip 
    ```

+   **步骤 3**：使用`hipcc`编译生成的 HIP 代码

    ```
    $ hipcc  --offload-arch=gfx90a  -o  program.hip.exe  program.cu.hip 
    ```

尽管使用`hipify-perl`很简单，但该工具可能不适合大型应用程序，因为它严重依赖于用 HIP 字符串替换 CUDA 字符串（例如，它将`*cuda*`替换为`*hip*`）。此外，`hipify-perl`缺乏[区分设备/主机函数调用](https://docs.amd.com/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl)的能力。这里的替代方案是使用下一节将要描述的`hipify-clang`工具。

#### Hipify-clang

如[HIPIFY 文档](https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl)所述，`hipify-clang`工具基于 clang，用于将 CUDA 源代码转换为 HIP 源代码。与`hipify-perl`工具相比，该工具在转换 CUDA 代码方面更为健壮。此外，它通过提供辅助功能来促进代码的分析。

简而言之，`hipify-clang`需要`LLVM+CLANG`和`CUDA`。有关构建`hipify-clang`的详细信息，请参阅[此处](https://github.com/ROCm/HIPIFY)。请注意，`hipify-clang`在 LUMI-G 上可用。然而，问题可能与管理 CUDA-toolkit 的安装有关。为了避免安装过程中的任何潜在问题，我们选择使用 CUDA singularity 容器。在此，我们提供了一个逐步指南，用于运行`hipify-clang`：

+   **步骤 1**：拉取 CUDA singularity 容器，例如。

    ```
    $ singularity  pull  docker://nvcr.io/nvidia/cuda:11.4.3-devel-ubuntu20.04 
    ```

+   **步骤 2**：加载 rocm 模块并启动 CUDA singularity

    ```
    $ module  load  rocm/6.0.3
    $ singularity  shell  -B  $PWD,/opt:/opt  cuda_11.4.0-devel-ubuntu20.04.sif 
    ```

    其中，主机当前目录`$PWD`挂载到容器中的相应目录，主机中的`/opt`目录挂载到容器内部的相应目录。

+   **步骤 3**：设置环境变量`$PATH`。为了在容器内部运行`hipify-clang`，可以设置环境变量`$PATH`，该变量定义了查找二进制文件`hipify-clang`的路径。

    ```
    $ export  PATH=/opt/rocm-6.0.3/bin:$PATH 
    ```

    注意，我们使用的 rocm 版本是`rocm-6.0.3`。

+   **步骤 4**：在 singularity 容器内部运行`hipify-clang`

    ```
    $ hipify-clang  program.cu  -o  hip_program.cu.hip  --cuda-path=/usr/local/cuda-11.4  -I  /usr/local/cuda-11.4/include 
    ```

    在这里，应指定 cuda 路径以及`*includes*`和`*defines*`文件的路径。CUDA 源代码和生成的输出代码分别是`program.cu`和`hip_program.cu.hip`。

    生成的 hip 代码的编译过程语法与上一节中描述的类似（参见 hipify-perl 部分的**步骤 3**）。

`Hipify`练习的代码示例可以通过在`content/examples/exercise_hipify`子目录中克隆此存储库来访问：

> ```
> $ git  clone  https://github.com/ENCCS/gpu-programming.git
> $ cd  gpu-programming/content/examples/exercise_hipify
> $ ls 
> ```

练习 I：使用`hipify-perl`将 CUDA 代码转换为 HIP

1.1 生成`hipify-perl`工具。

1.2 使用`Hipify-perl`工具将位于`/exercise_hipify/Hipify_perl`的 CUDA 代码`vec_add_cuda.cu`转换为 HIP。

1.3 使用`hipcc`编译器包装器编译生成的 HIP 代码并运行它。

练习 II：使用`hipify-clang`将 CUDA 代码转换为 HIP

2.1 使用`Hipify-clang`工具将位于`/exercise_hipify/Hipify_clang`的 CUDA 代码`vec_add_cuda.cu`转换为 HIP。

2.2 使用 `hipcc` 编译器包装器编译生成的 HIP 代码并运行。

### 使用 Clacc 将 OpenACC 转换为 OpenMP

[Clacc](https://github.com/llvm-doe-org/llvm-project/tree/clacc/main) 是一个工具，用于将 OpenACC 应用程序转换为使用 Clang/LLVM 编译器环境的 OpenMP 转发。请注意，该工具针对 OpenACC C，而 OpenACC Fortran 已经在 AMD GPU 上得到支持。如 [GitHub 仓库](https://github.com/llvm-doe-org/llvm-project/tree/clacc/main) 中所示，编译器 `Clacc` 是位于 `\install` 目录下的 `\bin` 子目录中的 `Clang` 可执行文件，具体描述如下。

在以下内容中，我们提供了一个逐步指南，用于构建和使用 Clacc：

+   **步骤 1**：构建和安装 [Clacc](https://github.com/llvm-doe-org/llvm-project/tree/clacc/main)。

    ```
    $ git  clone  -b  clacc/main  https://github.com/llvm-doe-org/llvm-project.git
    $ cd  llvm-project
    $ mkdir  build  &&  cd  build
    $ cmake  -DCMAKE_INSTALL_PREFIX=../install  \
      -DCMAKE_BUILD_TYPE=Release  \
      -DLLVM_ENABLE_PROJECTS="clang;lld"  \
      -DLLVM_ENABLE_RUNTIMES=openmp  \
      -DLLVM_TARGETS_TO_BUILD="host;AMDGPU"  \
      -DCMAKE_C_COMPILER=gcc  \
      -DCMAKE_CXX_COMPILER=g++  \
      ../llvm
    $ make
    $ make  install 
    ```

+   **步骤 2**：设置环境变量，以便能够从 `/install` 目录工作，这是最简单的方法。我们假设 `/install` 目录位于路径 `/project/project_xxxxxx/Clacc/llvm-project`。

对于更高级的使用，例如修改 `Clacc`，我们建议读者参考[“从构建目录使用”](https://github.com/llvm-doe-org/llvm-project/blob/clacc/main/README.md)。

> ```
> $ export  PATH=/scratch/project_465002387/clacc/llvm-project/install/bin:$PATH
> $ export  LD_LIBRARY_PATH=/scratch/project_465002387/clacc/llvm-project/install/lib:$LD_LIBRARY_PATH 
> ```

+   **步骤 3**：将需要打印输出的 openACC_code.c 代码源代码转换为 openMP_code.c：

    ```
    $ clang  -fopenacc-print=omp  -fopenacc-structured-ref-count-omp=no-ompx-hold  openACC_code.c  >  openMP_code.c 
    ```

    在这里引入了标志 `-fopenacc-structured-ref-count-omp=no-ompx-hold` 来禁用 `ompx_hold` 映射类型修饰符，该修饰符用于 OpenACC `copy` 子句的翻译。`ompx_hold` 是一个 OpenMP 扩展，可能尚未被其他编译器支持。

+   **步骤 4**：使用 [cc 编译器包装器](https://docs.lumi-supercomputer.eu/development/compiling/prgenv/) 编译代码。

    ```
    module load CrayEnv
    module load PrgEnv-cray
    module load craype-accel-amd-gfx90a
    module load rocm/6.0.3

    cc -fopenmp -o executable openMP_code.c 
    ```

访问练习材料

`Clacc` 练习的代码示例可以通过在内容目录 `examples/exercise_clacc` 中克隆此仓库来访问：

```
$ git  clone  https://github.com/ENCCS/gpu-programming.git
$ cd  gpu-programming/content/examples/exercise_clacc
$ ls 
```

练习：将 OpenACC 代码转换为 OpenMP

1.  使用 `Clacc` 编译器将位于 `/exercise_clacc` 的 OpenACC 代码 `openACC_code.c` 转换。

1.  使用 `cc` 编译器包装器编译生成的 OpenMP 代码并运行。

### 使用 SYCLomatic 将 CUDA 转换为 SYCL/DPC++

Intel 提供了一个用于 CUDA 到 SYCL 代码迁移的工具，包含在 Intel oneAPI Basekit 中。

它没有安装在 LUMI 上，但一般工作流程与 HIPify Clang 相似，并且也需要现有的 CUDA 安装：

> ```
> $ dpct  program.cu
> $ cd  dpct_output/
> $ icpx  -fsycl  program.dp.cpp 
> ```

SYCLomatic 可以通过使用 `-in-root` 和 `-out-root` 标志递归地处理目录来迁移较大的项目。它还可以使用编译数据库（由 CMake 和其他构建系统支持）来处理更复杂的项目布局。

请注意，由 SYCLomatic 生成的代码依赖于 oneAPI 特定扩展，因此不能直接与其他 SYCL 实现（如 AdaptiveCpp（hipSYCL））一起使用。可以通过将 `--no-incremental-migration` 标志添加到 `dpct` 命令中来最小化，但无法完全避免使用此兼容层。因为这需要手动操作，因为某些 CUDA 概念不能直接映射到 SYCL。

此外，CUDA 应用程序可能假设某些硬件行为，例如 32 宽的 warps。如果目标硬件不同（例如，LUMI 中使用的 AMD MI250 GPU，warp 的大小为 64），则算法可能需要手动调整。

### 结论

这简要概述了使用现有工具将 CUDA 代码转换为 HIP 和 SYCL，以及将 OpenACC 代码转换为 OpenMP 转发的用法。一般来说，大型应用程序的转换过程可能不完整，因此需要手动修改以完成迁移过程。然而，值得注意的是，转换过程的准确性要求应用程序根据 CUDA 和 OpenACC 语法正确编写。

## 参见

+   [Hipify GitHub](https://github.com/ROCm/HIPIFY)

+   [Hipify 参考指南 v5.1](https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html)

+   [HIP 示例](https://github.com/olcf-tutorials/simple_HIP_examples/tree/master/vector_addition)

+   [将 CUDA 迁移到 HIP](https://www.admin-magazine.com/HPC/Articles/Porting-CUDA-to-HIP)

+   [Clacc 主仓库 README](https://github.com/llvm-doe-org/llvm-project/blob/clacc/main/README.md)

+   [SYCLomatic 主图](https://www.intel.com/content/www/us/en/developer/articles/technical/syclomatic-new-cuda-to-sycl-code-migration-tool.html)

+   [SYCLomatic 文档](https://oneapi-src.github.io/SYCLomatic/get_started/index.html)

重点

+   存在将工具从 CUDA 转换到 HIP 和 SYCL 以及从 OpenACC 转换到 OpenMP 的有用工具，但可能需要手动修改。

## 从 CPU 迁移到 GPU

在将代码迁移以利用 GPU 的并行处理能力时，需要遵循几个步骤，并在实际编写要在 GPU 上执行的并行代码之前做一些额外的工作：

+   **识别目标部分**：首先识别对执行时间贡献显著的代码部分。这些通常是计算密集型部分，如循环或矩阵运算。帕累托原则表明，大约 10-20% 的代码占用了 80-90% 的执行时间。

+   **等效 GPU 库**：如果原始代码使用 CPU 库（如 BLAS、FFT 等），则识别等效 GPU 库至关重要。例如，cuBLAS 或 hipBLAS 可以替换基于 CPU 的 BLAS 库。利用特定于 GPU 的库可以确保高效利用 GPU。

+   **重构循环**：在直接将循环移植到 GPU 时，需要进行一些重构以适应 GPU 架构。这通常涉及将循环拆分为多个步骤或修改操作以利用迭代之间的独立性并改进内存访问模式。原始循环的每个步骤都可以映射到一个内核，由多个 GPU 线程执行，每个线程对应一个迭代。

+   **内存访问优化**：考虑代码中的内存访问模式。当内存访问是归一化和对齐的时，GPU 表现最佳。通过最小化全局内存访问并最大化共享内存或寄存器的利用，可以显著提高性能。审查代码以确保 GPU 执行的优化内存访问。

### 讨论

> 这将如何移植？（n_soap ≈ 100，n_sites ⩾ 10000，k_max ≈ 20*n_sites）
> 
> > 检查以下 Fortran 代码（如果你不读 Fortran：do-loops == for-loops）
> > 
> > ```
> > k2  =  0
> > do i  =  1,  n_sites
> >   do j  =  1,  n_neigh(i)
> >   k2  =  k2  +  1
> >   counter  =  0
> >   counter2  =  0
> >   do n  =  1,  n_max
> >   do np  =  n,  n_max
> >   do l  =  0,  l_max
> >   if(  skip_soap_component(l,  np,  n)  )cycle
> > 
> >   counter  =  counter+1
> >   do m  =  0,  l
> >   k  =  1  +  l*(l+1)/2  +  m
> >   counter2  =  counter2  +  1
> >   multiplicity  =  multiplicity_array(counter2)
> >   soap_rad_der(counter,  k2)  =  soap_rad_der(counter,  k2)  +  multiplicity  *  real(  cnk_rad_der(k,  n,  k2)  *  conjg(cnk(k,  np,  i))  +  cnk(k,  n,  i)  *  conjg(cnk_rad_der(k,  np,  k2))  )
> >   soap_azi_der(counter,  k2)  =  soap_azi_der(counter,  k2)  +  multiplicity  *  real(  cnk_azi_der(k,  n,  k2)  *  conjg(cnk(k,  np,  i))  +  cnk(k,  n,  i)  *  conjg(cnk_azi_der(k,  np,  k2))  )
> >   soap_pol_der(counter,  k2)  =  soap_pol_der(counter,  k2)  +  multiplicity  *  real(  cnk_pol_der(k,  n,  k2)  *  conjg(cnk(k,  np,  i))  +  cnk(k,  n,  i)  *  conjg(cnk_pol_der(k,  np,  k2))  )
> >   end do
> >  end do
> >  end do
> >  end do
> > 
> >   soap_rad_der(1:n_soap,  k2)  =  soap_rad_der(1:n_soap,  k2)  /  sqrt_dot_p(i)  -  soap(1:n_soap,  i)  /  sqrt_dot_p(i)**3  *  dot_product(  soap(1:n_soap,  i),  soap_rad_der(1:n_soap,  k2)  )
> >   soap_azi_der(1:n_soap,  k2)  =  soap_azi_der(1:n_soap,  k2)  /  sqrt_dot_p(i)  -  soap(1:n_soap,  i)  /  sqrt_dot_p(i)**3  *  dot_product(  soap(1:n_soap,  i),  soap_azi_der(1:n_soap,  k2)  )
> >   soap_pol_der(1:n_soap,  k2)  =  soap_pol_der(1:n_soap,  k2)  /  sqrt_dot_p(i)  -  soap(1:n_soap,  i)  /  sqrt_dot_p(i)**3  *  dot_product(  soap(1:n_soap,  i),  soap_pol_der(1:n_soap,  k2)  )
> > 
> >   if(  j  ==  1  )then
> >   k3  =  k2
> >   else
> >   soap_cart_der(1,  1:n_soap,  k2)  =  dsin(thetas(k2))  *  dcos(phis(k2))  *  soap_rad_der(1:n_soap,  k2)  -  dcos(thetas(k2))  *  dcos(phis(k2))  /  rjs(k2)  *  soap_pol_der(1:n_soap,  k2)  -  dsin(phis(k2))  /  rjs(k2)  *  soap_azi_der(1:n_soap,  k2)
> >   soap_cart_der(2,  1:n_soap,  k2)  =  dsin(thetas(k2))  *  dsin(phis(k2))  *  soap_rad_der(1:n_soap,  k2)  -  dcos(thetas(k2))  *  dsin(phis(k2))  /  rjs(k2)  *  soap_pol_der(1:n_soap,  k2)  +  dcos(phis(k2))  /  rjs(k2)  *  soap_azi_der(1:n_soap,  k2)
> >   soap_cart_der(3,  1:n_soap,  k2)  =  dcos(thetas(k2))  *  soap_rad_der(1:n_soap,  k2)  +  dsin(thetas(k2))  /  rjs(k2)  *  soap_pol_der(1:n_soap,  k2)
> >   soap_cart_der(1,  1:n_soap,  k3)  =  soap_cart_der(1,  1:n_soap,  k3)  -  soap_cart_der(1,  1:n_soap,  k2)
> >   soap_cart_der(2,  1:n_soap,  k3)  =  soap_cart_der(2,  1:n_soap,  k3)  -  soap_cart_der(2,  1:n_soap,  k2)
> >   soap_cart_der(3,  1:n_soap,  k3)  =  soap_cart_der(3,  1:n_soap,  k3)  -  soap_cart_der(3,  1:n_soap,  k2)
> >   end if
> >  end do
> > end do 
> > ```
> > 
> 一些初步步骤：
> 
> > +   代码可以被（必须）拆分为 3-4 个内核。为什么？
> > +   
> > +   检查是否存在可能导致迭代之间产生虚假依赖的变量，如索引 k2
> > +   
> > +   对于 GPU 来说，将工作分配到索引 i 是否高效？关于内存访问呢？注意 Fortran 中的数组是二维的。
> > +   
> > +   是否可以合并一些循环？合并嵌套循环可以减少开销并改进内存访问模式，从而提高 GPU 性能。
> > +   
> > +   在 GPU 中最佳的内存访问是什么？回顾代码中的内存访问模式。通过在适当的地方使用共享内存或寄存器来最小化全局内存访问，确保内存访问是归一化和对齐的，以最大化 GPU 内存吞吐量。

代码重构！

+   寄存器有限，内核使用寄存器越多，活跃线程就越少（占用率低）。

+   为了计算 soap_rad_der(is,k2)，CUDA 线程需要访问所有之前值 soap_rad_der(1:nsoap,k2)。

+   为了计算 soap_cart_der(1, 1:n_soap, k3)，需要访问所有值（k3+1:k2+n_neigh(i)）。

+   注意第一部分中的索引。矩阵被转置以获得更好的访问模式。

> ```
>  !omp target teams distribute parallel do private (i)
>   do k2  =  1,  k2_max
>   i=list_of_i(k2)
>   counter  =  0
>   counter2  =  0
>   do n  =  1,  n_max
>   do np  =  n,  n_max
>   do l  =  0,  l_max
>   if(  skip_soap_component(l,  np,  n)  )  then
>  cycle
>  endif
>   counter  =  counter+1
>   do m  =  0,  l
>   k  =  1  +  l*(l+1)/2  +  m
>   counter2  =  counter2  +  1
>   multiplicity  =  multiplicity_array(counter2)
>   tsoap_rad_der(k2,counter)  =  tsoap_rad_der(k2,counter)  +  multiplicity  *  real(  tcnk_rad_der(k2,k,n)  *  conjg(tcnk(i,k,np))  +  tcnk(i,k,n)  *  conjg(tcnk_rad_der(k2,k,np))  )
>   tsoap_azi_der(k2,counter)  =  tsoap_azi_der(k2,counter)  +  multiplicity  *  real(  tcnk_azi_der(k2,k,n)  *  conjg(tcnk(i,k,np))  +  tcnk(i,k,n)  *  conjg(tcnk_azi_der(k2,k,np))  )
>   tsoap_pol_der(k2,counter)  =  tsoap_pol_der(k2,counter)  +  multiplicity  *  real(  tcnk_pol_der(k2,k,n)  *  conjg(tcnk(i,k,np))  +  tcnk(i,k,n)  *  conjg(tcnk_pol_der(k2,k,np))  )
>   end do
>  end do
>  end do
>  end do
>  end do
> 
> ! Before the next part the variables are transposed again to their original layout.
> 
>   !omp target teams  distribute private(i)
>   do k2  =  1,  k2_max
>   i=list_of_i(k2)
>   locdot=0.d0
> 
>   !omp parallel do reduction(+:locdot_rad_der,locdot_azi_der,locdot_pol_der)
>   do is=1,nsoap
>   locdot_rad_der=locdot_rad_der+soap(is,  i)  *  soap_rad_der(is,  k2)
>   locdot_azi_der=locdot_azi_der+soap(is,  i)  *  soap_azi_der(is,  k2)
>   locdot_pol_der=locdot_pol_der+soap(is,  i)  *  soap_pol_der(is,  k2)
>   enddo
>   dot_soap_rad_der(k2)=  locdot_rad_der
>   dot_soap_azi_der(k2)=  locdot_azi_der
>   dot_soap_pol_der(k2)=  locdot_pol_der
>   end do
> 
>   !omp target teams distribute
>   do k2  =  1,  k2_max
>   i=list_of_i(k2)
> 
>   !omp parallel do
>   do is=1,nsoap
>   soap_rad_der(is,  k2)  =  soap_rad_der(is,  k2)  /  sqrt_dot_p(i)  -  soap(is,  i)  /  sqrt_dot_p(i)**3  *  dot_soap_rad_der(k2)
>   soap_azi_der(is,  k2)  =  soap_azi_der(is,  k2)  /  sqrt_dot_p(i)  -  soap(is,  i)  /  sqrt_dot_p(i)**3  *  dot_soap_azi_der(k2)
>   soap_pol_der(is,  k2)  =  soap_pol_der(is,  k2)  /  sqrt_dot_p(i)  -  soap(is,  i)  /  sqrt_dot_p(i)**3  *  dot_soap_pol_der(k2)
>   end do
>  end do
> 
>   !omp teams distribute private(k3)
>   do k2  =  1,  k2_max
>   k3=list_k2k3(k2)
> 
>   !omp parallel do private (is)
>   do is=1,n_soap
>   if(  k3  /=  k2)then
>   soap_cart_der(1,  is,  k2)  =  dsin(thetas(k2))  *  dcos(phis(k2))  *  soap_rad_der(1:n_soap,  k2)  -  dcos(thetas(k2))  *  dcos(phis(k2))  /  rjs(k2)  *  soap_pol_der(1:n_soap,  k2)  -  dsin(phis(k2))  /  rjs(k2)  *  soap_azi_der(is,  k2)
>   soap_cart_der(2,  is,  k2)  =  dsin(thetas(k2))  *  dsin(phis(k2))  *  soap_rad_der(1:n_soap,  k2)  -  dcos(thetas(k2))  *  dsin(phis(k2))  /  rjs(k2)  *  soap_pol_der(1:n_soap,  k2)  +  dcos(phis(k2))  /  rjs(k2)  *  soap_azi_der(is,  k2)
>   soap_cart_der(3,  is,  k2)  =  dcos(thetas(k2))  *  soap_rad_der(is,  k2)  +  dsin(thetas(k2))  /  rjs(k2)  *  soap_pol_der(is,  k2)
>   end if
>  end do
>  end do
> 
>   !omp teams distribute private(k3)
>   do i  =  1,  n_sites
>   k3=list_k3(i)
> 
>   !omp parallel do private(is, k2)
>   do is=1,n_soap
>   do k2=k3+1,k3+n_neigh(i)
>   soap_cart_der(1,  is,  k3)  =  soap_cart_der(1,  is,  k3)  -  soap_cart_der(1,  is,  k2)
>   soap_cart_der(2,  is,  k3)  =  soap_cart_der(2,  is,  k3)  -  soap_cart_der(2,  is,  k2)
>   soap_cart_der(3,  is,  k3)  =  soap_cart_der(3,  is,  k3)  -  soap_cart_der(3,  is,  k2)
>   end do
>  end do
>  end do 
> ```

重点

+   识别与基于 CPU 的库等效的 GPU 库，并利用它们以确保高效的 GPU 利用。

+   识别代码中计算密集部分的重要性，这些部分对执行时间有显著贡献。

+   需要重构循环以适应 GPU 架构。

+   内存访问优化对于高效 GPU 执行的重要性，包括归一化和对齐的内存访问模式。

### 讨论

> 这将如何移植？（n_soap ≈ 100，n_sites ⩾ 10000，k_max ≈ 20*n_sites）
> 
> > 检查以下 Fortran 代码（如果你不读 Fortran：do-loops == for-loops）
> > 
> > ```
> > k2  =  0
> > do i  =  1,  n_sites
> >   do j  =  1,  n_neigh(i)
> >   k2  =  k2  +  1
> >   counter  =  0
> >   counter2  =  0
> >   do n  =  1,  n_max
> >   do np  =  n,  n_max
> >   do l  =  0,  l_max
> >   if(  skip_soap_component(l,  np,  n)  )cycle
> > 
> >   counter  =  counter+1
> >   do m  =  0,  l
> >   k  =  1  +  l*(l+1)/2  +  m
> >   counter2  =  counter2  +  1
> >   multiplicity  =  multiplicity_array(counter2)
> >   soap_rad_der(counter,  k2)  =  soap_rad_der(counter,  k2)  +  multiplicity  *  real(  cnk_rad_der(k,  n,  k2)  *  conjg(cnk(k,  np,  i))  +  cnk(k,  n,  i)  *  conjg(cnk_rad_der(k,  np,  k2))  )
> >   soap_azi_der(counter,  k2)  =  soap_azi_der(counter,  k2)  +  multiplicity  *  real(  cnk_azi_der(k,  n,  k2)  *  conjg(cnk(k,  np,  i))  +  cnk(k,  n,  i)  *  conjg(cnk_azi_der(k,  np,  k2))  )
> >   soap_pol_der(counter,  k2)  =  soap_pol_der(counter,  k2)  +  multiplicity  *  real(  cnk_pol_der(k,  n,  k2)  *  conjg(cnk(k,  np,  i))  +  cnk(k,  n,  i)  *  conjg(cnk_pol_der(k,  np,  k2))  )
> >   end do
> >  end do
> >  end do
> >  end do
> > 
> >   soap_rad_der(1:n_soap,  k2)  =  soap_rad_der(1:n_soap,  k2)  /  sqrt_dot_p(i)  -  soap(1:n_soap,  i)  /  sqrt_dot_p(i)**3  *  dot_product(  soap(1:n_soap,  i),  soap_rad_der(1:n_soap,  k2)  )
> >   soap_azi_der(1:n_soap,  k2)  =  soap_azi_der(1:n_soap,  k2)  /  sqrt_dot_p(i)  -  soap(1:n_soap,  i)  /  sqrt_dot_p(i)**3  *  dot_product(  soap(1:n_soap,  i),  soap_azi_der(1:n_soap,  k2)  )
> >   soap_pol_der(1:n_soap,  k2)  =  soap_pol_der(1:n_soap,  k2)  /  sqrt_dot_p(i)  -  soap(1:n_soap,  i)  /  sqrt_dot_p(i)**3  *  dot_product(  soap(1:n_soap,  i),  soap_pol_der(1:n_soap,  k2)  )
> > 
> >   if(  j  ==  1  )then
> >   k3  =  k2
> >   else
> >   soap_cart_der(1,  1:n_soap,  k2)  =  dsin(thetas(k2))  *  dcos(phis(k2))  *  soap_rad_der(1:n_soap,  k2)  -  dcos(thetas(k2))  *  dcos(phis(k2))  /  rjs(k2)  *  soap_pol_der(1:n_soap,  k2)  -  dsin(phis(k2))  /  rjs(k2)  *  soap_azi_der(1:n_soap,  k2)
> >   soap_cart_der(2,  1:n_soap,  k2)  =  dsin(thetas(k2))  *  dsin(phis(k2))  *  soap_rad_der(1:n_soap,  k2)  -  dcos(thetas(k2))  *  dsin(phis(k2))  /  rjs(k2)  *  soap_pol_der(1:n_soap,  k2)  +  dcos(phis(k2))  /  rjs(k2)  *  soap_azi_der(1:n_soap,  k2)
> >   soap_cart_der(3,  1:n_soap,  k2)  =  dcos(thetas(k2))  *  soap_rad_der(1:n_soap,  k2)  +  dsin(thetas(k2))  /  rjs(k2)  *  soap_pol_der(1:n_soap,  k2)
> >   soap_cart_der(1,  1:n_soap,  k3)  =  soap_cart_der(1,  1:n_soap,  k3)  -  soap_cart_der(1,  1:n_soap,  k2)
> >   soap_cart_der(2,  1:n_soap,  k3)  =  soap_cart_der(2,  1:n_soap,  k3)  -  soap_cart_der(2,  1:n_soap,  k2)
> >   soap_cart_der(3,  1:n_soap,  k3)  =  soap_cart_der(3,  1:n_soap,  k3)  -  soap_cart_der(3,  1:n_soap,  k2)
> >   end if
> >  end do
> > end do 
> > ```
> > 
> 一些初步步骤：
> 
> > +   代码可以被（必须）拆分为 3-4 个内核。为什么？
> > +   
> > +   检查是否存在可能导致迭代之间产生虚假依赖的变量，如索引 k2
> > +   
> > +   对于 GPU 来说，将工作分配到索引 i 是否高效？关于内存访问呢？注意 Fortran 中的数组是二维的
> > +   
> > +   是否可以合并一些循环？合并嵌套循环可以减少开销并改善内存访问模式，从而提高 GPU 性能。
> > +   
> > +   在 GPU 中最佳的内存访问方式是什么？审查代码中的内存访问模式。通过在适当的地方利用共享内存或寄存器来最小化全局内存访问。确保内存访问是合并和对齐的，以最大化 GPU 内存吞吐量

重构后的代码！

+   寄存器有限，内核使用寄存器越多，活跃线程就越少（占用率低）。

+   为了计算 soap_rad_der(is,k2)，CUDA 线程需要访问所有之前值 soap_rad_der(1:nsoap,k2)。

+   为了计算 soap_cart_der(1, 1:n_soap, k3)，需要访问所有值（k3+1:k2+n_neigh(i)）。

+   注意第一部分中的索引。矩阵被转置以获得更好的访问模式。

> ```
>  !omp target teams distribute parallel do private (i)
>   do k2  =  1,  k2_max
>   i=list_of_i(k2)
>   counter  =  0
>   counter2  =  0
>   do n  =  1,  n_max
>   do np  =  n,  n_max
>   do l  =  0,  l_max
>   if(  skip_soap_component(l,  np,  n)  )  then
>  cycle
>  endif
>   counter  =  counter+1
>   do m  =  0,  l
>   k  =  1  +  l*(l+1)/2  +  m
>   counter2  =  counter2  +  1
>   multiplicity  =  multiplicity_array(counter2)
>   tsoap_rad_der(k2,counter)  =  tsoap_rad_der(k2,counter)  +  multiplicity  *  real(  tcnk_rad_der(k2,k,n)  *  conjg(tcnk(i,k,np))  +  tcnk(i,k,n)  *  conjg(tcnk_rad_der(k2,k,np))  )
>   tsoap_azi_der(k2,counter)  =  tsoap_azi_der(k2,counter)  +  multiplicity  *  real(  tcnk_azi_der(k2,k,n)  *  conjg(tcnk(i,k,np))  +  tcnk(i,k,n)  *  conjg(tcnk_azi_der(k2,k,np))  )
>   tsoap_pol_der(k2,counter)  =  tsoap_pol_der(k2,counter)  +  multiplicity  *  real(  tcnk_pol_der(k2,k,n)  *  conjg(tcnk(i,k,np))  +  tcnk(i,k,n)  *  conjg(tcnk_pol_der(k2,k,np))  )
>   end do
>  end do
>  end do
>  end do
>  end do
> 
> ! Before the next part the variables are transposed again to their original layout.
> 
>   !omp target teams  distribute private(i)
>   do k2  =  1,  k2_max
>   i=list_of_i(k2)
>   locdot=0.d0
> 
>   !omp parallel do reduction(+:locdot_rad_der,locdot_azi_der,locdot_pol_der)
>   do is=1,nsoap
>   locdot_rad_der=locdot_rad_der+soap(is,  i)  *  soap_rad_der(is,  k2)
>   locdot_azi_der=locdot_azi_der+soap(is,  i)  *  soap_azi_der(is,  k2)
>   locdot_pol_der=locdot_pol_der+soap(is,  i)  *  soap_pol_der(is,  k2)
>   enddo
>   dot_soap_rad_der(k2)=  locdot_rad_der
>   dot_soap_azi_der(k2)=  locdot_azi_der
>   dot_soap_pol_der(k2)=  locdot_pol_der
>   end do
> 
>   !omp target teams distribute
>   do k2  =  1,  k2_max
>   i=list_of_i(k2)
> 
>   !omp parallel do
>   do is=1,nsoap
>   soap_rad_der(is,  k2)  =  soap_rad_der(is,  k2)  /  sqrt_dot_p(i)  -  soap(is,  i)  /  sqrt_dot_p(i)**3  *  dot_soap_rad_der(k2)
>   soap_azi_der(is,  k2)  =  soap_azi_der(is,  k2)  /  sqrt_dot_p(i)  -  soap(is,  i)  /  sqrt_dot_p(i)**3  *  dot_soap_azi_der(k2)
>   soap_pol_der(is,  k2)  =  soap_pol_der(is,  k2)  /  sqrt_dot_p(i)  -  soap(is,  i)  /  sqrt_dot_p(i)**3  *  dot_soap_pol_der(k2)
>   end do
>  end do
> 
>   !omp teams distribute private(k3)
>   do k2  =  1,  k2_max
>   k3=list_k2k3(k2)
> 
>   !omp parallel do private (is)
>   do is=1,n_soap
>   if(  k3  /=  k2)then
>   soap_cart_der(1,  is,  k2)  =  dsin(thetas(k2))  *  dcos(phis(k2))  *  soap_rad_der(1:n_soap,  k2)  -  dcos(thetas(k2))  *  dcos(phis(k2))  /  rjs(k2)  *  soap_pol_der(1:n_soap,  k2)  -  dsin(phis(k2))  /  rjs(k2)  *  soap_azi_der(is,  k2)
>   soap_cart_der(2,  is,  k2)  =  dsin(thetas(k2))  *  dsin(phis(k2))  *  soap_rad_der(1:n_soap,  k2)  -  dcos(thetas(k2))  *  dsin(phis(k2))  /  rjs(k2)  *  soap_pol_der(1:n_soap,  k2)  +  dcos(phis(k2))  /  rjs(k2)  *  soap_azi_der(is,  k2)
>   soap_cart_der(3,  is,  k2)  =  dcos(thetas(k2))  *  soap_rad_der(is,  k2)  +  dsin(thetas(k2))  /  rjs(k2)  *  soap_pol_der(is,  k2)
>   end if
>  end do
>  end do
> 
>   !omp teams distribute private(k3)
>   do i  =  1,  n_sites
>   k3=list_k3(i)
> 
>   !omp parallel do private(is, k2)
>   do is=1,n_soap
>   do k2=k3+1,k3+n_neigh(i)
>   soap_cart_der(1,  is,  k3)  =  soap_cart_der(1,  is,  k3)  -  soap_cart_der(1,  is,  k2)
>   soap_cart_der(2,  is,  k3)  =  soap_cart_der(2,  is,  k3)  -  soap_cart_der(2,  is,  k2)
>   soap_cart_der(3,  is,  k3)  =  soap_cart_der(3,  is,  k3)  -  soap_cart_der(3,  is,  k2)
>   end do
>  end do
>  end do 
> ```

关键点

+   识别与基于 CPU 的库等效的 GPU 库，并利用它们以确保高效的 GPU 利用率。

+   识别代码中计算密集部分的重要性，这些部分对执行时间有显著贡献。

+   需要重构循环以适应 GPU 架构。

+   内存访问优化对高效 GPU 执行的重要性，包括合并和对齐的内存访问模式。

## 在不同的 GPU 框架之间移植

你可能也会发现自己处于需要将代码从一个特定的 GPU 框架移植到另一个框架的情况。本节概述了不同的工具，这些工具可以将 CUDA 和 OpenACC 代码分别转换为 HIP 和 OpenMP。此转换过程使应用程序能够针对各种 GPU 架构，特别是 NVIDIA 和 AMD GPU。在此，我们关注 [hipify](https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html) 和 [clacc](https://csmd.ornl.gov/project/clacc) 工具。本指南改编自 [NRIS 文档](https://documentation.sigma2.no/code_development/guides/cuda_translating-tools.html)。

### 使用 Hipify 将 CUDA 转换为 HIP

在本节中，我们介绍了使用 `hipify-perl` 和 `hipify-clang` 工具将 CUDA 代码转换为 HIP 的方法。

#### Hipify-perl

`hipify-perl` 工具是一个基于 perl 的脚本，它将 CUDA 语法转换为 HIP 语法（例如，请参阅[此处](https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl)以获取更多详细信息）。例如，在一个包含 CUDA 函数 `cudaMalloc` 和 `cudaDeviceSynchronize` 的 CUDA 代码中，该工具将用 HIP 函数 `hipMalloc` 替换 `cudaMalloc`。同样，CUDA 函数 `cudaDeviceSynchronize` 将被 HIP 函数 `hipDeviceSynchronize` 替换。以下是在 LUMI-G 上运行 `hipify-perl` 的基本步骤。

+   **步骤 1**：生成 `hipify-perl` 脚本

    ```
    $ module  load  rocm/6.0.3
    $ hipify-clang  --perl 
    ```

+   **步骤 2**：运行生成的 `hipify-perl`

    ```
    $ hipify-perl  program.cu  >  program.cu.hip 
    ```

+   **步骤 3**：使用 `hipcc` 编译器包装器编译生成的 HIP 代码

    ```
    $ hipcc  --offload-arch=gfx90a  -o  program.hip.exe  program.cu.hip 
    ```

尽管 `hipify-perl` 的使用很简单，但该工具可能不适合大型应用程序，因为它主要依赖于将 CUDA 字符串替换为 HIP 字符串（例如，将 `*cuda*` 替换为 `*hip*`）。此外，`hipify-perl` 缺乏[区分设备/主机函数调用](https://docs.amd.com/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl)的能力。这里的替代方案是使用下一节将要描述的 `hipify-clang` 工具。

#### Hipify-clang

如 [HIPIFY 文档](https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl) 所述，`hipify-clang` 工具基于 clang，用于将 CUDA 源代码转换为 HIP 源代码。与 `hipify-perl` 工具相比，该工具在翻译 CUDA 代码方面更稳健。此外，它通过提供辅助功能来促进代码分析。

简而言之，`hipify-clang` 需要 `LLVM+CLANG` 和 `CUDA`。有关构建 `hipify-clang` 的详细信息，请参阅[此处](https://github.com/ROCm/HIPIFY)。请注意，`hipify-clang` 在 LUMI-G 上可用。然而，问题可能与管理 CUDA-toolkit 的安装有关。为了避免安装过程中可能出现的任何问题，我们选择使用 CUDA singularity 容器。在此，我们提供了一个逐步指南，用于运行 `hipify-clang`：

+   **步骤 1**：拉取 CUDA singularity 容器，例如。

    ```
    $ singularity  pull  docker://nvcr.io/nvidia/cuda:11.4.3-devel-ubuntu20.04 
    ```

+   **步骤 2**：加载 rocm 模块并启动 CUDA singularity

    ```
    $ module  load  rocm/6.0.3
    $ singularity  shell  -B  $PWD,/opt:/opt  cuda_11.4.0-devel-ubuntu20.04.sif 
    ```

    其中，主机中的当前目录 `$PWD` 被挂载到容器中的相应目录，主机中的 `/opt` 目录被挂载到容器内的相应目录。

+   **步骤 3**：设置环境变量 `$PATH`。为了在容器内运行 `hipify-clang`，可以设置环境变量 `$PATH`，该变量定义了查找二进制文件 `hipify-clang` 的路径。

    ```
    $ export  PATH=/opt/rocm-6.0.3/bin:$PATH 
    ```

    注意，我们使用的 rocm 版本是 `rocm-6.0.3`。

+   **步骤 4**：在 singularity 容器内运行 `hipify-clang`

    ```
    $ hipify-clang  program.cu  -o  hip_program.cu.hip  --cuda-path=/usr/local/cuda-11.4  -I  /usr/local/cuda-11.4/include 
    ```

    在这里，应指定 cuda 路径以及 `*includes*` 和 `*defines*` 文件的路经。CUDA 源代码和生成的输出代码分别是 program.cu 和 hip_program.cu.hip。

    生成 hip 代码的编译过程语法与上一节中描述的类似（参见 hipify-perl 节的 **步骤 3**）。

可以通过克隆此存储库在内容/示例/exercise_hipify 子目录中访问 `Hipify` 练习的代码示例：

> ```
> $ git  clone  https://github.com/ENCCS/gpu-programming.git
> $ cd  gpu-programming/content/examples/exercise_hipify
> $ ls 
> ```

练习 I：使用 `hipify-perl` 将 CUDA 代码翻译成 HIP

1.1 生成 `hipify-perl` 工具。

1.2 使用 `Hipify-perl` 工具将位于 `/exercise_hipify/Hipify_perl` 的 CUDA 代码 `vec_add_cuda.cu` 转换为 HIP。

1.3 使用 `hipcc` 编译器包装器编译生成的 HIP 代码并运行它。

练习 II：使用 `hipify-clang` 将 CUDA 代码翻译成 HIP

2.1 使用`Hipify-clang`工具将位于`/exercise_hipify/Hipify_clang`的 CUDA 代码`vec_add_cuda.cu`转换为 HIP。

2.2 使用`hipcc`编译器包装器编译生成的 HIP 代码并运行它。

### 使用 Clacc 将 OpenACC 转换为 OpenMP

[Clacc](https://github.com/llvm-doe-org/llvm-project/tree/clacc/main)是一个工具，可以将 OpenACC 应用程序转换为使用 Clang/LLVM 编译器环境的 OpenMP 卸载。请注意，该工具特定于 OpenACC C，而 OpenACC Fortran 已在 AMD GPU 上得到支持。如[GitHub 存储库](https://github.com/llvm-doe-org/llvm-project/tree/clacc/main)中所示，编译器`Clacc`是描述如下`\install`目录`\bin`子目录中的`Clang`的可执行文件。

在以下内容中，我们提供了一个逐步指南来构建和使用 Clacc：

+   **步骤 1**：构建和安装[Clacc](https://github.com/llvm-doe-org/llvm-project/tree/clacc/main)。

    ```
    $ git  clone  -b  clacc/main  https://github.com/llvm-doe-org/llvm-project.git
    $ cd  llvm-project
    $ mkdir  build  &&  cd  build
    $ cmake  -DCMAKE_INSTALL_PREFIX=../install  \
      -DCMAKE_BUILD_TYPE=Release  \
      -DLLVM_ENABLE_PROJECTS="clang;lld"  \
      -DLLVM_ENABLE_RUNTIMES=openmp  \
      -DLLVM_TARGETS_TO_BUILD="host;AMDGPU"  \
      -DCMAKE_C_COMPILER=gcc  \
      -DCMAKE_CXX_COMPILER=g++  \
      ../llvm
    $ make
    $ make  install 
    ```

+   **步骤 2**：设置环境变量以便能够从`/install`目录工作，这是最简单的方法。我们假设`/install`目录位于路径`/project/project_xxxxxx/Clacc/llvm-project`中。

对于更高级的使用，例如修改`Clacc`，我们建议读者参考[“从构建目录使用”](https://github.com/llvm-doe-org/llvm-project/blob/clacc/main/README.md)

> ```
> $ export  PATH=/scratch/project_465002387/clacc/llvm-project/install/bin:$PATH
> $ export  LD_LIBRARY_PATH=/scratch/project_465002387/clacc/llvm-project/install/lib:$LD_LIBRARY_PATH 
> ```

+   **步骤 3**：将 openACC_code.c 代码源到源转换为要打印到文件 openMP_code.c 中的代码：

    ```
    $ clang  -fopenacc-print=omp  -fopenacc-structured-ref-count-omp=no-ompx-hold  openACC_code.c  >  openMP_code.c 
    ```

    这里引入了标志`-fopenacc-structured-ref-count-omp=no-ompx-hold`来禁用`ompx_hold`映射类型修饰符，该修饰符用于 OpenACC `copy`子句的转换。`ompx_hold`是 OpenMP 扩展，可能尚未被其他编译器支持。

+   **步骤 4**：使用[cc 编译器包装器](https://docs.lumi-supercomputer.eu/development/compiling/prgenv/)编译代码

    ```
    module load CrayEnv
    module load PrgEnv-cray
    module load craype-accel-amd-gfx90a
    module load rocm/6.0.3

    cc -fopenmp -o executable openMP_code.c 
    ```

访问练习材料

可以通过在内容目录`content/examples/exercise_clacc`下克隆此存储库来访问`Clacc`练习的代码示例：

```
$ git  clone  https://github.com/ENCCS/gpu-programming.git
$ cd  gpu-programming/content/examples/exercise_clacc
$ ls 
```

练习：将 OpenACC 代码转换为 OpenMP

1.  使用`Clacc`编译器将位于`/exercise_clacc`的 OpenACC 代码`openACC_code.c`转换为 HIP。

1.  使用`cc`编译器包装器编译生成的 OpenMP 代码并运行它。

### 使用 SYCLomatic 将 CUDA 转换为 SYCL/DPC++

Intel 提供了一种 CUDA 到 SYCL 代码迁移的工具，包含在 Intel oneAPI Basekit 中。

Clacc 没有安装在 LUMI 上，但总体工作流程与 HIPify Clang 类似，也要求存在现有的 CUDA 安装：

> ```
> $ dpct  program.cu
> $ cd  dpct_output/
> $ icpx  -fsycl  program.dp.cpp 
> ```

SYCLomatic 可以通过使用`-in-root`和`-out-root`标志递归处理目录来迁移更大的项目。它还可以使用编译数据库（由 CMake 和其他构建系统支持）来处理更复杂的项目布局。

请注意，由 SYCLomatic 生成的代码依赖于 oneAPI 特定扩展，因此不能直接与其他 SYCL 实现一起使用，例如 AdaptiveCpp（hipSYCL）。可以将 `--no-incremental-migration` 标志添加到 `dpct` 命令中，以最小化但不完全避免使用此兼容层。这将需要手动操作，因为某些 CUDA 概念不能直接映射到 SYCL。

此外，CUDA 应用程序可能假设某些硬件行为，例如 32 宽的 warp。如果目标硬件不同（例如，LUMI 中使用的 AMD MI250 GPU，warp 的大小为 64），则算法可能需要手动调整。

### 结论

这就结束了关于使用现有工具将 CUDA 代码转换为 HIP 和 SYCL，以及将 OpenACC 代码转换为 OpenMP 载荷的简要概述。一般来说，大型应用程序的转换过程可能是不完整的，因此需要手动修改以完成移植过程。然而，值得注意的是，转换过程的准确性要求应用程序根据 CUDA 和 OpenACC 语法正确编写。

### 使用 Hipify 将 CUDA 转换为 HIP

在本节中，我们将介绍如何使用 `hipify-perl` 和 `hipify-clang` 工具将 CUDA 代码转换为 HIP。

#### Hipify-perl

`hipify-perl` 工具是一个基于 perl 的脚本，它将 CUDA 语法转换为 HIP 语法（更多详情请参见[此处](https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl)）。例如，在一个包含 CUDA 函数 `cudaMalloc` 和 `cudaDeviceSynchronize` 的 CUDA 代码中，该工具会将 `cudaMalloc` 替换为 HIP 函数 `hipMalloc`。同样，CUDA 函数 `cudaDeviceSynchronize` 也会被替换为 HIP 函数 `hipDeviceSynchronize`。以下是在 LUMI-G 上运行 `hipify-perl` 的基本步骤。

+   **步骤 1**：生成 `hipify-perl` 脚本

    ```
    $ module  load  rocm/6.0.3
    $ hipify-clang  --perl 
    ```

+   **步骤 2**：运行生成的 `hipify-perl`

    ```
    $ hipify-perl  program.cu  >  program.cu.hip 
    ```

+   **步骤 3**：使用 `hipcc` 编译生成的 HIP 代码

    ```
    $ hipcc  --offload-arch=gfx90a  -o  program.hip.exe  program.cu.hip 
    ```

尽管使用 `hipify-perl` 的操作很简单，但这个工具可能不适合大型应用程序，因为它主要依赖于将 CUDA 字符串替换为 HIP 字符串（例如，将 `*cuda*` 替换为 `*hip*`）。此外，`hipify-perl` 缺乏区分设备/主机函数调用的能力[参见 HIPify 参考指南](https://docs.amd.com/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl)。在这种情况下，可以使用下一节中将要描述的 `hipify-clang` 工具。

#### Hipify-clang

如 [HIPIFY 文档](https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl) 所述，`hipify-clang` 工具基于 clang，用于将 CUDA 源代码转换为 HIP 源代码。与 `hipify-perl` 工具相比，该工具在翻译 CUDA 代码方面更为稳健。此外，它通过提供辅助功能来促进代码的分析。

简而言之，`hipify-clang` 需要 `LLVM+CLANG` 和 `CUDA`。有关构建 `hipify-clang` 的详细信息，请参阅[此处](https://github.com/ROCm/HIPIFY)。请注意，`hipify-clang` 可在 LUMI-G 上使用。然而，问题可能与管理 CUDA-toolkit 的安装有关。为了避免安装过程中可能出现的任何问题，我们选择使用 CUDA singularity 容器。以下是如何运行 `hipify-clang` 的分步指南：

+   **步骤 1**：拉取 CUDA singularity 容器，例如

    ```
    $ singularity  pull  docker://nvcr.io/nvidia/cuda:11.4.3-devel-ubuntu20.04 
    ```

+   **步骤 2**：加载 rocm 模块并启动 CUDA singularity

    ```
    $ module  load  rocm/6.0.3
    $ singularity  shell  -B  $PWD,/opt:/opt  cuda_11.4.0-devel-ubuntu20.04.sif 
    ```

    其中主机当前目录 `$PWD` 映射到容器中的相应目录，主机中的 `/opt` 目录映射到容器内的相应目录。

+   **步骤 3**：设置环境变量 `$PATH`。为了在容器内运行 `hipify-clang`，可以设置环境变量 `$PATH`，该变量定义了查找二进制文件 `hipify-clang` 的路径。

    ```
    $ export  PATH=/opt/rocm-6.0.3/bin:$PATH 
    ```

    注意，我们使用的 ROCm 版本是 `rocm-6.0.3`。

+   **步骤 4**：在 singularity 容器内运行 `hipify-clang`

    ```
    $ hipify-clang  program.cu  -o  hip_program.cu.hip  --cuda-path=/usr/local/cuda-11.4  -I  /usr/local/cuda-11.4/include 
    ```

    在这里，应指定 cuda 路径以及 `*includes*` 和 `*defines*` 文件的路经。CUDA 源代码和生成的输出代码分别是 program.cu 和 hip_program.cu.hip。

    生成 hip 代码的编译过程语法与上一节中描述的类似（请参阅 hipify-perl 节的 **步骤 3**）。

可以通过克隆此存储库在内容/示例/exercise_hipify 子目录中访问 `Hipify` 练习的代码示例：

> ```
> $ git  clone  https://github.com/ENCCS/gpu-programming.git
> $ cd  gpu-programming/content/examples/exercise_hipify
> $ ls 
> ```

练习 I：使用 `hipify-perl` 将 CUDA 代码翻译成 HIP

1.1 生成 `hipify-perl` 工具。

1.2 使用 `Hipify-perl` 工具将位于 `/exercise_hipify/Hipify_perl` 的 CUDA 代码 `vec_add_cuda.cu` 转换为 HIP。

1.3 使用 `hipcc` 编译器包装器编译生成的 HIP 代码并运行它。

练习 II：使用 `hipify-clang` 将 CUDA 代码翻译成 HIP

2.1 使用 `Hipify-clang` 工具将位于 `/exercise_hipify/Hipify_clang` 的 CUDA 代码 `vec_add_cuda.cu` 转换为 HIP。

2.2 使用 `hipcc` 编译器包装器编译生成的 HIP 代码并运行它。

#### Hipify-perl

`hipify-perl` 工具是一个基于 perl 的脚本，它将 CUDA 语法转换为 HIP 语法（例如，[此处](https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl)提供了更多详细信息）。例如，在一个包含 CUDA 函数 `cudaMalloc` 和 `cudaDeviceSynchronize` 的 CUDA 代码中，该工具将 `cudaMalloc` 替换为 HIP 函数 `hipMalloc`。同样，CUDA 函数 `cudaDeviceSynchronize` 将被替换为 HIP 函数 `hipDeviceSynchronize`。以下是在 LUMI-G 上运行 `hipify-perl` 的基本步骤。

+   **步骤 1**：生成 `hipify-perl` 脚本

    ```
    $ module  load  rocm/6.0.3
    $ hipify-clang  --perl 
    ```

+   **步骤 2**：运行生成的 `hipify-perl`

    ```
    $ hipify-perl  program.cu  >  program.cu.hip 
    ```

+   **步骤 3**：使用 `hipcc` 编译器包装器编译生成的 HIP 代码

    ```
    $ hipcc  --offload-arch=gfx90a  -o  program.hip.exe  program.cu.hip 
    ```

尽管使用 `hipify-perl` 简单，但该工具可能不适合大型应用程序，因为它严重依赖于用 HIP 字符串替换 CUDA 字符串（例如，将 `*cuda*` 替换为 `*hip*`）。此外，`hipify-perl` 缺乏区分设备/主机函数调用的能力。[区分设备/主机函数调用](https://docs.amd.com/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl)。这里的替代方案是使用下一节将要描述的 `hipify-clang` 工具。

#### Hipify-clang

如 [HIPIFY 文档](https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl) 所述，`hipify-clang` 工具基于 clang，用于将 CUDA 源代码转换为 HIP 源代码。与 `hipify-perl` 工具相比，该工具在转换 CUDA 代码方面更为健壮。此外，它通过提供辅助功能来促进代码分析。

简而言之，`hipify-clang` 需要 `LLVM+CLANG` 和 `CUDA`。有关构建 `hipify-clang` 的详细信息，请参阅[此处](https://github.com/ROCm/HIPIFY)。请注意，`hipify-clang` 可在 LUMI-G 上使用。然而，问题可能与管理 CUDA-toolkit 的安装有关。为了避免安装过程中的任何潜在问题，我们选择使用 CUDA singularity 容器。在此，我们提供了一个逐步指南，用于运行 `hipify-clang`：

+   **步骤 1**：拉取 CUDA singularity 容器，例如

    ```
    $ singularity  pull  docker://nvcr.io/nvidia/cuda:11.4.3-devel-ubuntu20.04 
    ```

+   **步骤 2**：加载 ROCm 模块并启动 CUDA singularity

    ```
    $ module  load  rocm/6.0.3
    $ singularity  shell  -B  $PWD,/opt:/opt  cuda_11.4.0-devel-ubuntu20.04.sif 
    ```

    其中，主机当前目录 `$PWD` 映射到容器的相应目录，主机中的 `/opt` 目录映射到容器内部的相应目录。

+   **步骤 3**：设置环境变量 `$PATH`。为了在容器内部运行 `hipify-clang`，可以设置环境变量 `$PATH`，该变量定义了查找二进制文件 `hipify-clang` 的路径。

    ```
    $ export  PATH=/opt/rocm-6.0.3/bin:$PATH 
    ```

    注意，我们使用的 ROCm 版本是 `rocm-6.0.3`。

+   **步骤 4**：在 singularity 容器内部运行 `hipify-clang`

    ```
    $ hipify-clang  program.cu  -o  hip_program.cu.hip  --cuda-path=/usr/local/cuda-11.4  -I  /usr/local/cuda-11.4/include 
    ```

    这里应指定 cuda 路径以及 `*includes*` 和 `*defines*` 文件的路经。CUDA 源代码和生成的输出代码分别是 program.cu 和 hip_program.cu.hip。

    生成的 HIP 代码的编译过程语法与上一节中描述的类似（参见 hipify-perl 节的 **步骤 3**）。

可以通过克隆此存储库在内容目录下的 examples/exercise_hipify 子目录中访问 `Hipify` 练习的代码示例：

> ```
> $ git  clone  https://github.com/ENCCS/gpu-programming.git
> $ cd  gpu-programming/content/examples/exercise_hipify
> $ ls 
> ```

练习 I：使用 `hipify-perl` 将 CUDA 代码转换为 HIP

1.1 生成 `hipify-perl` 工具。

1.2 使用 `Hipify-perl` 工具将位于 `/exercise_hipify/Hipify_perl` 的 CUDA 代码 `vec_add_cuda.cu` 转换为 HIP。

1.3 使用 `hipcc` 编译器包装器编译生成的 HIP 代码并运行。

练习 II：使用 `hipify-clang` 将 CUDA 代码转换为 HIP

2.1 使用 `Hipify-clang` 工具将位于 `/exercise_hipify/Hipify_clang` 的 CUDA 代码 `vec_add_cuda.cu` 转换为 HIP。

2.2 使用 `hipcc` 编译器包装器编译生成的 HIP 代码并运行它。

### 使用 Clacc 将 OpenACC 转换为 OpenMP

[Clacc](https://github.com/llvm-doe-org/llvm-project/tree/clacc/main) 是一个工具，可以将 OpenACC 应用程序转换为使用 Clang/LLVM 编译器环境的 OpenMP 转发。请注意，该工具特定于 OpenACC C，而 OpenACC Fortran 已经在 AMD GPU 上得到支持。如 [GitHub 存储库](https://github.com/llvm-doe-org/llvm-project/tree/clacc/main) 中所示，编译器 `Clacc` 是 `\install` 目录下 `\bin` 子目录中的 `Clang` 可执行文件，如下所述。

在以下内容中，我们提供了一个逐步指南，用于构建和使用 Clacc：

+   **步骤 1**：构建和安装 [Clacc](https://github.com/llvm-doe-org/llvm-project/tree/clacc/main)。

    ```
    $ git  clone  -b  clacc/main  https://github.com/llvm-doe-org/llvm-project.git
    $ cd  llvm-project
    $ mkdir  build  &&  cd  build
    $ cmake  -DCMAKE_INSTALL_PREFIX=../install  \
      -DCMAKE_BUILD_TYPE=Release  \
      -DLLVM_ENABLE_PROJECTS="clang;lld"  \
      -DLLVM_ENABLE_RUNTIMES=openmp  \
      -DLLVM_TARGETS_TO_BUILD="host;AMDGPU"  \
      -DCMAKE_C_COMPILER=gcc  \
      -DCMAKE_CXX_COMPILER=g++  \
      ../llvm
    $ make
    $ make  install 
    ```

+   **步骤 2**：设置环境变量，以便能够从 `/install` 目录工作，这是最简单的方法。我们假设 `/install` 目录位于路径 `/project/project_xxxxxx/Clacc/llvm-project`。

对于更高级的使用，例如修改 `Clacc`，我们建议读者参考[“从构建目录使用”](https://github.com/llvm-doe-org/llvm-project/blob/clacc/main/README.md)。

> ```
> $ export  PATH=/scratch/project_465002387/clacc/llvm-project/install/bin:$PATH
> $ export  LD_LIBRARY_PATH=/scratch/project_465002387/clacc/llvm-project/install/lib:$LD_LIBRARY_PATH 
> ```

+   **步骤 3**：将 openACC_code.c 代码从源代码转换为要打印到 openMP_code.c 文件的代码：

    ```
    $ clang  -fopenacc-print=omp  -fopenacc-structured-ref-count-omp=no-ompx-hold  openACC_code.c  >  openMP_code.c 
    ```

    这里引入了标志 `-fopenacc-structured-ref-count-omp=no-ompx-hold` 来禁用 `ompx_hold` 映射类型修饰符，该修饰符用于 OpenACC `copy` 子句的翻译。`ompx_hold` 是一个可能尚未被其他编译器支持的 OpenMP 扩展。

+   **步骤 4** 使用 [cc 编译器包装器](https://docs.lumi-supercomputer.eu/development/compiling/prgenv/) 编译代码

    ```
    module load CrayEnv
    module load PrgEnv-cray
    module load craype-accel-amd-gfx90a
    module load rocm/6.0.3

    cc -fopenmp -o executable openMP_code.c 
    ```

访问练习材料

可以通过在内容/示例/exercise_clacc 子目录中克隆此存储库来访问 `Clacc` 练习的代码示例：

```
$ git  clone  https://github.com/ENCCS/gpu-programming.git
$ cd  gpu-programming/content/examples/exercise_clacc
$ ls 
```

练习：将 OpenACC 代码转换为 OpenMP

1.  使用 `Clacc` 编译器将位于 `/exercise_clacc` 的 OpenACC 代码 `openACC_code.c` 转换。

1.  使用 `cc` 编译器包装器编译生成的 OpenMP 代码并运行它。

### 使用 SYCLomatic 将 CUDA 转换为 SYCL/DPC++

英特尔提供了一种将 CUDA 转换为 SYCL 代码的工具，包含在 Intel oneAPI Basekit 中。

它没有安装在 LUMI 上，但一般的工作流程与 HIPify Clang 类似，也要求有一个现有的 CUDA 安装：

> ```
> $ dpct  program.cu
> $ cd  dpct_output/
> $ icpx  -fsycl  program.dp.cpp 
> ```

SYCLomatic 可以通过使用 `-in-root` 和 `-out-root` 标志递归地处理目录来迁移较大的项目。它还可以使用编译数据库（由 CMake 和其他构建系统支持）来处理更复杂的项目布局。

请注意，由 SYCLomatic 生成的代码依赖于 oneAPI 特定扩展，因此不能直接与其他 SYCL 实现一起使用，例如 AdaptiveCpp (hipSYCL)。可以将 `--no-incremental-migration` 标志添加到 `dpct` 命令中，以最小化，但不是完全避免使用此兼容层。这将需要手动工作，因为某些 CUDA 概念不能直接映射到 SYCL。

此外，CUDA 应用程序可能假设某些硬件行为，例如 32 宽的 warp。如果目标硬件不同（例如，LUMI 中使用的 AMD MI250 GPU，warp 的大小为 64），算法可能需要手动调整。

### 结论

这总结了使用现有工具将 CUDA 代码转换为 HIP 和 SYCL，以及将 OpenACC 代码转换为 OpenMP 转发的简要概述。一般来说，大型应用程序的转换过程可能不完整，因此需要手动修改以完成迁移过程。然而，值得注意的是，转换过程的准确性要求应用程序根据 CUDA 和 OpenACC 语法正确编写。

## 参见

+   [Hipify GitHub](https://github.com/ROCm/HIPIFY)

+   [HIPify 参考指南 v5.1](https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html)

+   [HIP 示例](https://github.com/olcf-tutorials/simple_HIP_examples/tree/master/vector_addition)

+   [将 CUDA 迁移到 HIP](https://www.admin-magazine.com/HPC/Articles/Porting-CUDA-to-HIP)

+   [Clacc 主仓库 README](https://github.com/llvm-doe-org/llvm-project/blob/clacc/main/README.md)

+   [SYCLomatic 主页](https://www.intel.com/content/www/us/en/developer/articles/technical/syclomatic-new-cuda-to-sycl-code-migration-tool.html)

+   [SYCLomatic 文档](https://oneapi-src.github.io/SYCLomatic/get_started/index.html)

关键点

+   存在着一些有用的工具，可以自动将工具从 CUDA 转换为 HIP 和 SYCL，以及从 OpenACC 转换为 OpenMP，但它们可能需要手动修改*。

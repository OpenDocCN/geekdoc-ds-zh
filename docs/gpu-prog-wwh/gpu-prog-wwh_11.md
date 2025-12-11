# 基于内核的可移植模型

> [`enccs.github.io/gpu-programming/8-portable-kernel-models/`](https://enccs.github.io/gpu-programming/8-portable-kernel-models/)

*GPU 编程：为什么、何时以及如何？* **   基于内核的可移植模型

+   [在 GitHub 上编辑](https://github.com/ENCCS/gpu-programming/blob/main/content/8-portable-kernel-models.rst)

* * *

问题

+   如何使用 alpaka、C++ StdPar、Kokkos、OpenCL 和 SYCL 来编程 GPU？

+   这些编程模型之间有什么区别。

目标

+   能够使用基于内核的可移植模型编写简单代码

+   理解 Kokkos 和 SYCL 中关于内存和同步的不同方法是如何工作的

教师备注

+   60 分钟教学

+   30 分钟练习

跨平台可移植生态系统目标是在多个架构上运行相同的代码，从而减少代码重复。它们通常基于 C++，并使用函数对象/lambda 函数来定义循环体（即内核），这些可以在多个架构上运行，如 CPU、GPU 和来自不同供应商的 FPGA。OpenCL 是这一规则的例外，它最初只提供 C API（尽管目前也提供了 C++ API），并为内核代码使用单独的源模型。然而，与许多传统的 CUDA 或 HIP 实现不同，可移植生态系统要求如果用户希望它在 CPU 和 GPU 上运行，则内核只需编写一次。一些值得注意的跨平台可移植生态系统包括 alpaka、Kokkos、OpenCL、RAJA 和 SYCL。Kokkos、alpaka 和 RAJA 是独立的项目，而 OpenCL 和 SYCL 是多个项目遵循的标准，这些项目实现了（并扩展了）它们。例如，一些值得注意的 SYCL 实现包括[Intel oneAPI DPC++](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html)、[AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp/)（之前称为 hipSYCL 或 Open SYCL）、[triSYCL](https://github.com/triSYCL/triSYCL)和[ComputeCPP](https://developer.codeplay.com/products/computecpp/ce/home/)。

## C++ StdPar

在 C++17 中，引入了对标准算法并行执行的支持。大多数通过标准 `<algorithms>` 头文件提供的算法都被赋予了接受[*执行策略*](https://en.cppreference.com/w/cpp/algorithm)参数的重载，这使得程序员可以请求并行执行标准库函数。虽然主要目标是允许低成本的、高级接口运行现有的算法，如`std::sort`在多个 CPU 核心上，但实现允许使用其他硬件，并且函数如`std::for_each`或`std::transform`在编写算法时提供了极大的灵活性。

C++ StdPar，也称为并行 STL 或 PSTL，可以被认为是类似于指令驱动的模型，因为它非常高级，并且不给予程序员对数据移动或对硬件特定功能（如共享（局部）内存）的精细控制。甚至要运行的 GPU 也是自动选择的，因为标准 C++ 没有概念上的 *device*（但有一些供应商扩展允许程序员有更多的控制）。然而，对于已经依赖于 C++ 标准库算法的应用程序，StdPar 可以是一种在最小代码修改的情况下获得 CPU 和 GPU 性能优势的好方法。

对于 GPU 编程，所有三个供应商都提供了他们的 StdPar 实现，可以将代码卸载到 GPU 上：NVIDIA 有 `nvc++`，AMD 有实验性的 [roc-stdpar](https://github.com/ROCm/roc-stdpar)，Intel 通过他们的 oneAPI 编译器提供 StdPar 卸载。[AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp/) 提供了一个独立的 StdPar 实现，能够针对所有三个供应商的设备。虽然 StdPar 是 C++ 标准的一部分，但不同编译器之间对 StdPar 的支持程度和成熟度差异很大：并非所有编译器都支持所有算法，将算法映射到硬件和进行数据移动的不同启发式方法可能会影响性能。

### StdPar 编译

构建过程很大程度上取决于所使用的编译器：

+   AdaptiveCpp：在调用 `acpp` 时添加 `--acpp-stdpar` 标志。

+   Intel oneAPI：在调用 `icpx` 时添加 `-fsycl -fsycl-pstl-offload=gpu` 标志。

+   NVIDIA NVC++：在调用 `nvc++` 时添加 `-stdpar` 标志（不支持使用普通 `nvcc`）。

### StdPar 编程

在其最简单的形式中，使用 C++ 标准并行性需要包含一个额外的 `<execution>` 头文件，并将一个参数添加到支持的标准库函数中。

例如，让我们看看以下按顺序排序向量的代码：

```
#include  <algorithm>
#include  <vector>

void  f(std::vector<int>&  a)  {
  std::sort(a.begin(),  a.end());
} 
```

要使其在 GPU 上运行排序操作，只需要进行微小的修改：

```
#include  <algorithm>
#include  <vector>
#include  <execution> // To get std::execution

void  f(std::vector<int>&  a)  {
  std::sort(
  std::execution::par_unseq,  // This algorithm can be run in parallel
  a.begin(),  a.end()
  );
} 
```

现在，当使用支持的编译器编译时，代码将在 GPU 上运行排序操作。

虽然一开始可能看起来非常有限，但许多标准算法，如 `std::transform`、`std::accumulate`、`std::transform_reduce` 和 `std::for_each`，可以在数组上运行自定义函数，从而允许将任意算法卸载，只要它不违反 GPU 内核的典型限制，例如不抛出任何异常和不进行系统调用。

### StdPar 执行策略

在 C++ 中，有四种不同的执行策略可供选择：

+   `std::execution::seq`: 串行运行算法，不进行并行化。

+   `std::execution::par`: 允许并行化算法（类似于使用多个线程），

+   `std::execution::unseq`: 允许向量化算法（类似于使用 SIMD），

+   `std::execution::par_unseq`: 允许同时向量化和平行化算法。

`par` 和 `unseq` 之间的主要区别与线程进度和锁有关：使用 `unseq` 或 `par_unseq` 要求算法在进程之间不包含互斥锁和其他锁，而 `par` 没有这样的限制。

对于 GPU，最佳选择是 `par_unseq`，因为它对编译器在操作顺序方面的要求最低。虽然 `par` 在某些情况下也受到支持，但最好避免使用它，这不仅因为编译器支持有限，而且也表明算法可能不适合硬件。

## Kokkos

Kokkos 是一个开源的性能可移植生态系统，用于在大型的异构硬件架构上并行化，其开发主要在桑迪亚国家实验室进行。该项目始于 2011 年，作为一个并行 C++ 编程模型，但后来扩展成为一个更广泛的生态系统，包括 Kokkos Core（编程模型）、Kokkos Kernels（数学库）和 Kokkos Tools（调试、分析和调优工具）。通过为 C++ 标准委员会准备提案，该项目还旨在影响 ISO/C++ 语言标准，以便最终 Kokkos 的功能将成为语言标准的原生功能。更详细的介绍可以在[这里](https://www.sandia.gov/news/publications/hpc-annual-reports/article/kokkos/)找到。

Kokkos 库为多种不同的并行编程模型提供了一个抽象层，目前包括 CUDA、HIP、SYCL、HPX、OpenMP 和 C++ 线程。因此，它允许在不同厂商制造的硬件之间实现更好的可移植性，但同时也给软件堆栈引入了额外的依赖。例如，当使用 CUDA 时，只需要 CUDA 安装即可，但当使用 Kokkos 与 NVIDIA GPU 一起时，则需要 Kokkos 和 CUDA 的安装。Kokkos 并不是并行编程中非常受欢迎的选择，因此，与更成熟的编程模型（如 CUDA）相比，学习和使用 Kokkos 可能会更加困难，因为关于 CUDA 的搜索结果和 Stack Overflow 讨论的数量要多得多。

### Kokkos 编译

此外，一些跨平台可移植性库的挑战之一是，即使在同一系统上，不同的项目可能也需要为可移植性库设置不同的编译组合。例如，在 Kokkos 中，一个项目可能希望默认的执行空间是 CUDA 设备，而另一个项目则可能需要 CPU。即使项目偏好相同的执行空间，一个项目可能希望统一内存成为默认的内存空间，而另一个项目可能希望使用固定 GPU 内存。在单个系统上维护大量库实例可能会变得很麻烦。

然而，Kokkos 提供了一种简单的方法来同时编译 Kokkos 库和用户项目。这是通过指定 Kokkos 编译设置（见[这里](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Compiling.html)）并在用户 Makefile 中包含 Kokkos Makefile 来实现的。CMake 也受到支持。这样，用户应用程序和 Kokkos 库一起编译。以下是一个使用 CUDA（Volta 架构）作为后端（默认执行空间）和统一内存作为默认内存空间的单文件 Kokkos 项目（hello.cpp）的示例 Makefile：

```
default:  build

# Set compiler
KOKKOS_PATH  =  $(shell  pwd)/kokkos
CXX  =  hipcc
# CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper

# Variables for the Makefile.kokkos
KOKKOS_DEVICES  =  "HIP"
# KOKKOS_DEVICES = "Cuda"
KOKKOS_ARCH  =  "VEGA90A"
# KOKKOS_ARCH = "Volta70"
KOKKOS_CUDA_OPTIONS  =  "enable_lambda,force_uvm"

# Include Makefile.kokkos
include $(KOKKOS_PATH)/Makefile.kokkos

build:  $(KOKKOS_LINK_DEPENDS) $(KOKKOS_CPP_DEPENDS) hello.cpp
  $(CXX)  $(KOKKOS_CPPFLAGS)  $(KOKKOS_CXXFLAGS)  $(KOKKOS_LDFLAGS)  hello.cpp  $(KOKKOS_LIBS)  -o  hello 
```

要使用上述 Makefile 构建 **hello.cpp** 项目，除了将 Kokkos 项目克隆到当前目录外，不需要进行其他步骤。

### Kokkos 编程

当开始使用 Kokkos 编写项目时，第一步是了解 Kokkos 的初始化和终止。Kokkos 必须通过调用 `Kokkos::initialize(int& argc, char* argv[])` 来初始化，并通过调用 `Kokkos::finalize()` 来终止。更多详情请参阅[这里](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Initialization.html)。

Kokkos 使用执行空间模型来抽象并行硬件的细节。执行空间实例映射到可用的后端选项，如 CUDA、OpenMP、HIP 或 SYCL。如果程序员在源代码中没有明确选择执行空间，则使用默认的执行空间 `Kokkos::DefaultExecutionSpace`。这是在编译 Kokkos 库时选择的。Kokkos 的执行空间模型在[这里](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Machine-Model.html#kokkos-spaces)有更详细的描述。

类似地，Kokkos 使用内存空间模型来处理不同类型的内存，例如主机内存或设备内存。如果没有明确定义，Kokkos 将使用在 Kokkos 编译期间指定的默认内存空间，具体请参阅[这里](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Machine-Model.html#kokkos-memory-spaces)。

以下是一个初始化 Kokkos 并打印执行空间和内存空间实例的 Kokkos 程序示例：

```
#include  <Kokkos_Core.hpp>
#include  <iostream>

int  main(int  argc,  char*  argv[])  {
  Kokkos::initialize(argc,  argv);
  std::cout  <<  "Execution Space: "  <<
  typeid(Kokkos::DefaultExecutionSpace).name()  <<  std::endl;
  std::cout  <<  "Memory Space: "  <<
  typeid(Kokkos::DefaultExecutionSpace::memory_space).name()  <<  std::endl;
  Kokkos::finalize();
  return  0;
} 
```

使用 Kokkos，数据可以通过原始指针或通过 Kokkos 视图来访问。使用原始指针时，可以将内存分配到默认内存空间，使用 `Kokkos::kokkos_malloc(n * sizeof(int))`。Kokkos 视图是一种数据类型，它提供了一种更有效地访问与特定 Kokkos 内存空间（如主机内存或设备内存）对应的数据的方法。可以通过 `Kokkos::View<int*> a("a", n)` 创建一个类型为 int* 的一维视图，其中 `"a"` 是标签，`n` 是以整数数量表示的分配大小。Kokkos 在编译时确定数据的最佳布局，以获得最佳的整体性能，这取决于计算机架构。此外，Kokkos 会自动处理此类内存的释放。有关 Kokkos 视图的更多详细信息，请参阅[这里](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/View.html)。

最后，Kokkos 提供了三种不同的并行操作：`parallel_for`、`parallel_reduce` 和 `parallel_scan`。`parallel_for` 操作用于并行执行循环。`parallel_reduce` 操作用于并行执行循环并将结果归约到单个值。`parallel_scan` 操作实现了前缀扫描。`parallel_for` 和 `parallel_reduce` 的用法将在本章后面的示例中演示。有关并行操作的更多详细信息，请参阅[这里](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/ParallelDispatch.html)。

### 简单步骤运行 Kokkos hello.cpp 示例

以下内容在 AMD VEGA90A 设备上直接使用应该可以工作（需要 ROCm 安装）。在 NVIDIA Volta V100 设备上（需要 CUDA 安装），请使用 Makefile 中注释掉的变量。

1.  `git clone https://github.com/kokkos/kokkos.git`

1.  将上面的 Makefile 复制到当前文件夹中（确保最后一行的缩进是制表符，而不是空格）

1.  将上面的 hello.cpp 文件复制到当前文件夹

1.  `make`

1.  `./hello`

## OpenCL

OpenCL 是一个跨平台的开放标准 API，用于编写在由 CPU、GPU、FPGA 和其他设备组成的异构平台上执行的并行程序。OpenCL 的第一个版本（1.0）于 2008 年 12 月发布，OpenCL 的最新版本（3.0）于 2020 年 9 月发布。OpenCL 获得了包括 AMD、ARM、Intel、NVIDIA 和 Qualcomm 在内的许多厂商的支持。它是一个免版税标准，OpenCL 规范由 Khronos Group 维护。OpenCL 提供了一个基于 C 的低级编程接口，但最近也提供了一种 C++ 接口。

### OpenCL 编译

OpenCL 支持两种编译程序的模式：在线和离线。在线编译发生在运行时，当宿主程序调用一个函数来编译源代码时。在线模式允许动态生成和加载内核，但可能会因为编译时间和可能的错误而带来一些开销。离线编译发生在运行之前，当内核的源代码被编译成可以被宿主程序加载的二进制格式时。这种模式允许更快的执行和更好的内核优化，但可能会限制程序的移植性，因为二进制文件只能在编译时指定的架构上运行。

OpenCL 随带几个并行编程生态系统，例如 NVIDIA CUDA 和 Intel oneAPI。例如，在成功安装这些包并设置环境之后，可以通过如 `icx cl_devices.c -lOpenCL`（Intel oneAPI）或 `nvcc cl_devices.c -lOpenCL`（NVIDIA CUDA）这样的命令简单地编译一个 OpenCL 程序，其中 `cl_devices.c` 是编译后的文件。与大多数其他编程模型不同，OpenCL 将内核存储为文本，并在运行时（即时编译）为设备编译，因此不需要任何特殊的编译器支持：只要所需的库和头文件安装在了标准位置，就可以使用 `gcc cl_devices.c -lOpenCL`（或使用 C++ API 时的 `g++`）来编译代码。

安装在 LUMI 上的 AMD 编译器支持 OpenCL C 和 C++ API，后者有一些限制。要编译程序，您可以使用 GPU 分区的 AMD 编译器：

```
$ module  load  LUMI/24.03  partition/G
$ module  load  rocm/6.0.3
$ module  load  PrgEnv-cray-amd
$ CC  program.cpp  -lOpenCL  -o  program  # C++ program
$ cc  program.c  -lOpenCL  -o  program  # C program 
```

### OpenCL 编程

OpenCL 程序由两部分组成：在主机设备（通常是 CPU）上运行的宿主程序和在一个或多个计算设备（如 GPU）上运行的内核。宿主程序负责管理所选平台上的设备、分配内存对象、构建和排队内核以及管理内存对象。

编写 OpenCL 程序的第一步是初始化 OpenCL 环境，通过选择平台和设备，创建与所选设备关联的上下文或上下文，并为每个设备创建一个命令队列。选择默认设备、创建与设备关联的上下文和队列的简单示例如下。

```
// Initialize OpenCL
cl::Device  device  =  cl::Device::getDefault();
cl::Context  context(device);
cl::CommandQueue  queue(context,  device); 
```

```
// Initialize OpenCL
cl_int  err;  // Error code returned by API calls
cl_platform_id  platform;
err  =  clGetPlatformIDs(1,  &platform,  NULL);
assert(err  ==  CL_SUCCESS);  // Checking error codes is skipped later for brevity
cl_device_id  device;
err  =  clGetDeviceIDs(platform,  CL_DEVICE_TYPE_GPU,  1,  &device,  NULL);
cl_context  context  =  clCreateContext(NULL,  1,  &device,  NULL,  NULL,  &err);
cl_command_queue  queue  =  clCreateCommandQueue(context,  device,  0,  &err); 
```

OpenCL 提供了两种主要的编程模型来管理主机和加速设备设备的内存层次结构：缓冲区和共享虚拟内存（SVM）。缓冲区是 OpenCL 的传统内存模型，其中主机和设备有独立的地址空间，程序员必须显式指定内存分配以及如何以及在哪里访问内存。这可以通过`cl::Buffer`类和如`cl::CommandQueue::enqueueReadBuffer()`等函数来完成。缓冲区自 OpenCL 的早期版本以来就得到了支持，并且在不同架构上运行良好。缓冲区还可以利用特定于设备的内存功能，如常量或局部内存。

SVM 是 OpenCL 中的一种较新的内存模型，在 2.0 版本中引入，其中主机和设备共享单个虚拟地址空间。因此，程序员可以使用相同的指针从主机和设备访问数据，从而简化编程工作。在 OpenCL 中，SVM 有不同级别，如粗粒度缓冲区 SVM、细粒度缓冲区 SVM 和细粒度系统 SVM。所有级别都允许在主机和设备之间使用相同的指针，但它们在内存区域的粒度和同步要求上有所不同。此外，SVM 的支持并非在所有 OpenCL 平台和设备上都是通用的，例如，NVIDIA V100 和 A100 这样的 GPU 仅支持粗粒度 SVM 缓冲区。这一级别需要显式同步从主机和设备对内存的访问（使用如`cl::CommandQueue::enqueueMapSVM()`和`cl::CommandQueue::enqueueUnmapSVM()`等函数），这使得 SVM 的使用不太方便。值得注意的是，这与 CUDA 提供的常规统一内存不同，CUDA 更接近于 OpenCL 中的细粒度系统 SVM 级别。

OpenCL 使用独立源内核模型，其中内核代码通常保存在单独的文件中，这些文件可以在运行时编译。该模型允许将内核源代码作为字符串传递给 OpenCL 驱动程序，之后程序对象可以在特定设备上执行。尽管被称为独立源内核模型，但内核也可以在主机程序编译单元中以字符串的形式定义，这在某些情况下可能更方便。

使用独立源内核模型的在线编译相较于需要离线编译内核为特定设备二进制的二进制模型具有多个优势。在线编译保留了 OpenCL 的可移植性和灵活性，因为相同的内核源代码可以在任何支持的设备上运行。此外，基于运行时信息（如输入大小、工作组大小或设备能力）的内核动态优化也是可能的。以下是一个 OpenCL 内核的示例，它由主机编译单元中的字符串定义，并将全局线程索引分配到全局设备内存中。

```
static  const  std::string  kernel_source  =  R"(
 __kernel void dot(__global int *a) {
 int i = get_global_id(0);
 a[i] = i;
 }
)"; 
```

上面的名为 `dot` 并存储在字符串 `kernel_source` 中的内核可以设置在主机代码中构建，如下所示：

```
cl::Program  program(context,  kernel_source);
program.build({device});
cl::Kernel  kernel_dot(program,  "dot"); 
```

```
cl_int  err;
cl_program  program  =  clCreateProgramWithSource(context,  1,  &kernel_source,  NULL,  &err);
err  =  clBuildProgram(program,  1,  &device,  NULL,  NULL,  NULL);
cl_kernel  kernel_dot  =  clCreateKernel(program,  "vector_add",  &err); 
```

## SYCL

[SYCL](https://www.khronos.org/sycl/) 是一个免版税的开放标准 C++ 编程模型，用于多设备编程。它为异构系统提供了一种高级、单源编程模型，包括 GPU。该标准有几个实现。对于 GPU 编程，[Intel oneAPI DPC++](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html) 和 [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp/)（也称为 hipSYCL）是桌面和 HPC GPU 中最受欢迎的；[ComputeCPP](https://developer.codeplay.com/products/computecpp/ce/home/) 是嵌入式设备的良好选择。相同的标准兼容 SYCL 代码应该可以与任何实现一起工作，但它们不是二进制兼容的。

SYCL 标准的最新版本是 SYCL 2020，这是我们将在本课程中使用的版本。

### SYCL 编译

#### Intel oneAPI DPC++

对于 Intel GPU，只需安装 [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) 即可。然后，编译就像 `icpx -fsycl file.cpp` 那么简单。

还可以使用 oneAPI 为 NVIDIA 和 AMD GPU。除了 oneAPI Base Toolkit 之外，还需要安装供应商提供的运行时（CUDA 或 HIP）以及相应的 [Codeplay oneAPI 插件](https://codeplay.com/solutions/oneapi/)。然后，可以使用包含在 oneAPI 中的 Intel LLVM 编译器编译代码：

+   `clang++ -fsycl -fsycl-targets=nvidia_gpu_sm_86 file.cpp` 用于针对 CUDA 8.6 NVIDIA GPU，

+   `clang++ -fsycl -fsycl-targets=amd_gpu_gfx90a` 用于针对 GFX90a AMD GPU。

#### AdaptiveCpp

使用 AdaptiveCpp 为 NVIDIA 或 AMD GPU 也需要首先安装 CUDA 或 HIP。然后可以使用 `acpp` 编译代码，指定目标设备。例如，以下是如何编译支持 AMD 和 NVIDIA 设备的程序：

+   `acpp --acpp-targets='hip:gfx90a;cuda:sm_70' file.cpp`

#### 在 LUMI 上使用 SYCL

LUMI 没有系统范围内安装任何 SYCL 框架，但 CSC 模块中有一个最新的 AdaptiveCpp 安装：

```
$ module  load  LUMI/24.03  partition/G
$ module  load  rocm/6.0.3
$ module  use  /appl/local/csc/modulefiles
$ module  load  acpp/24.06.0 
```

默认编译目标预设为 MI250 GPU，因此要编译单个 C++ 文件，只需调用 `acpp -O2 file.cpp` 即可。

当运行使用 AdaptiveCpp 构建的程序时，经常会看到警告“dag_direct_scheduler: Detected a requirement that is neither of discard access mode”，这反映了在使用缓冲区访问模型时缺少优化提示。警告是无害的，可以忽略。

### SYCL 编程

在许多方面，SYCL 与 OpenCL 类似，但像 Kokkos 一样使用具有内核 lambdas 的单源模型。

要将任务提交到设备，首先必须创建一个 sycl::queue，它用作管理任务调度和执行的方式。在最简单的情况下，这就是所需的全部初始化：

```
int  main()  {
  // Create an out-of-order queue on the default device:
  sycl::queue  q;
  // Now we can submit tasks to q!
} 
```

如果想要更多控制，可以显式指定设备，或者可以向队列传递额外的属性：

```
// Iterate over all available devices
for  (const  auto  &device  :  sycl::device::get_devices())  {
  // Print the device name
  std::cout  <<  "Creating a queue on "  <<  device.get_info<sycl::info::device::name>()  <<  "\n";
  // Create an in-order queue for the current device
  sycl::queue  q(device,  {sycl::property::queue::in_order()});
  // Now we can submit tasks to q!
} 
```

内存管理可以通过两种不同的方式完成：*缓冲访问器*模型和*统一共享内存*（USM）。内存管理模型的选择也会影响 GPU 任务的同步方式。

在*缓冲访问器*模型中，使用`sycl::buffer`对象来表示数据数组。缓冲区没有映射到任何单一内存空间，并且可以透明地在 GPU 和 CPU 内存之间迁移。`sycl::buffer`中的数据不能直接读取或写入，必须创建访问器。`sycl::accessor`对象指定数据访问的位置（主机或某个 GPU 内核）以及访问模式（只读、只写、读写）。这种方法允许通过构建数据依赖的有向无环图（DAG）来优化任务调度：如果内核*A*创建了一个对缓冲区的只写访问器，然后内核*B*提交了一个对同一缓冲区的只读访问器，然后请求主机端的只读访问器，那么可以推断出*A*必须在*B*启动之前完成，并且结果必须在主机任务可以继续之前复制到主机，但主机任务可以与内核*B*并行运行。由于任务之间的依赖关系可以自动构建，因此默认情况下 SYCL 使用*乱序队列*：当两个任务提交到同一个`sycl::queue`时，不能保证第二个任务只有在第一个任务完成后才会启动。在启动内核时，必须创建访问器：

```
// Create a buffer of n integers
auto  buf  =  sycl::buffer<int>(sycl::range<1>(n));
// Submit a kernel into a queue; cgh is a helper object
q.submit(&  {
  // Create write-only accessor for buf
  auto  acc  =  buf.get_access<sycl::access_mode::write>(cgh);
  // Define a kernel: n threads execute the following lambda
  cgh.parallel_for<class  KernelName>(sycl::range<1>{n},  =  {
  // The data is written to the buffer via acc
  acc[i]  =  /*...*/
  });
});
/* If we now submit another kernel with accessor to buf, it will not
 * start running until the kernel above is done */ 
```

缓冲访问器模型简化了异构编程的许多方面，并防止了许多与同步相关的错误，但它只允许对数据移动和内核执行进行非常粗略的控制。

*USM*模型类似于 NVIDIA CUDA 或 AMD HIP 管理内存的方式。程序员必须显式地在设备上分配内存（`sycl::malloc_device`）、在主机上（`sycl::malloc_host`）或在共享内存空间中（`sycl::malloc_shared`）。尽管名为统一共享内存，并且与 OpenCL 的 SVM 相似，但并非所有 USM 分配都是共享的：例如，由`sycl::malloc_device`分配的内存不能从主机访问。分配函数返回可以直接使用的内存指针，无需访问器。这意味着程序员必须确保主机和设备任务之间的正确同步，以避免数据竞争。使用 USM 时，通常更方便使用*顺序队列*而不是默认的*乱序队列*。有关 USM 的更多信息，请参阅[SYCL 2020 规范的第 4.8 节](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:usm)。

```
// Create a shared (migratable) allocation of n integers
// Unlike with buffers, we need to specify a queue (or, explicitly, a device and a context)
int*  v  =  sycl::malloc_shared<int>(n,  q);
// Submit a kernel into a queue; cgh is a helper object
q.submit(&  {
  // Define a kernel: n threads execute the following lambda
  cgh.parallel_for<class  KernelName>(sycl::range<1>{n},  =  {
  // The data is directly written to v
  v[i]  =  /*...*/
  });
});
// If we want to access v, we have to ensure that the kernel has finished
q.wait();
// After we're done, the memory must be deallocated
sycl::free(v,  q); 
```

### 练习

练习：在 SYCL 中实现 SAXPY

在这个练习中，我们想要编写（填空）一段简单的代码，用于执行 SAXPY（向量加法）。

要交互式地编译和运行代码，首先进行分配并加载 AdaptiveCpp 模块：

```
$ salloc  -A  project_465002387  -N  1  -t  1:00:00  -p  standard-g  --gpus-per-node=1
....
salloc: Granted job allocation 123456

$ module  load  LUMI/24.03  partition/G
$ module  use  /appl/local/csc/modulefiles
$ module  load  rocm/6.0.3  acpp/24.06.0 
```

现在，你可以运行一个简单的设备检测实用程序来检查 GPU 是否可用（注意 `srun`）：

> ```
> $ srun  acpp-info  -l
> =================Backend information===================
> Loaded backend 0: HIP
>  Found device: AMD Instinct MI250X
> Loaded backend 1: OpenMP
>  Found device: hipSYCL OpenMP host device 
> ```

如果你还没有做，请使用 `git clone https://github.com/ENCCS/gpu-programming.git` 或 **更新** 它使用 `git pull origin main` 来克隆仓库。

现在，让我们看看 `content/examples/portable-kernel-models/exercise-sycl-saxpy.cpp` 中的示例代码：

```
#include  <iostream>
#include  <sycl/sycl.hpp>
#include  <vector>

int  main()  {
  // Create an in-order queue
  sycl::queue  q{sycl::property::queue::in_order()};
  // Print the device name, just for fun
  std::cout  <<  "Running on "
  <<  q.get_device().get_info<sycl::info::device::name>()  <<  std::endl;
  const  int  n  =  1024;  // Vector size

  // Allocate device and host memory for the first input vector
  float  *d_x  =  sycl::malloc_device<float>(n,  q);
  float  *h_x  =  sycl::malloc_host<float>(n,  q);
 // Bonus question: Can we use `std::vector` here instead of `malloc_host`? // TODO: Allocate second input vector on device and host, d_y and h_y  // Allocate device and host memory for the output vector
  float  *d_z  =  sycl::malloc_device<float>(n,  q);
  float  *h_z  =  sycl::malloc_host<float>(n,  q);

  // Initialize values on host
  for  (int  i  =  0;  i  <  n;  i++)  {
  h_x[i]  =  i;
 // TODO: Initialize h_y somehow  }
  const  float  alpha  =  0.42f;

  q.copy<float>(h_x,  d_x,  n);
 // TODO: Copy h_y to d_y // Bonus question: Why don't we need to wait before using the data? 
  // Run the kernel
  q.parallel_for(sycl::range<1>{n},  =  {
 // TODO: Modify the code to compute z[i] = alpha * x[i] + y[i]  d_z[i]  =  alpha  *  d_x[i];
  });

 // TODO: Copy d_z to h_z // TODO: Wait for the copy to complete 
  // Check the results
  bool  ok  =  true;
  for  (int  i  =  0;  i  <  n;  i++)  {
  float  ref  =  alpha  *  h_x[i]  +  h_y[i];  // Reference value
  float  tol  =  1e-5;  // Relative tolerance
  if  (std::abs((h_z[i]  -  ref))  >  tol  *  std::abs(ref))  {
  std::cout  <<  i  <<  " "  <<  h_z[i]  <<  " "  <<  h_x[i]  <<  " "  <<  h_y[i]
  <<  std::endl;
  ok  =  false;
  break;
  }
  }
  if  (ok)
  std::cout  <<  "Results are correct!"  <<  std::endl;
  else
  std::cout  <<  "Results are NOT correct!"  <<  std::endl;

  // Free allocated memory
  sycl::free(d_x,  q);
  sycl::free(h_x,  q);
 // TODO: Free d_y, h_y.  sycl::free(d_y,  q);
  sycl::free(h_y,  q);

  return  0;
} 
```

要编译和运行代码，请使用以下命令：

```
$ acpp  -O3  exercise-sycl-saxpy.cpp  -o  exercise-sycl-saxpy
$ srun  ./exercise-sycl-saxpy
Running on AMD Instinct MI250X
Results are correct! 
```

代码将无法直接编译！你的任务是填写由 `TODO` 注释指示的缺失部分。你还可以通过代码中的“Bonus questions”来测试你的理解。

如果你感到卡壳，可以查看 `exercise-sycl-saxpy-solution.cpp` 文件。

## alpaka

[alpaka](https://github.com/alpaka-group/alpaka3) 库是一个开源的仅包含头文件的 C++20 抽象库，用于加速器开发。

其目的是通过抽象底层并行级别，在加速器之间提供性能可移植性。该项目提供了一个单一的 C++ API，使开发者能够编写一次并行代码，并在不同的硬件架构上运行而无需修改。名称“alpaka”来自 **A**bstractions for **L**evels of **P**arallelism, **A**lgorithms, and **K**ernels for **A**ccelerators。该库是平台无关的，并支持多个设备的并发和协作使用，包括主机 CPU（x86、ARM 和 RISC-V）以及来自不同厂商的 GPU（NVIDIA、AMD 和 Intel）。提供了各种加速器后端，如 CUDA、HIP、SYCL、OpenMP 和串行执行，可以根据目标设备进行选择。只需要一个用户内核的实现，以具有标准化接口的函数对象的形式表达。这消除了编写专门的 CUDA、HIP、SYCL、OpenMP、Intel TBB 或线程代码的需求。此外，可以将多个加速器后端组合起来，以针对单个系统甚至单个应用程序中的不同厂商硬件进行目标定位。

抽象基于一个虚拟索引域，该域被分解成称为帧的等大小块。**alpaka** 提供了一个统一的抽象来遍历这些帧，与底层硬件无关。要并行化的算法将分块索引域和本地工作线程映射到数据上，将计算表示为在并行线程（SIMT）中执行的内核，从而也利用了 SIMD 单元。与 CUDA、HIP 和 SYCL 等原生并行模型不同，**alpaka** 内核不受三维的限制。通过共享内存显式缓存帧内的数据允许开发者充分发挥计算设备性能。此外，**alpaka** 提供了诸如 iota、transform、transform-reduce、reduce 和 concurrent 等原始函数，简化了可移植高性能应用程序的开发。主机、设备、映射和管理多维视图提供了一种自然的方式来操作数据。

在这里，我们展示了 **alpaka3** 的使用方法，它是 [alpaka](https://github.com/alpaka-group/alpaka) 的完全重写。计划在 2026 年第二季度/第三季度的第一次发布之前，将这个独立的代码库合并回主线 alpaka 仓库。尽管如此，代码经过了良好的测试，并且可以用于今天的开发。

### 在您的系统上安装 alpaka

为了便于使用，我们建议按照以下说明使用 CMake 安装 alpaka。有关在项目中使用 alpaka 的其他方法，请参阅 [alpaka3 文档](https://alpaka3.readthedocs.io/en/latest/basic/install.html)。

1.  **克隆仓库**

    从 GitHub 克隆 alpaka 源代码到您选择的目录：

    ```
    git  clone  https://github.com/alpaka-group/alpaka3.git
    cd  alpaka 
    ```

1.  **设置安装目录**

    将 `ALPAKA_DIR` 环境变量设置为想要安装 alpaka 的目录。这可以是您选择的任何目录，只要您有写入权限。

    ```
    export  ALPAKA_DIR=/path/to/your/alpaka/install/dir 
    ```

1.  **构建和安装**

    创建一个构建目录，并使用 CMake 构建和安装 alpaka。我们使用 `CMAKE_INSTALL_PREFIX` 来告诉 CMake 将库安装在哪里。

    ```
    mkdir  build
    cmake  -B  build  -S  .  -DCMAKE_INSTALL_PREFIX=$ALPAKA_DIR
    cmake  --build  build  --parallel 
    ```

1.  **更新环境**

    为了确保其他项目可以找到您的 alpaka 安装，您应该将安装目录添加到您的 `CMAKE_PREFIX_PATH` 中。您可以通过将以下行添加到您的 shell 配置文件（例如 `~/.bashrc`）来实现：

    ```
    export  CMAKE_PREFIX_PATH=$ALPAKA_DIR:$CMAKE_PREFIX_PATH 
    ```

    您需要源您的 shell 配置文件或打开一个新的终端，以便更改生效。

### alpaka 编译

我们建议使用 CMake 构建使用 alpaka 的项目。可以采用各种策略来处理为特定设备或设备集构建应用程序。在这里，我们展示了入门的最小方法，但这绝不是设置项目的唯一方法。请参阅 [alpaka3 文档](https://alpaka3.readthedocs.io/en/latest/basic/install.html)，了解在项目中使用 alpaka 的其他方法，包括在 CMake 中定义设备规范以使源代码与目标加速器无关的方法。

以下示例演示了一个使用 alpaka3 的单文件项目的`CMakeLists.txt`（以下章节中展示的`main.cpp`）：

> ```
> cmake_minimum_required(VERSION  3.25)
> project(myAlpakaApp  VERSION  1.0)
> 
> # Find installed alpaka
> find_package(alpaka  REQUIRED)
> 
> # Build the executable
> add_executable(myAlpakaApp  main.cpp)
> target_link_libraries(myAlpakaApp  PRIVATE  alpaka::alpaka)
> alpaka_finalize(myAlpakaApp) 
> ```

#### 在 LUMI 上使用 alpaka

要加载在 LUMI 上使用 AMD GPU 的 HIP 环境，可以使用以下模块 -

```
$ module  load  LUMI/24.03  partition/G
$ module  load  rocm/6.0.3
$ module  load  buildtools/24.03
$ module  load  PrgEnv-amd
$ module  load  craype-accel-amd-gfx90a
$ export  CXX=hipcc 
```

### alpaka 编程

在开始使用 alpaka3 时，第一步是理解**设备选择模型**。与需要显式初始化调用的框架不同，alpaka3 使用设备规范来确定使用哪个后端和硬件。设备规范由两个组件组成：

+   **API**：并行编程接口（host、cuda、hip、oneApi）

+   **设备类型**：硬件类型（cpu、nvidiaGpu、amdGpu、intelGpu）

在这里，我们指定并使用这些规范在运行时选择和初始化设备。设备选择过程在 alpaka3 文档中有详细描述。

alpaka3 使用**执行空间模型**来抽象并行硬件细节。使用`alpaka::onHost::makeDeviceSelector(devSpec)`创建设备选择器，它返回一个可以查询可用设备并为所选后端创建设备实例的对象。

以下示例演示了一个基本的 alpaka 程序，该程序初始化一个设备并打印有关它的信息：

> ```
> #include  <alpaka/alpaka.hpp>
> #include  <cstdlib>
> #include  <iostream>
> 
> namespace  ap  =  alpaka;
> 
> auto  getDeviceSpec()
> {
>   /* Select a device, possible combinations of api+deviceKind:
>  * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
>  * oneApi+amdGpu, oneApi+nvidiaGpu
>  */
>   return  ap::onHost::DeviceSpec{ap::api::hip,  ap::deviceKind::amdGpu};
> }
> 
> int  main(int  argc,  char**  argv)
> {
>   // Initialize device specification and selector
>   ap::onHost::DeviceSpec  devSpec  =  getDeviceSpec();
>   auto  deviceSelector  =  ap::onHost::makeDeviceSelector(devSpec);
> 
>   // Query available devices
>   auto  num_devices  =  deviceSelector.getDeviceCount();
>   std::cout  <<  "Number of available devices: "  <<  num_devices  <<  "\n";
> 
>   if  (num_devices  ==  0)  {
>   std::cerr  <<  "No devices found for the selected backend\n";
>   return  EXIT_FAILURE;
>   }
> 
>   // Select and initialize the first device
>   auto  device  =  deviceSelector.makeDevice(0);
>   std::cout  <<  "Using device: "  <<  device.getName()  <<  "\n";
> 
>   return  EXIT_SUCCESS;
> } 
> ```

alpaka3 通过缓冲区和视图提供内存管理抽象。可以使用`alpaka::allocBuf<T, Idx>(device, extent)`在主机或设备上分配内存。主机和设备之间的数据传输通过`alpaka::memcpy(queue, dst, src)`处理。该库自动管理不同架构上的内存布局以实现最佳性能。

对于并行执行，alpaka3 提供内核抽象。内核定义为函数式对象或 lambda 函数，并使用定义并行化策略的工作划分规范执行。该框架支持各种并行模式，包括逐元素操作、归约和扫描。

#### **alpaka**功能巡礼

现在我们将快速探索 alpaka 最常用的功能，并简要介绍一些基本用法。常用的 alpaka 功能快速参考可在[这里](https://alpaka3.readthedocs.io/en/latest/basic/cheatsheet.html)找到。

**通用设置**：只需包含一次综合头文件，你就可以开始使用 alpaka 了。

```
#include  <alpaka/alpaka.hpp>

namespace  myProject
{
  namespace  ap  =  alpaka;
  // Your code here
} 
```

**加速器、平台和设备管理**：通过设备选择器结合所需的 API 和适当的硬件类型来选择设备。

```
auto  devSelector  =  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
if  (devSelector.getDeviceCount()  ==  0)
{
  throw  std::runtime_error("No device found!");
}
auto  device  =  devSelector.makeDevice(0); 
```

**队列和事件**：为每个设备创建阻塞或非阻塞队列，记录事件，并根据需要同步工作。

```
auto  queue  =  device.makeQueue();
auto  nonBlockingQueue  =  device.makeQueue(ap::queueKind::nonBlocking);
auto  blockingQueue  =  device.makeQueue(ap::queueKind::blocking);

auto  event  =  device.makeEvent();
queue.enqueue(event);
ap::onHost::wait(event);
ap::onHost::wait(queue); 
```

**内存管理**：分配主机、设备、映射、统一或延迟缓冲区，创建非拥有视图，并使用 memcpy、memset 和 fill 移动数据。

```
auto  hostBuffer  =  ap::onHost::allocHost<DataType>(extent3D);
auto  devBuffer  =  ap::onHost::alloc<DataType>(device,  extentMd);
auto  devMappedBuffer  =  ap::onHost::allocMapped<DataType>(device,  extentMd);

auto  hostView  =  ap::makeView(api::host,  externPtr,  ap::Vec{numElements});
auto  devNonOwningView  =  devBuffer.getView();

ap::onHost::memset(queue,  devBuffer,  uint8_t{0});
ap::onHost::memcpy(queue,  devBuffer,  hostBuffer);
ap::onHost::fill(queue,  devBuffer,  DataType{42}); 
```

**内核执行**：手动构建 FrameSpec 或请求针对您数据类型调优的一个，然后使用自动或显式执行器排队内核。

```
constexpr  uint32_t  dim  =  2u;
using  IdxType  =  size_t;
using  DataType  =  int;

IdxType  valueX,  valueY;
auto  extentMD  =  ap::Vec{valueY,  valueX};

auto  frameSpec  =  ap::onHost::FrameSpec{numFramesMd,  frameExtentMd};
auto  tunedSpec  =  ap::onHost::getFrameSpec<DataType>(device,  extentMd);

queue.enqueue(tunedSpec,  ap::KernelBundle{kernel,  kernelArgs...});

auto  executor  =  ap::exec::cpuSerial;
queue.enqueue(executor,  tunedSpec,  ap::KernelBundle{kernel,  kernelArgs...}); 
```

**内核实现**：将内核作为带有 ALPAKA_FN_ACC 注解的函数编写，使用共享内存、同步、原子操作和数学助手直接在内核体内部。

```
struct  MyKernel
{
  ALPAKA_FN_ACC  void  operator()(ap::onAcc::concepts::Acc  auto  const&  acc,  auto...  args)  const
  {
  auto  idxMd  =  acc.getIdxWithin(ap::onAcc::origin::grid,  ap::onAcc::unit::blocks);

  auto  sharedMdArray  =
  ap::onAcc::declareSharedMdArray<float,  ap::uniqueId()>(acc,  ap::CVec<uint32_t,  3,  4>{});

  ap::onAcc::syncBlockThreads(acc);
  auto  old  =  onAcc::atomicAdd(acc,  args...);
  ap::onAcc::memFence(acc,  ap::onAcc::scope::block);
  auto  sinValue  =  ap::math::sin(args[0]);
  }
}; 
```

#### 简单步骤运行 alpaka3 示例

以下示例适用于 CMake 3.25+ 和适当的 C++ 编译器的系统。对于 GPU 执行，请确保已安装相应的运行时（CUDA、ROCm 或 oneAPI）。

1.  为你的项目创建一个目录：

    ```
    mkdir  my_alpaka_project  &&  cd  my_alpaka_project 
    ```

1.  将上面的 CMakeLists.txt 复制到当前文件夹

1.  将 main.cpp 文件复制到当前文件夹

1.  配置和构建：

    ```
    cmake  -B  build  -S  .  -Dalpaka_DEP_HIP=ON
    cmake  --build  build  --parallel 
    ```

1.  运行可执行文件：

    ```
    ./build/myAlpakaApp 
    ```

注意

设备规范系统允许你在 CMake 配置时间选择目标设备。格式为 `"api:deviceKind"`，其中：

+   **api**：并行编程接口（`host`、`cuda`、`hip`、`oneApi`）

+   **deviceKind**：设备的类型（`cpu`、`nvidiaGpu`、`amdGpu`、`intelGpu`）

可用的组合有：`host:cpu`、`cuda:nvidiaGpu`、`hip:amdGpu`、`oneApi:cpu`、`oneApi:intelGpu`、`oneApi:nvidiaGpu`、`oneApi:amdGpu`

警告

只有当 CUDA SDK、HIP SDK 或 OneAPI SDK 可用时，CUDA、HIP 或 Intel 后端才有效

#### 预期输出

```
Number of available devices: 1
Using device: [Device Name] 
```

设备名称将根据你的硬件而变化（例如，“NVIDIA A100”、“AMD MI250X”或你的 CPU 型号）。

### 编译和执行示例

你可以从示例部分测试 alpaka 提供的示例。这些示例在 LUMI 上需要硬编码 AMD ROCm 平台的使用。要仅使用 CPU，你可以简单地替换 `ap::onHost::makeDeviceSelector(ap::api::hip, ap::deviceKind::amdGpu);` 为 `ap::onHost::makeDeviceSelector(ap::api::host, ap::deviceKind::cpu);`

以下步骤假设你已经下载了 alpaka，并且 alpaka 源代码的路径存储在环境变量 `ALPAKA_DIR` 中。要测试示例，请将代码复制到文件 `main.cpp`

或者，[点击此处](https://godbolt.org/z/69exnG4xb) 在 godbolt 编译器探索器中尝试第一个示例。

```
# use the following in C++ code
# auto devSelector = ap::onHost::makeDeviceSelector(ap::api::hip, ap::deviceKind::amdGpu);
# We use CC to refer to the compiler to work smoothly with the LUMI environment
CC  -I  $ALPAKA_DIR/include/  -std=c++20  -x  hip  --offload-arch=gfx90a  main.cpp
./a.out 
```

```
# use the following in C++ code
# auto devSelector = ap::onHost::makeDeviceSelector(ap::api::host, ap::deviceKind::cpu);
# We use CC to refer to the compiler to work smoothly with the LUMI environment
CC  -I  $ALPAKA_DIR/include/  -std=c++20  main.cpp
./a.out 
```

```
# use the following in C++ code
# auto devSelector = ap::onHost::makeDeviceSelector(ap::api::cuda, ap::deviceKind::nvidiaGpu);
nvcc  -I  $ALPAKA_DIR/include/  -std=c++20  --expt-relaxed-constexpr  -x  cuda  main.cpp
./a.out 
```

```
# use the following in C++ code
# auto devSelector = ap::onHost::makeDeviceSelector(ap::api::oneApi, ap::deviceKind::cpu);
icpx  -I  $ALPAKA_DIR/include/  -std=c++20  -fsycl  -fsycl-targets=spir64_x86_64  main.cpp
./a.out 
```

```
# use the following in C++ code
# auto devSelector = ap::onHost::makeDeviceSelector(ap::api::oneApi, ap::deviceKind::intelGpu);
icpx  -I  $ALPAKA_DIR/include/  -std=c++20  -fsycl  -fsycl-targets=spir64  main.cpp
./a.out 
```

注意

要使用 AMD 或 NVIDIA Gpus 的 oneAPI Sycl，你必须安装相应的 Codeplay oneAPI 插件，如[此处](https://codeplay.com/solutions/oneapi/plugins/)所述。

```
# use the following in C++ code
# auto devSelector = ap::onHost::makeDeviceSelector(ap::api::oneApi, ap::deviceKind::amdGpu);
icpx  -I  $ALPAKA_DIR/include/  -std=c++20  -fsycl  -fsycl-targets=amd_gpu_gfx90a  main.cpp
./a.out 
```

注意

要使用 AMD 或 NVIDIA Gpus 的 oneAPI Sycl，你必须安装相应的 Codeplay oneAPI 插件，如[此处](https://codeplay.com/solutions/oneapi/plugins/)所述。

```
# use the following in C++ code
# auto devSelector = ap::onHost::makeDeviceSelector(ap::api::oneApi, ap::deviceKind::nvidiaGpu);
icpx  -I  $ALPAKA_DIR/include/  -std=c++20  -fsycl  -fsycl-targets=nvptx64-nvidia-cuda  -Xsycl-target-backend=nvptx64-nvidia-cuda  --offload-arch=sm_80  main.cpp
./a.out 
```

### 练习

练习：在 alpaka 中编写向量加法内核

在这个练习中，我们希望编写（填空）一个简单的内核来添加两个向量。

要交互式地编译和运行代码，我们首先需要在一个 GPU 节点上获取一个分配并加载 alpaka 的模块：

```
$ srun  -p  dev-g  --gpus  1  -N  1  -n  1  --time=00:20:00  --account=project_465002387  --pty  bash
....
srun: job 1234 queued and waiting for resources
srun: job 1234 has been allocated resources

$ module  load  LUMI/24.03  partition/G
$ module  load  rocm/6.0.3
$ module  load  buildtools/24.03
$ module  load  PrgEnv-amd
$ module  load  craype-accel-amd-gfx90a
$ export  CXX=hipcc 
```

现在你可以运行一个简单的设备检测工具来检查是否有 GPU 可用（注意 `srun`）：

```
$ rocm-smi

======================================= ROCm System Management Interface =======================================
================================================= Concise Info =================================================
Device  [Model : Revision]    Temp    Power  Partitions      SCLK    MCLK     Fan  Perf    PwrCap  VRAM%  GPU%
 Name (20 chars)       (Edge)  (Avg)  (Mem, Compute)
================================================================================================================
0       [0x0b0c : 0x00]       45.0°C  N/A    N/A, N/A        800Mhz  1600Mhz  0%   manual  0.0W      0%   0%
 AMD INSTINCT MI200 (
================================================================================================================
============================================= End of ROCm SMI Log ============================================== 
```

现在，让我们看看设置练习的代码：

下面我们使用 CMake 的 fetch content 快速开始使用 alpaka。

CMakeLists.txt

```
cmake_minimum_required(VERSION  3.25)
project(vectorAdd  LANGUAGES  CXX  VERSION  1.0)
#Use CMake's FetchContent to download and integrate alpaka3 directly from GitHub
include(FetchContent)
#Declare where to fetch alpaka3 from
#This will download the library at configure time
FetchContent_Declare(alpaka3  GIT_REPOSITORY  https://github.com/alpaka-group/alpaka3.git  GIT_TAG  dev)
#Make alpaka3 available for use in this project
#This downloads, configures, and makes the library targets available
FetchContent_MakeAvailable(alpaka3)
#Finalize the alpaka FetchContent setup
alpaka_FetchContent_Finalize() #Create the executable target from the source file
add_executable(vectorAdd  main.cpp)
#Link the alpaka library to the executable
target_link_libraries(vectorAdd  PRIVATE  alpaka::alpaka)
#Finalize the alpaka configuration for this target
#This sets up backend - specific compiler flags and dependencies
alpaka_finalize(vectorAdd) 
```

下面我们有主要的 alpaka 代码，使用高级转换函数在设备上进行向量加法

main.cpp

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

auto  main()  ->  int  {
  unsigned  n  =  5;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */

  // auto devSelector = ap::onHost::makeDeviceSelector(ap::api::host,
  // ap::deviceKind::cpu);
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue  queue  =  devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate unified memory that is accessible on host and device
  auto  a  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  b  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  c  =  ap::onHost::allocUnified<int>(devAcc,  n);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

  // Run element-wise vector addition on device
 ap::onHost::transform(queue,  c,  std::plus{},  a,  b); 
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  c[i]);
  }

  return  0;
} 
```

为了设置我们的项目，我们创建一个文件夹，并将我们的 CMakeLists.txt 和 main.cpp 放在那里。

```
$ mkdir  alpakaExercise  &&  cd  alpakaExercise
$ vim  CMakeLists.txt
and now paste the CMakeLsits here (Press i, followed by Ctrl+Shift+V)
Press esc and then :wq to exit vim
$ vim  main.cpp
Similarly, paste the C++ code here 
```

要编译和运行代码，请使用以下命令：

```
configure step, we additionaly specify that HIP is available
$ cmake  -B  build  -S  .  -Dalpaka_DEP_HIP=ON
build
$ cmake  --build  build  --parallel
run
$ ./build/vectorAdd
Using alpaka device: AMD Instinct MI250X id=0
c[0] = 1
c[1] = 2
c[2] = 3
c[3] = 4
c[4] = 5 
```

现在的任务将是编写和启动你的第一个 alpaka 内核。这个内核将执行向量加法，我们将使用这个代替 transform 辅助器。

编写向量加法内核

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

struct  AddKernel  {
 constexpr  void  operator()(ap::onAcc::concepts::Acc  auto  const  &acc, ap::concepts::IMdSpan  auto  c, ap::concepts::IMdSpan  auto  const  a, ap::concepts::IMdSpan  auto  const  b)  const  { for  (auto  idx  :  ap::onAcc::makeIdxMap(acc,  ap::onAcc::worker::threadsInGrid, ap::IdxRange{c.getExtents()}))  { c[idx]  =  a[idx]  +  b[idx]; } } }; 
auto  main()  ->  int  {
  unsigned  n  =  5;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */

  // auto devSelector = ap::onHost::makeDeviceSelector(ap::api::host,
  // ap::deviceKind::cpu);
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue  queue  =  devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate unified memory that is accessible on host and device
  auto  a  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  b  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  c  =  ap::onHost::allocUnified<int>(devAcc,  n);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

 auto  frameSpec  =  ap::onHost::getFrameSpec<int>(devAcc,  c.getExtents()); 
 // Call the element-wise addition kernel on device queue.enqueue(frameSpec,  ap::KernelBundle{AddKernel{},  c,  a,  b}); 
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  c[i]);
  }

  return  0;
} 
```

## 示例

### 带统一内存的并行 for 循环

```
#include  <algorithm>
#include  <cstdio>
#include  <execution>
#include  <vector>

int  main()  {
  unsigned  n  =  5;

  // Allocate arrays
  std::vector<int>  a(n),  b(n),  c(n);

  // Initialize values
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

  // Run element-wise multiplication on device
  std::transform(std::execution::par_unseq,  a.begin(),  a.end(),  b.begin(),
  c.begin(),  [](int  i,  int  j)  {  return  i  *  j;  });

  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  c[i]);
  }

  return  0;
} 
```

```
#include  <Kokkos_Core.hpp>

int  main(int  argc,  char  *argv[])  {

  // Initialize Kokkos
  Kokkos::initialize(argc,  argv);

  {
  unsigned  n  =  5;

  // Allocate on Kokkos default memory space (Unified Memory)
  int  *a  =  (int  *)Kokkos::kokkos_malloc(n  *  sizeof(int));
  int  *b  =  (int  *)Kokkos::kokkos_malloc(n  *  sizeof(int));
  int  *c  =  (int  *)Kokkos::kokkos_malloc(n  *  sizeof(int));

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

  // Run element-wise multiplication on device
  Kokkos::parallel_for(n,  KOKKOS_LAMBDA(const  int  i)  {  c[i]  =  a[i]  *  b[i];  });

  // Kokkos synchronization
  Kokkos::fence();

  // Print results
  for  (unsigned  i  =  0;  i  <  n;  i++)
  printf("c[%d] = %d\n",  i,  c[i]);

  // Free Kokkos allocation (Unified Memory)
  Kokkos::kokkos_free(a);
  Kokkos::kokkos_free(b);
  Kokkos::kokkos_free(c);
  }

  // Finalize Kokkos
  Kokkos::finalize();
  return  0;
} 
```

```
// We're using OpenCL C API here, since SVM support in C++ API is unstable on
// ROCm
#define CL_TARGET_OPENCL_VERSION 220
#include  <CL/cl.h>
#include  <stdio.h>

// For larger kernels, we can store source in a separate file
static  const  char  *kernel_source  =
  "                                                 \
 __kernel void dot(__global const int *a, __global const int *b, __global int *c) { \
 int i = get_global_id(0);                                                        \
 c[i] = a[i] * b[i];                                                              \
 }                                                                                  \
 ";

int  main(int  argc,  char  *argv[])  {

  // Initialize OpenCL
  cl_platform_id  platform;
  clGetPlatformIDs(1,  &platform,  NULL);
  cl_device_id  device;
  clGetDeviceIDs(platform,  CL_DEVICE_TYPE_GPU,  1,  &device,  NULL);
  cl_context  context  =  clCreateContext(NULL,  1,  &device,  NULL,  NULL,  NULL);
  cl_command_queue  queue  =  clCreateCommandQueue(context,  device,  0,  NULL);

  // Compile OpenCL program for found device.
  cl_program  program  =
  clCreateProgramWithSource(context,  1,  &kernel_source,  NULL,  NULL);
  clBuildProgram(program,  1,  &device,  NULL,  NULL,  NULL);
  cl_kernel  kernel  =  clCreateKernel(program,  "dot",  NULL);

  // Set problem dimensions
  unsigned  n  =  5;

  // Create SVM buffer objects on host side
  int  *a  =  clSVMAlloc(context,  CL_MEM_READ_ONLY,  n  *  sizeof(int),  0);
  int  *b  =  clSVMAlloc(context,  CL_MEM_READ_ONLY,  n  *  sizeof(int),  0);
  int  *c  =  clSVMAlloc(context,  CL_MEM_WRITE_ONLY,  n  *  sizeof(int),  0);

  // Pass arguments to device kernel
  clSetKernelArgSVMPointer(kernel,  0,  a);
  clSetKernelArgSVMPointer(kernel,  1,  b);
  clSetKernelArgSVMPointer(kernel,  2,  c);

  // Create mappings for host and initialize values
  clEnqueueSVMMap(queue,  CL_TRUE,  CL_MAP_WRITE,  a,  n  *  sizeof(int),  0,  NULL,
  NULL);
  clEnqueueSVMMap(queue,  CL_TRUE,  CL_MAP_WRITE,  b,  n  *  sizeof(int),  0,  NULL,
  NULL);
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }
  clEnqueueSVMUnmap(queue,  a,  0,  NULL,  NULL);
  clEnqueueSVMUnmap(queue,  b,  0,  NULL,  NULL);

  size_t  globalSize  =  n;
  clEnqueueNDRangeKernel(queue,  kernel,  1,  NULL,  &globalSize,  NULL,  0,  NULL,
  NULL);

  // Create mapping for host and print results
  clEnqueueSVMMap(queue,  CL_TRUE,  CL_MAP_READ,  c,  n  *  sizeof(int),  0,  NULL,
  NULL);
  for  (unsigned  i  =  0;  i  <  n;  i++)
  printf("c[%d] = %d\n",  i,  c[i]);
  clEnqueueSVMUnmap(queue,  c,  0,  NULL,  NULL);

  // Free SVM buffers
  clSVMFree(context,  a);
  clSVMFree(context,  b);
  clSVMFree(context,  c);

  return  0;
} 
```

```
#include  <sycl/sycl.hpp>

int  main()  {

  sycl::queue  q;
  unsigned  n  =  5;

  // Allocate shared memory (Unified Shared Memory)
  int  *a  =  sycl::malloc_shared<int>(n,  q);
  int  *b  =  sycl::malloc_shared<int>(n,  q);
  int  *c  =  sycl::malloc_shared<int>(n,  q);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

  // Run element-wise multiplication on device
  q.parallel_for(sycl::range<1>{n},  =  {
  c[i]  =  a[i]  *  b[i];
  }).wait();

  // Print results
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  c[i]);
  }

  // Free shared memory allocation (Unified Memory)
  sycl::free(a,  q);
  sycl::free(b,  q);
  sycl::free(c,  q);

  return  0;
} 
```

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

auto  main()  ->  int  {
  unsigned  n  =  5;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue  queue  =  devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate unified memory that is accessible on host and device
  auto  a  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  b  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  c  =  ap::onHost::allocUnified<int>(devAcc,  n);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

  // Run element-wise multiplication on device
  ap::onHost::transform(queue,  c,  std::multiplies{},  a,  b);

  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  c[i]);
  }

  return  0;
} 
```

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

struct  MulKernel  {
  constexpr  void  operator()(ap::onAcc::concepts::Acc  auto  const&  acc,
  ap::concepts::IMdSpan  auto  c,
  ap::concepts::IMdSpan  auto  const  a,
  ap::concepts::IMdSpan  auto  const  b)  const  {
  for  (auto  idx  :  ap::onAcc::makeIdxMap(acc,  ap::onAcc::worker::threadsInGrid,
  ap::IdxRange{c.getExtents()}))  {
  c[idx]  =  a[idx]  *  b[idx];
  }
  }
};

auto  main()  ->  int  {
  unsigned  n  =  5;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue  queue  =  devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate unified memory that is accessible on host and device
  auto  a  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  b  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  c  =  ap::onHost::allocUnified<int>(devAcc,  n);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

  unsigned  frameExtent  =  32u;
  auto  frameSpec  =
  ap::onHost::FrameSpec{ap::divExZero(n,  frameExtent),  frameExtent};

  // Run element-wise multiplication on device
  queue.enqueue(frameSpec,  ap::KernelBundle{MulKernel{},  c,  a,  b});

  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  c[i]);
  }

  return  0;
} 
```

### 带 GPU 缓冲区的并行 for 循环

```
#include  <Kokkos_Core.hpp>

int  main(int  argc,  char  *argv[])  {

  // Initialize Kokkos
  Kokkos::initialize(argc,  argv);

  {
  unsigned  n  =  5;

  // Allocate space for 5 ints on Kokkos host memory space
  Kokkos::View<int  *,  Kokkos::HostSpace>  h_a("h_a",  n);
  Kokkos::View<int  *,  Kokkos::HostSpace>  h_b("h_b",  n);
  Kokkos::View<int  *,  Kokkos::HostSpace>  h_c("h_c",  n);

  // Allocate space for 5 ints on Kokkos default memory space (eg, GPU memory)
  Kokkos::View<int  *>  a("a",  n);
  Kokkos::View<int  *>  b("b",  n);
  Kokkos::View<int  *>  c("c",  n);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  h_a[i]  =  i;
  h_b[i]  =  1;
  }

  // Copy from host to device
  Kokkos::deep_copy(a,  h_a);
  Kokkos::deep_copy(b,  h_b);

  // Run element-wise multiplication on device
  Kokkos::parallel_for(n,  KOKKOS_LAMBDA(const  int  i)  {  c[i]  =  a[i]  *  b[i];  });

  // Copy from device to host
  Kokkos::deep_copy(h_c,  c);

  // Print results
  for  (unsigned  i  =  0;  i  <  n;  i++)
  printf("c[%d] = %d\n",  i,  h_c[i]);
  }

  // Finalize Kokkos
  Kokkos::finalize();
  return  0;
} 
```

```
// We're using OpenCL C++ API here; there is also C API in <CL/cl.h>
#define CL_TARGET_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110
#include  <CL/cl.hpp>

// For larger kernels, we can store source in a separate file
static  const  std::string  kernel_source  =  R"(
 __kernel void dot(__global const int *a, __global const int *b, __global int *c) {
 int i = get_global_id(0);
 c[i] = a[i] * b[i];
 }
  )";

int  main(int  argc,  char  *argv[])  {

  // Initialize OpenCL
  cl::Device  device  =  cl::Device::getDefault();
  cl::Context  context(device);
  cl::CommandQueue  queue(context,  device);

  // Compile OpenCL program for found device.
  cl::Program  program(context,  kernel_source);
  program.build({device});
  cl::Kernel  kernel_dot(program,  "dot");

  {
  // Set problem dimensions
  unsigned  n  =  5;

  std::vector<int>  a(n),  b(n),  c(n);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

  // Create buffers and copy input data to device.
  cl::Buffer  dev_a(context,  CL_MEM_READ_ONLY  |  CL_MEM_COPY_HOST_PTR,
  n  *  sizeof(int),  a.data());
  cl::Buffer  dev_b(context,  CL_MEM_READ_ONLY  |  CL_MEM_COPY_HOST_PTR,
  n  *  sizeof(int),  b.data());
  cl::Buffer  dev_c(context,  CL_MEM_WRITE_ONLY,  n  *  sizeof(int));

  // Pass arguments to device kernel
  kernel_dot.setArg(0,  dev_a);
  kernel_dot.setArg(1,  dev_b);
  kernel_dot.setArg(2,  dev_c);

  // We don't need to apply any offset to thread IDs
  queue.enqueueNDRangeKernel(kernel_dot,  cl::NullRange,  cl::NDRange(n),
  cl::NullRange);

  // Read result
  queue.enqueueReadBuffer(dev_c,  CL_TRUE,  0,  n  *  sizeof(int),  c.data());

  // Print results
  for  (unsigned  i  =  0;  i  <  n;  i++)
  printf("c[%d] = %d\n",  i,  c[i]);
  }

  return  0;
} 
```

```
#include  <sycl/sycl.hpp>

int  main()  {

  sycl::queue  q;
  unsigned  n  =  5;

  // Allocate space for 5 ints
  auto  a_buf  =  sycl::buffer<int>(sycl::range<1>(n));
  auto  b_buf  =  sycl::buffer<int>(sycl::range<1>(n));
  auto  c_buf  =  sycl::buffer<int>(sycl::range<1>(n));

  // Initialize values
  // We should use curly braces to limit host accessors' lifetime
  //    and indicate when we're done working with them:
  {
  auto  a_host_acc  =  a_buf.get_host_access();
  auto  b_host_acc  =  b_buf.get_host_access();
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a_host_acc[i]  =  i;
  b_host_acc[i]  =  1;
  }
  }

  // Submit a SYCL kernel into a queue
  q.submit(&  {
  // Create read accessors over a_buf and b_buf
  auto  a_acc  =  a_buf.get_access<sycl::access_mode::read>(cgh);
  auto  b_acc  =  b_buf.get_access<sycl::access_mode::read>(cgh);
  // Create write accesor over c_buf
  auto  c_acc  =  c_buf.get_access<sycl::access_mode::write>(cgh);
  // Run element-wise multiplication on device
  cgh.parallel_for<class  vec_add>(sycl::range<1>{n},  =  {
  c_acc[i]  =  a_acc[i]  *  b_acc[i];
  });
  });

  // No need to synchronize, creating the accessor for c_buf will do it
  // automatically
  {
  const  auto  c_host_acc  =  c_buf.get_host_access();
  // Print results
  for  (unsigned  i  =  0;  i  <  n;  i++)
  printf("c[%d] = %d\n",  i,  c_host_acc[i]);
  }

  return  0;
} 
```

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

auto  main()  ->  int  {
  unsigned  n  =  5;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue  queue  =  devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate memory that is accessible on host
  auto  h_a  =  ap::onHost::allocHost<int>(n);
  auto  h_b  =  ap::onHost::allocHostLike(h_a);
  auto  h_c  =  ap::onHost::allocHostLike(h_a);

  // Allocate memory on the device and inherit the extents from h_a
  auto  a  =  ap::onHost::allocLike(devAcc,  h_a);
  auto  b  =  ap::onHost::allocLike(devAcc,  h_a);
  auto  c  =  ap::onHost::allocLike(devAcc,  h_a);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  h_a[i]  =  i;
  h_b[i]  =  1;
  }

  // Copy host memory element wise to the device memory
  ap::onHost::memcpy(queue,  a,  h_a);
  ap::onHost::memcpy(queue,  b,  h_b);

  // Run element-wise multiplication on device
  ap::onHost::transform(queue,  c,  std::multiplies{},  a,  b);

  // Copy the device result back to host memory
  ap::onHost::memcpy(queue,  h_c,  c);

  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  h_c[i]);
  }

  return  0;
} 
```

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

struct  MulKernel  {
  constexpr  void  operator()(ap::onAcc::concepts::Acc  auto  const&  acc,
  ap::concepts::IMdSpan  auto  c,
  ap::concepts::IMdSpan  auto  const  a,
  ap::concepts::IMdSpan  auto  const  b)  const  {
  for  (auto  idx  :  ap::onAcc::makeIdxMap(acc,  ap::onAcc::worker::threadsInGrid,
  ap::IdxRange{c.getExtents()}))  {
  c[idx]  =  a[idx]  *  b[idx];
  }
  }
};

auto  main()  ->  int  {
  unsigned  n  =  5;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue  queue  =  devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate memory that is accessible on host
  auto  h_a  =  ap::onHost::allocHost<int>(n);
  auto  h_b  =  ap::onHost::allocHostLike(h_a);
  auto  h_c  =  ap::onHost::allocHostLike(h_a);

  // allocate memory on the device and inherit the extents from a
  auto  a  =  ap::onHost::allocLike(devAcc,  h_a);
  auto  b  =  ap::onHost::allocLike(devAcc,  h_a);
  auto  c  =  ap::onHost::allocLike(devAcc,  h_a);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  h_a[i]  =  i;
  h_b[i]  =  1;
  }

  // Copy host memory element wise to the device memory
  ap::onHost::memcpy(queue,  a,  h_a);
  ap::onHost::memcpy(queue,  b,  h_b);

  unsigned  frameExtent  =  32u;
  auto  frameSpec  =
  ap::onHost::FrameSpec{ap::divExZero(n,  frameExtent),  frameExtent};

  // Run element-wise multiplication on device
  queue.enqueue(frameSpec,  ap::KernelBundle{MulKernel{},  c,  a,  b});

  // Copy the device result back to host memory
  ap::onHost::memcpy(queue,  h_c,  c);

  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  h_c[i]);
  }

  return  0;
} 
```

### 异步并行 for 内核

```
#include  <Kokkos_Core.hpp>

int  main(int  argc,  char  *argv[])  {

  // Initialize Kokkos
  Kokkos::initialize(argc,  argv);

  {
  unsigned  n  =  5;
  unsigned  nx  =  20;

  // Allocate on Kokkos default memory space (Unified Memory)
  int  *a  =  (int  *)Kokkos::kokkos_malloc(nx  *  sizeof(int));

  // Create 'n' execution space instances (maps to streams in CUDA/HIP)
  auto  ex  =  Kokkos::Experimental::partition_space(
  Kokkos::DefaultExecutionSpace(),  1,  1,  1,  1,  1);

  // Launch 'n' potentially asynchronous kernels
  // Each kernel has their own execution space instances
  for  (unsigned  region  =  0;  region  <  n;  region++)  {
  Kokkos::parallel_for(
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
  ex[region],  nx  /  n  *  region,  nx  /  n  *  (region  +  1)),
  KOKKOS_LAMBDA(const  int  i)  {  a[i]  =  region  +  i;  });
  }

  // Sync execution space instances (maps to streams in CUDA/HIP)
  for  (unsigned  region  =  0;  region  <  n;  region++)
  ex[region].fence();

  // Print results
  for  (unsigned  i  =  0;  i  <  nx;  i++)
  printf("a[%d] = %d\n",  i,  a[i]);

  // Free Kokkos allocation (Unified Memory)
  Kokkos::kokkos_free(a);
  }

  // Finalize Kokkos
  Kokkos::finalize();
  return  0;
} 
```

```
// We're using OpenCL C API here, since SVM support in C++ API is unstable on
// ROCm
#define CL_TARGET_OPENCL_VERSION 200
#include  <CL/cl.h>
#include  <stdio.h>

// For larger kernels, we can store source in a separate file
static  const  char  *kernel_source  =  "              \
 __kernel void async(__global int *a) { \
 int i = get_global_id(0);            \
 int region = i / get_global_size(0); \
 a[i] = region + i;                   \
 }                                      \
 ";

int  main(int  argc,  char  *argv[])  {
  // Initialize OpenCL
  cl_platform_id  platform;
  clGetPlatformIDs(1,  &platform,  NULL);
  cl_device_id  device;
  clGetDeviceIDs(platform,  CL_DEVICE_TYPE_GPU,  1,  &device,  NULL);
  cl_context  context  =  clCreateContext(NULL,  1,  &device,  NULL,  NULL,  NULL);
  cl_command_queue  queue  =  clCreateCommandQueue(context,  device,  0,  NULL);

  // Compile OpenCL program for found device.
  cl_program  program  =
  clCreateProgramWithSource(context,  1,  &kernel_source,  NULL,  NULL);
  clBuildProgram(program,  1,  &device,  NULL,  NULL,  NULL);
  cl_kernel  kernel  =  clCreateKernel(program,  "async",  NULL);

  // Set problem dimensions
  unsigned  n  =  5;
  unsigned  nx  =  20;

  // Create SVM buffer objects on host side
  int  *a  =  clSVMAlloc(context,  CL_MEM_WRITE_ONLY,  nx  *  sizeof(int),  0);

  // Pass arguments to device kernel
  clSetKernelArgSVMPointer(kernel,  0,  a);

  // Launch multiple potentially asynchronous kernels on different parts of the
  // array
  for  (unsigned  region  =  0;  region  <  n;  region++)  {
  size_t  offset  =  (nx  /  n)  *  region;
  size_t  size  =  nx  /  n;
  clEnqueueNDRangeKernel(queue,  kernel,  1,  &offset,  &size,  NULL,  0,  NULL,
  NULL);
  }

  // Create mapping for host and print results
  clEnqueueSVMMap(queue,  CL_TRUE,  CL_MAP_READ,  a,  nx  *  sizeof(int),  0,  NULL,
  NULL);
  for  (unsigned  i  =  0;  i  <  nx;  i++)
  printf("a[%d] = %d\n",  i,  a[i]);
  clEnqueueSVMUnmap(queue,  a,  0,  NULL,  NULL);

  // Free SVM buffers
  clSVMFree(context,  a);

  return  0;
} 
```

```
#include  <sycl/sycl.hpp>

int  main()  {

  sycl::queue  q;
  unsigned  n  =  5;
  unsigned  nx  =  20;

  // Allocate shared memory (Unified Shared Memory)
  int  *a  =  sycl::malloc_shared<int>(nx,  q);

  // Launch multiple potentially asynchronous kernels on different parts of the
  // array
  for  (unsigned  region  =  0;  region  <  n;  region++)  {
  q.parallel_for(sycl::range<1>{n},  =  {
  const  int  iShifted  =  i  +  nx  /  n  *  region;
  a[iShifted]  =  region  +  iShifted;
  });
  }

  // Synchronize
  q.wait();

  // Print results
  for  (unsigned  i  =  0;  i  <  nx;  i++)
  printf("a[%d] = %d\n",  i,  a[i]);

  // Free shared memory allocation (Unified Memory)
  sycl::free(a,  q);

  return  0;
} 
```

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

auto  main()  ->  int  {
  unsigned  n  =  5;
  unsigned  nx  =  20;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Non-blocking device queue (requires synchronization)
  using  QueueType  =
  ap::onHost::Queue<ALPAKA_TYPEOF(devAcc),  ap::queueKind::NonBlocking>;
  std::vector<QueueType>  queues;
  for  (unsigned  region  =  0;  region  <  n;  region++)  {
  queues.emplace_back(devAcc.makeQueue(ap::queueKind::nonBlocking));
  }

  // Allocate unified memory that is accessible on host and device
  auto  a  =  ap::onHost::allocUnified<int>(devAcc,  nx);

  // Run element-wise multiplication on device
  for  (unsigned  region  =  0;  region  <  n;  region++)  {
  unsigned  nPerRegion  =  nx  /  n;
  unsigned  regionOffset  =  nPerRegion  *  region;
  ap::onHost::iota<int>(queues[region],  regionOffset,
  a.getSubView(regionOffset,  nx  -  regionOffset));
  }
  // Wait for the device, includes all queues
  ap::onHost::wait(devAcc);

  for  (unsigned  i  =  0;  i  <  nx;  i++)  printf("a[%d] = %d\n",  i,  a[i]);

  return  0;
} 
```

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

struct  IdxAssignKernel  {
  constexpr  void  operator()(ap::onAcc::concepts::Acc  auto  const&  acc,
  ap::concepts::IMdSpan  auto  a,
  unsigned  region,
  unsigned  n)  const  {
  unsigned  nPerRegion  =  a.getExtents().x()  /  n;
  unsigned  regionOffset  =  nPerRegion  *  region;
  for  (auto  [idx]  :
  ap::onAcc::makeIdxMap(acc,  ap::onAcc::worker::threadsInGrid,
  ap::IdxRange{regionOffset,  regionOffset  +  nPerRegion}))  {
  a[idx]  =  idx;
  }
  }
};

auto  main()  ->  int  {
  unsigned  n  =  5;
  unsigned  nx  =  20;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Non-blocking device queue (requires synchronization)
  using  QueueType  =
  ap::onHost::Queue<ALPAKA_TYPEOF(devAcc),  ap::queueKind::NonBlocking>;
  std::vector<QueueType>  queues;
  for  (unsigned  region  =  0;  region  <  n;  region++)  {
  queues.emplace_back(devAcc.makeQueue(ap::queueKind::nonBlocking));
  }

  // Allocate unified memory that is accessible on host and device
  auto  a  =  ap::onHost::allocUnified<int>(devAcc,  nx);

  unsigned  frameExtent  =  32u;
  auto  frameSpec  =
  ap::onHost::FrameSpec{ap::divExZero(nx,  frameExtent),  frameExtent};

  // Run element-wise multiplication on device
  for  (unsigned  region  =  0;  region  <  n;  region++)  {
  queues[region].enqueue(
  frameSpec,  ap::KernelBundle{IdxAssignKernel{},  a,  region,  n});
  }
  // Wait for the device, includes all queues
  ap::onHost::wait(devAcc);

  for  (unsigned  i  =  0;  i  <  nx;  i++)  printf("a[%d] = %d\n",  i,  a[i]);

  return  0;
} 
```

### 减少

```
#include  <cstdio>
#include  <execution>
#include  <numeric>
#include  <vector>

int  main()  {
  unsigned  n  =  10;

  std::vector<int>  a(n);

  std::iota(a.begin(),  a.end(),  0);  // Fill the array

  // Run reduction on the device
  int  sum  =  std::reduce(std::execution::par_unseq,  a.cbegin(),  a.cend(),  0,
  std::plus<int>{});

  // Print results
  printf("sum = %d\n",  sum);

  return  0;
} 
```

```
#include  <Kokkos_Core.hpp>

int  main(int  argc,  char  *argv[])  {

  // Initialize Kokkos
  Kokkos::initialize(argc,  argv);

  {
  unsigned  n  =  10;

  // Initialize sum variable
  int  sum  =  0;

  // Run sum reduction kernel
  Kokkos::parallel_reduce(
  n,  KOKKOS_LAMBDA(const  int  i,  int  &lsum)  {  lsum  +=  i;  },  sum);

  // Kokkos synchronization
  Kokkos::fence();

  // Print results
  printf("sum = %d\n",  sum);
  }

  // Finalize Kokkos
  Kokkos::finalize();
  return  0;
} 
```

```
// We're using OpenCL C++ API here; there is also C API in <CL/cl.h>
#define CL_TARGET_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110
#include  <CL/cl.hpp>

// For larger kernels, we can store source in a separate file
static  const  std::string  kernel_source  =  R"(
 __kernel void reduce(__global int* sum, __local int* local_mem) {

 // Get work group and work item information
 int gsize = get_global_size(0); // global work size
 int gid = get_global_id(0); // global work item index
 int lsize = get_local_size(0); // local work size
 int lid = get_local_id(0); // local work item index

 // Store reduced item into local memory
 local_mem[lid] = gid; // initialize local memory
 barrier(CLK_LOCAL_MEM_FENCE); // synchronize local memory

 // Perform reduction across the local work group
 for (int s = 1; s < lsize; s *= 2) { // loop over local memory with stride doubling each iteration
 if (lid % (2 * s) == 0 && (lid + s) < lsize) {
 local_mem[lid] += local_mem[lid + s];
 }
 barrier(CLK_LOCAL_MEM_FENCE); // synchronize local memory
 }

 if (lid == 0) { // only one work item per work group
 atomic_add(sum, local_mem[0]); // add partial sum to global sum atomically
 }
 }
  )";

int  main(int  argc,  char  *argv[])  {

  // Initialize OpenCL
  cl::Device  device  =  cl::Device::getDefault();
  cl::Context  context(device);
  cl::CommandQueue  queue(context,  device);

  // Compile OpenCL program for found device
  cl::Program  program(context,  kernel_source);
  program.build({device});
  cl::Kernel  kernel_reduce(program,  "reduce");

  {
  // Set problem dimensions
  unsigned  n  =  10;

  // Initialize sum variable
  int  sum  =  0;

  // Create buffer for sum
  cl::Buffer  buffer(context,  CL_MEM_READ_WRITE  |  CL_MEM_COPY_HOST_PTR,
  sizeof(int),  &sum);

  // Pass arguments to device kernel
  kernel_reduce.setArg(0,  buffer);  // pass buffer to device
  kernel_reduce.setArg(1,  sizeof(int),  NULL);  // allocate local memory

  // Enqueue kernel
  queue.enqueueNDRangeKernel(kernel_reduce,  cl::NullRange,  cl::NDRange(n),
  cl::NullRange);

  // Read result
  queue.enqueueReadBuffer(buffer,  CL_TRUE,  0,  sizeof(int),  &sum);

  // Print result
  printf("sum = %d\n",  sum);
  }

  return  0;
} 
```

```
// We use built-in sycl::reduction mechanism in this example.
// The manual implementation of the reduction kernel can be found in
// the "Non-portable kernel models" chapter.

#include  <sycl/sycl.hpp>

int  main()  {
  sycl::queue  q;
  unsigned  n  =  10;

  // Initialize sum
  int  sum  =  0;
  {
  // Create a buffer for sum to get the reduction results
  sycl::buffer<int>  sum_buf{&sum,  1};

  // Submit a SYCL kernel into a queue
  q.submit(&  {
  // Create temporary object describing variables with reduction semantics
  auto  sum_acc  =  sum_buf.get_access<sycl::access_mode::read_write>(cgh);
  // We can use built-in reduction primitive
  auto  sum_reduction  =  sycl::reduction(sum_acc,  sycl::plus<int>());

  // A reference to the reducer is passed to the lambda
  cgh.parallel_for(
  sycl::range<1>{n},  sum_reduction,
  =  {  reducer.combine(idx[0]);  });
  }).wait();
  // The contents of sum_buf are copied back to sum by the destructor of
  // sum_buf
  }
  // Print results
  printf("sum = %d\n",  sum);
} 
```

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

auto  main()  ->  int  {
  unsigned  n  =  10;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue  queue  =  devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate unified memory that is accessible on host and device
  auto  sum  =  ap::onHost::allocUnified<int>(devAcc,  1);

  // Run element-wise multiplication on device
  ap::onHost::reduce(queue,  0,  sum,  std::plus{},  ap::LinearizedIdxGenerator{n});

  // Print results
  printf("sum = %d\n",  sum[0]);

  return  0;
} 
```

## 跨平台可移植生态系统优缺点

### 一般观察

> +   代码重复量最小化。
> +   
> +   同一段代码可以编译成来自不同供应商的多种架构。
> +   
> +   与 CUDA 相比，学习资源有限（Stack Overflow、课程材料、文档）。

### 基于 Lambda 的内核模型（Kokkos, SYCL）

> +   更高级别的抽象。
> +   
> +   初始移植所需的底层架构知识较少。
> +   
> +   非常好读的源代码（C++ API）。
> +   
> +   这些模型相对较新，尚未非常流行。

### 基于 Functor 的内核模型（alpaka）

> +   非常好的可移植性。
> +   
> +   更高级别的抽象。
> +   
> +   低级 API 始终可用，这提供了更多控制并允许微调。
> +   
> +   为主机和内核代码提供用户友好的 C++ API。
> +   
> +   小型社区和生态系统。

### 独立源代码内核模型（OpenCL）

> +   非常好的可移植性。
> +   
> +   成熟的生态系统。
> +   
> +   供应商提供的库数量有限。
> +   
> +   低级 API 提供了更多控制并允许微调。
> +   
> +   提供 C 和 C++ API（C++ API 支持较差）。
> +   
> +   低级 API 和独立源代码内核模型对用户不太友好。

### C++标准并行（StdPar、PSTL）

> +   非常高级的抽象。
> +   
> +   容易加速依赖于 STL 算法的代码。
> +   
> +   对硬件的控制非常有限。
> +   
> +   编译器的支持正在改善，但尚未成熟。

重点

+   通用代码组织与非可移植的基于内核的模型相似。

+   只要没有使用特定于供应商的功能，相同的代码可以在任何 GPU 上运行。上一页 下一页

* * *

© 版权所有 2023-2024，贡献者。

使用[Read the Docs](https://readthedocs.org)提供的[主题](https://github.com/readthedocs/sphinx_rtd_theme)和[Sphinx](https://www.sphinx-doc.org/)构建。问题

+   如何使用 alpaka、C++ StdPar、Kokkos、OpenCL 和 SYCL 编程 GPU？

+   这些编程模型之间有什么区别？

目标

+   能够使用可移植的基于内核的模型编写简单代码

+   理解 Kokkos 和 SYCL 中不同内存和同步方法的工作原理

教师备注

+   60 分钟教学

+   30 分钟练习

跨平台可移植性生态系统的目标是允许相同的代码在多个架构上运行，从而减少代码重复。它们通常基于 C++，并使用函数对象/lambda 函数来定义循环体（即内核），这可以在多个架构上运行，如来自不同供应商的 CPU、GPU 和 FPGA。一个例外是 OpenCL，它最初只提供了 C API（尽管目前也提供了 C++ API），并且为内核代码使用单独的源模型。然而，与许多传统的 CUDA 或 HIP 实现 不同，可移植性生态系统要求如果用户希望它在 CPU 和 GPU 上运行，例如，内核只需编写一次。一些值得注意的跨平台可移植性生态系统包括 alpaka、Kokkos、OpenCL、RAJA 和 SYCL。Kokkos、alpaka 和 RAJA 是独立的项目，而 OpenCL 和 SYCL 是由多个项目遵循的标准，这些项目实现了（并扩展了）它们。例如，一些值得注意的 SYCL 实现包括 [Intel oneAPI DPC++](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html)、[AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp/)（之前称为 hipSYCL 或 Open SYCL）、[triSYCL](https://github.com/triSYCL/triSYCL) 和 [ComputeCPP](https://developer.codeplay.com/products/computecpp/ce/home/)。

## C++ StdPar

在 C++17 中，对标准算法并行执行的支持已经被引入。大多数通过标准 `<algorithms>` 头文件提供的算法都被赋予了接受一个 [*执行策略*](https://en.cppreference.com/w/cpp/algorithm) 参数的重载，这允许程序员请求并行执行标准库函数。虽然主要目标是允许低成本的、高级接口运行现有的算法，如 `std::sort` 在许多 CPU 核心上，但实现允许使用其他硬件，并且 `std::for_each` 或 `std::transform` 函数在编写算法时提供了很大的灵活性。

C++ StdPar，也称为并行 STL 或 PSTL，可以被认为是类似于指令驱动的模型，因为它非常高级，并且不给予程序员对数据移动的精细控制或对硬件特定功能（如共享（局部）内存）的访问。甚至用于运行的 GPU 也是自动选择的，因为标准 C++ 没有概念上的 *设备*（尽管有供应商扩展允许程序员有更多的控制）。然而，对于已经依赖于 C++ 标准库算法的应用程序，StdPar 可以是一种在最小代码修改的情况下获得 CPU 和 GPU 性能收益的好方法。

对于 GPU 编程，所有三个供应商都提供了他们的 StdPar 实现，可以将代码卸载到 GPU 上：NVIDIA 有 `nvc++`，AMD 有实验性的 [roc-stdpar](https://github.com/ROCm/roc-stdpar)，Intel 通过他们的 oneAPI 编译器提供 StdPar 卸载。[AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp/) 提供了一个独立的 StdPar 实现，能够针对所有三个供应商的设备。虽然 StdPar 是 C++ 标准的一部分，但不同编译器对 StdPar 的支持和成熟度差异很大：并非所有编译器都支持所有算法，将算法映射到硬件和进行数据移动的不同启发式方法可能会影响性能。

### StdPar 编译

构建过程在很大程度上取决于所使用的编译器：

+   AdaptiveCpp: 在调用 `acpp` 时添加 `--acpp-stdpar` 标志。

+   Intel oneAPI：在调用 `icpx` 时添加 `-fsycl -fsycl-pstl-offload=gpu` 标志。

+   NVIDIA NVC++：在调用 `nvc++` 时添加 `-stdpar` 标志（不支持使用普通 `nvcc`）。

### StdPar 编程

在其最简单的形式中，使用 C++ 标准并行性需要包含一个额外的 `<execution>` 头文件，并将一个参数添加到受支持的标准库函数中。

例如，让我们看看以下按顺序排序向量的代码：

```
#include  <algorithm>
#include  <vector>

void  f(std::vector<int>&  a)  {
  std::sort(a.begin(),  a.end());
} 
```

要使其在 GPU 上运行排序，只需要进行微小的修改：

```
#include  <algorithm>
#include  <vector>
#include  <execution> // To get std::execution

void  f(std::vector<int>&  a)  {
  std::sort(
  std::execution::par_unseq,  // This algorithm can be run in parallel
  a.begin(),  a.end()
  );
} 
```

现在，当使用支持的编译器编译时，代码将在 GPU 上运行排序。

虽然一开始可能看起来非常有限制，但许多标准算法，如 `std::transform`、`std::accumulate`、`std::transform_reduce` 和 `std::for_each`，可以在数组上运行自定义函数，从而允许将任意算法卸载，只要它不违反 GPU 内核的典型限制，例如不抛出任何异常和不进行系统调用。

### StdPar 执行策略

在 C++ 中，有四种不同的执行策略可供选择：

+   `std::execution::seq`：串行运行算法，不并行化它。

+   `std::execution::par`：允许并行化算法（就像使用多个线程一样），

+   `std::execution::unseq`：允许矢量化算法（就像使用 SIMD 一样），

+   `std::execution::par_unseq`：允许矢量化并并行化算法。

`par` 和 `unseq` 之间的主要区别与线程进度和锁有关：使用 `unseq` 或 `par_unseq` 要求算法在进程之间不包含互斥锁和其他锁，而 `par` 没有这种限制。

对于 GPU，最佳选择是 `par_unseq`，因为它对编译器在操作顺序方面的要求最低。虽然 `par` 在某些情况下也受到支持，但最好避免使用它，这不仅因为编译器支持有限，而且也表明该算法可能不适合硬件。

## Kokkos

Kokkos 是一个开源的性能可移植生态系统，用于在大型的异构硬件架构上并行化，其开发主要在桑迪亚国家实验室进行。该项目始于 2011 年，最初是一个并行 C++编程模型，但后来扩展成为一个更广泛的生态系统，包括 Kokkos Core（编程模型）、Kokkos Kernels（数学库）和 Kokkos Tools（调试、分析和调优工具）。通过为 C++标准委员会准备提案，该项目还旨在影响 ISO/C++语言标准，使得最终 Kokkos 的功能将成为语言标准的一部分。更详细的介绍可以在[这里](https://www.sandia.gov/news/publications/hpc-annual-reports/article/kokkos/)找到。

Kokkos 库为各种不同的并行编程模型提供了一个抽象层，目前包括 CUDA、HIP、SYCL、HPX、OpenMP 和 C++线程。因此，它允许在不同厂商制造的硬件之间实现更好的可移植性，但引入了软件堆栈的额外依赖。例如，当使用 CUDA 时，只需要 CUDA 安装，但当使用 Kokkos 与 NVIDIA GPU 一起时，需要 Kokkos 和 CUDA 的安装。Kokkos 并不是并行编程中非常受欢迎的选择，因此，与 CUDA 等更成熟的编程模型相比，学习和使用 Kokkos 可能会更加困难，对于 CUDA，可以找到大量的搜索结果和 Stack Overflow 讨论。

### Kokkos 编译

此外，一些跨平台可移植库的一个挑战是，即使在同一系统上，不同的项目可能需要不同的编译设置组合来满足可移植库的需求。例如，在 Kokkos 中，一个项目可能希望默认执行空间是 CUDA 设备，而另一个项目可能需要 CPU。即使项目偏好相同的执行空间，一个项目可能希望统一内存是默认内存空间，而另一个项目可能希望使用固定 GPU 内存。在单个系统上维护大量库实例可能会变得很麻烦。

然而，Kokkos 提供了一种简单的方法来同时编译 Kokkos 库和用户项目。这是通过指定 Kokkos 编译设置（见[这里](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Compiling.html)）并在用户 Makefile 中包含 Kokkos Makefile 来实现的。CMake 也受到支持。这样，用户应用程序和 Kokkos 库一起编译。以下是一个使用 CUDA（Volta 架构）作为后端（默认执行空间）和统一内存作为默认内存空间的单文件 Kokkos 项目的示例 Makefile：

```
default:  build

# Set compiler
KOKKOS_PATH  =  $(shell  pwd)/kokkos
CXX  =  hipcc
# CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper

# Variables for the Makefile.kokkos
KOKKOS_DEVICES  =  "HIP"
# KOKKOS_DEVICES = "Cuda"
KOKKOS_ARCH  =  "VEGA90A"
# KOKKOS_ARCH = "Volta70"
KOKKOS_CUDA_OPTIONS  =  "enable_lambda,force_uvm"

# Include Makefile.kokkos
include $(KOKKOS_PATH)/Makefile.kokkos

build:  $(KOKKOS_LINK_DEPENDS) $(KOKKOS_CPP_DEPENDS) hello.cpp
  $(CXX)  $(KOKKOS_CPPFLAGS)  $(KOKKOS_CXXFLAGS)  $(KOKKOS_LDFLAGS)  hello.cpp  $(KOKKOS_LIBS)  -o  hello 
```

要使用上述 Makefile 构建**hello.cpp**项目，除了将 Kokkos 项目克隆到当前目录外，不需要进行其他步骤。

### Kokkos 编程

当开始使用 Kokkos 编写项目时，第一步是了解 Kokkos 的初始化和终止。Kokkos 必须通过调用 `Kokkos::initialize(int& argc, char* argv[])` 进行初始化，并通过调用 `Kokkos::finalize()` 进行终止。更多详细信息请参阅[此处](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Initialization.html)。

Kokkos 使用执行空间模型来抽象并行硬件的细节。执行空间实例映射到可用的后端选项，如 CUDA、OpenMP、HIP 或 SYCL。如果程序员在源代码中没有明确选择执行空间，则使用默认执行空间 `Kokkos::DefaultExecutionSpace`。这是在编译 Kokkos 库时选择的。Kokkos 执行空间模型在[此处](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Machine-Model.html#kokkos-spaces)有更详细的描述。

类似地，Kokkos 使用内存空间模型来处理不同类型的内存，例如主机内存或设备内存。如果没有明确定义，Kokkos 将使用在 Kokkos 编译期间指定的默认内存空间，具体描述见[此处](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Machine-Model.html#kokkos-memory-spaces)。

以下是一个初始化 Kokkos 并打印执行空间和内存空间实例的 Kokkos 程序示例：

```
#include  <Kokkos_Core.hpp>
#include  <iostream>

int  main(int  argc,  char*  argv[])  {
  Kokkos::initialize(argc,  argv);
  std::cout  <<  "Execution Space: "  <<
  typeid(Kokkos::DefaultExecutionSpace).name()  <<  std::endl;
  std::cout  <<  "Memory Space: "  <<
  typeid(Kokkos::DefaultExecutionSpace::memory_space).name()  <<  std::endl;
  Kokkos::finalize();
  return  0;
} 
```

使用 Kokkos，数据可以通过原始指针或通过 Kokkos 视图进行访问。使用原始指针时，可以通过 `Kokkos::kokkos_malloc(n * sizeof(int))` 在默认内存空间中进行内存分配。Kokkos 视图是一种数据类型，它提供了一种更有效地访问与特定 Kokkos 内存空间（如主机内存或设备内存）相对应的数据的方法。可以通过 `Kokkos::View<int*> a("a", n)` 创建一个类型为 int* 的一维视图，其中 `"a"` 是标签，`n` 是以整数数量表示的分配大小。Kokkos 在编译时确定数据的最佳布局，以实现最佳的整体性能，这取决于计算机架构。此外，Kokkos 会自动处理此类内存的释放。有关 Kokkos 视图的更多详细信息，请参阅[此处](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/View.html)。

最后，Kokkos 提供了三种不同的并行操作：`parallel_for`、`parallel_reduce` 和 `parallel_scan`。`parallel_for` 操作用于并行执行循环。`parallel_reduce` 操作用于并行执行循环并将结果归约到单个值。`parallel_scan` 操作实现前缀扫描。`parallel_for` 和 `parallel_reduce` 的用法将在本章后面的示例中演示。有关并行操作的更多详细信息，请参阅[此处](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/ParallelDispatch.html)。

### 简单步骤运行 Kokkos hello.cpp 示例

以下内容应在 AMD VEGA90A 设备上直接使用（需要 ROCm 安装）。在 NVIDIA Volta V100 设备上（需要 CUDA 安装），请在 Makefile 中取消注释的变量。

1.  `git clone https://github.com/kokkos/kokkos.git`

1.  将上面的 Makefile 复制到当前文件夹中（确保最后一行的缩进是制表符，而不是空格）

1.  将上面的 hello.cpp 文件复制到当前文件夹

1.  `make`

1.  `./hello`

## OpenCL

OpenCL 是一个跨平台的开放标准 API，用于编写在由 CPU、GPU、FPGA 和其他设备组成的异构平台上执行的并行程序。OpenCL 的第一个版本（1.0）于 2008 年 12 月发布，最新的 OpenCL 版本（3.0）于 2020 年 9 月发布。OpenCL 由包括 AMD、ARM、Intel、NVIDIA 和 Qualcomm 在内的多个厂商支持。它是一个免版税标准，OpenCL 规范由 Khronos Group 维护。OpenCL 提供了一个基于 C 的低级编程接口，但最近也提供了一种 C++接口。

### OpenCL 编译

OpenCL 支持两种编译程序的模式：在线和离线。在线编译发生在运行时，当主机程序调用一个函数来编译源代码时。在线模式允许动态生成和加载内核，但可能因为编译时间和可能的错误而带来一些开销。离线编译发生在运行之前，当内核的源代码被编译成主机程序可以加载的二进制格式。这种模式允许内核更快地执行和更好的优化，但可能会限制程序的便携性，因为二进制只能在编译时指定的架构上运行。

OpenCL 附带了一些并行编程生态系统，例如 NVIDIA CUDA 和 Intel oneAPI。例如，在成功安装此类包并设置环境后，可以通过如`icx cl_devices.c -lOpenCL`（Intel oneAPI）或`nvcc cl_devices.c -lOpenCL`（NVIDIA CUDA）等命令简单地编译 OpenCL 程序，其中`cl_devices.c`是编译后的文件。与大多数其他编程模型不同，OpenCL 将内核存储为文本，并在运行时（即时编译）为设备编译，因此不需要任何特殊的编译器支持：只要所需的库和头文件安装在标准位置，就可以使用`gcc cl_devices.c -lOpenCL`（或使用 C++ API 时使用`g++`）来编译代码。

安装在 LUMI 上的 AMD 编译器支持 OpenCL C 和 C++ API，后者有一些限制。要编译程序，您可以使用 GPU 分区上的 AMD 编译器：

```
$ module  load  LUMI/24.03  partition/G
$ module  load  rocm/6.0.3
$ module  load  PrgEnv-cray-amd
$ CC  program.cpp  -lOpenCL  -o  program  # C++ program
$ cc  program.c  -lOpenCL  -o  program  # C program 
```

### OpenCL 编程

OpenCL 程序由两部分组成：在主机设备（通常是 CPU）上运行的宿主程序以及在一个或多个计算设备（如 GPU）上运行的内核。宿主程序负责管理所选平台上的设备、分配内存对象、构建和排队内核以及管理内存对象等任务。

编写 OpenCL 程序的第一步是初始化 OpenCL 环境，通过选择平台和设备，创建与所选设备关联的上下文或上下文，并为每个设备创建一个命令队列。以下是一个选择默认设备、创建与设备关联的上下文和队列的简单示例。

```
// Initialize OpenCL
cl::Device  device  =  cl::Device::getDefault();
cl::Context  context(device);
cl::CommandQueue  queue(context,  device); 
```

```
// Initialize OpenCL
cl_int  err;  // Error code returned by API calls
cl_platform_id  platform;
err  =  clGetPlatformIDs(1,  &platform,  NULL);
assert(err  ==  CL_SUCCESS);  // Checking error codes is skipped later for brevity
cl_device_id  device;
err  =  clGetDeviceIDs(platform,  CL_DEVICE_TYPE_GPU,  1,  &device,  NULL);
cl_context  context  =  clCreateContext(NULL,  1,  &device,  NULL,  NULL,  &err);
cl_command_queue  queue  =  clCreateCommandQueue(context,  device,  0,  &err); 
```

OpenCL 提供了两种主要的编程模型来管理主机和加速器设备的内存层次结构：缓冲区和共享虚拟内存（SVM）。缓冲区是 OpenCL 的传统内存模型，其中主机和设备有独立的地址空间，程序员必须显式指定内存分配以及如何以及在哪里访问内存。这可以通过`cl::Buffer`类和如`cl::CommandQueue::enqueueReadBuffer()`这样的函数来完成。缓冲区自 OpenCL 的早期版本以来就得到了支持，并且在不同架构上都能很好地工作。缓冲区还可以利用特定于设备的内存特性，如常量或局部内存。

SVM 是 OpenCL 中较新的内存模型，在 2.0 版本中引入，其中主机和设备共享一个单一的虚拟地址空间。因此，程序员可以使用相同的指针来访问主机和设备上的数据，从而简化编程工作。在 OpenCL 中，SVM 有不同级别，如粗粒度缓冲区 SVM、细粒度缓冲区 SVM 和细粒度系统 SVM。所有级别都允许在主机和设备之间使用相同的指针，但它们在内存区域的粒度和同步要求上有所不同。此外，SVM 的支持并不是在所有 OpenCL 平台和设备上都是通用的，例如，NVIDIA V100 和 A100 这样的 GPU 只支持粗粒度 SVM 缓冲区。这一级别需要显式同步从主机和设备对内存的访问（使用如`cl::CommandQueue::enqueueMapSVM()`和`cl::CommandQueue::enqueueUnmapSVM()`这样的函数），这使得 SVM 的使用不太方便。值得注意的是，这与 CUDA 提供的常规统一内存不同，CUDA 提供的统一内存更接近于 OpenCL 中的细粒度系统 SVM 级别。

OpenCL 使用单独源内核模型，其中内核代码通常保存在单独的文件中，这些文件可以在运行时编译。该模型允许将内核源代码作为字符串传递给 OpenCL 驱动程序，然后程序对象可以在特定设备上执行。尽管被称为单独源内核模型，但内核也可以在主机程序编译单元中定义为字符串，这在某些情况下可能更方便。

与二进制模型相比，使用单独源内核模型的在线编译具有多个优势，后者需要离线编译内核到特定于设备的二进制文件，这些文件在运行时由应用程序加载。在线编译保留了 OpenCL 的可移植性和灵活性，因为相同的内核源代码可以在任何支持的设备上运行。此外，基于运行时信息（如输入大小、工作组大小或设备功能）的内核动态优化也是可能的。以下是一个 OpenCL 内核的示例，它由主机编译单元中的字符串定义，并将全局线程索引分配到全局设备内存中。

```
static  const  std::string  kernel_source  =  R"(
 __kernel void dot(__global int *a) {
 int i = get_global_id(0);
 a[i] = i;
 }
)"; 
```

上面的名为 `dot` 并存储在字符串 `kernel_source` 中的内核可以设置在主机代码中构建，如下所示：

```
cl::Program  program(context,  kernel_source);
program.build({device});
cl::Kernel  kernel_dot(program,  "dot"); 
```

```
cl_int  err;
cl_program  program  =  clCreateProgramWithSource(context,  1,  &kernel_source,  NULL,  &err);
err  =  clBuildProgram(program,  1,  &device,  NULL,  NULL,  NULL);
cl_kernel  kernel_dot  =  clCreateKernel(program,  "vector_add",  &err); 
```

## SYCL

[SYCL](https://www.khronos.org/sycl/) 是一个免版税、开放标准的 C++ 编程模型，用于多设备编程。它为异构系统提供了一种高级、单源编程模型，包括 GPU。该标准有几个实现。对于 GPU 编程，[Intel oneAPI DPC++](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html) 和 [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp/)（也称为 hipSYCL）是桌面和 HPC GPU 中最受欢迎的；[ComputeCPP](https://developer.codeplay.com/products/computecpp/ce/home/) 是嵌入式设备的良好选择。遵循相同标准合规的 SYCL 代码应该可以在任何实现上运行，但它们不是二进制兼容的。

SYCL 标准的最新版本是 SYCL 2020，这是我们将在本课程中使用的版本。

### SYCL 编译

#### Intel oneAPI DPC++

对于 Intel GPU，只需安装 [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) 即可。然后，编译就像 `icpx -fsycl file.cpp` 一样简单。

也可以使用 oneAPI 来针对 NVIDIA 和 AMD GPU。除了 oneAPI Base Toolkit 之外，还需要安装供应商提供的运行时（CUDA 或 HIP）以及相应的 [Codeplay oneAPI 插件](https://codeplay.com/solutions/oneapi/)。然后，可以使用包含在 oneAPI 中的 Intel LLVM 编译器编译代码：

+   `clang++ -fsycl -fsycl-targets=nvidia_gpu_sm_86 file.cpp` 用于针对 CUDA 8.6 NVIDIA GPU，

+   `clang++ -fsycl -fsycl-targets=amd_gpu_gfx90a` 用于针对 GFX90a AMD GPU。

#### AdaptiveCpp

使用 AdaptiveCpp 为 NVIDIA 或 AMD GPU 编程也要求首先安装 CUDA 或 HIP。然后可以使用`acpp`编译代码，指定目标设备。例如，以下是编译支持 AMD 和 NVIDIA 设备的程序的方法：

+   `acpp --acpp-targets='hip:gfx90a;cuda:sm_70' file.cpp`

#### 在 LUMI 上使用 SYCL

LUMI 没有系统范围内安装任何 SYCL 框架，但 CSC 模块中有一个最新的 AdaptiveCpp 安装：

```
$ module  load  LUMI/24.03  partition/G
$ module  load  rocm/6.0.3
$ module  use  /appl/local/csc/modulefiles
$ module  load  acpp/24.06.0 
```

默认编译目标已预设为 MI250 GPU，因此要编译单个 C++文件，只需调用`acpp -O2 file.cpp`即可。

当运行使用 AdaptiveCpp 构建的应用程序时，经常会看到警告“dag_direct_scheduler: 检测到一个既不是丢弃访问模式的要求”，这反映了在使用缓冲区访问器模型时缺少优化提示。这个警告是无害的，可以忽略。

### SYCL 编程

在许多方面，SYCL 与 OpenCL 相似，但像 Kokkos 一样使用单源模型和内核 lambda。

要将任务提交到设备，首先必须创建一个 sycl::queue，它被用作管理任务调度和执行的方式。在最简单的情况下，这就是所需的全部初始化：

```
int  main()  {
  // Create an out-of-order queue on the default device:
  sycl::queue  q;
  // Now we can submit tasks to q!
} 
```

如果想要更多控制，可以显式指定设备，或者向队列传递额外的属性：

```
// Iterate over all available devices
for  (const  auto  &device  :  sycl::device::get_devices())  {
  // Print the device name
  std::cout  <<  "Creating a queue on "  <<  device.get_info<sycl::info::device::name>()  <<  "\n";
  // Create an in-order queue for the current device
  sycl::queue  q(device,  {sycl::property::queue::in_order()});
  // Now we can submit tasks to q!
} 
```

内存管理可以通过两种不同的方式完成：*缓冲区访问器*模型和*统一共享内存*（USM）。内存管理模型的选择也会影响 GPU 任务的同步方式。

在*缓冲区访问器*模型中，使用`sycl::buffer`对象来表示数据数组。缓冲区没有映射到任何单一内存空间，并且可以在 GPU 和 CPU 内存之间透明迁移。`sycl::buffer`中的数据不能直接读取或写入，必须创建访问器。`sycl::accessor`对象指定数据访问的位置（主机或某个 GPU 内核）以及访问模式（只读、只写、读写）。这种方法通过构建数据依赖的有向无环图（DAG）来优化任务调度：如果内核*A*创建了一个对缓冲区的只写访问器，然后内核*B*提交了一个对同一缓冲区的只读访问器，然后请求主机端的只读访问器，那么可以推断出*A*必须在*B*启动之前完成，并且结果必须在主机任务可以继续之前复制到主机，但主机任务可以与内核*B*并行运行。由于任务之间的依赖关系可以自动构建，因此默认情况下 SYCL 使用*乱序队列*：当两个任务提交到同一个`sycl::queue`时，不能保证第二个任务只有在第一个任务完成后才会启动。在启动内核时，必须创建访问器：

```
// Create a buffer of n integers
auto  buf  =  sycl::buffer<int>(sycl::range<1>(n));
// Submit a kernel into a queue; cgh is a helper object
q.submit(&  {
  // Create write-only accessor for buf
  auto  acc  =  buf.get_access<sycl::access_mode::write>(cgh);
  // Define a kernel: n threads execute the following lambda
  cgh.parallel_for<class  KernelName>(sycl::range<1>{n},  =  {
  // The data is written to the buffer via acc
  acc[i]  =  /*...*/
  });
});
/* If we now submit another kernel with accessor to buf, it will not
 * start running until the kernel above is done */ 
```

缓冲区访问模型简化了异构编程的许多方面，并防止了许多与同步相关的错误，但它只允许对数据移动和内核执行进行非常粗略的控制。

*USM* 模型类似于 NVIDIA CUDA 或 AMD HIP 管理内存的方式。程序员必须显式地在设备上（`sycl::malloc_device`）、在主机上（`sycl::malloc_host`）或在共享内存空间中（`sycl::malloc_shared`）分配内存。尽管其名称为统一共享内存，并且与 OpenCL 的 SVM 相似，但并非所有 USM 分配都是共享的：例如，由 `sycl::malloc_device` 分配的内存不能从主机访问。分配函数返回可以直接使用的内存指针，无需访问器。这意味着程序员必须确保主机和设备任务之间的正确同步，以避免数据竞争。使用 USM 时，通常更方便使用 *顺序队列* 而不是默认的 *乱序队列*。有关 USM 的更多信息，请参阅 [SYCL 2020 规范的第 4.8 节](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:usm)。

```
// Create a shared (migratable) allocation of n integers
// Unlike with buffers, we need to specify a queue (or, explicitly, a device and a context)
int*  v  =  sycl::malloc_shared<int>(n,  q);
// Submit a kernel into a queue; cgh is a helper object
q.submit(&  {
  // Define a kernel: n threads execute the following lambda
  cgh.parallel_for<class  KernelName>(sycl::range<1>{n},  =  {
  // The data is directly written to v
  v[i]  =  /*...*/
  });
});
// If we want to access v, we have to ensure that the kernel has finished
q.wait();
// After we're done, the memory must be deallocated
sycl::free(v,  q); 
```

### 练习

练习：在 SYCL 中实现 SAXPY

在这个练习中，我们希望编写（填空）一个简单的代码，执行 SAXPY（向量加法）。

要交互式地编译和运行代码，首先进行分配并加载 AdaptiveCpp 模块：

```
$ salloc  -A  project_465002387  -N  1  -t  1:00:00  -p  standard-g  --gpus-per-node=1
....
salloc: Granted job allocation 123456

$ module  load  LUMI/24.03  partition/G
$ module  use  /appl/local/csc/modulefiles
$ module  load  rocm/6.0.3  acpp/24.06.0 
```

现在，你可以运行一个简单的设备检测实用程序来检查是否有 GPU 可用（注意 `srun`）：

> ```
> $ srun  acpp-info  -l
> =================Backend information===================
> Loaded backend 0: HIP
>  Found device: AMD Instinct MI250X
> Loaded backend 1: OpenMP
>  Found device: hipSYCL OpenMP host device 
> ```

如果你还没有做，请使用 `git clone https://github.com/ENCCS/gpu-programming.git` 或 **更新** 它使用 `git pull origin main` 来克隆仓库。

现在，让我们看看 `content/examples/portable-kernel-models/exercise-sycl-saxpy.cpp` 中的示例代码：

```
#include  <iostream>
#include  <sycl/sycl.hpp>
#include  <vector>

int  main()  {
  // Create an in-order queue
  sycl::queue  q{sycl::property::queue::in_order()};
  // Print the device name, just for fun
  std::cout  <<  "Running on "
  <<  q.get_device().get_info<sycl::info::device::name>()  <<  std::endl;
  const  int  n  =  1024;  // Vector size

  // Allocate device and host memory for the first input vector
  float  *d_x  =  sycl::malloc_device<float>(n,  q);
  float  *h_x  =  sycl::malloc_host<float>(n,  q);
 // Bonus question: Can we use `std::vector` here instead of `malloc_host`? // TODO: Allocate second input vector on device and host, d_y and h_y  // Allocate device and host memory for the output vector
  float  *d_z  =  sycl::malloc_device<float>(n,  q);
  float  *h_z  =  sycl::malloc_host<float>(n,  q);

  // Initialize values on host
  for  (int  i  =  0;  i  <  n;  i++)  {
  h_x[i]  =  i;
 // TODO: Initialize h_y somehow  }
  const  float  alpha  =  0.42f;

  q.copy<float>(h_x,  d_x,  n);
 // TODO: Copy h_y to d_y // Bonus question: Why don't we need to wait before using the data? 
  // Run the kernel
  q.parallel_for(sycl::range<1>{n},  =  {
 // TODO: Modify the code to compute z[i] = alpha * x[i] + y[i]  d_z[i]  =  alpha  *  d_x[i];
  });

 // TODO: Copy d_z to h_z // TODO: Wait for the copy to complete 
  // Check the results
  bool  ok  =  true;
  for  (int  i  =  0;  i  <  n;  i++)  {
  float  ref  =  alpha  *  h_x[i]  +  h_y[i];  // Reference value
  float  tol  =  1e-5;  // Relative tolerance
  if  (std::abs((h_z[i]  -  ref))  >  tol  *  std::abs(ref))  {
  std::cout  <<  i  <<  " "  <<  h_z[i]  <<  " "  <<  h_x[i]  <<  " "  <<  h_y[i]
  <<  std::endl;
  ok  =  false;
  break;
  }
  }
  if  (ok)
  std::cout  <<  "Results are correct!"  <<  std::endl;
  else
  std::cout  <<  "Results are NOT correct!"  <<  std::endl;

  // Free allocated memory
  sycl::free(d_x,  q);
  sycl::free(h_x,  q);
 // TODO: Free d_y, h_y.  sycl::free(d_y,  q);
  sycl::free(h_y,  q);

  return  0;
} 
```

要编译和运行代码，请使用以下命令：

```
$ acpp  -O3  exercise-sycl-saxpy.cpp  -o  exercise-sycl-saxpy
$ srun  ./exercise-sycl-saxpy
Running on AMD Instinct MI250X
Results are correct! 
```

代码不能直接编译！你的任务是填写由 `TODO` 注释指示的缺失部分。你还可以通过代码中的“附加问题”来测试你的理解。

如果你感到困惑，请查看 `exercise-sycl-saxpy-solution.cpp` 文件。

## alpaka

[alpaka](https://github.com/alpaka-group/alpaka3) 库是一个开源的仅头文件 C++20 抽象库，用于加速器开发。

其目的是通过抽象底层并行级别，在加速器之间提供性能可移植性。该项目提供了一个单源 C++ API，使开发者能够编写一次并行代码，并在不同的硬件架构上运行而无需修改。名称“alpaka”来源于**A**bstractions for **L**evels of **P**arallelism, **A**lgorithms, and **K**ernels for **A**ccelerators。该库是平台无关的，并支持多个设备的并发和协作使用，包括主机 CPU（x86、ARM 和 RISC-V）以及来自不同供应商的 GPU（NVIDIA、AMD 和 Intel）。提供了各种加速器后端，如 CUDA、HIP、SYCL、OpenMP 和串行执行，可以根据目标设备进行选择。只需要一个用户内核的实现，以具有标准化接口的函数对象的形式表达。这消除了编写专门的 CUDA、HIP、SYCL、OpenMP、Intel TBB 或线程代码的需求。此外，可以将多个加速器后端组合起来，以针对单个系统甚至单个应用程序中的不同供应商硬件。

抽象基于一个虚拟索引域，该域被分解成大小相等的块，称为帧。**alpaka**提供了一种统一的抽象来遍历这些帧，独立于底层硬件。要并行化的算法将分块索引域和本地工作线程映射到数据上，将计算表示为在并行线程（SIMT）中执行的内核，从而也利用了 SIMD 单元。与 CUDA、HIP 和 SYCL 等本地并行模型不同，**alpaka**内核不受三维的限制。通过共享内存显式缓存帧内的数据允许开发者充分发挥计算设备性能。此外，**alpaka**还提供了诸如 iota、transform、transform-reduce、reduce 和 concurrent 等原始函数，简化了可移植高性能应用程序的开发。主机、设备、映射和管理多维视图提供了一种自然的方式来操作数据。

在这里，我们展示了 **alpaka3** 的用法，它是 [alpaka](https://github.com/alpaka-group/alpaka) 的完全重写。计划在 2026 年第二季度/第三季度的第一次发布之前，将这个独立的代码库合并回 alpaka 的主仓库。尽管如此，代码经过了良好的测试，并且可以用于今天的开发。

### 在您的系统上安装 alpaka

为了便于使用，我们建议按照以下说明使用 CMake 安装 alpaka。有关在项目中使用 alpaka 的其他方法，请参阅[alpaka3 文档](https://alpaka3.readthedocs.io/en/latest/basic/install.html)。

1.  **克隆仓库**

    从 GitHub 克隆 alpaka 源代码到您选择的目录：

    ```
    git  clone  https://github.com/alpaka-group/alpaka3.git
    cd  alpaka 
    ```

1.  **设置安装目录**

    将`ALPAKA_DIR`环境变量设置为想要安装 alpaka 的目录。这可以是您选择的任何目录，只要您有写入权限。

    ```
    export  ALPAKA_DIR=/path/to/your/alpaka/install/dir 
    ```

1.  **构建和安装**

    创建一个构建目录，并使用 CMake 构建 alpaka。我们使用 `CMAKE_INSTALL_PREFIX` 来告诉 CMake 将库安装在哪里。

    ```
    mkdir  build
    cmake  -B  build  -S  .  -DCMAKE_INSTALL_PREFIX=$ALPAKA_DIR
    cmake  --build  build  --parallel 
    ```

1.  **更新环境**

    为了确保其他项目可以找到你的 alpaka 安装，你应该将安装目录添加到你的 `CMAKE_PREFIX_PATH` 中。你可以通过将以下行添加到你的 shell 配置文件（例如 `~/.bashrc`）来实现：

    ```
    export  CMAKE_PREFIX_PATH=$ALPAKA_DIR:$CMAKE_PREFIX_PATH 
    ```

    你需要源码你的 shell 配置文件或打开一个新的终端以使更改生效。

### alpaka 编译

我们建议使用 CMake 构建使用 alpaka 的项目。可以采用各种策略来处理为特定设备或设备集构建你的应用程序。这里我们展示了入门的最小方法，但这绝对不是设置项目的唯一方法。请参阅 [alpaka3 文档](https://alpaka3.readthedocs.io/en/latest/basic/install.html)，了解在项目中使用 alpaka 的替代方法，包括在 CMake 中定义设备规范以使源代码与目标加速器无关的方法。

以下示例演示了一个使用 alpaka3 的单文件项目的 `CMakeLists.txt`（下面章节中展示的 `main.cpp`）：

> ```
> cmake_minimum_required(VERSION  3.25)
> project(myAlpakaApp  VERSION  1.0)
> 
> # Find installed alpaka
> find_package(alpaka  REQUIRED)
> 
> # Build the executable
> add_executable(myAlpakaApp  main.cpp)
> target_link_libraries(myAlpakaApp  PRIVATE  alpaka::alpaka)
> alpaka_finalize(myAlpakaApp) 
> ```

#### 在 LUMI 上使用 alpaka

要加载在 LUMI 上使用 HIP 的 AMD GPU 的环境，可以使用以下模块 -

```
$ module  load  LUMI/24.03  partition/G
$ module  load  rocm/6.0.3
$ module  load  buildtools/24.03
$ module  load  PrgEnv-amd
$ module  load  craype-accel-amd-gfx90a
$ export  CXX=hipcc 
```

### alpaka 编程

当开始使用 alpaka3 时，第一步是理解**设备选择模型**。与需要显式初始化调用的框架不同，alpaka3 使用设备规范来确定使用哪个后端和硬件。设备规范由两个组件组成：

+   **API**：并行编程接口（主机、cuda、hip、oneApi）

+   **设备类型**：硬件类型（cpu、nvidiaGpu、amdGpu、intelGpu）

在这里，我们指定并使用这些在运行时选择和初始化设备。设备选择过程在 alpaka3 文档中有详细描述。

alpaka3 使用**执行空间模型**来抽象并行硬件的细节。使用 `alpaka::onHost::makeDeviceSelector(devSpec)` 创建一个设备选择器，它返回一个可以查询可用设备并为所选后端创建设备实例的对象。

以下示例演示了一个基本的 alpaka 程序，该程序初始化一个设备并打印有关该设备的信息：

> ```
> #include  <alpaka/alpaka.hpp>
> #include  <cstdlib>
> #include  <iostream>
> 
> namespace  ap  =  alpaka;
> 
> auto  getDeviceSpec()
> {
>   /* Select a device, possible combinations of api+deviceKind:
>  * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
>  * oneApi+amdGpu, oneApi+nvidiaGpu
>  */
>   return  ap::onHost::DeviceSpec{ap::api::hip,  ap::deviceKind::amdGpu};
> }
> 
> int  main(int  argc,  char**  argv)
> {
>   // Initialize device specification and selector
>   ap::onHost::DeviceSpec  devSpec  =  getDeviceSpec();
>   auto  deviceSelector  =  ap::onHost::makeDeviceSelector(devSpec);
> 
>   // Query available devices
>   auto  num_devices  =  deviceSelector.getDeviceCount();
>   std::cout  <<  "Number of available devices: "  <<  num_devices  <<  "\n";
> 
>   if  (num_devices  ==  0)  {
>   std::cerr  <<  "No devices found for the selected backend\n";
>   return  EXIT_FAILURE;
>   }
> 
>   // Select and initialize the first device
>   auto  device  =  deviceSelector.makeDevice(0);
>   std::cout  <<  "Using device: "  <<  device.getName()  <<  "\n";
> 
>   return  EXIT_SUCCESS;
> } 
> ```

alpaka3 通过缓冲区和视图提供内存管理抽象。可以使用 `alpaka::allocBuf<T, Idx>(device, extent)` 在主机或设备上分配内存。主机和设备之间的数据传输通过 `alpaka::memcpy(queue, dst, src)` 处理。库自动管理不同架构上的内存布局以实现最佳性能。

对于并行执行，alpaka3 提供了内核抽象。内核定义为函数式对象或 lambda 函数，并使用定义并行化策略的工作划分规范执行。框架支持各种并行模式，包括逐元素操作、归约和扫描。

#### **alpaka** 功能巡览

现在我们将快速探索 alpaka 最常用的功能，并简要介绍一些基本用法。常用 alpaka 功能的快速参考可在此处找到。[链接](https://alpaka3.readthedocs.io/en/latest/basic/cheatsheet.html)

**通用设置**：包含一次综合头文件，然后即可开始使用 alpaka。

```
#include  <alpaka/alpaka.hpp>

namespace  myProject
{
  namespace  ap  =  alpaka;
  // Your code here
} 
```

**加速器、平台和设备管理**：通过将所需的 API 与适当的硬件类型结合使用设备选择器来选择设备。

```
auto  devSelector  =  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
if  (devSelector.getDeviceCount()  ==  0)
{
  throw  std::runtime_error("No device found!");
}
auto  device  =  devSelector.makeDevice(0); 
```

**队列和事件**：为每个设备创建阻塞或非阻塞队列，记录事件，并根据需要同步工作。

```
auto  queue  =  device.makeQueue();
auto  nonBlockingQueue  =  device.makeQueue(ap::queueKind::nonBlocking);
auto  blockingQueue  =  device.makeQueue(ap::queueKind::blocking);

auto  event  =  device.makeEvent();
queue.enqueue(event);
ap::onHost::wait(event);
ap::onHost::wait(queue); 
```

**内存管理**：分配主机、设备、映射、统一或延迟缓冲区，创建非拥有视图，并使用 memcpy、memset 和 fill 可移植地移动数据。

```
auto  hostBuffer  =  ap::onHost::allocHost<DataType>(extent3D);
auto  devBuffer  =  ap::onHost::alloc<DataType>(device,  extentMd);
auto  devMappedBuffer  =  ap::onHost::allocMapped<DataType>(device,  extentMd);

auto  hostView  =  ap::makeView(api::host,  externPtr,  ap::Vec{numElements});
auto  devNonOwningView  =  devBuffer.getView();

ap::onHost::memset(queue,  devBuffer,  uint8_t{0});
ap::onHost::memcpy(queue,  devBuffer,  hostBuffer);
ap::onHost::fill(queue,  devBuffer,  DataType{42}); 
```

**内核执行**：手动构建 FrameSpec 或请求针对您的数据类型调优的 FrameSpec，然后使用自动或显式执行器排队内核。

```
constexpr  uint32_t  dim  =  2u;
using  IdxType  =  size_t;
using  DataType  =  int;

IdxType  valueX,  valueY;
auto  extentMD  =  ap::Vec{valueY,  valueX};

auto  frameSpec  =  ap::onHost::FrameSpec{numFramesMd,  frameExtentMd};
auto  tunedSpec  =  ap::onHost::getFrameSpec<DataType>(device,  extentMd);

queue.enqueue(tunedSpec,  ap::KernelBundle{kernel,  kernelArgs...});

auto  executor  =  ap::exec::cpuSerial;
queue.enqueue(executor,  tunedSpec,  ap::KernelBundle{kernel,  kernelArgs...}); 
```

**内核实现**：将内核编写为带有 ALPAKA_FN_ACC 注解的函数式对象，并在内核体内部直接使用共享内存、同步、原子操作和数学助手。

```
struct  MyKernel
{
  ALPAKA_FN_ACC  void  operator()(ap::onAcc::concepts::Acc  auto  const&  acc,  auto...  args)  const
  {
  auto  idxMd  =  acc.getIdxWithin(ap::onAcc::origin::grid,  ap::onAcc::unit::blocks);

  auto  sharedMdArray  =
  ap::onAcc::declareSharedMdArray<float,  ap::uniqueId()>(acc,  ap::CVec<uint32_t,  3,  4>{});

  ap::onAcc::syncBlockThreads(acc);
  auto  old  =  onAcc::atomicAdd(acc,  args...);
  ap::onAcc::memFence(acc,  ap::onAcc::scope::block);
  auto  sinValue  =  ap::math::sin(args[0]);
  }
}; 
```

#### 简单步骤运行 alpaka3 示例

以下示例适用于 CMake 3.25+ 及以上版本和合适的 C++ 编译器。对于 GPU 执行，请确保已安装相应的运行时（CUDA、ROCm 或 oneAPI）。

1.  为您的项目创建一个目录：

    ```
    mkdir  my_alpaka_project  &&  cd  my_alpaka_project 
    ```

1.  将上面的 CMakeLists.txt 文件复制到当前文件夹

1.  将 main.cpp 文件复制到当前文件夹

1.  配置和构建：

    ```
    cmake  -B  build  -S  .  -Dalpaka_DEP_HIP=ON
    cmake  --build  build  --parallel 
    ```

1.  运行可执行文件：

    ```
    ./build/myAlpakaApp 
    ```

注意

设备指定系统允许你在 CMake 配置时选择目标设备。格式为 `"api:deviceKind"`，其中：

+   **api**：并行编程接口（`host`、`cuda`、`hip`、`oneApi`）

+   **deviceKind**：设备类型（`cpu`、`nvidiaGpu`、`amdGpu`、`intelGpu`）

可用的组合有：`host:cpu`、`cuda:nvidiaGpu`、`hip:amdGpu`、`oneApi:cpu`、`oneApi:intelGpu`、`oneApi:nvidiaGpu`、`oneApi:amdGpu`

警告

只有当 CUDA SDK、HIP SDK 或 OneAPI SDK 可用分别时，CUDA、HIP 或 Intel 后端才有效。

#### 预期输出

```
Number of available devices: 1
Using device: [Device Name] 
```

设备名称将根据您的硬件而变化（例如，“NVIDIA A100”、“AMD MI250X”或您的 CPU 型号）。

### 编译和执行示例

您可以从示例部分测试 alpaka 提供的示例。这些示例已硬编码了在 LUMI 上所需的 AMD ROCm 平台的使用。要仅使用 CPU，您只需将 `ap::onHost::makeDeviceSelector(ap::api::hip, ap::deviceKind::amdGpu);` 替换为 `ap::onHost::makeDeviceSelector(ap::api::host, ap::deviceKind::cpu);`

以下步骤假设你已经下载了 alpaka，并且**alapka**源代码的路径存储在环境变量`ALPAKA_DIR`中。要测试示例，请将代码复制到文件`main.cpp`中。

或者，[点击此处](https://godbolt.org/z/69exnG4xb)在 godbolt 编译器探索器中尝试第一个示例。

```
# use the following in C++ code
# auto devSelector = ap::onHost::makeDeviceSelector(ap::api::hip, ap::deviceKind::amdGpu);
# We use CC to refer to the compiler to work smoothly with the LUMI environment
CC  -I  $ALPAKA_DIR/include/  -std=c++20  -x  hip  --offload-arch=gfx90a  main.cpp
./a.out 
```

```
# use the following in C++ code
# auto devSelector = ap::onHost::makeDeviceSelector(ap::api::host, ap::deviceKind::cpu);
# We use CC to refer to the compiler to work smoothly with the LUMI environment
CC  -I  $ALPAKA_DIR/include/  -std=c++20  main.cpp
./a.out 
```

```
# use the following in C++ code
# auto devSelector = ap::onHost::makeDeviceSelector(ap::api::cuda, ap::deviceKind::nvidiaGpu);
nvcc  -I  $ALPAKA_DIR/include/  -std=c++20  --expt-relaxed-constexpr  -x  cuda  main.cpp
./a.out 
```

```
# use the following in C++ code
# auto devSelector = ap::onHost::makeDeviceSelector(ap::api::oneApi, ap::deviceKind::cpu);
icpx  -I  $ALPAKA_DIR/include/  -std=c++20  -fsycl  -fsycl-targets=spir64_x86_64  main.cpp
./a.out 
```

```
# use the following in C++ code
# auto devSelector = ap::onHost::makeDeviceSelector(ap::api::oneApi, ap::deviceKind::intelGpu);
icpx  -I  $ALPAKA_DIR/include/  -std=c++20  -fsycl  -fsycl-targets=spir64  main.cpp
./a.out 
```

注意

要使用 AMD 或 NVIDIA GPU 上的 oneAPI Sycl，你必须安装相应的 Codeplay oneAPI 插件，如[此处](https://codeplay.com/solutions/oneapi/plugins/)所述。

```
# use the following in C++ code
# auto devSelector = ap::onHost::makeDeviceSelector(ap::api::oneApi, ap::deviceKind::amdGpu);
icpx  -I  $ALPAKA_DIR/include/  -std=c++20  -fsycl  -fsycl-targets=amd_gpu_gfx90a  main.cpp
./a.out 
```

注意

要使用 AMD 或 NVIDIA GPU 上的 oneAPI Sycl，你必须安装相应的 Codeplay oneAPI 插件，如[此处](https://codeplay.com/solutions/oneapi/plugins/)所述。

```
# use the following in C++ code
# auto devSelector = ap::onHost::makeDeviceSelector(ap::api::oneApi, ap::deviceKind::nvidiaGpu);
icpx  -I  $ALPAKA_DIR/include/  -std=c++20  -fsycl  -fsycl-targets=nvptx64-nvidia-cuda  -Xsycl-target-backend=nvptx64-nvidia-cuda  --offload-arch=sm_80  main.cpp
./a.out 
```

### 练习

练习：在 alpaka 中编写向量加法内核

在这个练习中，我们希望编写（填空）一个简单的内核来添加两个向量。

要交互式编译和运行代码，首先我们需要在 GPU 节点上获取一个分配并加载 alpaka 模块：

```
$ srun  -p  dev-g  --gpus  1  -N  1  -n  1  --time=00:20:00  --account=project_465002387  --pty  bash
....
srun: job 1234 queued and waiting for resources
srun: job 1234 has been allocated resources

$ module  load  LUMI/24.03  partition/G
$ module  load  rocm/6.0.3
$ module  load  buildtools/24.03
$ module  load  PrgEnv-amd
$ module  load  craype-accel-amd-gfx90a
$ export  CXX=hipcc 
```

现在你可以运行一个简单的设备检测工具来检查 GPU 是否可用（注意`srun`）：

```
$ rocm-smi

======================================= ROCm System Management Interface =======================================
================================================= Concise Info =================================================
Device  [Model : Revision]    Temp    Power  Partitions      SCLK    MCLK     Fan  Perf    PwrCap  VRAM%  GPU%
 Name (20 chars)       (Edge)  (Avg)  (Mem, Compute)
================================================================================================================
0       [0x0b0c : 0x00]       45.0°C  N/A    N/A, N/A        800Mhz  1600Mhz  0%   manual  0.0W      0%   0%
 AMD INSTINCT MI200 (
================================================================================================================
============================================= End of ROCm SMI Log ============================================== 
```

现在，让我们看看设置练习的代码：

下面我们使用 CMake 的 fetch content 功能快速开始使用 alpaka。

CMakeLists.txt

```
cmake_minimum_required(VERSION  3.25)
project(vectorAdd  LANGUAGES  CXX  VERSION  1.0)
#Use CMake's FetchContent to download and integrate alpaka3 directly from GitHub
include(FetchContent)
#Declare where to fetch alpaka3 from
#This will download the library at configure time
FetchContent_Declare(alpaka3  GIT_REPOSITORY  https://github.com/alpaka-group/alpaka3.git  GIT_TAG  dev)
#Make alpaka3 available for use in this project
#This downloads, configures, and makes the library targets available
FetchContent_MakeAvailable(alpaka3)
#Finalize the alpaka FetchContent setup
alpaka_FetchContent_Finalize() #Create the executable target from the source file
add_executable(vectorAdd  main.cpp)
#Link the alpaka library to the executable
target_link_libraries(vectorAdd  PRIVATE  alpaka::alpaka)
#Finalize the alpaka configuration for this target
#This sets up backend - specific compiler flags and dependencies
alpaka_finalize(vectorAdd) 
```

下面是我们使用高级 transform 函数在设备上执行向量加法的主要 alpaka 代码。

main.cpp

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

auto  main()  ->  int  {
  unsigned  n  =  5;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */

  // auto devSelector = ap::onHost::makeDeviceSelector(ap::api::host,
  // ap::deviceKind::cpu);
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue  queue  =  devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate unified memory that is accessible on host and device
  auto  a  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  b  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  c  =  ap::onHost::allocUnified<int>(devAcc,  n);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

  // Run element-wise vector addition on device
 ap::onHost::transform(queue,  c,  std::plus{},  a,  b); 
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  c[i]);
  }

  return  0;
} 
```

要设置我们的项目，我们创建一个文件夹并将我们的 CMakeLists.txt 和 main.cpp 放在那里。

```
$ mkdir  alpakaExercise  &&  cd  alpakaExercise
$ vim  CMakeLists.txt
and now paste the CMakeLsits here (Press i, followed by Ctrl+Shift+V)
Press esc and then :wq to exit vim
$ vim  main.cpp
Similarly, paste the C++ code here 
```

要编译和运行代码，请使用以下命令：

```
configure step, we additionaly specify that HIP is available
$ cmake  -B  build  -S  .  -Dalpaka_DEP_HIP=ON
build
$ cmake  --build  build  --parallel
run
$ ./build/vectorAdd
Using alpaka device: AMD Instinct MI250X id=0
c[0] = 1
c[1] = 2
c[2] = 3
c[3] = 4
c[4] = 5 
```

现在的任务是编写和启动你的第一个 alpaka 内核。这个内核将执行向量加法，我们将使用这个代替 transform 辅助函数。

编写向量加法内核

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

struct  AddKernel  {
 constexpr  void  operator()(ap::onAcc::concepts::Acc  auto  const  &acc, ap::concepts::IMdSpan  auto  c, ap::concepts::IMdSpan  auto  const  a, ap::concepts::IMdSpan  auto  const  b)  const  { for  (auto  idx  :  ap::onAcc::makeIdxMap(acc,  ap::onAcc::worker::threadsInGrid, ap::IdxRange{c.getExtents()}))  { c[idx]  =  a[idx]  +  b[idx]; } } }; 
auto  main()  ->  int  {
  unsigned  n  =  5;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */

  // auto devSelector = ap::onHost::makeDeviceSelector(ap::api::host,
  // ap::deviceKind::cpu);
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue  queue  =  devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate unified memory that is accessible on host and device
  auto  a  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  b  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  c  =  ap::onHost::allocUnified<int>(devAcc,  n);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

 auto  frameSpec  =  ap::onHost::getFrameSpec<int>(devAcc,  c.getExtents()); 
 // Call the element-wise addition kernel on device queue.enqueue(frameSpec,  ap::KernelBundle{AddKernel{},  c,  a,  b}); 
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  c[i]);
  }

  return  0;
} 
```

## 示例

### 带统一内存的并行 for 循环

```
#include  <algorithm>
#include  <cstdio>
#include  <execution>
#include  <vector>

int  main()  {
  unsigned  n  =  5;

  // Allocate arrays
  std::vector<int>  a(n),  b(n),  c(n);

  // Initialize values
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

  // Run element-wise multiplication on device
  std::transform(std::execution::par_unseq,  a.begin(),  a.end(),  b.begin(),
  c.begin(),  [](int  i,  int  j)  {  return  i  *  j;  });

  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  c[i]);
  }

  return  0;
} 
```

```
#include  <Kokkos_Core.hpp>

int  main(int  argc,  char  *argv[])  {

  // Initialize Kokkos
  Kokkos::initialize(argc,  argv);

  {
  unsigned  n  =  5;

  // Allocate on Kokkos default memory space (Unified Memory)
  int  *a  =  (int  *)Kokkos::kokkos_malloc(n  *  sizeof(int));
  int  *b  =  (int  *)Kokkos::kokkos_malloc(n  *  sizeof(int));
  int  *c  =  (int  *)Kokkos::kokkos_malloc(n  *  sizeof(int));

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

  // Run element-wise multiplication on device
  Kokkos::parallel_for(n,  KOKKOS_LAMBDA(const  int  i)  {  c[i]  =  a[i]  *  b[i];  });

  // Kokkos synchronization
  Kokkos::fence();

  // Print results
  for  (unsigned  i  =  0;  i  <  n;  i++)
  printf("c[%d] = %d\n",  i,  c[i]);

  // Free Kokkos allocation (Unified Memory)
  Kokkos::kokkos_free(a);
  Kokkos::kokkos_free(b);
  Kokkos::kokkos_free(c);
  }

  // Finalize Kokkos
  Kokkos::finalize();
  return  0;
} 
```

```
// We're using OpenCL C API here, since SVM support in C++ API is unstable on
// ROCm
#define CL_TARGET_OPENCL_VERSION 220
#include  <CL/cl.h>
#include  <stdio.h>

// For larger kernels, we can store source in a separate file
static  const  char  *kernel_source  =
  "                                                 \
 __kernel void dot(__global const int *a, __global const int *b, __global int *c) { \
 int i = get_global_id(0);                                                        \
 c[i] = a[i] * b[i];                                                              \
 }                                                                                  \
 ";

int  main(int  argc,  char  *argv[])  {

  // Initialize OpenCL
  cl_platform_id  platform;
  clGetPlatformIDs(1,  &platform,  NULL);
  cl_device_id  device;
  clGetDeviceIDs(platform,  CL_DEVICE_TYPE_GPU,  1,  &device,  NULL);
  cl_context  context  =  clCreateContext(NULL,  1,  &device,  NULL,  NULL,  NULL);
  cl_command_queue  queue  =  clCreateCommandQueue(context,  device,  0,  NULL);

  // Compile OpenCL program for found device.
  cl_program  program  =
  clCreateProgramWithSource(context,  1,  &kernel_source,  NULL,  NULL);
  clBuildProgram(program,  1,  &device,  NULL,  NULL,  NULL);
  cl_kernel  kernel  =  clCreateKernel(program,  "dot",  NULL);

  // Set problem dimensions
  unsigned  n  =  5;

  // Create SVM buffer objects on host side
  int  *a  =  clSVMAlloc(context,  CL_MEM_READ_ONLY,  n  *  sizeof(int),  0);
  int  *b  =  clSVMAlloc(context,  CL_MEM_READ_ONLY,  n  *  sizeof(int),  0);
  int  *c  =  clSVMAlloc(context,  CL_MEM_WRITE_ONLY,  n  *  sizeof(int),  0);

  // Pass arguments to device kernel
  clSetKernelArgSVMPointer(kernel,  0,  a);
  clSetKernelArgSVMPointer(kernel,  1,  b);
  clSetKernelArgSVMPointer(kernel,  2,  c);

  // Create mappings for host and initialize values
  clEnqueueSVMMap(queue,  CL_TRUE,  CL_MAP_WRITE,  a,  n  *  sizeof(int),  0,  NULL,
  NULL);
  clEnqueueSVMMap(queue,  CL_TRUE,  CL_MAP_WRITE,  b,  n  *  sizeof(int),  0,  NULL,
  NULL);
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }
  clEnqueueSVMUnmap(queue,  a,  0,  NULL,  NULL);
  clEnqueueSVMUnmap(queue,  b,  0,  NULL,  NULL);

  size_t  globalSize  =  n;
  clEnqueueNDRangeKernel(queue,  kernel,  1,  NULL,  &globalSize,  NULL,  0,  NULL,
  NULL);

  // Create mapping for host and print results
  clEnqueueSVMMap(queue,  CL_TRUE,  CL_MAP_READ,  c,  n  *  sizeof(int),  0,  NULL,
  NULL);
  for  (unsigned  i  =  0;  i  <  n;  i++)
  printf("c[%d] = %d\n",  i,  c[i]);
  clEnqueueSVMUnmap(queue,  c,  0,  NULL,  NULL);

  // Free SVM buffers
  clSVMFree(context,  a);
  clSVMFree(context,  b);
  clSVMFree(context,  c);

  return  0;
} 
```

```
#include  <sycl/sycl.hpp>

int  main()  {

  sycl::queue  q;
  unsigned  n  =  5;

  // Allocate shared memory (Unified Shared Memory)
  int  *a  =  sycl::malloc_shared<int>(n,  q);
  int  *b  =  sycl::malloc_shared<int>(n,  q);
  int  *c  =  sycl::malloc_shared<int>(n,  q);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

  // Run element-wise multiplication on device
  q.parallel_for(sycl::range<1>{n},  =  {
  c[i]  =  a[i]  *  b[i];
  }).wait();

  // Print results
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  c[i]);
  }

  // Free shared memory allocation (Unified Memory)
  sycl::free(a,  q);
  sycl::free(b,  q);
  sycl::free(c,  q);

  return  0;
} 
```

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

auto  main()  ->  int  {
  unsigned  n  =  5;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue  queue  =  devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate unified memory that is accessible on host and device
  auto  a  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  b  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  c  =  ap::onHost::allocUnified<int>(devAcc,  n);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

  // Run element-wise multiplication on device
  ap::onHost::transform(queue,  c,  std::multiplies{},  a,  b);

  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  c[i]);
  }

  return  0;
} 
```

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

struct  MulKernel  {
  constexpr  void  operator()(ap::onAcc::concepts::Acc  auto  const&  acc,
  ap::concepts::IMdSpan  auto  c,
  ap::concepts::IMdSpan  auto  const  a,
  ap::concepts::IMdSpan  auto  const  b)  const  {
  for  (auto  idx  :  ap::onAcc::makeIdxMap(acc,  ap::onAcc::worker::threadsInGrid,
  ap::IdxRange{c.getExtents()}))  {
  c[idx]  =  a[idx]  *  b[idx];
  }
  }
};

auto  main()  ->  int  {
  unsigned  n  =  5;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue  queue  =  devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate unified memory that is accessible on host and device
  auto  a  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  b  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  c  =  ap::onHost::allocUnified<int>(devAcc,  n);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

  unsigned  frameExtent  =  32u;
  auto  frameSpec  =
  ap::onHost::FrameSpec{ap::divExZero(n,  frameExtent),  frameExtent};

  // Run element-wise multiplication on device
  queue.enqueue(frameSpec,  ap::KernelBundle{MulKernel{},  c,  a,  b});

  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  c[i]);
  }

  return  0;
} 
```

### 带 GPU 缓冲区的并行 for 循环

```
#include  <Kokkos_Core.hpp>

int  main(int  argc,  char  *argv[])  {

  // Initialize Kokkos
  Kokkos::initialize(argc,  argv);

  {
  unsigned  n  =  5;

  // Allocate space for 5 ints on Kokkos host memory space
  Kokkos::View<int  *,  Kokkos::HostSpace>  h_a("h_a",  n);
  Kokkos::View<int  *,  Kokkos::HostSpace>  h_b("h_b",  n);
  Kokkos::View<int  *,  Kokkos::HostSpace>  h_c("h_c",  n);

  // Allocate space for 5 ints on Kokkos default memory space (eg, GPU memory)
  Kokkos::View<int  *>  a("a",  n);
  Kokkos::View<int  *>  b("b",  n);
  Kokkos::View<int  *>  c("c",  n);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  h_a[i]  =  i;
  h_b[i]  =  1;
  }

  // Copy from host to device
  Kokkos::deep_copy(a,  h_a);
  Kokkos::deep_copy(b,  h_b);

  // Run element-wise multiplication on device
  Kokkos::parallel_for(n,  KOKKOS_LAMBDA(const  int  i)  {  c[i]  =  a[i]  *  b[i];  });

  // Copy from device to host
  Kokkos::deep_copy(h_c,  c);

  // Print results
  for  (unsigned  i  =  0;  i  <  n;  i++)
  printf("c[%d] = %d\n",  i,  h_c[i]);
  }

  // Finalize Kokkos
  Kokkos::finalize();
  return  0;
} 
```

```
// We're using OpenCL C++ API here; there is also C API in <CL/cl.h>
#define CL_TARGET_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110
#include  <CL/cl.hpp>

// For larger kernels, we can store source in a separate file
static  const  std::string  kernel_source  =  R"(
 __kernel void dot(__global const int *a, __global const int *b, __global int *c) {
 int i = get_global_id(0);
 c[i] = a[i] * b[i];
 }
  )";

int  main(int  argc,  char  *argv[])  {

  // Initialize OpenCL
  cl::Device  device  =  cl::Device::getDefault();
  cl::Context  context(device);
  cl::CommandQueue  queue(context,  device);

  // Compile OpenCL program for found device.
  cl::Program  program(context,  kernel_source);
  program.build({device});
  cl::Kernel  kernel_dot(program,  "dot");

  {
  // Set problem dimensions
  unsigned  n  =  5;

  std::vector<int>  a(n),  b(n),  c(n);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

  // Create buffers and copy input data to device.
  cl::Buffer  dev_a(context,  CL_MEM_READ_ONLY  |  CL_MEM_COPY_HOST_PTR,
  n  *  sizeof(int),  a.data());
  cl::Buffer  dev_b(context,  CL_MEM_READ_ONLY  |  CL_MEM_COPY_HOST_PTR,
  n  *  sizeof(int),  b.data());
  cl::Buffer  dev_c(context,  CL_MEM_WRITE_ONLY,  n  *  sizeof(int));

  // Pass arguments to device kernel
  kernel_dot.setArg(0,  dev_a);
  kernel_dot.setArg(1,  dev_b);
  kernel_dot.setArg(2,  dev_c);

  // We don't need to apply any offset to thread IDs
  queue.enqueueNDRangeKernel(kernel_dot,  cl::NullRange,  cl::NDRange(n),
  cl::NullRange);

  // Read result
  queue.enqueueReadBuffer(dev_c,  CL_TRUE,  0,  n  *  sizeof(int),  c.data());

  // Print results
  for  (unsigned  i  =  0;  i  <  n;  i++)
  printf("c[%d] = %d\n",  i,  c[i]);
  }

  return  0;
} 
```

```
#include  <sycl/sycl.hpp>

int  main()  {

  sycl::queue  q;
  unsigned  n  =  5;

  // Allocate space for 5 ints
  auto  a_buf  =  sycl::buffer<int>(sycl::range<1>(n));
  auto  b_buf  =  sycl::buffer<int>(sycl::range<1>(n));
  auto  c_buf  =  sycl::buffer<int>(sycl::range<1>(n));

  // Initialize values
  // We should use curly braces to limit host accessors' lifetime
  //    and indicate when we're done working with them:
  {
  auto  a_host_acc  =  a_buf.get_host_access();
  auto  b_host_acc  =  b_buf.get_host_access();
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a_host_acc[i]  =  i;
  b_host_acc[i]  =  1;
  }
  }

  // Submit a SYCL kernel into a queue
  q.submit(&  {
  // Create read accessors over a_buf and b_buf
  auto  a_acc  =  a_buf.get_access<sycl::access_mode::read>(cgh);
  auto  b_acc  =  b_buf.get_access<sycl::access_mode::read>(cgh);
  // Create write accesor over c_buf
  auto  c_acc  =  c_buf.get_access<sycl::access_mode::write>(cgh);
  // Run element-wise multiplication on device
  cgh.parallel_for<class  vec_add>(sycl::range<1>{n},  =  {
  c_acc[i]  =  a_acc[i]  *  b_acc[i];
  });
  });

  // No need to synchronize, creating the accessor for c_buf will do it
  // automatically
  {
  const  auto  c_host_acc  =  c_buf.get_host_access();
  // Print results
  for  (unsigned  i  =  0;  i  <  n;  i++)
  printf("c[%d] = %d\n",  i,  c_host_acc[i]);
  }

  return  0;
} 
```

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

auto  main()  ->  int  {
  unsigned  n  =  5;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue  queue  =  devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate memory that is accessible on host
  auto  h_a  =  ap::onHost::allocHost<int>(n);
  auto  h_b  =  ap::onHost::allocHostLike(h_a);
  auto  h_c  =  ap::onHost::allocHostLike(h_a);

  // Allocate memory on the device and inherit the extents from h_a
  auto  a  =  ap::onHost::allocLike(devAcc,  h_a);
  auto  b  =  ap::onHost::allocLike(devAcc,  h_a);
  auto  c  =  ap::onHost::allocLike(devAcc,  h_a);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  h_a[i]  =  i;
  h_b[i]  =  1;
  }

  // Copy host memory element wise to the device memory
  ap::onHost::memcpy(queue,  a,  h_a);
  ap::onHost::memcpy(queue,  b,  h_b);

  // Run element-wise multiplication on device
  ap::onHost::transform(queue,  c,  std::multiplies{},  a,  b);

  // Copy the device result back to host memory
  ap::onHost::memcpy(queue,  h_c,  c);

  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  h_c[i]);
  }

  return  0;
} 
```

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

struct  MulKernel  {
  constexpr  void  operator()(ap::onAcc::concepts::Acc  auto  const&  acc,
  ap::concepts::IMdSpan  auto  c,
  ap::concepts::IMdSpan  auto  const  a,
  ap::concepts::IMdSpan  auto  const  b)  const  {
  for  (auto  idx  :  ap::onAcc::makeIdxMap(acc,  ap::onAcc::worker::threadsInGrid,
  ap::IdxRange{c.getExtents()}))  {
  c[idx]  =  a[idx]  *  b[idx];
  }
  }
};

auto  main()  ->  int  {
  unsigned  n  =  5;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue  queue  =  devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate memory that is accessible on host
  auto  h_a  =  ap::onHost::allocHost<int>(n);
  auto  h_b  =  ap::onHost::allocHostLike(h_a);
  auto  h_c  =  ap::onHost::allocHostLike(h_a);

  // allocate memory on the device and inherit the extents from a
  auto  a  =  ap::onHost::allocLike(devAcc,  h_a);
  auto  b  =  ap::onHost::allocLike(devAcc,  h_a);
  auto  c  =  ap::onHost::allocLike(devAcc,  h_a);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  h_a[i]  =  i;
  h_b[i]  =  1;
  }

  // Copy host memory element wise to the device memory
  ap::onHost::memcpy(queue,  a,  h_a);
  ap::onHost::memcpy(queue,  b,  h_b);

  unsigned  frameExtent  =  32u;
  auto  frameSpec  =
  ap::onHost::FrameSpec{ap::divExZero(n,  frameExtent),  frameExtent};

  // Run element-wise multiplication on device
  queue.enqueue(frameSpec,  ap::KernelBundle{MulKernel{},  c,  a,  b});

  // Copy the device result back to host memory
  ap::onHost::memcpy(queue,  h_c,  c);

  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  h_c[i]);
  }

  return  0;
} 
```

### 异步并行 for 内核

```
#include  <Kokkos_Core.hpp>

int  main(int  argc,  char  *argv[])  {

  // Initialize Kokkos
  Kokkos::initialize(argc,  argv);

  {
  unsigned  n  =  5;
  unsigned  nx  =  20;

  // Allocate on Kokkos default memory space (Unified Memory)
  int  *a  =  (int  *)Kokkos::kokkos_malloc(nx  *  sizeof(int));

  // Create 'n' execution space instances (maps to streams in CUDA/HIP)
  auto  ex  =  Kokkos::Experimental::partition_space(
  Kokkos::DefaultExecutionSpace(),  1,  1,  1,  1,  1);

  // Launch 'n' potentially asynchronous kernels
  // Each kernel has their own execution space instances
  for  (unsigned  region  =  0;  region  <  n;  region++)  {
  Kokkos::parallel_for(
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
  ex[region],  nx  /  n  *  region,  nx  /  n  *  (region  +  1)),
  KOKKOS_LAMBDA(const  int  i)  {  a[i]  =  region  +  i;  });
  }

  // Sync execution space instances (maps to streams in CUDA/HIP)
  for  (unsigned  region  =  0;  region  <  n;  region++)
  ex[region].fence();

  // Print results
  for  (unsigned  i  =  0;  i  <  nx;  i++)
  printf("a[%d] = %d\n",  i,  a[i]);

  // Free Kokkos allocation (Unified Memory)
  Kokkos::kokkos_free(a);
  }

  // Finalize Kokkos
  Kokkos::finalize();
  return  0;
} 
```

```
// We're using OpenCL C API here, since SVM support in C++ API is unstable on
// ROCm
#define CL_TARGET_OPENCL_VERSION 200
#include  <CL/cl.h>
#include  <stdio.h>

// For larger kernels, we can store source in a separate file
static  const  char  *kernel_source  =  "              \
 __kernel void async(__global int *a) { \
 int i = get_global_id(0);            \
 int region = i / get_global_size(0); \
 a[i] = region + i;                   \
 }                                      \
 ";

int  main(int  argc,  char  *argv[])  {
  // Initialize OpenCL
  cl_platform_id  platform;
  clGetPlatformIDs(1,  &platform,  NULL);
  cl_device_id  device;
  clGetDeviceIDs(platform,  CL_DEVICE_TYPE_GPU,  1,  &device,  NULL);
  cl_context  context  =  clCreateContext(NULL,  1,  &device,  NULL,  NULL,  NULL);
  cl_command_queue  queue  =  clCreateCommandQueue(context,  device,  0,  NULL);

  // Compile OpenCL program for found device.
  cl_program  program  =
  clCreateProgramWithSource(context,  1,  &kernel_source,  NULL,  NULL);
  clBuildProgram(program,  1,  &device,  NULL,  NULL,  NULL);
  cl_kernel  kernel  =  clCreateKernel(program,  "async",  NULL);

  // Set problem dimensions
  unsigned  n  =  5;
  unsigned  nx  =  20;

  // Create SVM buffer objects on host side
  int  *a  =  clSVMAlloc(context,  CL_MEM_WRITE_ONLY,  nx  *  sizeof(int),  0);

  // Pass arguments to device kernel
  clSetKernelArgSVMPointer(kernel,  0,  a);

  // Launch multiple potentially asynchronous kernels on different parts of the
  // array
  for  (unsigned  region  =  0;  region  <  n;  region++)  {
  size_t  offset  =  (nx  /  n)  *  region;
  size_t  size  =  nx  /  n;
  clEnqueueNDRangeKernel(queue,  kernel,  1,  &offset,  &size,  NULL,  0,  NULL,
  NULL);
  }

  // Create mapping for host and print results
  clEnqueueSVMMap(queue,  CL_TRUE,  CL_MAP_READ,  a,  nx  *  sizeof(int),  0,  NULL,
  NULL);
  for  (unsigned  i  =  0;  i  <  nx;  i++)
  printf("a[%d] = %d\n",  i,  a[i]);
  clEnqueueSVMUnmap(queue,  a,  0,  NULL,  NULL);

  // Free SVM buffers
  clSVMFree(context,  a);

  return  0;
} 
```

```
#include  <sycl/sycl.hpp>

int  main()  {

  sycl::queue  q;
  unsigned  n  =  5;
  unsigned  nx  =  20;

  // Allocate shared memory (Unified Shared Memory)
  int  *a  =  sycl::malloc_shared<int>(nx,  q);

  // Launch multiple potentially asynchronous kernels on different parts of the
  // array
  for  (unsigned  region  =  0;  region  <  n;  region++)  {
  q.parallel_for(sycl::range<1>{n},  =  {
  const  int  iShifted  =  i  +  nx  /  n  *  region;
  a[iShifted]  =  region  +  iShifted;
  });
  }

  // Synchronize
  q.wait();

  // Print results
  for  (unsigned  i  =  0;  i  <  nx;  i++)
  printf("a[%d] = %d\n",  i,  a[i]);

  // Free shared memory allocation (Unified Memory)
  sycl::free(a,  q);

  return  0;
} 
```

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

auto  main()  ->  int  {
  unsigned  n  =  5;
  unsigned  nx  =  20;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Non-blocking device queue (requires synchronization)
  using  QueueType  =
  ap::onHost::Queue<ALPAKA_TYPEOF(devAcc),  ap::queueKind::NonBlocking>;
  std::vector<QueueType>  queues;
  for  (unsigned  region  =  0;  region  <  n;  region++)  {
  queues.emplace_back(devAcc.makeQueue(ap::queueKind::nonBlocking));
  }

  // Allocate unified memory that is accessible on host and device
  auto  a  =  ap::onHost::allocUnified<int>(devAcc,  nx);

  // Run element-wise multiplication on device
  for  (unsigned  region  =  0;  region  <  n;  region++)  {
  unsigned  nPerRegion  =  nx  /  n;
  unsigned  regionOffset  =  nPerRegion  *  region;
  ap::onHost::iota<int>(queues[region],  regionOffset,
  a.getSubView(regionOffset,  nx  -  regionOffset));
  }
  // Wait for the device, includes all queues
  ap::onHost::wait(devAcc);

  for  (unsigned  i  =  0;  i  <  nx;  i++)  printf("a[%d] = %d\n",  i,  a[i]);

  return  0;
} 
```

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

struct  IdxAssignKernel  {
  constexpr  void  operator()(ap::onAcc::concepts::Acc  auto  const&  acc,
  ap::concepts::IMdSpan  auto  a,
  unsigned  region,
  unsigned  n)  const  {
  unsigned  nPerRegion  =  a.getExtents().x()  /  n;
  unsigned  regionOffset  =  nPerRegion  *  region;
  for  (auto  [idx]  :
  ap::onAcc::makeIdxMap(acc,  ap::onAcc::worker::threadsInGrid,
  ap::IdxRange{regionOffset,  regionOffset  +  nPerRegion}))  {
  a[idx]  =  idx;
  }
  }
};

auto  main()  ->  int  {
  unsigned  n  =  5;
  unsigned  nx  =  20;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Non-blocking device queue (requires synchronization)
  using  QueueType  =
  ap::onHost::Queue<ALPAKA_TYPEOF(devAcc),  ap::queueKind::NonBlocking>;
  std::vector<QueueType>  queues;
  for  (unsigned  region  =  0;  region  <  n;  region++)  {
  queues.emplace_back(devAcc.makeQueue(ap::queueKind::nonBlocking));
  }

  // Allocate unified memory that is accessible on host and device
  auto  a  =  ap::onHost::allocUnified<int>(devAcc,  nx);

  unsigned  frameExtent  =  32u;
  auto  frameSpec  =
  ap::onHost::FrameSpec{ap::divExZero(nx,  frameExtent),  frameExtent};

  // Run element-wise multiplication on device
  for  (unsigned  region  =  0;  region  <  n;  region++)  {
  queues[region].enqueue(
  frameSpec,  ap::KernelBundle{IdxAssignKernel{},  a,  region,  n});
  }
  // Wait for the device, includes all queues
  ap::onHost::wait(devAcc);

  for  (unsigned  i  =  0;  i  <  nx;  i++)  printf("a[%d] = %d\n",  i,  a[i]);

  return  0;
} 
```

### 归约

```
#include  <cstdio>
#include  <execution>
#include  <numeric>
#include  <vector>

int  main()  {
  unsigned  n  =  10;

  std::vector<int>  a(n);

  std::iota(a.begin(),  a.end(),  0);  // Fill the array

  // Run reduction on the device
  int  sum  =  std::reduce(std::execution::par_unseq,  a.cbegin(),  a.cend(),  0,
  std::plus<int>{});

  // Print results
  printf("sum = %d\n",  sum);

  return  0;
} 
```

```
#include  <Kokkos_Core.hpp>

int  main(int  argc,  char  *argv[])  {

  // Initialize Kokkos
  Kokkos::initialize(argc,  argv);

  {
  unsigned  n  =  10;

  // Initialize sum variable
  int  sum  =  0;

  // Run sum reduction kernel
  Kokkos::parallel_reduce(
  n,  KOKKOS_LAMBDA(const  int  i,  int  &lsum)  {  lsum  +=  i;  },  sum);

  // Kokkos synchronization
  Kokkos::fence();

  // Print results
  printf("sum = %d\n",  sum);
  }

  // Finalize Kokkos
  Kokkos::finalize();
  return  0;
} 
```

```
// We're using OpenCL C++ API here; there is also C API in <CL/cl.h>
#define CL_TARGET_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110
#include  <CL/cl.hpp>

// For larger kernels, we can store source in a separate file
static  const  std::string  kernel_source  =  R"(
 __kernel void reduce(__global int* sum, __local int* local_mem) {

 // Get work group and work item information
 int gsize = get_global_size(0); // global work size
 int gid = get_global_id(0); // global work item index
 int lsize = get_local_size(0); // local work size
 int lid = get_local_id(0); // local work item index

 // Store reduced item into local memory
 local_mem[lid] = gid; // initialize local memory
 barrier(CLK_LOCAL_MEM_FENCE); // synchronize local memory

 // Perform reduction across the local work group
 for (int s = 1; s < lsize; s *= 2) { // loop over local memory with stride doubling each iteration
 if (lid % (2 * s) == 0 && (lid + s) < lsize) {
 local_mem[lid] += local_mem[lid + s];
 }
 barrier(CLK_LOCAL_MEM_FENCE); // synchronize local memory
 }

 if (lid == 0) { // only one work item per work group
 atomic_add(sum, local_mem[0]); // add partial sum to global sum atomically
 }
 }
  )";

int  main(int  argc,  char  *argv[])  {

  // Initialize OpenCL
  cl::Device  device  =  cl::Device::getDefault();
  cl::Context  context(device);
  cl::CommandQueue  queue(context,  device);

  // Compile OpenCL program for found device
  cl::Program  program(context,  kernel_source);
  program.build({device});
  cl::Kernel  kernel_reduce(program,  "reduce");

  {
  // Set problem dimensions
  unsigned  n  =  10;

  // Initialize sum variable
  int  sum  =  0;

  // Create buffer for sum
  cl::Buffer  buffer(context,  CL_MEM_READ_WRITE  |  CL_MEM_COPY_HOST_PTR,
  sizeof(int),  &sum);

  // Pass arguments to device kernel
  kernel_reduce.setArg(0,  buffer);  // pass buffer to device
  kernel_reduce.setArg(1,  sizeof(int),  NULL);  // allocate local memory

  // Enqueue kernel
  queue.enqueueNDRangeKernel(kernel_reduce,  cl::NullRange,  cl::NDRange(n),
  cl::NullRange);

  // Read result
  queue.enqueueReadBuffer(buffer,  CL_TRUE,  0,  sizeof(int),  &sum);

  // Print result
  printf("sum = %d\n",  sum);
  }

  return  0;
} 
```

```
// We use built-in sycl::reduction mechanism in this example.
// The manual implementation of the reduction kernel can be found in
// the "Non-portable kernel models" chapter.

#include  <sycl/sycl.hpp>

int  main()  {
  sycl::queue  q;
  unsigned  n  =  10;

  // Initialize sum
  int  sum  =  0;
  {
  // Create a buffer for sum to get the reduction results
  sycl::buffer<int>  sum_buf{&sum,  1};

  // Submit a SYCL kernel into a queue
  q.submit(&  {
  // Create temporary object describing variables with reduction semantics
  auto  sum_acc  =  sum_buf.get_access<sycl::access_mode::read_write>(cgh);
  // We can use built-in reduction primitive
  auto  sum_reduction  =  sycl::reduction(sum_acc,  sycl::plus<int>());

  // A reference to the reducer is passed to the lambda
  cgh.parallel_for(
  sycl::range<1>{n},  sum_reduction,
  =  {  reducer.combine(idx[0]);  });
  }).wait();
  // The contents of sum_buf are copied back to sum by the destructor of
  // sum_buf
  }
  // Print results
  printf("sum = %d\n",  sum);
} 
```

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

auto  main()  ->  int  {
  unsigned  n  =  10;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue  queue  =  devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate unified memory that is accessible on host and device
  auto  sum  =  ap::onHost::allocUnified<int>(devAcc,  1);

  // Run element-wise multiplication on device
  ap::onHost::reduce(queue,  0,  sum,  std::plus{},  ap::LinearizedIdxGenerator{n});

  // Print results
  printf("sum = %d\n",  sum[0]);

  return  0;
} 
```

## 跨平台可移植生态系统的好处和坏处

### 一般观察

> +   代码重复量最小化。
> +   
> +   同一段代码可以编译成不同供应商的多种架构。
> +   
> +   与 CUDA 相比，学习资源有限（Stack Overflow、课程材料、文档）。

### 基于 Lambda 的内核模型（Kokkos、SYCL）

> +   更高的抽象级别。
> +   
> +   初始移植所需的底层架构知识较少。
> +   
> +   非常好读的源代码（C++ API）。
> +   
> +   这些模型相对较新，尚未非常流行。

### 基于 Functor 的内核模型（alpaka）

> +   非常好的可移植性。
> +   
> +   更高的抽象级别。
> +   
> +   低级 API 始终可用，提供更多控制并允许精细调整。
> +   
> +   适用于主机和内核代码的用户友好 C++ API。
> +   
> +   小型社区和生态系统。

### 分离源内核模型（OpenCL）

> +   非常好的可移植性。
> +   
> +   成熟的生态系统。
> +   
> +   有限的供应商提供的库数量。
> +   
> +   低级 API 提供更多控制并允许精细调整。
> +   
> +   提供 C 和 C++ API（C++ API 支持度较低）。
> +   
> +   低级 API 和分离源内核模型对用户不太友好。

### C++标准并行性（StdPar、PSTL）

> +   非常高的抽象级别。
> +   
> +   容易加速依赖于 STL 算法的代码。
> +   
> +   对硬件的控制非常有限。
> +   
> +   编译器的支持正在改善，但还远未成熟。

重点

+   通用代码组织与非可移植的基于内核的模型类似。

+   只要不使用特定于供应商的功能，相同的代码就可以在任何 GPU 上运行。

## C++ StdPar

在 C++17 中，对标准算法并行执行的支持已经被引入。大多数通过标准`<algorithms>`头文件提供的算法都被赋予了接受[*执行策略*](https://en.cppreference.com/w/cpp/algorithm)参数的重载，这允许程序员请求并行执行标准库函数。虽然主要目标是允许低成本的、高级别的接口在许多 CPU 核心上运行现有的算法，如`std::sort`，但实现允许使用其他硬件，并且`std::for_each`或`std::transform`等函数在编写算法时提供了很大的灵活性。

C++ StdPar，也称为并行 STL 或 PSTL，可以被认为是类似于指令驱动的模型，因为它非常高级，不提供程序员对数据移动的细粒度控制或对硬件特定功能（如共享（局部）内存）的访问。甚至运行的 GPU 也是自动选择的，因为标准 C++没有*设备*的概念（但有一些供应商扩展允许程序员有更多的控制）。然而，对于已经依赖于 C++标准库算法的应用程序，StdPar 可以通过最小的代码修改来获得 CPU 和 GPU 的性能优势。

对于 GPU 编程，三家供应商都提供了他们的 StdPar 实现，可以将代码卸载到 GPU 上：NVIDIA 有`nvc++`，AMD 有实验性的[roc-stdpar](https://github.com/ROCm/roc-stdpar)，而 Intel 通过他们的 oneAPI 编译器提供 StdPar 卸载功能。[AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp/)提供了一个独立的 StdPar 实现，能够针对三家供应商的设备。虽然 StdPar 是 C++标准的一部分，但不同编译器对 StdPar 的支持水平和成熟度差异很大：并非所有编译器都支持所有算法，而且将算法映射到硬件以及管理数据移动的不同启发式方法可能会影响性能。

### StdPar 编译

构建过程很大程度上取决于所使用的编译器：

+   AdaptiveCpp：在调用`acpp`时添加`--acpp-stdpar`标志。

+   Intel oneAPI：在调用`icpx`时添加`-fsycl -fsycl-pstl-offload=gpu`标志。

+   NVIDIA NVC++：在调用`nvc++`时添加`-stdpar`标志（不支持使用普通的`nvcc`）。

### StdPar 编程

在最简单的形式中，使用 C++标准并行性需要包含额外的`<execution>`头文件，并将一个参数添加到支持的标准库函数中。

例如，让我们看看以下按顺序排序向量的代码：

```
#include  <algorithm>
#include  <vector>

void  f(std::vector<int>&  a)  {
  std::sort(a.begin(),  a.end());
} 
```

要使其在 GPU 上运行排序，只需要进行微小的修改：

```
#include  <algorithm>
#include  <vector>
#include  <execution> // To get std::execution

void  f(std::vector<int>&  a)  {
  std::sort(
  std::execution::par_unseq,  // This algorithm can be run in parallel
  a.begin(),  a.end()
  );
} 
```

现在，当使用支持的编译器编译时，代码将在 GPU 上运行排序。

虽然一开始可能看起来非常受限，但许多标准算法，例如`std::transform`、`std::accumulate`、`std::transform_reduce`和`std::for_each`，可以在数组上运行自定义函数，从而允许将任意算法卸载，只要它不违反 GPU 内核的典型限制，例如不抛出任何异常和不进行系统调用。

### StdPar 执行策略

在 C++中，有四种不同的执行策略可供选择：

+   `std::execution::seq`：串行运行算法，不进行并行化。

+   `std::execution::par`：允许并行化算法（类似于使用多个线程），

+   `std::execution::unseq`：允许向量化算法（类似于使用 SIMD），

+   `std::execution::par_unseq`：允许向量化并并行化算法。

`par`和`unseq`之间的主要区别与线程进度和锁有关：使用`unseq`或`par_unseq`要求算法在进程之间不包含互斥锁和其他锁，而`par`没有这个限制。

对于 GPU，最佳选择是`par_unseq`，因为这在操作顺序方面对编译器的要求最低。虽然`par`在某些情况下也受到支持，但最好避免使用，这既是因为编译器支持有限，也是因为算法可能不适合硬件的迹象。

### StdPar 编译

构建过程很大程度上取决于所使用的编译器：

+   AdaptiveCpp：在调用`acpp`时添加`--acpp-stdpar`标志。

+   Intel oneAPI：在调用`icpx`时添加`-fsycl -fsycl-pstl-offload=gpu`标志。

+   NVIDIA NVC++：在调用`nvc++`时添加`-stdpar`标志（不支持使用普通`nvcc`）。

### StdPar 编程

在最简单的情况下，使用 C++标准并行性需要包含一个额外的`<execution>`头文件，并将一个参数添加到受支持的标准库函数中。

例如，让我们看看以下按顺序排序向量的代码：

```
#include  <algorithm>
#include  <vector>

void  f(std::vector<int>&  a)  {
  std::sort(a.begin(),  a.end());
} 
```

要使其在 GPU 上运行排序，只需要进行微小的修改：

```
#include  <algorithm>
#include  <vector>
#include  <execution> // To get std::execution

void  f(std::vector<int>&  a)  {
  std::sort(
  std::execution::par_unseq,  // This algorithm can be run in parallel
  a.begin(),  a.end()
  );
} 
```

现在，当使用支持的编译器编译时，代码将在 GPU 上运行排序。

虽然一开始可能看起来非常受限，但许多标准算法，例如`std::transform`、`std::accumulate`、`std::transform_reduce`和`std::for_each`，可以在数组上运行自定义函数，从而允许将任意算法卸载，只要它不违反 GPU 内核的典型限制，例如不抛出任何异常和不进行系统调用。

### StdPar 执行策略

在 C++中，有四种不同的执行策略可供选择：

+   `std::execution::seq`：串行运行算法，不进行并行化。

+   `std::execution::par`：允许并行化算法（类似于使用多个线程），

+   `std::execution::unseq`: 允许算法向量化（类似于使用 SIMD），

+   `std::execution::par_unseq`：允许对算法进行向量化和平行化。

`par`和`unseq`之间的主要区别与线程进度和锁有关：使用`unseq`或`par_unseq`要求算法在进程之间不包含互斥锁和其他锁，而`par`没有这个限制。

对于 GPU，最佳选择是`par_unseq`，因为它对编译器在操作顺序方面的要求最少。虽然`par`在某些情况下也得到支持，但最好避免使用，这不仅因为编译器支持有限，而且也表明算法可能不适合硬件。

## Kokkos

Kokkos 是一个开源的性能可移植生态系统，用于在大型的异构硬件架构上并行化，其开发主要在桑迪亚国家实验室进行。该项目始于 2011 年，最初是一个并行 C++编程模型，但后来扩展成为一个更广泛的生态系统，包括 Kokkos Core（编程模型）、Kokkos Kernels（数学库）和 Kokkos Tools（调试、分析和调优工具）。通过为 C++标准委员会准备提案，该项目还旨在影响 ISO/C++语言标准，使得最终 Kokkos 的功能将成为语言标准的原生部分。更详细的介绍可以在[这里](https://www.sandia.gov/news/publications/hpc-annual-reports/article/kokkos/)找到。

Kokkos 库为各种不同的并行编程模型提供了一个抽象层，目前包括 CUDA、HIP、SYCL、HPX、OpenMP 和 C++线程。因此，它允许在不同厂商制造的硬件之间实现更好的可移植性，但引入了软件堆栈的额外依赖。例如，当使用 CUDA 时，只需要 CUDA 安装即可，但当使用 Kokkos 与 NVIDIA GPU 一起时，需要 Kokkos 和 CUDA 的安装。Kokkos 并不是并行编程的一个非常受欢迎的选择，因此，与 CUDA 等更成熟的编程模型相比，学习和使用 Kokkos 可能会更加困难，因为 CUDA 有大量的搜索结果和 Stack Overflow 讨论。

### Kokkos 编译

此外，一些跨平台可移植性库的挑战之一是，即使在同一系统上，不同的项目可能需要不同的编译设置组合来满足可移植性库。例如，在 Kokkos 中，一个项目可能希望默认的执行空间是 CUDA 设备，而另一个项目可能需要 CPU。即使项目偏好相同的执行空间，一个项目可能希望统一内存成为默认的内存空间，而另一个项目可能希望使用固定 GPU 内存。在单个系统上维护大量库实例可能会变得很麻烦。

然而，Kokkos 提供了一种简单的方法来同时编译 Kokkos 库和用户项目。这是通过指定 Kokkos 编译设置（见[这里](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Compiling.html)）并在用户 Makefile 中包含 Kokkos Makefile 来实现的。CMake 也受到支持。这样，用户应用程序和 Kokkos 库一起编译。以下是一个使用 CUDA（Volta 架构）作为后端（默认执行空间）和统一内存作为默认内存空间的单文件 Kokkos 项目的示例 Makefile：

```
default:  build

# Set compiler
KOKKOS_PATH  =  $(shell  pwd)/kokkos
CXX  =  hipcc
# CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper

# Variables for the Makefile.kokkos
KOKKOS_DEVICES  =  "HIP"
# KOKKOS_DEVICES = "Cuda"
KOKKOS_ARCH  =  "VEGA90A"
# KOKKOS_ARCH = "Volta70"
KOKKOS_CUDA_OPTIONS  =  "enable_lambda,force_uvm"

# Include Makefile.kokkos
include $(KOKKOS_PATH)/Makefile.kokkos

build:  $(KOKKOS_LINK_DEPENDS) $(KOKKOS_CPP_DEPENDS) hello.cpp
  $(CXX)  $(KOKKOS_CPPFLAGS)  $(KOKKOS_CXXFLAGS)  $(KOKKOS_LDFLAGS)  hello.cpp  $(KOKKOS_LIBS)  -o  hello 
```

要使用上述 Makefile 构建 **hello.cpp** 项目，除了将 Kokkos 项目克隆到当前目录外，无需进行其他步骤。

### Kokkos 编程

当开始使用 Kokkos 编写项目时，第一步是了解 Kokkos 的初始化和终止。Kokkos 必须通过调用 `Kokkos::initialize(int& argc, char* argv[])` 来初始化，并通过调用 `Kokkos::finalize()` 来终止。更多详细信息请参阅[这里](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Initialization.html)。

Kokkos 使用执行空间模型来抽象并行硬件的细节。执行空间实例映射到可用的后端选项，如 CUDA、OpenMP、HIP 或 SYCL。如果程序员在源代码中没有明确选择执行空间，则使用默认的执行空间 `Kokkos::DefaultExecutionSpace`。这是在编译 Kokkos 库时选择的。Kokkos 执行空间模型在[这里](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Machine-Model.html#kokkos-spaces)有更详细的描述。

同样，Kokkos 使用内存空间模型来处理不同类型的内存，如主机内存或设备内存。如果没有明确定义，Kokkos 将使用在 Kokkos 编译期间指定的默认内存空间，如[这里](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Machine-Model.html#kokkos-memory-spaces)所述。

以下是一个初始化 Kokkos 并打印执行空间和内存空间实例的 Kokkos 程序示例：

```
#include  <Kokkos_Core.hpp>
#include  <iostream>

int  main(int  argc,  char*  argv[])  {
  Kokkos::initialize(argc,  argv);
  std::cout  <<  "Execution Space: "  <<
  typeid(Kokkos::DefaultExecutionSpace).name()  <<  std::endl;
  std::cout  <<  "Memory Space: "  <<
  typeid(Kokkos::DefaultExecutionSpace::memory_space).name()  <<  std::endl;
  Kokkos::finalize();
  return  0;
} 
```

使用 Kokkos，数据可以通过原始指针或通过 Kokkos 视图来访问。使用原始指针，可以将内存分配到默认内存空间，使用 `Kokkos::kokkos_malloc(n * sizeof(int))`。Kokkos 视图是一种数据类型，提供了一种更有效地访问对应于特定 Kokkos 内存空间（如主机内存或设备内存）中的数据的方法。可以通过 `Kokkos::View<int*> a("a", n)` 创建一个类型为 int* 的一维视图，其中 `"a"` 是标签，`n` 是以整数数量表示的分配大小。Kokkos 在编译时确定数据的最佳布局，以实现最佳的整体性能，这取决于计算机架构。此外，Kokkos 会自动处理此类内存的释放。有关 Kokkos 视图的更多详细信息，请参阅[这里](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/View.html)。

最后，Kokkos 提供了三种不同的并行操作：`parallel_for`、`parallel_reduce` 和 `parallel_scan`。`parallel_for` 操作用于并行执行循环。`parallel_reduce` 操作用于并行执行循环并将结果归约到单个值。`parallel_scan` 操作实现前缀扫描。`parallel_for` 和 `parallel_reduce` 的用法将在本章后面的示例中演示。有关并行操作的更多详细信息，请参阅[这里](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/ParallelDispatch.html)。

### 简单步骤运行 Kokkos hello.cpp 示例

以下内容在 AMD VEGA90A 设备上直接使用应能正常工作（需要 ROCm 安装）。在 NVIDIA Volta V100 设备上（需要 CUDA 安装），请在 Makefile 中使用注释掉的变量。

1.  `git clone https://github.com/kokkos/kokkos.git`

1.  将上面的 Makefile 复制到当前文件夹中（确保最后一行的缩进是制表符，而不是空格）

1.  将上面的 hello.cpp 文件复制到当前文件夹

1.  `make`

1.  `./hello`

### Kokkos 编译

此外，一些跨平台可移植库的挑战之一是，即使在同一系统上，不同的项目可能需要不同的编译设置组合来满足库的可移植性。例如，在 Kokkos 中，一个项目可能希望默认的执行空间是 CUDA 设备，而另一个则可能需要 CPU。即使项目偏好相同的执行空间，一个项目可能希望统一内存成为默认的内存空间，而另一个则可能希望使用固定 GPU 内存。在单个系统上维护大量库实例可能会变得很麻烦。

然而，Kokkos 提供了一种简单的方法来同时编译 Kokkos 库和用户项目。这是通过指定 Kokkos 编译设置（见[这里](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Compiling.html)）并在用户 Makefile 中包含 Kokkos Makefile 来实现的。CMake 也受到支持。这样，用户应用程序和 Kokkos 库将一起编译。以下是一个使用 CUDA（Volta 架构）作为后端（默认执行空间）和统一内存作为默认内存空间的单文件 Kokkos 项目的示例 Makefile：

```
default:  build

# Set compiler
KOKKOS_PATH  =  $(shell  pwd)/kokkos
CXX  =  hipcc
# CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper

# Variables for the Makefile.kokkos
KOKKOS_DEVICES  =  "HIP"
# KOKKOS_DEVICES = "Cuda"
KOKKOS_ARCH  =  "VEGA90A"
# KOKKOS_ARCH = "Volta70"
KOKKOS_CUDA_OPTIONS  =  "enable_lambda,force_uvm"

# Include Makefile.kokkos
include $(KOKKOS_PATH)/Makefile.kokkos

build:  $(KOKKOS_LINK_DEPENDS) $(KOKKOS_CPP_DEPENDS) hello.cpp
  $(CXX)  $(KOKKOS_CPPFLAGS)  $(KOKKOS_CXXFLAGS)  $(KOKKOS_LDFLAGS)  hello.cpp  $(KOKKOS_LIBS)  -o  hello 
```

要使用上述 Makefile 构建 **hello.cpp** 项目，除了将 Kokkos 项目克隆到当前目录外，无需其他步骤。

### Kokkos 编程

当开始使用 Kokkos 编写项目时，第一步是理解 Kokkos 的初始化和终止。Kokkos 必须通过调用 `Kokkos::initialize(int& argc, char* argv[])` 来初始化，并通过调用 `Kokkos::finalize()` 来终止。更多详细信息请见[这里](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Initialization.html)。

Kokkos 使用执行空间模型来抽象并行硬件的细节。执行空间实例映射到可用的后端选项，如 CUDA、OpenMP、HIP 或 SYCL。如果程序员在源代码中没有明确选择执行空间，则使用默认执行空间 `Kokkos::DefaultExecutionSpace`。这是在编译 Kokkos 库时选择的。Kokkos 执行空间模型在[这里](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Machine-Model.html#kokkos-spaces)有更详细的描述。

类似地，Kokkos 使用内存空间模型来处理不同类型的内存，例如主机内存或设备内存。如果没有明确定义，Kokkos 将使用在 Kokkos 编译期间指定的默认内存空间，具体描述请见[这里](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Machine-Model.html#kokkos-memory-spaces)。

以下是一个初始化 Kokkos 并打印执行空间和内存空间实例的 Kokkos 程序示例：

```
#include  <Kokkos_Core.hpp>
#include  <iostream>

int  main(int  argc,  char*  argv[])  {
  Kokkos::initialize(argc,  argv);
  std::cout  <<  "Execution Space: "  <<
  typeid(Kokkos::DefaultExecutionSpace).name()  <<  std::endl;
  std::cout  <<  "Memory Space: "  <<
  typeid(Kokkos::DefaultExecutionSpace::memory_space).name()  <<  std::endl;
  Kokkos::finalize();
  return  0;
} 
```

使用 Kokkos，数据可以通过原始指针或通过 Kokkos Views 访问。使用原始指针，可以将内存分配到默认内存空间，使用 `Kokkos::kokkos_malloc(n * sizeof(int))`。Kokkos Views 是一种数据类型，提供了一种更有效地访问与特定 Kokkos 内存空间（如主机内存或设备内存）对应的数据的方法。可以通过 `Kokkos::View<int*> a("a", n)` 创建一个类型为 int* 的一维视图，其中 `"a"` 是标签，`n` 是以整数数量表示的分配大小。Kokkos 在编译时确定数据的最佳布局，以获得最佳的整体性能，这取决于计算机架构。此外，Kokkos 会自动处理此类内存的释放。有关 Kokkos Views 的更多详细信息，请参阅[这里](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/View.html)。

最后，Kokkos 提供了三种不同的并行操作：`parallel_for`、`parallel_reduce` 和 `parallel_scan`。`parallel_for` 操作用于并行执行循环。`parallel_reduce` 操作用于并行执行循环并将结果归约到单个值。`parallel_scan` 操作实现了前缀扫描。`parallel_for` 和 `parallel_reduce` 的用法将在本章后面的示例中演示。有关并行操作的更多详细信息，请参阅[这里](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/ParallelDispatch.html)。

### 按简单步骤运行 Kokkos hello.cpp 示例

以下内容在 AMD VEGA90A 设备上直接使用即可（需要 ROCm 安装）。在 NVIDIA Volta V100 设备上（需要 CUDA 安装），请使用 Makefile 中的注释变量。

1.  `git clone https://github.com/kokkos/kokkos.git`

1.  将上面的 Makefile 复制到当前文件夹中（确保最后一行的缩进是制表符，而不是空格）

1.  将上面的 hello.cpp 文件复制到当前文件夹

1.  `make`

1.  `./hello`

## OpenCL

OpenCL 是一个跨平台的开放标准 API，用于编写在由 CPU、GPU、FPGA 和其他设备组成的异构平台上执行的并行程序。OpenCL 的第一个版本（1.0）于 2008 年 12 月发布，最新版本 OpenCL（3.0）于 2020 年 9 月发布。OpenCL 获得了包括 AMD、ARM、Intel、NVIDIA 和 Qualcomm 在内的多个厂商的支持。它是一个免版税的标准，OpenCL 规范由 Khronos Group 维护。OpenCL 提供了一个基于 C 的低级编程接口，但最近也提供了一种 C++ 接口。

### OpenCL 编译

OpenCL 支持两种编译程序的模式：在线和离线。在线编译发生在运行时，当主机程序调用一个函数来编译源代码时。在线模式允许动态生成和加载内核，但可能会因为编译时间和可能的错误而带来一些开销。离线编译发生在运行之前，当内核的源代码被编译成主机程序可以加载的二进制格式。这种模式允许内核更快地执行和更好的优化，但可能会限制程序的移植性，因为二进制只能在编译时指定的架构上运行。

OpenCL 附带了一些并行编程生态系统，例如 NVIDIA CUDA 和 Intel oneAPI。例如，在成功安装此类包并设置环境后，可以通过如`icx cl_devices.c -lOpenCL`（Intel oneAPI）或`nvcc cl_devices.c -lOpenCL`（NVIDIA CUDA）等命令简单地编译 OpenCL 程序，其中`cl_devices.c`是编译后的文件。与大多数其他编程模型不同，OpenCL 将内核存储为文本，并在运行时（即时编译）为设备编译，因此不需要任何特殊的编译器支持：只要将所需的库和头文件安装到标准位置，就可以使用`gcc cl_devices.c -lOpenCL`（或使用 C++ API 时使用`g++`）编译代码。

安装在 LUMI 上的 AMD 编译器支持 OpenCL C 和 C++ API，后者有一些限制。要编译程序，您可以在 GPU 分区上使用 AMD 编译器：

```
$ module  load  LUMI/24.03  partition/G
$ module  load  rocm/6.0.3
$ module  load  PrgEnv-cray-amd
$ CC  program.cpp  -lOpenCL  -o  program  # C++ program
$ cc  program.c  -lOpenCL  -o  program  # C program 
```

### OpenCL 编程

OpenCL 程序由两部分组成：一部分是运行在主机设备（通常为 CPU）上的主机程序，另一部分是运行在计算设备（如 GPU）上的一或多个内核。主机程序负责管理所选平台上的设备、分配内存对象、构建和排队内核以及管理内存对象等任务。

编写 OpenCL 程序的第一步是初始化 OpenCL 环境，通过选择平台和设备，创建与所选设备关联的上下文或上下文，并为每个设备创建一个命令队列。以下是一个选择默认设备、创建与设备关联的上下文和队列的简单示例。

```
// Initialize OpenCL
cl::Device  device  =  cl::Device::getDefault();
cl::Context  context(device);
cl::CommandQueue  queue(context,  device); 
```

```
// Initialize OpenCL
cl_int  err;  // Error code returned by API calls
cl_platform_id  platform;
err  =  clGetPlatformIDs(1,  &platform,  NULL);
assert(err  ==  CL_SUCCESS);  // Checking error codes is skipped later for brevity
cl_device_id  device;
err  =  clGetDeviceIDs(platform,  CL_DEVICE_TYPE_GPU,  1,  &device,  NULL);
cl_context  context  =  clCreateContext(NULL,  1,  &device,  NULL,  NULL,  &err);
cl_command_queue  queue  =  clCreateCommandQueue(context,  device,  0,  &err); 
```

OpenCL 提供了两种主要的编程模型来管理主机和加速设备内存层次结构：缓冲区和共享虚拟内存（SVM）。缓冲区是 OpenCL 的传统内存模型，其中主机和设备拥有独立的地址空间，程序员必须显式指定内存分配以及如何以及在哪里访问内存。这可以通过`cl::Buffer`类和如`cl::CommandQueue::enqueueReadBuffer()`这样的函数来实现。缓冲区自 OpenCL 的早期版本起就得到了支持，并且在不同架构上都能良好工作。缓冲区还可以利用设备特定的内存特性，例如常量或局部内存。

SVM 是 OpenCL 的一个较新的内存模型，自 2.0 版本引入，其中主机和设备共享一个单一的虚拟地址空间。因此，程序员可以使用相同的指针从主机和设备访问数据，从而简化编程工作。在 OpenCL 中，SVM 有不同级别，如粗粒度缓冲区 SVM、细粒度缓冲区 SVM 和细粒度系统 SVM。所有级别都允许在主机和设备之间使用相同的指针，但它们在内存区域的粒度和同步要求上有所不同。此外，SVM 的支持并不是所有 OpenCL 平台和设备都通用的，例如，NVIDIA V100 和 A100 这样的 GPU 只支持粗粒度 SVM 缓冲区。这一级别需要显式同步从主机和设备对内存的访问（使用如`cl::CommandQueue::enqueueMapSVM()`和`cl::CommandQueue::enqueueUnmapSVM()`这样的函数），这使得 SVM 的使用不太方便。值得注意的是，这与 CUDA 提供的常规统一内存不同，CUDA 更接近于 OpenCL 中的细粒度系统 SVM 级别。

OpenCL 使用一个独立的内核源模型，其中内核代码通常保存在单独的文件中，这些文件可能在运行时进行编译。该模型允许内核源代码作为字符串传递给 OpenCL 驱动程序，之后程序对象可以在特定设备上执行。尽管被称为独立的内核源模型，但内核也可以在主机程序编译单元中以字符串的形式定义，这在某些情况下可能是一个更方便的方法。

与需要离线编译内核到特定设备二进制的二进制模型相比，使用独立内核源模型的在线编译有几个优点。在线编译保留了 OpenCL 的可移植性和灵活性，因为相同的内核源代码可以在任何支持的设备上运行。此外，基于运行时信息（如输入大小、工作组大小或设备能力）的内核动态优化也是可能的。下面是一个 OpenCL 内核的示例，该内核在主机编译单元中以字符串形式定义，并将全局线程索引赋值到全局设备内存中。

```
static  const  std::string  kernel_source  =  R"(
 __kernel void dot(__global int *a) {
 int i = get_global_id(0);
 a[i] = i;
 }
)"; 
```

上面的名为`dot`并存储在字符串`kernel_source`中的内核可以设置为在主机代码中构建，如下所示：

```
cl::Program  program(context,  kernel_source);
program.build({device});
cl::Kernel  kernel_dot(program,  "dot"); 
```

```
cl_int  err;
cl_program  program  =  clCreateProgramWithSource(context,  1,  &kernel_source,  NULL,  &err);
err  =  clBuildProgram(program,  1,  &device,  NULL,  NULL,  NULL);
cl_kernel  kernel_dot  =  clCreateKernel(program,  "vector_add",  &err); 
```

### OpenCL 编译

OpenCL 支持两种编译程序的模式：在线和离线。在线编译发生在运行时，当主机程序调用一个函数来编译源代码时。在线模式允许动态生成和加载内核，但可能因为编译时间和可能的错误而带来一些开销。离线编译发生在运行之前，当内核的源代码被编译成主机程序可以加载的二进制格式。这种模式允许内核更快地执行和更好的优化，但可能会限制程序的移植性，因为二进制只能在编译时指定的架构上运行。

OpenCL 附带了一些并行编程生态系统，例如 NVIDIA CUDA 和 Intel oneAPI。例如，在成功安装这些包并设置环境后，可以通过如`icx cl_devices.c -lOpenCL`（Intel oneAPI）或`nvcc cl_devices.c -lOpenCL`（NVIDIA CUDA）这样的命令简单地编译一个 OpenCL 程序，其中`cl_devices.c`是编译后的文件。与大多数其他编程模型不同，OpenCL 将内核存储为文本，并在运行时（即时编译）为设备编译，因此不需要任何特殊的编译器支持：只要所需的库和头文件安装在标准位置，就可以使用`gcc cl_devices.c -lOpenCL`（或使用 C++ API 时使用`g++`）来编译代码。

安装在 LUMI 上的 AMD 编译器支持 OpenCL C 和 C++ API，后者有一些限制。要编译程序，您可以使用 GPU 分区上的 AMD 编译器：

```
$ module  load  LUMI/24.03  partition/G
$ module  load  rocm/6.0.3
$ module  load  PrgEnv-cray-amd
$ CC  program.cpp  -lOpenCL  -o  program  # C++ program
$ cc  program.c  -lOpenCL  -o  program  # C program 
```

### OpenCL 编程

OpenCL 程序由两部分组成：在主机设备（通常是 CPU）上运行的宿主程序以及在一个或多个计算设备（如 GPU）上运行的内核。宿主程序负责管理所选平台上的设备、分配内存对象、构建和排队内核以及管理内存对象。

编写 OpenCL 程序的第一步是初始化 OpenCL 环境，通过选择平台和设备，创建与所选设备关联的上下文或上下文，并为每个设备创建一个命令队列。以下是一个选择默认设备、创建与设备关联的上下文和队列的简单示例。

```
// Initialize OpenCL
cl::Device  device  =  cl::Device::getDefault();
cl::Context  context(device);
cl::CommandQueue  queue(context,  device); 
```

```
// Initialize OpenCL
cl_int  err;  // Error code returned by API calls
cl_platform_id  platform;
err  =  clGetPlatformIDs(1,  &platform,  NULL);
assert(err  ==  CL_SUCCESS);  // Checking error codes is skipped later for brevity
cl_device_id  device;
err  =  clGetDeviceIDs(platform,  CL_DEVICE_TYPE_GPU,  1,  &device,  NULL);
cl_context  context  =  clCreateContext(NULL,  1,  &device,  NULL,  NULL,  &err);
cl_command_queue  queue  =  clCreateCommandQueue(context,  device,  0,  &err); 
```

OpenCL 提供了两种主要的编程模型来管理主机和加速设备内存层次结构：缓冲区和共享虚拟内存（SVM）。缓冲区是 OpenCL 的传统内存模型，其中主机和设备拥有独立的地址空间，程序员必须显式指定内存分配以及如何以及在哪里访问内存。这可以通过`cl::Buffer`类和如`cl::CommandQueue::enqueueReadBuffer()`这样的函数来实现。缓冲区自 OpenCL 的早期版本起就得到了支持，并且在不同架构上都能良好工作。缓冲区还可以利用设备特定的内存特性，例如常量或局部内存。

SVM 是 OpenCL 的一个较新的内存模型，自 2.0 版本引入，其中主机和设备共享一个单一的虚拟地址空间。因此，程序员可以使用相同的指针从主机和设备访问数据，从而简化编程工作。在 OpenCL 中，SVM 有不同级别，如粗粒度缓冲区 SVM、细粒度缓冲区 SVM 和细粒度系统 SVM。所有级别都允许在主机和设备之间使用相同的指针，但它们在内存区域的粒度和同步要求上有所不同。此外，SVM 的支持并不是所有 OpenCL 平台和设备都通用的，例如，NVIDIA V100 和 A100 这样的 GPU 只支持粗粒度 SVM 缓冲区。这一级别需要显式同步从主机和设备对内存的访问（使用如`cl::CommandQueue::enqueueMapSVM()`和`cl::CommandQueue::enqueueUnmapSVM()`这样的函数），这使得 SVM 的使用不太方便。值得注意的是，这与 CUDA 提供的常规统一内存不同，CUDA 更接近于 OpenCL 中的细粒度系统 SVM 级别。

OpenCL 使用一个独立的内核源模型，其中内核代码通常保存在单独的文件中，这些文件可能在运行时进行编译。该模型允许内核源代码作为字符串传递给 OpenCL 驱动程序，之后程序对象可以在特定设备上执行。尽管被称为独立的内核源模型，但内核也可以在主机程序编译单元中以字符串的形式定义，这在某些情况下可能是一个更方便的方法。

与需要离线编译内核到特定设备二进制的二进制模型相比，使用独立内核源模型的在线编译有几个优点。在线编译保留了 OpenCL 的可移植性和灵活性，因为相同的内核源代码可以在任何支持的设备上运行。此外，基于运行时信息（如输入大小、工作组大小或设备能力）的内核动态优化也是可能的。下面是一个 OpenCL 内核的示例，该内核在主机编译单元中以字符串形式定义，并将全局线程索引赋值到全局设备内存中。

```
static  const  std::string  kernel_source  =  R"(
 __kernel void dot(__global int *a) {
 int i = get_global_id(0);
 a[i] = i;
 }
)"; 
```

上面的名为 `dot` 的内核存储在字符串 `kernel_source` 中，可以设置为在主机代码中构建，如下所示：

```
cl::Program  program(context,  kernel_source);
program.build({device});
cl::Kernel  kernel_dot(program,  "dot"); 
```

```
cl_int  err;
cl_program  program  =  clCreateProgramWithSource(context,  1,  &kernel_source,  NULL,  &err);
err  =  clBuildProgram(program,  1,  &device,  NULL,  NULL,  NULL);
cl_kernel  kernel_dot  =  clCreateKernel(program,  "vector_add",  &err); 
```

## SYCL

[SYCL](https://www.khronos.org/sycl/) 是一个免版税、开放标准的 C++ 编程模型，用于多设备编程。它为异构系统提供了一种高级、单源编程模型，包括 GPU。该标准有几个实现。对于 GPU 编程，[Intel oneAPI DPC++](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html) 和 [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp/)（也称为 hipSYCL）是桌面和 HPC GPU 中最受欢迎的；[ComputeCPP](https://developer.codeplay.com/products/computecpp/ce/home/) 是嵌入式设备的良好选择。相同的标准兼容 SYCL 代码应该与任何实现兼容，但它们不是二进制兼容的。

SYCL 标准的最新版本是 SYCL 2020，这是我们将在本课程中使用的版本。

### SYCL 编译

#### Intel oneAPI DPC++

对于 Intel GPU，只需安装 [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html)。然后，编译就像 `icpx -fsycl file.cpp` 那么简单。

还可以使用 oneAPI 来支持 NVIDIA 和 AMD GPU。除了 oneAPI Base Toolkit 之外，还需要安装供应商提供的运行时（CUDA 或 HIP）以及相应的 [Codeplay oneAPI 插件](https://codeplay.com/solutions/oneapi/)。然后，可以使用包含在 oneAPI 中的 Intel LLVM 编译器编译代码：

+   `clang++ -fsycl -fsycl-targets=nvidia_gpu_sm_86 file.cpp` 用于针对 CUDA 8.6 NVIDIA GPU，

+   `clang++ -fsycl -fsycl-targets=amd_gpu_gfx90a` 用于针对 GFX90a AMD GPU。

#### AdaptiveCpp

使用 AdaptiveCpp 为 NVIDIA 或 AMD GPU 编程还需要首先安装 CUDA 或 HIP。然后可以使用 `acpp` 编译代码，指定目标设备。例如，以下是如何编译支持 AMD 和 NVIDIA 设备的程序：

+   `acpp --acpp-targets='hip:gfx90a;cuda:sm_70' file.cpp`

#### 在 LUMI 上使用 SYCL

LUMI 没有安装任何全局的 SYCL 框架，但在 CSC 模块中有一个最近的 AdaptiveCpp 安装：

```
$ module  load  LUMI/24.03  partition/G
$ module  load  rocm/6.0.3
$ module  use  /appl/local/csc/modulefiles
$ module  load  acpp/24.06.0 
```

默认的编译目标是预置为 MI250 GPU，因此要编译单个 C++ 文件，只需调用 `acpp -O2 file.cpp` 即可。

当运行使用 AdaptiveCpp 构建的程序时，经常会看到警告“dag_direct_scheduler: Detected a requirement that is neither of discard access mode”，这反映了在使用缓冲区访问模型时缺少优化提示。警告是无害的，可以忽略。

### SYCL 编程

在许多方面，SYCL 与 OpenCL 类似，但像 Kokkos 一样使用单个源模型和内核 lambda。

要将任务提交到设备，首先必须创建一个 sycl::queue，它用作管理任务调度和执行的方式。在最简单的情况下，这就是所需的全部初始化：

```
int  main()  {
  // Create an out-of-order queue on the default device:
  sycl::queue  q;
  // Now we can submit tasks to q!
} 
```

如果需要更多控制，可以显式指定设备，或者向队列传递额外的属性：

```
// Iterate over all available devices
for  (const  auto  &device  :  sycl::device::get_devices())  {
  // Print the device name
  std::cout  <<  "Creating a queue on "  <<  device.get_info<sycl::info::device::name>()  <<  "\n";
  // Create an in-order queue for the current device
  sycl::queue  q(device,  {sycl::property::queue::in_order()});
  // Now we can submit tasks to q!
} 
```

内存管理可以通过两种不同的方式完成：*缓冲区访问器*模型和*统一共享内存*（USM）。内存管理模型的选择也会影响 GPU 任务的同步方式。

在*缓冲区访问器*模型中，使用`sycl::buffer`对象来表示数据数组。缓冲区没有映射到任何单一内存空间，并且可以在 GPU 和 CPU 内存之间透明迁移。`sycl::buffer`中的数据不能直接读取或写入，必须创建访问器。`sycl::accessor`对象指定数据访问的位置（主机或某个 GPU 内核）以及访问模式（只读、只写、读写）。这种方法允许通过构建数据依赖的有向无环图（DAG）来优化任务调度：如果内核*A*创建了一个对缓冲区的只写访问器，然后内核*B*提交了一个对同一缓冲区的只读访问器，然后请求主机端的只读访问器，那么可以推断出*A*必须在*B*启动之前完成，并且结果必须在主机任务可以继续之前复制到主机，但主机任务可以与内核*B*并行运行。由于任务之间的依赖关系可以自动构建，因此默认情况下 SYCL 使用*乱序队列*：当两个任务提交到同一个`sycl::queue`时，不能保证第二个任务只有在第一个任务完成后才会启动。在启动内核时，必须创建访问器：

```
// Create a buffer of n integers
auto  buf  =  sycl::buffer<int>(sycl::range<1>(n));
// Submit a kernel into a queue; cgh is a helper object
q.submit(&  {
  // Create write-only accessor for buf
  auto  acc  =  buf.get_access<sycl::access_mode::write>(cgh);
  // Define a kernel: n threads execute the following lambda
  cgh.parallel_for<class  KernelName>(sycl::range<1>{n},  =  {
  // The data is written to the buffer via acc
  acc[i]  =  /*...*/
  });
});
/* If we now submit another kernel with accessor to buf, it will not
 * start running until the kernel above is done */ 
```

缓冲区访问器模型简化了异构编程的许多方面，并防止了许多与同步相关的错误，但它只允许对数据移动和内核执行进行非常粗略的控制。

*USM*模型类似于 NVIDIA CUDA 或 AMD HIP 管理内存的方式。程序员必须显式地在设备上（`sycl::malloc_device`）、在主机上（`sycl::malloc_host`）或在共享内存空间中（`sycl::malloc_shared`）分配内存。尽管名为统一共享内存，并且与 OpenCL 的 SVM 相似，但并非所有 USM 分配都是共享的：例如，由`sycl::malloc_device`分配的内存不能从主机访问。分配函数返回可以直接使用的内存指针，无需访问器。这意味着程序员必须确保主机和设备任务之间的正确同步，以避免数据竞争。使用 USM 时，通常更方便使用*顺序队列*而不是默认的*乱序队列*。有关 USM 的更多信息，请参阅[SYCL 2020 规范的第 4.8 节](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:usm)。

```
// Create a shared (migratable) allocation of n integers
// Unlike with buffers, we need to specify a queue (or, explicitly, a device and a context)
int*  v  =  sycl::malloc_shared<int>(n,  q);
// Submit a kernel into a queue; cgh is a helper object
q.submit(&  {
  // Define a kernel: n threads execute the following lambda
  cgh.parallel_for<class  KernelName>(sycl::range<1>{n},  =  {
  // The data is directly written to v
  v[i]  =  /*...*/
  });
});
// If we want to access v, we have to ensure that the kernel has finished
q.wait();
// After we're done, the memory must be deallocated
sycl::free(v,  q); 
```

### 练习

练习：在 SYCL 中实现 SAXPY

在这个练习中，我们希望编写（填空）一个简单的代码，执行 SAXPY（向量加法）。

要交互式地编译和运行代码，首先进行分配并加载 AdaptiveCpp 模块：

```
$ salloc  -A  project_465002387  -N  1  -t  1:00:00  -p  standard-g  --gpus-per-node=1
....
salloc: Granted job allocation 123456

$ module  load  LUMI/24.03  partition/G
$ module  use  /appl/local/csc/modulefiles
$ module  load  rocm/6.0.3  acpp/24.06.0 
```

现在你可以运行一个简单的设备检测实用程序来检查是否有 GPU 可用（注意`srun`）：

> ```
> $ srun  acpp-info  -l
> =================Backend information===================
> Loaded backend 0: HIP
>  Found device: AMD Instinct MI250X
> Loaded backend 1: OpenMP
>  Found device: hipSYCL OpenMP host device 
> ```

如果还没有这样做，请使用`git clone https://github.com/ENCCS/gpu-programming.git`克隆仓库或使用`git pull origin main`更新它。

现在，让我们看看`content/examples/portable-kernel-models/exercise-sycl-saxpy.cpp`中的示例代码：

```
#include  <iostream>
#include  <sycl/sycl.hpp>
#include  <vector>

int  main()  {
  // Create an in-order queue
  sycl::queue  q{sycl::property::queue::in_order()};
  // Print the device name, just for fun
  std::cout  <<  "Running on "
  <<  q.get_device().get_info<sycl::info::device::name>()  <<  std::endl;
  const  int  n  =  1024;  // Vector size

  // Allocate device and host memory for the first input vector
  float  *d_x  =  sycl::malloc_device<float>(n,  q);
  float  *h_x  =  sycl::malloc_host<float>(n,  q);
 // Bonus question: Can we use `std::vector` here instead of `malloc_host`? // TODO: Allocate second input vector on device and host, d_y and h_y  // Allocate device and host memory for the output vector
  float  *d_z  =  sycl::malloc_device<float>(n,  q);
  float  *h_z  =  sycl::malloc_host<float>(n,  q);

  // Initialize values on host
  for  (int  i  =  0;  i  <  n;  i++)  {
  h_x[i]  =  i;
 // TODO: Initialize h_y somehow  }
  const  float  alpha  =  0.42f;

  q.copy<float>(h_x,  d_x,  n);
 // TODO: Copy h_y to d_y // Bonus question: Why don't we need to wait before using the data? 
  // Run the kernel
  q.parallel_for(sycl::range<1>{n},  =  {
 // TODO: Modify the code to compute z[i] = alpha * x[i] + y[i]  d_z[i]  =  alpha  *  d_x[i];
  });

 // TODO: Copy d_z to h_z // TODO: Wait for the copy to complete 
  // Check the results
  bool  ok  =  true;
  for  (int  i  =  0;  i  <  n;  i++)  {
  float  ref  =  alpha  *  h_x[i]  +  h_y[i];  // Reference value
  float  tol  =  1e-5;  // Relative tolerance
  if  (std::abs((h_z[i]  -  ref))  >  tol  *  std::abs(ref))  {
  std::cout  <<  i  <<  " "  <<  h_z[i]  <<  " "  <<  h_x[i]  <<  " "  <<  h_y[i]
  <<  std::endl;
  ok  =  false;
  break;
  }
  }
  if  (ok)
  std::cout  <<  "Results are correct!"  <<  std::endl;
  else
  std::cout  <<  "Results are NOT correct!"  <<  std::endl;

  // Free allocated memory
  sycl::free(d_x,  q);
  sycl::free(h_x,  q);
 // TODO: Free d_y, h_y.  sycl::free(d_y,  q);
  sycl::free(h_y,  q);

  return  0;
} 
```

要编译和运行代码，请使用以下命令：

```
$ acpp  -O3  exercise-sycl-saxpy.cpp  -o  exercise-sycl-saxpy
$ srun  ./exercise-sycl-saxpy
Running on AMD Instinct MI250X
Results are correct! 
```

代码不能直接编译！你的任务是填写由`TODO`注释指示的缺失部分。你还可以通过代码中的“Bonus questions”测试你的理解。

如果你觉得卡住了，可以查看`exercise-sycl-saxpy-solution.cpp`文件。

### SYCL 编译

#### Intel oneAPI DPC++

对于针对 Intel GPU，只需安装[Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html)。然后，编译就像`icpx -fsycl file.cpp`一样简单。

也可以使用 oneAPI 针对 NVIDIA 和 AMD GPU。除了 oneAPI Base Toolkit 外，还需要安装供应商提供的运行时（CUDA 或 HIP）以及相应的[Codeplay oneAPI 插件](https://codeplay.com/solutions/oneapi/)。然后，可以使用包含在 oneAPI 中的 Intel LLVM 编译器编译代码：

+   `clang++ -fsycl -fsycl-targets=nvidia_gpu_sm_86 file.cpp`用于针对 CUDA 8.6 NVIDIA GPU，

+   `clang++ -fsycl -fsycl-targets=amd_gpu_gfx90a`用于针对 GFX90a AMD GPU。

#### AdaptiveCpp

使用 AdaptiveCpp 针对 NVIDIA 或 AMD GPU 也需要首先安装 CUDA 或 HIP。然后可以使用`acpp`编译代码，指定目标设备。例如，以下是编译支持 AMD 和 NVIDIA 设备的程序的方法：

+   `acpp --acpp-targets='hip:gfx90a;cuda:sm_70' file.cpp`

#### 在 LUMI 上使用 SYCL

LUMI 没有系统范围内安装任何 SYCL 框架，但最新的 AdaptiveCpp 安装可在 CSC 模块中找到：

```
$ module  load  LUMI/24.03  partition/G
$ module  load  rocm/6.0.3
$ module  use  /appl/local/csc/modulefiles
$ module  load  acpp/24.06.0 
```

默认编译目标已预设为 MI250 GPU，因此要编译单个 C++文件，只需调用`acpp -O2 file.cpp`即可。

当运行使用 AdaptiveCpp 构建的应用程序时，经常会看到警告“dag_direct_scheduler: 检测到一个既不是丢弃访问模式也不是的要求”，这反映了在使用缓冲区访问模型时缺少优化提示。警告是无害的，可以忽略。

#### Intel oneAPI DPC++

对于针对 Intel GPU，只需安装[Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html)。然后，编译就像`icpx -fsycl file.cpp`一样简单。

也可以使用 oneAPI 针对 NVIDIA 和 AMD GPU。除了 oneAPI Base Toolkit 外，还需要安装供应商提供的运行时（CUDA 或 HIP）以及相应的[Codeplay oneAPI 插件](https://codeplay.com/solutions/oneapi/)。然后，可以使用包含在 oneAPI 中的 Intel LLVM 编译器编译代码：

+   `clang++ -fsycl -fsycl-targets=nvidia_gpu_sm_86 file.cpp`用于针对 CUDA 8.6 NVIDIA GPU，

+   `clang++ -fsycl -fsycl-targets=amd_gpu_gfx90a`用于针对 GFX90a AMD GPU。

#### AdaptiveCpp

在 NVIDIA 或 AMD GPU 上使用 AdaptiveCpp 也需要首先安装 CUDA 或 HIP。然后可以使用`acpp`编译代码，指定目标设备。例如，以下是编译支持 AMD 和 NVIDIA 设备的程序的方法：

+   `acpp --acpp-targets='hip:gfx90a;cuda:sm_70' file.cpp`

#### 在 LUMI 上使用 SYCL

LUMI 没有系统范围内安装任何 SYCL 框架，但 CSC 模块中有最新的 AdaptiveCpp 安装：

```
$ module  load  LUMI/24.03  partition/G
$ module  load  rocm/6.0.3
$ module  use  /appl/local/csc/modulefiles
$ module  load  acpp/24.06.0 
```

默认编译目标预设为 MI250 GPU，因此要编译单个 C++文件，只需要调用`acpp -O2 file.cpp`。

当运行使用 AdaptiveCpp 构建的应用程序时，经常会看到警告“dag_direct_scheduler: Detected a requirement that is neither of discard access mode”，这反映了在使用缓冲区访问器模型时缺少优化提示。警告是无害的，可以忽略。

### SYCL 编程

SYCL 在很多方面与 OpenCL 相似，但像 Kokkos 一样，使用单源模型和内核 lambda。

要将任务提交到设备，首先必须创建一个`sycl::queue`，它被用作管理任务调度和执行的方式。在最简单的情况下，这就是所需的全部初始化：

```
int  main()  {
  // Create an out-of-order queue on the default device:
  sycl::queue  q;
  // Now we can submit tasks to q!
} 
```

如果想要更多控制，可以显式指定设备，或者可以向队列传递额外的属性：

```
// Iterate over all available devices
for  (const  auto  &device  :  sycl::device::get_devices())  {
  // Print the device name
  std::cout  <<  "Creating a queue on "  <<  device.get_info<sycl::info::device::name>()  <<  "\n";
  // Create an in-order queue for the current device
  sycl::queue  q(device,  {sycl::property::queue::in_order()});
  // Now we can submit tasks to q!
} 
```

内存管理可以通过两种不同的方式完成：*缓冲区访问器*模型和*统一共享内存*（USM）。内存管理模型的选择也影响 GPU 任务的同步。

在*缓冲区访问器*模型中，使用`sycl::buffer`对象来表示数据数组。缓冲区没有映射到任何单一内存空间，并且可以在 GPU 和 CPU 内存之间透明迁移。`sycl::buffer`中的数据不能直接读取或写入，必须创建访问器。`sycl::accessor`对象指定数据访问的位置（主机或某个 GPU 内核）和访问模式（只读、只写、读写）。这种方法通过构建数据依赖的有向无环图（DAG）来优化任务调度：如果内核*A*创建了一个对缓冲区的只写访问器，然后内核*B*提交了一个对同一缓冲区的只读访问器，然后请求主机端的只读访问器，那么可以推断出*A*必须在*B*启动之前完成，并且结果必须在主机任务可以继续之前复制到主机，但主机任务可以与内核*B*并行运行。由于任务之间的依赖关系可以自动构建，因此默认情况下 SYCL 使用*乱序队列*：当两个任务提交到同一个`sycl::queue`时，不能保证第二个任务只有在第一个任务完成后才会启动。在启动内核时，必须创建访问器：

```
// Create a buffer of n integers
auto  buf  =  sycl::buffer<int>(sycl::range<1>(n));
// Submit a kernel into a queue; cgh is a helper object
q.submit(&  {
  // Create write-only accessor for buf
  auto  acc  =  buf.get_access<sycl::access_mode::write>(cgh);
  // Define a kernel: n threads execute the following lambda
  cgh.parallel_for<class  KernelName>(sycl::range<1>{n},  =  {
  // The data is written to the buffer via acc
  acc[i]  =  /*...*/
  });
});
/* If we now submit another kernel with accessor to buf, it will not
 * start running until the kernel above is done */ 
```

缓冲区访问模型简化了异构编程的许多方面，并防止了许多与同步相关的错误，但它只允许对数据移动和内核执行进行非常粗略的控制。

*USM* 模型类似于 NVIDIA CUDA 或 AMD HIP 管理内存的方式。程序员必须显式地在设备上（`sycl::malloc_device`）、在主机上（`sycl::malloc_host`）或在共享内存空间中（`sycl::malloc_shared`）分配内存。尽管其名称为统一共享内存，并且与 OpenCL 的 SVM 相似，但并非所有 USM 分配都是共享的：例如，由 `sycl::malloc_device` 分配的内存不能从主机访问。分配函数返回可以直接使用的内存指针，无需访问器。这意味着程序员必须确保主机和设备任务之间的正确同步，以避免数据竞争。使用 USM 时，通常更方便使用 *顺序队列* 而不是默认的 *乱序队列*。有关 USM 的更多信息，请参阅 [SYCL 2020 规范的第 4.8 节](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:usm)。

```
// Create a shared (migratable) allocation of n integers
// Unlike with buffers, we need to specify a queue (or, explicitly, a device and a context)
int*  v  =  sycl::malloc_shared<int>(n,  q);
// Submit a kernel into a queue; cgh is a helper object
q.submit(&  {
  // Define a kernel: n threads execute the following lambda
  cgh.parallel_for<class  KernelName>(sycl::range<1>{n},  =  {
  // The data is directly written to v
  v[i]  =  /*...*/
  });
});
// If we want to access v, we have to ensure that the kernel has finished
q.wait();
// After we're done, the memory must be deallocated
sycl::free(v,  q); 
```

### 练习

练习：实现 SYCL 中的 SAXPY

在这个练习中，我们希望编写（填空）一个简单的代码，执行 SAXPY（向量加法）。

要交互式地编译和运行代码，首先进行分配并加载 AdaptiveCpp 模块：

```
$ salloc  -A  project_465002387  -N  1  -t  1:00:00  -p  standard-g  --gpus-per-node=1
....
salloc: Granted job allocation 123456

$ module  load  LUMI/24.03  partition/G
$ module  use  /appl/local/csc/modulefiles
$ module  load  rocm/6.0.3  acpp/24.06.0 
```

现在你可以运行一个简单的设备检测实用程序来检查 GPU 是否可用（注意`srun`）：

> ```
> $ srun  acpp-info  -l
> =================Backend information===================
> Loaded backend 0: HIP
>  Found device: AMD Instinct MI250X
> Loaded backend 1: OpenMP
>  Found device: hipSYCL OpenMP host device 
> ```

如果你还没有做，请使用 `git clone https://github.com/ENCCS/gpu-programming.git` 或 **使用 `git pull origin main` 更新** 仓库。

现在，让我们看看 `content/examples/portable-kernel-models/exercise-sycl-saxpy.cpp` 中的示例代码：

```
#include  <iostream>
#include  <sycl/sycl.hpp>
#include  <vector>

int  main()  {
  // Create an in-order queue
  sycl::queue  q{sycl::property::queue::in_order()};
  // Print the device name, just for fun
  std::cout  <<  "Running on "
  <<  q.get_device().get_info<sycl::info::device::name>()  <<  std::endl;
  const  int  n  =  1024;  // Vector size

  // Allocate device and host memory for the first input vector
  float  *d_x  =  sycl::malloc_device<float>(n,  q);
  float  *h_x  =  sycl::malloc_host<float>(n,  q);
 // Bonus question: Can we use `std::vector` here instead of `malloc_host`? // TODO: Allocate second input vector on device and host, d_y and h_y  // Allocate device and host memory for the output vector
  float  *d_z  =  sycl::malloc_device<float>(n,  q);
  float  *h_z  =  sycl::malloc_host<float>(n,  q);

  // Initialize values on host
  for  (int  i  =  0;  i  <  n;  i++)  {
  h_x[i]  =  i;
 // TODO: Initialize h_y somehow  }
  const  float  alpha  =  0.42f;

  q.copy<float>(h_x,  d_x,  n);
 // TODO: Copy h_y to d_y // Bonus question: Why don't we need to wait before using the data? 
  // Run the kernel
  q.parallel_for(sycl::range<1>{n},  =  {
 // TODO: Modify the code to compute z[i] = alpha * x[i] + y[i]  d_z[i]  =  alpha  *  d_x[i];
  });

 // TODO: Copy d_z to h_z // TODO: Wait for the copy to complete 
  // Check the results
  bool  ok  =  true;
  for  (int  i  =  0;  i  <  n;  i++)  {
  float  ref  =  alpha  *  h_x[i]  +  h_y[i];  // Reference value
  float  tol  =  1e-5;  // Relative tolerance
  if  (std::abs((h_z[i]  -  ref))  >  tol  *  std::abs(ref))  {
  std::cout  <<  i  <<  " "  <<  h_z[i]  <<  " "  <<  h_x[i]  <<  " "  <<  h_y[i]
  <<  std::endl;
  ok  =  false;
  break;
  }
  }
  if  (ok)
  std::cout  <<  "Results are correct!"  <<  std::endl;
  else
  std::cout  <<  "Results are NOT correct!"  <<  std::endl;

  // Free allocated memory
  sycl::free(d_x,  q);
  sycl::free(h_x,  q);
 // TODO: Free d_y, h_y.  sycl::free(d_y,  q);
  sycl::free(h_y,  q);

  return  0;
} 
```

要编译和运行代码，请使用以下命令：

```
$ acpp  -O3  exercise-sycl-saxpy.cpp  -o  exercise-sycl-saxpy
$ srun  ./exercise-sycl-saxpy
Running on AMD Instinct MI250X
Results are correct! 
```

代码不能直接编译！你的任务是填写由 `TODO` 注释指示的缺失部分。你也可以通过代码中的“附加问题”来测试你的理解。

如果你觉得卡住了，请查看 `exercise-sycl-saxpy-solution.cpp` 文件。

## alpaka

[alpaka](https://github.com/alpaka-group/alpaka3) 库是一个开源的仅头文件 C++20 抽象库，用于加速器开发。

其目的是通过抽象底层并行级别，在加速器之间提供性能可移植性。该项目提供了一个单源 C++ API，使开发者能够编写一次并行代码，并在不同的硬件架构上运行而无需修改。名称“alpaka”来源于**A**bstractions for **L**evels of **P**arallelism、**A**lgorithms 和 **K**ernels for **A**ccelerators。该库是平台无关的，并支持多个设备的并发和协作使用，包括主机 CPU（x86、ARM 和 RISC-V）以及来自不同供应商的 GPU（NVIDIA、AMD 和 Intel）。提供了各种加速器后端，如 CUDA、HIP、SYCL、OpenMP 和串行执行，可以根据目标设备进行选择。只需要一个用户内核的实现，以具有标准化接口的函数对象的形式表达。这消除了编写专门的 CUDA、HIP、SYCL、OpenMP、Intel TBB 或线程代码的需求。此外，可以将多个加速器后端组合起来，以针对单个系统甚至单个应用程序中的不同供应商硬件。

抽象基于一个虚拟索引域，该域被分解成大小相等的块，称为帧。**alpaka** 提供了一个统一的抽象来遍历这些帧，独立于底层硬件。要并行化的算法将分块索引域和本地工作线程映射到数据上，将计算表示为在并行线程（SIMT）中执行的内核，从而也利用了 SIMD 单元。与 CUDA、HIP 和 SYCL 等原生并行模型不同，**alpaka** 内核不受三维的限制。通过共享内存显式缓存帧内的数据允许开发者充分发挥计算设备性能。此外，**alpaka** 还提供了原始函数，如 iota、transform、transform-reduce、reduce 和 concurrent，简化了可移植高性能应用程序的开发。主机、设备、映射和管理多维视图提供了一种自然的数据操作方式。

在这里，我们展示了 **alpaka3** 的用法，这是一个对 [alpaka](https://github.com/alpaka-group/alpaka) 的完整重写。计划在 2026 年第二季度/第三季度首次发布之前，将这个独立的代码库合并回主线 alpaka 仓库。尽管如此，代码经过充分测试，可以用于今天的开发。

### 在您的系统上安装 alpaka

为了便于使用，我们建议按照以下说明使用 CMake 安装 alpaka。有关在项目中使用 alpaka 的其他方法，请参阅 [alpaka3 文档](https://alpaka3.readthedocs.io/en/latest/basic/install.html)。

1.  **克隆仓库**

    从 GitHub 克隆 alpaka 源代码到您选择的目录：

    ```
    git  clone  https://github.com/alpaka-group/alpaka3.git
    cd  alpaka 
    ```

1.  **设置安装目录**

    将 `ALPAKA_DIR` 环境变量设置为想要安装 alpaka 的目录。这可以是您选择的任何目录，只要您有写入权限。

    ```
    export  ALPAKA_DIR=/path/to/your/alpaka/install/dir 
    ```

1.  **构建和安装**

    创建一个构建目录并使用 CMake 构建和安装 alpaka。我们使用 `CMAKE_INSTALL_PREFIX` 来告诉 CMake 将库安装在哪里。

    ```
    mkdir  build
    cmake  -B  build  -S  .  -DCMAKE_INSTALL_PREFIX=$ALPAKA_DIR
    cmake  --build  build  --parallel 
    ```

1.  **更新环境**

    为了确保其他项目可以找到您的 alpaka 安装，您应该将安装目录添加到您的 `CMAKE_PREFIX_PATH` 中。您可以通过将以下行添加到您的 shell 配置文件（例如 `~/.bashrc`）来实现：

    ```
    export  CMAKE_PREFIX_PATH=$ALPAKA_DIR:$CMAKE_PREFIX_PATH 
    ```

    您需要源码您的 shell 配置文件或打开一个新的终端以使更改生效。

### alpaka 编译

我们建议使用 CMake 构建使用 alpaka 的项目。可以采用各种策略来处理为特定设备或设备集构建应用程序。这里我们展示了开始的最小方法，但这绝不是设置项目的唯一方法。请参阅 [alpaka3 文档](https://alpaka3.readthedocs.io/en/latest/basic/install.html)，了解在项目中使用 alpaka 的替代方法，包括在 CMake 中定义设备规范以使源代码与目标加速器无关的方法。

以下示例演示了一个使用 alpaka3 的单文件项目的 `CMakeLists.txt`（以下部分中展示的 `main.cpp`）：

> ```
> cmake_minimum_required(VERSION  3.25)
> project(myAlpakaApp  VERSION  1.0)
> 
> # Find installed alpaka
> find_package(alpaka  REQUIRED)
> 
> # Build the executable
> add_executable(myAlpakaApp  main.cpp)
> target_link_libraries(myAlpakaApp  PRIVATE  alpaka::alpaka)
> alpaka_finalize(myAlpakaApp) 
> ```

#### 在 LUMI 上使用 alpaka

要加载在 LUMI 上使用 HIP 的 AMD GPU 的环境，可以使用以下模块 -

```
$ module  load  LUMI/24.03  partition/G
$ module  load  rocm/6.0.3
$ module  load  buildtools/24.03
$ module  load  PrgEnv-amd
$ module  load  craype-accel-amd-gfx90a
$ export  CXX=hipcc 
```

### alpaka 编程

使用 alpaka3 开始时，第一步是理解 **设备选择模型**。与需要显式初始化调用的框架不同，alpaka3 使用设备规范来确定使用哪个后端和硬件。设备规范由两个组件组成：

+   **API**: 并行编程接口（主机、cuda、hip、oneApi）

+   **设备类型**: 硬件类型（cpu、nvidiaGpu、amdGpu、intelGpu）

在这里，我们指定并使用这些在运行时选择和初始化设备。设备选择过程在 alpaka3 文档中有详细描述。

alpaka3 使用 **执行空间模型** 来抽象并行硬件细节。使用 `alpaka::onHost::makeDeviceSelector(devSpec)` 创建一个设备选择器，它返回一个可以查询可用设备并为所选后端创建设备实例的对象。

以下示例演示了一个基本的 alpaka 程序，该程序初始化一个设备并打印有关该设备的信息：

> ```
> #include  <alpaka/alpaka.hpp>
> #include  <cstdlib>
> #include  <iostream>
> 
> namespace  ap  =  alpaka;
> 
> auto  getDeviceSpec()
> {
>   /* Select a device, possible combinations of api+deviceKind:
>  * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
>  * oneApi+amdGpu, oneApi+nvidiaGpu
>  */
>   return  ap::onHost::DeviceSpec{ap::api::hip,  ap::deviceKind::amdGpu};
> }
> 
> int  main(int  argc,  char**  argv)
> {
>   // Initialize device specification and selector
>   ap::onHost::DeviceSpec  devSpec  =  getDeviceSpec();
>   auto  deviceSelector  =  ap::onHost::makeDeviceSelector(devSpec);
> 
>   // Query available devices
>   auto  num_devices  =  deviceSelector.getDeviceCount();
>   std::cout  <<  "Number of available devices: "  <<  num_devices  <<  "\n";
> 
>   if  (num_devices  ==  0)  {
>   std::cerr  <<  "No devices found for the selected backend\n";
>   return  EXIT_FAILURE;
>   }
> 
>   // Select and initialize the first device
>   auto  device  =  deviceSelector.makeDevice(0);
>   std::cout  <<  "Using device: "  <<  device.getName()  <<  "\n";
> 
>   return  EXIT_SUCCESS;
> } 
> ```

alpaka3 通过缓冲区和视图提供内存管理抽象。可以使用 `alpaka::allocBuf<T, Idx>(device, extent)` 在主机或设备上分配内存。主机和设备之间的数据传输通过 `alpaka::memcpy(queue, dst, src)` 处理。该库自动管理不同架构上的内存布局以实现最佳性能。

对于并行执行，alpaka3 提供了内核抽象。内核被定义为函数式对象或 lambda 函数，并使用定义并行化策略的工作划分规范来执行。该框架支持各种并行模式，包括逐元素操作、归约和扫描。

#### **alpaka** 功能概览

现在我们将快速探索 alpaka 最常用的功能，并简要介绍一些基本用法。常用 alpaka 功能的快速参考可在[此处](https://alpaka3.readthedocs.io/en/latest/basic/cheatsheet.html)找到。

**一般设置**：包含一次综合头文件，您就可以开始使用 alpaka。

```
#include  <alpaka/alpaka.hpp>

namespace  myProject
{
  namespace  ap  =  alpaka;
  // Your code here
} 
```

**加速器、平台和设备管理**：通过将所需的 API 与适当的硬件类型结合使用设备选择器来选择设备。

```
auto  devSelector  =  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
if  (devSelector.getDeviceCount()  ==  0)
{
  throw  std::runtime_error("No device found!");
}
auto  device  =  devSelector.makeDevice(0); 
```

**队列和事件**：为每个设备创建阻塞或非阻塞队列，记录事件，并根据需要同步工作。

```
auto  queue  =  device.makeQueue();
auto  nonBlockingQueue  =  device.makeQueue(ap::queueKind::nonBlocking);
auto  blockingQueue  =  device.makeQueue(ap::queueKind::blocking);

auto  event  =  device.makeEvent();
queue.enqueue(event);
ap::onHost::wait(event);
ap::onHost::wait(queue); 
```

**内存管理**：分配主机、设备、映射、统一或延迟缓冲区，创建非拥有视图，并使用 memcpy、memset 和 fill 可移植地移动数据。

```
auto  hostBuffer  =  ap::onHost::allocHost<DataType>(extent3D);
auto  devBuffer  =  ap::onHost::alloc<DataType>(device,  extentMd);
auto  devMappedBuffer  =  ap::onHost::allocMapped<DataType>(device,  extentMd);

auto  hostView  =  ap::makeView(api::host,  externPtr,  ap::Vec{numElements});
auto  devNonOwningView  =  devBuffer.getView();

ap::onHost::memset(queue,  devBuffer,  uint8_t{0});
ap::onHost::memcpy(queue,  devBuffer,  hostBuffer);
ap::onHost::fill(queue,  devBuffer,  DataType{42}); 
```

**内核执行**：手动构建 FrameSpec 或请求针对您数据类型优化的 FrameSpec，然后使用自动或显式执行器将内核入队。

```
constexpr  uint32_t  dim  =  2u;
using  IdxType  =  size_t;
using  DataType  =  int;

IdxType  valueX,  valueY;
auto  extentMD  =  ap::Vec{valueY,  valueX};

auto  frameSpec  =  ap::onHost::FrameSpec{numFramesMd,  frameExtentMd};
auto  tunedSpec  =  ap::onHost::getFrameSpec<DataType>(device,  extentMd);

queue.enqueue(tunedSpec,  ap::KernelBundle{kernel,  kernelArgs...});

auto  executor  =  ap::exec::cpuSerial;
queue.enqueue(executor,  tunedSpec,  ap::KernelBundle{kernel,  kernelArgs...}); 
```

**内核实现**：将内核编写为带有 ALPAKA_FN_ACC 注解的函数式对象，在内核体内部直接使用共享内存、同步、原子操作和数学助手。

```
struct  MyKernel
{
  ALPAKA_FN_ACC  void  operator()(ap::onAcc::concepts::Acc  auto  const&  acc,  auto...  args)  const
  {
  auto  idxMd  =  acc.getIdxWithin(ap::onAcc::origin::grid,  ap::onAcc::unit::blocks);

  auto  sharedMdArray  =
  ap::onAcc::declareSharedMdArray<float,  ap::uniqueId()>(acc,  ap::CVec<uint32_t,  3,  4>{});

  ap::onAcc::syncBlockThreads(acc);
  auto  old  =  onAcc::atomicAdd(acc,  args...);
  ap::onAcc::memFence(acc,  ap::onAcc::scope::block);
  auto  sinValue  =  ap::math::sin(args[0]);
  }
}; 
```

#### 简单步骤运行 alpaka3 示例

以下示例适用于 CMake 3.25+ 和适当的 C++ 编译器。对于 GPU 执行，请确保已安装相应的运行时（CUDA、ROCm 或 oneAPI）。

1.  为您的项目创建一个目录：

    ```
    mkdir  my_alpaka_project  &&  cd  my_alpaka_project 
    ```

1.  将上面的 CMakeLists.txt 复制到当前文件夹

1.  将 main.cpp 文件复制到当前文件夹

1.  配置和构建：

    ```
    cmake  -B  build  -S  .  -Dalpaka_DEP_HIP=ON
    cmake  --build  build  --parallel 
    ```

1.  运行可执行文件：

    ```
    ./build/myAlpakaApp 
    ```

注意

设备规范系统允许您在 CMake 配置时选择目标设备。格式为 `"api:deviceKind"`，其中：

+   **api**：并行编程接口（`host`、`cuda`、`hip`、`oneApi`）

+   **deviceKind**：设备类型（`cpu`、`nvidiaGpu`、`amdGpu`、`intelGpu`）

可用的组合有：`host:cpu`、`cuda:nvidiaGpu`、`hip:amdGpu`、`oneApi:cpu`、`oneApi:intelGpu`、`oneApi:nvidiaGpu`、`oneApi:amdGpu`

警告

只有当 CUDA SDK、HIP SDK 或 OneAPI SDK 分别可用时，CUDA、HIP 或 Intel 后端才有效。

#### 预期输出

```
Number of available devices: 1
Using device: [Device Name] 
```

设备名称将根据您的硬件而变化（例如，“NVIDIA A100”、“AMD MI250X”或您的 CPU 型号）。

### 编译和执行示例

您可以从示例部分测试 alpaka 提供的示例。这些示例已硬编码了在 LUMI 上所需的 AMD ROCm 平台的用法。要仅使用 CPU，您只需将 `ap::onHost::makeDeviceSelector(ap::api::hip, ap::deviceKind::amdGpu);` 替换为 `ap::onHost::makeDeviceSelector(ap::api::host, ap::deviceKind::cpu);`

以下步骤假设您已经下载了 alpaka，并且 **alapka** 源代码的路径已存储在环境变量 `ALPAKA_DIR` 中。要测试示例，请将代码复制到文件 `main.cpp`

或者，[点击此处](https://godbolt.org/z/69exnG4xb) 尝试在 godbolt 编译器探索器中使用第一个示例。

```
# use the following in C++ code
# auto devSelector = ap::onHost::makeDeviceSelector(ap::api::hip, ap::deviceKind::amdGpu);
# We use CC to refer to the compiler to work smoothly with the LUMI environment
CC  -I  $ALPAKA_DIR/include/  -std=c++20  -x  hip  --offload-arch=gfx90a  main.cpp
./a.out 
```

```
# use the following in C++ code
# auto devSelector = ap::onHost::makeDeviceSelector(ap::api::host, ap::deviceKind::cpu);
# We use CC to refer to the compiler to work smoothly with the LUMI environment
CC  -I  $ALPAKA_DIR/include/  -std=c++20  main.cpp
./a.out 
```

```
# use the following in C++ code
# auto devSelector = ap::onHost::makeDeviceSelector(ap::api::cuda, ap::deviceKind::nvidiaGpu);
nvcc  -I  $ALPAKA_DIR/include/  -std=c++20  --expt-relaxed-constexpr  -x  cuda  main.cpp
./a.out 
```

```
# use the following in C++ code
# auto devSelector = ap::onHost::makeDeviceSelector(ap::api::oneApi, ap::deviceKind::cpu);
icpx  -I  $ALPAKA_DIR/include/  -std=c++20  -fsycl  -fsycl-targets=spir64_x86_64  main.cpp
./a.out 
```

```
# use the following in C++ code
# auto devSelector = ap::onHost::makeDeviceSelector(ap::api::oneApi, ap::deviceKind::intelGpu);
icpx  -I  $ALPAKA_DIR/include/  -std=c++20  -fsycl  -fsycl-targets=spir64  main.cpp
./a.out 
```

注意

要使用 AMD 或 NVIDIA Gpus 的 oneAPI Sycl，您必须安装相应的 Codeplay oneAPI 插件，具体说明请参阅[此处](https://codeplay.com/solutions/oneapi/plugins/)。

```
# use the following in C++ code
# auto devSelector = ap::onHost::makeDeviceSelector(ap::api::oneApi, ap::deviceKind::amdGpu);
icpx  -I  $ALPAKA_DIR/include/  -std=c++20  -fsycl  -fsycl-targets=amd_gpu_gfx90a  main.cpp
./a.out 
```

注意

要使用 AMD 或 NVIDIA Gpus 的 oneAPI Sycl，您必须安装相应的 Codeplay oneAPI 插件，具体说明请参阅[此处](https://codeplay.com/solutions/oneapi/plugins/)。

```
# use the following in C++ code
# auto devSelector = ap::onHost::makeDeviceSelector(ap::api::oneApi, ap::deviceKind::nvidiaGpu);
icpx  -I  $ALPAKA_DIR/include/  -std=c++20  -fsycl  -fsycl-targets=nvptx64-nvidia-cuda  -Xsycl-target-backend=nvptx64-nvidia-cuda  --offload-arch=sm_80  main.cpp
./a.out 
```

### 练习

练习：在 alpaka 中编写向量加法内核

在这个练习中，我们希望编写（填空）一个简单的内核来添加两个向量。

为了交互式地编译和运行代码，首先我们需要在 GPU 节点上获取一个分配并加载 alpaka 的模块：

```
$ srun  -p  dev-g  --gpus  1  -N  1  -n  1  --time=00:20:00  --account=project_465002387  --pty  bash
....
srun: job 1234 queued and waiting for resources
srun: job 1234 has been allocated resources

$ module  load  LUMI/24.03  partition/G
$ module  load  rocm/6.0.3
$ module  load  buildtools/24.03
$ module  load  PrgEnv-amd
$ module  load  craype-accel-amd-gfx90a
$ export  CXX=hipcc 
```

现在您可以运行一个简单的设备检测实用程序来检查是否有 GPU 可用（注意 `srun`）：

```
$ rocm-smi

======================================= ROCm System Management Interface =======================================
================================================= Concise Info =================================================
Device  [Model : Revision]    Temp    Power  Partitions      SCLK    MCLK     Fan  Perf    PwrCap  VRAM%  GPU%
 Name (20 chars)       (Edge)  (Avg)  (Mem, Compute)
================================================================================================================
0       [0x0b0c : 0x00]       45.0°C  N/A    N/A, N/A        800Mhz  1600Mhz  0%   manual  0.0W      0%   0%
 AMD INSTINCT MI200 (
================================================================================================================
============================================= End of ROCm SMI Log ============================================== 
```

现在，让我们看看设置练习的代码：

在下面，我们使用 CMake 的 fetch content 快速开始使用 alpaka。

CMakeLists.txt

```
cmake_minimum_required(VERSION  3.25)
project(vectorAdd  LANGUAGES  CXX  VERSION  1.0)
#Use CMake's FetchContent to download and integrate alpaka3 directly from GitHub
include(FetchContent)
#Declare where to fetch alpaka3 from
#This will download the library at configure time
FetchContent_Declare(alpaka3  GIT_REPOSITORY  https://github.com/alpaka-group/alpaka3.git  GIT_TAG  dev)
#Make alpaka3 available for use in this project
#This downloads, configures, and makes the library targets available
FetchContent_MakeAvailable(alpaka3)
#Finalize the alpaka FetchContent setup
alpaka_FetchContent_Finalize() #Create the executable target from the source file
add_executable(vectorAdd  main.cpp)
#Link the alpaka library to the executable
target_link_libraries(vectorAdd  PRIVATE  alpaka::alpaka)
#Finalize the alpaka configuration for this target
#This sets up backend - specific compiler flags and dependencies
alpaka_finalize(vectorAdd) 
```

下面是我们主要的 alpaka 代码，使用高级转换函数在设备上进行向量加法

main.cpp

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

auto  main()  ->  int  {
  unsigned  n  =  5;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */

  // auto devSelector = ap::onHost::makeDeviceSelector(ap::api::host,
  // ap::deviceKind::cpu);
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue  queue  =  devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate unified memory that is accessible on host and device
  auto  a  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  b  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  c  =  ap::onHost::allocUnified<int>(devAcc,  n);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

  // Run element-wise vector addition on device
 ap::onHost::transform(queue,  c,  std::plus{},  a,  b); 
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  c[i]);
  }

  return  0;
} 
```

为了设置我们的项目，我们创建一个文件夹，并将我们的 CMakeLists.txt 和 main.cpp 放在那里。

```
$ mkdir  alpakaExercise  &&  cd  alpakaExercise
$ vim  CMakeLists.txt
and now paste the CMakeLsits here (Press i, followed by Ctrl+Shift+V)
Press esc and then :wq to exit vim
$ vim  main.cpp
Similarly, paste the C++ code here 
```

要编译和运行代码，请使用以下命令：

```
configure step, we additionaly specify that HIP is available
$ cmake  -B  build  -S  .  -Dalpaka_DEP_HIP=ON
build
$ cmake  --build  build  --parallel
run
$ ./build/vectorAdd
Using alpaka device: AMD Instinct MI250X id=0
c[0] = 1
c[1] = 2
c[2] = 3
c[3] = 4
c[4] = 5 
```

现在您的任务将是编写和启动您的第一个 alpaka 内核。这个内核将执行向量加法，我们将使用这个而不是转换辅助函数。

编写向量加法内核

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

struct  AddKernel  {
 constexpr  void  operator()(ap::onAcc::concepts::Acc  auto  const  &acc, ap::concepts::IMdSpan  auto  c, ap::concepts::IMdSpan  auto  const  a, ap::concepts::IMdSpan  auto  const  b)  const  { for  (auto  idx  :  ap::onAcc::makeIdxMap(acc,  ap::onAcc::worker::threadsInGrid, ap::IdxRange{c.getExtents()}))  { c[idx]  =  a[idx]  +  b[idx]; } } }; 
auto  main()  ->  int  {
  unsigned  n  =  5;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */

  // auto devSelector = ap::onHost::makeDeviceSelector(ap::api::host,
  // ap::deviceKind::cpu);
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue  queue  =  devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate unified memory that is accessible on host and device
  auto  a  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  b  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  c  =  ap::onHost::allocUnified<int>(devAcc,  n);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

 auto  frameSpec  =  ap::onHost::getFrameSpec<int>(devAcc,  c.getExtents()); 
 // Call the element-wise addition kernel on device queue.enqueue(frameSpec,  ap::KernelBundle{AddKernel{},  c,  a,  b}); 
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  c[i]);
  }

  return  0;
} 
```

### 在您的系统上安装 alpaka

为了方便使用，我们建议按照以下说明使用 CMake 安装 alpaka。有关在项目中使用 alpaka 的其他方法，请参阅 [alpaka3 文档](https://alpaka3.readthedocs.io/en/latest/basic/install.html)。

1.  **克隆仓库**

    将 alpaka 源代码从 GitHub 克隆到您选择的目录：

    ```
    git  clone  https://github.com/alpaka-group/alpaka3.git
    cd  alpaka 
    ```

1.  **设置安装目录**

    将 `ALPAKA_DIR` 环境变量设置为要安装 alpaka 的目录。这可以是您选择的任何目录，您有写入权限。

    ```
    export  ALPAKA_DIR=/path/to/your/alpaka/install/dir 
    ```

1.  **构建和安装**

    创建一个构建目录，并使用 CMake 构建和安装 alpaka。我们使用 `CMAKE_INSTALL_PREFIX` 来告诉 CMake 将库安装在哪里。

    ```
    mkdir  build
    cmake  -B  build  -S  .  -DCMAKE_INSTALL_PREFIX=$ALPAKA_DIR
    cmake  --build  build  --parallel 
    ```

1.  **更新环境**

    为了确保其他项目可以找到您的 alpaka 安装，您应该将安装目录添加到您的 `CMAKE_PREFIX_PATH` 中。您可以通过将以下行添加到您的 shell 配置文件（例如 `~/.bashrc`）来实现：

    ```
    export  CMAKE_PREFIX_PATH=$ALPAKA_DIR:$CMAKE_PREFIX_PATH 
    ```

    您需要源码您的 shell 配置文件或打开一个新的终端以使更改生效。

### alpaka 编译

我们建议使用 CMake 构建使用 alpaka 的项目。可以采用各种策略来处理为特定设备或设备集构建应用程序。这里我们展示了开始的最小方法，但这绝对不是设置项目的唯一方法。请参阅[alpaka3 文档](https://alpaka3.readthedocs.io/en/latest/basic/install.html)，了解在项目中使用 alpaka 的替代方法，包括通过在 CMake 中定义设备规范来使源代码与目标加速器无关的方法。

以下示例演示了一个使用 alpaka3 的单文件项目的`CMakeLists.txt`（以下章节中展示的`main.cpp`）：

> ```
> cmake_minimum_required(VERSION  3.25)
> project(myAlpakaApp  VERSION  1.0)
> 
> # Find installed alpaka
> find_package(alpaka  REQUIRED)
> 
> # Build the executable
> add_executable(myAlpakaApp  main.cpp)
> target_link_libraries(myAlpakaApp  PRIVATE  alpaka::alpaka)
> alpaka_finalize(myAlpakaApp) 
> ```

#### 在 LUMI 上使用 alpaka

要加载在 LUMI 上使用 AMD GPU 的 HIP 环境，可以使用以下模块 -

```
$ module  load  LUMI/24.03  partition/G
$ module  load  rocm/6.0.3
$ module  load  buildtools/24.03
$ module  load  PrgEnv-amd
$ module  load  craype-accel-amd-gfx90a
$ export  CXX=hipcc 
```

#### 在 LUMI 上使用 alpaka

要加载在 LUMI 上使用 AMD GPU 的 HIP 环境，可以使用以下模块 -

```
$ module  load  LUMI/24.03  partition/G
$ module  load  rocm/6.0.3
$ module  load  buildtools/24.03
$ module  load  PrgEnv-amd
$ module  load  craype-accel-amd-gfx90a
$ export  CXX=hipcc 
```

### alpaka 编程

当从 alpaka3 开始时，第一步是理解**设备选择模型**。与需要显式初始化调用的框架不同，alpaka3 使用设备规范来确定使用哪个后端和硬件。设备规范由两个组件组成：

+   **API**: 并行编程接口（host、cuda、hip、oneApi）

+   **设备类型**: 硬件类型（cpu、nvidiaGpu、amdGpu、intelGpu）

在这里，我们指定并使用这些内容在运行时选择和初始化设备。设备选择过程在 alpaka3 文档中有详细描述。

alpaka3 使用**执行空间模型**来抽象并行硬件细节。使用`alpaka::onHost::makeDeviceSelector(devSpec)`创建设备选择器，它返回一个可以查询可用设备并为所选后端创建设备实例的对象。

以下示例演示了一个基本的 alpaka 程序，该程序初始化一个设备并打印有关该设备的信息：

> ```
> #include  <alpaka/alpaka.hpp>
> #include  <cstdlib>
> #include  <iostream>
> 
> namespace  ap  =  alpaka;
> 
> auto  getDeviceSpec()
> {
>   /* Select a device, possible combinations of api+deviceKind:
>  * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
>  * oneApi+amdGpu, oneApi+nvidiaGpu
>  */
>   return  ap::onHost::DeviceSpec{ap::api::hip,  ap::deviceKind::amdGpu};
> }
> 
> int  main(int  argc,  char**  argv)
> {
>   // Initialize device specification and selector
>   ap::onHost::DeviceSpec  devSpec  =  getDeviceSpec();
>   auto  deviceSelector  =  ap::onHost::makeDeviceSelector(devSpec);
> 
>   // Query available devices
>   auto  num_devices  =  deviceSelector.getDeviceCount();
>   std::cout  <<  "Number of available devices: "  <<  num_devices  <<  "\n";
> 
>   if  (num_devices  ==  0)  {
>   std::cerr  <<  "No devices found for the selected backend\n";
>   return  EXIT_FAILURE;
>   }
> 
>   // Select and initialize the first device
>   auto  device  =  deviceSelector.makeDevice(0);
>   std::cout  <<  "Using device: "  <<  device.getName()  <<  "\n";
> 
>   return  EXIT_SUCCESS;
> } 
> ```

alpaka3 通过缓冲区和视图提供内存管理抽象。可以使用`alpaka::allocBuf<T, Idx>(device, extent)`在主机或设备上分配内存。主机和设备之间的数据传输通过`alpaka::memcpy(queue, dst, src)`处理。库自动管理不同架构上的内存布局以实现最佳性能。

对于并行执行，alpaka3 提供内核抽象。内核被定义为函数式对象或 lambda 函数，并使用定义并行化策略的工作划分规范来执行。该框架支持各种并行模式，包括逐元素操作、归约和扫描。

#### **alpaka**功能概览

现在我们将快速探索 alpaka 最常用的功能，并简要介绍一些基本用法。常用 alpaka 功能的快速参考可在[这里](https://alpaka3.readthedocs.io/en/latest/basic/cheatsheet.html)找到。

**通用设置**: 一次性包含综合头文件，然后即可开始使用 alpaka。

```
#include  <alpaka/alpaka.hpp>

namespace  myProject
{
  namespace  ap  =  alpaka;
  // Your code here
} 
```

**加速器、平台和设备管理**：通过使用设备选择器将所需的 API 与适当的硬件类型组合来选择设备。

```
auto  devSelector  =  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
if  (devSelector.getDeviceCount()  ==  0)
{
  throw  std::runtime_error("No device found!");
}
auto  device  =  devSelector.makeDevice(0); 
```

**队列和事件**：为每个设备创建阻塞或非阻塞队列，记录事件，并在需要时同步工作。

```
auto  queue  =  device.makeQueue();
auto  nonBlockingQueue  =  device.makeQueue(ap::queueKind::nonBlocking);
auto  blockingQueue  =  device.makeQueue(ap::queueKind::blocking);

auto  event  =  device.makeEvent();
queue.enqueue(event);
ap::onHost::wait(event);
ap::onHost::wait(queue); 
```

**内存管理**：分配主机、设备、映射、统一或延迟缓冲区，创建非拥有视图，并使用 memcpy、memset 和 fill 可移植地移动数据。

```
auto  hostBuffer  =  ap::onHost::allocHost<DataType>(extent3D);
auto  devBuffer  =  ap::onHost::alloc<DataType>(device,  extentMd);
auto  devMappedBuffer  =  ap::onHost::allocMapped<DataType>(device,  extentMd);

auto  hostView  =  ap::makeView(api::host,  externPtr,  ap::Vec{numElements});
auto  devNonOwningView  =  devBuffer.getView();

ap::onHost::memset(queue,  devBuffer,  uint8_t{0});
ap::onHost::memcpy(queue,  devBuffer,  hostBuffer);
ap::onHost::fill(queue,  devBuffer,  DataType{42}); 
```

**内核执行**：手动构建 FrameSpec 或请求针对您的数据类型调优的 FrameSpec，然后使用自动或显式执行器将内核入队。

```
constexpr  uint32_t  dim  =  2u;
using  IdxType  =  size_t;
using  DataType  =  int;

IdxType  valueX,  valueY;
auto  extentMD  =  ap::Vec{valueY,  valueX};

auto  frameSpec  =  ap::onHost::FrameSpec{numFramesMd,  frameExtentMd};
auto  tunedSpec  =  ap::onHost::getFrameSpec<DataType>(device,  extentMd);

queue.enqueue(tunedSpec,  ap::KernelBundle{kernel,  kernelArgs...});

auto  executor  =  ap::exec::cpuSerial;
queue.enqueue(executor,  tunedSpec,  ap::KernelBundle{kernel,  kernelArgs...}); 
```

**内核实现**：将内核编写为带有 ALPAKA_FN_ACC 注解的函数式对象，在内核体内部直接使用共享内存、同步、原子操作和数学助手。

```
struct  MyKernel
{
  ALPAKA_FN_ACC  void  operator()(ap::onAcc::concepts::Acc  auto  const&  acc,  auto...  args)  const
  {
  auto  idxMd  =  acc.getIdxWithin(ap::onAcc::origin::grid,  ap::onAcc::unit::blocks);

  auto  sharedMdArray  =
  ap::onAcc::declareSharedMdArray<float,  ap::uniqueId()>(acc,  ap::CVec<uint32_t,  3,  4>{});

  ap::onAcc::syncBlockThreads(acc);
  auto  old  =  onAcc::atomicAdd(acc,  args...);
  ap::onAcc::memFence(acc,  ap::onAcc::scope::block);
  auto  sinValue  =  ap::math::sin(args[0]);
  }
}; 
```

#### 简单步骤运行 alpaka3 示例

以下示例适用于 CMake 3.25+ 及适当的 C++ 编译器。对于 GPU 执行，请确保已安装相应的运行时（CUDA、ROCm 或 oneAPI）。

1.  为您的项目创建一个目录：

    ```
    mkdir  my_alpaka_project  &&  cd  my_alpaka_project 
    ```

1.  将上面的 CMakeLists.txt 复制到当前文件夹

1.  将 main.cpp 文件复制到当前文件夹

1.  配置和构建：

    ```
    cmake  -B  build  -S  .  -Dalpaka_DEP_HIP=ON
    cmake  --build  build  --parallel 
    ```

1.  运行可执行文件：

    ```
    ./build/myAlpakaApp 
    ```

注意

设备规范系统允许你在 CMake 配置时间选择目标设备。格式为 `"api:deviceKind"`，其中：

+   **api**：并行编程接口（`host`、`cuda`、`hip`、`oneApi`）

+   **deviceKind**：设备类型（`cpu`、`nvidiaGpu`、`amdGpu`、`intelGpu`）

可用的组合有：`host:cpu`、`cuda:nvidiaGpu`、`hip:amdGpu`、`oneApi:cpu`、`oneApi:intelGpu`、`oneApi:nvidiaGpu`、`oneApi:amdGpu`

警告

只有当 CUDA SDK、HIP SDK 或 OneAPI SDK 分别可用时，CUDA、HIP 或 Intel 后端才有效。

#### 预期输出

```
Number of available devices: 1
Using device: [Device Name] 
```

设备名称将根据您的硬件而变化（例如，“NVIDIA A100”、“AMD MI250X”或您的 CPU 型号）。

#### **alpaka** 功能巡礼

现在我们将快速探索 alpaka 最常用的功能，并简要介绍一些基本用法。常用的 alpaka 功能快速参考可在[这里](https://alpaka3.readthedocs.io/en/latest/basic/cheatsheet.html)找到。

**通用设置**：包含一次综合头文件，您就可以开始使用 alpaka 了。

```
#include  <alpaka/alpaka.hpp>

namespace  myProject
{
  namespace  ap  =  alpaka;
  // Your code here
} 
```

**加速器、平台和设备管理**：通过使用设备选择器将所需的 API 与适当的硬件类型组合来选择设备。

```
auto  devSelector  =  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
if  (devSelector.getDeviceCount()  ==  0)
{
  throw  std::runtime_error("No device found!");
}
auto  device  =  devSelector.makeDevice(0); 
```

**队列和事件**：为每个设备创建阻塞或非阻塞队列，记录事件，并在需要时同步工作。

```
auto  queue  =  device.makeQueue();
auto  nonBlockingQueue  =  device.makeQueue(ap::queueKind::nonBlocking);
auto  blockingQueue  =  device.makeQueue(ap::queueKind::blocking);

auto  event  =  device.makeEvent();
queue.enqueue(event);
ap::onHost::wait(event);
ap::onHost::wait(queue); 
```

**内存管理**：分配主机、设备、映射、统一或延迟缓冲区，创建非拥有视图，并使用 memcpy、memset 和 fill 可移植地移动数据。

```
auto  hostBuffer  =  ap::onHost::allocHost<DataType>(extent3D);
auto  devBuffer  =  ap::onHost::alloc<DataType>(device,  extentMd);
auto  devMappedBuffer  =  ap::onHost::allocMapped<DataType>(device,  extentMd);

auto  hostView  =  ap::makeView(api::host,  externPtr,  ap::Vec{numElements});
auto  devNonOwningView  =  devBuffer.getView();

ap::onHost::memset(queue,  devBuffer,  uint8_t{0});
ap::onHost::memcpy(queue,  devBuffer,  hostBuffer);
ap::onHost::fill(queue,  devBuffer,  DataType{42}); 
```

**内核执行**：手动构建 FrameSpec 或请求针对您的数据类型调优的 FrameSpec，然后使用自动或显式执行器将内核入队。

```
constexpr  uint32_t  dim  =  2u;
using  IdxType  =  size_t;
using  DataType  =  int;

IdxType  valueX,  valueY;
auto  extentMD  =  ap::Vec{valueY,  valueX};

auto  frameSpec  =  ap::onHost::FrameSpec{numFramesMd,  frameExtentMd};
auto  tunedSpec  =  ap::onHost::getFrameSpec<DataType>(device,  extentMd);

queue.enqueue(tunedSpec,  ap::KernelBundle{kernel,  kernelArgs...});

auto  executor  =  ap::exec::cpuSerial;
queue.enqueue(executor,  tunedSpec,  ap::KernelBundle{kernel,  kernelArgs...}); 
```

**内核实现**：将内核编写为带有 ALPAKA_FN_ACC 注解的函数式对象，在内核体内部直接使用共享内存、同步、原子操作和数学助手。

```
struct  MyKernel
{
  ALPAKA_FN_ACC  void  operator()(ap::onAcc::concepts::Acc  auto  const&  acc,  auto...  args)  const
  {
  auto  idxMd  =  acc.getIdxWithin(ap::onAcc::origin::grid,  ap::onAcc::unit::blocks);

  auto  sharedMdArray  =
  ap::onAcc::declareSharedMdArray<float,  ap::uniqueId()>(acc,  ap::CVec<uint32_t,  3,  4>{});

  ap::onAcc::syncBlockThreads(acc);
  auto  old  =  onAcc::atomicAdd(acc,  args...);
  ap::onAcc::memFence(acc,  ap::onAcc::scope::block);
  auto  sinValue  =  ap::math::sin(args[0]);
  }
}; 
```

#### 简单步骤运行 alpaka3 示例

以下示例适用于 CMake 3.25+ 和适当的 C++ 编译器。对于 GPU 执行，请确保已安装相应的运行时（CUDA、ROCm 或 oneAPI）。

1.  为您的项目创建一个目录：

    ```
    mkdir  my_alpaka_project  &&  cd  my_alpaka_project 
    ```

1.  将上面的 CMakeLists.txt 复制到当前文件夹

1.  将 main.cpp 文件复制到当前文件夹

1.  配置和构建：

    ```
    cmake  -B  build  -S  .  -Dalpaka_DEP_HIP=ON
    cmake  --build  build  --parallel 
    ```

1.  运行可执行文件：

    ```
    ./build/myAlpakaApp 
    ```

注意

设备指定系统允许您在 CMake 配置时选择目标设备。格式为 `"api:deviceKind"`，其中：

+   **api**：并行编程接口（`host`、`cuda`、`hip`、`oneApi`）

+   **deviceKind**：设备类型（`cpu`、`nvidiaGpu`、`amdGpu`、`intelGpu`）

可用的组合有：`host:cpu`、`cuda:nvidiaGpu`、`hip:amdGpu`、`oneApi:cpu`、`oneApi:intelGpu`、`oneApi:nvidiaGpu`、`oneApi:amdGpu`

警告

只有当 CUDA SDK、HIP SDK 或 OneAPI SDK 可用时，CUDA、HIP 或 Intel 后端才有效

#### 预期输出

```
Number of available devices: 1
Using device: [Device Name] 
```

设备名称将根据您的硬件而有所不同（例如，“NVIDIA A100”，“AMD MI250X”或您的 CPU 型号）。

### 编译和执行示例

您可以从示例部分测试提供的 **alpaka** 示例。这些示例已硬编码了在 LUMI 上所需的 AMD ROCm 平台的用法。要仅使用 CPU，您只需将 `ap::onHost::makeDeviceSelector(ap::api::hip, ap::deviceKind::amdGpu);` 替换为 `ap::onHost::makeDeviceSelector(ap::api::host, ap::deviceKind::cpu);`

以下步骤假设您已经下载了 alpaka，并且 **alapka** 源代码的路径已存储在环境变量 `ALPAKA_DIR` 中。要测试示例，请将代码复制到文件 `main.cpp`

或者，[点击此处](https://godbolt.org/z/69exnG4xb) 尝试使用 godbolt 编译器探索器中的第一个示例。

```
# use the following in C++ code
# auto devSelector = ap::onHost::makeDeviceSelector(ap::api::hip, ap::deviceKind::amdGpu);
# We use CC to refer to the compiler to work smoothly with the LUMI environment
CC  -I  $ALPAKA_DIR/include/  -std=c++20  -x  hip  --offload-arch=gfx90a  main.cpp
./a.out 
```

```
# use the following in C++ code
# auto devSelector = ap::onHost::makeDeviceSelector(ap::api::host, ap::deviceKind::cpu);
# We use CC to refer to the compiler to work smoothly with the LUMI environment
CC  -I  $ALPAKA_DIR/include/  -std=c++20  main.cpp
./a.out 
```

```
# use the following in C++ code
# auto devSelector = ap::onHost::makeDeviceSelector(ap::api::cuda, ap::deviceKind::nvidiaGpu);
nvcc  -I  $ALPAKA_DIR/include/  -std=c++20  --expt-relaxed-constexpr  -x  cuda  main.cpp
./a.out 
```

```
# use the following in C++ code
# auto devSelector = ap::onHost::makeDeviceSelector(ap::api::oneApi, ap::deviceKind::cpu);
icpx  -I  $ALPAKA_DIR/include/  -std=c++20  -fsycl  -fsycl-targets=spir64_x86_64  main.cpp
./a.out 
```

```
# use the following in C++ code
# auto devSelector = ap::onHost::makeDeviceSelector(ap::api::oneApi, ap::deviceKind::intelGpu);
icpx  -I  $ALPAKA_DIR/include/  -std=c++20  -fsycl  -fsycl-targets=spir64  main.cpp
./a.out 
```

注意

要使用 AMD 或 NVIDIA GPU 的 oneAPI Sycl，你必须安装相应的 Codeplay oneAPI 插件，具体说明[在此](https://codeplay.com/solutions/oneapi/plugins/)。

```
# use the following in C++ code
# auto devSelector = ap::onHost::makeDeviceSelector(ap::api::oneApi, ap::deviceKind::amdGpu);
icpx  -I  $ALPAKA_DIR/include/  -std=c++20  -fsycl  -fsycl-targets=amd_gpu_gfx90a  main.cpp
./a.out 
```

注意

要使用 AMD 或 NVIDIA GPU 的 oneAPI Sycl，你必须安装相应的 Codeplay oneAPI 插件，具体说明[在此](https://codeplay.com/solutions/oneapi/plugins/)。

```
# use the following in C++ code
# auto devSelector = ap::onHost::makeDeviceSelector(ap::api::oneApi, ap::deviceKind::nvidiaGpu);
icpx  -I  $ALPAKA_DIR/include/  -std=c++20  -fsycl  -fsycl-targets=nvptx64-nvidia-cuda  -Xsycl-target-backend=nvptx64-nvidia-cuda  --offload-arch=sm_80  main.cpp
./a.out 
```

### 练习

练习：在 alpaka 中编写向量加法内核

在这个练习中，我们希望编写（填空）一个简单的内核来添加两个向量。

要交互式地编译和运行代码，我们首先需要在一个 GPU 节点上获取一个分配并加载 alpaka 的模块：

```
$ srun  -p  dev-g  --gpus  1  -N  1  -n  1  --time=00:20:00  --account=project_465002387  --pty  bash
....
srun: job 1234 queued and waiting for resources
srun: job 1234 has been allocated resources

$ module  load  LUMI/24.03  partition/G
$ module  load  rocm/6.0.3
$ module  load  buildtools/24.03
$ module  load  PrgEnv-amd
$ module  load  craype-accel-amd-gfx90a
$ export  CXX=hipcc 
```

现在你可以运行一个简单的设备检测工具来检查是否有 GPU 可用（注意 `srun`）：

```
$ rocm-smi

======================================= ROCm System Management Interface =======================================
================================================= Concise Info =================================================
Device  [Model : Revision]    Temp    Power  Partitions      SCLK    MCLK     Fan  Perf    PwrCap  VRAM%  GPU%
 Name (20 chars)       (Edge)  (Avg)  (Mem, Compute)
================================================================================================================
0       [0x0b0c : 0x00]       45.0°C  N/A    N/A, N/A        800Mhz  1600Mhz  0%   manual  0.0W      0%   0%
 AMD INSTINCT MI200 (
================================================================================================================
============================================= End of ROCm SMI Log ============================================== 
```

现在，让我们看看设置练习的代码：

下面我们使用 CMake 的 fetch content 快速开始使用 alpaka。

CMakeLists.txt

```
cmake_minimum_required(VERSION  3.25)
project(vectorAdd  LANGUAGES  CXX  VERSION  1.0)
#Use CMake's FetchContent to download and integrate alpaka3 directly from GitHub
include(FetchContent)
#Declare where to fetch alpaka3 from
#This will download the library at configure time
FetchContent_Declare(alpaka3  GIT_REPOSITORY  https://github.com/alpaka-group/alpaka3.git  GIT_TAG  dev)
#Make alpaka3 available for use in this project
#This downloads, configures, and makes the library targets available
FetchContent_MakeAvailable(alpaka3)
#Finalize the alpaka FetchContent setup
alpaka_FetchContent_Finalize() #Create the executable target from the source file
add_executable(vectorAdd  main.cpp)
#Link the alpaka library to the executable
target_link_libraries(vectorAdd  PRIVATE  alpaka::alpaka)
#Finalize the alpaka configuration for this target
#This sets up backend - specific compiler flags and dependencies
alpaka_finalize(vectorAdd) 
```

下面是我们使用高级转换函数在设备上执行向量加法的主要 alpaka 代码

main.cpp

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

auto  main()  ->  int  {
  unsigned  n  =  5;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */

  // auto devSelector = ap::onHost::makeDeviceSelector(ap::api::host,
  // ap::deviceKind::cpu);
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue  queue  =  devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate unified memory that is accessible on host and device
  auto  a  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  b  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  c  =  ap::onHost::allocUnified<int>(devAcc,  n);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

  // Run element-wise vector addition on device
 ap::onHost::transform(queue,  c,  std::plus{},  a,  b); 
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  c[i]);
  }

  return  0;
} 
```

要设置我们的项目，我们创建一个文件夹，并将我们的 CMakeLists.txt 和 main.cpp 放在那里。

```
$ mkdir  alpakaExercise  &&  cd  alpakaExercise
$ vim  CMakeLists.txt
and now paste the CMakeLsits here (Press i, followed by Ctrl+Shift+V)
Press esc and then :wq to exit vim
$ vim  main.cpp
Similarly, paste the C++ code here 
```

要编译和运行代码，请使用以下命令：

```
configure step, we additionaly specify that HIP is available
$ cmake  -B  build  -S  .  -Dalpaka_DEP_HIP=ON
build
$ cmake  --build  build  --parallel
run
$ ./build/vectorAdd
Using alpaka device: AMD Instinct MI250X id=0
c[0] = 1
c[1] = 2
c[2] = 3
c[3] = 4
c[4] = 5 
```

现在的任务是编写和启动你的第一个 alpaka 内核。这个内核将执行向量加法，我们将使用这个代替 transform 辅助器。

编写向量加法内核。

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

struct  AddKernel  {
 constexpr  void  operator()(ap::onAcc::concepts::Acc  auto  const  &acc, ap::concepts::IMdSpan  auto  c, ap::concepts::IMdSpan  auto  const  a, ap::concepts::IMdSpan  auto  const  b)  const  { for  (auto  idx  :  ap::onAcc::makeIdxMap(acc,  ap::onAcc::worker::threadsInGrid, ap::IdxRange{c.getExtents()}))  { c[idx]  =  a[idx]  +  b[idx]; } } }; 
auto  main()  ->  int  {
  unsigned  n  =  5;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */

  // auto devSelector = ap::onHost::makeDeviceSelector(ap::api::host,
  // ap::deviceKind::cpu);
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue  queue  =  devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate unified memory that is accessible on host and device
  auto  a  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  b  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  c  =  ap::onHost::allocUnified<int>(devAcc,  n);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

 auto  frameSpec  =  ap::onHost::getFrameSpec<int>(devAcc,  c.getExtents()); 
 // Call the element-wise addition kernel on device queue.enqueue(frameSpec,  ap::KernelBundle{AddKernel{},  c,  a,  b}); 
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  c[i]);
  }

  return  0;
} 
```

## 示例。

### 带统一内存的并行 for 循环。

```
#include  <algorithm>
#include  <cstdio>
#include  <execution>
#include  <vector>

int  main()  {
  unsigned  n  =  5;

  // Allocate arrays
  std::vector<int>  a(n),  b(n),  c(n);

  // Initialize values
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

  // Run element-wise multiplication on device
  std::transform(std::execution::par_unseq,  a.begin(),  a.end(),  b.begin(),
  c.begin(),  [](int  i,  int  j)  {  return  i  *  j;  });

  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  c[i]);
  }

  return  0;
} 
```

```
#include  <Kokkos_Core.hpp>

int  main(int  argc,  char  *argv[])  {

  // Initialize Kokkos
  Kokkos::initialize(argc,  argv);

  {
  unsigned  n  =  5;

  // Allocate on Kokkos default memory space (Unified Memory)
  int  *a  =  (int  *)Kokkos::kokkos_malloc(n  *  sizeof(int));
  int  *b  =  (int  *)Kokkos::kokkos_malloc(n  *  sizeof(int));
  int  *c  =  (int  *)Kokkos::kokkos_malloc(n  *  sizeof(int));

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

  // Run element-wise multiplication on device
  Kokkos::parallel_for(n,  KOKKOS_LAMBDA(const  int  i)  {  c[i]  =  a[i]  *  b[i];  });

  // Kokkos synchronization
  Kokkos::fence();

  // Print results
  for  (unsigned  i  =  0;  i  <  n;  i++)
  printf("c[%d] = %d\n",  i,  c[i]);

  // Free Kokkos allocation (Unified Memory)
  Kokkos::kokkos_free(a);
  Kokkos::kokkos_free(b);
  Kokkos::kokkos_free(c);
  }

  // Finalize Kokkos
  Kokkos::finalize();
  return  0;
} 
```

```
// We're using OpenCL C API here, since SVM support in C++ API is unstable on
// ROCm
#define CL_TARGET_OPENCL_VERSION 220
#include  <CL/cl.h>
#include  <stdio.h>

// For larger kernels, we can store source in a separate file
static  const  char  *kernel_source  =
  "                                                 \
 __kernel void dot(__global const int *a, __global const int *b, __global int *c) { \
 int i = get_global_id(0);                                                        \
 c[i] = a[i] * b[i];                                                              \
 }                                                                                  \
 ";

int  main(int  argc,  char  *argv[])  {

  // Initialize OpenCL
  cl_platform_id  platform;
  clGetPlatformIDs(1,  &platform,  NULL);
  cl_device_id  device;
  clGetDeviceIDs(platform,  CL_DEVICE_TYPE_GPU,  1,  &device,  NULL);
  cl_context  context  =  clCreateContext(NULL,  1,  &device,  NULL,  NULL,  NULL);
  cl_command_queue  queue  =  clCreateCommandQueue(context,  device,  0,  NULL);

  // Compile OpenCL program for found device.
  cl_program  program  =
  clCreateProgramWithSource(context,  1,  &kernel_source,  NULL,  NULL);
  clBuildProgram(program,  1,  &device,  NULL,  NULL,  NULL);
  cl_kernel  kernel  =  clCreateKernel(program,  "dot",  NULL);

  // Set problem dimensions
  unsigned  n  =  5;

  // Create SVM buffer objects on host side
  int  *a  =  clSVMAlloc(context,  CL_MEM_READ_ONLY,  n  *  sizeof(int),  0);
  int  *b  =  clSVMAlloc(context,  CL_MEM_READ_ONLY,  n  *  sizeof(int),  0);
  int  *c  =  clSVMAlloc(context,  CL_MEM_WRITE_ONLY,  n  *  sizeof(int),  0);

  // Pass arguments to device kernel
  clSetKernelArgSVMPointer(kernel,  0,  a);
  clSetKernelArgSVMPointer(kernel,  1,  b);
  clSetKernelArgSVMPointer(kernel,  2,  c);

  // Create mappings for host and initialize values
  clEnqueueSVMMap(queue,  CL_TRUE,  CL_MAP_WRITE,  a,  n  *  sizeof(int),  0,  NULL,
  NULL);
  clEnqueueSVMMap(queue,  CL_TRUE,  CL_MAP_WRITE,  b,  n  *  sizeof(int),  0,  NULL,
  NULL);
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }
  clEnqueueSVMUnmap(queue,  a,  0,  NULL,  NULL);
  clEnqueueSVMUnmap(queue,  b,  0,  NULL,  NULL);

  size_t  globalSize  =  n;
  clEnqueueNDRangeKernel(queue,  kernel,  1,  NULL,  &globalSize,  NULL,  0,  NULL,
  NULL);

  // Create mapping for host and print results
  clEnqueueSVMMap(queue,  CL_TRUE,  CL_MAP_READ,  c,  n  *  sizeof(int),  0,  NULL,
  NULL);
  for  (unsigned  i  =  0;  i  <  n;  i++)
  printf("c[%d] = %d\n",  i,  c[i]);
  clEnqueueSVMUnmap(queue,  c,  0,  NULL,  NULL);

  // Free SVM buffers
  clSVMFree(context,  a);
  clSVMFree(context,  b);
  clSVMFree(context,  c);

  return  0;
} 
```

```
#include  <sycl/sycl.hpp>

int  main()  {

  sycl::queue  q;
  unsigned  n  =  5;

  // Allocate shared memory (Unified Shared Memory)
  int  *a  =  sycl::malloc_shared<int>(n,  q);
  int  *b  =  sycl::malloc_shared<int>(n,  q);
  int  *c  =  sycl::malloc_shared<int>(n,  q);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

  // Run element-wise multiplication on device
  q.parallel_for(sycl::range<1>{n},  =  {
  c[i]  =  a[i]  *  b[i];
  }).wait();

  // Print results
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  c[i]);
  }

  // Free shared memory allocation (Unified Memory)
  sycl::free(a,  q);
  sycl::free(b,  q);
  sycl::free(c,  q);

  return  0;
} 
```

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

auto  main()  ->  int  {
  unsigned  n  =  5;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue  queue  =  devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate unified memory that is accessible on host and device
  auto  a  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  b  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  c  =  ap::onHost::allocUnified<int>(devAcc,  n);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

  // Run element-wise multiplication on device
  ap::onHost::transform(queue,  c,  std::multiplies{},  a,  b);

  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  c[i]);
  }

  return  0;
} 
```

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

struct  MulKernel  {
  constexpr  void  operator()(ap::onAcc::concepts::Acc  auto  const&  acc,
  ap::concepts::IMdSpan  auto  c,
  ap::concepts::IMdSpan  auto  const  a,
  ap::concepts::IMdSpan  auto  const  b)  const  {
  for  (auto  idx  :  ap::onAcc::makeIdxMap(acc,  ap::onAcc::worker::threadsInGrid,
  ap::IdxRange{c.getExtents()}))  {
  c[idx]  =  a[idx]  *  b[idx];
  }
  }
};

auto  main()  ->  int  {
  unsigned  n  =  5;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue  queue  =  devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate unified memory that is accessible on host and device
  auto  a  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  b  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  c  =  ap::onHost::allocUnified<int>(devAcc,  n);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

  unsigned  frameExtent  =  32u;
  auto  frameSpec  =
  ap::onHost::FrameSpec{ap::divExZero(n,  frameExtent),  frameExtent};

  // Run element-wise multiplication on device
  queue.enqueue(frameSpec,  ap::KernelBundle{MulKernel{},  c,  a,  b});

  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  c[i]);
  }

  return  0;
} 
```

### 带 GPU 缓冲区的并行 for 循环。

```
#include  <Kokkos_Core.hpp>

int  main(int  argc,  char  *argv[])  {

  // Initialize Kokkos
  Kokkos::initialize(argc,  argv);

  {
  unsigned  n  =  5;

  // Allocate space for 5 ints on Kokkos host memory space
  Kokkos::View<int  *,  Kokkos::HostSpace>  h_a("h_a",  n);
  Kokkos::View<int  *,  Kokkos::HostSpace>  h_b("h_b",  n);
  Kokkos::View<int  *,  Kokkos::HostSpace>  h_c("h_c",  n);

  // Allocate space for 5 ints on Kokkos default memory space (eg, GPU memory)
  Kokkos::View<int  *>  a("a",  n);
  Kokkos::View<int  *>  b("b",  n);
  Kokkos::View<int  *>  c("c",  n);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  h_a[i]  =  i;
  h_b[i]  =  1;
  }

  // Copy from host to device
  Kokkos::deep_copy(a,  h_a);
  Kokkos::deep_copy(b,  h_b);

  // Run element-wise multiplication on device
  Kokkos::parallel_for(n,  KOKKOS_LAMBDA(const  int  i)  {  c[i]  =  a[i]  *  b[i];  });

  // Copy from device to host
  Kokkos::deep_copy(h_c,  c);

  // Print results
  for  (unsigned  i  =  0;  i  <  n;  i++)
  printf("c[%d] = %d\n",  i,  h_c[i]);
  }

  // Finalize Kokkos
  Kokkos::finalize();
  return  0;
} 
```

```
// We're using OpenCL C++ API here; there is also C API in <CL/cl.h>
#define CL_TARGET_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110
#include  <CL/cl.hpp>

// For larger kernels, we can store source in a separate file
static  const  std::string  kernel_source  =  R"(
 __kernel void dot(__global const int *a, __global const int *b, __global int *c) {
 int i = get_global_id(0);
 c[i] = a[i] * b[i];
 }
  )";

int  main(int  argc,  char  *argv[])  {

  // Initialize OpenCL
  cl::Device  device  =  cl::Device::getDefault();
  cl::Context  context(device);
  cl::CommandQueue  queue(context,  device);

  // Compile OpenCL program for found device.
  cl::Program  program(context,  kernel_source);
  program.build({device});
  cl::Kernel  kernel_dot(program,  "dot");

  {
  // Set problem dimensions
  unsigned  n  =  5;

  std::vector<int>  a(n),  b(n),  c(n);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

  // Create buffers and copy input data to device.
  cl::Buffer  dev_a(context,  CL_MEM_READ_ONLY  |  CL_MEM_COPY_HOST_PTR,
  n  *  sizeof(int),  a.data());
  cl::Buffer  dev_b(context,  CL_MEM_READ_ONLY  |  CL_MEM_COPY_HOST_PTR,
  n  *  sizeof(int),  b.data());
  cl::Buffer  dev_c(context,  CL_MEM_WRITE_ONLY,  n  *  sizeof(int));

  // Pass arguments to device kernel
  kernel_dot.setArg(0,  dev_a);
  kernel_dot.setArg(1,  dev_b);
  kernel_dot.setArg(2,  dev_c);

  // We don't need to apply any offset to thread IDs
  queue.enqueueNDRangeKernel(kernel_dot,  cl::NullRange,  cl::NDRange(n),
  cl::NullRange);

  // Read result
  queue.enqueueReadBuffer(dev_c,  CL_TRUE,  0,  n  *  sizeof(int),  c.data());

  // Print results
  for  (unsigned  i  =  0;  i  <  n;  i++)
  printf("c[%d] = %d\n",  i,  c[i]);
  }

  return  0;
} 
```

```
#include  <sycl/sycl.hpp>

int  main()  {

  sycl::queue  q;
  unsigned  n  =  5;

  // Allocate space for 5 ints
  auto  a_buf  =  sycl::buffer<int>(sycl::range<1>(n));
  auto  b_buf  =  sycl::buffer<int>(sycl::range<1>(n));
  auto  c_buf  =  sycl::buffer<int>(sycl::range<1>(n));

  // Initialize values
  // We should use curly braces to limit host accessors' lifetime
  //    and indicate when we're done working with them:
  {
  auto  a_host_acc  =  a_buf.get_host_access();
  auto  b_host_acc  =  b_buf.get_host_access();
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a_host_acc[i]  =  i;
  b_host_acc[i]  =  1;
  }
  }

  // Submit a SYCL kernel into a queue
  q.submit(&  {
  // Create read accessors over a_buf and b_buf
  auto  a_acc  =  a_buf.get_access<sycl::access_mode::read>(cgh);
  auto  b_acc  =  b_buf.get_access<sycl::access_mode::read>(cgh);
  // Create write accesor over c_buf
  auto  c_acc  =  c_buf.get_access<sycl::access_mode::write>(cgh);
  // Run element-wise multiplication on device
  cgh.parallel_for<class  vec_add>(sycl::range<1>{n},  =  {
  c_acc[i]  =  a_acc[i]  *  b_acc[i];
  });
  });

  // No need to synchronize, creating the accessor for c_buf will do it
  // automatically
  {
  const  auto  c_host_acc  =  c_buf.get_host_access();
  // Print results
  for  (unsigned  i  =  0;  i  <  n;  i++)
  printf("c[%d] = %d\n",  i,  c_host_acc[i]);
  }

  return  0;
} 
```

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

auto  main()  ->  int  {
  unsigned  n  =  5;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue  queue  =  devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate memory that is accessible on host
  auto  h_a  =  ap::onHost::allocHost<int>(n);
  auto  h_b  =  ap::onHost::allocHostLike(h_a);
  auto  h_c  =  ap::onHost::allocHostLike(h_a);

  // Allocate memory on the device and inherit the extents from h_a
  auto  a  =  ap::onHost::allocLike(devAcc,  h_a);
  auto  b  =  ap::onHost::allocLike(devAcc,  h_a);
  auto  c  =  ap::onHost::allocLike(devAcc,  h_a);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  h_a[i]  =  i;
  h_b[i]  =  1;
  }

  // Copy host memory element wise to the device memory
  ap::onHost::memcpy(queue,  a,  h_a);
  ap::onHost::memcpy(queue,  b,  h_b);

  // Run element-wise multiplication on device
  ap::onHost::transform(queue,  c,  std::multiplies{},  a,  b);

  // Copy the device result back to host memory
  ap::onHost::memcpy(queue,  h_c,  c);

  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  h_c[i]);
  }

  return  0;
} 
```

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

struct  MulKernel  {
  constexpr  void  operator()(ap::onAcc::concepts::Acc  auto  const&  acc,
  ap::concepts::IMdSpan  auto  c,
  ap::concepts::IMdSpan  auto  const  a,
  ap::concepts::IMdSpan  auto  const  b)  const  {
  for  (auto  idx  :  ap::onAcc::makeIdxMap(acc,  ap::onAcc::worker::threadsInGrid,
  ap::IdxRange{c.getExtents()}))  {
  c[idx]  =  a[idx]  *  b[idx];
  }
  }
};

auto  main()  ->  int  {
  unsigned  n  =  5;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue  queue  =  devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate memory that is accessible on host
  auto  h_a  =  ap::onHost::allocHost<int>(n);
  auto  h_b  =  ap::onHost::allocHostLike(h_a);
  auto  h_c  =  ap::onHost::allocHostLike(h_a);

  // allocate memory on the device and inherit the extents from a
  auto  a  =  ap::onHost::allocLike(devAcc,  h_a);
  auto  b  =  ap::onHost::allocLike(devAcc,  h_a);
  auto  c  =  ap::onHost::allocLike(devAcc,  h_a);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  h_a[i]  =  i;
  h_b[i]  =  1;
  }

  // Copy host memory element wise to the device memory
  ap::onHost::memcpy(queue,  a,  h_a);
  ap::onHost::memcpy(queue,  b,  h_b);

  unsigned  frameExtent  =  32u;
  auto  frameSpec  =
  ap::onHost::FrameSpec{ap::divExZero(n,  frameExtent),  frameExtent};

  // Run element-wise multiplication on device
  queue.enqueue(frameSpec,  ap::KernelBundle{MulKernel{},  c,  a,  b});

  // Copy the device result back to host memory
  ap::onHost::memcpy(queue,  h_c,  c);

  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  h_c[i]);
  }

  return  0;
} 
```

### 异步并行内核。

```
#include  <Kokkos_Core.hpp>

int  main(int  argc,  char  *argv[])  {

  // Initialize Kokkos
  Kokkos::initialize(argc,  argv);

  {
  unsigned  n  =  5;
  unsigned  nx  =  20;

  // Allocate on Kokkos default memory space (Unified Memory)
  int  *a  =  (int  *)Kokkos::kokkos_malloc(nx  *  sizeof(int));

  // Create 'n' execution space instances (maps to streams in CUDA/HIP)
  auto  ex  =  Kokkos::Experimental::partition_space(
  Kokkos::DefaultExecutionSpace(),  1,  1,  1,  1,  1);

  // Launch 'n' potentially asynchronous kernels
  // Each kernel has their own execution space instances
  for  (unsigned  region  =  0;  region  <  n;  region++)  {
  Kokkos::parallel_for(
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
  ex[region],  nx  /  n  *  region,  nx  /  n  *  (region  +  1)),
  KOKKOS_LAMBDA(const  int  i)  {  a[i]  =  region  +  i;  });
  }

  // Sync execution space instances (maps to streams in CUDA/HIP)
  for  (unsigned  region  =  0;  region  <  n;  region++)
  ex[region].fence();

  // Print results
  for  (unsigned  i  =  0;  i  <  nx;  i++)
  printf("a[%d] = %d\n",  i,  a[i]);

  // Free Kokkos allocation (Unified Memory)
  Kokkos::kokkos_free(a);
  }

  // Finalize Kokkos
  Kokkos::finalize();
  return  0;
} 
```

```
// We're using OpenCL C API here, since SVM support in C++ API is unstable on
// ROCm
#define CL_TARGET_OPENCL_VERSION 200
#include  <CL/cl.h>
#include  <stdio.h>

// For larger kernels, we can store source in a separate file
static  const  char  *kernel_source  =  "              \
 __kernel void async(__global int *a) { \
 int i = get_global_id(0);            \
 int region = i / get_global_size(0); \
 a[i] = region + i;                   \
 }                                      \
 ";

int  main(int  argc,  char  *argv[])  {
  // Initialize OpenCL
  cl_platform_id  platform;
  clGetPlatformIDs(1,  &platform,  NULL);
  cl_device_id  device;
  clGetDeviceIDs(platform,  CL_DEVICE_TYPE_GPU,  1,  &device,  NULL);
  cl_context  context  =  clCreateContext(NULL,  1,  &device,  NULL,  NULL,  NULL);
  cl_command_queue  queue  =  clCreateCommandQueue(context,  device,  0,  NULL);

  // Compile OpenCL program for found device.
  cl_program  program  =
  clCreateProgramWithSource(context,  1,  &kernel_source,  NULL,  NULL);
  clBuildProgram(program,  1,  &device,  NULL,  NULL,  NULL);
  cl_kernel  kernel  =  clCreateKernel(program,  "async",  NULL);

  // Set problem dimensions
  unsigned  n  =  5;
  unsigned  nx  =  20;

  // Create SVM buffer objects on host side
  int  *a  =  clSVMAlloc(context,  CL_MEM_WRITE_ONLY,  nx  *  sizeof(int),  0);

  // Pass arguments to device kernel
  clSetKernelArgSVMPointer(kernel,  0,  a);

  // Launch multiple potentially asynchronous kernels on different parts of the
  // array
  for  (unsigned  region  =  0;  region  <  n;  region++)  {
  size_t  offset  =  (nx  /  n)  *  region;
  size_t  size  =  nx  /  n;
  clEnqueueNDRangeKernel(queue,  kernel,  1,  &offset,  &size,  NULL,  0,  NULL,
  NULL);
  }

  // Create mapping for host and print results
  clEnqueueSVMMap(queue,  CL_TRUE,  CL_MAP_READ,  a,  nx  *  sizeof(int),  0,  NULL,
  NULL);
  for  (unsigned  i  =  0;  i  <  nx;  i++)
  printf("a[%d] = %d\n",  i,  a[i]);
  clEnqueueSVMUnmap(queue,  a,  0,  NULL,  NULL);

  // Free SVM buffers
  clSVMFree(context,  a);

  return  0;
} 
```

```
#include  <sycl/sycl.hpp>

int  main()  {

  sycl::queue  q;
  unsigned  n  =  5;
  unsigned  nx  =  20;

  // Allocate shared memory (Unified Shared Memory)
  int  *a  =  sycl::malloc_shared<int>(nx,  q);

  // Launch multiple potentially asynchronous kernels on different parts of the
  // array
  for  (unsigned  region  =  0;  region  <  n;  region++)  {
  q.parallel_for(sycl::range<1>{n},  =  {
  const  int  iShifted  =  i  +  nx  /  n  *  region;
  a[iShifted]  =  region  +  iShifted;
  });
  }

  // Synchronize
  q.wait();

  // Print results
  for  (unsigned  i  =  0;  i  <  nx;  i++)
  printf("a[%d] = %d\n",  i,  a[i]);

  // Free shared memory allocation (Unified Memory)
  sycl::free(a,  q);

  return  0;
} 
```

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

auto  main()  ->  int  {
  unsigned  n  =  5;
  unsigned  nx  =  20;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Non-blocking device queue (requires synchronization)
  using  QueueType  =
  ap::onHost::Queue<ALPAKA_TYPEOF(devAcc),  ap::queueKind::NonBlocking>;
  std::vector<QueueType>  queues;
  for  (unsigned  region  =  0;  region  <  n;  region++)  {
  queues.emplace_back(devAcc.makeQueue(ap::queueKind::nonBlocking));
  }

  // Allocate unified memory that is accessible on host and device
  auto  a  =  ap::onHost::allocUnified<int>(devAcc,  nx);

  // Run element-wise multiplication on device
  for  (unsigned  region  =  0;  region  <  n;  region++)  {
  unsigned  nPerRegion  =  nx  /  n;
  unsigned  regionOffset  =  nPerRegion  *  region;
  ap::onHost::iota<int>(queues[region],  regionOffset,
  a.getSubView(regionOffset,  nx  -  regionOffset));
  }
  // Wait for the device, includes all queues
  ap::onHost::wait(devAcc);

  for  (unsigned  i  =  0;  i  <  nx;  i++)  printf("a[%d] = %d\n",  i,  a[i]);

  return  0;
} 
```

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

struct  IdxAssignKernel  {
  constexpr  void  operator()(ap::onAcc::concepts::Acc  auto  const&  acc,
  ap::concepts::IMdSpan  auto  a,
  unsigned  region,
  unsigned  n)  const  {
  unsigned  nPerRegion  =  a.getExtents().x()  /  n;
  unsigned  regionOffset  =  nPerRegion  *  region;
  for  (auto  [idx]  :
  ap::onAcc::makeIdxMap(acc,  ap::onAcc::worker::threadsInGrid,
  ap::IdxRange{regionOffset,  regionOffset  +  nPerRegion}))  {
  a[idx]  =  idx;
  }
  }
};

auto  main()  ->  int  {
  unsigned  n  =  5;
  unsigned  nx  =  20;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Non-blocking device queue (requires synchronization)
  using  QueueType  =
  ap::onHost::Queue<ALPAKA_TYPEOF(devAcc),  ap::queueKind::NonBlocking>;
  std::vector<QueueType>  queues;
  for  (unsigned  region  =  0;  region  <  n;  region++)  {
  queues.emplace_back(devAcc.makeQueue(ap::queueKind::nonBlocking));
  }

  // Allocate unified memory that is accessible on host and device
  auto  a  =  ap::onHost::allocUnified<int>(devAcc,  nx);

  unsigned  frameExtent  =  32u;
  auto  frameSpec  =
  ap::onHost::FrameSpec{ap::divExZero(nx,  frameExtent),  frameExtent};

  // Run element-wise multiplication on device
  for  (unsigned  region  =  0;  region  <  n;  region++)  {
  queues[region].enqueue(
  frameSpec,  ap::KernelBundle{IdxAssignKernel{},  a,  region,  n});
  }
  // Wait for the device, includes all queues
  ap::onHost::wait(devAcc);

  for  (unsigned  i  =  0;  i  <  nx;  i++)  printf("a[%d] = %d\n",  i,  a[i]);

  return  0;
} 
```

### 聚合操作。

```
#include  <cstdio>
#include  <execution>
#include  <numeric>
#include  <vector>

int  main()  {
  unsigned  n  =  10;

  std::vector<int>  a(n);

  std::iota(a.begin(),  a.end(),  0);  // Fill the array

  // Run reduction on the device
  int  sum  =  std::reduce(std::execution::par_unseq,  a.cbegin(),  a.cend(),  0,
  std::plus<int>{});

  // Print results
  printf("sum = %d\n",  sum);

  return  0;
} 
```

```
#include  <Kokkos_Core.hpp>

int  main(int  argc,  char  *argv[])  {

  // Initialize Kokkos
  Kokkos::initialize(argc,  argv);

  {
  unsigned  n  =  10;

  // Initialize sum variable
  int  sum  =  0;

  // Run sum reduction kernel
  Kokkos::parallel_reduce(
  n,  KOKKOS_LAMBDA(const  int  i,  int  &lsum)  {  lsum  +=  i;  },  sum);

  // Kokkos synchronization
  Kokkos::fence();

  // Print results
  printf("sum = %d\n",  sum);
  }

  // Finalize Kokkos
  Kokkos::finalize();
  return  0;
} 
```

```
// We're using OpenCL C++ API here; there is also C API in <CL/cl.h>
#define CL_TARGET_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110
#include  <CL/cl.hpp>

// For larger kernels, we can store source in a separate file
static  const  std::string  kernel_source  =  R"(
 __kernel void reduce(__global int* sum, __local int* local_mem) {

 // Get work group and work item information
 int gsize = get_global_size(0); // global work size
 int gid = get_global_id(0); // global work item index
 int lsize = get_local_size(0); // local work size
 int lid = get_local_id(0); // local work item index

 // Store reduced item into local memory
 local_mem[lid] = gid; // initialize local memory
 barrier(CLK_LOCAL_MEM_FENCE); // synchronize local memory

 // Perform reduction across the local work group
 for (int s = 1; s < lsize; s *= 2) { // loop over local memory with stride doubling each iteration
 if (lid % (2 * s) == 0 && (lid + s) < lsize) {
 local_mem[lid] += local_mem[lid + s];
 }
 barrier(CLK_LOCAL_MEM_FENCE); // synchronize local memory
 }

 if (lid == 0) { // only one work item per work group
 atomic_add(sum, local_mem[0]); // add partial sum to global sum atomically
 }
 }
  )";

int  main(int  argc,  char  *argv[])  {

  // Initialize OpenCL
  cl::Device  device  =  cl::Device::getDefault();
  cl::Context  context(device);
  cl::CommandQueue  queue(context,  device);

  // Compile OpenCL program for found device
  cl::Program  program(context,  kernel_source);
  program.build({device});
  cl::Kernel  kernel_reduce(program,  "reduce");

  {
  // Set problem dimensions
  unsigned  n  =  10;

  // Initialize sum variable
  int  sum  =  0;

  // Create buffer for sum
  cl::Buffer  buffer(context,  CL_MEM_READ_WRITE  |  CL_MEM_COPY_HOST_PTR,
  sizeof(int),  &sum);

  // Pass arguments to device kernel
  kernel_reduce.setArg(0,  buffer);  // pass buffer to device
  kernel_reduce.setArg(1,  sizeof(int),  NULL);  // allocate local memory

  // Enqueue kernel
  queue.enqueueNDRangeKernel(kernel_reduce,  cl::NullRange,  cl::NDRange(n),
  cl::NullRange);

  // Read result
  queue.enqueueReadBuffer(buffer,  CL_TRUE,  0,  sizeof(int),  &sum);

  // Print result
  printf("sum = %d\n",  sum);
  }

  return  0;
} 
```

```
// We use built-in sycl::reduction mechanism in this example.
// The manual implementation of the reduction kernel can be found in
// the "Non-portable kernel models" chapter.

#include  <sycl/sycl.hpp>

int  main()  {
  sycl::queue  q;
  unsigned  n  =  10;

  // Initialize sum
  int  sum  =  0;
  {
  // Create a buffer for sum to get the reduction results
  sycl::buffer<int>  sum_buf{&sum,  1};

  // Submit a SYCL kernel into a queue
  q.submit(&  {
  // Create temporary object describing variables with reduction semantics
  auto  sum_acc  =  sum_buf.get_access<sycl::access_mode::read_write>(cgh);
  // We can use built-in reduction primitive
  auto  sum_reduction  =  sycl::reduction(sum_acc,  sycl::plus<int>());

  // A reference to the reducer is passed to the lambda
  cgh.parallel_for(
  sycl::range<1>{n},  sum_reduction,
  =  {  reducer.combine(idx[0]);  });
  }).wait();
  // The contents of sum_buf are copied back to sum by the destructor of
  // sum_buf
  }
  // Print results
  printf("sum = %d\n",  sum);
} 
```

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

auto  main()  ->  int  {
  unsigned  n  =  10;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue  queue  =  devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate unified memory that is accessible on host and device
  auto  sum  =  ap::onHost::allocUnified<int>(devAcc,  1);

  // Run element-wise multiplication on device
  ap::onHost::reduce(queue,  0,  sum,  std::plus{},  ap::LinearizedIdxGenerator{n});

  // Print results
  printf("sum = %d\n",  sum[0]);

  return  0;
} 
```

### 带统一内存的并行 for 循环。

```
#include  <algorithm>
#include  <cstdio>
#include  <execution>
#include  <vector>

int  main()  {
  unsigned  n  =  5;

  // Allocate arrays
  std::vector<int>  a(n),  b(n),  c(n);

  // Initialize values
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

  // Run element-wise multiplication on device
  std::transform(std::execution::par_unseq,  a.begin(),  a.end(),  b.begin(),
  c.begin(),  [](int  i,  int  j)  {  return  i  *  j;  });

  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  c[i]);
  }

  return  0;
} 
```

```
#include  <Kokkos_Core.hpp>

int  main(int  argc,  char  *argv[])  {

  // Initialize Kokkos
  Kokkos::initialize(argc,  argv);

  {
  unsigned  n  =  5;

  // Allocate on Kokkos default memory space (Unified Memory)
  int  *a  =  (int  *)Kokkos::kokkos_malloc(n  *  sizeof(int));
  int  *b  =  (int  *)Kokkos::kokkos_malloc(n  *  sizeof(int));
  int  *c  =  (int  *)Kokkos::kokkos_malloc(n  *  sizeof(int));

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

  // Run element-wise multiplication on device
  Kokkos::parallel_for(n,  KOKKOS_LAMBDA(const  int  i)  {  c[i]  =  a[i]  *  b[i];  });

  // Kokkos synchronization
  Kokkos::fence();

  // Print results
  for  (unsigned  i  =  0;  i  <  n;  i++)
  printf("c[%d] = %d\n",  i,  c[i]);

  // Free Kokkos allocation (Unified Memory)
  Kokkos::kokkos_free(a);
  Kokkos::kokkos_free(b);
  Kokkos::kokkos_free(c);
  }

  // Finalize Kokkos
  Kokkos::finalize();
  return  0;
} 
```

```
// We're using OpenCL C API here, since SVM support in C++ API is unstable on
// ROCm
#define CL_TARGET_OPENCL_VERSION 220
#include  <CL/cl.h>
#include  <stdio.h>

// For larger kernels, we can store source in a separate file
static  const  char  *kernel_source  =
  "                                                 \
 __kernel void dot(__global const int *a, __global const int *b, __global int *c) { \
 int i = get_global_id(0);                                                        \
 c[i] = a[i] * b[i];                                                              \
 }                                                                                  \
 ";

int  main(int  argc,  char  *argv[])  {

  // Initialize OpenCL
  cl_platform_id  platform;
  clGetPlatformIDs(1,  &platform,  NULL);
  cl_device_id  device;
  clGetDeviceIDs(platform,  CL_DEVICE_TYPE_GPU,  1,  &device,  NULL);
  cl_context  context  =  clCreateContext(NULL,  1,  &device,  NULL,  NULL,  NULL);
  cl_command_queue  queue  =  clCreateCommandQueue(context,  device,  0,  NULL);

  // Compile OpenCL program for found device.
  cl_program  program  =
  clCreateProgramWithSource(context,  1,  &kernel_source,  NULL,  NULL);
  clBuildProgram(program,  1,  &device,  NULL,  NULL,  NULL);
  cl_kernel  kernel  =  clCreateKernel(program,  "dot",  NULL);

  // Set problem dimensions
  unsigned  n  =  5;

  // Create SVM buffer objects on host side
  int  *a  =  clSVMAlloc(context,  CL_MEM_READ_ONLY,  n  *  sizeof(int),  0);
  int  *b  =  clSVMAlloc(context,  CL_MEM_READ_ONLY,  n  *  sizeof(int),  0);
  int  *c  =  clSVMAlloc(context,  CL_MEM_WRITE_ONLY,  n  *  sizeof(int),  0);

  // Pass arguments to device kernel
  clSetKernelArgSVMPointer(kernel,  0,  a);
  clSetKernelArgSVMPointer(kernel,  1,  b);
  clSetKernelArgSVMPointer(kernel,  2,  c);

  // Create mappings for host and initialize values
  clEnqueueSVMMap(queue,  CL_TRUE,  CL_MAP_WRITE,  a,  n  *  sizeof(int),  0,  NULL,
  NULL);
  clEnqueueSVMMap(queue,  CL_TRUE,  CL_MAP_WRITE,  b,  n  *  sizeof(int),  0,  NULL,
  NULL);
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }
  clEnqueueSVMUnmap(queue,  a,  0,  NULL,  NULL);
  clEnqueueSVMUnmap(queue,  b,  0,  NULL,  NULL);

  size_t  globalSize  =  n;
  clEnqueueNDRangeKernel(queue,  kernel,  1,  NULL,  &globalSize,  NULL,  0,  NULL,
  NULL);

  // Create mapping for host and print results
  clEnqueueSVMMap(queue,  CL_TRUE,  CL_MAP_READ,  c,  n  *  sizeof(int),  0,  NULL,
  NULL);
  for  (unsigned  i  =  0;  i  <  n;  i++)
  printf("c[%d] = %d\n",  i,  c[i]);
  clEnqueueSVMUnmap(queue,  c,  0,  NULL,  NULL);

  // Free SVM buffers
  clSVMFree(context,  a);
  clSVMFree(context,  b);
  clSVMFree(context,  c);

  return  0;
} 
```

```
#include  <sycl/sycl.hpp>

int  main()  {

  sycl::queue  q;
  unsigned  n  =  5;

  // Allocate shared memory (Unified Shared Memory)
  int  *a  =  sycl::malloc_shared<int>(n,  q);
  int  *b  =  sycl::malloc_shared<int>(n,  q);
  int  *c  =  sycl::malloc_shared<int>(n,  q);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

  // Run element-wise multiplication on device
  q.parallel_for(sycl::range<1>{n},  =  {
  c[i]  =  a[i]  *  b[i];
  }).wait();

  // Print results
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  c[i]);
  }

  // Free shared memory allocation (Unified Memory)
  sycl::free(a,  q);
  sycl::free(b,  q);
  sycl::free(c,  q);

  return  0;
} 
```

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

auto  main()  ->  int  {
  unsigned  n  =  5;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue  queue  =  devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate unified memory that is accessible on host and device
  auto  a  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  b  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  c  =  ap::onHost::allocUnified<int>(devAcc,  n);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

  // Run element-wise multiplication on device
  ap::onHost::transform(queue,  c,  std::multiplies{},  a,  b);

  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  c[i]);
  }

  return  0;
} 
```

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

struct  MulKernel  {
  constexpr  void  operator()(ap::onAcc::concepts::Acc  auto  const&  acc,
  ap::concepts::IMdSpan  auto  c,
  ap::concepts::IMdSpan  auto  const  a,
  ap::concepts::IMdSpan  auto  const  b)  const  {
  for  (auto  idx  :  ap::onAcc::makeIdxMap(acc,  ap::onAcc::worker::threadsInGrid,
  ap::IdxRange{c.getExtents()}))  {
  c[idx]  =  a[idx]  *  b[idx];
  }
  }
};

auto  main()  ->  int  {
  unsigned  n  =  5;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue  queue  =  devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate unified memory that is accessible on host and device
  auto  a  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  b  =  ap::onHost::allocUnified<int>(devAcc,  n);
  auto  c  =  ap::onHost::allocUnified<int>(devAcc,  n);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

  unsigned  frameExtent  =  32u;
  auto  frameSpec  =
  ap::onHost::FrameSpec{ap::divExZero(n,  frameExtent),  frameExtent};

  // Run element-wise multiplication on device
  queue.enqueue(frameSpec,  ap::KernelBundle{MulKernel{},  c,  a,  b});

  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  c[i]);
  }

  return  0;
} 
```

### 带 GPU 缓冲区的并行 for 循环。

```
#include  <Kokkos_Core.hpp>

int  main(int  argc,  char  *argv[])  {

  // Initialize Kokkos
  Kokkos::initialize(argc,  argv);

  {
  unsigned  n  =  5;

  // Allocate space for 5 ints on Kokkos host memory space
  Kokkos::View<int  *,  Kokkos::HostSpace>  h_a("h_a",  n);
  Kokkos::View<int  *,  Kokkos::HostSpace>  h_b("h_b",  n);
  Kokkos::View<int  *,  Kokkos::HostSpace>  h_c("h_c",  n);

  // Allocate space for 5 ints on Kokkos default memory space (eg, GPU memory)
  Kokkos::View<int  *>  a("a",  n);
  Kokkos::View<int  *>  b("b",  n);
  Kokkos::View<int  *>  c("c",  n);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  h_a[i]  =  i;
  h_b[i]  =  1;
  }

  // Copy from host to device
  Kokkos::deep_copy(a,  h_a);
  Kokkos::deep_copy(b,  h_b);

  // Run element-wise multiplication on device
  Kokkos::parallel_for(n,  KOKKOS_LAMBDA(const  int  i)  {  c[i]  =  a[i]  *  b[i];  });

  // Copy from device to host
  Kokkos::deep_copy(h_c,  c);

  // Print results
  for  (unsigned  i  =  0;  i  <  n;  i++)
  printf("c[%d] = %d\n",  i,  h_c[i]);
  }

  // Finalize Kokkos
  Kokkos::finalize();
  return  0;
} 
```

```
// We're using OpenCL C++ API here; there is also C API in <CL/cl.h>
#define CL_TARGET_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110
#include  <CL/cl.hpp>

// For larger kernels, we can store source in a separate file
static  const  std::string  kernel_source  =  R"(
 __kernel void dot(__global const int *a, __global const int *b, __global int *c) {
 int i = get_global_id(0);
 c[i] = a[i] * b[i];
 }
  )";

int  main(int  argc,  char  *argv[])  {

  // Initialize OpenCL
  cl::Device  device  =  cl::Device::getDefault();
  cl::Context  context(device);
  cl::CommandQueue  queue(context,  device);

  // Compile OpenCL program for found device.
  cl::Program  program(context,  kernel_source);
  program.build({device});
  cl::Kernel  kernel_dot(program,  "dot");

  {
  // Set problem dimensions
  unsigned  n  =  5;

  std::vector<int>  a(n),  b(n),  c(n);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a[i]  =  i;
  b[i]  =  1;
  }

  // Create buffers and copy input data to device.
  cl::Buffer  dev_a(context,  CL_MEM_READ_ONLY  |  CL_MEM_COPY_HOST_PTR,
  n  *  sizeof(int),  a.data());
  cl::Buffer  dev_b(context,  CL_MEM_READ_ONLY  |  CL_MEM_COPY_HOST_PTR,
  n  *  sizeof(int),  b.data());
  cl::Buffer  dev_c(context,  CL_MEM_WRITE_ONLY,  n  *  sizeof(int));

  // Pass arguments to device kernel
  kernel_dot.setArg(0,  dev_a);
  kernel_dot.setArg(1,  dev_b);
  kernel_dot.setArg(2,  dev_c);

  // We don't need to apply any offset to thread IDs
  queue.enqueueNDRangeKernel(kernel_dot,  cl::NullRange,  cl::NDRange(n),
  cl::NullRange);

  // Read result
  queue.enqueueReadBuffer(dev_c,  CL_TRUE,  0,  n  *  sizeof(int),  c.data());

  // Print results
  for  (unsigned  i  =  0;  i  <  n;  i++)
  printf("c[%d] = %d\n",  i,  c[i]);
  }

  return  0;
} 
```

```
#include  <sycl/sycl.hpp>

int  main()  {

  sycl::queue  q;
  unsigned  n  =  5;

  // Allocate space for 5 ints
  auto  a_buf  =  sycl::buffer<int>(sycl::range<1>(n));
  auto  b_buf  =  sycl::buffer<int>(sycl::range<1>(n));
  auto  c_buf  =  sycl::buffer<int>(sycl::range<1>(n));

  // Initialize values
  // We should use curly braces to limit host accessors' lifetime
  //    and indicate when we're done working with them:
  {
  auto  a_host_acc  =  a_buf.get_host_access();
  auto  b_host_acc  =  b_buf.get_host_access();
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  a_host_acc[i]  =  i;
  b_host_acc[i]  =  1;
  }
  }

  // Submit a SYCL kernel into a queue
  q.submit(&  {
  // Create read accessors over a_buf and b_buf
  auto  a_acc  =  a_buf.get_access<sycl::access_mode::read>(cgh);
  auto  b_acc  =  b_buf.get_access<sycl::access_mode::read>(cgh);
  // Create write accesor over c_buf
  auto  c_acc  =  c_buf.get_access<sycl::access_mode::write>(cgh);
  // Run element-wise multiplication on device
  cgh.parallel_for<class  vec_add>(sycl::range<1>{n},  =  {
  c_acc[i]  =  a_acc[i]  *  b_acc[i];
  });
  });

  // No need to synchronize, creating the accessor for c_buf will do it
  // automatically
  {
  const  auto  c_host_acc  =  c_buf.get_host_access();
  // Print results
  for  (unsigned  i  =  0;  i  <  n;  i++)
  printf("c[%d] = %d\n",  i,  c_host_acc[i]);
  }

  return  0;
} 
```

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

auto  main()  ->  int  {
  unsigned  n  =  5;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue  queue  =  devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate memory that is accessible on host
  auto  h_a  =  ap::onHost::allocHost<int>(n);
  auto  h_b  =  ap::onHost::allocHostLike(h_a);
  auto  h_c  =  ap::onHost::allocHostLike(h_a);

  // Allocate memory on the device and inherit the extents from h_a
  auto  a  =  ap::onHost::allocLike(devAcc,  h_a);
  auto  b  =  ap::onHost::allocLike(devAcc,  h_a);
  auto  c  =  ap::onHost::allocLike(devAcc,  h_a);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  h_a[i]  =  i;
  h_b[i]  =  1;
  }

  // Copy host memory element wise to the device memory
  ap::onHost::memcpy(queue,  a,  h_a);
  ap::onHost::memcpy(queue,  b,  h_b);

  // Run element-wise multiplication on device
  ap::onHost::transform(queue,  c,  std::multiplies{},  a,  b);

  // Copy the device result back to host memory
  ap::onHost::memcpy(queue,  h_c,  c);

  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  h_c[i]);
  }

  return  0;
} 
```

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

struct  MulKernel  {
  constexpr  void  operator()(ap::onAcc::concepts::Acc  auto  const&  acc,
  ap::concepts::IMdSpan  auto  c,
  ap::concepts::IMdSpan  auto  const  a,
  ap::concepts::IMdSpan  auto  const  b)  const  {
  for  (auto  idx  :  ap::onAcc::makeIdxMap(acc,  ap::onAcc::worker::threadsInGrid,
  ap::IdxRange{c.getExtents()}))  {
  c[idx]  =  a[idx]  *  b[idx];
  }
  }
};

auto  main()  ->  int  {
  unsigned  n  =  5;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue  queue  =  devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate memory that is accessible on host
  auto  h_a  =  ap::onHost::allocHost<int>(n);
  auto  h_b  =  ap::onHost::allocHostLike(h_a);
  auto  h_c  =  ap::onHost::allocHostLike(h_a);

  // allocate memory on the device and inherit the extents from a
  auto  a  =  ap::onHost::allocLike(devAcc,  h_a);
  auto  b  =  ap::onHost::allocLike(devAcc,  h_a);
  auto  c  =  ap::onHost::allocLike(devAcc,  h_a);

  // Initialize values on host
  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  h_a[i]  =  i;
  h_b[i]  =  1;
  }

  // Copy host memory element wise to the device memory
  ap::onHost::memcpy(queue,  a,  h_a);
  ap::onHost::memcpy(queue,  b,  h_b);

  unsigned  frameExtent  =  32u;
  auto  frameSpec  =
  ap::onHost::FrameSpec{ap::divExZero(n,  frameExtent),  frameExtent};

  // Run element-wise multiplication on device
  queue.enqueue(frameSpec,  ap::KernelBundle{MulKernel{},  c,  a,  b});

  // Copy the device result back to host memory
  ap::onHost::memcpy(queue,  h_c,  c);

  for  (unsigned  i  =  0;  i  <  n;  i++)  {
  printf("c[%d] = %d\n",  i,  h_c[i]);
  }

  return  0;
} 
```

### 异步并行内核。

```
#include  <Kokkos_Core.hpp>

int  main(int  argc,  char  *argv[])  {

  // Initialize Kokkos
  Kokkos::initialize(argc,  argv);

  {
  unsigned  n  =  5;
  unsigned  nx  =  20;

  // Allocate on Kokkos default memory space (Unified Memory)
  int  *a  =  (int  *)Kokkos::kokkos_malloc(nx  *  sizeof(int));

  // Create 'n' execution space instances (maps to streams in CUDA/HIP)
  auto  ex  =  Kokkos::Experimental::partition_space(
  Kokkos::DefaultExecutionSpace(),  1,  1,  1,  1,  1);

  // Launch 'n' potentially asynchronous kernels
  // Each kernel has their own execution space instances
  for  (unsigned  region  =  0;  region  <  n;  region++)  {
  Kokkos::parallel_for(
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
  ex[region],  nx  /  n  *  region,  nx  /  n  *  (region  +  1)),
  KOKKOS_LAMBDA(const  int  i)  {  a[i]  =  region  +  i;  });
  }

  // Sync execution space instances (maps to streams in CUDA/HIP)
  for  (unsigned  region  =  0;  region  <  n;  region++)
  ex[region].fence();

  // Print results
  for  (unsigned  i  =  0;  i  <  nx;  i++)
  printf("a[%d] = %d\n",  i,  a[i]);

  // Free Kokkos allocation (Unified Memory)
  Kokkos::kokkos_free(a);
  }

  // Finalize Kokkos
  Kokkos::finalize();
  return  0;
} 
```

```
// We're using OpenCL C API here, since SVM support in C++ API is unstable on
// ROCm
#define CL_TARGET_OPENCL_VERSION 200
#include  <CL/cl.h>
#include  <stdio.h>

// For larger kernels, we can store source in a separate file
static  const  char  *kernel_source  =  "              \
 __kernel void async(__global int *a) { \
 int i = get_global_id(0);            \
 int region = i / get_global_size(0); \
 a[i] = region + i;                   \
 }                                      \
 ";

int  main(int  argc,  char  *argv[])  {
  // Initialize OpenCL
  cl_platform_id  platform;
  clGetPlatformIDs(1,  &platform,  NULL);
  cl_device_id  device;
  clGetDeviceIDs(platform,  CL_DEVICE_TYPE_GPU,  1,  &device,  NULL);
  cl_context  context  =  clCreateContext(NULL,  1,  &device,  NULL,  NULL,  NULL);
  cl_command_queue  queue  =  clCreateCommandQueue(context,  device,  0,  NULL);

  // Compile OpenCL program for found device.
  cl_program  program  =
  clCreateProgramWithSource(context,  1,  &kernel_source,  NULL,  NULL);
  clBuildProgram(program,  1,  &device,  NULL,  NULL,  NULL);
  cl_kernel  kernel  =  clCreateKernel(program,  "async",  NULL);

  // Set problem dimensions
  unsigned  n  =  5;
  unsigned  nx  =  20;

  // Create SVM buffer objects on host side
  int  *a  =  clSVMAlloc(context,  CL_MEM_WRITE_ONLY,  nx  *  sizeof(int),  0);

  // Pass arguments to device kernel
  clSetKernelArgSVMPointer(kernel,  0,  a);

  // Launch multiple potentially asynchronous kernels on different parts of the
  // array
  for  (unsigned  region  =  0;  region  <  n;  region++)  {
  size_t  offset  =  (nx  /  n)  *  region;
  size_t  size  =  nx  /  n;
  clEnqueueNDRangeKernel(queue,  kernel,  1,  &offset,  &size,  NULL,  0,  NULL,
  NULL);
  }

  // Create mapping for host and print results
  clEnqueueSVMMap(queue,  CL_TRUE,  CL_MAP_READ,  a,  nx  *  sizeof(int),  0,  NULL,
  NULL);
  for  (unsigned  i  =  0;  i  <  nx;  i++)
  printf("a[%d] = %d\n",  i,  a[i]);
  clEnqueueSVMUnmap(queue,  a,  0,  NULL,  NULL);

  // Free SVM buffers
  clSVMFree(context,  a);

  return  0;
} 
```

```
#include  <sycl/sycl.hpp>

int  main()  {

  sycl::queue  q;
  unsigned  n  =  5;
  unsigned  nx  =  20;

  // Allocate shared memory (Unified Shared Memory)
  int  *a  =  sycl::malloc_shared<int>(nx,  q);

  // Launch multiple potentially asynchronous kernels on different parts of the
  // array
  for  (unsigned  region  =  0;  region  <  n;  region++)  {
  q.parallel_for(sycl::range<1>{n},  =  {
  const  int  iShifted  =  i  +  nx  /  n  *  region;
  a[iShifted]  =  region  +  iShifted;
  });
  }

  // Synchronize
  q.wait();

  // Print results
  for  (unsigned  i  =  0;  i  <  nx;  i++)
  printf("a[%d] = %d\n",  i,  a[i]);

  // Free shared memory allocation (Unified Memory)
  sycl::free(a,  q);

  return  0;
} 
```

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

auto  main()  ->  int  {
  unsigned  n  =  5;
  unsigned  nx  =  20;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Non-blocking device queue (requires synchronization)
  using  QueueType  =
  ap::onHost::Queue<ALPAKA_TYPEOF(devAcc),  ap::queueKind::NonBlocking>;
  std::vector<QueueType>  queues;
  for  (unsigned  region  =  0;  region  <  n;  region++)  {
  queues.emplace_back(devAcc.makeQueue(ap::queueKind::nonBlocking));
  }

  // Allocate unified memory that is accessible on host and device
  auto  a  =  ap::onHost::allocUnified<int>(devAcc,  nx);

  // Run element-wise multiplication on device
  for  (unsigned  region  =  0;  region  <  n;  region++)  {
  unsigned  nPerRegion  =  nx  /  n;
  unsigned  regionOffset  =  nPerRegion  *  region;
  ap::onHost::iota<int>(queues[region],  regionOffset,
  a.getSubView(regionOffset,  nx  -  regionOffset));
  }
  // Wait for the device, includes all queues
  ap::onHost::wait(devAcc);

  for  (unsigned  i  =  0;  i  <  nx;  i++)  printf("a[%d] = %d\n",  i,  a[i]);

  return  0;
} 
```

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

struct  IdxAssignKernel  {
  constexpr  void  operator()(ap::onAcc::concepts::Acc  auto  const&  acc,
  ap::concepts::IMdSpan  auto  a,
  unsigned  region,
  unsigned  n)  const  {
  unsigned  nPerRegion  =  a.getExtents().x()  /  n;
  unsigned  regionOffset  =  nPerRegion  *  region;
  for  (auto  [idx]  :
  ap::onAcc::makeIdxMap(acc,  ap::onAcc::worker::threadsInGrid,
  ap::IdxRange{regionOffset,  regionOffset  +  nPerRegion}))  {
  a[idx]  =  idx;
  }
  }
};

auto  main()  ->  int  {
  unsigned  n  =  5;
  unsigned  nx  =  20;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Non-blocking device queue (requires synchronization)
  using  QueueType  =
  ap::onHost::Queue<ALPAKA_TYPEOF(devAcc),  ap::queueKind::NonBlocking>;
  std::vector<QueueType>  queues;
  for  (unsigned  region  =  0;  region  <  n;  region++)  {
  queues.emplace_back(devAcc.makeQueue(ap::queueKind::nonBlocking));
  }

  // Allocate unified memory that is accessible on host and device
  auto  a  =  ap::onHost::allocUnified<int>(devAcc,  nx);

  unsigned  frameExtent  =  32u;
  auto  frameSpec  =
  ap::onHost::FrameSpec{ap::divExZero(nx,  frameExtent),  frameExtent};

  // Run element-wise multiplication on device
  for  (unsigned  region  =  0;  region  <  n;  region++)  {
  queues[region].enqueue(
  frameSpec,  ap::KernelBundle{IdxAssignKernel{},  a,  region,  n});
  }
  // Wait for the device, includes all queues
  ap::onHost::wait(devAcc);

  for  (unsigned  i  =  0;  i  <  nx;  i++)  printf("a[%d] = %d\n",  i,  a[i]);

  return  0;
} 
```

### 聚合操作。

```
#include  <cstdio>
#include  <execution>
#include  <numeric>
#include  <vector>

int  main()  {
  unsigned  n  =  10;

  std::vector<int>  a(n);

  std::iota(a.begin(),  a.end(),  0);  // Fill the array

  // Run reduction on the device
  int  sum  =  std::reduce(std::execution::par_unseq,  a.cbegin(),  a.cend(),  0,
  std::plus<int>{});

  // Print results
  printf("sum = %d\n",  sum);

  return  0;
} 
```

```
#include  <Kokkos_Core.hpp>

int  main(int  argc,  char  *argv[])  {

  // Initialize Kokkos
  Kokkos::initialize(argc,  argv);

  {
  unsigned  n  =  10;

  // Initialize sum variable
  int  sum  =  0;

  // Run sum reduction kernel
  Kokkos::parallel_reduce(
  n,  KOKKOS_LAMBDA(const  int  i,  int  &lsum)  {  lsum  +=  i;  },  sum);

  // Kokkos synchronization
  Kokkos::fence();

  // Print results
  printf("sum = %d\n",  sum);
  }

  // Finalize Kokkos
  Kokkos::finalize();
  return  0;
} 
```

```
// We're using OpenCL C++ API here; there is also C API in <CL/cl.h>
#define CL_TARGET_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110
#include  <CL/cl.hpp>

// For larger kernels, we can store source in a separate file
static  const  std::string  kernel_source  =  R"(
 __kernel void reduce(__global int* sum, __local int* local_mem) {

 // Get work group and work item information
 int gsize = get_global_size(0); // global work size
 int gid = get_global_id(0); // global work item index
 int lsize = get_local_size(0); // local work size
 int lid = get_local_id(0); // local work item index

 // Store reduced item into local memory
 local_mem[lid] = gid; // initialize local memory
 barrier(CLK_LOCAL_MEM_FENCE); // synchronize local memory

 // Perform reduction across the local work group
 for (int s = 1; s < lsize; s *= 2) { // loop over local memory with stride doubling each iteration
 if (lid % (2 * s) == 0 && (lid + s) < lsize) {
 local_mem[lid] += local_mem[lid + s];
 }
 barrier(CLK_LOCAL_MEM_FENCE); // synchronize local memory
 }

 if (lid == 0) { // only one work item per work group
 atomic_add(sum, local_mem[0]); // add partial sum to global sum atomically
 }
 }
  )";

int  main(int  argc,  char  *argv[])  {

  // Initialize OpenCL
  cl::Device  device  =  cl::Device::getDefault();
  cl::Context  context(device);
  cl::CommandQueue  queue(context,  device);

  // Compile OpenCL program for found device
  cl::Program  program(context,  kernel_source);
  program.build({device});
  cl::Kernel  kernel_reduce(program,  "reduce");

  {
  // Set problem dimensions
  unsigned  n  =  10;

  // Initialize sum variable
  int  sum  =  0;

  // Create buffer for sum
  cl::Buffer  buffer(context,  CL_MEM_READ_WRITE  |  CL_MEM_COPY_HOST_PTR,
  sizeof(int),  &sum);

  // Pass arguments to device kernel
  kernel_reduce.setArg(0,  buffer);  // pass buffer to device
  kernel_reduce.setArg(1,  sizeof(int),  NULL);  // allocate local memory

  // Enqueue kernel
  queue.enqueueNDRangeKernel(kernel_reduce,  cl::NullRange,  cl::NDRange(n),
  cl::NullRange);

  // Read result
  queue.enqueueReadBuffer(buffer,  CL_TRUE,  0,  sizeof(int),  &sum);

  // Print result
  printf("sum = %d\n",  sum);
  }

  return  0;
} 
```

```
// We use built-in sycl::reduction mechanism in this example.
// The manual implementation of the reduction kernel can be found in
// the "Non-portable kernel models" chapter.

#include  <sycl/sycl.hpp>

int  main()  {
  sycl::queue  q;
  unsigned  n  =  10;

  // Initialize sum
  int  sum  =  0;
  {
  // Create a buffer for sum to get the reduction results
  sycl::buffer<int>  sum_buf{&sum,  1};

  // Submit a SYCL kernel into a queue
  q.submit(&  {
  // Create temporary object describing variables with reduction semantics
  auto  sum_acc  =  sum_buf.get_access<sycl::access_mode::read_write>(cgh);
  // We can use built-in reduction primitive
  auto  sum_reduction  =  sycl::reduction(sum_acc,  sycl::plus<int>());

  // A reference to the reducer is passed to the lambda
  cgh.parallel_for(
  sycl::range<1>{n},  sum_reduction,
  =  {  reducer.combine(idx[0]);  });
  }).wait();
  // The contents of sum_buf are copied back to sum by the destructor of
  // sum_buf
  }
  // Print results
  printf("sum = %d\n",  sum);
} 
```

```
#include  <alpaka/alpaka.hpp>

namespace  ap  =  alpaka;

auto  main()  ->  int  {
  unsigned  n  =  10;

  /* Select a device, possible combinations:
 * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
 * oneApi+amdGpu, oneApi+nvidiaGpu
 */
  auto  devSelector  =
  ap::onHost::makeDeviceSelector(ap::api::hip,  ap::deviceKind::amdGpu);
  ap::onHost::Device  devAcc  =  devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n",  devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue  queue  =  devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate unified memory that is accessible on host and device
  auto  sum  =  ap::onHost::allocUnified<int>(devAcc,  1);

  // Run element-wise multiplication on device
  ap::onHost::reduce(queue,  0,  sum,  std::plus{},  ap::LinearizedIdxGenerator{n});

  // Print results
  printf("sum = %d\n",  sum[0]);

  return  0;
} 
```

## 跨平台可移植性生态系统的优缺点。

### 一般观察。

> +   代码重复量最小化。
> +   
> +   同一段代码可以编译成不同供应商的多种架构。
> +   
> +   与 CUDA 相比，学习资源有限（Stack Overflow、课程材料、文档）。

### 基于 Lambda 的内核模型（Kokkos，SYCL）。

> +   更高的抽象级别。
> +   
> +   初始移植不需要对底层架构有太多了解。
> +   
> +   非常好读的源代码（C++ API）。
> +   
> +   这些模型相对较新，尚未非常流行。

### 基于 Functor 的内核模型（alpaka）。

> +   非常好的可移植性。
> +   
> +   更高的抽象级别。
> +   
> +   低级 API 始终可用，提供更多控制并允许微调。
> +   
> +   适用于主机和内核代码的用户友好 C++ API。
> +   
> +   小型社区和生态系统。

### 独立源代码内核模型（OpenCL）。

> +   非常好的可移植性。
> +   
> +   成熟的生态系统。
> +   
> +   供应商提供的库数量有限。
> +   
> +   低级 API 提供更多控制并允许微调。
> +   
> +   C 和 C++ API 都可用（C++ API 支持得不太好）。
> +   
> +   低级 API 和独立源代码内核模型对用户不太友好。

### C++标准并行性（StdPar，PSTL）。

> +   非常高的抽象级别。
> +   
> +   容易加速依赖于 STL 算法的代码。
> +   
> +   对硬件的控制非常有限。
> +   
> +   编译器的支持正在改善，但远未成熟。

关键点。

+   通用代码组织与不可移植的基于内核的模型类似。

+   只要不使用供应商特定的功能，相同的代码可以在任何 GPU 上运行。

### 一般观察。

> +   代码重复量最小化。
> +   
> +   同一段代码可以编译成不同供应商的多种架构。
> +   
> +   与 CUDA 相比，学习资源有限（Stack Overflow、课程材料、文档）。

### 基于 Lambda 的内核模型（Kokkos，SYCL）。

> +   更高的抽象级别。
> +   
> +   初始移植不需要对底层架构有太多了解。
> +   
> +   非常好读的源代码（C++ API）。
> +   
> +   这些模型相对较新，尚未非常流行。

### 基于 Functor 的内核模型（alpaka）。

> +   非常好的可移植性。
> +   
> +   更高的抽象级别。
> +   
> +   低级 API 始终可用，提供更多控制并允许微调。
> +   
> +   适用于主机和内核代码的用户友好 C++ API。
> +   
> +   小型社区和生态系统。

### 独立源代码内核模型（OpenCL）。

> +   非常好的可移植性。
> +   
> +   成熟的生态系统。
> +   
> +   供应商提供的库数量有限。
> +   
> +   低级 API 提供更多控制并允许微调。
> +   
> +   C 和 C++ API 都可用（C++ API 支持得不太好）。
> +   
> +   低级 API 和分离源代码的内核模型对用户不太友好。

### C++标准并行性（StdPar，PSTL）

> +   非常高的抽象级别。
> +   
> +   容易加速依赖于 STL 算法的代码。
> +   
> +   对硬件的控制非常有限。
> +   
> +   编译器的支持正在改善，但远未成熟。

重点

+   通用代码组织与不可移植的基于内核的模型相似。

+   只要不使用特定供应商的功能，相同的代码可以在任何 GPU 上运行。*

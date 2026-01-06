# 内存分页

> 原文：[`en.algorithmica.org/hpc/cpu-cache/paging/`](https://en.algorithmica.org/hpc/cpu-cache/paging/)

再次考虑这个步进增量循环：

```cpp
const int N = (1 << 13); int a[D * N];   for (int i = 0; i < D * N; i += D)  a[i] += 1; 
```

我们改变步长$D$并按比例增加数组大小，使得总迭代次数$N$保持不变。由于总的内存访问次数也保持不变，对于所有$D \geq 16$，我们应该正好获取$N$个缓存行——或者更确切地说，$64 \cdot N = 2⁶ \cdot 2^{13} = 2^{19}$字节。这正好适合 L2 缓存，无论步长大小如何，吞吐量图应该看起来是平的。

这次，我们考虑一个更广泛的$D$值范围，直到 1024。从大约 256 开始，图表显然不是平的：

![](img/b651bca2cb9d842c46dee30f3adfcf8d.png)

这种异常也是由于缓存系统引起的，尽管标准的 L1-L3 数据缓存与此无关。虚拟内存才是问题所在，特别是*转换后备缓冲区*（TLB），这是一个负责检索虚拟内存页面物理地址的缓存。

在[我的 CPU](https://en.wikichip.org/wiki/amd/microarchitectures/zen_2)上，有两个级别的 TLB：

+   L1 TLB 有 64 个条目，如果页面大小是 4K，那么它可以处理$64 \times 4K = 512K$的活跃内存而不需要访问 L2 TLB。

+   L2 TLB 有 2048 个条目，它可以处理$2048 \times 4K = 8M$的内存而不需要访问页面表。

当$D$等于 256 时，分配了多少内存？你已经猜到了：$8K \times 256 \times 4B = 8M$，正好是 L2 TLB 可以处理的极限。当$D$超过这个值时，一些请求开始被重定向到主页面表，它具有很大的延迟和非常有限的吞吐量，这会阻塞整个计算。

### [#](https://en.algorithmica.org/hpc/cpu-cache/paging/#changing-page-size)更改页面大小

那个无延迟的 8MB 内存似乎是一个非常严格的限制。虽然我们无法改变硬件的特性来提高它，但我们*可以*增加页面大小，这反过来会减少对 TLB 容量的压力。

现代操作系统允许我们全局设置页面大小，也可以为单个分配设置。CPU 只支持一组定义的页面大小——例如，我的 CPU 可以使用 4K 或 2M 页面。另一个典型的页面大小是 1G——它通常只与具有数百 GB RAM 的服务器级硬件相关。超过默认的 4K 的任何东西在 Linux 上被称为*大页*，在 Windows 上称为*大页*。

在 Linux 上，有一个特殊的系统文件控制着大页的分配。以下是让内核在每次分配时都给你大页的方法：

```cpp
$ echo always > /sys/kernel/mm/transparent_hugepage/enabled 
```

以这种方式全局启用大页并不总是好主意，因为它会降低内存粒度，并提高进程消耗的最小内存量——而且一些环境中的进程数量超过了可用的内存兆字节。因此，在该文件中除了`always`和`never`之外，还有一个第三种选择：

```cpp
$ cat /sys/kernel/mm/transparent_hugepage/enabled always [madvise] never 
```

`madvise`是一个特殊的系统调用，允许程序建议内核是否使用大页，这可以用于按需分配大页。如果启用，你可以在 C++中使用它如下：

```cpp
#include <sys/mman.h>  void *ptr = std::aligned_alloc(page_size, array_size); madvise(ptr, array_size, MADV_HUGEPAGE); 
```

只有当内存区域具有相应的对齐方式时，你才能请求使用大页进行分配。

Windows 具有类似的功能。它的内存 API 将这两个功能合并为一个：

```cpp
#include "memoryapi.h"  void *ptr = VirtualAlloc(NULL, array_size,  MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE); 
```

在这两种情况下，`array_size`应该是`page_size`的倍数。

### [#](https://en.algorithmica.org/hpc/cpu-cache/paging/#impact-of-huge-pages)大页的影响

两种分配大页的变体立即使曲线平坦：

![](img/0ec1bff689bffbef53892443318cc4f6.png)

启用大页还可以通过高达 10-15%的幅度提高不适合 L2 缓存的数组的延迟：

![](img/e0a470f7d9fd055e77703ed863939964.png)

通常情况下，当你有任何类型的稀疏读取时，启用大页是一个好主意，因为它们通常略微提高性能，(几乎)永远不会损害性能。

话虽如此，如果可能的话，你不应该依赖大页，因为它们可能由于硬件或计算环境限制而不可用。有许多其他 原因 理由表明，在空间上对数据访问进行分组可能是有益的，这会自动解决分页问题。[← 缓存关联性](https://en.algorithmica.org/hpc/cpu-cache/associativity/)[AoS 和 SoA →](https://en.algorithmica.org/hpc/cpu-cache/aos-soa/)

# 对齐和打包

> 原文：[`en.algorithmica.org/hpc/cpu-cache/alignment/`](https://en.algorithmica.org/hpc/cpu-cache/alignment/)

事实是，内存被划分为 64B 缓存行，这使得操作跨越缓存行边界的字数据变得困难。当您需要检索一些原始类型，例如 32 位整数时，您确实希望它在单个缓存行上定位——这不仅因为检索两个缓存行需要更多的内存带宽，而且在硬件中拼接结果需要宝贵的晶体管空间。

这个方面极大地影响了算法设计和编译器选择数据结构内存布局的方式。

### [#](https://en.algorithmica.org/hpc/cpu-cache/alignment/#aligned-allocation)对齐分配

默认情况下，当您分配某种原始类型的数组时，您保证所有元素的地址都是它们大小的整数倍，这确保了它们只跨越单个缓存行。例如，您保证`int`数组第一个和每个其他元素的地址是 4 字节的倍数（`sizeof int`）。

有时您需要确保这个最小对齐更高。例如，许多 SIMD 应用程序以 32 字节的数据块读取和写入数据，并且这些 32 字节属于同一缓存行对于性能至关重要关键性能。在这种情况下，您可以在定义静态数组变量时使用`alignas`指定符：

```cpp
alignas(32) float a[n]; 
```

要动态分配内存对齐数组，您可以使用`std::aligned_alloc`，它接受对齐值和数组的大小（以字节为单位），并返回分配内存的指针——就像`new`运算符一样：

```cpp
void *a = std::aligned_alloc(32, 4 * n); 
```

您还可以将内存对齐到大于缓存行的大小大于缓存行。唯一的限制是大小参数必须是对齐的整数倍。

您也可以在定义`struct`时使用`alignas`指定符：

```cpp
struct alignas(64) Data {
    // ...
}; 
```

每当分配`Data`的一个实例时，它将位于缓存行的开始处。缺点是结构的实际大小将向上舍入到最接近的 64 字节倍数。这样做是为了，例如，当分配`Data`数组时，不仅仅是第一个元素是正确对齐的。

### [#](https://en.algorithmica.org/hpc/cpu-cache/alignment/#structure-alignment)结构对齐

当我们需要分配一组非均匀元素时，这个问题变得更加复杂，这种情况适用于结构体。而不是像玩俄罗斯方块一样尝试重新排列`struct`的成员，以便每个成员都在单个缓存行内——这并不总是可能的，因为结构体本身不必放在缓存行的开始处——大多数 C/C++编译器也依赖于内存对齐的机制。

结构体对齐同样确保所有成员原始类型（`char`、`int`、`float*`等）的地址都是其大小的倍数，这自动保证了它们中的每一个只跨越一个缓存行。它是通过以下方式实现的：

+   如果需要，通过添加一个可变数量的空白字节来填充每个结构体成员，以满足下一个成员的对齐要求；

+   将结构体本身的对齐要求设置为成员类型对齐要求中的最大值，这样当分配结构体类型的数组或将其用作另一个结构体的成员类型时，所有原始类型的对齐要求都得到满足。

为了更好地理解，考虑以下玩具示例：

```cpp
struct Data {
    char a;
    short b;
    int c;
    char d;
}; 
```

当简洁存储时，此结构体每个实例需要总共$1 + 2 + 4 + 1 = 8$字节，但即使假设整个结构体的对齐为 4 字节（其最大的成员，`int`），只有`a`将合适，而`b`、`c`和`d`不是大小对齐的，并且可能跨越缓存行边界。

为了解决这个问题，编译器插入一些未命名的成员，以确保每个后续成员获得正确的最小对齐：

```cpp
struct Data {
    char a;    // 1 byte
    char x[1]; // 1 byte for the following "short" to be aligned on a 2-byte boundary
    short b;   // 2 bytes 
    int c;     // 4 bytes (largest member, setting the alignment of the whole structure)
    char d;    // 1 byte
    char y[3]; // 3 bytes to make total size of the structure 12 bytes (divisible by 4)
};

// sizeof(Data) = 12
// alignof(Data) = alignof(int) = sizeof(int) = 4 
```

这可能会浪费空间，但可以节省大量的 CPU 周期。这种权衡在大多数情况下是有益的，因此大多数编译器默认启用结构体对齐。

### [#](https://en.algorithmica.org/hpc/cpu-cache/alignment/#optimizing-member-order)优化成员顺序

填充仅在尚未对齐的成员之前或结构体末尾插入。通过改变结构体中成员的顺序，可以改变所需的填充字节数和结构体的总大小。

在前面的例子中，我们可以这样重新排序结构体成员：

```cpp
struct Data {
    int c;
    short b;
    char a;
    char d;
}; 
```

现在，每个结构体成员都对齐，没有任何填充，结构体的尺寸仅为 8 字节。结构体的大小及其性能似乎取决于其成员定义的顺序，这看起来很愚蠢，但这是为了二进制兼容性所必需的。

作为经验法则，从最大的数据类型到最小的数据类型放置你的类型定义——这个贪婪算法在除非你有某些奇怪的、不是 2 的幂次方大小的类型，如 10 字节的`long double`^(1))的情况下是保证有效的。

### [#](https://en.algorithmica.org/hpc/cpu-cache/alignment/#structure-packing)结构体填充

如果你了解自己在做什么，你可以禁用结构体填充，并将数据尽可能紧密地打包。

你必须要求编译器这样做，因为这种功能目前既不是 C 也不是 C++标准的一部分。在 GCC 和 Clang 中，这是通过`packed`属性来实现的：

```cpp
struct __attribute__ ((packed)) Data {
    long long a;
    bool b;
}; 
```

这使得`Data`实例仅占用 9 字节，而不是对齐所需的 16 字节，但代价是可能需要读取两个缓存行来获取其元素。

### [#](https://en.algorithmica.org/hpc/cpu-cache/alignment/#bit-fields)位字段

你还可以使用填充与*位域*结合，这允许你显式地设置成员的位数大小：

```cpp
struct __attribute__ ((packed)) Data {
    char a;     // 1 byte
    int b : 24; // 3 bytes
}; 
```

这种结构在紧凑模式下占用 4 个字节，在填充模式下占用 8 个字节。成员的位数不必是 8 的倍数，整个结构的大小也不必是 8 的倍数。在`Data`数组的相邻元素中，如果字节数不是整数，它们将被“合并”。它还允许你设置一个超过基本类型的宽度，这充当填充——尽管在过程中会抛出一个警告。

这个特性并不那么普遍，因为 CPU 没有 3 字节算术或类似的东西，在加载时必须进行一些低效的字节到字节的转换：

```cpp
int load(char *p) {
    char x = p[0], y = p[1], z = p[2];
    return (x << 16) + (y << 8) + z;
} 
```

当存在非整数字节时，开销甚至更大——它需要通过移位和与掩码来处理。

这个过程可以通过先加载一个 4 字节的`int`，然后使用掩码丢弃其最高位来优化。

```cpp
int load(int *p) {
    int x = *p;
    return x & ((1<<24) - 1);
} 
```

编译器通常不会这样做，因为这在技术上是不合法的：那个第 4 个字节可能位于你无权访问的内存页上，所以即使你打算立即丢弃它，操作系统也不会让你加载它。

* * *

1.  80 位的`long double`至少需要 10 个字节，但确切的格式取决于编译器——例如，它可能将其填充到 12 或 16 个字节以最小化对齐问题（64 位的 GCC 和 Clang 默认使用 16 个字节；您可以通过指定`-mlong-double-64/80/128`或`-m96/128bit-long-double` [选项](https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html)来覆盖此设置）。↩︎ [← 预取](https://en.algorithmica.org/hpc/cpu-cache/prefetching/)[指针替代方案 →](https://en.algorithmica.org/hpc/cpu-cache/pointers/)

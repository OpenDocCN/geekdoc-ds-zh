# 预计算

> 原文：[`en.algorithmica.org/hpc/compilation/precalc/`](https://en.algorithmica.org/hpc/compilation/precalc/)

当编译器可以推断出某个变量不依赖于任何用户提供的数据时，它可以在编译时计算其值，并通过将其嵌入生成的机器代码中将其转换为常量。

这种优化对性能有很大帮助，但它不是 C++标准的一部分，因此编译器**不必**这样做。当编译时计算既难以实现又耗时，编译器可能会放弃这个机会。

### [#](https://en.algorithmica.org/hpc/compilation/precalc/#constant-expressions)常量表达式

为了获得更可靠的解决方案，在现代 C++中，你可以将一个函数标记为`constexpr`；如果它通过传递常量被调用，其值保证在编译时计算：

```cpp
constexpr int fibonacci(int n) {
    if (n <= 2)
        return 1;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

static_assert(fibonacci(10) == 55); 
```

这些函数有一些限制，例如它们只能调用其他`constexpr`函数，不能进行内存分配，但除此之外，它们将“原样”执行。

注意，尽管`constexpr`函数在运行时不会产生任何开销，但它们仍然会增加编译时间，所以至少在某种程度上要关心它们的效率，不要将 NP 完全问题放入其中：

```cpp
constexpr int fibonacci(int n) {
    int a = 1, b = 1;
    while (n--) {
        int c = a + b;
        a = b;
        b = c;
    }
    return b;
} 
```

在早期的 C++标准中，曾经有许多限制，例如你无法在它们中使用任何状态，并且必须依赖递归，所以整个过程更像 Haskell 编程而不是 C++。自从 C++17 以来，你甚至可以使用命令式风格计算静态数组，这对于预计算查找表很有用：

```cpp
struct Precalc {
    int isqrt[1000];

    constexpr Precalc() : isqrt{} {
        for (int i = 0; i < 1000; i++)
            isqrt[i] = int(sqrt(i));
    }
};

constexpr Precalc P;

static_assert(P.isqrt[42] == 6); 
```

注意，当你调用`constexpr`函数并传递非常量时，编译器可能或在编译时计算它们：

```cpp
for (int i = 0; i < 100; i++)
    cout << fibonacci(i) << endl; 
```

在这个例子中，尽管技术上我们执行了固定次数的迭代，并且使用编译时已知的参数调用`fibonacci`，但它们在技术上并不是编译时常量。是否优化这个循环取决于编译器——对于大量计算，它通常选择不进行优化。[← 合约编程](https://en.algorithmica.org/hpc/compilation/contracts/)[../性能分析 →](https://en.algorithmica.org/hpc/profiling/)

# 搜索树

> 原文：[`en.algorithmica.org/hpc/data-structures/b-tree/`](https://en.algorithmica.org/hpc/data-structures/b-tree/)

在上一篇文章中，我们设计和实现了*静态*B-树以加速排序数组中的二分搜索。在其最后一节中，我们简要讨论了如何使它们*动态*，同时保留从 S+树中获得的性能提升，并通过在 S+树的内部节点中添加和跟踪显式指针来验证我们的预测。

在本文中，我们继续探讨那个命题，并设计了一个功能最少的整数键搜索树，实现了相对于`std::set`的 18x/8x 速度提升，相对于`absl::btree`的`lower_bound`和`insert`查询分别达到 7x/2x 速度提升——同时仍有很大的改进空间。

该结构的内存开销大约为 32 位整数的 30%，最终实现少于 150 行 C++代码[under 150 lines of C++](https://github.com/sslotin/amh-code/blob/main/b-tree/btree-final.cc)。它可以很容易地推广到其他算术类型和小/固定长度的字符串，如哈希、国家代码和股票符号。

## [#](https://en.algorithmica.org/hpc/data-structures/b-tree/#b-tree)B-树

与我们在其他案例研究中通常所做的微小增量改进不同，在本文中，我们将仅实现一个名为*B-树*的数据结构，它基于 B+树，但有几个小的不同之处：

+   B-树中的节点不存储指针或任何元数据，除了指向内部节点子节点的指针（而 B+树的叶子节点存储指向下一个叶子节点的指针）。这使得我们能够完美地将键放置在叶子节点上的缓存行中。

+   我们将键$i$定义为子节点$i$的子树中的*最大*键，而不是子节点$(i + 1)$的子树中的*最小*键。这使得我们在达到叶子节点后不需要获取任何其他节点（在 B+树中，叶子节点中的所有键可能都小于搜索键，因此我们需要转到下一个叶子节点以获取其第一个元素）。

我们还使用节点大小$B=32$，这比典型值小。它不是$16$的原因，因为对于 S+树来说是最优的，是因为我们还有与获取指针相关的额外开销，而通过减少树的高度约 20%带来的好处超过了处理每个节点中两倍元素的成本，还因为这也提高了需要平均每$\frac{B}{2}$次插入时执行昂贵节点分割的`insert`查询的运行时间。

### [#](https://en.algorithmica.org/hpc/data-structures/b-tree/#memory-layout)内存布局

尽管从软件工程的角度来看这可能不是最佳方法，但我们将简单地在一个大预分配数组中存储整个树，而不区分叶节点和内部节点：

```cpp
const int R = 1e8;
alignas(64) int tree[R]; 
```

我们还预先用无穷大填充这个数组，以简化实现：

```cpp
for (int i = 0; i < R; i++)
    tree[i] = INT_MAX; 
```

（一般来说，与使用`new`的`std::set`或其他结构比较在技术上是不诚实的，但内存分配和初始化在这里不是瓶颈，所以这不会显著影响评估。）

两种节点类型都按顺序以排序顺序存储它们的键，并通过数组中第一个键的索引来识别：

+   叶节点最多有$(B - 1)$个键，但用无穷大填充到$B$个元素。

+   内部节点最多有$(B - 2)$个键填充到$B$个元素，以及最多$(B - 1)$个子节点索引，这些索引也填充到$B$个元素。

这些设计决策并非任意：

+   填充确保叶节点恰好占用 2 个缓存行，内部节点恰好占用 4 个缓存行。

+   我们特别使用索引而不是指针来节省缓存空间，并使使用 SIMD 移动它们更快。

    （从现在起，我们将“指针”和“索引”互换使用。）

+   尽管索引存储在单独的缓存行中，我们仍然将索引存储在键的后面，因为我们有原因。

+   我们故意在叶节点中“浪费”一个数组单元，在内部节点中浪费$2+1=3$个单元，因为我们需要它们在节点分割期间存储临时结果。

初始时，我们只有一个作为根的空叶节点：

```cpp
const int B = 32;

int root = 0;   // where the keys of the root start
int n_tree = B; // number of allocated array cells
int H = 1;      // current tree height 
```

要“分配”一个新节点，如果它是一个叶节点，我们只需将`n_tree`增加$B$，如果是内部节点，则增加$2 B$。

由于新节点只能通过分割一个完整节点来创建，除了根节点之外，每个节点至少会有一半是满的。这意味着我们需要为每个整数元素分配 4 到 8 个字节（内部节点将贡献大约 1/16 的数量），前者是在插入顺序时的情况，后者是在输入具有对抗性时的情况。当查询均匀分布时，节点平均大约有 75%的容量，相当于每个元素大约 5.2 个字节。

与基于指针的二元树相比，B 树在内存效率上非常高。例如，`std::set`至少需要三个指针（左子树、右子树和父节点），单独就花费了$3 \times 8 = 24$字节，再加上至少另外$8$字节来存储键和元信息，这是由于结构填充。

### [#](https://en.algorithmica.org/hpc/data-structures/b-tree/#searching)搜索

当超过 90%的操作是查找时，这是一个非常常见的场景，即使不是这种情况，每个其他树操作通常也以定位键开始，所以我们将从实现和优化搜索开始。

当我们实现 S-trees 时，由于混合/打包指令的复杂性，我们最终以排列顺序存储键。对于*动态树*问题，以排列顺序存储键会使插入的实现变得更加困难，因此我们将改变方法。

考虑到在排序数组中找到元素 `x` 的潜在位置的另一种方法是“第一个不小于 `x` 的元素的索引”，而不是“小于 `x` 的元素的数量。”这个观察产生了以下想法：将键与 `x` 进行比较，将向量掩码聚合到一个 32 位掩码中（其中每个位可以对应任何元素，只要映射是双射的），然后调用 `popcnt`，返回小于 `x` 的元素数量。

这个技巧让我们可以有效地执行局部搜索，而不需要任何洗牌操作：

```cpp
typedef __m256i reg;

reg cmp(reg x, int *node) {
    reg y = _mm256_load_si256((reg*) node);
    return _mm256_cmpgt_epi32(x, y);
}

// returns how many keys are less than x
unsigned rank32(reg x, int *node) {
    reg m1 = cmp(x, node);
    reg m2 = cmp(x, node + 8);
    reg m3 = cmp(x, node + 16);
    reg m4 = cmp(x, node + 24);

    // take lower 16 bits from m1/m3 and higher 16 bits from m2/m4
    m1 = _mm256_blend_epi16(m1, m2, 0b01010101);
    m3 = _mm256_blend_epi16(m3, m4, 0b01010101);
    m1 = _mm256_packs_epi16(m1, m3); // can also use blendv here, but packs is simpler

    unsigned mask = _mm256_movemask_epi8(m1);
    return __builtin_popcount(mask);    
} 
```

注意，由于这个程序，我们必须用无穷大填充“键区域”，这阻止我们在空出的单元格中存储元数据（除非我们也愿意在加载 SIMD 通道时花费几个周期来屏蔽它）。

现在，要实现 `lower_bound`，我们可以像在 S+ 树中做的那样遍历树，但在计算子节点编号后获取指针：

```cpp
int lower_bound(int _x) {
    unsigned k = root;
    reg x = _mm256_set1_epi32(_x);

    for (int h = 0; h < H - 1; h++) {
        unsigned i = rank32(x, &tree[k]);
        k = tree[k + B + i];
    }

    unsigned i = rank32(x, &tree[k]);

    return tree[k + i];
} 
```

实现搜索很简单，并且不会引入太多开销。困难的部分是实现插入。

### [#](https://en.algorithmica.org/hpc/data-structures/b-tree/#insertion)插入

在一方面，正确实现插入需要大量的代码，但另一方面，其中大部分代码执行频率很低，所以我们不必太关心其性能。通常，我们只需要到达叶节点（我们已经知道了如何做到这一点）并插入一个新的键，将一些键的子串向右移动一个位置。偶尔，我们还需要分割节点和/或更新一些祖先，但这相对较少，所以让我们首先关注最常见的执行路径。

要将一个键插入到 $(B - 1)$ 个排序元素的数组中，我们可以将它们加载到向量寄存器中，然后使用一个预计算的掩码将它们向右移动一个位置，该掩码告诉哪些元素需要写入给定的 `i`：

```cpp
struct Precalc {
    alignas(64) int mask[B][B];

    constexpr Precalc() : mask{} {
        for (int i = 0; i < B; i++)
            for (int j = i; j < B - 1; j++)
                // everything from i to B - 2 inclusive needs to be moved
                mask[i][j] = -1;
    }
};

constexpr Precalc P;

void insert(int *node, int i, int x) {
    // need to iterate right-to-left to not overwrite the first element of the next lane
    for (int j = B - 8; j >= 0; j -= 8) {
        // load the keys
        reg t = _mm256_load_si256((reg*) &node[j]);
        // load the corresponding mask
        reg mask = _mm256_load_si256((reg*) &P.mask[i][j]);
        // mask-write them one position to the right
        _mm256_maskstore_epi32(&node[j + 1], mask, t);
    }
    node[i] = x; // finally, write the element itself
} 
```

这种 constexpr 魔法是我们使用的唯一 C++ 功能。

有其他方法可以做到这一点，其中一些可能更高效，但我们现在将停止在这里。

当我们分割一个节点时，需要将一半的键移动到另一个节点，因此让我们编写另一个原始操作来完成它：

```cpp
// move the second half of a node and fill it with infinities
void move(int *from, int *to) {
    const reg infs = _mm256_set1_epi32(INT_MAX);
    for (int i = 0; i < B / 2; i += 8) {
        reg t = _mm256_load_si256((reg*) &from[B / 2 + i]);
        _mm256_store_si256((reg*) &to[i], t);
        _mm256_store_si256((reg*) &from[B / 2 + i], infs);
    }
} 
```

实现了这两个向量函数后，我们现在可以非常小心地实现插入：

```cpp
void insert(int _x) {
    // the beginning of the procedure is the same as in lower_bound,
    // except that we save the path in case we need to update some of our ancestors
    unsigned sk[10], si[10]; // k and i on each iteration
    //           ^------^ We assume that the tree height does not exceed 10
    //                    (which would require at least 16^10 elements)

    unsigned k = root;
    reg x = _mm256_set1_epi32(_x);

    for (int h = 0; h < H - 1; h++) {
        unsigned i = rank32(x, &tree[k]);

        // optionally update the key i right away
        tree[k + i] = (_x > tree[k + i] ? _x : tree[k + i]);
        sk[h] = k, si[h] = i; // and save the path

        k = tree[k + B + i];
    }

    unsigned i = rank32(x, &tree[k]);

    // we can start computing the is-full check before insertion completes
    bool filled  = (tree[k + B - 2] != INT_MAX);

    insert(tree + k, i, _x);

    if (filled) {
        // the node needs to be split, so we create a new leaf node
        move(tree + k, tree + n_tree);

        int v = tree[k + B / 2 - 1]; // new key to be inserted
        int p = n_tree;              // pointer to the newly created node

        n_tree += B;

        for (int h = H - 2; h >= 0; h--) {
            // ascend and repeat until we reach the root or find a the node is not split
            k = sk[h], i = si[h];

            filled = (tree[k + B - 3] != INT_MAX);

            // the node already has a correct key (the right one)
            //                  and a correct pointer (the left one)
            insert(tree + k,     i,     v);
            insert(tree + k + B, i + 1, p);

            if (!filled)
                return; // we're done

            // create a new internal node
            move(tree + k,     tree + n_tree);     // move keys
            move(tree + k + B, tree + n_tree + B); // move pointers

            v = tree[k + B / 2 - 1];
            tree[k + B / 2 - 1] = INT_MAX;

            p = n_tree;
            n_tree += 2 * B;
        }

        // if reach here, this means we've reached the root,
        // and it was split into two, so we need a new root
        tree[n_tree] = v;

        tree[n_tree + B] = root;
        tree[n_tree + B + 1] = p;

        root = n_tree;
        n_tree += 2 * B;
        H++;
    }
} 
```

虽然存在许多低效之处，但幸运的是，`if (filled)` 的主体执行频率非常低——大约每 $\frac{B}{2}$ 次插入——因此插入性能并不是我们的首要任务，所以我们只需将其保留即可。

## [#](https://en.algorithmica.org/hpc/data-structures/b-tree/#evaluation)评估

我们只实现了 `insert` 和 `lower_bound`，因此这是我们将会测量的内容。

我们希望评估过程花费合理的时间，因此我们的基准是一个在两个步骤之间交替的循环：

+   将结构大小从 $1.17^k$ 增加到 $1.17^{k+1}$，通过单个 `insert` 操作并测量所需时间。

+   执行 $10⁶$ 次随机的 `lower_bound` 查询并测量所需时间。

我们从大小 $10⁴$ 开始，结束于 $10⁷$，总共大约有 $50$ 个数据点。我们在 $0, 2^{30})$ 范围内均匀地生成两种查询类型的数据，并在各个阶段独立生成。由于数据生成过程允许重复的键，我们与 `std::multiset` 和 `absl::btree_multiset`^([1) 进行了比较，尽管为了简洁起见，我们仍然将它们称为 `std::set` 和 `absl::btree`。我们还为所有三次运行在系统级别上启用了 hugepages。

B−树的表现与我们最初预测的相符 — 至少对于查找操作：

![](img/26083816f86f5976244114da1efa0dab.png)

相对速度提升随着结构大小的变化而变化 — 相比 STL，速度提升 7-18 倍/3-8 倍，相比 Abseil，速度提升 3-7 倍/1.5-2 倍：

![](img/18c97b4d8a35a1b9df83ca03db53208d.png)

插入操作比 `absl::btree` 快 1.5-2 倍，后者使用标量代码执行所有操作。我最好的猜测是插入操作之所以那么慢，是因为数据依赖性：由于树节点可能会改变，CPU 在完成上一个查询之前不能开始处理下一个查询（两个查询的 真实延迟 大约相等，并且是 `lower_bound` 倒数吞吐量的 ~3 倍）。

![](img/0449323c1e75d28258465c347b9aff02.png)

当结构大小较小时，`lower_bound` 的 倒数吞吐量 以离散的步骤增加：当只有根节点需要访问时，它从 3.5ns 开始，然后增长到 6.5ns（两个节点），然后到 12ns（三个节点），然后达到 L2 缓存（图中未显示）并开始更平滑地增加，但仍然在树高度增加时出现明显的峰值。

有趣的是，即使 B−树只存储单个键，它的性能也优于 `absl::btree`：它在 分支预测错误 上停滞了大约 5ns，而 B−树中的搜索是完全无分支的。

### [#](https://en.algorithmica.org/hpc/data-structures/b-tree/#possible-optimizations)可能的优化

在我们之前的数据结构优化努力中，尽可能多地使变量成为编译时常量有很大帮助：编译器可以将这些常量硬编码到机器代码中，简化算术运算，展开所有循环，并为我们做许多其他好事。

如果我们的树高度是常数，这根本不会是问题，但事实并非如此。它*大部分*是常数：高度很少改变，实际上，在基准测试的约束下，最大高度仅为 6。

我们可以预先编译`insert`和`lower_bound`函数，用于几个不同的编译时常量高度，并在树生长时在这之间切换。C++的惯用方法是使用虚拟函数，但我更喜欢明确地使用原始函数指针，如下所示：

```cpp
void (*insert_ptr)(int);
int (*lower_bound_ptr)(int);

void insert(int x) {
    insert_ptr(x);
}

int lower_bound(int x) {
    return lower_bound_ptr(x);
} 
```

我们现在定义了具有树高度作为参数的模板函数，并在`insert`函数内的`grow-tree`块中，随着树的生长改变指针：

```cpp
template <int H>
void insert_impl(int _x) {
    // ...
}

template <int H>
void insert_impl(int _x) {
    // ...
    if (/* tree grows */) {
        // ...
        insert_ptr = &insert_impl<H + 1>;
        lower_bound_ptr = &lower_bound_impl<H + 1>;
    }
}

template <>
void insert_impl<10>(int x) {
    std::cerr << "This depth was not supposed to be reached" << std::endl;
    exit(1);
} 
```

我尝试了，但使用这种方法并没有获得任何性能提升，但我仍然对这个方法抱有很高的期望，因为编译器可以（理论上）移除`sk`和`si`，完全移除任何临时存储，并且只读取和计算一次，极大地优化了`insert`过程。

插入也可以通过使用更大的块大小来优化，因为节点分裂会变得很少，但这会以更慢的查找为代价。我们还可以尝试为不同层使用不同的节点大小：叶子节点可能应该比内部节点大。

**另一个想法**是在插入时将额外的键移动到兄弟节点，尽可能延迟节点分裂。

其中一种特定的修改方法被称为 B*树。如果当前节点已满，它会将最后一个键移动到下一个节点；当两个节点都满了，它会同时分裂它们，产生三个三分之二满的节点。这减少了内存开销（节点平均将会有六分之五满），增加了分支因子，减少了树的高度，这有助于所有操作。

这种技术甚至可以扩展到，比如说，三到四的分裂，尽管进一步的泛化将付出更慢的`insert`的代价。

**还有另一个想法**是去除（一些）指针。例如，对于大型树，我们可能可以承受一个小的 S+树，用于大约$16 \cdot 17$个元素作为根，每次它不经常改变时，我们从头开始重建它。不幸的是，你不能将其扩展到整个树：我相信某处有一篇论文说，如果我们不进行$\Omega(\sqrt n)$的操作，就无法将动态结构完全隐式化。

我们还可以尝试一些非树数据结构，例如[跳表](https://en.wikipedia.org/wiki/Skip_list)。甚至有人尝试了[向量化的成功尝试](https://doublequan.github.io/)——尽管速度提升并不那么令人印象深刻。我对跳表，特别是，能被改进的希望很低，尽管它可能在并发设置中实现更高的总吞吐量。

### [#](https://en.algorithmica.org/hpc/data-structures/b-tree/#other-operations)其他操作

要*删除*一个键，我们可以用类似的方法通过相同的 mask-store 技巧定位并从节点中移除它。之后，如果节点至少有一半是满的，我们就完成了。否则，我们尝试从下一个兄弟节点借一个键。如果兄弟节点有超过 $\frac{B}{2}$ 个键，我们就附加它的第一个键并将它的键向左移动一个位置。否则，当前节点和下一个节点都少于 $\frac{B}{2}$ 个键，因此我们可以合并它们，之后我们前往父节点并迭代地在那里删除一个键。

我们还可能想要实现的是*迭代*。从`l`到`r`批量加载每个键是一个非常常见的模式——例如，在数据库中的`SELECT abc ORDER BY xyz`类型的查询中——B+树通常在数据层存储指向下一个节点的指针，以便进行此类快速迭代。在 B-树中，由于我们使用的是更小的节点大小，如果我们这样做，可能会遇到指针追踪问题。前往父节点并读取所有它的$B$个指针可能更快，因为它消除了这个问题。因此，祖先节点栈（我们在`insert`中使用的`sk`和`si`数组）可以用作迭代器，甚至可能比在节点中单独存储指针更好。

我们可以轻松实现`std::set`几乎能做的所有事情，但 B-树，就像任何其他 B 树一样，由于指针稳定性的要求，不太可能成为`std::set`的直接替代品：指向元素的指针应该保持有效，除非元素被删除，这在不断分割和合并节点时很难实现。这不仅是对搜索树，也是对大多数数据结构的一个主要问题：同时拥有指针稳定性和高性能几乎是不可能的。

## [#](https://en.algorithmica.org/hpc/data-structures/b-tree/#acknowledgements)致谢

感谢 Google 的[Danila Kutenin](https://danlark.org/)就适用性和在 Abseil 中使用 B 树的意义进行了有意义的讨论。

* * *

1.  如果你认为仅仅比较 Abseil 的 B 树还不够令人信服，[请随意](https://github.com/sslotin/amh-code/tree/main/b-tree) 将你喜欢的搜索树添加到基准测试中。↑ [← 静态 B 树](https://en.algorithmica.org/hpc/data-structures/s-tree/)[段树 →](https://en.algorithmica.org/hpc/data-structures/segment-trees/)

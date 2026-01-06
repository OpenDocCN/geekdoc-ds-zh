# 高斯消元法

> [`www.algorithm-archive.org/contents/gaussian_elimination/gaussian_elimination.html`](https://www.algorithm-archive.org/contents/gaussian_elimination/gaussian_elimination.html)

假设我们有一个方程组，

我们想要解出 \(x\)，\(y\) 和 \(z\)。嗯，一种方法就是使用 *高斯消元法*，你可能在前几节数学课中已经接触过了。

第一步是将方程组转换成一个矩阵，使用每个变量前面的系数，其中每一行对应一个方程，每一列对应一个独立变量，如 \(x\)，\(y\) 或 \(z\)。对于之前的方程组，这可能看起来像这样：

或者更简单地说：

起初，将方程组转换成这样的矩阵似乎没有帮助，让我们换一种方式来考虑。

#### 行阶梯形

而不是上面显示的复杂方程式，想象一下如果系统看起来像这样：

然后，我们就可以解出 \(x\) 并将这个值代入前两个方程，通过一个称为回代的过程来解出 \(y\) 和 \(z\)。在矩阵形式中，这组方程看起来像这样：

这种矩阵形式有一个特定的名称：*行阶梯形*。基本上，任何矩阵都可以被认为是行阶梯形，如果其首项系数或 *主元*（从左到右读取时每行的第一个非零元素）位于上一行的主元右侧。

这创建了一个矩阵，有时看起来像上三角矩阵；然而，这并不意味着所有行阶梯形矩阵都是上三角的。例如，以下所有矩阵都是行阶梯形：

其中的前两个具有合适的维度来解一个方程组；然而，最后两个矩阵分别是欠约束和过约束的，这意味着它们不能为方程组提供适当的解。话虽如此，这并不意味着每个正确形式的矩阵都可以求解。例如，如果你再次将第二个矩阵转换成方程组，最后一行转换成 \(0 = 1\)，这是一个矛盾。这是由于矩阵是奇异的，这个特定方程组没有解。尽管如此，所有这些矩阵都是行阶梯形。

#### *简化行阶梯形*

行阶梯形很好，但如果我们的方程组看起来简单如这样会更好：

然后，我们就可以确切地知道 \(x\)，\(y\) 和 \(z\) 的值，没有任何麻烦！在矩阵形式中，它看起来像这样：

这引入了另一种矩阵配置：***简化行阶梯形***。一个矩阵如果是简化行阶梯形，则满足以下条件：

1.  它是行阶梯形。

1.  每个主元都是 1，并且是它所在列的唯一非零项。

所有以下示例都是简化行阶梯形：

再次，在这些操作中（看起来像单位矩阵的那个），在求解方程组的上下文中是可取的，但将任何矩阵转换为这种形式都会立即给出一个明确的答案：我能否求解我的方程组？

除了求解方程组之外，以这种形式重塑矩阵使得很容易推断矩阵的其他属性，例如其秩——线性无关列的最大数量。在简化行阶梯形中，秩简单地是主元的数量。

目前，我希望动机是清晰的：我们希望将一个矩阵转换为行阶梯形，然后转换为简化行阶梯形，以便使大型方程组变得容易求解，因此我们需要一些方法来完成这个任务。一般来说，术语“高斯消元法”指的是将矩阵转换为行阶梯形的过程，而将行阶梯形矩阵转换为简化行阶梯形的过程称为“高斯-若尔当消元法”。尽管如此，这里的符号有时是不一致的。一些作者使用“高斯消元法”这个术语来包括高斯-若尔当消元法。此外，高斯-若尔当消元的过程有时被称为“回代”，这也很令人困惑，因为该术语也可以用来表示从行阶梯形求解方程组，而不简化到简化行阶梯形。因此，在本章中，我们将使用以下定义：

+   **高斯消元法**：将矩阵转换为行阶梯形的过程

+   **高斯-若尔当消元法**：将行阶梯形矩阵转换为*简化*行阶梯形的过程

+   **回代**：直接求解行阶梯形矩阵的过程，*而不转换为简化行阶梯形*

## **分析法**

高斯消元法本质上是分析性的，可以手动用于小型的方程组；然而，对于大型方程组，这（当然）变得繁琐，我们需要找到一个合适的数值解。因此，我将本节分为两部分。一部分将涵盖分析框架，另一部分将涵盖你可以用你喜欢的编程语言编写的算法。

最后，简化大型方程组归结为你在看似随机的矩阵上玩的一个游戏，你有 3 种可能的移动。你可以：

1.  交换任何两行。

1.  将任何一行乘以一个非零的比例值。

1.  将任何一行加到任何其他行的倍数上。

就这些。在继续之前，我建议你尝试重新创建我们上面制作的行阶梯形矩阵。也就是说，做以下操作：

有很多不同的策略可以用来做这件事，没有一种策略比其他策略更好。一种方法是从上面的行中减去一个倍数，使得主元值下面的所有值都为零。这个过程如果先交换一些行可能会更容易，并且可以对每个主元执行。

在得到行阶梯形矩阵后，下一步是找到简化行阶梯形。换句话说，我们做以下操作：

这里，思路与上面类似，同样适用相同的规则。在这种情况下，我们可能从最右侧的列开始，向上减去而不是向下减去。

## 计算方法

高斯消元法的解析方法可能看起来很简单，但计算方法并不明显地来自于我们之前所玩的游戏。最终，计算方法归结为两个独立的步骤，并且具有复杂度。

作为备注，这个过程遍历提供的矩阵中的所有行。当我们说“当前行”（`curr_row`）时，我们指的是当时我们正在进行的特定行迭代编号，并且像之前一样，“主元”对应于该行中的第一个非零元素。

#### 第一步

对于当前行下方主元列中的每个元素，找到最大值，并将具有最大值的行与当前行交换。此时，*主元*被认为是交换后的最高行中的第一个元素。

例如，在这个情况下，最大值是：

找到这个值之后，我们只需将带有该值的行与当前行交换：

在这种情况下，新的主元是。

在代码中，这个过程可能看起来是这样的：

```
# finding the maximum element for each column
max_index = argmax(abs.(A[row:end,col])) + row-1

# Check to make sure matrix is good!
if (A[max_index, col] == 0)
    println("matrix is singular!")
    continue
end

# swap row with highest value for that column to the top
temp_vector = A[max_index, :]
A[max_index, :] = A[row, :]
A[row, :] = temp_vector 
```

```
// finding the maximum element
for (int i = row + 1; i < row; i++) {
    if (Math.abs(a[i][col]) > Math.abs(a[pivot][col])) {
        pivot = i;
    }
}

if (a[pivot][col] == 0) {
    System.err.println("The matrix is singular");
    continue;
}

if (row != pivot) {
    // Swap the row with the highest valued element
    // with the current row
    swapRow(a, col, pivot);
} 
```

```
void swap_rows(double *a, const size_t i, const size_t pivot,
               const size_t cols) {

    for (size_t j = 0; j < cols; ++j) {
        double tmp = a[i * cols + j];
        a[i * cols + j] = a[pivot * cols + j];
        a[pivot * cols + j] = tmp;
    }
} 
```

```
size_t pivot = row;

for (size_t i = row + 1; i < rows; ++i) {
    if (fabs(a[i * cols + col]) > fabs(a[pivot * cols + col])) {
        pivot = i;
    }
}

if (a[pivot * cols + col] == 0) {
    printf("The matrix is singular.\n");
    continue;
}

if (col != pivot) {
    swap_rows(a, col, pivot, cols);
} 
```

```
 std::size_t pivot = i;

for (std::size_t j = i + 1; j < rows; j++) {
  if (fabs(eqns[j][i]) > fabs(eqns[pivot][i])) pivot = j;
}

if (eqns[pivot][i] == 0.0)
  continue;  // But continuing to simplify the matrix as much as possible

if (i != pivot)  // Swapping the rows if new row with higher maxVals is found
  std::swap(eqns[pivot], eqns[i]);  // C++ swap function 
```

```
swapRows :: Int -> Int -> Matrix a -> Matrix a
swapRows r1 r2 m
  | r1 == r2 = m
  | otherwise =
    m //
    concat [[((r2, c), m ! (r1, c)), ((r1, c), m ! (r2, c))] | c <- [c1 .. cn]]
  where
    ((_, c1), (_, cn)) = bounds m 
```

```
(target, pivot) =
  maximumBy (compare `on` abs . snd) [(k, m ! (k, c)) | k <- [r .. rn]]
m' = swapRows r target m 
```

```
let pivot = row;
for (let i = row + 1; i < rows; ++i) {
  if (Math.abs(a[i][col]) > Math.abs(a[pivot][col])) {
    pivot = i;
  }
}

if (a[pivot][col] === 0) {
  console.log("The matrix is singular.");
  continue;
}

if (col !== pivot) {
  const t = a[col];
  a[col] = a[pivot];
  a[pivot] = t;
} 
```

```
// 1\. find highest value in column below row to be pivot
p, highest := r, 0.
for i, row := range a[r:] {
    if abs := math.Abs(row[c]); abs > highest {
        p = r + i
        highest = abs
    }
}
highest = a[p][c] // correct sign

if highest == 0. {
    if !singular {
        singular = true
        fmt.Println("This matrix is singular.")
    }
    continue
} 
```

```
temp = A[pivot_row, :].copy()
A[pivot_row, :] = A[max_i, :]
A[max_i, :] = temp

# Skip on singular matrix,  not actually a pivot
if A[pivot_row, pivot_col] == 0:
    continue 
```

```
 // find the maximum element for this column
    let mut max_row = k;
    let mut max_value = a[(k, k)].abs();
    for row in (k + 1)..a.rows {
        if max_value < a[(row, k)].abs() {
            max_value = a[(row, k)].abs();
            max_row = row;
        }
    }

    // Check to make sure the matrix is good
    if a[(max_row, k)] == 0.0 {
        println!("Matrix is singular, aborting");
        return;
    }

    // swap the row with the highest value for this kumn to the top
    a.swap_rows(k, max_row);

    // Loop over all remaining rows
    for i in k + 1..a.rows {
        // find the fraction
        let fraction = a[(i, k)] / a[(k, k)];

        // Loop through all columns for that row
        for j in (k + 1)..a.cols {
            // re-evaluate each element
            a[(i, j)] -= a[(k, j)] * fraction;
        }

        // set lower elements to 0
        a[(i, k)] = 0.0;
    }
} 
```

作为备注，如果最大值是  ，则矩阵是奇异的，系统没有唯一解。这很有道理，因为如果一列中的最大值是 0，那么整个列必须是 0，因此当我们把矩阵作为一组方程来读取时，不可能有唯一解。尽管如此，高斯消元法更为通用，即使矩阵作为一组方程不一定可解，我们也可以继续进行。如果你的最终目标是解一组方程，那么在找到  之后，你可以随时退出。

#### 第二步

对于当前主元行下方和主元列内的每一行，找到一个与该列中值与主元值比相对应的分数。之后，从每个相应的行元素中减去当前主元行乘以该分数。这个过程本质上是从每个下方行中减去当前行的最优倍数（类似于上面游戏中的第 3 步）。理想情况下，这应该在当前行主元值下方创建一个 0。

例如，在这个矩阵中，下一行是  ，主元值是 ，所以分数是 。

找到分数后，我们只需减去  ，如下所示：

之后，对其他所有行重复此过程。

下面是它在代码中的样子：

```
# Loop for all remaining rows
for i = (row+1):rows

    # finding fraction
    fraction = A[i,col]/A[row,col]

    # loop through all columns for that row
    for j = (col+1):cols

         # re-evaluate each element
         A[i,j] -= A[row,j]*fraction

    end 
```

```
for (int i = row + 1; i < rows; i++) {
    // finding the inverse
    double scale = a[i][col] / a[row][col];
    // loop through all columns in current row
    for (int j = col + 1; j < cols; j++) {

        // Subtract rows
        a[i][j] -= a[row][j] * scale;
    } 
```

```
for (size_t i = row + 1; i < rows; ++i) {
    double scale = a[i * cols + col] / a[row * cols + col];

    for (size_t j = col + 1; j < cols; ++j) {
        a[i * cols + j] -= a[row * cols + j] * scale;
    } 
```

```
for (std::size_t j = i + 1; j < rows; j++) {
  double scale = eqns[j][i] / eqns[i][i];

  for (std::size_t k = i + 1; k < cols; k++)   // k doesn't start at 0, since
    eqns[j][k] -= scale * eqns[i][k];  // values before from 0 to i
                                       // are already 0
  eqns[j][i] = 0.0;
} 
```

```
subRows ::
     Fractional a
  => (Int, Int) -- pivot location
  -> (Int, Int) -- rows to cover
  -> (Int, Int) -- columns to cover
  -> Matrix a
  -> Matrix a
subRows (r, c) (r1, rn) (c1, cn) m =
  accum
    (-)
    m
    [ ((i, j), m ! (i, c) * m ! (r, j) / m ! (r, c))
    | i <- [r1 .. rn]
    , j <- [c1 .. cn]
    ] 
```

```
| otherwise = go (r + 1, c + 1) $ subRows (r, c) (r + 1, rn) (c, cn) m' 
```

```
for (let i = row + 1; i < rows; ++i) {
  const scale = a[i][col] / a[row][col];

  for (let j = col + 1; j < cols; ++j) {
    a[i][j] -= a[row][j] * scale;
  } 
```

```
for _, row := range a[r+1:] {
    // 3\. find fraction from pivot value
    frac := row[c] / highest

    // 4\. subtract row to set rest of column to zero
    for j := range row {
        row[j] -= frac * a[r][j]
    }

    // 5\. ensure col goes to zero (no float rounding)
    row[c] = 0.
} 
```

```
# Zero out elements below pivot
for r in range(pivot_row + 1,  A.shape[0]):
    # Get fraction
    frac = -A[r, pivot_col] / A[pivot_row, pivot_col]
    # Add rows
    A[r, :] += frac * A[pivot_row, :] 
```

```
// Loop over all remaining rows
for i in k + 1..a.rows {
    // find the fraction
    let fraction = a[(i, k)] / a[(k, k)];

    // Loop through all columns for that row
    for j in (k + 1)..a.cols {
        // re-evaluate each element
        a[(i, j)] -= a[(k, j)] * fraction;
    }

    // set lower elements to 0
    a[(i, k)] = 0.0;
} 
```

#### 总共

当我们把所有东西放在一起时，看起来是这样的：

```
function gaussian_elimination!(A::Array{Float64,2})

    rows = size(A,1)
    cols = size(A,2)

    # Row index
    row = 1

    # Main loop going through all columns
    for col = 1:(cols-1)

        # finding the maximum element for each column
        max_index = argmax(abs.(A[row:end,col])) + row-1

        # Check to make sure matrix is good!
        if (A[max_index, col] == 0)
            println("matrix is singular!")
            continue
        end

        # swap row with highest value for that column to the top
        temp_vector = A[max_index, :]
        A[max_index, :] = A[row, :]
        A[row, :] = temp_vector

        # Loop for all remaining rows
        for i = (row+1):rows

            # finding fraction
            fraction = A[i,col]/A[row,col]

            # loop through all columns for that row
            for j = (col+1):cols

                 # re-evaluate each element
                 A[i,j] -= A[row,j]*fraction

            end

            # Set lower elements to 0
            A[i,col] = 0
        end
        row += 1
    end
end 
```

```
void gaussian_elimination(double *a, const size_t rows, const size_t cols) {
    size_t row = 0;

    for (size_t col = 0; col < cols - 1; ++col) {
        size_t pivot = row;

        for (size_t i = row + 1; i < rows; ++i) {
            if (fabs(a[i * cols + col]) > fabs(a[pivot * cols + col])) {
                pivot = i;
            }
        }

        if (a[pivot * cols + col] == 0) {
            printf("The matrix is singular.\n");
            continue;
        }

        if (col != pivot) {
            swap_rows(a, col, pivot, cols);
        }

        for (size_t i = row + 1; i < rows; ++i) {
            double scale = a[i * cols + col] / a[row * cols + col];

            for (size_t j = col + 1; j < cols; ++j) {
                a[i * cols + j] -= a[row * cols + j] * scale;
            }

            a[i * cols + col] = 0;
        }

        row++;
    }
} 
```

```
void gaussianElimination(std::vector<std::vector<double> > &eqns) {
  // 'eqns' is the matrix, 'rows' is no. of vars
  std::size_t rows = eqns.size(), cols = eqns[0].size();

  for (std::size_t i = 0; i < rows - 1; i++) {
      std::size_t pivot = i;

    for (std::size_t j = i + 1; j < rows; j++) {
      if (fabs(eqns[j][i]) > fabs(eqns[pivot][i])) pivot = j;
    }

    if (eqns[pivot][i] == 0.0)
      continue;  // But continuing to simplify the matrix as much as possible

    if (i != pivot)  // Swapping the rows if new row with higher maxVals is found
      std::swap(eqns[pivot], eqns[i]);  // C++ swap function

    for (std::size_t j = i + 1; j < rows; j++) {
      double scale = eqns[j][i] / eqns[i][i];

      for (std::size_t k = i + 1; k < cols; k++)   // k doesn't start at 0, since
        eqns[j][k] -= scale * eqns[i][k];  // values before from 0 to i
                                           // are already 0
      eqns[j][i] = 0.0;
    }
  }
} 
```

```
swapRows :: Int -> Int -> Matrix a -> Matrix a
swapRows r1 r2 m
  | r1 == r2 = m
  | otherwise =
    m //
    concat [[((r2, c), m ! (r1, c)), ((r1, c), m ! (r2, c))] | c <- [c1 .. cn]]
  where
    ((_, c1), (_, cn)) = bounds m

subRows ::
     Fractional a
  => (Int, Int) -- pivot location
  -> (Int, Int) -- rows to cover
  -> (Int, Int) -- columns to cover
  -> Matrix a
  -> Matrix a
subRows (r, c) (r1, rn) (c1, cn) m =
  accum
    (-)
    m
    [ ((i, j), m ! (i, c) * m ! (r, j) / m ! (r, c))
    | i <- [r1 .. rn]
    , j <- [c1 .. cn]
    ]

gaussianElimination :: (Fractional a, Ord a) => Matrix a -> Matrix a
gaussianElimination mat = go (r1, c1) mat 
```

```
def gaussian_elimination(A):

    pivot_row = 0

    # Go by column
    for pivot_col in range(min(A.shape[0], A.shape[1])):

        # Swap row with highest element in col
        max_i = np.argmax(abs(A[pivot_row:, pivot_col])) + pivot_row

        temp = A[pivot_row, :].copy()
        A[pivot_row, :] = A[max_i, :]
        A[max_i, :] = temp

        # Skip on singular matrix,  not actually a pivot
        if A[pivot_row, pivot_col] == 0:
            continue

        # Zero out elements below pivot
        for r in range(pivot_row + 1,  A.shape[0]):
            # Get fraction
            frac = -A[r, pivot_col] / A[pivot_row, pivot_col]
            # Add rows
            A[r, :] += frac * A[pivot_row, :]

        pivot_row += 1 
```

```
static void gaussianElimination(double[][] a) {
    int row = 0;

    int rows = a.length;
    int cols = a[0].length;

    for (int col = 0; col < cols - 1; col++) {
        int pivot = row;

        // finding the maximum element
        for (int i = row + 1; i < row; i++) {
            if (Math.abs(a[i][col]) > Math.abs(a[pivot][col])) {
                pivot = i;
            }
        }

        if (a[pivot][col] == 0) {
            System.err.println("The matrix is singular");
            continue;
        }

        if (row != pivot) {
            // Swap the row with the highest valued element
            // with the current row
            swapRow(a, col, pivot);
        }

        for (int i = row + 1; i < rows; i++) {
            // finding the inverse
            double scale = a[i][col] / a[row][col];
            // loop through all columns in current row
            for (int j = col + 1; j < cols; j++) {

                // Subtract rows
                a[i][j] -= a[row][j] * scale;
            }

            // Set lower elements to 0
            a[i][col] = 0;
        }
        row++;
    }
} 
```

```
function gaussianElimination(a) {
  const rows = a.length
  const cols = a[0].length
  let row = 0;
  for (let col = 0; col < cols - 1; ++col) {

    let pivot = row;
    for (let i = row + 1; i < rows; ++i) {
      if (Math.abs(a[i][col]) > Math.abs(a[pivot][col])) {
        pivot = i;
      }
    }

    if (a[pivot][col] === 0) {
      console.log("The matrix is singular.");
      continue;
    }

    if (col !== pivot) {
      const t = a[col];
      a[col] = a[pivot];
      a[pivot] = t;
    }

    for (let i = row + 1; i < rows; ++i) {
      const scale = a[i][col] / a[row][col];

      for (let j = col + 1; j < cols; ++j) {
        a[i][j] -= a[row][j] * scale;
      }

      a[i][col] = 0;
    }

    ++row;
  }
  return a;
} 
```

```
func gaussianElimination(a [][]float64) {
    singular := false
    rows := len(a)
    cols := len(a[0])

    for c, r := 0, 0; c < cols && r < rows; c++ {
        // 1\. find highest value in column below row to be pivot
        p, highest := r, 0.
        for i, row := range a[r:] {
            if abs := math.Abs(row[c]); abs > highest {
                p = r + i
                highest = abs
            }
        }
        highest = a[p][c] // correct sign

        if highest == 0. {
            if !singular {
                singular = true
                fmt.Println("This matrix is singular.")
            }
            continue
        }

        // 2\. swap pivot with current row
        if p != r {
            a[r], a[p] = a[p], a[r]
        }

        for _, row := range a[r+1:] {
            // 3\. find fraction from pivot value
            frac := row[c] / highest

            // 4\. subtract row to set rest of column to zero
            for j := range row {
                row[j] -= frac * a[r][j]
            }

            // 5\. ensure col goes to zero (no float rounding)
            row[c] = 0.
        }

        r++
    }
} 
```

```
fn gaussian_elimination(a: &mut Matrix) {
    for k in 0..min(a.cols, a.rows) {
        // find the maximum element for this column
        let mut max_row = k;
        let mut max_value = a[(k, k)].abs();
        for row in (k + 1)..a.rows {
            if max_value < a[(row, k)].abs() {
                max_value = a[(row, k)].abs();
                max_row = row;
            }
        }

        // Check to make sure the matrix is good
        if a[(max_row, k)] == 0.0 {
            println!("Matrix is singular, aborting");
            return;
        }

        // swap the row with the highest value for this kumn to the top
        a.swap_rows(k, max_row);

        // Loop over all remaining rows
        for i in k + 1..a.rows {
            // find the fraction
            let fraction = a[(i, k)] / a[(k, k)];

            // Loop through all columns for that row
            for j in (k + 1)..a.cols {
                // re-evaluate each element
                a[(i, j)] -= a[(k, j)] * fraction;
            }

            // set lower elements to 0
            a[(i, k)] = 0.0;
        }
    }
} 
```

为了清楚起见：如果在过程中发现矩阵是奇异的，那么方程组要么是超定的，要么是欠定的，并且不存在一般解。因此，许多这种方法实现会在发现矩阵没有唯一解时立即停止。在这个实现中，我们允许更一般的情况，并选择在矩阵奇异时简单地输出。如果你打算解一个方程组，那么在你知道没有唯一解时立即停止这个方法是有意义的，所以可能需要对这段代码进行一些小的修改！

那么，我们接下来该做什么呢？嗯，我们继续减少矩阵；然而，这里有两种方法可以做到这一点：

1.  使用高斯-若尔当消元法进一步将矩阵简化为*简化*行阶梯形

1.  如果矩阵允许，直接使用*回代*来解决系统

让我们从高斯-若尔当消元法开始，然后进行回代

## 高斯-若尔当消元法

高斯-若尔当消元法正是我们上面所说的；然而，在这种情况下，我们通常是从下往上工作，而不是从上往下。我们基本上需要找到每一行的主元，并通过将整个行除以主元值将该值设为 1。之后，我们向上减去，直到主元上方的所有值都为 0，然后再从右到左（而不是像之前那样从左到右）移动到下一列。下面是相应的代码：

```
function gauss_jordan_elimination!(A::Array{Float64,2})

    rows = size(A,1)
    cols = size(A,2)

    # After this, we know what row to start on (r-1)
    # to go back through the matrix
    row = 1
    for col = 1:cols-1
        if (A[row, col] != 0)

            # divide row by pivot and leaving pivot as 1
            for i = cols:-1:col
                A[row,i] /= A[row,col]
            end

            # subtract value from above row and set values above pivot to 0
            for i = 1:row-1
                for j = cols:-1:col
                    A[i,j] -= A[i,col]*A[row,j]
                end
            end
            row += 1
        end
    end
end 
```

```
void gauss_jordan(double *a, const size_t cols) {
    size_t row = 0;

    for (size_t col = 0; col < cols - 1; ++col) {
        if (a[row * cols + col] != 0) {
            for (size_t i = cols - 1; i > col - 1; --i) {
                a[row * cols + i] /= a[row * cols + col];
            }

            for (size_t i = 0; i < row; ++i) {
                for (size_t j = cols - 1; j > col - 1; --j) {
                    a[i * cols + j] -= a[i * cols + col] * a[row * cols + j];
                }
            }

            row++;
        }
    }
} 
```

```
void gaussJordan(std::vector<std::vector<double> > &eqns) {
  // 'eqns' is the (Row-echelon) matrix, 'rows' is no. of vars
  std::size_t rows = eqns.size();

  for (std::size_t i = rows - 1; i < rows; i--) {

    if (eqns[i][i] != 0) {

      eqns[i][rows] /= eqns[i][i];
      eqns[i][i] = 1;  // We know that the only entry in this row is 1

      // subtracting rows from below
      for (std::size_t j = i - 1; j < i; j--) {
        eqns[j][rows] -= eqns[j][i] * eqns[i][rows];
        eqns[j][i] = 0;  // We also set all the other values in row to 0 directly
      }
    }
  }
} 
```

```
((r1, c1), (rn, cn)) = bounds mat
go (r, c) m
  | c == cn = m
  | pivot == 0 = go (r, c + 1) m
  | otherwise = go (r + 1, c + 1) $ subRows (r, c) (r + 1, rn) (c, cn) m'
  where
    (target, pivot) =
      maximumBy (compare `on` abs . snd) [(k, m ! (k, c)) | k <- [r .. rn]]
    m' = swapRows r target m 
```

```
# Assumes A is already row echelon form
def gauss_jordan_elimination(A):

    col = 0

    # Scan for pivots
    for row in range(A.shape[0]):
        while col < A.shape[1] and A[row, col] == 0:
            col += 1

        if col >= A.shape[1]:
            continue

        # Set each pivot to one via row scaling
        A[row, :] /= A[row, col]

        # Zero out elements above pivot
        for r in range(row):
            A[r, :] -= A[r, col] * A[row, :] 
```

```
static void gaussJordan(double[][] a) {
    int row = 0;

    int cols = a[0].length;

    for (int col = 0; col < cols - 1; col++) {
        if (a[row][col] != 0) {
            for (int i = cols - 1; i > col - 1; i--) {
                // divide row by pivot so the pivot is set to 1
                a[row][i] /= a[row][col];
            }

            // subtract the value form above row and set values above pivot to 0
            for (int i = 0; i < row; i++) {
                for (int j = cols - 1; j > col - 1; j--) {
                    a[i][j] -= a[i][col] * a[row][j];
                }
            }
            row++;
        }
    }
} 
```

```
function gaussJordan(a) {
  const cols = a[0].length;
  let row = 0;

  for (let col = 0; col < cols - 1; ++col) {
    if (a[row][col] !== 0) {
      for (let i = cols - 1; i > col - 1; --i) {
        a[row][i] /= a[row][col];
      }

      for (let i = 0; i < row; ++i) {
        for (let j = cols - 1; j > col - 1; --j) {
          a[i][j] -= a[i][col] * a[row][j];
        }
      }

      ++row;
    }
  }
} 
```

```
func gaussJordan(a [][]float64) {
    for r := len(a) - 1; r >= 0; r-- {
        // Find pivot col
        p := -1
        for c, cell := range a[r] {
            if cell != 0. {
                p = c
                break
            }
        }
        if p < 0 {
            continue
        }

        // Scale pivot r to 1.
        scale := a[r][p]
        for c := range a[r][p:] {
            a[r][p+c] /= scale
        }
        // Subtract pivot row from each row above
        for _, row := range a[:r] {
            scale = row[p]
            for c, cell := range a[r][p:] {
                row[p+c] -= cell * scale
            }
        }
    }
} 
```

```
fn gauss_jordan(a: &mut Matrix) {
    let mut row = 0;
    for k in 0..(a.cols - 1) {
        if a[(row, k)] != 0.0 {
            for i in (k..a.cols).rev() {
                a[(row, i)] /= a[(row, k)];
            }

            for i in 0..row {
                for j in (k..a.cols).rev() {
                    a[(i, j)] -= a[(i, k)] * a[(row, j)];
                }
            }

            row += 1;
        }
    }
} 
```

作为备注：高斯-若尔当消元法也可以通过遵循相同的程序来找到矩阵的逆，但需要在每个方程的左侧而不是右侧放置一个单位矩阵。这个过程很简单，但在这里不会涉及，仅仅是因为有更快的方法来找到矩阵的逆；然而，如果你想看到这个过程，请告诉我，我可以为了完整性添加它。

## 回代

回代的思想很简单：我们创建一个解矩阵，并通过迭代求解每个变量，将所有之前的变量插入进去。例如，如果我们的矩阵看起来像这样：

我们可以快速求解 \( x \)，然后使用这个结果通过插入 \( x \) 来求解 \( y \)。之后，我们只需要以类似的方式求解 \( z \)。在代码中，这涉及到保持所有替换值的滚动总和，从解列中减去这个总和，然后除以系数变量。在代码中，它看起来像这样：

```
function back_substitution(A::Array{Float64,2})

    rows = size(A,1)
    cols = size(A,2)

    # Creating the solution Vector
    soln = zeros(rows)

    for i = rows:-1:1
        sum = 0.0
        for j = rows:-1:i
            sum += soln[j]*A[i,j]
        end
        soln[i] = (A[i, cols] - sum) / A[i, i]
    end

    return soln
end 
```

```
void back_substitution(const double *a, double *x, const int rows,
                       const int cols) {

    for (int i = rows - 1; i >= 0; --i) {
        double sum = 0.0;

        for (int j = cols - 2; j > i; --j) {
            sum += x[j] * a[i * cols + j];
        }

        x[i] = (a[i * cols + cols - 1] - sum) / a[i * cols + i];
    }
} 
```

```
std::vector<double> backSubs(const std::vector<std::vector<double> > &eqns) {
  // 'eqns' is matrix, 'rows' is no. of variables
  std::size_t rows = eqns.size();

  std::vector<double> ans(rows);
  for (std::size_t i = rows - 1; i < rows; i--) {
    double sum = 0.0;

    for (std::size_t j = i + 1; j < rows; j++) sum += eqns[i][j] * ans[j];

    if (eqns[i][i] != 0)
      ans[i] = (eqns[i][rows] - sum) / eqns[i][i];
    else
      return std::vector<double>(0);
  }
  return ans;
} 
```

```
fn back_substitution(a: &Matrix) -> Vec<f64> {
    let mut soln = vec![0.0; a.rows];

    soln[a.rows - 1] = a[(a.rows - 1, a.cols - 1)] / a[(a.rows - 1, a.cols - 2)];

    for i in (0..a.rows - 1).rev() {
        let mut sum = 0.0;
        for j in (i..a.rows).rev() {
            sum += soln[j] * a[(i, j)];
        }
        soln[i] = (a[(i, a.cols - 1)] - sum) / a[(i, i)];
    }

    soln
} 
```

```
gaussJordan :: (Fractional a, Eq a) => Matrix a -> Matrix a
gaussJordan mat = go (r1, c1) mat
  where
    ((r1, c1), (rn, cn)) = bounds mat
    go (r, c) m
      | c == cn = m 
```

```
# Assumes A has a unique solution and A in row echelon form
def back_substitution(A):

    sol = np.zeros(A.shape[0]).T

    # Go by pivots along diagonal
    for pivot_i in range(A.shape[0] - 1,  -1,  -1):
        s = 0
        for col in range(pivot_i + 1,  A.shape[1] - 1):
            s += A[pivot_i, col] * sol[col]
        sol[pivot_i] = (A[pivot_i, A.shape[1] - 1] - s) / A[pivot_i, pivot_i]

    return sol 
```

```
static double[] backSubstitution(double[][] a) {
    int rows = a.length;
    int cols = a[0].length;

    double[] solution = new double[rows];

    for (int i = rows - 1; i >= 0; i--) {
        double sum = 0;

        for (int j = cols - 2; j > i; j--) {
            sum += solution[j] * a[i][j];
        }
        solution[i] = (a[i][cols - 1] - sum) / a[i][i];
    }
    return solution;
} 
```

```
function backSubstitution(a) {
  const rows = a.length;
  const cols = a[0].length;
  const sol = [];

  for (let i = rows - 1; i >= 0; --i) {

    let sum = 0;
    for (let j = cols - 2; j > i; --j) {
      sum += sol[j] * a[i][j];
    }

    sol[i] = (a[i][cols - 1] - sum) / a[i][i];
  }
  return sol;
} 
```

```
func backSubstitution(a [][]float64) []float64 {
    rows := len(a)
    cols := len(a[0])
    x := make([]float64, rows)
    for r := rows - 1; r >= 0; r-- {
        sum := 0.

        for c := cols - 2; c > r; c-- {
            sum += x[c] * a[r][c]
        }

        x[r] = (a[r][cols-1] - sum) / a[r][r]
    }
    return x
} 
```

## 可视表示

到目前为止，我们一直使用高斯消元作为解方程组的方法；然而，通常有一个更简单的方法来找到类似的解，只需绘制我们矩阵中的每一行。对于两个方程和两个未知数的情况，我们会绘制对应每个方程的两条线，它们交点的位置就是 x 和 y 的解。同样，对于三个方程和三个未知数的情况，我们会绘制三个平面，它们交点的位置就是 x、y 和 z 的解。

那么，如果我们可以直接绘制我们的方程组来找到解，高斯消元有什么意义呢？好吧，当我们开始超越三维时，这个类比很快就会破裂，所以很明显我们需要一种方法来处理更高维的系统。话虽如此，当我们为三维情况绘制矩阵进行高斯消元时，看到发生的事情特别有趣。

<res/GE_vis.mp4>

您的浏览器不支持视频标签。

如上图所示，在三维空间中，这些平面会摇摆直到达到行阶梯形，其中一个平面与 x 轴和 y 轴平行。在这个时候，找到解的 x 坐标是显而易见的，因为它只是平行平面的 y 截距。从那里，随着矩阵移动到简化行阶梯形，矩阵的解释变得更加容易。在这个形式中，解就是相应平面的 x、y 和 z 截距。

这种可视化可能对一些读者来说很明显，但我最初发现它特别有启发性。通过执行高斯消元，我们正在操纵我们的平面，使它们可以一目了然地解释——这正是我们用矩阵解释所做的事情！

## 结论

有了这些，我们就有两种可能的方法来减少我们的方程组并找到解。如果我们确定我们的矩阵不是奇异的并且存在解，那么使用回代来找到我们的解是最快的。如果没有解或者我们正在尝试找到简化行阶梯形矩阵，那么高斯-若尔当消元法是最好的。正如我们一开始所说的，文献中高斯消元的表示法相当模糊，所以我们希望这里提供的定义足够清晰和一致，以涵盖所有情况。

关于接下来会发生什么……好吧，我们将会享受到一场盛宴！上面的算法显然有 3 个`for`循环，其复杂度为 O(n³)，这是非常糟糕的！如果我们能将矩阵简化为特定的**三对角矩阵**，我们实际上可以在 O(n)时间内解出系统。如何做到？我们可以使用一种称为**三对角矩阵算法**（TDMA）的算法，也称为**托马斯算法**。

还有许多其他类似的求解器，我们将在适当的时候介绍。

## 视频解释

这里有一个描述高斯消元的视频：

[`www.youtube-nocookie.com/embed/2tlwSqblrvU`](https://www.youtube-nocookie.com/embed/2tlwSqblrvU)

## 示例代码

```
function gaussian_elimination!(A::Array{Float64,2})

    rows = size(A,1)
    cols = size(A,2)

    # Row index
    row = 1

    # Main loop going through all columns
    for col = 1:(cols-1)

        # finding the maximum element for each column
        max_index = argmax(abs.(A[row:end,col])) + row-1

        # Check to make sure matrix is good!
        if (A[max_index, col] == 0)
            println("matrix is singular!")
            continue
        end

        # swap row with highest value for that column to the top
        temp_vector = A[max_index, :]
        A[max_index, :] = A[row, :]
        A[row, :] = temp_vector

        # Loop for all remaining rows
        for i = (row+1):rows

            # finding fraction
            fraction = A[i,col]/A[row,col]

            # loop through all columns for that row
            for j = (col+1):cols

                 # re-evaluate each element
                 A[i,j] -= A[row,j]*fraction

            end

            # Set lower elements to 0
            A[i,col] = 0
        end
        row += 1
    end
end

function back_substitution(A::Array{Float64,2})

    rows = size(A,1)
    cols = size(A,2)

    # Creating the solution Vector
    soln = zeros(rows)

    for i = rows:-1:1
        sum = 0.0
        for j = rows:-1:i
            sum += soln[j]*A[i,j]
        end
        soln[i] = (A[i, cols] - sum) / A[i, i]
    end

    return soln
end

function gauss_jordan_elimination!(A::Array{Float64,2})

    rows = size(A,1)
    cols = size(A,2)

    # After this, we know what row to start on (r-1)
    # to go back through the matrix
    row = 1
    for col = 1:cols-1
        if (A[row, col] != 0)

            # divide row by pivot and leaving pivot as 1
            for i = cols:-1:col
                A[row,i] /= A[row,col]
            end

            # subtract value from above row and set values above pivot to 0
            for i = 1:row-1
                for j = cols:-1:col
                    A[i,j] -= A[i,col]*A[row,j]
                end
            end
            row += 1
        end
    end
end

function main()
    A = [2. 3 4 6;
         1 2 3 4;
         3 -4 0 10]

    gaussian_elimination!(A)
    println(A)

    gauss_jordan_elimination!(A)
    println(A)

    soln = back_substitution(A)
    println(soln)

end

main() 
```

```
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void swap_rows(double *a, const size_t i, const size_t pivot,
               const size_t cols) {

    for (size_t j = 0; j < cols; ++j) {
        double tmp = a[i * cols + j];
        a[i * cols + j] = a[pivot * cols + j];
        a[pivot * cols + j] = tmp;
    }
}

void gaussian_elimination(double *a, const size_t rows, const size_t cols) {
    size_t row = 0;

    for (size_t col = 0; col < cols - 1; ++col) {
        size_t pivot = row;

        for (size_t i = row + 1; i < rows; ++i) {
            if (fabs(a[i * cols + col]) > fabs(a[pivot * cols + col])) {
                pivot = i;
            }
        }

        if (a[pivot * cols + col] == 0) {
            printf("The matrix is singular.\n");
            continue;
        }

        if (col != pivot) {
            swap_rows(a, col, pivot, cols);
        }

        for (size_t i = row + 1; i < rows; ++i) {
            double scale = a[i * cols + col] / a[row * cols + col];

            for (size_t j = col + 1; j < cols; ++j) {
                a[i * cols + j] -= a[row * cols + j] * scale;
            }

            a[i * cols + col] = 0;
        }

        row++;
    }
}

void back_substitution(const double *a, double *x, const int rows,
                       const int cols) {

    for (int i = rows - 1; i >= 0; --i) {
        double sum = 0.0;

        for (int j = cols - 2; j > i; --j) {
            sum += x[j] * a[i * cols + j];
        }

        x[i] = (a[i * cols + cols - 1] - sum) / a[i * cols + i];
    }
}

void gauss_jordan(double *a, const size_t cols) {
    size_t row = 0;

    for (size_t col = 0; col < cols - 1; ++col) {
        if (a[row * cols + col] != 0) {
            for (size_t i = cols - 1; i > col - 1; --i) {
                a[row * cols + i] /= a[row * cols + col];
            }

            for (size_t i = 0; i < row; ++i) {
                for (size_t j = cols - 1; j > col - 1; --j) {
                    a[i * cols + j] -= a[i * cols + col] * a[row * cols + j];
                }
            }

            row++;
        }
    }
}

int main() {
    double a[3][4] = {{3.0, 2.0, -4.0, 3.0},
                      {2.0, 3.0, 3.0, 15.0},
                      {5.0, -3.0, 1.0, 14.0}};

    gaussian_elimination((double *)a, 3, 4);

    printf("Gaussian elimination:\n");
    for (size_t i = 0; i < 3; ++i) {
        printf("[");
        for (size_t j = 0; j < 4; ++j) {
            printf("%f ", a[i][j]);
        }
        printf("]\n");
    }

    printf("\nGauss-Jordan:\n");

    gauss_jordan((double *)a, 4);

    for (size_t i = 0; i < 3; ++i) {
        printf("[");
        for (size_t j = 0; j < 4; ++j) {
            printf("%f ", a[i][j]);
        }
        printf("]\n");
    }

    printf("\nSolutions are:\n");

    double x[3] = {0, 0, 0};
    back_substitution((double *)a, x, 3, 4);

    printf("(%f,%f,%f)\n", x[0], x[1], x[2]);

    return 0;
} 
```

```
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

void gaussianElimination(std::vector<std::vector<double> > &eqns) {
  // 'eqns' is the matrix, 'rows' is no. of vars
  std::size_t rows = eqns.size(), cols = eqns[0].size();

  for (std::size_t i = 0; i < rows - 1; i++) {
      std::size_t pivot = i;

    for (std::size_t j = i + 1; j < rows; j++) {
      if (fabs(eqns[j][i]) > fabs(eqns[pivot][i])) pivot = j;
    }

    if (eqns[pivot][i] == 0.0)
      continue;  // But continuing to simplify the matrix as much as possible

    if (i != pivot)  // Swapping the rows if new row with higher maxVals is found
      std::swap(eqns[pivot], eqns[i]);  // C++ swap function

    for (std::size_t j = i + 1; j < rows; j++) {
      double scale = eqns[j][i] / eqns[i][i];

      for (std::size_t k = i + 1; k < cols; k++)   // k doesn't start at 0, since
        eqns[j][k] -= scale * eqns[i][k];  // values before from 0 to i
                                           // are already 0
      eqns[j][i] = 0.0;
    }
  }
}

void gaussJordan(std::vector<std::vector<double> > &eqns) {
  // 'eqns' is the (Row-echelon) matrix, 'rows' is no. of vars
  std::size_t rows = eqns.size();

  for (std::size_t i = rows - 1; i < rows; i--) {

    if (eqns[i][i] != 0) {

      eqns[i][rows] /= eqns[i][i];
      eqns[i][i] = 1;  // We know that the only entry in this row is 1

      // subtracting rows from below
      for (std::size_t j = i - 1; j < i; j--) {
        eqns[j][rows] -= eqns[j][i] * eqns[i][rows];
        eqns[j][i] = 0;  // We also set all the other values in row to 0 directly
      }
    }
  }
}

std::vector<double> backSubs(const std::vector<std::vector<double> > &eqns) {
  // 'eqns' is matrix, 'rows' is no. of variables
  std::size_t rows = eqns.size();

  std::vector<double> ans(rows);
  for (std::size_t i = rows - 1; i < rows; i--) {
    double sum = 0.0;

    for (std::size_t j = i + 1; j < rows; j++) sum += eqns[i][j] * ans[j];

    if (eqns[i][i] != 0)
      ans[i] = (eqns[i][rows] - sum) / eqns[i][i];
    else
      return std::vector<double>(0);
  }
  return ans;
}

void printMatrix(const std::vector<std::vector<double> > &matrix) {
  for (std::size_t row = 0; row < matrix.size(); row++) {
    std::cout << "[";

    for (std::size_t col = 0; col < matrix[row].size() - 1; col++)
      std::cout << std::setw(8) << std::fixed << std::setprecision(3)
                << matrix[row][col];

    std::cout << " |" << std::setw(8) << std::fixed << std::setprecision(3)
              << matrix[row].back() << " ]" << std::endl;
  }
}

int main() {
  std::vector<std::vector<double> > equations{
      {2, 3, 4, 6},
      {1, 2, 3, 4},
      {3, -4, 0, 10}};

  std::cout << "Initial matrix:" << std::endl;
  printMatrix(equations);
  std::cout << std::endl;

  gaussianElimination(equations);
  std::cout << "Matrix after gaussian elimination:" << std::endl;
  printMatrix(equations);
  std::cout << std::endl;

  std::vector<double> ans = backSubs(equations);
  std::cout << "Solution from backsubstitution" << std::endl;
  std::cout << "x = " << ans[0] << ", y = " << ans[1] << ", z = " << ans[2]
            << std::endl
            << std::endl;

  gaussJordan(equations);
  std::cout << "Matrix after Gauss Jordan:" << std::endl;
  printMatrix(equations);
  std::cout << std::endl;
} 
```

```
// submitted by jess 3jane

use std::cmp::min;
use std::ops::{Index, IndexMut};

pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    fn new(rows: usize, cols: usize, data: &[f64]) -> Matrix {
        Matrix {
            rows,
            cols,
            data: data.to_vec(),
        }
    }

    fn swap_rows(&mut self, a: usize, b: usize) {
        for col in 0..self.cols {
            self.data.swap(a * self.cols + col, b * self.cols + col);
        }
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;
    fn index(&self, (row, col): (usize, usize)) -> &f64 {
        &self.data[row * self.cols + col]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut f64 {
        &mut self.data[row * self.cols + col]
    }
}

fn gaussian_elimination(a: &mut Matrix) {
    for k in 0..min(a.cols, a.rows) {
        // find the maximum element for this column
        let mut max_row = k;
        let mut max_value = a[(k, k)].abs();
        for row in (k + 1)..a.rows {
            if max_value < a[(row, k)].abs() {
                max_value = a[(row, k)].abs();
                max_row = row;
            }
        }

        // Check to make sure the matrix is good
        if a[(max_row, k)] == 0.0 {
            println!("Matrix is singular, aborting");
            return;
        }

        // swap the row with the highest value for this kumn to the top
        a.swap_rows(k, max_row);

        // Loop over all remaining rows
        for i in k + 1..a.rows {
            // find the fraction
            let fraction = a[(i, k)] / a[(k, k)];

            // Loop through all columns for that row
            for j in (k + 1)..a.cols {
                // re-evaluate each element
                a[(i, j)] -= a[(k, j)] * fraction;
            }

            // set lower elements to 0
            a[(i, k)] = 0.0;
        }
    }
}

fn gauss_jordan(a: &mut Matrix) {
    let mut row = 0;
    for k in 0..(a.cols - 1) {
        if a[(row, k)] != 0.0 {
            for i in (k..a.cols).rev() {
                a[(row, i)] /= a[(row, k)];
            }

            for i in 0..row {
                for j in (k..a.cols).rev() {
                    a[(i, j)] -= a[(i, k)] * a[(row, j)];
                }
            }

            row += 1;
        }
    }
}

fn back_substitution(a: &Matrix) -> Vec<f64> {
    let mut soln = vec![0.0; a.rows];

    soln[a.rows - 1] = a[(a.rows - 1, a.cols - 1)] / a[(a.rows - 1, a.cols - 2)];

    for i in (0..a.rows - 1).rev() {
        let mut sum = 0.0;
        for j in (i..a.rows).rev() {
            sum += soln[j] * a[(i, j)];
        }
        soln[i] = (a[(i, a.cols - 1)] - sum) / a[(i, i)];
    }

    soln
}

fn main() {
    // The example matrix from the text
    let mut a = Matrix::new(
        3,
        4,
        &vec![2.0, 3.0, 4.0, 6.0, 1.0, 2.0, 3.0, 4.0, 3.0, -4.0, 0.0, 10.0],
    );

    gaussian_elimination(&mut a);
    gauss_jordan(&mut a);
    let soln = back_substitution(&a);
    println!("Solution: {:?}", soln);
} 
```

```
import Data.Array
import Data.Function (on)
import Data.List (intercalate, maximumBy)
import Data.Ratio

type Matrix a = Array (Int, Int) a

type Vector a = Array Int a

swapRows :: Int -> Int -> Matrix a -> Matrix a
swapRows r1 r2 m
  | r1 == r2 = m
  | otherwise =
    m //
    concat [[((r2, c), m ! (r1, c)), ((r1, c), m ! (r2, c))] | c <- [c1 .. cn]]
  where
    ((_, c1), (_, cn)) = bounds m

subRows ::
     Fractional a
  => (Int, Int) -- pivot location
  -> (Int, Int) -- rows to cover
  -> (Int, Int) -- columns to cover
  -> Matrix a
  -> Matrix a
subRows (r, c) (r1, rn) (c1, cn) m =
  accum
    (-)
    m
    [ ((i, j), m ! (i, c) * m ! (r, j) / m ! (r, c))
    | i <- [r1 .. rn]
    , j <- [c1 .. cn]
    ]

gaussianElimination :: (Fractional a, Ord a) => Matrix a -> Matrix a
gaussianElimination mat = go (r1, c1) mat
  where
    ((r1, c1), (rn, cn)) = bounds mat
    go (r, c) m
      | c == cn = m
      | pivot == 0 = go (r, c + 1) m
      | otherwise = go (r + 1, c + 1) $ subRows (r, c) (r + 1, rn) (c, cn) m'
      where
        (target, pivot) =
          maximumBy (compare `on` abs . snd) [(k, m ! (k, c)) | k <- [r .. rn]]
        m' = swapRows r target m

gaussJordan :: (Fractional a, Eq a) => Matrix a -> Matrix a
gaussJordan mat = go (r1, c1) mat
  where
    ((r1, c1), (rn, cn)) = bounds mat
    go (r, c) m
      | c == cn = m
      | m ! (r, c) == 0 = go (r, c + 1) m
      | otherwise = go (r + 1, c + 1) $ subRows (r, c) (r1, r - 1) (c, cn) m'
      where
        m' = accum (/) m [((r, j), m ! (r, c)) | j <- [c .. cn]]

backSubstitution :: (Fractional a) => Matrix a -> Vector a
backSubstitution m = sol
  where
    ((r1, _), (rn, cn)) = bounds m
    sol =
      listArray (r1, rn) [(m ! (r, cn) - sum' r) / m ! (r, r) | r <- [r1 .. rn]]
    sum' r = sum [m ! (r, k) * sol ! k | k <- [r + 1 .. rn]]

printM :: (Show a) => Matrix a -> String
printM m =
  let ((r1, c1), (rn, cn)) = bounds m
   in unlines
        [ intercalate "\t" [show $ m ! (r, c) | c <- [c1 .. cn]]
        | r <- [r1 .. rn]
        ]

printV :: (Show a) => Vector a -> String
printV = unlines . map show . elems

main :: IO ()
main = do
  let mat = [2, 3, 4, 6, 1, 2, 3, 4, 3, -4, 0, 10] :: [Ratio Int]
      m = listArray ((1, 1), (3, 4)) mat
  putStrLn "Original Matrix:"
  putStrLn $ printM m
  putStrLn "Echelon form"
  putStrLn $ printM $ gaussianElimination m
  putStrLn "Reduced echelon form"
  putStrLn $ printM $ gaussJordan $ gaussianElimination m
  putStrLn "Solution from back substitution"
  putStrLn $ printV $ backSubstitution $ gaussianElimination m 
```

```
import numpy as np

def gaussian_elimination(A):

    pivot_row = 0

    # Go by column
    for pivot_col in range(min(A.shape[0], A.shape[1])):

        # Swap row with highest element in col
        max_i = np.argmax(abs(A[pivot_row:, pivot_col])) + pivot_row

        temp = A[pivot_row, :].copy()
        A[pivot_row, :] = A[max_i, :]
        A[max_i, :] = temp

        # Skip on singular matrix,  not actually a pivot
        if A[pivot_row, pivot_col] == 0:
            continue

        # Zero out elements below pivot
        for r in range(pivot_row + 1,  A.shape[0]):
            # Get fraction
            frac = -A[r, pivot_col] / A[pivot_row, pivot_col]
            # Add rows
            A[r, :] += frac * A[pivot_row, :]

        pivot_row += 1

# Assumes A is already row echelon form
def gauss_jordan_elimination(A):

    col = 0

    # Scan for pivots
    for row in range(A.shape[0]):
        while col < A.shape[1] and A[row, col] == 0:
            col += 1

        if col >= A.shape[1]:
            continue

        # Set each pivot to one via row scaling
        A[row, :] /= A[row, col]

        # Zero out elements above pivot
        for r in range(row):
            A[r, :] -= A[r, col] * A[row, :]

# Assumes A has a unique solution and A in row echelon form
def back_substitution(A):

    sol = np.zeros(A.shape[0]).T

    # Go by pivots along diagonal
    for pivot_i in range(A.shape[0] - 1,  -1,  -1):
        s = 0
        for col in range(pivot_i + 1,  A.shape[1] - 1):
            s += A[pivot_i, col] * sol[col]
        sol[pivot_i] = (A[pivot_i, A.shape[1] - 1] - s) / A[pivot_i, pivot_i]

    return sol

def main():
    A = np.array([[2, 3, 4, 6],
                  [1, 2, 3, 4,],
                  [3, -4, 0, 10]], dtype=float)

    print("Original")
    print(A, "\n")

    gaussian_elimination(A)
    print("Gaussian elimination")
    print(A, "\n")

    print("Back subsitution")
    print(back_substitution(A), "\n")

    gauss_jordan_elimination(A)
    print("Gauss-Jordan")
    print(A, "\n")

if __name__ == "__main__":
    main() 
```

```
import java.util.Arrays;

public class GaussianElimination {

    static void gaussianElimination(double[][] a) {
        int row = 0;

        int rows = a.length;
        int cols = a[0].length;

        for (int col = 0; col < cols - 1; col++) {
            int pivot = row;

            // finding the maximum element
            for (int i = row + 1; i < row; i++) {
                if (Math.abs(a[i][col]) > Math.abs(a[pivot][col])) {
                    pivot = i;
                }
            }

            if (a[pivot][col] == 0) {
                System.err.println("The matrix is singular");
                continue;
            }

            if (row != pivot) {
                // Swap the row with the highest valued element
                // with the current row
                swapRow(a, col, pivot);
            }

            for (int i = row + 1; i < rows; i++) {
                // finding the inverse
                double scale = a[i][col] / a[row][col];
                // loop through all columns in current row
                for (int j = col + 1; j < cols; j++) {

                    // Subtract rows
                    a[i][j] -= a[row][j] * scale;
                }

                // Set lower elements to 0
                a[i][col] = 0;
            }
            row++;
        }
    }

    static void gaussJordan(double[][] a) {
        int row = 0;

        int cols = a[0].length;

        for (int col = 0; col < cols - 1; col++) {
            if (a[row][col] != 0) {
                for (int i = cols - 1; i > col - 1; i--) {
                    // divide row by pivot so the pivot is set to 1
                    a[row][i] /= a[row][col];
                }

                // subtract the value form above row and set values above pivot to 0
                for (int i = 0; i < row; i++) {
                    for (int j = cols - 1; j > col - 1; j--) {
                        a[i][j] -= a[i][col] * a[row][j];
                    }
                }
                row++;
            }
        }
    }

    static double[] backSubstitution(double[][] a) {
        int rows = a.length;
        int cols = a[0].length;

        double[] solution = new double[rows];

        for (int i = rows - 1; i >= 0; i--) {
            double sum = 0;

            for (int j = cols - 2; j > i; j--) {
                sum += solution[j] * a[i][j];
            }
            solution[i] = (a[i][cols - 1] - sum) / a[i][i];
        }
        return solution;
    }

    static void swapRow(double[][] a, int rowA, int rowB) {
        double[] temp = a[rowA];
        a[rowA] = a[rowB];
        a[rowB] = temp;
    }

    public static void main(String[] args) {
        double[][] a = {
            { 3, 2, -4, 3 },
            { 2, 3, 3, 15 },
            { 5, -3, 1, 14 }
        };

        gaussianElimination(a);
        System.out.println("Gaussian elimination:");
        Arrays.stream(a).forEach(x -> System.out.println(Arrays.toString(x)));

        gaussJordan(a);
        System.out.println("\nGauss-Jordan:");
        Arrays.stream(a).forEach(x -> System.out.println(Arrays.toString(x)));

        System.out.println("\nSolutions:");
        System.out.println(Arrays.toString(backSubstitution(a)));
    }
} 
```

```
function gaussianElimination(a) {
  const rows = a.length
  const cols = a[0].length
  let row = 0;
  for (let col = 0; col < cols - 1; ++col) {

    let pivot = row;
    for (let i = row + 1; i < rows; ++i) {
      if (Math.abs(a[i][col]) > Math.abs(a[pivot][col])) {
        pivot = i;
      }
    }

    if (a[pivot][col] === 0) {
      console.log("The matrix is singular.");
      continue;
    }

    if (col !== pivot) {
      const t = a[col];
      a[col] = a[pivot];
      a[pivot] = t;
    }

    for (let i = row + 1; i < rows; ++i) {
      const scale = a[i][col] / a[row][col];

      for (let j = col + 1; j < cols; ++j) {
        a[i][j] -= a[row][j] * scale;
      }

      a[i][col] = 0;
    }

    ++row;
  }
  return a;
}

function backSubstitution(a) {
  const rows = a.length;
  const cols = a[0].length;
  const sol = [];

  for (let i = rows - 1; i >= 0; --i) {

    let sum = 0;
    for (let j = cols - 2; j > i; --j) {
      sum += sol[j] * a[i][j];
    }

    sol[i] = (a[i][cols - 1] - sum) / a[i][i];
  }
  return sol;
}

function gaussJordan(a) {
  const cols = a[0].length;
  let row = 0;

  for (let col = 0; col < cols - 1; ++col) {
    if (a[row][col] !== 0) {
      for (let i = cols - 1; i > col - 1; --i) {
        a[row][i] /= a[row][col];
      }

      for (let i = 0; i < row; ++i) {
        for (let j = cols - 1; j > col - 1; --j) {
          a[i][j] -= a[i][col] * a[row][j];
        }
      }

      ++row;
    }
  }
}

function printMatrixRow(row) {
  const text = row
    .map(v => (v < 0 ? " " : "  ") + v.toPrecision(8))
    .join("");

  console.log(text);
}

function printMatrix(a) {
  for (const row of a) {
    printMatrixRow(row);
  }
}

const a = [
  [3,  2, -4,  3],
  [2,  3,  3, 15],
  [5, -3,  1, 14]
];

gaussianElimination(a);
console.log("Gaussian elimination:");
printMatrix(a);

gaussJordan(a);
console.log("\nGauss-Jordan:");
printMatrix(a);

const sol = backSubstitution(a);
console.log("\nSolutions are:");
printMatrixRow(sol); 
```

```
// Package demonstrates Gaussian Elimination
package main

import (
    "fmt"
    "math"
)

func gaussianElimination(a [][]float64) {
    singular := false
    rows := len(a)
    cols := len(a[0])

    for c, r := 0, 0; c < cols && r < rows; c++ {
        // 1\. find highest value in column below row to be pivot
        p, highest := r, 0.
        for i, row := range a[r:] {
            if abs := math.Abs(row[c]); abs > highest {
                p = r + i
                highest = abs
            }
        }
        highest = a[p][c] // correct sign

        if highest == 0. {
            if !singular {
                singular = true
                fmt.Println("This matrix is singular.")
            }
            continue
        }

        // 2\. swap pivot with current row
        if p != r {
            a[r], a[p] = a[p], a[r]
        }

        for _, row := range a[r+1:] {
            // 3\. find fraction from pivot value
            frac := row[c] / highest

            // 4\. subtract row to set rest of column to zero
            for j := range row {
                row[j] -= frac * a[r][j]
            }

            // 5\. ensure col goes to zero (no float rounding)
            row[c] = 0.
        }

        r++
    }
}

func gaussJordan(a [][]float64) {
    for r := len(a) - 1; r >= 0; r-- {
        // Find pivot col
        p := -1
        for c, cell := range a[r] {
            if cell != 0. {
                p = c
                break
            }
        }
        if p < 0 {
            continue
        }

        // Scale pivot r to 1.
        scale := a[r][p]
        for c := range a[r][p:] {
            a[r][p+c] /= scale
        }
        // Subtract pivot row from each row above
        for _, row := range a[:r] {
            scale = row[p]
            for c, cell := range a[r][p:] {
                row[p+c] -= cell * scale
            }
        }
    }
}

func backSubstitution(a [][]float64) []float64 {
    rows := len(a)
    cols := len(a[0])
    x := make([]float64, rows)
    for r := rows - 1; r >= 0; r-- {
        sum := 0.

        for c := cols - 2; c > r; c-- {
            sum += x[c] * a[r][c]
        }

        x[r] = (a[r][cols-1] - sum) / a[r][r]
    }
    return x
}

func printMatrixRow(row []float64) {
    fmt.Print("[")
    for _, cell := range row {
        fmt.Printf("%9.4f ", cell)
    }
    fmt.Println("]")
}

func printMatrix(a [][]float64) {
    for _, row := range a {
        printMatrixRow(row)
    }
    fmt.Println()
}

func main() {
    a := [][]float64{
        {2, 3, 4, 6},
        {1, 2, 3, 4},
        {3, -4, 0, 10},
    }
    fmt.Println("Original Matrix:")
    printMatrix(a)

    fmt.Println("Gaussian elimination:")
    gaussianElimination(a)
    printMatrix(a)

    gaussJordan(a)
    fmt.Println("Gauss-Jordan:")
    printMatrix(a)

    fmt.Println("Solutions are:")
    x := backSubstitution(a)
    printMatrixRow(x)
} 
```

## 许可证

##### 代码示例

代码示例授权于 MIT 许可（可在[LICENSE.md](https://github.com/algorithm-archivists/algorithm-archive/blob/main/LICENSE.md)中找到）。

##### 文本

本章的文本由[James Schloss](https://github.com/leios)编写，并授权于[Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/legalcode)。

[](https://creativecommons.org/licenses/by-sa/4.0/)

![](https://creativecommons.org/licenses/by-sa/4.0/)

##### 图片/图形

+   动画"GEvis"由[James Schloss](https://github.com/leios)创建，并授权于[Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/legalcode)。

##### 提交请求

在初始许可([#560](https://github.com/algorithm-archivists/algorithm-archive/pull/560))之后，以下提交请求已修改了本章的文本或图形：

+   无

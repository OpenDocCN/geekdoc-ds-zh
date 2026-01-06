# 托马斯算法

> 原文：[`www.algorithm-archive.org/contents/thomas_algorithm/thomas_algorithm.html`](https://www.algorithm-archive.org/contents/thomas_algorithm/thomas_algorithm.html)

如高斯消元章节中所述，托马斯算法（或 TDMA，三对角矩阵算法）允许程序员在某些情况下将代码的计算成本从降低到！这是通过利用高斯消元的一个特殊情况来实现的，其中矩阵看起来像这样：

这种矩阵形状被称为*三对角矩阵*（当然，不包括我们方程组的右侧！）。现在，起初，可能不明显它如何有帮助。首先，它使得系统更容易编码：我们可以将其分为四个独立的向量，分别对应于，，和（在某些实现中，你会看到缺失的和被设置为零以获得相同大小的四个向量）。其次，最重要的是，这样短且规则的方程容易进行解析求解。

我们将首先应用那些阅读过高斯消元章节的人所熟悉的机制。我们的第一个目标是消除项，并将对角线值设为。和项将被转换成和。第一行尤其容易转换，因为没有，我们只需要将行除以：

假设我们已经找到了转换前行的方法。我们如何转换下一行？我们有

让我们分两步转换行。

**第一步**：使用转换消除：

**第二步**：使用转换得到：

太棒了！有了最后两个公式，我们可以在单次遍历中计算出所有的和，从行开始，因为我们已经知道了的值。

当然，我们真正需要的是解。现在是回代的时候了！

如果我们用方程而不是矩阵来表示我们的系统，我们得到

加上最后一行，这甚至更简单：。一个免费的解决方案！也许我们可以从最后一个解决方案回溯？让我们（几乎）转换上面的方程：

就这样，我们可以在单次遍历中计算出所有的，从末尾开始。

总体来说，我们只需要两次遍历，这就是为什么我们的算法是！转换也很简单，不是吗？

## 示例代码

```
function thomas(a::Vector{Float64}, b::Vector{Float64}, c::Vector{Float64},
                d::Vector{Float64}, n::Int64)

    x = copy(d)
    c_prime = copy(c)

    # Setting initial elements
    c_prime[1] /= b[1]
    x[1] /= b[1]

    for i = 2:n
        # Scale factor is for c_prime and x
        scale = 1.0 / (b[i] - c_prime[i-1]*a[i])
        c_prime[i] *= scale
        x[i] = (x[i] - a[i] * x[i-1]) * scale
    end

    # Back-substitution
    for i = n-1:-1:1
        x[i] -= (c_prime[i] * x[i+1])
    end

    return x

end

function main()
    a = [0.0, 2.0, 3.0]
    b = [1.0, 3.0, 6.0]
    c = [4.0, 5.0, 0.0]
    d = [7.0, 5.0, 3.0]

    println(
        """The system
        $(join((b[1], c[1], "",   "|", d[1]), "\t"))
        $(join((a[2], b[2], c[2], "|", d[2]), "\t"))
        $(join(("",   a[3], b[3], "|", d[3]), "\t"))
        Has the solution:"""
    )

    soln = thomas(a, b, c, d, 3)

    println(soln)
end

main() 
```

```
#include <stdio.h>
#include <string.h>

void thomas(double * const a, double * const b, double * const c,
            double * const x, const size_t size) {

    double y[size];
    memset(y, 0, size * sizeof(double));

    y[0] = c[0] / b[0];
    x[0] = x[0] / b[0];

    for (size_t i = 1; i < size; ++i) {
        double scale = 1.0 / (b[i] - a[i] * y[i - 1]);
        y[i] = c[i] * scale;
        x[i] = (x[i] - a[i] * x[i - 1]) * scale;
    }

    for (size_t i = size - 2; i < size - 1; --i) {
        x[i] -= y[i] * x[i + 1];
    }
}

int main() {
    double a[] = {0.0, 2.0, 3.0};
    double b[] = {1.0, 3.0, 6.0};
    double c[] = {4.0, 5.0, 0.0};
    double x[] = {7.0, 5.0, 3.0};

    printf("The system,\n");
    printf("[1.0  4.0  0.0][x] = [7.0]\n");
    printf("[2.0  3.0  5.0][y] = [5.0]\n");
    printf("[0.0  3.0  6.0][z] = [3.0]\n");
    printf("has the solution:\n");

    thomas(a, b, c, x, 3);

    for (size_t i = 0; i < 3; ++i) {
        printf("[%f]\n", x[i]);
    }

    return 0;
} 
```

```
# Author: gammison

# note this example is inplace and destructive
def thomas(a, b, c, d):

    # set the initial elements
    c[0] = c[0] / b[0]
    d[0] = d[0] / b[0]

    n = len(d) # number of equations to solve
    for i in range(1, n):
        # scale factor for c and d
        scale = 1 / (b[i] - c[i-1] * a[i])

        c[i] *= scale
        d[i] = (d[i] - a[i] * d[i-1]) * scale

    # do the back substitution
    for i in range(n-2, -1, -1):
        d[i] -= c[i] * d[i+1]

    return d

def main():
    # example for matrix
    # [1  4  0][x]   [7]
    # [2  3  5][y] = [5]
    # [0  3  6][z]   [3]

    #                 [.8666]
    # soln will equal [1.533]
    #                 [-.266]
    # note we index a from 1 and c from 0
    a = [0, 2, 3]
    b = [1, 3, 6]
    c = [4, 5, 0]
    d = [7, 5, 3]

    soln = thomas(a, b, c, d)
    print(soln)

if __name__ == '__main__':
    main() 
```

你可以在[这个项目](https://scratch.mit.edu/projects/169418273/)中找到这个算法的实现。

![](img/d03db70d5015dc976f04bfb124fe8ea4.png)

```
public class Thomas {
    private static double[] thomasAlgorithm(double[] a, double[] b, double[] c, double[] x) {
        int size = a.length;
        double[] y = new double[size]; // This is needed so that we don't have to modify c
        double[] solution = new double[size];

        // Set initial elements
        y[0] = c[0] / b[0];
        solution[0] = x[0] / b[0];

        for (int i = 1; i < size; ++i) {
            // Scale factor is for c and x
            double scale = 1.0 / (b[i] - a[i] * y[i - 1]);
            y[i] = c[i] * scale;
            solution[i] = (x[i] - a[i] * solution[i - 1]) * scale;
        }

        // Back-substitution
        for (int i = size - 2; i >= 0; --i) {
            solution[i] -= y[i] * solution[i + 1];
        }

        return solution;
    }

    public static void main(String[] args) {
        double[] a = {0.0, 2.0, 3.0};
        double[] b = {1.0, 3.0, 6.0};
        double[] c = {4.0, 5.0, 0.0};
        double[] x = {7.0, 5.0, 3.0};
        double[] solution = thomasAlgorithm(a, b, c, x);

        System.out.format("The system,\n");
        System.out.format("[%.1f, %.1f, %.1f][x] = [%.1f]\n", b[0], c[0], 0f, x[0]);
        System.out.format("[%.1f, %.1f, %.1f][y] = [%.1f]\n", a[1], b[1], c[1], x[1]);
        System.out.format("[%.1f, %.1f, %.1f][z] = [%.1f]\n", 0f, a[2], b[2], x[2]);
        System.out.format("has the solution:\n");

        for (int i = 0; i < solution.length; i++) {
            System.out.format("[% .5f]\n", solution[i]);
        }
    }
} 
```

```
import Data.List (zip4)
import Data.Ratio

thomas :: Fractional a => [a] -> [a] -> [a] -> [a] -> [a]
thomas a b c = init . scanr back 0 . tail . scanl forward (0, 0) . zip4 a b c
  where
    forward (c', d') (a, b, c, d) =
      let denominator = b - a * c'
       in (c / denominator, (d - a * d') / denominator)
    back (c, d) x = d - c * x

main :: IO ()
main = do
  let a = [0, 2, 3] :: [Ratio Int]
      b = [1, 3, 6]
      c = [4, 5, 0]
      d = [7, 5, 3]
  print $ thomas a b c d 
```

```
package main

import "fmt"

func thomas(a, b, c, d []float64) []float64 {
    c[0] = c[0] / b[0]
    d[0] = d[0] / b[0]

    for i := 1; i < len(d); i++ {
        scale := 1. / (b[i] - c[i-1]*a[i])
        c[i] *= scale
        d[i] = (d[i] - a[i]*d[i-1]) * scale
    }

    for i := len(d) - 2; i >= 0; i-- {
        d[i] -= c[i] * d[i+1]
    }

    return d
}

func main() {
    a := []float64{0., 2., 3.}
    b := []float64{1., 3., 6.}
    c := []float64{4., 5., 0.}
    d := []float64{7., 5., 3.}

    fmt.Println("The system,")
    fmt.Println("[1.0  4.0  0.0][x] = [7.0]")
    fmt.Println("[2.0  3.0  5.0][y] = [5.0]")
    fmt.Println("[0.0  3.0  6.0][z] = [3.0]")
    fmt.Println("has the solution:")
    solve := thomas(a, b, c, d)
    for _, i := range solve {
        fmt.Printf("[%f]\n", i)
    }
} 
```

```
fn thomas(a []f32, b []f32, c []f32, d []f32) []f32 {
    mut new_c := c
    mut new_d := d
    new_c[0] = new_c[0] / b[0]
    new_d[0] = new_d[0] / b[0]

    for i := 1; i < d.len; i++ {
        scale := 1. / (b[i] - new_c[i-1]*a[i])
        new_c[i] *= scale
        new_d[i] = (new_d[i] - a[i]*new_d[i-1]) * scale
    }

    for i := d.len - 2; i >= 0; i-- {
        new_d[i] -= new_c[i] * new_d[i+1]
    }

    return new_d
}

fn main() {
    a := [0.0, 2.0, 3.0]
    b := [1.0, 3.0, 6.0]
    c := [4.0, 5.0, 0.0]
    d := [7.0, 5.0, 3.0]

    println("The system,")
    println("[1.0  4.0  0.0][x] = [7.0]")
    println("[2.0  3.0  5.0][y] = [5.0]")
    println("[0.0  3.0  6.0][z] = [3.0]")
    println("has the solution:")
    solution := thomas(a, b, c, d)
    for i in solution {
        println("[$i]")
    }
} 
```

```
func thomas(a: [Double], b: [Double], c: [Double], d: [Double]) -> [Double] {
    var a = a
    var b = b
    var c = c
    var d = d

    // set the initial elements
    c[0] = c[0] / b[0]
    d[0] = d[0] / b[0]

    let n = d.count // number of equations to solve
    for i in 1..<n {
        // scale factor for c and d
        let scale = 1 / (b[i] - c[i-1] * a[i])

        c[i] = c[i] * scale
        d[i] = (d[i] - a[i] * d[i-1]) * scale
    }

    // do the back substitution
    for i in stride(from: n-2, to: -1, by: -1) {
        d[i] = d[i] - c[i] * d[i+1]
    }

    return d
}

func main() {
    let a = [0.0, 2.0, 3.0]
    let b = [1.0, 3.0, 6.0]
    let c = [4.0, 5.0, 0.0]
    let d = [7.0, 5.0, 3.0]

    print(thomas(a: a, b: b, c: c, d: d))
}

main() 
```

```
<?php
declare(strict_types=1);

function thomas_algorithm(array $a, array $b, array $c, array $x, int $size): array
{
    $y = [];
    $y[0] = $b[0] == 0 ? 0 : $c[0] / $b[0];
    $x[0] = $b[0] == 0 ? 0 : $x[0] / $b[0];

    for ($i = 1; $i < $size; ++$i) {
        $scale = (float)(1 / ($b[$i] - $a[$i] * $y[$i - 1]));
        $y[$i] = $c[$i] * $scale;
        $x[$i] = ($x[$i] - $a[$i] * $x[$i - 1]) * $scale;
    }

    for ($i = $size - 2; $i >= 0; --$i)
        $x[$i] -= $y[$i] & $x[$i + 1];

    return $x;
}

$a = [0.0, 2.0, 3.0];
$b = [1.0, 3.0, 6.0];
$c = [4.0, 5.0, 0.0];
$x = [7.0, 5.0, 3.0];

printf('The system,%s', PHP_EOL);
printf('  [%s, %s, %s][x] = [%s]%s', $b[0], $c[0], 0, $x[0], PHP_EOL);
printf('  [%s, %s, %s][y] = [%s]%s', $a[1], $b[1], $c[1], $x[1], PHP_EOL);
printf('  [%s, %s, %s][z] = [%s]%s', 0, $a[2], $b[2], $x[2], PHP_EOL);
printf('has the solution:%s', PHP_EOL);

$solution = thomas_algorithm($a, $a, $c, $x, count($x));
for ($i = 0; $i < count($solution); $i++)
    printf('  [%s]%s', $solution[$i], PHP_EOL); 
```

```
proc thomas_algorithm(a, b, c_in, d_in: seq[float]): seq[float] = 

  let n: int = len(d_in)

  var c: seq[float] = c_in
  var d: seq[float] = d_in

  c[0] /= b[0]
  d[0] /= b[0]

  for i in 1..n - 1:
    let scale: float = (1 / (b[i] - c[i - 1] * a[i]))

    c[i] *= scale
    d[i] = (d[i] - a[i] * d[i - 1]) * scale

  for i in countdown(n - 2,0):
    d[i] -= c[i] * d[i + 1]

  return d

const x: seq[float] = @[0.0, 2.0, 3.0]
const y: seq[float] = @[1.0, 3.0, 6.0]
const z: seq[float] = @[4.0, 5.0, 0.0]
const w: seq[float] = @[7.0, 5.0, 3.0]            

echo "The system,"
echo "[1.0 4.0 0.0][x] = [7.0]"
echo "[2.0 3.0 5.0][y] = [5.0]"
echo "[0.0 3.0 6.0][z] = [3.0]"

echo "has the solution:"

const soln: seq[float] = thomas_algorithm(x, y, z, w)

for i in 0..len(w) - 1:
  echo soln[i] 
```

```
#include <cstddef>
#include <iostream>
#include <vector>

void thomas(
    std::vector<double> const& a,
    std::vector<double> const& b,
    std::vector<double> const& c,
    std::vector<double>& x) {
  auto y = std::vector<double>(a.size(), 0.0);

  y[0] = c[0] / b[0];
  x[0] = x[0] / b[0];

  for (std::size_t i = 1; i < a.size(); ++i) {
    const auto scale = 1.0 / (b[i] - a[i] * y[i - 1]);
    y[i] = c[i] * scale;
    x[i] = (x[i] - a[i] * x[i - 1]) * scale;
  }

  for (std::size_t i = a.size() - 2; i < a.size(); --i) {
    x[i] -= y[i] * x[i + 1];
  }
}

int main() {
  const std::vector<double> a = {0.0, 2.0, 3.0};
  const std::vector<double> b = {1.0, 3.0, 6.0};
  const std::vector<double> c = {4.0, 5.0, 0.0};
  std::vector<double> x = {7.0, 5.0, 3.0};

  std::cout << "The system\n";
  std::cout << "[1.0  4.0  0.0][x] = [7.0]\n";
  std::cout << "[2.0  3.0  5.0][y] = [5.0]\n";
  std::cout << "[0.0  3.0  6.0][z] = [3.0]\n";
  std::cout << "has the solution:\n";

  thomas(a, b, c, x);

  for (auto const& val : x) {
    std::cout << "[" << val << "]\n";
  }

  return 0;
} 
```

```
local function thomas(a, b, c, d)

  -- Create tables and set initial elements
  local c_prime = {c[1] / b[1]}
  local result = {d[1] / b[1]}

  for i = 2, #a do
    -- Scale factor is for c_prime and result
    local scale = 1.0 / (b[i] - a[i] * c_prime[i - 1])
    c_prime[i] = c[i] * scale
    result[i] = (d[i] - a[i] * result[i - 1]) * scale
  end

  -- Back-substitution
  for i = #a-1, 1, -1 do
    result[i] = result[i] - (c_prime[i] * result [i + 1])
  end

  return result
end

local a = {0.0, 2.0, 3.0}
local b = {1.0, 3.0, 6.0}
local c = {4.0, 5.0, 0.0}
local d = {7.0, 5.0, 3.0}

print("The system")
print(b[1], c[1], "",   "|", d[1])
print(a[2], b[2], c[2], "|", d[2])
print("",   a[3], b[3], "|", d[3])
print("Has the solution:")

local solution = thomas(a, b, c, d)

print(table.unpack(solution)) 
```

```
def thomas(a, b, c, d)
  c_prime = c.dup
  x = d.dup

  # Setting initial elements
  c_prime[0] /= b[0]
  x[0] /= b[0]

  1.upto(a.size - 1) do |i|
    # Scale factor is for c_prime and x
    scale = 1.0 / (b[i] - c_prime[i - 1]*a[i])
    c_prime[i] *= scale
    x[i] = (x[i] - a[i] * x[i - 1]) * scale
  end

  # Back-substitution
  (a.size - 2).downto(0) do |i|
    x[i] -= (c_prime[i] * x[i + 1])
  end

  x
end

def main
  a = [0.0, 2.0, 3.0]
  b = [1.0, 3.0, 6.0]
  c = [4.0, 5.0, 0.0]
  d = [7.0, 5.0, 3.0]

  puts "The system"
  puts [b[0], c[0], "",   "|", d[0]].join("\t")
  puts [a[1], b[1], c[1], "|", d[1]].join("\t")
  puts ["",   a[2], b[2], "|", d[2]].join("\t")
  puts "Has the solution:"

  soln = thomas(a, b, c, d)

  puts soln.join("\t")
end

main 
```

```
private fun thomas(a: DoubleArray, b: DoubleArray, c: DoubleArray, d: DoubleArray): DoubleArray {
    val cPrime = c.clone()
    val x = d.clone()
    val size = a.size
    cPrime[0] /= b[0]
    x[0] /= b[0]
    for (i in 1 until size) {
        val scale = 1.0 / (b[i] - cPrime[i - 1] * a[i])
        cPrime[i] *= scale
        x[i] = (x[i] - a[i] * x[i - 1]) * scale
    }
    for (i in (size - 2) downTo 0) {
        x[i] -= cPrime[i] * x[i + 1]
    }
    return x
}

fun main(args: Array<String>) {
    val a = doubleArrayOf(0.0, 2.0, 3.0)
    val b = doubleArrayOf(1.0, 3.0, 6.0)
    val c = doubleArrayOf(4.0, 5.0, 0.0)
    val x = doubleArrayOf(7.0, 5.0, 3.0)
    val solution = thomas(a, b, c, x)

    println("System:")
    println("[%.1f, %.1f, %.1f][x] = [%.1f]".format(b[0], c[0], 0f, x[0]))
    println("[%.1f, %.1f, %.1f][y] = [%.1f]".format(a[1], b[1], c[1], x[1]))
    println("[%.1f, %.1f, %.1f][z] = [%.1f]\n".format(0f, a[2], b[2], x[2]))
    println("Solution:")
    for (i in solution.indices) {
        println("[% .5f]".format(solution[i]))
    }
} 
```

```
;;;; Thomas algorithm implementation in Common Lisp

(defmacro divf (place divisor)
  "Divides the value at place by divisor"
  `(setf ,place (/ ,place ,divisor)))

(defun helper (v1 v2 v3 row)
  (- (svref v1 row) (* (svref v2 row) (svref v3 (1- row)))))

(defun thomas (diagonal-a diagonal-b diagonal-c last-column)
  "Returns the solutions to a tri-diagonal matrix non-destructively"
  ;; We have to copy the inputs to ensure non-destructiveness
  (let ((a (copy-seq diagonal-a))
         (b (copy-seq diagonal-b))
         (c (copy-seq diagonal-c))
         (d (copy-seq last-column)))
    (divf (svref c 0) (svref b 0))
    (divf (svref d 0) (svref b 0))
    (loop
      for i from 1 upto (1- (length a)) do
      (divf (svref c i) (helper b a c i))
      (setf (svref d i) (/ (helper d a d i) (helper b a c i))))
    (loop
      for i from (- (length a) 2) downto 0 do
      (decf (svref d i) (* (svref c i) (svref d (1+ i)))))
    d))

(defparameter diagonal-a #(0 2 3))
(defparameter diagonal-b #(1 3 6))
(defparameter diagonal-c #(4 5 0))
(defparameter last-column #(7 5 3))

;; should print 0.8666667 1.5333333 -0.26666668
(format t "~{~f ~}~%" (coerce (thomas diagonal-a diagonal-b diagonal-c last-column) 'list)) 
```

```
# note this example is inplace and destructive
def thomas(a, b, c, d)
  # set the initial elements
  c[0] = c[0] / b[0]
  d[0] = d[0] / b[0]

  n = d.length # number of equations to solve
  (1...n).each do |i|
    scale = 1 / (b[i] - c[i - 1] * a[i]) # scale factor for c and d
    c[i] *= scale
    d[i] = (d[i] - a[i] * d[i - 1]) * scale
  end

  # do the back substitution
  (n - 2).downto(0).each do |j|
    d[j] -= c[j] * d[j + 1]
  end

  d
end

# example for matrix
# [1  4  0][x]   [7]
# [2  3  5][y] = [5]
# [0  3  6][z]   [3]

#                 [.8666]
# soln will equal [1.533]
#                 [-.266]
# note we index a from 1 and c from 0

a = [0.0, 2.0, 3.0]
b = [1.0, 3.0, 6.0]
c = [4.0, 5.0, 0.0]
d = [7.0, 5.0, 3.0]

soln = thomas(a, b, c, d)
puts soln 
```

```
function thomas(a, b, c, x) {
    const y = [];

    y[0] = c[0] / b[0];
    x[0] = x[0] / b[0];

    for (let i = 1; i < a.length; i++) {
        const scale = 1.0 / (b[i] - a[i] * y[i - 1]);
        y[i] = c[i] * scale;
        x[i] = (x[i] - a[i] * x[i - 1]) * scale;
    }

    for (let i = a.length - 2; i >= 0; i--)
        x[i] -= y[i] * x[i + 1];
}

let a = [0.0, 2.0, 3.0];
let b = [1.0, 3.0, 6.0];
let c = [4.0, 5.0, 0.0];
let x = [7.0, 5.0, 3.0];

console.log("The system,");
console.log("[1.0  4.0  0.0][x] = [7.0]");
console.log("[2.0  3.0  5.0][y] = [5.0]");
console.log("[0.0  3.0  6.0][z] = [3.0]");
console.log("has the solution:\n");

thomas(a, b, c, x);

for (let i = 0; i < 3; i++)
    console.log("[" + x[i] + "]"); 
```

```
fn thomas(a: &[f64], b: &[f64], c: &[f64], x: &[f64]) -> Vec<f64> {
    let size = a.len();
    let mut y = vec![0.0; size];
    let mut z = Vec::from(x);

    y[0] = c[0] / b[0];
    z[0] = x[0] / b[0];

    for i in 1..size {
        let scale = 1.0 / (b[i] - a[i] * y[i - 1]);
        y[i] = c[i] * scale;
        z[i] = (z[i] - a[i] * z[i - 1]) * scale;
    }

    for i in (0..(size - 1)).rev() {
        z[i] -= y[i] * z[i + 1];
    }

    z
}

fn main() {
    let a = vec![0.0, 2.0, 3.0];
    let b = vec![1.0, 3.0, 6.0];
    let c = vec![4.0, 5.0, 0.0];
    let x = vec![7.0, 5.0, 3.0];

    println!("The system");
    println!("[{:?} {:?} {:?}][x] = [{:?}]", a[0], b[0], c[0], &x[0]);
    println!("[{:?} {:?} {:?}][x] = [{:?}]", a[1], b[1], c[1], &x[1]);
    println!("[{:?} {:?} {:?}][x] = [{:?}]", a[2], b[2], c[2], &x[2]);
    println!("has the solution");

    let y = thomas(&a, &b, &c, &x);

    y.iter()
        .for_each(|i| println!("[{:>19}]", format!("{:18}", format!("{:?}", i))));
} 
```

## 许可证

##### 代码示例

代码示例许可在 MIT 许可下（可在[LICENSE.md](https://github.com/algorithm-archivists/algorithm-archive/blob/main/LICENSE.md)中找到）。

##### 文本

本章的文本由[James Schloss](https://github.com/leios)编写，并许可在[Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/legalcode)下。

[](https://creativecommons.org/licenses/by-sa/4.0/)

![](https://creativecommons.org/licenses/by-sa/4.0/)

##### 拉取请求

在初始许可([#560](https://github.com/algorithm-archivists/algorithm-archive/pull/560))之后，以下拉取请求已修改了本章的文本或图形：

+   无

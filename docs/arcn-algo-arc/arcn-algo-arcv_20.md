# 卷积定理

> 原文：[`www.algorithm-archive.org/contents/convolutions/convolutional_theorem/convolutional_theorem.html`](https://www.algorithm-archive.org/contents/convolutions/convolutional_theorem/convolutional_theorem.html)

重要提示：在修订傅里叶变换和快速傅里叶变换（FFT）章节之后，本节将进行扩展。

现在，让我告诉你一点计算上的魔法：

**傅里叶变换可以用来执行卷积！**

这听起来很疯狂，但解释起来也非常困难，所以让我尽力而为。正如在傅里叶变换章节中所述，傅里叶变换允许程序员从实空间移动到频域。当我们把一个波变换到频域时，我们可以看到一个与该波频率相关的单个峰值。无论我们向傅里叶变换发送什么函数，频域图像都可以解释为一系列具有指定频率的不同波。每个这样的波都由另一个 \( \phi \) 项参数化，其中 \( \phi \) 是频域中元素的值，\( \phi \) 是时域中的值，而 \( \phi \) 是信号的总体长度。这样，每个波都可以看作是一个复指数。

所以，这里有一个想法：如果我们取两个函数 \( f \) 和 \( g \)，并将它们移动到频域，变成 \( F \) 和 \( G \)，然后我们可以将这两个函数相乘并将它们转换回 \( f \) 来混合信号。这样，我们将得到一个第三个函数，它关联着两个输入函数的频域图像。这被称为**卷积定理**，其形式如下：

其中 \( F \) 表示傅里叶变换。

起初，这可能看起来并不特别直观，但请记住，频域本质上是由一组指数组成的。正如乘法作为卷积部分中提到的，10 进制空间中的乘法也是一种卷积。卷积定理将这个概念扩展到与任何一组指数的乘法，而不仅仅是 10 进制。显然，这个描述仍然缺少一些解释，但我保证在修订傅里叶变换章节时我们会添加更多内容！

通过在代码中使用快速傅里叶变换（FFT），这可以将两个长度为 \( n \) 的数组的标准卷积过程从 \( O(n²) \) 减少到 \( O(n \log n) \)。这意味着卷积定理对于创建某些大型输入的快速卷积方法是基本的。

```
# using the convolutional theorem
function convolve_fft(signal1::Array{T}, signal2::Array{T}) where {T <: Number}
    return ifft(fft(signal1).*fft(signal2))
end 
```

这种方法还有一个额外的优点，那就是它将*始终输出一个与你的信号大小相同的数组*；然而，如果你的信号大小不等，我们需要用零填充较小的信号。此外，请注意，傅里叶变换是一个周期性或循环操作，因此这种方法中没有真正的边缘，相反，数组“环绕”到另一边，就像我们在一维卷积的周期性边界条件案例中展示的那样，形成一个循环卷积。

## 示例代码

对于这个示例代码，我们将使用两个锯齿波函数，就像我们在一维卷积章节中做的那样(one-dimensional convolutions)：

```
using FFTW
using LinearAlgebra
using DelimitedFiles

# using the convolutional theorem
function convolve_fft(signal1::Array{T}, signal2::Array{T}) where {T <: Number}
    return ifft(fft(signal1).*fft(signal2))
end

function main()

    # sawtooth functions for x and y
    x = [float(i)/200 for i = 1:200]
    y = [float(i)/200 for i = 1:200]

    # Normalization is not strictly necessary, but good practice
    normalize!(x)
    normalize!(y)

    # cyclic convolution via the convolutional theorem
    fft_output = convolve_fft(x, y)

    # outputting convolutions to different files for plotting in external code
    # note: we are outputting just the real component because the imaginary
    #       component is virtually 0
    writedlm("fft.dat", real(fft_output))

end

main() 
```

```
from scipy.fft import fft, ifft
import numpy as np

# using the convolutional theorem
def convolve_fft(signal1, signal2):
    return ifft(np.multiply(fft(signal1),fft(signal2)))

# Sawtooth functions
x = [float(i)/200 for i in range(1,101)]
y = [float(i)/200 for i in range(1,101)]

x /= np.linalg.norm(x)
y /= np.linalg.norm(y)

# Convolving the two signals
fft_output = convolve_fft(x, y)

np.savetxt("fft.dat", np.real(fft_output)) 
```

这应该产生以下输出：

![图片](img/0e5df4fc813d4fa6da84a546ad5b2a9f.png)

## 许可证

##### 代码示例

代码示例受 MIT 许可协议保护（可在[LICENSE.md](https://github.com/algorithm-archivists/algorithm-archive/blob/main/LICENSE.md)中找到）。

##### 图片/图形

+   图片"Cyclic"由[James Schloss](https://github.com/leios)创建，并受[Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/legalcode)许可。

##### 文本

本章的文本由[James Schloss](https://github.com/leios)编写，并受[Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/legalcode)许可。

[](https://creativecommons.org/licenses/by-sa/4.0/)

![示例代码图片](img/c7782818305d7cc32e33f74558a972b7.png)[(https://creativecommons.org/licenses/by-sa/4.0/)]

##### 拉取请求

在初始许可后([#560](https://github.com/algorithm-archivists/algorithm-archive/pull/560))，以下拉取请求已修改本章的文本或图形：

+   无

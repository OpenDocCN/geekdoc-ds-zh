光线追踪之一周末的指南

一个周末的光线追踪[彼得·雪莉](https://github.com/petershirley), [特雷弗·大卫·布莱克](https://github.com/trevordblack), [史蒂夫·霍拉斯](https://github.com/hollasch)版本 4.0.2, 2025-04-25 版权所有 2018-2024 彼得·雪莉。保留所有权利。

# 概述

这些年我教过很多图形学课程。通常我会用光线追踪来做，因为这样你必须编写所有的代码，但你仍然可以用没有 API 的方式得到酷炫的图像。我决定将我的课程笔记改编成教程，以便你能尽快进入一个酷炫的程序。这不会是一个功能齐全的光线追踪器，但它确实具有使光线追踪成为电影中必备技术的间接照明。遵循这些步骤，如果你兴奋并想要进一步追求，你制作的光线追踪器的架构将适合扩展到一个更广泛的光线追踪器。

当有人说“光线追踪”时，它可能意味着很多不同的东西。我将要描述的是技术上的路径追踪器，而且相当通用。虽然代码将会相当简单（让计算机做工作！）但我相信你会对能制作出的图像感到非常满意。

我会按照我实际操作的方式带你编写光线追踪器，并提供一些调试技巧。到那时，你将拥有一个能够生成一些精美图像的光线追踪器。你应该能在周末内完成这个任务。如果你花了更长的时间，不要担心。我使用 C++作为驱动语言，但你不必这样做。然而，我建议你这样做，因为 C++速度快、可移植，并且大多数电影和视频游戏的渲染器都是用 C++编写的。请注意，我避免使用 C++的“现代特性”，但继承和运算符重载对于光线追踪器来说太有用，不能放弃。

> 我没有在网上提供代码，但代码是真实的，除了`vec3`类中的一些简单运算符外，我都展示了所有代码。我非常相信通过输入代码来学习，但代码可用时我会使用它，所以当代码不可用时，我只实践我所宣扬的。所以请不要问我！

我保留了最后一部分，因为我在这里做了 180 度的大转弯。一些读者在比较代码时发现了细微的错误，这得到了帮助。所以请务必输入代码，但你可以在 GitHub 上的[RayTracing 项目](https://github.com/RayTracing/raytracing.github.io/)中找到每本书的完成源代码。

关于这些书籍的实现代码——我们对于包含的代码的哲学优先考虑以下目标：

+   代码应实现书中涵盖的概念。

+   我们使用 C++，但尽可能简单。我们的编程风格非常类似于 C 语言，但我们利用现代特性，使其代码更容易使用或理解。

+   我们的编码风格尽可能地延续了原始书籍中建立的风格，以保持连贯性。

+   每行代码长度保持在每行 96 个字符，以保持代码库和书中代码列表的一致性。

因此，代码提供了一个基线实现，留下了大量的改进空间供读者享受。有无数种优化和现代化的方法；我们优先考虑简单的解决方案。

我们假设读者对向量（如点积和向量加法）有一定的熟悉度。如果你不知道这些，请稍作复习。如果你需要复习，或者第一次学习这些内容，请查看 Morgan McGuire 的在线[*Graphics Codex*](https://graphicscodex.com/)，Steve Marschner 和 Peter Shirley 的[*计算机图形学基础*](https://graphicscodex.com/)，或者 J.D. Foley 和 Andy Van Dam 的[*计算机图形学：原理与实践*](https://graphicscodex.com/)。

查看项目的 README 文件以获取有关此项目、GitHub 上的存储库、目录结构、构建和运行以及如何制作或引用更正和贡献的信息。

查看我们的[进一步阅读 wiki 页面](https://github.com/RayTracing/raytracing.github.io/wiki/Further-Readings)以获取更多与项目相关的资源。

这些书籍已经格式化，可以直接从你的浏览器打印出来。我们还包含每个版本的书籍 PDF 文件[与每个发布版本一起](https://github.com/RayTracing/raytracing.github.io/releases/)，在“资产”部分。

如果你想与我们联系，请随时通过以下邮箱发送邮件：

+   彼得·雪莉，ptrshrl@gmail.com

+   史蒂夫·霍拉斯奇，steve@hollasch.net

+   特雷弗·大卫·布莱克，trevordblack@trevord.black

最后，如果你在实现过程中遇到问题，有一般性问题，或者想要分享你自己的想法或工作，请参阅 GitHub 项目上的[GitHub 讨论论坛](https://github.com/RayTracing/raytracing.github.io/discussions/)。

感谢所有在这个项目上伸出援手的人。你可以在本书末尾的致谢部分找到他们。

让我们开始吧！

# 输出图像

## PPM 图像格式

每次启动渲染器时，你需要一种查看图像的方法。最直接的方法是将它写入文件。问题是，有如此多的格式。其中许多都很复杂。我总是从一个简单的 ppm 文本文件开始。以下是从维基百科的一个很好的描述：

[![](https://raytracing.github.io/images/fig-1.01-ppm.jpg)](https://raytracing.github.io/images/fig-1.01-ppm.jpg)**图 1:** PPM 示例让我们编写一些 C++ 代码来输出这样的内容：

```
#include <iostream>

int main() {

    // Image

    int image_width = 256;
    int image_height = 256;

    // Render

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            auto r = double(i) / (image_width-1);
            auto g = double(j) / (image_height-1);
            auto b = 0.0;

            int ir = int(255.999 * r);
            int ig = int(255.999 * g);
            int ib = int(255.999 * b);

            std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }
}
```

**列表 1:** `[main.cc]` 创建你的第一个图像

在此代码中需要注意一些事项：

1.  像素是按行写出的。

1.  每行像素是从左到右写出的。

1.  这些行是从上到下写出的。

1.  按照惯例，每个红色/绿色/蓝色组件在内部都由范围从 0.0 到 1.0 的实值变量表示。在打印出来之前，这些必须缩放到 0 到 255 之间的整数值。

1.  红色从完全关闭（黑色）到完全开启（亮红色）从左到右变化，绿色从顶部完全关闭（黑色）到底部完全开启（亮绿色）变化。红色和绿色光结合在一起形成黄色，因此我们应该预期右下角是黄色。

## 创建图像文件

因为文件是写入标准输出流，所以你需要将其重定向到图像文件。通常这通过命令行使用 `>` 重定向运算符来完成。

在 Windows 上，你会从 CMake 运行此命令以获取调试构建：

```
cmake -B build
cmake --build build
```

然后这样运行你新构建的程序：

```
build\Debug\inOneWeekend.exe > image.ppm
```

之后，为了提高速度，最好运行优化后的构建。在这种情况下，你会这样构建：

```
cmake --build build --config release
```

然后这样运行优化后的程序：

```
build\Release\inOneWeekend.exe > image.ppm
```

上面的示例假设你正在使用 CMake 构建，使用与包含源文件中的 `CMakeLists.txt` 文件相同的方法。使用你最舒适的构建环境（和语言）。

在 Mac 或 Linux 上，发布构建，你会这样启动程序：

```
build/inOneWeekend > image.ppm
```

完整的构建和运行说明可以在项目 README 中找到。

在我的 Mac 上，我在`ToyViewer`中打开输出文件（如果你的查看器不支持，请尝试使用你喜欢的图像查看器，并在 Google 中搜索“ppm viewer”），显示以下结果：

[![](https://raytracing.github.io/images/img-1.01-first-ppm-image.png)](https://raytracing.github.io/images/img-1.01-first-ppm-image.png)图 1：第一个 PPM 图像

哈喽！这是“hello world”的图形。如果你的图像看起来不像这样，请用文本编辑器打开输出文件，看看它是什么样子。它应该从以下内容开始：

```
P3
256 256
255
0 0 0
1 0 0
2 0 0
3 0 0
4 0 0
5 0 0
6 0 0
7 0 0
8 0 0
9 0 0
10 0 0
11 0 0
12 0 0
...
```

**列表 2**：第一个图像输出

如果你的 PPM 文件看起来不像这样，那么请仔细检查你的格式化代码。如果它确实看起来像这样但无法渲染，那么你可能存在行结束差异或其他类似的问题，这可能会使你的图像查看器困惑。为了帮助调试这个问题，你可以在 GitHub 项目的`images`目录中找到一个名为`test.ppm`的文件。这应该有助于确保你的查看器可以处理 PPM 格式，并作为与你的生成 PPM 文件进行比较的依据。

一些读者报告说，他们在 Windows 上查看生成的文件时遇到了问题。在这种情况下，问题通常是 PPM 被以 UTF-16 格式写入，通常来自 PowerShell。如果你遇到这个问题，请参阅[讨论 1114](https://github.com/RayTracing/raytracing.github.io/discussions/1114)以获取此问题的帮助。

如果一切显示正确，那么你基本上就完成了系统和 IDE 的问题——本系列剩余部分的所有内容都使用这种相同的简单机制来生成渲染图像。

如果你想要生成其他图像格式，我非常喜欢`stb_image.h`，这是一个 GitHub 上可用的仅头文件图像库，网址为[`github.com/nothings/stb`](https://github.com/nothings/stb)。

## 添加进度指示器

在我们继续之前，让我们给我们的输出添加一个进度指示器。这是一种跟踪长时间渲染进度的便捷方式，也可以用来识别由于无限循环或其他问题而停滞的运行。

我们程序将图像输出到标准输出流（`std::cout`），所以请保持不变，而是写入到日志输出流（`std::clog`）：

```
 for (int j = 0; j < image_height; ++j) {        std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;        for (int i = 0; i < image_width; i++) {
            auto r = double(i) / (image_width-1);
            auto g = double(j) / (image_height-1);
            auto b = 0.0;

            int ir = int(255.999 * r);
            int ig = int(255.999 * g);
            int ib = int(255.999 * b);

            std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }

    std::clog << "\rDone.                 \n";
```

**列表 3**：`[main.cc]`带有进度报告的主渲染循环

现在运行时，你会看到剩余扫描线的运行计数。希望这运行得如此之快，以至于你甚至看不到它！别担心——将来你将有足够的时间观察缓慢更新的进度条，随着我们扩展光线追踪器。

# vec3 类

几乎所有图形程序都有一些用于存储几何向量和颜色的类。在许多系统中，这些向量是 4D 的（3D 位置加上用于几何的齐次坐标，或 RGB 加上 alpha 透明度组件用于颜色）。就我们的目的而言，三个坐标就足够了。我们将使用相同的类`vec3`来表示颜色、位置、方向、偏移等。有些人不喜欢这样做，因为它不能阻止你做一些愚蠢的事情，比如从一个颜色中减去一个位置。他们有很好的观点，但当我们没有明显错误时，我们将始终选择“更少的代码”路线。尽管如此，我们仍然为`vec3`声明了两个别名：`point3`和`color`。由于这两个类型只是`vec3`的别名，所以如果你将一个`color`传递给期望一个`point3`的函数，你不会收到警告，而且没有任何东西阻止你将一个`point3`加到一个`color`上，但这会使代码更容易阅读和理解。

我们在新的`vec3.h`头文件的顶部定义了`vec3`类，并在底部定义了一组有用的向量实用函数：

```
#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

class vec3 {
  public:
    double e[3];

    vec3() : e{0,0,0} {}
    vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

    double x() const { return e[0]; }
    double y() const { return e[1]; }
    double z() const { return e[2]; }

    vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    double operator[](int i) const { return e[i]; }
    double& operator[](int i) { return e[i]; }

    vec3& operator+=(const vec3& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    vec3& operator*=(double t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    vec3& operator/=(double t) {
        return *this *= 1/t;
    }

    double length() const {
        return std::sqrt(length_squared());
    }

    double length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }
};

// point3 is just an alias for vec3, but useful for geometric clarity in the code.
using point3 = vec3;

// Vector Utility Functions

inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

inline vec3 operator*(double t, const vec3& v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

inline vec3 operator*(const vec3& v, double t) {
    return t * v;
}

inline vec3 operator/(const vec3& v, double t) {
    return (1/t) * v;
}

inline double dot(const vec3& u, const vec3& v) {
    return u.e[0] * v.e[0]
         + u.e[1] * v.e[1]
         + u.e[2] * v.e[2];
}

inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

inline vec3 unit_vector(const vec3& v) {
    return v / v.length();
}

#endif
```

**列表 4:** `[vec3.h]` vec3 定义和辅助函数

我们在这里使用`double`，但一些光线追踪器使用`float`。`double`具有更高的精度和范围，但与`float`相比大小是其两倍。如果您的编程环境内存有限（例如硬件着色器），这种大小的增加可能很重要。两者都行——跟随您的个人喜好。

## 颜色实用函数

使用我们新的`vec3`类，我们将创建一个新的`color.h`头文件，并定义一个实用函数，该函数将单个像素的颜色写入标准输出流。

```
#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"

#include <iostream>

using color = vec3;

void write_color(std::ostream& out, const color& pixel_color) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // Translate the [0,1] component values to the byte range [0,255].
    int rbyte = int(255.999 * r);
    int gbyte = int(255.999 * g);
    int bbyte = int(255.999 * b);

    // Write out the pixel color components.
    out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}

#endif
```

**列表 5:** `[color.h]` 颜色实用函数现在我们可以更改我们的主程序来使用这两个：

```
#include "color.h"
#include "vec3.h"
#include <iostream>

int main() {

    // Image

    int image_width = 256;
    int image_height = 256;

    // Render

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = 0; j < image_height; j++) {
        std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
        for (int i = 0; i < image_width; i++) {            auto pixel_color = color(double(i)/(image_width-1), double(j)/(image_height-1), 0);
            write_color(std::cout, pixel_color);        }
    }

    std::clog << "\rDone.                 \n";
}
```

**列表 6:** `[main.cc]` 首个 PPM 图像的最终代码

你应该得到与之前完全相同的图片。

# 光线、简单相机和背景

## 光线类

所有光线追踪器都有的一个东西是一个射线类和计算沿射线所看到的颜色。让我们把射线想象成一个函数 <nobr aria-hidden="true">P(t)=A+tb</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">P</mi></mrow><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo><mo>=</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">A</mi></mrow><mo>+</mo><mi>t</mi><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">b</mi></mrow>. 这里 <nobr aria-hidden="true">P</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">P</mi></mrow> 是沿 3D 线的 3D 位置。 <nobr aria-hidden="true">A</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">A</mi></mrow> 是射线的起点，而 <nobr aria-hidden="true">b</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">b</mi></mrow> 是射线的方向。射线参数 <nobr aria-hidden="true">t</nobr><mi>t</mi> 是一个实数（代码中的 `double`）。插入不同的 <nobr aria-hidden="true">t</nobr><mi>t</mi> 值，<nobr aria-hidden="true">P(t)</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">P</mi></mrow><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo> 会沿着射线移动点。加入负的 <nobr aria-hidden="true">t</nobr><mi>t</mi> 值，你可以在 3D 线上的任何地方移动。对于正的 <nobr aria-hidden="true">t</nobr><mi>t</mi>，你只能得到 <nobr aria-hidden="true">A</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">A</mi></mrow> 前面的部分，这通常被称为半线或射线。

[![](https://raytracing.github.io/images/fig-1.02-lerp.jpg)](https://raytracing.github.io/images/fig-1.02-lerp.jpg)**图 2:** 线性插值我们可以将射线的概念表示为一个类，并将函数 <nobr aria-hidden="true">P(t)</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">P</mi></mrow><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo> 表示为我们将要称为 `ray::at(t)` 的函数：

```
#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class ray {
  public:
    ray() {}

    ray(const point3& origin, const vec3& direction) : orig(origin), dir(direction) {}

    const point3& origin() const { return orig; }
    const vec3& direction() const { return dir; }

    point3 at(double t) const {
        return orig + t*dir;
    }

  private:
    point3 orig;
    vec3 dir;
};

#endif
```

**列表 7:** `[ray.h]` 射线类

（对于不熟悉 C++ 的人来说，`ray::origin()` 和 `ray::direction()` 函数都返回对其成员的不可变引用。调用者可以直接使用该引用，或者根据需要创建一个可变副本。）

## 将射线发送到场景中

现在我们准备转弯并制作一个光线追踪器。其核心是，光线追踪器通过像素发送射线并计算这些射线的方向上所看到的颜色。涉及的步骤是

1.  计算从“眼睛”通过像素的射线

1.  确定射线与哪些对象相交，

1.  计算最近交点的颜色。

当首次开发光线追踪器时，我总是先做一个简单的相机，以便让代码运行起来。

我经常因为经常混淆 <nobr aria-hidden="true">x</nobr><mi>x</mi> 和 <nobr aria-hidden="true">y</nobr><mi>y</mi> 而陷入调试的麻烦，所以我们将使用非正方形图像。正方形图像的宽高比是 1:1，因为它的宽度和高度相同。由于我们想要非正方形图像，我们将选择 16:9，因为它非常常见。16:9 的宽高比意味着图像宽度和高度的比例是 16:9。换句话说，给定一个 16:9 宽高比的图像，

<nobr aria-hidden="true">宽/高=16/9=1.7778</nobr><mtext>宽</mtext><mrow class="MJX-TeXAtom-ORD"><mo>/</mo></mrow><mtext>高</mtext><mo>=</mo><mn>16</mn><mrow class="MJX-TeXAtom-ORD"><mo>/</mo></mrow><mn>9</mn><mo>=</mo><mn>1.7778</mn>

对于一个实际例子，一个宽度为 800 像素、高度为 400 像素的图像具有 2:1 的宽高比。

图像的宽高比可以通过其宽度和高度的比值来确定。然而，由于我们心中已经有了特定的宽高比，因此设置图像的宽度和宽高比会更简单，然后利用这个比值来计算其高度。这样，我们可以通过改变图像宽度来放大或缩小图像，而不会破坏我们想要的宽高比。我们确实需要确保在求解图像高度时，得到的高度至少为 1。

除了设置渲染图像的像素维度外，我们还需要设置一个虚拟的*视口*，通过它传递我们的场景光线。视口是 3D 世界中的一个虚拟矩形，包含图像像素位置的网格。如果像素在水平方向上的间距与垂直方向上的间距相同，那么包含它们的视口将具有与渲染图像相同的宽高比。两个相邻像素之间的距离称为像素间距，正方形像素是标准。

为了开始，我们将选择一个任意的视口高度为 2.0，并将视口宽度缩放以获得所需的宽高比。以下是这段代码的片段：

```
auto aspect_ratio = 16.0 / 9.0;
int image_width = 400;

// Calculate the image height, and ensure that it's at least 1.
int image_height = int(image_width / aspect_ratio);
image_height = (image_height < 1) ? 1 : image_height;

// Viewport widths less than one are ok since they are real valued.
auto viewport_height = 2.0;
auto viewport_width = viewport_height * (double(image_width)/image_height);
```

**列表 8：渲染图像设置**

如果你想知道为什么我们在计算`viewport_width`时不直接使用`aspect_ratio`，那是因为设置给`aspect_ratio`的值是理想比例，它可能不是`image_width`和`image_height`之间的*实际*比例。如果`image_height`被允许是实数值——而不仅仅是整数——那么使用`aspect_ratio`将是可行的。但是，`image_width`和`image_height`之间的*实际*比例可能会根据代码的两个部分而变化。首先，`image_height`会被四舍五入到最接近的整数，这可能会增加比例。其次，我们不允许`image_height`小于 1，这也可能改变实际的宽高比。

注意，`aspect_ratio` 是一个理想的比例，我们尽可能地用基于整数的图像宽度与高度的比率来近似。为了使我们的视口比例与图像比例完全匹配，我们使用计算出的图像宽高比来确定最终的视口宽度。

接下来我们将定义相机中心：一个在 3D 空间中的点，所有场景光线都将从这个点发出（这也通常被称为 *视点*）。从相机中心到视口中心的向量将与视口垂直。我们最初将视口和相机中心点之间的距离设置为 1 个单位。这个距离通常被称为 *焦距*。

为了简化，我们将以相机中心在 <nobr aria-hidden="true">(0,0,0)</nobr><mo stretchy="false">(</mo><mn>0</mn><mo>,</mo><mn>0</mn><mo>,</mo><mn>0</mn><mo stretchy="false">)</mo> 为起点。我们还将使 y 轴向上，x 轴向右，负 z 轴指向观察方向。（这通常被称为 *右手坐标系*。）

[![](https://raytracing.github.io/images/fig-1.03-cam-geom.jpg)](https://raytracing.github.io/images/fig-1.03-cam-geom.jpg)**图 3：** 相机几何

现在是不可避免的棘手部分。虽然我们的 3D 空间有上述惯例，但这与我们的图像坐标相冲突，我们希望在图像的左上角有零像素，然后向下到右下角的最后像素。这意味着我们的图像坐标 Y 轴是反转的：Y 值随着图像向下增加。

在扫描我们的图像时，我们将从左上角的像素（像素 <nobr aria-hidden="true">0,0</nobr><mn>0</mn><mo>,</mo><mn>0</mn>）开始，从左到右扫描每一行，然后逐行从上到下扫描。为了帮助导航像素网格，我们将使用从左边缘到右边缘的向量（<nobr aria-hidden="true">Vu</nobr><mrow class="MJX-TeXAtom-ORD"><msub><mi mathvariant="bold">V</mi><mi mathvariant="bold">u</mi></msub></mrow>），以及从上边缘到下边缘的向量（<nobr aria-hidden="true">Vv</nobr><mrow class="MJX-TeXAtom-ORD"><msub><mi mathvariant="bold">V</mi><mi mathvariant="bold">v</mi></msub></mrow>）。

我们的像素网格将嵌入视口边缘，距离为像素间距离的一半。这样，我们的视口区域就被均匀地分成了宽度 × 高度相同的区域。以下是我们的视口和像素网格的示意图：

[![](https://raytracing.github.io/images/fig-1.04-pixel-grid.jpg)](https://raytracing.github.io/images/fig-1.04-pixel-grid.jpg)**图 4：** 视口和像素网格

在这个图中，我们有视口、7×5 分辨率图像的像素网格、视口左上角 <nobr aria-hidden="true">Q</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">Q</mi></mrow>、像素位置 <nobr aria-hidden="true">P0,0</nobr><mrow class="MJX-TeXAtom-ORD"><msub><mi mathvariant="bold">P</mi><mrow class="MJX-TeXAtom-ORD"><mn mathvariant="bold">0</mn><mo mathvariant="bold">,</mo><mn mathvariant="bold">0</mn></mrow></msub></mrow>，视口向量 <nobr aria-hidden="true">Vu</nobr><mrow class="MJX-TeXAtom-ORD"><msub><mi mathvariant="bold">V</mi><mi mathvariant="bold">u</mi></msub></mrow> (`viewport_u`)，视口向量 <nobr aria-hidden="true">Vv</nobr><mrow class="MJX-TeXAtom-ORD"><msub><mi mathvariant="bold">V</mi><mi mathvariant="bold">v</mi></msub></mrow> (`viewport_v`)，以及像素增量向量 <nobr aria-hidden="true">Δu</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">Δ</mi><mi mathvariant="bold">u</mi></mrow> 和 <nobr aria-hidden="true">Δv</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">Δ</mi><mi mathvariant="bold">v</mi></mrow>.

从所有这些中汲取灵感，以下是实现相机的代码。我们将创建一个函数 `ray_color(const ray& r)`，该函数返回给定场景射线的颜色——我们目前将其设置为总是返回黑色。

```
#include "color.h"#include "ray.h"#include "vec3.h"

#include <iostream>

color ray_color(const ray& r) {
    return color(0,0,0);
}
int main() {

    // Image

    auto aspect_ratio = 16.0 / 9.0;
    int image_width = 400;

    // Calculate the image height, and ensure that it's at least 1.
    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    // Camera

    auto focal_length = 1.0;
    auto viewport_height = 2.0;
    auto viewport_width = viewport_height * (double(image_width)/image_height);
    auto camera_center = point3(0, 0, 0);

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    auto viewport_u = vec3(viewport_width, 0, 0);
    auto viewport_v = vec3(0, -viewport_height, 0);

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    auto pixel_delta_u = viewport_u / image_width;
    auto pixel_delta_v = viewport_v / image_height;

    // Calculate the location of the upper left pixel.
    auto viewport_upper_left = camera_center
                             - vec3(0, 0, focal_length) - viewport_u/2 - viewport_v/2;
    auto pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    // Render

    std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";

    for (int j = 0; j < image_height; j++) {
        std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
        for (int i = 0; i < image_width; i++) {            auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
            auto ray_direction = pixel_center - camera_center;
            ray r(camera_center, ray_direction);

            color pixel_color = ray_color(r);            write_color(std::cout, pixel_color);
        }
    }

    std::clog << "\rDone.                 \n";
}
```

**列表 9:** `[main.cc]` 创建场景射线

注意到在上述代码中，我没有将 `ray_direction` 设置为单位向量，因为我认为这样做可以使代码更简单且稍微快一点。

现在我们将填充 `ray_color(ray)` 函数以实现一个简单的渐变。这个函数将根据 <nobr aria-hidden="true">y</nobr><mi>y</mi> 坐标的高度线性混合白色和蓝色，在将射线方向缩放为单位长度之后（因此 <nobr aria-hidden="true">−1.0<y<1.0</nobr><mo>−</mo><mn>1.0</mn><mo><</mo><mi>y</mi><mo><</mo><mn>1.0</mn>）。因为我们是在规范化向量后查看 <nobr aria-hidden="true">y</nobr><mi>y</mi> 高度，所以你会注意到除了垂直渐变外，还有水平渐变到颜色中。

我将使用一个标准的图形技巧来线性缩放 <nobr aria-hidden="true">0.0≤a≤1.0</nobr><mn>0.0</mn><mo>≤</mo><mi>a</mi><mo>≤</mo><mn>1.0</mn>. 当 <nobr aria-hidden="true">a=1.0</nobr><mi>a</mi><mo>=</mo><mn>1.0</mn> 时，我想要蓝色。当 <nobr aria-hidden="true">a=0.0</nobr><mi>a</mi><mo>=</mo><mn>0.0</mn> 时，我想要白色。在两者之间，我想要混合。这形成了一个“线性混合”，或“线性插值”。这通常被称为两个值之间的 *lerp*。lerp 总是以下形式

<nobr aria-hidden="true">blendedValue=(1−a)⋅startValue+a⋅endValue,</nobr><mrow class="MJX-TeXAtom-ORD"><mi class="MJX-tex-mathit" mathvariant="italic">b</mi><mi class="MJX-tex-mathit" mathvariant="italic">l</mi><mi class="MJX-tex-mathit" mathvariant="italic">e</mi><mi class="MJX-tex-mathit" mathvariant="italic">n</mi><mi class="MJX-tex-mathit" mathvariant="italic">d</mi><mi class="MJX-tex-mathit" mathvariant="italic">e</mi><mi class="MJX-tex-mathit" mathvariant="italic">d</mi><mi class="MJX-tex-mathit" mathvariant="italic">V</mi><mi class="MJX-tex-mathit" mathvariant="italic">a</mi><mi class="MJX-tex-mathit" mathvariant="italic">l</mi><mi class="MJX-tex-mathit" mathvariant="italic">u</mi><mi class="MJX-tex-mathit" mathvariant="italic">e</mi></mrow><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mi>a</mi><mo stretchy="false">)</mo><mo>⋅</mo><mrow class="MJX-TeXAtom-ORD"><mi class="MJX-tex-mathit" mathvariant="italic">s</mi><mi class="MJX-tex-mathit" mathvariant="italic">t</mi><mi class="MJX-tex-mathit" mathvariant="italic">a</mi><mi class="MJX-tex-mathit" mathvariant="italic">r</mi><mi class="MJX-tex-mathit" mathvariant="italic">t</mi><mi class="MJX-tex-mathit" mathvariant="italic">V</mi><mi class="MJX-tex-mathit" mathvariant="italic">a</mi><mi class="MJX-tex-mathit" mathvariant="italic">l</mi><mi class="MJX-tex-mathit" mathvariant="italic">u</mi><mi class="MJX-tex-mathit" mathvariant="italic">e</mi></mrow><mo>+</mo><mi>a</mi><mo>⋅</mo><mrow class="MJX-TeXAtom-ORD"><mi class="MJX-tex-mathit" mathvariant="italic">e</mi><mi class="MJX-tex-mathit" mathvariant="italic">n</mi><mi class="MJX-tex-mathit" mathvariant="italic">d</mi><mi class="MJX-tex-mathit" mathvariant="italic">V</mi><mi class="MJX-tex-mathit" mathvariant="italic">a</mi><mi class="MJX-tex-mathit" mathvariant="italic">l</mi><mi class="MJX-tex-mathit" mathvariant="italic">u</mi><mi class="MJX-tex-mathit" mathvariant="italic">e</mi></mrow><mo>,</mo>

其中 <nobr aria-hidden="true">a</nobr><mi>a</mi> 从零到一变化。

将所有这些放在一起，我们得到以下结果：

```
#include "color.h"
#include "ray.h"
#include "vec3.h"

#include <iostream>

color ray_color(const ray& r) {    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5*(unit_direction.y() + 1.0);
    return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);}

...
```

**列表 10:** `[main.cc]` 渲染从蓝色到白色的渐变。在我们的情况下，这会产生：[![](https://raytracing.github.io/images/img-1.02-blue-to-white.png)](https://raytracing.github.io/images/img-1.02-blue-to-white.png)图 2：根据射线 Y 坐标的蓝色到白色渐变

# 添加球体

让我们在我们的光线追踪器中添加一个单独的对象。人们经常在光线追踪器中使用球体，因为计算射线是否击中球体相对简单。

## 射线-球面交点

以原点为中心的半径为 <nobr aria-hidden="true">r</nobr><mi>r</mi> 的球体方程是一个重要的数学方程：

<nobr aria-hidden="true">x2+y2+z2=r2</nobr><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><msup><mi>y</mi><mn>2</mn></msup><mo>+</mo><msup><mi>z</mi><mn>2</mn></msup><mo>=</mo><msup><mi>r</mi><mn>2</mn></msup>

你也可以这样理解，如果给定点 <nobr aria-hidden="true">(x,y,z)</nobr><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>,</mo><mi>z</mi><mo stretchy="false">)</mo> 在球面上，那么 <nobr aria-hidden="true">x2+y2+z2=r2</nobr><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><msup><mi>y</mi><mn>2</mn></msup><mo>+</mo><msup><mi>z</mi><mn>2</mn></msup><mo>=</mo><msup><mi>r</mi><mn>2</mn></msup>. 如果给定点 <nobr aria-hidden="true">(x,y,z)</nobr><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>,</mo><mi>z</mi><mo stretchy="false">)</mo> 在球内，那么 <nobr aria-hidden="true">x2+y2+z2<r2</nobr><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><msup><mi>y</mi><mn>2</mn></msup><mo>+</mo><msup><mi>z</mi><mn>2</mn></msup><mo><</mo><msup><mi>r</mi><mn>2</mn></msup>, 如果给定点 <nobr aria-hidden="true">(x,y,z)</nobr><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>,</mo><mi>z</mi><mo stretchy="false">)</mo> 在球外，那么 <nobr aria-hidden="true">x2+y2+z2>r2</nobr><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><msup><mi>y</mi><mn>2</mn></msup><mo>+</mo><msup><mi>z</mi><mn>2</mn></msup><mo>></mo><msup><mi>r</mi><mn>2</mn></msup>.

如果我们想要允许球心位于任意点 <nobr aria-hidden="true">(Cx,Cy,Cz)</nobr><mo stretchy="false">(</mo><msub><mi>C</mi><mi>x</mi></msub><mo>,</mo><msub><mi>C</mi><mi>y</mi></msub><mo>,</mo><msub><mi>C</mi><mi>z</mi></msub><mo stretchy="false">)</mo>，那么方程就变得不那么简洁了：

<nobr aria-hidden="true">(Cx−x)2+(Cy−y)2+(Cz−z)2=r2</nobr><mo stretchy="false">(</mo><msub><mi>C</mi><mi>x</mi></msub><mo>−</mo><mi>x</mi><msup><mo stretchy="false">)</mo><mn>2</mn></msup><mo>+</mo><mo stretchy="false">(</mo><msub><mi>C</mi><mi>y</mi></msub><mo>−</mo><mi>y</mi><msup><mo stretchy="false">)</mo><mn>2</mn></msup><mo>+</mo><mo stretchy="false">(</mo><msub><mi>C</mi><mi>z</mi></msub><mo>−</mo><mi>z</mi><msup><mo stretchy="false">)</mo><mn>2</mn></msup><mo>=</mo><msup><mi>r</mi><mn>2</mn></msup>

在图形学中，你几乎总是希望你的公式以向量的形式表示，这样所有关于 <nobr aria-hidden="true">x</nobr><mi>x</mi>/<nobr aria-hidden="true">y</nobr><mi>y</mi>/<nobr aria-hidden="true">z</nobr><mi>z</mi> 的内容都可以简单地使用一个 `vec3` 类来表示。你可能注意到，从点 <nobr aria-hidden="true">P=(x,y,z)</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">P</mi></mrow><mo>=</mo><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>,</mo><mi>z</mi><mo stretchy="false">)</mo> 到中心 <nobr aria-hidden="true">C=(Cx,Cy,Cz)</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">C</mi></mrow><mo>=</mo><mo stretchy="false">(</mo><msub><mi>C</mi><mi>x</mi></msub><mo>,</mo><msub><mi>C</mi><mi>y</mi></msub><mo>,</mo><msub><mi>C</mi><mi>z</mi></msub><mo stretchy="false">)</mo> 的向量是 <nobr aria-hidden="true">(C−P)</nobr><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">C</mi></mrow><mo>−</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">P</mi></mrow><mo stretchy="false">)</mo>.

如果我们使用点积的定义：

<nobr aria-hidden="true">(C−P)⋅(C−P)=(Cx−x)2+(Cy−y)2+(Cz−z)2</nobr><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">C</mi></mrow><mo>−</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">P</mi></mrow><mo stretchy="false">)</mo><mo>⋅</mo><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">C</mi></mrow><mo>−</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">P</mi></mrow><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><msub><mi>C</mi><mi>x</mi></msub><mo>−</mo><mi>x</mi><msup><mo stretchy="false">)</mo><mn>2</mn></msup><mo>+</mo><mo stretchy="false">(</mo><msub><mi>C</mi><mi>y</mi></msub><mo>−</mo><mi>y</mi><msup><mo stretchy="false">)</mo><mn>2</mn></msup><mo>+</mo><mo stretchy="false">(</mo><msub><mi>C</mi><mi>z</mi></msub><mo>−</mo><mi>z</mi><msup><mo stretchy="false">)</mo><mn>2</mn></msup>

然后，我们可以将球面的方程重写为向量形式：

<nobr aria-hidden="true">(C−P)⋅(C−P)=r2</nobr><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">C</mi></mrow><mo>−</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">P</mi></mrow><mo stretchy="false">)</mo><mo>⋅</mo><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">C</mi></mrow><mo>−</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">P</mi></mrow><mo stretchy="false">)</mo><mo>=</mo><msup><mi>r</mi><mn>2</mn></msup>

我们可以将其读作“任何满足此方程的点 <nobr aria-hidden="true">P</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">P</mi></mrow> 都位于球面上”。我们想知道我们的射线 <nobr aria-hidden="true">P(t)=Q+td</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">P</mi></mrow><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo><mo>=</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">Q</mi></mrow><mo>+</mo><mi>t</mi><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">d</mi></mrow> 是否在任何地方击中球面。如果它击中了球面，那么存在某个 <nobr aria-hidden="true">t</nobr><mi>t</mi>，使得 <nobr aria-hidden="true">P(t)</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">P</mi></mrow><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo> 满足球面方程。因此，我们正在寻找任何 <nobr aria-hidden="true">t</nobr><mi>t</mi>，其中这是真的：

<nobr aria-hidden="true">(C−P(t))⋅(C−P(t))=r2</nobr><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">C</mi></mrow><mo>−</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">P</mi></mrow><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo><mo stretchy="false">)</mo><mo>⋅</mo><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">C</mi></mrow><mo>−</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">P</mi></mrow><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo><mo stretchy="false">)</mo><mo>=</mo><msup><mi>r</mi><mn>2</mn></msup>

可以通过将 <nobr aria-hidden="true">P(t)</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">P</mi></mrow><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo> 替换为其展开形式来找到：

<nobr aria-hidden="true">(C−(Q+td))⋅(C−(Q+td))=r2</nobr><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">C</mi></mrow><mo>−</mo><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">Q</mi></mrow><mo>+</mo><mi>t</mi><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">d</mi></mrow><mo stretchy="false">)</mo><mo stretchy="false">)</mo><mo>⋅</mo><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">C</mi></mrow><mo>−</mo><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">Q</mi></mrow><mo>+</mo><mi>t</mi><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">d</mi></mrow><mo stretchy="false">)</mo><mo stretchy="false">)</mo><mo>=</mo><msup><mi>r</mi><mn>2</mn></msup>

我们有三个向量在左边与右边的三个向量点乘。如果我们求解完整的点积，我们会得到九个向量。你当然可以逐个写出来，但我们不需要那么辛苦。如果你记得，我们想要求解的是 <nobr aria-hidden="true">t</nobr><mi>t</mi>，所以我们将项根据是否有 <nobr aria-hidden="true">t</nobr><mi>t</mi> 来分开：

<nobr aria-hidden="true">(−td+(C−Q))⋅(−td+(C−Q))=r2</nobr><mo stretchy="false">(</mo><mo>−</mo><mi>t</mi><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">d</mi></mrow><mo>+</mo><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">C</mi></mrow><mo>−</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">Q</mi></mrow><mo stretchy="false">)</mo><mo stretchy="false">)</mo><mo>⋅</mo><mo stretchy="false">(</mo><mo>−</mo><mi>t</mi><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">d</mi></mrow><mo>+</mo><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">C</mi></mrow><mo>−</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">Q</mi></mrow><mo stretchy="false">)</mo><mo stretchy="false">)</mo><mo>=</mo><msup><mi>r</mi><mn>2</mn></msup>

现在我们遵循向量代数的规则来分配点积：

<nobr aria-hidden="true">t2d⋅d−2td⋅(C−Q)+(C−Q)⋅(C−Q)=r2</nobr><msup><mi>t</mi><mn>2</mn></msup><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">d</mi></mrow><mo>⋅</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">d</mi></mrow><mo>−</mo><mn>2</mn><mi>t</mi><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">d</mi></mrow><mo>⋅</mo><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">C</mi></mrow><mo>−</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">Q</mi></mrow><mo stretchy="false">)</mo><mo>+</mo><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">C</mi></mrow><mo>−</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">Q</mi></mrow><mo stretchy="false">)</mo><mo>⋅</mo><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">C</mi></mrow><mo>−</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">Q</mi></mrow><mo stretchy="false">)</mo><mo>=</mo><msup><mi>r</mi><mn>2</mn></msup>

将半径的平方移到左边：

<nobr aria-hidden="true">t2d⋅d−2td⋅(C−Q)+(C−Q)⋅(C−Q)−r2=0</nobr><msup><mi>t</mi><mn>2</mn></msup><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">d</mi></mrow><mo>⋅</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">d</mi></mrow><mo>−</mo><mn>2</mn><mi>t</mi><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">d</mi></mrow><mo>⋅</mo><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">C</mi></mrow><mo>−</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">Q</mi></mrow><mo stretchy="false">)</mo><mo>+</mo><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">C</mi></mrow><mo>−</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">Q</mi></mrow><mo stretchy="false">)</mo><mo>⋅</mo><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">C</mi></mrow><mo>−</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">Q</mi></mrow><mo stretchy="false">)</mo><mo>−</mo><msup><mi>r</mi><mn>2</mn></msup><mo>=</mo><mn>0</mn>

很难看清楚这个方程具体是什么，但方程中的向量和 <nobr aria-hidden="true">r</nobr><mi>r</mi> 都是常数且已知的。此外，我们拥有的唯一向量通过点积被简化为标量。唯一的未知数是 <nobr aria-hidden="true">t</nobr><mi>t</mi>，我们有一个 <nobr aria-hidden="true">t2</nobr><msup><mi>t</mi><mn>2</mn></msup>，这意味着这个方程是二次的。你可以通过使用二次公式来解二次方程 <nobr aria-hidden="true">ax2+bx+c=0</nobr><mi>a</mi><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><mi>b</mi><mi>x</mi><mo>+</mo><mi>c</mi><mo>=</mo><mn>0</mn>：

<nobr aria-hidden="true">−b±b2−4ac−−−−−−−√2a</nobr><mfrac><mrow><mo>−</mo><mi>b</mi><mo>±</mo><msqrt><msup><mi>b</mi><mn>2</mn></msup><mo>−</mo><mn>4</mn><mi>a</mi><mi>c</mi></msqrt></mrow><mrow><mn>2</mn><mi>a</mi></mrow></mfrac>

因此，在求解射线与球面交点方程中的 <nobr aria-hidden="true">t</nobr><mi>t</mi> 时，我们得到了以下 <nobr aria-hidden="true">a</nobr><mi>a</mi>、<nobr aria-hidden="true">b</nobr><mi>b</mi> 和 <nobr aria-hidden="true">c</nobr><mi>c</mi> 的值：

<nobr aria-hidden="true">a=d⋅d</nobr><mi>a</mi><mo>=</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">d</mi></mrow><mo>⋅</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">d</mi></mrow><nobr aria-hidden="true">b=−2d⋅(C−Q)</nobr><mi>b</mi><mo>=</mo><mo>−</mo><mn>2</mn><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">d</mi></mrow><mo>⋅</mo><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">C</mi></mrow><mo>−</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">Q</mi></mrow><mo stretchy="false">)</mo><nobr aria-hidden="true">c=(C−Q)⋅(C−Q)−r2</nobr><mi>c</mi><mo>=</mo><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">C</mi></mrow><mo>−</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">Q</mi></mrow><mo stretchy="false">)</mo><mo>⋅</mo><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">C</mi></mrow><mo>−</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">Q</mi></mrow><mo stretchy="false">)</mo><mo>−</mo><msup><mi>r</mi><mn>2</mn></msup>

使用上述所有内容，你可以解出 <nobr aria-hidden="true">t</nobr><mi>t</mi>，但其中有一个平方根部分可以是正数（意味着有两个实数解），负数（意味着没有实数解），或者零（意味着有一个实数解）。在图形学中，代数几乎总是直接与几何相关。我们得到的是：

[![](https://raytracing.github.io/images/fig-1.05-ray-sphere.jpg)](https://raytracing.github.io/images/fig-1.05-ray-sphere.jpg)**图 5:** 光线与球体交点结果

## 创建我们的第一个光线追踪图像

如果我们将这个数学公式硬编码到我们的程序中，我们可以通过在 z 轴上放置一个小球体在-1 的位置，然后为与之相交的任何像素着色红色来测试我们的代码。

```
bool hit_sphere(const point3& center, double radius, const ray& r) {
    vec3 oc = center - r.origin();
    auto a = dot(r.direction(), r.direction());
    auto b = -2.0 * dot(r.direction(), oc);
    auto c = dot(oc, oc) - radius*radius;
    auto discriminant = b*b - 4*a*c;
    return (discriminant >= 0);
}
color ray_color(const ray& r) {    if (hit_sphere(point3(0,0,-1), 0.5, r))
        return color(1, 0, 0);
    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5*(unit_direction.y() + 1.0);
    return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
}
```

**列表 11:** `[main.cc]` 渲染红色球体我们得到的是这个：[![](https://raytracing.github.io/images/img-1.03-red-sphere.png)](https://raytracing.github.io/images/img-1.03-red-sphere.png)图像 3：一个简单的红色球体

现在缺少各种东西——比如着色、反射光线和多个对象——但我们已经比开始时更接近完成一半了！需要注意的一点是，我们通过解二次方程来测试光线是否与球体相交，并查看是否存在解，但带有负值 <nobr aria-hidden="true">t</nobr><mi>t</mi> 的解同样有效。如果你将球体中心改为 <nobr aria-hidden="true">z=+1</nobr><mi>z</mi><mo>=</mo><mo>+</mo><mn>1</mn>，你将得到完全相同的图像，因为此解无法区分 *相机前面的对象* 和 *相机后面的对象*。这不是一个特性！我们将在下一部分修复这些问题。

# 表面法线和多个对象

## 使用表面法线进行着色

首先，让我们获取一个表面法线，以便我们可以进行着色。这是一个垂直于交点处的表面的向量。

对于我们的代码中的法向量，我们需要做出一个关键的设计决策：法向量将具有任意长度，还是将被归一化到单位长度。

跳过在归一化向量时涉及的开方运算似乎很有吸引力，以防万一不需要它。然而，在实践中，有三个重要的观察。首先，如果需要单位长度的法向量，那么最好一开始就一次性完成，而不是每次需要单位长度时都反复“以防万一”。其次，我们在几个地方确实需要单位长度的法向量。第三，如果你需要法向量具有单位长度，那么通常可以通过理解特定的几何类，在其构造函数中或在`hit()`函数中有效地生成该向量。例如，通过除以球体半径，可以简单地使球面法向量成为单位长度，从而完全避免开方运算。

考虑到所有这些，我们将采用所有法向量都将具有单位长度的策略。

对于球体，外向法向量是指向入射点减去中心的方向：

[![](https://raytracing.github.io/images/fig-1.06-sphere-normal.jpg)](https://raytracing.github.io/images/fig-1.06-sphere-normal.jpg)**图 6：** 球面法向量几何

在地球上，这意味着从地球中心到你的向量直接向上。现在让我们将其加入代码中，并对其进行着色。我们目前还没有任何光源或其他东西，所以让我们用颜色图来可视化法线。用于可视化法线的一个常见技巧（因为它简单且直观，可以假设 <nobr aria-hidden="true">n</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">n</mi></mrow> 是一个单位长度向量——因此每个分量都在 -1 和 1 之间）是将每个分量映射到 0 到 1 的区间，然后将 <nobr aria-hidden="true">(x,y,z)</nobr><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>,</mo><mi>z</mi><mo stretchy="false">)</mo> 映射到 <nobr aria-hidden="true">(red,green,blue)</nobr><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi class="MJX-tex-mathit" mathvariant="italic">r</mi><mi class="MJX-tex-mathit" mathvariant="italic">e</mi><mi class="MJX-tex-mathit" mathvariant="italic">d</mi></mrow><mo>,</mo><mrow class="MJX-TeXAtom-ORD"><mi class="MJX-tex-mathit" mathvariant="italic">g</mi><mi class="MJX-tex-mathit" mathvariant="italic">r</mi><mi class="MJX-tex-mathit" mathvariant="italic">e</mi><mi class="MJX-tex-mathit" mathvariant="italic">e</mi><mi class="MJX-tex-mathit" mathvariant="italic">n</mi></mrow><mo>,</mo><mrow class="MJX-TeXAtom-ORD"><mi class="MJX-tex-mathit" mathvariant="italic">b</mi><mi class="MJX-tex-mathit" mathvariant="italic">l</mi><mi class="MJX-tex-mathit" mathvariant="italic">u</mi><mi class="MJX-tex-mathit" mathvariant="italic">e</mi></mrow><mo stretchy="false">)</mo>. 对于法线，我们需要击中点，而不仅仅是是否击中（这是我们目前所计算的全部）。场景中只有一个球体，并且它正位于摄像机前方，所以我们暂时不用担心 <nobr aria-hidden="true">t</nobr><mi>t</mi> 的负值。我们只需假设最近的击中点（最小的 <nobr aria-hidden="true">t</nobr><mi>t</mi>）是我们想要的点。这些代码中的更改使我们能够计算和可视化 <nobr aria-hidden="true">n</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">n</mi></mrow>:

```
double hit_sphere(const point3& center, double radius, const ray& r) {    vec3 oc = center - r.origin();
    auto a = dot(r.direction(), r.direction());
    auto b = -2.0 * dot(r.direction(), oc);
    auto c = dot(oc, oc) - radius*radius;
    auto discriminant = b*b - 4*a*c;

    if (discriminant < 0) {
        return -1.0;
    } else {
        return (-b - std::sqrt(discriminant) ) / (2.0*a);
    }}

color ray_color(const ray& r) {    auto t = hit_sphere(point3(0,0,-1), 0.5, r);
    if (t > 0.0) {
        vec3 N = unit_vector(r.at(t) - vec3(0,0,-1));
        return 0.5*color(N.x()+1, N.y()+1, N.z()+1);
    }
    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5*(unit_direction.y() + 1.0);
    return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
}
```

**列表 12:** `[main.cc]` 在球面上渲染表面法线

## 简化射线-球面交点代码

让我们重新审视射线-球面函数：

```
double hit_sphere(const point3& center, double radius, const ray& r) {
    vec3 oc = center - r.origin();
    auto a = dot(r.direction(), r.direction());
    auto b = -2.0 * dot(r.direction(), oc);
    auto c = dot(oc, oc) - radius*radius;
    auto discriminant = b*b - 4*a*c;

    if (discriminant < 0) {
        return -1.0;
    } else {
        return (-b - std::sqrt(discriminant) ) / (2.0*a);
    }
}
```

**列表 13:** `[main.cc]` 射线-球面交点代码（之前）

首先回忆一下，一个向量与自身的点积等于该向量的长度的平方。

其次，注意 `b` 的方程中有一个负二的因子。考虑如果 <nobr aria-hidden="true">b=−2h</nobr><mi>b</mi><mo>=</mo><mo>−</mo><mn>2</mn><mi>h</mi> 发生在二次方程上会发生什么：

<nobr aria-hidden="true">−b±b2−4ac−−−−−−−√2a</nobr><mfrac><mrow><mo>−</mo><mi>b</mi><mo>±</mo><msqrt><msup><mi>b</mi><mn>2</mn></msup><mo>−</mo><mn>4</mn><mi>a</mi><mi>c</mi></msqrt></mrow><mrow><mn>2</mn><mi>a</mi></mrow></mfrac><nobr aria-hidden="true">=−(−2h)±(−2h)2−4ac−−−−−−−−−−−√2a</nobr><mo>=</mo><mfrac><mrow><mo>−</mo><mo stretchy="false">(</mo><mo>−</mo><mn>2</mn><mi>h</mi><mo stretchy="false">)</mo><mo>±</mo><msqrt><mo stretchy="false">(</mo><mo>−</mo><mn>2</mn><mi>h</mi><msup><mo stretchy="false">)</mo><mn>2</mn></msup><mo>−</mo><mn>4</mn><mi>a</mi><mi>c</mi></msqrt></mrow><mrow><mn>2</mn><mi>a</mi></mrow></mfrac><nobr aria-hidden="true">=2h±2h2−ac−−−−−−√2a</nobr><mo>=</mo><mfrac><mrow><mn>2</mn><mi>h</mi><mo>±</mo><mn>2</mn><msqrt><msup><mi>h</mi><mn>2</mn></msup><mo>−</mo><mi>a</mi><mi>c</mi></msqrt></mrow><mrow><mn>2</mn><mi>a</mi></mrow></mfrac><nobr aria-hidden="true">=h±h2−ac−−−−−−√a</nobr><mo>=</mo><mfrac><mrow><mi>h</mi><mo>±</mo><msqrt><msup><mi>h</mi><mn>2</mn></msup><mo>−</mo><mi>a</mi><mi>c</mi></msqrt></mrow><mi>a</mi></mfrac>

这简化得很好，所以我们将使用它。因此，求解 <nobr aria-hidden="true">h</nobr><mi>h</mi>：

<nobr aria-hidden="true">b=−2d⋅(C−Q)</nobr><mi>b</mi><mo>=</mo><mo>−</mo><mn>2</mn><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">d</mi></mrow><mo>⋅</mo><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">C</mi></mrow><mo>−</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">Q</mi></mrow><mo stretchy="false">)</mo><nobr aria-hidden="true">b=−2h</nobr><mi>b</mi><mo>=</mo><mo>−</mo><mn>2</mn><mi>h</mi><nobr aria-hidden="true">h=b−2=d⋅(C−Q)</nobr><mi>h</mi><mo>=</mo><mfrac><mi>b</mi><mrow><mo>−</mo><mn>2</mn></mrow></mfrac><mo>=</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">d</mi></mrow><mo>⋅</mo><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">C</mi></mrow><mo>−</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">Q</mi></mrow><mo stretchy="false">)</mo>使用这些观察结果，我们现在可以将球面交点代码简化为以下形式：

```
double hit_sphere(const point3& center, double radius, const ray& r) {
    vec3 oc = center - r.origin();    auto a = r.direction().length_squared();
    auto h = dot(r.direction(), oc);
    auto c = oc.length_squared() - radius*radius;
    auto discriminant = h*h - a*c;
    if (discriminant < 0) {
        return -1.0;
    } else {        return (h - std::sqrt(discriminant)) / a;    }
}
```

**列表 14:** `[main.cc]` 射线与球面交点代码（之后）

## Hittable 对象的抽象

现在，如果有多个球面呢？虽然使用球面数组很诱人，但一个更干净的方法是创建一个“抽象类”，用于任何射线可能击中的对象，并使球面和球面列表都成为可以被击中的对象。这个类应该叫什么名字有些令人困惑——如果它不是面向对象的编程，那么叫“对象”会很好。“表面”经常被使用，但缺点可能是我们可能还想有体积（雾、云等）。 “可击中”强调将它们联系在一起的成员函数。我不喜欢这些中的任何一个，但我们将使用“可击中”。

这个`hittable`抽象类将有一个接受光线作为参数的`hit`函数。大多数光线追踪器发现添加一个有效的碰撞时间间隔（`<nobr aria-hidden="true">tmin</nobr><msub><mi>t</mi><mrow class="MJX-TeXAtom-ORD"><mrow class="MJX-TeXAtom-ORD"><mi class="MJX-tex-mathit" mathvariant="italic">m</mi><mi class="MJX-tex-mathit" mathvariant="italic">i</mi><mi class="MJX-tex-mathit" mathvariant="italic">n</mi></mrow></mrow></msub>`到`<nobr aria-hidden="true">tmax</nobr><msub><mi>t</mi><mrow class="MJX-TeXAtom-ORD"><mrow class="MJX-TeXAtom-ORD"><mi class="MJX-tex-mathit" mathvariant="italic">m</mi><mi class="MJX-tex-mathit" mathvariant="italic">a</mi><mi class="MJX-tex-mathit" mathvariant="italic">x</mi></mrow></mrow></msub>`的有效区间很方便，这样碰撞只有在`<nobr aria-hidden="true">tmin<t<tmax</nobr><msub><mi>t</mi><mrow class="MJX-TeXAtom-ORD"><mrow class="MJX-TeXAtom-ORD"><mi class="MJX-tex-mathit" mathvariant="italic">m</mi><mi class="MJX-tex-mathit" mathvariant="italic">i</mi><mi class="MJX-tex-mathit" mathvariant="italic">n</mi></mrow></mrow></msub><mo><</mo><mi>t</mi><mo><</mo><msub><mi>t</mi><mrow class="MJX-TeXAtom-ORD"><mrow class="MJX-TeXAtom-ORD"><mi class="MJX-tex-mathit" mathvariant="italic">m</mi><mi class="MJX-tex-mathit" mathvariant="italic">a</mi><mi class="MJX-tex-mathit" mathvariant="italic">x</mi></mrow></mrow></msub>`时才“计算”在内。对于初始光线，这是正的`<nobr aria-hidden="true">t</nobr><mi>t</mi>`，但正如我们将看到的，将时间间隔`<nobr aria-hidden="true">tmin</nobr><msub><mi>t</mi><mrow class="MJX-TeXAtom-ORD"><mrow class="MJX-TeXAtom-ORD"><mi class="MJX-tex-mathit" mathvariant="italic">m</mi><mi class="MJX-tex-mathit" mathvariant="italic">i</mi><mi class="MJX-tex-mathit" mathvariant="italic">n</mi></mrow></mrow></msub>`到`<nobr aria-hidden="true">tmax</nobr><msub><mi>t</mi><mrow class="MJX-TeXAtom-ORD"><mrow class="MJX-TeXAtom-ORD"><mi class="MJX-tex-mathit" mathvariant="italic">m</mi><mi class="MJX-tex-mathit" mathvariant="italic">a</mi><mi class="MJX-tex-mathit" mathvariant="italic">x</mi></mrow></mrow></msub>`简化我们的代码是有帮助的。一个设计问题是，如果我们碰撞到某个物体，是否要计算法线。在我们进行搜索的过程中，我们可能会遇到更近的物体，我们只需要最近物体的法线。我将采用简单的解决方案，计算一些我将存储在某种结构中的东西。以下是抽象类：

```
#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"

class hit_record {
  public:
    point3 p;
    vec3 normal;
    double t;
};

class hittable {
  public:
    virtual ~hittable() = default;

    virtual bool hit(const ray& r, double ray_tmin, double ray_tmax, hit_record& rec) const = 0;
};

#endif
```

**列表 15**：`[hittable.h]` 可碰撞类，以及这里的球体：

```
#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "vec3.h"

class sphere : public hittable {
  public:
    sphere(const point3& center, double radius) : center(center), radius(std::fmax(0,radius)) {}

    bool hit(const ray& r, double ray_tmin, double ray_tmax, hit_record& rec) const override {
        vec3 oc = center - r.origin();
        auto a = r.direction().length_squared();
        auto h = dot(r.direction(), oc);
        auto c = oc.length_squared() - radius*radius;

        auto discriminant = h*h - a*c;
        if (discriminant < 0)
            return false;

        auto sqrtd = std::sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        auto root = (h - sqrtd) / a;
        if (root <= ray_tmin || ray_tmax <= root) {
            root = (h + sqrtd) / a;
            if (root <= ray_tmin || ray_tmax <= root)
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        rec.normal = (rec.p - center) / radius;

        return true;
    }

  private:
    point3 center;
    double radius;
};

#endif
```

**列表 16**：`[sphere.h]` 球体类

（注意这里我们使用 C++标准函数`std::fmax()`，它返回两个浮点数参数中的最大值。同样，我们稍后还会使用`std::fmin()`，它返回两个浮点数参数中的最小值。）

## 前面与后面

正常向量的第二个设计决策是它们是否应该始终指向外部。目前，找到的正常向量始终指向中心到交点的方向（正常向量指向外部）。如果光线从外部与球体相交，则正常向量与光线方向相反。如果光线从内部与球体相交，则正常向量（始终指向外部）与光线方向相同。或者，我们也可以让正常向量始终与光线方向相反。如果光线在球体外部，则正常向量指向外部，但如果光线在球体内部，则正常向量指向内部。

![图 7：球面表面法线几何的可能方向](https://raytracing.github.io/images/fig-1.07-normal-sides.jpg)**图 7：** 球面表面法线几何的可能方向

我们需要选择这些可能性之一，因为我们最终想要确定光线来自表面的哪一侧。这对于在每一侧渲染不同的物体很重要，比如两面纸上的文字，或者对于具有内外侧的物体，如玻璃球。

如果我们决定让法线始终指向外部，那么在着色时我们需要确定光线所在的一侧。我们可以通过比较光线与法线来解决这个问题。如果光线和法线方向相同，则光线在物体内部；如果光线和法线方向相反，则光线在物体外部。这可以通过计算两个向量的点积来确定，如果点积为正，则光线在球体内部。

```
if (dot(ray_direction, outward_normal) > 0.0) {
    // ray is inside the sphere
    ...
} else {
    // ray is outside the sphere
    ...
}
```

**列表 17：比较光线和法线**如果我们决定让法线始终指向光线方向，我们就无法使用点积来确定光线在表面的哪一侧。相反，我们需要存储该信息：

```
bool front_face;
if (dot(ray_direction, outward_normal) > 0.0) {
    // ray is inside the sphere
    normal = -outward_normal;
    front_face = false;
} else {
    // ray is outside the sphere
    normal = outward_normal;
    front_face = true;
}
```

**列表 18：记住表面的侧面**

我们可以设置法线始终指向表面“外部”或始终指向入射光线。这个决策取决于你是在几何相交时还是着色时确定表面的一侧。在这本书中，我们拥有的材质类型比几何类型多，所以我们将选择更简单的方法，在几何时间进行确定。这仅仅是一个偏好问题，你将在文献中看到两种实现。

我们将 `front_face` 布尔值添加到 `hit_record` 类中。我们还将添加一个函数来帮助我们进行这个计算：`set_face_normal()`。为了方便，我们将假设传递给新 `set_face_normal()` 函数的向量是单位长度。我们始终可以显式地规范化参数，但如果几何代码这样做更有效，因为通常在了解特定几何时更容易。

```
class hit_record {
  public:
    point3 p;
    vec3 normal;
    double t;    bool front_face;

    void set_face_normal(const ray& r, const vec3& outward_normal) {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to have unit length.

        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }};
```

**列表 19：`[hittable.h]` 添加前侧面跟踪到 hit_record 然后我们将表面侧面确定添加到类中：

```
class sphere : public hittable {
  public:
    ...
    bool hit(const ray& r, double ray_tmin, double ray_tmax, hit_record& rec) const {
        ...

        rec.t = root;
        rec.p = r.at(rec.t);        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        return true;
    }
    ...
};
```

**列表 20**：`[sphere.h]` 具有正常确定的球类

## 可击中对象的列表

我们有一个名为 `hittable` 的通用对象，射线可以与之相交。我们现在添加一个存储 `hittable` 列表的类：

```
#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"

#include <memory>
#include <vector>

using std::make_shared;
using std::shared_ptr;

class hittable_list : public hittable {
  public:
    std::vector<shared_ptr<hittable>> objects;

    hittable_list() {}
    hittable_list(shared_ptr<hittable> object) { add(object); }

    void clear() { objects.clear(); }

    void add(shared_ptr<hittable> object) {
        objects.push_back(object);
    }

    bool hit(const ray& r, double ray_tmin, double ray_tmax, hit_record& rec) const override {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_tmax;

        for (const auto& object : objects) {
            if (object->hit(r, ray_tmin, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }
};

#endif
```

**列表 21**：`[hittable_list.h]` hittable_list 类

## 一些新的 C++ 特性

`hittable_list` 类的代码使用了某些可能让你感到困惑的 C++ 特性：`vector`、`shared_ptr` 和 `make_shared`。

`shared_ptr<type>` 是指向已分配类型的指针，具有引用计数语义。每次将它的值赋给另一个共享指针（通常使用简单的赋值），引用计数就会增加。当共享指针超出作用域（例如在块或函数的末尾）时，引用计数就会减少。一旦计数达到零，对象就会被安全地删除。

通常，共享指针首先用新分配的对象进行初始化，例如：

```
shared_ptr<double> double_ptr = make_shared<double>(0.37);
shared_ptr<vec3>   vec3_ptr   = make_shared<vec3>(1.414214, 2.718281, 1.618034);
shared_ptr<sphere> sphere_ptr = make_shared<sphere>(point3(0,0,0), 1.0);
```

**列表 22**：使用 `shared_ptr` 的一个示例分配

`make_shared<thing>(thing_constructor_params ...)` 使用构造函数参数分配 `thing` 类型的新实例。它返回一个 `shared_ptr<thing>`。

由于类型可以由 `make_shared<type>(...)` 的返回类型自动推导，因此上述行可以使用 C++ 的 `auto` 类型说明符更简单地表达：

```
auto double_ptr = make_shared<double>(0.37);
auto vec3_ptr   = make_shared<vec3>(1.414214, 2.718281, 1.618034);
auto sphere_ptr = make_shared<sphere>(point3(0,0,0), 1.0);
```

**列表 23**：使用 `shared_ptr` 和 `auto` 类型的一个示例分配

我们将在代码中使用共享指针，因为它允许多个几何体共享一个公共实例（例如，使用相同颜色材料的多个球体），并且因为它使内存管理自动化，更容易推理。

`std::shared_ptr` 包含在 `<memory>` 头文件中。

你可能不熟悉的第二个 C++ 特性是 `std::vector`。这是一个类似于任意类型的数组样式的通用集合。上面，我们使用指向 `hittable` 的指针集合。`std::vector` 会随着添加更多值而自动增长：`objects.push_back(object)` 将一个值添加到 `std::vector` 成员变量 `objects` 的末尾。

`std::vector` 包含在 `<vector>` 头文件中。

最后，列表 21 中的 `using` 语句告诉编译器我们将从 `std` 库中获取 `shared_ptr` 和 `make_shared`，因此我们每次引用它们时不需要前缀 `std::`。

## 常用常量和实用函数

我们需要一些数学常量，我们将它们方便地定义在自己的头文件中。现在我们只需要无穷大，但我们也会在那里抛出我们自己的 pi 定义，我们稍后会需要。我们还将在这里放置一些常用的常量和未来的实用函数。这个新的头文件 `rtweekend.h` 将是我们的通用主头文件。

```
#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>

// C++ Std Usings

using std::make_shared;
using std::shared_ptr;

// Constants

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// Utility Functions

inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

// Common Headers

#include "color.h"
#include "ray.h"
#include "vec3.h"

#endif
```

**列表 24**：`[rtweekend.h]` rtweekend.h 公共头文件

程序文件将首先包含 `rtweekend.h`，因此所有其他头文件（我们的代码将驻留的地方）可以隐式地假设 `rtweekend.h` 已经被包含。头文件仍然需要显式地包含任何其他必要的头文件。我们将基于这些假设进行一些更新。

```
#include <iostream>
```

**列表 25:** `[color.h]` 假设包含 rtweekend.h 的 color.h

```
#include "ray.h"
```

**列表 26:** `[hittable.h]` 假设包含 rtweekend.h 的 hittable.h

```
#include <memory>#include <vector>

using std::make_shared;
using std::shared_ptr;
```

**列表 27:** `[hittable_list.h]` 假设包含 rtweekend.h 的 hittable_list.h

```
#include "vec3.h"
```

**列表 28:** `[sphere.h]` 假设包含 rtweekend.h 的 sphere.h

```
#include <cmath>
#include <iostream>
```

**列表 29:** `[vec3.h]` 假设包含 rtweekend.h 的 vec3.h，现在的新主程序：

```
#include "rtweekend.h"
#include "color.h"
#include "ray.h"
#include "vec3.h"#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
#include <iostream>
double hit_sphere(const point3& center, double radius, const ray& r) {
    ...
}
color ray_color(const ray& r, const hittable& world) {
    hit_record rec;
    if (world.hit(r, 0, infinity, rec)) {
        return 0.5 * (rec.normal + color(1,1,1));
    }
    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5*(unit_direction.y() + 1.0);
    return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
}

int main() {

    // Image

    auto aspect_ratio = 16.0 / 9.0;
    int image_width = 400;

    // Calculate the image height, and ensure that it's at least 1.
    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    // World

    hittable_list world;

    world.add(make_shared<sphere>(point3(0,0,-1), 0.5));
    world.add(make_shared<sphere>(point3(0,-100.5,-1), 100));
    // Camera

    auto focal_length = 1.0;
    auto viewport_height = 2.0;
    auto viewport_width = viewport_height * (double(image_width)/image_height);
    auto camera_center = point3(0, 0, 0);

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    auto viewport_u = vec3(viewport_width, 0, 0);
    auto viewport_v = vec3(0, -viewport_height, 0);

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    auto pixel_delta_u = viewport_u / image_width;
    auto pixel_delta_v = viewport_v / image_height;

    // Calculate the location of the upper left pixel.
    auto viewport_upper_left = camera_center
                             - vec3(0, 0, focal_length) - viewport_u/2 - viewport_v/2;
    auto pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    // Render

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = 0; j < image_height; j++) {
        std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
        for (int i = 0; i < image_width; i++) {
            auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
            auto ray_direction = pixel_center - camera_center;
            ray r(camera_center, ray_direction);

            color pixel_color = ray_color(r, world);            write_color(std::cout, pixel_color);
        }
    }

    std::clog << "\rDone.                 \n";
}
```

**列表 30:** `[main.cc]` 带有可碰撞体的新主程序

这会产生一个图，实际上只是对球体位置及其表面法线的可视化。这通常是查看任何几何模型缺陷或特定特征的好方法。

[![](https://raytracing.github.io/images/img-1.05-normals-sphere-ground.png)](https://raytracing.github.io/images/img-1.05-normals-sphere-ground.png) 图 5：带有地面的法线着色球体的渲染结果

## 区间类

在我们继续之前，我们将实现一个区间类来管理具有最小值和最大值的实值区间。随着我们的进行，我们将经常使用这个类。

```
#ifndef INTERVAL_H
#define INTERVAL_H

class interval {
  public:
    double min, max;

    interval() : min(+infinity), max(-infinity) {} // Default interval is empty

    interval(double min, double max) : min(min), max(max) {}

    double size() const {
        return max - min;
    }

    bool contains(double x) const {
        return min <= x && x <= max;
    }

    bool surrounds(double x) const {
        return min < x && x < max;
    }

    static const interval empty, universe;
};

const interval interval::empty    = interval(+infinity, -infinity);
const interval interval::universe = interval(-infinity, +infinity);

#endif
```

**列表 31:** `[interval.h]` 介绍新的区间类

```
// Common Headers

#include "color.h"#include "interval.h"#include "ray.h"
#include "vec3.h"
```

**列表 32:** `[rtweekend.h]` 包含新的区间类

```
class hittable {
  public:
    ...    virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0;};
```

**列表 33:** `[hittable.h]` 使用区间进行 hittable::hit()

```
class hittable_list : public hittable {
  public:
    ...    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {        hit_record temp_rec;
        bool hit_anything = false;        auto closest_so_far = ray_t.max;
        for (const auto& object : objects) {            if (object->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }
    ...
};
```

**列表 34:** `[hittable_list.h]` 使用区间进行 hittable_list::hit()

```
class sphere : public hittable {
  public:
    ...    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {        ...

        // Find the nearest root that lies in the acceptable range.
        auto root = (h - sqrtd) / a;        if (!ray_t.surrounds(root)) {            root = (h + sqrtd) / a;            if (!ray_t.surrounds(root))                return false;
        }
        ...
    }
    ...
};
```

**列表 35:** `[sphere.h]` 使用区间进行 sphere

```
color ray_color(const ray& r, const hittable& world) {
    hit_record rec;    if (world.hit(r, interval(0, infinity), rec)) {        return 0.5 * (rec.normal + color(1,1,1));
    }

    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5*(unit_direction.y() + 1.0);
    return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
}
```

**列表 36:** `[main.cc]` 使用区间的新主程序

# 将摄像机代码移动到自己的类中

在继续之前，现在是一个将我们的摄像机和场景渲染代码合并到单个新类中的好时机：`camera` 类。摄像机类将负责两个重要的任务：

1.  构建并将射线派发到世界中。

1.  使用这些射线的成果来构建渲染图像。

在这次重构中，我们将收集 `ray_color()` 函数，以及主程序中的图像、摄像机和渲染部分。新的摄像机类将包含两个公共方法 `initialize()` 和 `render()`，以及两个私有辅助方法 `get_ray()` 和 `ray_color()`。

最终，摄像机将遵循我们所能想到的最简单使用模式：默认构造无参数，然后拥有代码将通过简单赋值修改摄像机的公共变量，最后通过调用 `initialize()` 函数初始化一切。我们选择这种模式而不是拥有者调用带有大量参数的构造函数或定义并调用一大堆设置方法。相反，拥有代码只需要设置它明确关心的部分。最后，我们既可以由拥有代码调用 `initialize()`，也可以让摄像机在 `render()` 函数开始时自动调用此函数。我们将采用第二种方法。

在 `main()` 创建摄像机并设置默认值后，它将调用 `render()` 方法。`render()` 方法将准备摄像机进行渲染，然后执行渲染循环。

这是我们的新 `camera` 类的骨架：

```
#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.h"

class camera {
  public:
    /* Public Camera Parameters Here */

    void render(const hittable& world) {
        ...
    }

  private:
    /* Private Camera Variables Here */

    void initialize() {
        ...
    }

    color ray_color(const ray& r, const hittable& world) const {
        ...
    }
};

#endif
```

**列表 37:** `[camera.h]` 摄像机类骨架首先，让我们从 `main.cc` 中填充 `ray_color()` 函数：

```
class camera {
  ...

  private:
    ...

    color ray_color(const ray& r, const hittable& world) const {        hit_record rec;

        if (world.hit(r, interval(0, infinity), rec)) {
            return 0.5 * (rec.normal + color(1,1,1));
        }

        vec3 unit_direction = unit_vector(r.direction());
        auto a = 0.5*(unit_direction.y() + 1.0);
        return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);    }
};

#endif
```

**列表 38:** `[camera.h]` camera::ray_color 函数现在我们将几乎所有的内容从 `main()` 函数移入我们新的摄像机类。在 `main()` 函数中剩下的唯一事情是世界的构建。以下是带有新迁移代码的摄像机类：

```
class camera {
  public:    double aspect_ratio = 1.0;  // Ratio of image width over height
    int    image_width  = 100;  // Rendered image width in pixel count

    void render(const hittable& world) {
        initialize();

        std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

        for (int j = 0; j < image_height; j++) {
            std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
            for (int i = 0; i < image_width; i++) {
                auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
                auto ray_direction = pixel_center - center;
                ray r(center, ray_direction);

                color pixel_color = ray_color(r, world);
                write_color(std::cout, pixel_color);
            }
        }

        std::clog << "\rDone.                 \n";
    }
  private:    int    image_height;   // Rendered image height
    point3 center;         // Camera center
    point3 pixel00_loc;    // Location of pixel 0, 0
    vec3   pixel_delta_u;  // Offset to pixel to the right
    vec3   pixel_delta_v;  // Offset to pixel below

    void initialize() {
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        center = point3(0, 0, 0);

        // Determine viewport dimensions.
        auto focal_length = 1.0;
        auto viewport_height = 2.0;
        auto viewport_width = viewport_height * (double(image_width)/image_height);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        auto viewport_u = vec3(viewport_width, 0, 0);
        auto viewport_v = vec3(0, -viewport_height, 0);

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left =
            center - vec3(0, 0, focal_length) - viewport_u/2 - viewport_v/2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    }
    color ray_color(const ray& r, const hittable& world) const {
        ...
    }
};

#endif
```

**列表 39:** `[camera.h]` 工作摄像机类以下是大大简化后的主函数：

```
#include "rtweekend.h"

#include "camera.h"#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"

color ray_color(const ray& r, const hittable& world) {
    ...
}
int main() {    hittable_list world;

    world.add(make_shared<sphere>(point3(0,0,-1), 0.5));
    world.add(make_shared<sphere>(point3(0,-100.5,-1), 100));

    camera cam;

    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width  = 400;

    cam.render(world);}
```

**列表 40:** `[main.cc]` 使用新摄像机的新的主函数

运行这个新重构的程序应该会给出与之前相同的渲染图像。

# 抗锯齿

如果你将渲染图像放大，你可能会注意到我们渲染图像中边缘的“阶梯”性质。这种阶梯通常被称为“混叠”，或“锯齿”。当真实相机拍照时，通常沿着边缘没有锯齿，因为边缘像素是前景和背景的一些混合。考虑到我们的渲染图像与我们的渲染图像不同，真实世界的图像是连续的。换句话说，世界（以及它的任何真实图像）实际上具有无限分辨率。我们可以通过为每个像素平均大量样本来达到相同的效果。

通过每个像素中心的单个光线，我们正在执行通常称为 *点采样* 的操作。点采样的问题可以通过渲染一个远离的小棋盘来展示。如果这个棋盘由一个 8×8 的黑白瓷砖网格组成，但只有四个光线击中它，那么这四个光线可能只与白色瓷砖相交，或者只与黑色瓷砖相交，或者某种奇特的组合。在现实世界中，当我们用眼睛看一个远离的棋盘时，我们感知到的是灰色，而不是黑白分明的尖锐点。这是因为我们的眼睛自然地做了我们希望我们的光线追踪器做的事情：整合落在我们的渲染图像特定（离散）区域上的（连续函数）光线。

显然，仅仅通过多次在像素中心重新采样相同的射线并不会带来任何好处——我们每次都会得到相同的结果。相反，我们希望采样围绕像素的光，然后对这些样本进行积分以近似真正的连续结果。那么，我们如何积分围绕像素的光呢？

我们将采用最简单的模型：采样以像素为中心的方形区域，该区域延伸到四个相邻像素的中间。这不是最佳方法，但是最直接的方法。（参见[*一个像素不是一个小方块*](https://www.researchgate.net/publication/244986797)深入了解此主题。）

[![](https://raytracing.github.io/images/fig-1.08-pixel-samples.jpg)](https://raytracing.github.io/images/fig-1.08-pixel-samples.jpg)**图 8:** 像素采样

## 一些随机数实用工具

我们需要一个返回实数随机数的随机数生成器。此函数应返回一个规范化的随机数，按照惯例，它落在范围<nobr aria-hidden="true">0≤n<1</nobr><mn>0</mn><mo>≤</mo><mi>n</mi><mo><</mo><mn>1</mn>内。在 1 之前的重要的“小于”符号，因为我们有时会利用这一点。

一种简单的方法是使用`<cstdlib>`中可找到的`std::rand()`函数，它返回 0 到`RAND_MAX`范围内的随机整数。因此，我们可以通过以下代码片段在`rtweekend.h`中添加来获取所需的实数随机数：

```
#include <cmath>#include <cstdlib>#include <iostream>
#include <limits>
#include <memory>
...

// Utility Functions

inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

inline double random_double() {
    // Returns a random real in [0,1).
    return std::rand() / (RAND_MAX + 1.0);
}

inline double random_double(double min, double max) {
    // Returns a random real in [min,max).
    return min + (max-min)*random_double();
}
```

**列表 41:** `[rtweekend.h]` `random_double()`函数

C++传统上没有标准的随机数生成器，但 C++的新版本通过`<random>`头文件（尽管某些专家认为并不完美）解决了这个问题。如果您想使用它，可以使用以下条件获取随机数：

```
...

#include <random>
...

inline double random_double() {
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}
inline double random_double(double min, double max) {
    // Returns a random real in [min,max).
    return min + (max-min)*random_double();
}

... 
```

**列表 42:** `[rtweekend.h]` `random_double()`，另一种实现

## 使用多个样本生成像素

对于由多个样本组成的单个像素，我们将从围绕像素的区域中选择样本，并将得到的（颜色）光值平均在一起。

首先，我们将更新`write_color()`函数以考虑我们使用的样本数量：我们需要找到所有样本的平均值。为此，我们将添加每次迭代的完整颜色，然后在写入颜色之前进行一次除法（除以样本数量）。为了确保最终结果的颜色分量保持在适当的<nobr aria-hidden="true">[0,1]</nobr><mo stretchy="false">[</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo stretchy="false">]</mo>范围内，我们将添加并使用一个小型辅助函数：`interval::clamp(x)`。

```
class interval {
  public:
    ...

    bool surrounds(double x) const {
        return min < x && x < max;
    }

    double clamp(double x) const {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }    ...
};
```

**列表 43:** `[interval.h]` `interval::clamp()`实用函数

这里是包含区间限制函数的更新后的`write_color()`函数：

```
#include "interval.h"#include "vec3.h"

using color = vec3;

void write_color(std::ostream& out, const color& pixel_color) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // Translate the [0,1] component values to the byte range [0,255].    static const interval intensity(0.000, 0.999);
    int rbyte = int(256 * intensity.clamp(r));
    int gbyte = int(256 * intensity.clamp(g));
    int bbyte = int(256 * intensity.clamp(b));
    // Write out the pixel color components.
    out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}
```

**列表 44:** `[color.h]` 多样本`write_color()`函数

现在我们更新相机类以定义和使用新的 `camera::get_ray(i,j)` 函数，该函数将为每个像素生成不同的样本。这个函数将使用一个新的辅助函数 `sample_square()`，该函数在以原点为中心的单位正方形内生成一个随机样本点。然后我们将从这个理想正方形中的随机样本转换回我们当前正在采样的特定像素。

```
class camera {
  public:
    double aspect_ratio      = 1.0;  // Ratio of image width over height
    int    image_width       = 100;  // Rendered image width in pixel count    int    samples_per_pixel = 10;   // Count of random samples for each pixel
    void render(const hittable& world) {
        initialize();

        std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

        for (int j = 0; j < image_height; j++) {
            std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
            for (int i = 0; i < image_width; i++) {                color pixel_color(0,0,0);
                for (int sample = 0; sample < samples_per_pixel; sample++) {
                    ray r = get_ray(i, j);
                    pixel_color += ray_color(r, world);
                }
                write_color(std::cout, pixel_samples_scale * pixel_color);            }
        }

        std::clog << "\rDone.                 \n";
    }
    ...
  private:
    int    image_height;         // Rendered image height    double pixel_samples_scale;  // Color scale factor for a sum of pixel samples    point3 center;               // Camera center
    point3 pixel00_loc;          // Location of pixel 0, 0
    vec3   pixel_delta_u;        // Offset to pixel to the right
    vec3   pixel_delta_v;        // Offset to pixel below

    void initialize() {
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        pixel_samples_scale = 1.0 / samples_per_pixel;
        center = point3(0, 0, 0);
        ...
    }

    ray get_ray(int i, int j) const {
        // Construct a camera ray originating from the origin and directed at randomly sampled
        // point around the pixel location i, j.

        auto offset = sample_square();
        auto pixel_sample = pixel00_loc
                          + ((i + offset.x()) * pixel_delta_u)
                          + ((j + offset.y()) * pixel_delta_v);

        auto ray_origin = center;
        auto ray_direction = pixel_sample - ray_origin;

        return ray(ray_origin, ray_direction);
    }

    vec3 sample_square() const {
        // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
        return vec3(random_double() - 0.5, random_double() - 0.5, 0);
    }
    color ray_color(const ray& r, const hittable& world) const {
        ...
    }
};

#endif
```

**列表 45**：`[camera.h]` 带有每像素样本参数的相机

（除了上面提到的新的 `sample_square()` 函数外，你还可以在 Github 源代码中找到 `sample_disk()` 函数。这包括如果你想要尝试非正方形像素的情况，但在这本书中我们不会使用它。`sample_disk()` 函数依赖于稍后定义的 `random_in_unit_disk()` 函数。）

主程序更新为设置新的相机参数。

```
int main() {
    ...

    camera cam;

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 400;    cam.samples_per_pixel = 100;
    cam.render(world);
}
```

**列表 46**：`[main.cc]` 设置新的每像素样本参数。放大生成的图像，我们可以看到边缘像素的差异。[![](https://raytracing.github.io/images/img-1.06-antialias-before-after.png)](https://raytracing.github.io/images/img-1.06-antialias-before-after.png)图 6：抗锯齿前后的对比

# 漫反射材料

现在我们有了对象和每个像素的多个射线，我们可以制作一些看起来更逼真的材料。我们将从漫反射材料（也称为*哑光*）开始。一个问题是我们是否混合和匹配几何形状和材料（这样我们就可以将材料分配给多个球体，反之亦然），或者几何形状和材料是否紧密绑定（这对于几何形状和材料链接的程序化对象可能很有用）。我们将选择分开——这在大多数渲染器中是常见的——但请注意，还有其他方法。

## 简单的漫反射材料

漫反射物体不会发出自己的光，只是接受周围环境的颜色，但它们会用自己的固有颜色来调制这种颜色。从漫反射表面反射的光线方向是随机的，因此，如果我们向两个漫反射表面之间的裂缝中发送三束光线，它们各自都会有不同的随机行为：

[![](https://raytracing.github.io/images/fig-1.09-light-bounce.jpg)](https://raytracing.github.io/images/fig-1.09-light-bounce.jpg)**图 9**：光线反弹

它们也可能被吸收而不是反射。表面越暗，光线被吸收的可能性就越大（这就是为什么它是黑色的！）。实际上，任何随机化方向的算法都会产生看起来像哑光的表面。让我们从最直观的开始：一个表面可以随机地向所有方向等概率地反弹光线。对于这种材料，击中表面的光线有等概率地向表面外的任何方向反弹。

[![](https://raytracing.github.io/images/fig-1.10-random-vec-horizon.jpg)](https://raytracing.github.io/images/fig-1.10-random-vec-horizon.jpg)**图 10**：地平线以上的均匀反射

这份非常直观的材料是简单类型的光滑扩散，实际上——许多早期的光线追踪论文都使用了这种方法（在采用更精确的方法之前，我们将在稍后实施）。我们目前还没有随机反射光线的方法，因此我们需要在我们的向量工具头中添加几个函数。首先，我们需要的是生成任意随机向量的能力：

```
class vec3 {
  public:
    ...

    double length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }

    static vec3 random() {
        return vec3(random_double(), random_double(), random_double());
    }

    static vec3 random(double min, double max) {
        return vec3(random_double(min,max), random_double(min,max), random_double(min,max));
    }};
```

**列表 47:** `[vec3.h]` vec3 随机实用函数

然后，我们需要弄清楚如何操作随机向量，以便我们只得到半球表面的结果。有分析方法的实现，但实际上它们理解起来相当复杂，实现起来也相当复杂。相反，我们将使用通常最容易的算法：拒绝法。拒绝法通过反复生成随机样本，直到我们产生一个满足所需标准的样本。换句话说，继续拒绝不良样本，直到找到一个好的样本。

使用拒绝法在半球上生成随机向量有许多同样有效的方法，但出于我们的目的，我们将采用最简单的方法，即：

1.  在单位球体内生成一个随机向量

1.  将此向量归一化以扩展到球面

1.  如果归一化向量落在错误的半球上，则将其反转

首先，我们将使用拒绝法生成单位球体内的随机向量（即半径为 1 的球体）。在包含单位球体的立方体内随机选择一个点（即，其中 <nobr aria-hidden="true">x</nobr><mi>x</mi>, <nobr aria-hidden="true">y</nobr><mi>y</mi>, 和 <nobr aria-hidden="true">z</nobr><mi>z</mi> 都在范围 <nobr aria-hidden="true">[−1,+1]</nobr><mo stretchy="false">[</mo><mo>−</mo><mn>1</mn><mo>,</mo><mo>+</mo><mn>1</mn><mo stretchy="false">]</mo> 内）。如果这个点位于单位球体之外，那么就生成一个新的点，直到我们找到一个位于单位球体内部或其上的点。[![](https://raytracing.github.io/images/fig-1.11-sphere-vec.jpg)](https://raytracing.github.io/images/fig-1.11-sphere-vec.jpg)**图 11:** 在找到合适的向量之前，拒绝了两个向量（归一化前）[![](https://raytracing.github.io/images/fig-1.12-sphere-unit-vec.jpg)](https://raytracing.github.io/images/fig-1.12-sphere-unit-vec.jpg)**图 12:** 被接受的随机向量被归一化以产生一个单位向量

这是我们的函数的第一个草案：

```
...

inline vec3 unit_vector(const vec3& v) {
    return v / v.length();
}

inline vec3 random_unit_vector() {
    while (true) {
        auto p = vec3::random(-1,1);
        auto lensq = p.length_squared();
        if (lensq <= 1)
            return p / sqrt(lensq);
    }
}
```

**列表 48:** `[vec3.h]` random_unit_vector() 函数，版本一

很遗憾，我们有一个小的浮点数抽象泄漏需要处理。由于浮点数具有有限的精度，一个非常小的值在平方时可能会下溢为零。因此，如果所有三个坐标都足够小（即，非常接近球体的中心），则向量的范数将为零，从而归一化将产生错误的向量 <nobr aria-hidden="true">[±∞,±∞,±∞]</nobr><mo stretchy="false">[</mo><mo>±</mo><mi mathvariant="normal">∞</mi><mo>,</mo><mo>±</mo><mi mathvariant="normal">∞</mi><mo>,</mo><mo>±</mo><mi mathvariant="normal">∞</mi><mo stretchy="false">]</mo>。为了解决这个问题，我们还将拒绝位于中心周围这个“黑洞”内的点。使用双精度（64 位浮点数），我们可以安全地支持大于 <nobr aria-hidden="true">10−160</nobr><msup><mn>10</mn><mrow class="MJX-TeXAtom-ORD"><mo>−</mo><mn>160</mn></mrow></msup> 的值。

这里是我们的更健壮的函数：

```
inline vec3 random_unit_vector() {
    while (true) {
        auto p = vec3::random(-1,1);
        auto lensq = p.length_squared();        if (1e-160 < lensq && lensq <= 1)            return p / sqrt(lensq);
    }
}
```

**清单 49:** `[vec3.h]` random_unit_vector() 函数，版本二。现在我们有了随机单位向量，我们可以通过将其与表面法线进行比较来确定它是否位于正确的半球：[![](https://raytracing.github.io/images/fig-1.13-surface-normal.jpg)](https://raytracing.github.io/images/fig-1.13-surface-normal.jpg)**图 13:** 法线向量告诉我们需要哪个半球。我们可以计算表面法线与我们的随机向量的点积，以确定它是否位于正确的半球。如果点积为正，则向量位于正确的半球。如果点积为负，则需要反转向量。

```
...

inline vec3 random_unit_vector() {
    while (true) {
        auto p = vec3::random(-1,1);
        auto lensq = p.length_squared();
        if (1e-160 < lensq && lensq <= 1)
            return p / sqrt(lensq);
    }
}

inline vec3 random_on_hemisphere(const vec3& normal) {
    vec3 on_unit_sphere = random_unit_vector();
    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}
```

**清单 50:** `[vec3.h]` random_on_hemisphere() 函数

如果一条射线从材料上弹回并保留了 100% 的颜色，那么我们说该材料是**白色**的。如果一条射线从材料上弹回并保留了 0% 的颜色，那么我们说该材料是黑色。作为对我们新漫反射材料的第一次演示，我们将 `ray_color` 函数设置为返回弹跳 50% 的颜色。我们应该期望得到一个漂亮的灰色。

```
class camera {
  ...
  private:
    ...
    color ray_color(const ray& r, const hittable& world) const {
        hit_record rec;

        if (world.hit(r, interval(0, infinity), rec)) {            vec3 direction = random_on_hemisphere(rec.normal);
            return 0.5 * ray_color(ray(rec.p, direction), world);        }

        vec3 unit_direction = unit_vector(r.direction());
        auto a = 0.5*(unit_direction.y() + 1.0);
        return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
    }
};
```

**清单 51:** `[camera.h]` 使用随机射线方向进行 ray_color()... 确实，我们得到了相当漂亮的灰色球体：[![](https://raytracing.github.io/images/img-1.07-first-diffuse.png)](https://raytracing.github.io/images/img-1.07-first-diffuse.png)图 7：漫反射球体的首次渲染

## 限制子射线数量

这里有一个潜在的问题潜伏着。请注意，`ray_color` 函数是递归的。它何时会停止递归？当它未能击中任何东西时。然而，在某些情况下，这可能需要很长时间——足够长以至于会溢出堆栈。为了防止这种情况，让我们限制最大递归深度，在最大深度时返回无光贡献：

```
class camera {
  public:
    double aspect_ratio      = 1.0;  // Ratio of image width over height
    int    image_width       = 100;  // Rendered image width in pixel count
    int    samples_per_pixel = 10;   // Count of random samples for each pixel    int    max_depth         = 10;   // Maximum number of ray bounces into scene
    void render(const hittable& world) {
        initialize();

        std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

        for (int j = 0; j < image_height; j++) {
            std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
            for (int i = 0; i < image_width; i++) {
                color pixel_color(0,0,0);
                for (int sample = 0; sample < samples_per_pixel; sample++) {
                    ray r = get_ray(i, j);                    pixel_color += ray_color(r, max_depth, world);                }
                write_color(std::cout, pixel_samples_scale * pixel_color);
            }
        }

        std::clog << "\rDone.                 \n";
    }
    ...
  private:
    ...    color ray_color(const ray& r, int depth, const hittable& world) const {
        // If we've exceeded the ray bounce limit, no more light is gathered.
        if (depth <= 0)
            return color(0,0,0);
        hit_record rec;

        if (world.hit(r, interval(0, infinity), rec)) {
            vec3 direction = random_on_hemisphere(rec.normal);            return 0.5 * ray_color(ray(rec.p, direction), depth-1, world);        }

        vec3 unit_direction = unit_vector(r.direction());
        auto a = 0.5*(unit_direction.y() + 1.0);
        return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
    }
};
```

**清单 52:** `[camera.h]` camera::ray_color() 带深度限制

```
int main() {
    ...

    camera cam;

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 400;
    cam.samples_per_pixel = 100;    cam.max_depth         = 50;
    cam.render(world);
}
```

**列表 53:** `[main.cc]` 使用新的光线深度限制对于这个非常简单的场景，我们应该得到基本上相同的结果：[![](https://raytracing.github.io/images/img-1.08-second-diffuse.png)](https://raytracing.github.io/images/img-1.08-second-diffuse.png) 图 8：有限次反射的漫反射球体的第二次渲染

## 解决阴影痘痘问题

还有一个需要我们解决的微妙错误。当光线与表面相交时，它会尝试准确计算交点。不幸的是，这个计算容易受到浮点舍入误差的影响，这可能导致交点略微偏离。这意味着下一束光线的起源，即从表面随机散射的光线，不太可能完美地与表面齐平。它可能刚好在表面之上。它可能刚好在表面之下。如果光线的起源刚好在表面之下，那么它可能会再次与该表面相交。这意味着它将在 <nobr aria-hidden="true">t=0.00000001</nobr><mi>t</mi><mo>=</mo><mn>0.00000001</mn> 或 whatever floating point approximation the hit function gives us 处找到最近的表面。解决这个问题的最简单的方法就是忽略那些非常接近计算出的交点的碰撞：

```
class camera {
  ...
  private:
    ...
    color ray_color(const ray& r, int depth, const hittable& world) const {
        // If we've exceeded the ray bounce limit, no more light is gathered.
        if (depth <= 0)
            return color(0,0,0);

        hit_record rec;

        if (world.hit(r, interval(0.001, infinity), rec)) {            vec3 direction = random_on_hemisphere(rec.normal);
            return 0.5 * ray_color(ray(rec.p, direction), depth-1, world);
        }

        vec3 unit_direction = unit_vector(r.direction());
        auto a = 0.5*(unit_direction.y() + 1.0);
        return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
    }
};
```

**列表 54:** `[camera.h]` 使用容差计算反射光线起源这解决了阴影痘痘问题。是的，它确实被这样称呼。这是结果：[![](https://raytracing.github.io/images/img-1.09-no-acne.png)](https://raytracing.github.io/images/img-1.09-no-acne.png) 图 9：无阴影痘痘的漫反射球体

## 真实的朗伯反射

在半球面上均匀散射反射光线会产生一个很好的柔和漫反射模型，但我们肯定可以做得更好。对真实漫反射物体的更准确表示是 *朗伯* 分布。这种分布以与 <nobr aria-hidden="true">cos(ϕ)</nobr><mi>cos</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>ϕ</mi><mo stretchy="false">)</mo> 成正比的方式散射反射光线，其中 <nobr aria-hidden="true">ϕ</nobr><mi>ϕ</mi> 是反射光线与表面法线之间的角度。这意味着反射光线最有可能在接近表面法线的方向上散射，而不太可能在远离法线的方向上散射。这种非均匀的朗伯分布比我们之前的均匀散射更好地模拟了现实世界中的材料反射。

我们可以通过向法线向量添加一个随机单位向量来创建这种分布。在表面的交点处，有碰撞点，<nobr aria-hidden="true">p</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">p</mi></mrow>，以及表面的法线，<nobr aria-hidden="true">n</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">n</mi></mrow>。在交点处，这个表面恰好有两个面，因此只能有两个独特的单位球体与任何交点相切（每个面一个独特的球体）。这两个单位球体将沿着它们的半径长度从表面移动，对于一个单位球体，这个长度正好是一。

一个球体将沿着表面的法线方向（<nobr aria-hidden="true">n</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">n</mi></mrow>）移动，另一个球体将沿着相反方向（<nobr aria-hidden="true">−n</nobr><mrow class="MJX-TeXAtom-ORD"><mo mathvariant="bold">−</mo><mi mathvariant="bold">n</mi></mrow>）移动。这导致我们有两个单位大小的球体，它们仅在交点处刚好接触表面。由此，一个球体的中心位于 <nobr aria-hidden="true">(P+n)</nobr><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">P</mi></mrow><mo>+</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">n</mi></mrow><mo stretchy="false">)</mo>，而另一个球体的中心位于 <nobr aria-hidden="true">(P−n)</nobr><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">P</mi></mrow><mo>−</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">n</mi></mrow><mo stretchy="false">)</mo>。中心位于 <nobr aria-hidden="true">(P−n)</nobr><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">P</mi></mrow><mo>−</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">n</mi></mrow><mo stretchy="false">)</mo> 的球体被认为是表面**内部**的，而中心位于 <nobr aria-hidden="true">(P+n)</nobr><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">P</mi></mrow><mo>+</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">n</mi></mrow><mo stretchy="false">)</mo> 的球体被认为是表面**外部**的。

我们想要选择与射线原点位于表面同一侧的切线单位球。在这个单位半径球上随机选择一个点 <nobr aria-hidden="true">S</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">S</mi></mrow>，并从击中点 <nobr aria-hidden="true">P</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">P</mi></mrow> 发射一条射线到随机点 <nobr aria-hidden="true">S</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">S</mi></mrow>（这是向量 <nobr aria-hidden="true">(S−P)</nobr><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">S</mi></mrow><mo>−</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">P</mi></mrow><mo stretchy="false">)</mo>）：

[![](https://raytracing.github.io/images/fig-1.14-rand-unitvec.jpg)](https://raytracing.github.io/images/fig-1.14-rand-unitvec.jpg)**图 14**：根据 Lambertian 分布随机生成一个向量 The 变化实际上相当微小：

```
class camera {
    ...
    color ray_color(const ray& r, int depth, const hittable& world) const {
        // If we've exceeded the ray bounce limit, no more light is gathered.
        if (depth <= 0)
            return color(0,0,0);

        hit_record rec;

        if (world.hit(r, interval(0.001, infinity), rec)) {            vec3 direction = rec.normal + random_unit_vector();            return 0.5 * ray_color(ray(rec.p, direction), depth-1, world);
        }

        vec3 unit_direction = unit_vector(r.direction());
        auto a = 0.5*(unit_direction.y() + 1.0);
        return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
    }
};
```

**列表 55**：`[camera.h]` 中的 `ray_color()` 函数使用替换漫反射在渲染后我们得到一个类似的图像：[![](https://raytracing.github.io/images/img-1.10-correct-lambertian.png)](https://raytracing.github.io/images/img-1.10-correct-lambertian.png) 图像 10：Lambertian 球体的正确渲染

由于我们的场景是两个简单的球体，很难区分这两种漫反射方法，但你应该能够注意到两个重要的视觉差异：

1.  变化后阴影更加明显

1.  两个球体在变化后都从天空中染上了蓝色

这两个变化都是由于光线散射的不均匀性——更多的光线散射向法线。这意味着对于漫反射物体，它们将看起来 *更暗*，因为向相机反弹的光更少。对于阴影，更多的光线直接向上反弹，因此球体下方的区域更暗。

并非很多常见的日常物体都是完美漫反射的，因此我们对这些物体在光照下行为的视觉直觉可能形成得并不好。随着书中场景的逐渐复杂化，你被鼓励在这几种在此处展示的漫反射渲染器之间进行切换。大多数有趣的场景将包含大量的漫反射材料。通过理解不同漫反射方法对场景光照的影响，你可以获得宝贵的见解。

## 使用伽玛校正以获得准确的色彩强度

注意球体下的阴影。图片非常暗，但我们的球体只吸收了每次反弹的一半能量，因此它们是 50% 的反射体。球体应该看起来相当明亮（在现实生活中，是浅灰色），但它们看起来相当暗。如果我们通过我们的漫反射材料的完整亮度范围来观察，我们可以更清楚地看到这一点。我们首先将 `ray_color` 函数的反射率从 `0.5`（50%）设置为 `0.1`（10%）：

```
class camera {
    ...
    color ray_color(const ray& r, int depth, const hittable& world) const {
        // If we've exceeded the ray bounce limit, no more light is gathered.
        if (depth <= 0)
            return color(0,0,0);

        hit_record rec;

        if (world.hit(r, interval(0.001, infinity), rec)) {
            vec3 direction = rec.normal + random_unit_vector();            return 0.1 * ray_color(ray(rec.p, direction), depth-1, world);        }

        vec3 unit_direction = unit_vector(r.direction());
        auto a = 0.5*(unit_direction.y() + 1.0);
        return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
    }
};
```

**列表 56**：`[camera.h]` 中的 `ray_color()` 函数使用 10% 的反射率

我们以新的 10%反射率进行渲染。然后我们将反射率设置为 30%并再次渲染。我们重复这个过程，直到达到 50%、70%，最后是 90%。您可以在您选择的图片编辑器中从左到右叠加这些图像，应该能得到一个很好的所选色域亮度增加的视觉表示。这是我们迄今为止一直在使用的：

[![](https://raytracing.github.io/images/img-1.11-linear-gamut.png)](https://raytracing.github.io/images/img-1.11-linear-gamut.png)图 11：我们渲染器目前的色域

如果您仔细观察，或者使用颜色选择器，您应该会注意到 50%反射率的渲染（中间的那个）过于暗淡，无法在白色和黑色（中间灰）之间达到一半（中间灰）。实际上，70%的反射器更接近中间灰。这是因为几乎所有计算机程序都假设在写入图像文件之前，图像是“伽玛校正”的。这意味着 0 到 1 的值在存储为字节之前应用了一些转换。没有经过转换就写入数据的图像被称为处于“线性空间”，而经过转换的图像被称为处于“伽玛空间”。很可能是您使用的图像查看器期望一个伽玛空间的图像，但我们给它的是一个线性空间的图像。这就是我们的图像看起来不准确的原因。

存储图像在伽玛空间中有许多很好的理由，但就我们的目的而言，我们只需要意识到这一点。我们将我们的数据转换到伽玛空间，以便我们的图像查看器可以更准确地显示我们的图像。作为一个简单的近似，我们可以使用“伽玛 2”作为我们的转换，这是从伽玛空间到线性空间使用的幂。我们需要从线性空间转换到伽玛空间，这意味着取“伽玛 2”的倒数，即<sup class="MJX-TeXAtom-ORD"><span class="MJX-TeXAtom-ORD" style="vertical-align: -0.7em;"><mi class="MJX-tex-mathit" mathvariant="italic">1</mi></span></sup><span class="MJX-TeXAtom-ORD" style="vertical-align: -0.7em;"><mi class="MJX-tex-mathit" mathvariant="italic">g</mi></span><span class="MJX-TeXAtom-ORD" style="vertical-align: -0.7em;"><mi class="MJX-tex-mathit" mathvariant="italic">a</mi></span><span class="MJX-TeXAtom-ORD" style="vertical-align: -0.7em;"><mi class="MJX-tex-mathit" mathvariant="italic">m</mi></span><span class="MJX-TeXAtom-ORD" style="vertical-align: -0.7em;"><mi class="MJX-tex-mathit" mathvariant="italic">m</mi></span><span class="MJX-TeXAtom-ORD" style="vertical-align: -0.7em;"><mi class="MJX-tex-mathit" mathvariant="italic">a</mi></span>，这仅仅是平方根。我们还想确保我们能够稳健地处理负输入。

```
inline double linear_to_gamma(double linear_component) {
    if (linear_component > 0)
        return std::sqrt(linear_component);

    return 0;
}
void write_color(std::ostream& out, const color& pixel_color) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // Apply a linear to gamma transform for gamma 2
    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);
    // Translate the [0,1] component values to the byte range [0,255].
    static const interval intensity(0.000, 0.999);
    int rbyte = int(256 * intensity.clamp(r));
    int gbyte = int(256 * intensity.clamp(g));
    int bbyte = int(256 * intensity.clamp(b));

    // Write out the pixel color components.
    out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}
```

**列表 57**：`[color.h]` write_color()，使用伽玛校正使用这种伽玛校正，我们现在从暗到亮得到了一个更加一致的渐变：[![](https://raytracing.github.io/images/img-1.12-gamma-gamut.png)](https://raytracing.github.io/images/img-1.12-gamma-gamut.png)图 12：我们的渲染器色域，伽玛校正

# 金属

## 材料抽象类

如果我们想让不同的物体有不同的材料，我们有一个设计决策。我们可以有一个具有许多参数的通用材料类型，这样任何单个材料类型都可以忽略不影响它的参数。这不是一个坏方法。或者我们可以有一个封装独特行为的抽象材料类。我是后者的支持者。对于我们的程序，材料需要做两件事：

1.  产生一个散射射线（或者可以说它吸收了入射射线）。

1.  如果分散，请说明射线应该衰减多少。

这表明了抽象类：

```
#ifndef MATERIAL_H
#define MATERIAL_H

#include "hittable.h"

class material {
  public:
    virtual ~material() = default;

    virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered
    ) const {
        return false;
    }
};

#endif
```

**列表 58**：`[material.h]` 材料类

## 描述射线-物体相交的数据结构

`hit_record` 是为了避免大量参数，我们可以将任何我们想要的 信息放入其中。你可以使用参数而不是封装的类型，这只是口味的问题。击中物和材料需要在代码中能够引用对方的类型，因此存在一些引用的循环。在 C++ 中，我们添加 `class material;` 这一行来告诉编译器 `material` 是一个将在以后定义的类。由于我们只是指定了类的指针，编译器不需要知道类的细节，这样就解决了循环引用的问题。

```
class material;
class hit_record {
  public:
    point3 p;
    vec3 normal;    shared_ptr<material> mat;    double t;
    bool front_face;

    void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};
```

**列表 59**：`[hittable.h]` 带有附加材料指针的击中记录

`hit_record` 只是一种将大量参数放入一个类中的方法，这样我们就可以将它们作为一个组发送。当射线击中一个表面（例如一个特定的球体）时，`hit_record` 中的材料指针将被设置为指向在 `main()` 中设置球体时提供的材料指针。当 `ray_color()` 例程获取 `hit_record` 时，它可以调用材料指针的成员函数以找出是否有任何射线被散射。

为了实现这一点，`hit_record` 需要知道分配给球体的材料。

```
class sphere : public hittable {
  public:    sphere(const point3& center, double radius) : center(center), radius(std::fmax(0,radius)) {
        // TODO: Initialize the material pointer `mat`.
    }
    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        ...

        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);        rec.mat = mat;
        return true;
    }

  private:
    point3 center;
    double radius;    shared_ptr<material> mat;};
```

**列表 60**：`[sphere.h]` 带有附加材料信息的射线-球体相交

## 模拟光散射和反射

在这里以及在这些书籍的整个过程中，我们将使用术语 *albedo*（拉丁语意为“白色”）。Albedo 在某些学科中是一个精确的技术术语，但在所有情况下，它都用于定义某种形式的 *分数反射率*。Albedo 会随材料颜色变化，并且（正如我们将在玻璃材料中后来实现的那样）也可以随入射观察方向（入射射线的方向）变化。

拉姆伯特（漫反射）反射率可以始终根据其反射率 <nobr aria-hidden="true">R</nobr><mi>R</mi> 散射并衰减光线，或者它有时可以（概率 <nobr aria-hidden="true">1−R</nobr><mn>1</mn><mo>−</mo><mi>R</mi>）不衰减地散射（其中未散射的射线只是被材料吸收）。它也可以是这两种策略的混合。我们将选择始终散射，因此实现拉姆伯特材料变得简单：

```
class material {
    ...
};

class lambertian : public material {
  public:
    lambertian(const color& albedo) : albedo(albedo) {}

    bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
    const override {
        auto scatter_direction = rec.normal + random_unit_vector();
        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }

  private:
    color albedo;
};
```

**列表 61**：`[material.h]` 新的拉姆伯特材料类

注意第三个选项：我们可以以某个固定的概率 <nobr aria-hidden="true">p</nobr><mi>p</mi> 进行散射，并且衰减为 <nobr aria-hidden="true">albedo/p</nobr><mrow class="MJX-TeXAtom-ORD"><mi class="MJX-tex-mathit" mathvariant="italic">a</mi><mi class="MJX-tex-mathit" mathvariant="italic">l</mi><mi class="MJX-tex-mathit" mathvariant="italic">b</mi><mi class="MJX-tex-mathit" mathvariant="italic">e</mi><mi class="MJX-tex-mathit" mathvariant="italic">d</mi><mi class="MJX-tex-mathit" mathvariant="italic">o</mi></mrow><mrow class="MJX-TeXAtom-ORD"><mo>/</mo></mrow><mi class="MJX-tex-mathit" mathvariant="italic">p</mi>. 您的选择。

如果你仔细阅读上面的代码，你会注意到有一点点麻烦。如果我们生成的随机单位向量正好与法向量相反，这两个向量将相加为零，这将导致散射方向向量为零。这会导致后续出现不良情况（无穷大和 NaN），因此我们需要在传递之前拦截这种条件。

为了实现这一点，我们将创建一个新的向量方法——`vec3::near_zero()`——如果向量在所有维度上都非常接近零，则返回 true。

以下更改将使用 C++ 标准库函数 `std::fabs`，该函数返回其输入的绝对值。

```
class vec3 {
    ...

    double length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }

    bool near_zero() const {
        // Return true if the vector is close to zero in all dimensions.
        auto s = 1e-8;
        return (std::fabs(e[0]) < s) && (std::fabs(e[1]) < s) && (std::fabs(e[2]) < s);
    }
    ...
};
```

**列表 62**：`[vec3.h]` vec3::near_zero() 方法

```
class lambertian : public material {
  public:
    lambertian(const color& albedo) : albedo(albedo) {}

    bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
    const override {
        auto scatter_direction = rec.normal + random_unit_vector();

        // Catch degenerate scatter direction
        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;
        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }

  private:
    color albedo;
};
```

**列表 63**：`[material.h]` 拉姆伯特散射，防弹

## 镜面光反射

对于抛光金属，光线不会随机散射。关键问题是：光线是如何从金属镜子上反射的？向量数学是我们的朋友：

![图 15：光线反射](https://raytracing.github.io/images/fig-1.15-reflection.jpg)**图 15**：光线反射

反射射线的方向用红色表示，即 <nobr aria-hidden="true">v+2b</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">v</mi></mrow><mo>+</mo><mn>2</mn><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">b</mi></mrow>。在我们的设计中，<nobr aria-hidden="true">n</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">n</mi></mrow> 是一个单位向量（长度为 1），但 <nobr aria-hidden="true">v</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">v</mi></mrow> 可能不是。为了得到向量 <nobr aria-hidden="true">b</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">b</mi></mrow>，我们将法向量乘以 <nobr aria-hidden="true">v</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">v</mi></mrow> 在 <nobr aria-hidden="true">n</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">n</mi></mrow> 上的投影长度，该长度由点积 <nobr aria-hidden="true">v⋅n</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">v</mi></mrow><mo>⋅</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">n</mi></mrow> 给出。（如果 <nobr aria-hidden="true">n</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">n</mi></mrow> 不是一个单位向量，我们还需要将这个点积除以 <nobr aria-hidden="true">n</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">n</mi></mrow> 的长度。）最后，因为 <nobr aria-hidden="true">v</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">v</mi></mrow> 指向表面内部，而我们希望 <nobr aria-hidden="true">b</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">b</mi></mrow> 指向表面外部，所以我们需要取这个投影长度的负值。

将所有内容组合起来，我们得到以下反射向量的计算：

```
...

inline vec3 random_on_hemisphere(const vec3& normal) {
    ...
}

inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2*dot(v,n)*n;
}
```

**列表 64:** `[vec3.h]` vec3 反射函数金属材质仅使用该公式反射射线：

```
...

class lambertian : public material {
    ...
};

class metal : public material {
  public:
    metal(const color& albedo) : albedo(albedo) {}

    bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
    const override {
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        scattered = ray(rec.p, reflected);
        attenuation = albedo;
        return true;
    }

  private:
    color albedo;
};
```

**列表 65:** `[material.h]` 具有反射函数的金属材质我们需要修改 `ray_color()` 函数以适应所有更改：

```
#include "hittable.h"#include "material.h"...

class camera {
  ...
  private:
    ...
    color ray_color(const ray& r, int depth, const hittable& world) const {
        // If we've exceeded the ray bounce limit, no more light is gathered.
        if (depth <= 0)
            return color(0,0,0);

        hit_record rec;

        if (world.hit(r, interval(0.001, infinity), rec)) {            ray scattered;
            color attenuation;
            if (rec.mat->scatter(r, rec, attenuation, scattered))
                return attenuation * ray_color(scattered, depth-1, world);
            return color(0,0,0);        }

        vec3 unit_direction = unit_vector(r.direction());
        auto a = 0.5*(unit_direction.y() + 1.0);
        return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
    }
};
```

**列表 66:** `[camera.h]` 具有散射反射的射线颜色

现在我们将更新 `sphere` 构造函数以初始化材质指针 `mat`：

```
class sphere : public hittable {
  public:    sphere(const point3& center, double radius, shared_ptr<material> mat)
      : center(center), radius(std::fmax(0,radius)), mat(mat) {}
    ...
};
```

**列表 67:** `[sphere.h]` 使用材质初始化球体

## 金属球体场景

现在让我们在我们的场景中添加一些金属球体：

```
#include "rtweekend.h"

#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"#include "material.h"#include "sphere.h"

int main() {
    hittable_list world;

    auto material_ground = make_shared<lambertian>(color(0.8, 0.8, 0.0));
    auto material_center = make_shared<lambertian>(color(0.1, 0.2, 0.5));
    auto material_left   = make_shared<metal>(color(0.8, 0.8, 0.8));
    auto material_right  = make_shared<metal>(color(0.8, 0.6, 0.2));

    world.add(make_shared<sphere>(point3( 0.0, -100.5, -1.0), 100.0, material_ground));
    world.add(make_shared<sphere>(point3( 0.0,    0.0, -1.2),   0.5, material_center));
    world.add(make_shared<sphere>(point3(-1.0,    0.0, -1.0),   0.5, material_left));
    world.add(make_shared<sphere>(point3( 1.0,    0.0, -1.0),   0.5, material_right));
    camera cam;

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth         = 50;

    cam.render(world);
}
```

**列表 68:** `[main.cc]` 金属球体场景，它给出：[![](https://raytracing.github.io/images/img-1.13-metal-shiny.png)](https://raytracing.github.io/images/img-1.13-metal-shiny.png) 图 13：闪亮的金属

## 模糊反射

我们还可以通过使用一个小球体并选择射线的新的端点来随机化反射方向。我们将使用一个随机点，该点来自以原始端点为中心的球体表面，并按模糊因子进行缩放。

[![](https://raytracing.github.io/images/fig-1.16-reflect-fuzzy.jpg)](https://raytracing.github.io/images/fig-1.16-reflect-fuzzy.jpg)**图 16：生成模糊反射射线

模糊球体越大，反射就越模糊。这表明我们可以添加一个模糊性参数，它正好是球体的半径（因此零表示没有扰动）。但是，对于大球体或掠射光线，我们可能在表面下方发生散射。我们可以让表面吸收这些光线。

还要注意，为了让模糊球体有意义，它需要与反射向量保持一致的缩放比例，反射向量的长度可以是任意变化的。为了解决这个问题，我们需要对反射光线进行归一化。

```
class metal : public material {
  public:    metal(const color& albedo, double fuzz) : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {}
    bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
    const override {
        vec3 reflected = reflect(r_in.direction(), rec.normal);        reflected = unit_vector(reflected) + (fuzz * random_unit_vector());        scattered = ray(rec.p, reflected);
        attenuation = albedo;        return (dot(scattered.direction(), rec.normal) > 0);    }

  private:
    color albedo;    double fuzz;};
```

**列表 69**：`[material.h]` 金属材料的模糊性我们可以通过给金属添加模糊性 0.3 和 1.0 来尝试一下：

```
int main() {
    ...
    auto material_ground = make_shared<lambertian>(color(0.8, 0.8, 0.0));
    auto material_center = make_shared<lambertian>(color(0.1, 0.2, 0.5));    auto material_left   = make_shared<metal>(color(0.8, 0.8, 0.8), 0.3);
    auto material_right  = make_shared<metal>(color(0.8, 0.6, 0.2), 1.0);    ...
}
```

**列表 70**：`[main.cc]` 具有模糊性的金属球[![](https://raytracing.github.io/images/img-1.14-metal-fuzz.png)](https://raytracing.github.io/images/img-1.14-metal-fuzz.png)图 14：模糊金属

# 电介质

清晰的材料，如水、玻璃和钻石，是电介质。当光线击中它们时，它会分成一个反射光线和一个折射（透射）光线。我们将通过随机选择反射和折射来处理这个问题，每次交互只生成一个散射光线。

作为术语的快速回顾，一个**反射**光线击中表面然后“弹跳”到新的方向。

当折射光线从一种材料的周围过渡到该材料本身时（例如玻璃或水），它会弯曲。这就是为什么铅笔部分插入水中时看起来会弯曲。

折射光线弯曲的程度由材料的**折射率**决定。通常，这是一个单一值，描述了光线从真空进入材料时弯曲的程度。玻璃的折射率大约是 1.5–1.7，钻石大约是 2.4，而空气有一个很小的折射率 1.000293。

当一个透明材料嵌入到另一种透明材料中时，你可以用相对折射率来描述折射：物体的材料折射率除以周围材料的折射率。例如，如果你想渲染一个浸在水中玻璃球，那么玻璃球的有效折射率将是 1.125。这是由玻璃的折射率（1.5）除以水的折射率（1.333）得出的。

你可以通过快速网络搜索找到大多数常见材料的折射率。

## 折射

调试中最困难的部分是折射光线。我通常首先让所有光线在存在折射光线的情况下都发生折射。对于这个项目，我尝试在我们的场景中放入两个玻璃球，结果就是这样（我还没有告诉你这样做是对是错，但很快就会告诉你！）：

![![](https://raytracing.github.io/images/img-1.15-glass-first.png)](https://raytracing.github.io/images/img-1.15-glass-first.png)图 15：玻璃首次出现

这对吗？现实生活中玻璃球看起来很奇怪。但不是的，这是不对的。世界应该是颠倒的，没有奇怪的黑东西。我只是将光线直接打印在图像的中间，这显然是错误的。这通常能解决问题。

## 斯涅尔定律

折射由斯涅尔定律描述：

<nobr aria-hidden="true">η⋅sinθ=η′⋅sinθ′</nobr><mi>η</mi><mo>⋅</mo><mi>sin</mi><mo>⁡</mo><mi>θ</mi><mo>=</mo><msup><mi>η</mi><mo>′</mo></msup><mo>⋅</mo><mi>sin</mi><mo>⁡</mo><msup><mi>θ</mi><mo>′</mo></msup>

其中 <nobr aria-hidden="true">θ</nobr><mi>θ</mi> 和 <nobr aria-hidden="true">θ′</nobr><msup><mi>θ</mi><mo>′</mo></msup> 是从法线到角度，而 <nobr aria-hidden="true">η</nobr><mi>η</mi> 和 <nobr aria-hidden="true">η′</nobr><msup><mi>η</mi><mo>′</mo></msup>（发音为“eta”和“eta prime”）是折射率。几何关系如下：

![![](https://raytracing.github.io/images/fig-1.17-refraction.jpg)](https://raytracing.github.io/images/fig-1.17-refraction.jpg)**图 17：** 光线折射

为了确定折射光线的方向，我们需要求解 <nobr aria-hidden="true">sinθ′</nobr><mi>sin</mi><mo>⁡</mo><msup><mi>θ</mi><mo>′</mo></msup>:

<nobr aria-hidden="true">sinθ′=ηη′⋅sinθ</nobr><mi>sin</mi><mo>⁡</mo><msup><mi>θ</mi><mo>′</mo></msup><mo>=</mo><mfrac><mi>η</mi><msup><mi>η</mi><mo>′</mo></msup></mfrac><mo>⋅</mo><mi>sin</mi><mo>⁡</mo><mi>θ</mi>

在表面的折射侧有一个折射光线 <nobr aria-hidden="true">R′</nobr><mrow class="MJX-TeXAtom-ORD"><msup><mi mathvariant="bold">R</mi><mo>′</mo></msup></mrow> 和一个法线 <nobr aria-hidden="true">n′</nobr><mrow class="MJX-TeXAtom-ORD"><msup><mi mathvariant="bold">n</mi><mo>′</mo></msup></mrow>，它们之间存在一个角度，<nobr aria-hidden="true">θ′</nobr><msup><mi>θ</mi><mo>′</mo></msup>。我们可以将 <nobr aria-hidden="true">R′</nobr><mrow class="MJX-TeXAtom-ORD"><msup><mi mathvariant="bold">R</mi><mo>′</mo></msup></mrow> 分解为与 <nobr aria-hidden="true">n′</nobr><mrow class="MJX-TeXAtom-ORD"><msup><mi mathvariant="bold">n</mi><mo>′</mo></msup></mrow> 垂直和平行的部分：

<nobr aria-hidden="true">R′=R′⊥+R′∥</nobr><mrow class="MJX-TeXAtom-ORD"><msup><mi mathvariant="bold">R</mi><mo>′</mo></msup></mrow><mo>=</mo><msub><mrow class="MJX-TeXAtom-ORD"><msup><mi mathvariant="bold">R</mi><mo>′</mo></msup></mrow><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="normal">⊥</mi></mrow></msub><mo>+</mo><msub><mrow class="MJX-TeXAtom-ORD"><msup><mi mathvariant="bold">R</mi><mo>′</mo></msup></mrow><mrow class="MJX-TeXAtom-ORD"><mo>∥</mo></mrow></msub>

如果我们求解 <nobr aria-hidden="true">R′⊥</nobr><msub><mrow class="MJX-TeXAtom-ORD"><msup><mi mathvariant="bold">R</mi><mo>′</mo></msup></mrow><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="normal">⊥</mi></mrow></msub> 和 <nobr aria-hidden="true">R′∥</nobr><msub><mrow class="MJX-TeXAtom-ORD"><msup><mi mathvariant="bold">R</mi><mo>′</mo></msup></mrow><mrow class="MJX-TeXAtom-ORD"><mo>∥</mo></mrow></msub>，我们得到：

<nobr aria-hidden="true">R′⊥=ηη′(R+|R|cos(θ)n)</nobr><msub><mrow class="MJX-TeXAtom-ORD"><msup><mi mathvariant="bold">R</mi><mo>′</mo></msup></mrow><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="normal">⊥</mi></mrow></msub><mo>=</mo><mfrac><mi>η</mi><msup><mi>η</mi><mo>′</mo></msup></mfrac><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">R</mi></mrow><mo>+</mo><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">|</mo></mrow><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">R</mi></mrow><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">|</mo></mrow><mi>cos</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">n</mi></mrow><mo stretchy="false">)</mo><nobr aria-hidden="true">R′∥=−1−|R′⊥|2−−−−−−−−√n</nobr><msub><mrow class="MJX-TeXAtom-ORD"><msup><mi mathvariant="bold">R</mi><mo>′</mo></msup></mrow><mrow class="MJX-TeXAtom-ORD"><mo>∥</mo></mrow></msub><mo>=</mo><mo>−</mo><msqrt><mn>1</mn><mo>−</mo><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">|</mo></mrow><msub><mrow class="MJX-TeXAtom-ORD"><msup><mi mathvariant="bold">R</mi><mo>′</mo></msup></mrow><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="normal">⊥</mi></mrow></msub><msup><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">|</mo></mrow><mn>2</mn></msup></msqrt><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">n</mi></mrow>

如果你想亲自证明这一点，可以继续，但我们将把它作为事实并继续前进。本书的其余部分不需要你理解证明过程。

我们知道右侧每个项的值，除了 <nobr aria-hidden="true">cosθ</nobr><mi>cos</mi><mo>⁡</mo><mi>θ</mi>。众所周知，两个向量的点积可以用它们之间角度的余弦来解释：

<nobr aria-hidden="true">a⋅b=|a||b|cosθ</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">a</mi></mrow><mo>⋅</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">b</mi></mrow><mo>=</mo><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">|</mo></mrow><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">a</mi></mrow><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">|</mo></mrow><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">|</mo></mrow><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">b</mi></mrow><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">|</mo></mrow><mi>cos</mi><mo>⁡</mo><mi>θ</mi>

如果我们将 <nobr aria-hidden="true">a</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">a</mi></mrow> 和 <nobr aria-hidden="true">b</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">b</mi></mrow> 限制为单位向量：

<nobr aria-hidden="true">a⋅b=cosθ</nobr><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">a</mi></mrow><mo>⋅</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">b</mi></mrow><mo>=</mo><mi>cos</mi><mo>⁡</mo><mi>θ</mi>

我们现在可以用已知量来重新表示 <nobr aria-hidden="true">R′⊥</nobr><msub><mrow class="MJX-TeXAtom-ORD"><msup><mi mathvariant="bold">R</mi><mo>′</mo></msup></mrow><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="normal">⊥</mi></mrow></msub>：

<nobr aria-hidden="true">R′⊥=ηη′(R+(−R⋅n)n)</nobr><msub><mrow class="MJX-TeXAtom-ORD"><msup><mi mathvariant="bold">R</mi><mo>′</mo></msup></mrow><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="normal">⊥</mi></mrow></msub><mo>=</mo><mfrac><mi>η</mi><msup><mi>η</mi><mo>′</mo></msup></mfrac><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">R</mi></mrow><mo>+</mo><mo stretchy="false">(</mo><mrow class="MJX-TeXAtom-ORD"><mo mathvariant="bold">−</mo><mi mathvariant="bold">R</mi></mrow><mo>⋅</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">n</mi></mrow><mo stretchy="false">)</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">n</mi></mrow><mo stretchy="false">)</mo>当我们把它们组合在一起时，我们可以编写一个函数来计算 <nobr aria-hidden="true">R′</nobr><mrow class="MJX-TeXAtom-ORD"><msup><mi mathvariant="bold">R</mi><mo>′</mo></msup></mrow>：

```
...

inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2*dot(v,n)*n;
}

inline vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat) {
    auto cos_theta = std::fmin(dot(-uv, n), 1.0);
    vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    vec3 r_out_parallel = -std::sqrt(std::fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}
```

**列表 71**：`[vec3.h]` 折射函数始终折射的介电材料是：

```
...

class metal : public material {
    ...
};

class dielectric : public material {
  public:
    dielectric(double refraction_index) : refraction_index(refraction_index) {}

    bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
    const override {
        attenuation = color(1.0, 1.0, 1.0);
        double ri = rec.front_face ? (1.0/refraction_index) : refraction_index;

        vec3 unit_direction = unit_vector(r_in.direction());
        vec3 refracted = refract(unit_direction, rec.normal, ri);

        scattered = ray(rec.p, refracted);
        return true;
    }

  private:
    // Refractive index in vacuum or air, or the ratio of the material's refractive index over
    // the refractive index of the enclosing media
    double refraction_index;
};
```

**列表 72**：`[material.h]` 始终折射的介电材料类现在我们将场景更新为通过将左侧球体改为玻璃来展示折射，玻璃的折射率约为 1.5。

```
auto material_ground = make_shared<lambertian>(color(0.8, 0.8, 0.0));
auto material_center = make_shared<lambertian>(color(0.1, 0.2, 0.5));auto material_left   = make_shared<dielectric>(1.50);auto material_right  = make_shared<metal>(color(0.8, 0.6, 0.2), 1.0);
```

**列表 73**：`[main.cc]` 将左侧球体改为玻璃这给我们以下结果：[![](https://raytracing.github.io/images/img-1.16-glass-always-refract.png)](https://raytracing.github.io/images/img-1.16-glass-always-refract.png) 图 16：始终折射的玻璃球体

## 总内反射

折射中存在的一个棘手的问题是，对于某些射线角度，使用斯涅尔定律无法找到解决方案。当射线以足够大的斜角进入折射率较低的介质时，它可以以大于 90°的角度折射。如果我们回顾斯涅尔定律和 <nobr aria-hidden="true">sinθ′</nobr><mi>sin</mi><mo>⁡</mo><msup><mi>θ</mi><mo>′</mo></msup> 的推导：

<nobr aria-hidden="true">sinθ′=ηη′⋅sinθ</nobr><mi>sin</mi><mo>⁡</mo><msup><mi>θ</mi><mo>′</mo></msup><mo>=</mo><mfrac><mi>η</mi><msup><mi>η</mi><mo>′</mo></msup></mfrac><mo>⋅</mo><mi>sin</mi><mo>⁡</mo><mi>θ</mi>

如果光线在玻璃内部，外部是空气（<nobr aria-hidden="true">η=1.5</nobr><mi>η</mi><mo>=</mo><mn>1.5</mn> 和 <nobr aria-hidden="true">η′=1.0</nobr><msup><mi>η</mi><mo>′</mo></msup><mo>=</mo><mn>1.0</mn>）：

<nobr aria-hidden="true">sinθ′=1.5/1.0⋅sinθ</nobr><mi>sin</mi><mo>⁡</mo><msup><mi>θ</mi><mo>′</mo></msup><mo>=</mo><mfrac><mn>1.5</mn><mn>1.0</mn></mfrac><mo>⋅</mo><mi>sin</mi><mo>⁡</mo><mi>θ</mi>The value of <nobr aria-hidden="true">sinθ′</nobr><mi>sin</mi><mo>⁡</mo><msup><mi>θ</mi><mo>′</mo></msup> cannot be greater than 1\. So, if,<nobr aria-hidden="true">1.5/1.0⋅sinθ>1.0</nobr><mfrac><mn>1.5</mn><mn>1.0</mn></mfrac><mo>⋅</mo><mi>sin</mi><mo>⁡</mo><mi>θ</mi><mo>></mo><mn>1.0</mn>

方程式两边的等式被打破，不存在解。如果不存在解，玻璃不能折射，因此必须反射光线：

```
if (ri * sin_theta > 1.0) {
    // Must Reflect
    ...
} else {
    // Can Refract
    ...
}
```

**列表 74:** `[material.h]` 判断光线是否可以折射

在这里所有光线都被反射，因为实际上这通常是在固体物体内部，所以它被称为 *全内反射*。这就是为什么有时当你浸没在水中时，水-空气边界会像一个完美的镜子——如果你在水中向上看，你可以看到水面以上的东西，但是当你靠近水面并朝侧面看时，水面看起来像一面镜子。

我们可以使用三角恒等式求解 `sin_theta`：

<nobr aria-hidden="true">sinθ=√(1−cos²θ)</nobr><mi>sin</mi><mo>⁡</mo><mi>θ</mi><mo>=</mo><msqrt><mn>1</mn><mo>−</mo><msup><mi>cos</mi><mn>2</mn></msup><mo>⁡</mo><mi>θ</mi></msqrt>

和

<nobr aria-hidden="true">cosθ=R⋅n</nobr><mi>cos</mi><mo>⁡</mo><mi>θ</mi><mo>=</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">R</mi></mrow><mo>⋅</mo><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">n</mi></mrow>

```
double cos_theta = std::fmin(dot(-unit_direction, rec.normal), 1.0);
double sin_theta = std::sqrt(1.0 - cos_theta*cos_theta);

if (ri * sin_theta > 1.0) {
    // Must Reflect
    ...
} else {
    // Can Refract
    ...
}
```

**列表 75:** `[material.h]` 判断光线是否可以折射，以及总是折射（如果可能）的介质是：

```
class dielectric : public material {
  public:
    dielectric(double refraction_index) : refraction_index(refraction_index) {}

    bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
    const override {
        attenuation = color(1.0, 1.0, 1.0);
        double ri = rec.front_face ? (1.0/refraction_index) : refraction_index;

        vec3 unit_direction = unit_vector(r_in.direction());        double cos_theta = std::fmin(dot(-unit_direction, rec.normal), 1.0);
        double sin_theta = std::sqrt(1.0 - cos_theta*cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0;
        vec3 direction;

        if (cannot_refract)
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, ri);

        scattered = ray(rec.p, direction);        return true;
    }

  private:
    // Refractive index in vacuum or air, or the ratio of the material's refractive index over
    // the refractive index of the enclosing media
    double refraction_index;
};
```

**列表 76:** `[material.h]` 具有反射的介质材料类

衰减总是 1——玻璃表面不吸收任何东西。

如果我们使用新的 `dielectric::scatter()` 函数渲染先前的场景，我们看到……没有变化。咦？

嗯，结果是，给定一个折射率大于空气的材料的球体，没有入射角会产生全内反射——无论是光线进入球体的点还是光线离开的点。这是由于球体的几何形状，因为掠入射的光线总是会弯曲到一个更小的角度，然后在离开时弯曲回原来的角度。

那我们如何说明全内反射呢？嗯，如果球体的折射率小于它所在的介质，那么我们可以用浅掠入射角去撞击它，得到全外反射。这应该足以观察到效果。

我们将模拟一个充满水的世界（折射率约为 1.33），并将球体材料改为空气（折射率 1.00）——一个气泡！为此，将左球体材料的折射率改为

<nobr aria-hidden="true">空气折射率/水折射率</nobr><mfrac><mtext>空气折射率</mtext><mtext>水折射率</mtext></mfrac>

```
auto material_ground = make_shared<lambertian>(color(0.8, 0.8, 0.0));
auto material_center = make_shared<lambertian>(color(0.1, 0.2, 0.5));auto material_left   = make_shared<dielectric>(1.00 / 1.33);auto material_right  = make_shared<metal>(color(0.8, 0.6, 0.2), 1.0);
```

**列表 77:** `[main.cc]` 左球是水中的气泡 This change yields the following render:[![](https://raytracing.github.io/images/img-1.17-air-bubble-total-reflection.png)](https://raytracing.github.io/images/img-1.17-air-bubble-total-reflection.png)图 17：气泡有时折射，有时反射

在这里，你可以看到大致直接的光线会折射，而掠射光线会反射。

## Schlick 近似

现实中的玻璃具有随角度变化的反射率——以陡峭的角度看窗户，它变成了镜子。为此有一个很大的丑陋方程，但几乎每个人都使用 Christophe Schlick 提出的一个便宜且出奇准确的多项式近似。这产生了我们的全玻璃材料：

```
class dielectric : public material {
  public:
    dielectric(double refraction_index) : refraction_index(refraction_index) {}

    bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
    const override {
        attenuation = color(1.0, 1.0, 1.0);
        double ri = rec.front_face ? (1.0/refraction_index) : refraction_index;

        vec3 unit_direction = unit_vector(r_in.direction());
        double cos_theta = std::fmin(dot(-unit_direction, rec.normal), 1.0);
        double sin_theta = std::sqrt(1.0 - cos_theta*cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, ri) > random_double())            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, ri);

        scattered = ray(rec.p, direction);
        return true;
    }

  private:
    // Refractive index in vacuum or air, or the ratio of the material's refractive index over
    // the refractive index of the enclosing media
    double refraction_index;

    static double reflectance(double cosine, double refraction_index) {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - refraction_index) / (1 + refraction_index);
        r0 = r0*r0;
        return r0 + (1-r0)*std::pow((1 - cosine),5);
    }};
```

**列表 78:** `[material.h]` 全玻璃材料

## 空心玻璃球的建模

让我们模拟一个空心玻璃球。这是一个有一定厚度的球体，里面有一个空气球。如果你考虑光线穿过这样一个物体的路径，它将击中外球，折射，击中内球（假设我们确实击中了它），第二次折射，然后穿过球内的空气。然后它将继续前进，击中内球的内表面，折射回来，然后击中外球的内表面，最后折射并返回场景大气中。

外球只是用标准玻璃球建模，折射率约为 1.50（模拟从外部空气进入玻璃的折射）。内球则略有不同，因为*它的*折射率应该相对于周围的外球材料，从而模拟从玻璃进入内空气的过渡。

这实际上很简单就可以指定，因为电介质材料的 `refraction_index` 参数可以解释为物体折射率与包围介质折射率的*比值*。在这种情况下，内球将具有空气折射率（内球材料）与玻璃折射率（包围介质）的比值，或者 <nobr aria-hidden="true">1.00/1.50=0.67</nobr><mn>1.00</mn><mrow class="MJX-TeXAtom-ORD"><mo>/</mo></mrow><mn>1.50</mn><mo>=</mo><mn>0.67</mn>.

这里是代码：

```
...
auto material_ground = make_shared<lambertian>(color(0.8, 0.8, 0.0));
auto material_center = make_shared<lambertian>(color(0.1, 0.2, 0.5));auto material_left   = make_shared<dielectric>(1.50);
auto material_bubble = make_shared<dielectric>(1.00 / 1.50);auto material_right  = make_shared<metal>(color(0.8, 0.6, 0.2), 0.0);

world.add(make_shared<sphere>(point3( 0.0, -100.5, -1.0), 100.0, material_ground));
world.add(make_shared<sphere>(point3( 0.0,    0.0, -1.2),   0.5, material_center));
world.add(make_shared<sphere>(point3(-1.0,    0.0, -1.0),   0.5, material_left));world.add(make_shared<sphere>(point3(-1.0,    0.0, -1.0),   0.4, material_bubble));world.add(make_shared<sphere>(point3( 1.0,    0.0, -1.0),   0.5, material_right));
...
```

**列表 79:** `[main.cc]` 带空心玻璃球的场景 And here's the result:[![](https://raytracing.github.io/images/img-1.18-glass-hollow.png)](https://raytracing.github.io/images/img-1.18-glass-hollow.png)图 18：空心玻璃球

# 可定位相机

摄像机，就像电介质一样，调试起来很麻烦，所以我总是逐步开发我的摄像机。首先，让我们允许调整视场角（*fov*）。这是渲染图像边缘到边缘的视角。由于我们的图像不是正方形，水平方向和垂直方向的 fov 是不同的。我总是使用垂直 fov。我也通常用度数指定它，并在构造函数内部将其转换为弧度——这是一个个人喜好问题。

## 摄像机观看几何

首先，我们将保持从原点发出的光线朝向 <nobr aria-hidden="true">z=−1</nobr><mi>z</mi><mo>=</mo><mo>−</mo><mn>1</mn> 平面。我们可以将其设置为 <nobr aria-hidden="true">z=−2</nobr><mi>z</mi><mo>=</mo><mo>−</mo><mn>2</mn> 平面，或者任何其他平面，只要我们使 <nobr aria-hidden="true">h</nobr><mi>h</mi> 与该距离成比例。以下是我们的设置：

[![](https://raytracing.github.io/images/fig-1.18-cam-view-geom.jpg)](https://raytracing.github.io/images/fig-1.18-cam-view-geom.jpg)**图 18：** 摄像机观看几何（侧面）这表示 <nobr aria-hidden="true">h=tan(θ2)</nobr><mi>h</mi><mo>=</mo><mi>tan</mi><mo>⁡</mo><mo stretchy="false">(</mo><mfrac><mi>θ</mi><mn>2</mn></mfrac><mo stretchy="false">)</mo>. 我们现在的摄像机变为：

```
class camera {
  public:
    double aspect_ratio      = 1.0;  // Ratio of image width over height
    int    image_width       = 100;  // Rendered image width in pixel count
    int    samples_per_pixel = 10;   // Count of random samples for each pixel
    int    max_depth         = 10;   // Maximum number of ray bounces into scene

    double vfov = 90;  // Vertical view angle (field of view)
    void render(const hittable& world) {
    ...

  private:
    ...

    void initialize() {
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        pixel_samples_scale = 1.0 / samples_per_pixel;

        center = point3(0, 0, 0);

        // Determine viewport dimensions.
        auto focal_length = 1.0;        auto theta = degrees_to_radians(vfov);
        auto h = std::tan(theta/2);
        auto viewport_height = 2 * h * focal_length;        auto viewport_width = viewport_height * (double(image_width)/image_height);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        auto viewport_u = vec3(viewport_width, 0, 0);
        auto viewport_v = vec3(0, -viewport_height, 0);

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left =
            center - vec3(0, 0, focal_length) - viewport_u/2 - viewport_v/2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    }

    ...
};
```

**列表 80：** `[camera.h]` 可调节视场角（fov）的摄像机我们将通过一个简单的两个接触球体的场景来测试这些更改，使用 90°的视场角。

```
int main() {
    hittable_list world;

    auto R = std::cos(pi/4);

    auto material_left  = make_shared<lambertian>(color(0,0,1));
    auto material_right = make_shared<lambertian>(color(1,0,0));

    world.add(make_shared<sphere>(point3(-R, 0, -1), R, material_left));
    world.add(make_shared<sphere>(point3( R, 0, -1), R, material_right));
    camera cam;

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth         = 50;

    cam.vfov = 90;
    cam.render(world);
}
```

**列表 81：** `[main.cc]` 宽角摄像机场景这给我们带来了渲染效果：[![](https://raytracing.github.io/images/img-1.19-wide-view.png)](https://raytracing.github.io/images/img-1.19-wide-view.png)图 19：宽角视图

## 定位和定向摄像机

要获得任意视角，让我们首先命名我们关心的点。我们将放置摄像机的位置称为 *lookfrom*，我们看的点称为 *lookat*。（稍后，如果你愿意，你可以定义一个看的方向而不是看的点。）

我们还需要一种方法来指定摄像机的翻滚或侧倾：即绕着 lookat-lookfrom 轴的旋转。另一种思考方式是，即使你保持`lookfrom`和`lookat`不变，你仍然可以围绕你的鼻子旋转你的头。我们需要的是为摄像机指定一个“向上”向量的方法。

[![](https://raytracing.github.io/images/fig-1.19-cam-view-dir.jpg)](https://raytracing.github.io/images/fig-1.19-cam-view-dir.jpg)**图 19：** 摄像机视图方向

我们可以指定任何我们想要的向上向量，只要它不与视角方向平行。将这个向上向量投影到与视角方向正交的平面上，以获得相机相关的向上向量。我使用常见的命名惯例，将其命名为“视图向上” (*vup*) 向量。经过几次叉乘和向量归一化后，我们现在有一个完整的正交归一基 <nobr aria-hidden="true">(u,v,w)</nobr><mo stretchy="false">(</mo><mi>u</mi><mo>,</mo><mi>v</mi><mo>,</mo><mi>w</mi><mo stretchy="false">)</mo> 来描述我们的相机方向。 <nobr aria-hidden="true">u</nobr><mi>u</mi> 将是指向相机右侧的单位向量，<nobr aria-hidden="true">v</nobr><mi>v</mi> 是指向相机向上的单位向量，<nobr aria-hidden="true">w</nobr><mi>w</mi> 是指向与视角方向相反的单位向量（因为我们使用右手坐标系），并且相机中心在原点。

[![](https://raytracing.github.io/images/fig-1.20-cam-view-up.jpg)](https://raytracing.github.io/images/fig-1.20-cam-view-up.jpg)**图 20**：相机向上视图方向

和之前一样，当我们的固定相机面向 <nobr aria-hidden="true">−Z</nobr><mo>−</mo><mi>Z</mi> 时，我们的任意视角相机面向 <nobr aria-hidden="true">−w</nobr><mo>−</mo><mi>w</mi>。记住，我们可以——但不必——使用世界向上 <nobr aria-hidden="true">(0,1,0)</nobr><mo stretchy="false">(</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>,</mo><mn>0</mn><mo stretchy="false">)</mo> 来指定 vup。这是方便的，并且会自然地保持你的相机水平直到你决定尝试疯狂的相机角度。

```
class camera {
  public:
    double aspect_ratio      = 1.0;  // Ratio of image width over height
    int    image_width       = 100;  // Rendered image width in pixel count
    int    samples_per_pixel = 10;   // Count of random samples for each pixel
    int    max_depth         = 10;   // Maximum number of ray bounces into scene

    double vfov     = 90;              // Vertical view angle (field of view)    point3 lookfrom = point3(0,0,0);   // Point camera is looking from
    point3 lookat   = point3(0,0,-1);  // Point camera is looking at
    vec3   vup      = vec3(0,1,0);     // Camera-relative "up" direction
    ...

  private:
    int    image_height;         // Rendered image height
    double pixel_samples_scale;  // Color scale factor for a sum of pixel samples
    point3 center;               // Camera center
    point3 pixel00_loc;          // Location of pixel 0, 0
    vec3   pixel_delta_u;        // Offset to pixel to the right
    vec3   pixel_delta_v;        // Offset to pixel below    vec3   u, v, w;              // Camera frame basis vectors
    void initialize() {
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        pixel_samples_scale = 1.0 / samples_per_pixel;

        center = lookfrom;
        // Determine viewport dimensions.        auto focal_length = (lookfrom - lookat).length();        auto theta = degrees_to_radians(vfov);
        auto h = std::tan(theta/2);
        auto viewport_height = 2 * h * focal_length;
        auto viewport_width = viewport_height * (double(image_width)/image_height);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        // Calculate the vectors across the horizontal and down the vertical viewport edges.        vec3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
        vec3 viewport_v = viewport_height * -v;  // Vector down viewport vertical edge
        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.        auto viewport_upper_left = center - (focal_length * w) - viewport_u/2 - viewport_v/2;        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    }

    ...

  private:
};
```

**列表 82**：`[camera.h]` 可定位和可定向的相机我们将回到先前的场景，并使用新的视角：

```
int main() {
    hittable_list world;

    auto material_ground = make_shared<lambertian>(color(0.8, 0.8, 0.0));
    auto material_center = make_shared<lambertian>(color(0.1, 0.2, 0.5));
    auto material_left   = make_shared<dielectric>(1.50);
    auto material_bubble = make_shared<dielectric>(1.00 / 1.50);
    auto material_right  = make_shared<metal>(color(0.8, 0.6, 0.2), 1.0);

    world.add(make_shared<sphere>(point3( 0.0, -100.5, -1.0), 100.0, material_ground));
    world.add(make_shared<sphere>(point3( 0.0,    0.0, -1.2),   0.5, material_center));
    world.add(make_shared<sphere>(point3(-1.0,    0.0, -1.0),   0.5, material_left));
    world.add(make_shared<sphere>(point3(-1.0,    0.0, -1.0),   0.4, material_bubble));
    world.add(make_shared<sphere>(point3( 1.0,    0.0, -1.0),   0.5, material_right));
    camera cam;

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth         = 50;

    cam.vfov     = 90;    cam.lookfrom = point3(-2,2,1);
    cam.lookat   = point3(0,0,-1);
    cam.vup      = vec3(0,1,0);
    cam.render(world);
}
```

**列表 83**：`[main.cc]` 使用不同视角的场景：[![](https://raytracing.github.io/images/img-1.20-view-distant.png)](https://raytracing.github.io/images/img-1.20-view-distant.png) 图像 20：远景视图，并且我们可以更改视野：

```
 cam.vfov     = 20;
```

**列表 84**：`[main.cc]` 将视野更改为：[![](https://raytracing.github.io/images/img-1.21-view-zoom.png)](https://raytracing.github.io/images/img-1.21-view-zoom.png) 图像 21：放大视图

# 散焦模糊

现在我们最后的功能：*散焦模糊*。注意，摄影师称之为*景深*，所以请确保在你的光线追踪朋友中只使用术语*散焦模糊*。

我们在真实相机中看到散焦模糊的原因是因为它们需要一个大的孔（而不是仅仅是一个小孔）来收集光线。一个大孔会让所有东西都失焦，但如果我们在胶片/传感器前面放一个镜头，那么就会有一个特定的距离，在这个距离上所有东西都是清晰的。放置在这个距离上的物体将看起来是清晰的，而距离这个距离越远，它们就会线性地变得越来越模糊。你可以这样想：所有从焦点距离的特定点发出的光线——并且击中镜头——都会弯曲回图像传感器上的一个单一点。

我们称相机中心和所有东西都处于完美焦点平面的距离为**焦距**。请注意，焦距通常不等于焦距——焦距是相机中心和图像平面之间的距离。然而，在我们的模型中，这两个值将相同，因为我们将在焦点平面上放置我们的像素网格，它距离相机中心**焦距**。 

在物理相机中，焦距是由镜头和胶片/传感器之间的距离控制的。这就是为什么当你改变焦点时，你会看到镜头相对于相机移动（这可能在你的手机相机中也会发生，但传感器会移动）。"光圈"是一个孔，用于控制镜头的有效大小。对于真正的相机，如果你需要更多的光线，你使光圈变大，这将导致远离焦距的物体产生更多的模糊。对于我们的虚拟相机，我们可以有一个完美的传感器，永远不需要更多的光线，所以我们只在想要失焦模糊时使用光圈。

## 薄镜头近似

真实相机有一个复杂的复合镜头。对于我们的代码，我们可以模拟这个顺序：传感器，然后是镜头，然后是光圈。然后我们可以确定发送光线的位置，并在计算后翻转图像（图像在胶片上倒置）。然而，图形人员通常使用薄镜头近似：

[![](https://raytracing.github.io/images/fig-1.21-cam-lens.jpg)](https://raytracing.github.io/images/fig-1.21-cam-lens.jpg)**图 21：**相机镜头模型

我们不需要模拟相机内部的任何部分——为了渲染相机外的图像，这将是多余的复杂性。相反，我通常从无限薄的圆形“镜头”处开始发射光线，并将它们发送到焦平面上的目标像素（距离镜头`focal_length`），在这个平面上，3D 世界中的所有东西都处于完美的焦点。

在实践中，我们通过将视口放置在这个平面上来实现这一点。将所有这些放在一起：

1.  焦平面垂直于相机视图方向。

1.  焦距是相机中心和焦平面之间的距离。

1.  视口位于焦平面上，以相机视图方向向量为中心。

1.  像素位置的网格位于视口（位于 3D 世界中）内部。

1.  随机图像样本位置是从当前像素位置周围的区域选择的。

1.  相机从镜头上的随机点发射光线，通过当前图像样本位置。

[![](https://raytracing.github.io/images/fig-1.22-cam-film-plane.jpg)](https://raytracing.github.io/images/fig-1.22-cam-film-plane.jpg)**图 22：**相机焦平面

## 生成样本光线

没有散焦模糊时，所有场景光线都源自相机中心（或`lookfrom`）。为了实现散焦模糊，我们在相机中心构造一个圆盘。半径越大，散焦模糊越明显。你可以把我们的原始相机想象成具有半径为零的散焦圆盘（完全没有模糊），因此所有光线都源自圆盘中心（`lookfrom`）。

那么，散焦圆盘应该有多大？由于这个圆盘的大小控制着我们能得到多少散焦模糊，因此这应该是相机类的一个参数。我们只需将圆盘的半径作为相机参数即可，但模糊会根据投影距离而变化。一个稍微容易一些的参数是指定以视口中心为顶点、以相机中心为底部的圆锥的角度。这应该会给你提供更一致的结果，因为你在给定镜头中改变焦距。

由于我们将从散焦圆盘中选择随机点，我们需要一个函数来完成这个任务：`random_in_unit_disk()`。这个函数使用与我们在`random_unit_vector()`中使用的方法相同，只是针对二维。

```
...

inline vec3 unit_vector(const vec3& u) {
    return v / v.length();
}

inline vec3 random_in_unit_disk() {
    while (true) {
        auto p = vec3(random_double(-1,1), random_double(-1,1), 0);
        if (p.length_squared() < 1)
            return p;
    }
}
...
```

**列表 85**：`[vec3.h]` 在单位圆盘中生成随机点

现在让我们更新相机，使其从散焦圆盘发出光线：

```
class camera {
  public:
    double aspect_ratio      = 1.0;  // Ratio of image width over height
    int    image_width       = 100;  // Rendered image width in pixel count
    int    samples_per_pixel = 10;   // Count of random samples for each pixel
    int    max_depth         = 10;   // Maximum number of ray bounces into scene

    double vfov     = 90;              // Vertical view angle (field of view)
    point3 lookfrom = point3(0,0,0);   // Point camera is looking from
    point3 lookat   = point3(0,0,-1);  // Point camera is looking at
    vec3   vup      = vec3(0,1,0);     // Camera-relative "up" direction

    double defocus_angle = 0;  // Variation angle of rays through each pixel
    double focus_dist = 10;    // Distance from camera lookfrom point to plane of perfect focus
    ...

  private:
    int    image_height;         // Rendered image height
    double pixel_samples_scale;  // Color scale factor for a sum of pixel samples
    point3 center;               // Camera center
    point3 pixel00_loc;          // Location of pixel 0, 0
    vec3   pixel_delta_u;        // Offset to pixel to the right
    vec3   pixel_delta_v;        // Offset to pixel below
    vec3   u, v, w;              // Camera frame basis vectors    vec3   defocus_disk_u;       // Defocus disk horizontal radius
    vec3   defocus_disk_v;       // Defocus disk vertical radius
    void initialize() {
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        pixel_samples_scale = 1.0 / samples_per_pixel;

        center = lookfrom;

        // Determine viewport dimensions.        auto focal_length = (lookfrom - lookat).length();        auto theta = degrees_to_radians(vfov);
        auto h = std::tan(theta/2);        auto viewport_height = 2 * h * focus_dist;        auto viewport_width = viewport_height * (double(image_width)/image_height);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        vec3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
        vec3 viewport_v = viewport_height * -v;  // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors to the next pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.        auto viewport_upper_left = center - (focus_dist * w) - viewport_u/2 - viewport_v/2;        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        // Calculate the camera defocus disk basis vectors.
        auto defocus_radius = focus_dist * std::tan(degrees_to_radians(defocus_angle / 2));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;    }

    ray get_ray(int i, int j) const {        // Construct a camera ray originating from the defocus disk and directed at a randomly
        // sampled point around the pixel location i, j.
        auto offset = sample_square();
        auto pixel_sample = pixel00_loc
                          + ((i + offset.x()) * pixel_delta_u)
                          + ((j + offset.y()) * pixel_delta_v);

        auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample();        auto ray_direction = pixel_sample - ray_origin;

        return ray(ray_origin, ray_direction);
    }

    vec3 sample_square() const {
        ...
    }

    point3 defocus_disk_sample() const {
        // Returns a random point in the camera defocus disk.
        auto p = random_in_unit_disk();
        return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
    }
    color ray_color(const ray& r, int depth, const hittable& world) const {
        ...
    }
};
```

**列表 86**：`[camera.h]` 可调节景深的相机使用大光圈：

```
int main() {
    ...

    camera cam;

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth         = 50;

    cam.vfov     = 20;
    cam.lookfrom = point3(-2,2,1);
    cam.lookat   = point3(0,0,-1);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 10.0;
    cam.focus_dist    = 3.4;
    cam.render(world);
}
```

**列表 87**：`[main.cc]` 场景相机具有景深我们得到：[![](https://raytracing.github.io/images/img-1.22-depth-of-field.png)](https://raytracing.github.io/images/img-1.22-depth-of-field.png) 图像 22：具有景深的球体

# 接下来是什么？

## 最终渲染

让我们制作这本书的封面图像——许多随机的球体。

```
int main() {
    hittable_list world;

    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));
    camera cam;

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 1200;
    cam.samples_per_pixel = 500;
    cam.max_depth         = 50;

    cam.vfov     = 20;
    cam.lookfrom = point3(13,2,3);
    cam.lookat   = point3(0,0,0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0.6;
    cam.focus_dist    = 10.0;
    cam.render(world);
}
```

**列表 88**：`[main.cc]` 最终场景

（注意，上面的代码与项目示例代码略有不同：为了获得高质量图像，上面的`samples_per_pixel`设置为 500，这将需要相当长的时间来渲染。项目源代码在开发和验证过程中使用 10 的值，以实现合理的运行时间。）

这给出了：[![](https://raytracing.github.io/images/img-1.23-book1-final.jpg)](https://raytracing.github.io/images/img-1.23-book1-final.jpg) 图像 23：最终场景

你可能会注意到的一个有趣的现象是，玻璃球实际上并没有影子，这使得它们看起来像是悬浮在空中。这并不是一个错误——在现实生活中，你很少看到玻璃球，它们看起来也有些奇怪，而且在多云的日子里确实似乎在空中漂浮。在玻璃球下的大球体上的一个点仍然有大量的光线照射到它，因为天空是重新排列而不是被阻挡。

## 下一步

你现在有一个很酷的射线追踪器！接下来是什么？

### 书 2：《光线追踪：下周见》

本系列的第二本书基于你在这里开发的射线追踪器。这包括新的功能，如：

+   动态模糊——真实地渲染移动的物体。

+   界定体积层次结构——加快复杂场景的渲染速度。

+   纹理贴图——在物体上放置图像。

+   Perlin 噪声 — 一个非常有用的随机噪声生成器，适用于许多技术。

+   四边形 — 除了球体之外可以渲染的东西！也是实现圆盘、三角形、环形或其他任何二维原型的基石。

+   光线 — 向场景中添加光源。

+   变换 — 用于放置和旋转对象。

+   体积渲染 — 渲染烟雾、云彩和其他气态体积。

### 书籍 3：*光线追踪：你余生的其余部分*

这本书在第二本书的基础上进一步扩展了内容。这本书的很大一部分是关于提高渲染图像质量和渲染器性能，并专注于生成*正确*的光线并适当地累积它们。

这本书是为那些真正对编写专业级光线追踪器感兴趣的人，以及那些对实现高级效果（如次表面散射或嵌套电介质）的基础感兴趣的人。

### 其他方向

从这里你可以探索很多额外的方向，包括我们在这个系列中尚未（？）涵盖的技术。这些包括：

**三角形** — 大多数酷炫的模型都是以三角形的形式存在。模型 I/O 是最糟糕的，几乎每个人都试图使用别人的代码来完成这项工作。这也包括高效地处理大量三角形的*网格*，它们本身也带来了挑战。

**并行处理** — 在具有不同随机种子的 N 个核心上运行你的代码的 N 个副本。平均这 N 次运行。这种平均也可以分层进行，其中 N/2 对可以平均得到 N/4 个图像，然后这些图像的对也可以平均。这种方法应该可以很好地扩展到数千个核心，而无需编写太多代码。

**阴影光线** — 当向光源发射光线时，你可以确切地确定特定点的阴影情况。有了这个，你可以渲染清晰或柔和的阴影，为场景增加另一个真实感层次。

玩得开心，并请将你的酷炫图片发给我！

# 致谢

**原始手稿帮助**

+   Dave Hart

+   Jean Buckley

**网络发布**

+   [Berna Kabadayı](https://github.com/bernakabadayi)

+   [Lorenzo Mancini](https://github.com/lmancini)

+   [Lori Whippler Hollasch](https://github.com/lorihollasch)

+   [Ronald Wotzlaw](https://github.com/ronaldfw)

**校对和改进**

+   [Aaryaman Vasishta](https://github.com/jammm)

+   Andrew Kensler

+   [Antonio Gamiz](https://github.com/antoniogamiz)

+   Apoorva Joshi

+   [Aras Pranckevičius](https://github.com/aras-p)

+   [Arman Uguray](https://github.com/armansito)

+   Becker

+   Ben Kerl

+   Benjamin Summerton

+   Bennett Hardwick

+   [Benny Tsang](https://bthtsang.github.io/)

+   Dan Drummond

+   [David Chambers](https://github.com/dafhi)

+   David Hart

+   [Dimitry Ishenko](https://github.com/dimitry-ishenko)

+   [Dmitry Lomov](https://github.com/mu-lambda)

+   [Eric Haines](https://github.com/erich666)

+   Fabio Sancinetti

+   Filipe Scur

+   Frank He

+   [Gareth Martin](https://github.com/TheThief)

+   [Gerrit Wessendorf](https://github.com/celeph)

+   Grue Debry

+   [Gustaf Waldemarson](https://github.com/xaldew)

+   Ingo Wald

+   Jason Stone

+   [JC-ProgJava](https://github.com/JC-ProgJava)

+   Jean Buckley

+   [Jeff Smith](https://github.com/whydoubt)

+   Joey Cho

+   [John Kilpatrick](https://github.com/rjkilpatrick)

+   [Kaan Eraslan](https://github.com/D-K-E)

+   [Lorenzo Mancini](https://github.com/lmancini)

+   [Manas Kale](https://github.com/manas96)

+   Marcus Ottosson

+   [Mark Craig](https://github.com/mrmcsoftware)

+   Markus Boos

+   Matthew Heimlich

+   Nakata Daisuke

+   [Nate Rupsis](https://github.com/rupsis)

+   [Niccolò Tiezzi](https://github.com/niccolot)

+   Paul Melis

+   Phil Cristensen

+   [LollipopFt](https://github.com/LollipopFt)

+   [Ronald Wotzlaw](https://github.com/ronaldfw)

+   [Shaun P. Lee](https://github.com/shaunplee)

+   [Shota Kawajiri](https://github.com/estshorter)

+   Tatsuya Ogawa

+   Thiago Ize

+   [Thien Tran](https://github.com/gau-nernst)

+   Vahan Sosoyan

+   [WANG Lei](https://github.com/wlbksy)

+   [Yann Herklotz](https://github.com/ymherklotz)

+   [ZeHao Chen](https://github.com/oxine)

**特别感谢**感谢 [Limnu](https://limnu.com/) 团队对图表的帮助。

这些书籍完全使用 Morgan McGuire 的神奇且免费的 [Markdeep](https://casual-effects.com/markdeep/) 库编写。要查看其外观，请从您的浏览器中查看页面源代码。

感谢 [Helen Hu](https://github.com/hhu) 慷慨地将她的 [`github.com/RayTracing/`](https://github.com/RayTracing/) GitHub 组织捐赠给本项目。

<link rel="stylesheet" href="https://raytracing.github.io/books/../style/book.css">

# 引用此书

一致的引用使识别本作品的来源、位置和版本变得更容易。如果您正在引用此书，我们建议您尽可能使用以下格式之一。

## 基本数据

+   **标题 (系列)**："一周之内学习光线追踪系列"

+   **标题 (书籍)**："一周之内学习光线追踪"

+   **作者**：Peter Shirley, Trevor David Black, Steve Hollasch

+   **版本/版次**：v4.0.2

+   **日期**：2025-04-25

+   **URL (系列)**：[`raytracing.github.io`](https://raytracing.github.io)

+   **URL (书籍)**：[`raytracing.github.io/books/raytracinginoneweekend.html`](https://raytracing.github.io/books/raytracinginoneweekend.html)

## 碎片

### Markdown

```
[_Ray Tracing in One Weekend_](https://raytracing.github.io/books/RayTracingInOneWeekend.html)
```

### HTML

```
<a href="https://raytracing.github.io/books/RayTracingInOneWeekend.html">
    <cite>Ray Tracing in One Weekend</cite>
</a>
```

### LaTeX 和 BibTex

```
~\cite{Shirley2025RTW1}

@misc{Shirley2025RTW1,
   title = {Ray Tracing in One Weekend},
   author = {Peter Shirley, Trevor David Black, Steve Hollasch},
   year = {2025},
   month = {April},
   note = {\small \texttt{https://raytracing.github.io/books/RayTracingInOneWeekend.html}},
   url = {https://raytracing.github.io/books/RayTracingInOneWeekend.html}
}
```

### BibLaTeX

```
\usepackage{biblatex}

~\cite{Shirley2025RTW1}

@online{Shirley2025RTW1,
   title = {Ray Tracing in One Weekend},
   author = {Peter Shirley, Trevor David Black, Steve Hollasch},
   year = {2025},
   month = {April},
   url = {https://raytracing.github.io/books/RayTracingInOneWeekend.html}
}
```

### IEEE

```
“Ray Tracing in One Weekend.” raytracing.github.io/books/RayTracingInOneWeekend.html
(accessed MMM. DD, YYYY)
```

### MLA:

```
Ray Tracing in One Weekend. raytracing.github.io/books/RayTracingInOneWeekend.html
Accessed DD MMM. YYYY.
```

<link rel="stylesheet" href="../style/book.css">

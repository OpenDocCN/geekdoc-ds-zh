## 附录 A. CUDA 手册库

如 第一章 中所提到，本书随附的源代码是开源的，采用两段式 BSD 许可证。源代码的指针可在 [www.cudahandbook.com](http://www.cudahandbook.com) 找到，开发人员还可以在 [`github.com/ArchaeaSoftware/cudahandbook`](https://github.com/ArchaeaSoftware/cudahandbook) 上找到 Git 仓库。

本附录简要描述了 CUDA 手册库（chLib）的功能，它是一组便携式头文件，位于源代码项目的 `chLib/` 子目录中。chLib 并不打算在生产软件中重复使用，它提供了最小的功能，且源代码量最小，用于说明本书中涉及的概念。chLib 可移植到所有 CUDA 支持的操作系统，因此它通常需要暴露对这些操作系统功能交集的支持。

### A.1\. 定时

CUDA 手册库包括一个便携式定时库，它在 Windows 上使用 `QueryPerformanceCounter()`，在非 Windows 平台上使用 `gettimeofday()`。一个示例用法如下。

点击这里查看代码图像

float

TimeNULLKernelLaunches(int cIterations = 1000000 )

{

chTimerTimestamp start, stop;

chTimerGetTime( &start );

for ( int i = 0; i < cIterations; i++ ) {

NullKernel<<<1,1>>>();

}

cudaThreadSynchronize();

chTimerGetTime( &stop );

return 1e6*chTimerElapsedTime( &start, &stop ) /

(float) cIterations;

}

该函数计时指定数量的内核启动，并返回每次启动的微秒数。`chTimerTimestamp` 是一个高精度的时间戳，通常它是一个 64 位的计数器，随着时间单调递增，因此需要两个时间戳来计算时间间隔。

`chTimerGetTime()`函数获取当前时间的快照。`chTimerElapsedTime()`函数返回两个时间戳之间经过的秒数。这些计时器的分辨率非常精细（可能是微秒级别），因此`chTimerElapsedTime()`返回`double`类型。

点击这里查看代码图像

#ifdef _WIN32

#include <windows.h>

typedef LARGE_INTEGER chTimerTimestamp;

#else

typedef struct timeval chTimerTimestamp;

#endif

void chTimerGetTime(chTimerTimestamp *p);

double chTimerElapsedTime( chTimerTimestamp *pStart, chTimerTimestamp

*pEnd );

double chTimerBandwidth( chTimerTimestamp *pStart, chTimerTimestamp

*pEnd, double cBytes );

在 CUDA 支持的 GPU 上进行性能测量时，我们可能会使用 CUDA 事件，例如测量内核的设备内存带宽。使用 CUDA 事件进行计时是一把双刃剑：它们不太受系统级事件（如网络流量）的干扰，但这有时会导致过于乐观的计时结果。

### A.2\. 线程

chLib 包含一个简洁的线程库，支持创建一个“工作”CPU 线程池，并提供功能使父线程能够将工作“委派”给工作线程。线程是一个特别难以抽象的特性，因为不同的操作系统有着截然不同的支持方式。一些操作系统甚至有“线程池”，使得线程可以轻松回收，这样应用程序就不必让线程处于暂停状态，等待某个同步事件的触发。

Listing A.1 提供了 `chLib/chThread.h` 中的抽象线程支持。它包括一个 `processorCount()` 函数，用于返回可用的 CPU 核心数（许多使用多线程来利用多个 CPU 核心的应用程序，如我们在 第十四章 中的多线程 N-body 实现，想要为每个核心启动一个线程），以及一个 C++ 类 `workerThread`，它支持一些简单的线程操作。

创建与销毁

• `delegateSynchronous()`：父线程指定一个函数指针供 worker 执行，且该函数在 worker 线程完成之前不会返回。

• `delegateAsynchronous()`：父线程指定一个函数指针供 worker 异步运行；必须调用 `workerThread::waitAll` 来同步父线程与子线程。

• 成员函数 `waitAll()` 会等待直到所有指定的 worker 线程完成其委托的工作。

*Listing A.1.* `workerThread` 类。

点击这里查看代码图像

* * *

//

// 返回平台上的执行核心数量。

//

unsigned int processorCount();

//

// workerThread 类 - 包括一个线程 ID（在构造函数中指定）

//

class workerThread

{

public:

workerThread( int cpuThreadId = 0 );

virtual ~workerThread();

bool initialize( );

// 线程例程（平台特定）

static void threadRoutine( LPVOID );

//

// 从您的应用线程调用此方法来委托给 worker。

// 它不会返回，直到您的函数指针已被

// 使用给定参数调用。

//

bool delegateSynchronous( void (*pfn)(void *), void *parameter );

//

// 从应用线程调用此方法来委托给 worker

// 异步调用。由于它立即返回，您必须调用

// 稍后调用 waitAll

bool delegateAsynchronous( void (*pfn)(void *), void *parameter );

static bool waitAll( workerThread *p, size_t N );

};

* * *

### A.3. 驱动程序 API 功能

`chDrv.h`包含了一些对驱动程序 API 开发人员有用的工具：`chCUDADevice`类，如 Listing A.2 所示，简化了设备和上下文的管理。它的`loadModuleFromFile`方法简化了从`.cubin`或`.ptx`文件创建模块的过程。

此外，`chGetErrorString()`函数返回一个只读字符串，该字符串对应于一个错误值。除了为驱动程序 API 的`CUresult`类型实现`chDrv.h`中声明的此函数外，`chGetErrorString()`的特化版本还封装了 CUDA 运行时的`cudaGetErrorString()`函数。

*Listing A.2.* chCUDADevice 类。

点击这里查看代码图像

* * *

class chCUDADevice

{

public:

chCUDADevice();

virtual ~chCUDADevice();

CUresult 初始化(

int ordinal,

list<string>& moduleList,

unsigned int Flags = 0,

unsigned int numOptions = 0,

CUjit_option *options = NULL,

void **optionValues = NULL );

CUresult 从文件加载模块(

CUmodule *pModule,

string fileName,

unsigned int numOptions = 0,

CUjit_option *options = NULL,

void **optionValues = NULL );

CUdevice device() const { return m_device; }

CUcontext context() const { return m_context; }

CUmodule module( string s ) const { return (*m_modules.find(s)).

second; }

private:

CUdevice m_device;

CUcontext m_context;

map<string, CUmodule> m_modules;

};

* * *

### A.4\. Shmoos

“Shmoo 图”指的是当两个输入（例如电压和时钟频率）变化时，测试电路模式的图形显示。在编写代码以识别各种内核的最佳阻塞参数时，通过改变输入值（如线程块大小和循环展开因子）进行类似的测试是很有用的。Listing A.3 显示了`chShmooRange`类，该类封装了一个参数范围，以及`chShmooIterator`类，它使得`for`循环能够轻松地遍历给定范围。

*Listing A.3.* chShmooRange 和 chShmooIterator 类。

点击这里查看代码图像

* * *

class chShmooRange {

public:

chShmooRange( ) { }

void Initialize( int value );

bool Initialize( int min, int max, int step );

bool isStatic() const { return m_min==m_max; }

friend class chShmooIterator;

int min() const { return m_min; }

int max() const { return m_max; }

private:

bool m_initialized;

int m_min, m_max, m_step;

};

class chShmooIterator

{

public:

chShmooIterator( const chShmooRange& range );

int operator *() const { return m_i; }

operator bool() const { return m_i <= m_max; }

void operator++(int) { m_i += m_step; };

private:

int m_i;

int m_max;

int m_step;

};

* * *

命令行解析器还包括一个特殊化版本，可以基于命令行参数创建`chShmooRange`：将“min”、“max”和“step”添加到关键字前面，相应的范围将被返回。如果缺少这三者中的任何一个，函数将返回`false`。例如，`concurrencyKernelKernel`示例（位于`concurrency/`子目录下）对流数和时钟周期数范围进行测量。提取这些值的代码如下。

点击这里查看代码图像

chShmooRange streamsRange;

const int numStreams = 8;

if ( ! chCommandLineGet(&streamsRange, "Streams", argc, argv) ) {

streamsRange.Initialize( numStreams );

}

chShmooRange cyclesRange;

{

const int minCycles = 8;

const int maxCycles = 512;

const int stepCycles = 8;

cyclesRange.Initialize( minCycles, maxCycles, stepCycles );

chCommandLineGet( &cyclesRange, "Cycles", argc, argv );

}

用户可以按照如下方式指定应用程序的参数。

concurrencyKernelKernel -- minStreams 2 --maxStreams 16 stepStreams 2

### A.5\. 命令行解析

一个便携的命令行解析库（仅约 100 行 C++代码）位于`chCommandLine.h`中。它包括模板函数`chCommandLineGet()`，该函数返回给定类型的变量，以及`chCommandLineGetBool()`，该函数返回命令行中是否给定了特定的关键字。

点击此处查看代码图像

template<typename T> T

chCommandLineGet( T *p, const char *keyword, int argc, char *argv[] );

如前所述，`chCommandLineGet()`的一个特化版本会返回`chShmooRange`的实例。为了编译这个特化版本，必须在`chCommandLine.h`之前包含`chShmoo.h`。

### A.6\. 错误处理

`chError.h`实现了一组宏，使用基于`goto`的错误处理机制，详见第 1.2.3 节。这些宏执行以下操作：

• 将返回值赋给名为`status`的变量

• 检查`status`是否成功，如果处于调试模式，则将错误报告到`stderr`

• 如果`status`包含错误，`goto`到名为`Error`的标签

CUDA 运行时版本如下：

点击此处查看代码图像

#ifdef DEBUG

#define CUDART_CHECK( fn ) do { \

(status) = (fn); \

if ( cudaSuccess != (status) ) { \

fprintf( stderr, "CUDA 运行时失败（文件%s 的第%d 行）：\n\t" \

"%s 返回了 0x%x (%s)\n", \

__LINE__，__FILE__，#fn，status，cudaGetErrorString(status) ); \

goto Error; \

} \

} while (0);

#else

#define CUDART_CHECK( fn ) do { \

status = (fn); \

if ( cudaSuccess != (status) ) { \

goto Error; \

} \

} while (0);

#endif

`do..while`是 C 语言中的一种惯用法，通常用于宏中，使得宏调用的结果为一个单独的语句。使用这些宏时，如果没有定义变量`status`或标签`Error:`，则会生成编译错误。

使用`goto`的一个含义是，所有变量必须在代码块的顶部声明。否则，某些编译器会生成错误，因为`goto`语句可能会绕过初始化。当发生这种情况时，初始化的变量必须移到第一个`goto`语句之上，或移到基本代码块中，这样`goto`就不会在其作用域内。

Listing A.4 给出了遵循此习惯的示例函数。返回值和中间资源被初始化为可以通过清理代码处理的值。在这种情况下，函数分配的所有资源也都由该函数释放，因此清理代码和错误处理代码是相同的。那些仅释放它们分配的部分资源的函数，必须在不同的代码块中实现成功和失败的处理。

*Listing A.4.* `goto`基础的错误处理示例。

点击这里查看代码图片

* * *

double

TimedReduction(

int *answer, const int *deviceIn, size_t N,

int cBlocks, int cThreads,

pfnReduction hostReduction

)

{

double ret = 0.0;

int *deviceAnswer = 0;

int *partialSums = 0;

cudaEvent_t start = 0;

cudaEvent_t stop = 0;

cudaError_t status;

CUDART_CHECK( cudaMalloc( &deviceAnswer, sizeof(int) ) );

CUDART_CHECK( cudaMalloc( &partialSums, cBlocks*sizeof(int) ) );

CUDART_CHECK( cudaEventCreate( &start ) );

CUDART_CHECK( cudaEventCreate( &stop ) );

CUDART_CHECK( cudaThreadSynchronize() );

CUDART_CHECK( cudaEventRecord( start, 0 ) );

hostReduction(

deviceAnswer,

partialSums,

deviceIn,

N,

cBlocks,

cThreads );

CUDART_CHECK( cudaEventRecord( stop, 0 ) );

CUDART_CHECK( cudaMemcpy(

answer,

deviceAnswer,

sizeof(int),

cudaMemcpyDeviceToHost ) );

ret = chEventBandwidth( start, stop, N*sizeof(int) ) /

powf(2.0f,30.0f);

// 通过跳过到释放资源再返回

错误：

cudaFree( deviceAnswer );

cudaFree( partialSums );

cudaEventDestroy( start );

cudaEventDestroy( stop );

返回 ret;

}

* * *

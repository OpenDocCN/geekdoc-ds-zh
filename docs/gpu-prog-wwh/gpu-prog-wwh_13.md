# 使用 MPI 的多 GPU 编程

> 原文：[`enccs.github.io/gpu-programming/10-multiple_gpu/`](https://enccs.github.io/gpu-programming/10-multiple_gpu/)

*GPU 编程：为什么、何时以及如何？* **   使用 MPI 的多 GPU 编程

+   [在 GitHub 上编辑](https://github.com/ENCCS/gpu-programming/blob/main/content/10-multiple_gpu.rst)

* * *

问题

+   应采用什么方法将同步 OpenACC 和 OpenMP 转发模型扩展到跨多个节点的多个 GPU？

目标

+   了解如何将 MPI 与 OpenACC 或 OpenMP 转发模型相结合。

+   了解如何实现具有 GPU 感知性的 MPI 方法。

教师备注

+   30 分钟教学

+   30 分钟练习

## 简介

探索跨分布式节点上的多个 GPU（图形处理单元），这有可能在大型规模上充分利用现代高性能计算（HPC）系统的能力。这里，加速分布式系统计算的一种方法是将消息传递接口（MPI）与 GPU 编程模型（如 OpenACC 和 OpenMP 应用程序编程接口（API））相结合。这种组合既受到这些 API 简单性的驱动，也受到 MPI 广泛使用的驱动。

在本指南中，我们为熟悉 MPI 的读者提供了关于实现混合模型的见解，其中 MPI 通信框架与 OpenACC 或 OpenMP API 结合。特别关注从 OpenACC 和 OpenMP API 执行点对点操作（例如 MPI_Send 和 MPI_Recv）和集体操作（例如 MPI_Allreduce）。在此，我们讨论两种场景：（i）在 CPU 主机执行 MPI 操作后，将操作卸载到 GPU 设备的场景；（ii）在两个 GPU 之间执行 MPI 操作而不涉及 CPU 主机内存的场景。后者被称为 GPU 感知 MPI，其优点是减少由于通过主机内存进行异构通信而引起的数据传输时间，从而使得 HPC 应用程序更高效。

本指南组织如下：我们首先介绍如何将每个 MPI 进程分配到同一节点内的 GPU 设备。我们考虑了一种情况，即主机和设备具有不同的内存。随后将介绍带有和没有 GPU 感知 MPI 的混合 MPI-OpenACC/OpenMP 转发。最后，提供了一些练习题，以帮助理解这些概念。

## 将 MPI 进程分配到 GPU 设备

为了在分布式节点上利用多个 GPU 加速 MPI 应用程序，首先需要将每个 MPI 进程分配到 GPU 设备，以确保两个 MPI 进程不使用相同的 GPU 设备。这是必要的，以防止应用程序发生潜在的崩溃。这是因为 GPU 设计用于处理多个线程任务，而不是多个 MPI 进程。

确保两个 MPI 进程不使用同一 GPU 的一种方法，是确定哪些 MPI 进程运行在同一个节点上，这样每个进程就可以被分配到同一节点内的一个 GPU 设备。这可以通过使用 MPI_COMM_SPLIT_TYPE()例程将世界通信器拆分为子通信器组（或子通信器）来实现。例如，这样做。

```
 ! Split the world communicator into subgroups of commu, each of which
  ! contains processes that run on the same node, and which can create a
  ! shared memory region (via the type MPI_COMM_TYPE_SHARED).
  ! The call returns a new communicator "host_comm", which is created by
  ! each subgroup. 
  call MPI_COMM_SPLIT_TYPE(MPI_COMM_WORLD,  MPI_COMM_TYPE_SHARED,  0,&
  MPI_INFO_NULL,  host_comm,ierr)
  call MPI_COMM_RANK(host_comm,  host_rank,ierr) 
```

```
 // Split the world communicator into subgroups.
  MPI_Comm  host_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD,  MPI_COMM_TYPE_SHARED,  0,  MPI_INFO_NULL,
  &host_comm);
  int  host_rank;
  MPI_Comm_rank(host_comm,  &host_rank); 
```

在这里，每个子通信器的大小对应于每个节点上的 GPU 数量（这也就是每个节点的任务数量），每个子通信器包含一个由 rank 指定的进程列表。这些进程由 MPI_COMM_TYPE_SHARED（见[MPI 报告](https://www.mpi-forum.org/docs/mpi-4.0/mpi40-report.pdf)）定义的共享内存区域共享（更多详情）。调用 MPI_COMM_SPLIT_TYPE()例程返回一个在上述代码中标有“host_comm”的子通信器，其中 MPI-ranks 从 0 到每个节点进程数-1 进行排序。这些 MPI ranks 随后被分配到同一节点内的不同 GPU 设备。这个过程根据实现的指令基于模型进行。检索到的 MPI ranks 随后存储在变量**myDevice**中。该变量如代码所示传递给 OpenACC 或 OpenMP 例程。

示例：`分配设备`

```
 myDevice  =  host_rank

  ! Sets the device number and the device type to be used
  call acc_set_device_num(myDevice,  acc_get_device_type())

  ! Returns the number of devices available on the host
  numDevice  =  acc_get_num_devices(acc_get_device_type()) 
```

```
 myDevice  =  host_rank

  ! Sets the device number to use in device constructs by setting the initial value of the default-device-var 
  call omp_set_default_device(myDevice)

  ! Returns the number of devices available for offloading.
  numDevice  =  omp_get_num_devices() 
```

```
 int  myDevice  =  host_rank;
  // Set the device number to use in device constructs.
  omp_set_default_device(myDevice);

  // Return the number of devices available for offloading.
  int  numDevice  =  omp_get_num_devices(); 
```

另一个用于检索特定设备编号的有用函数，例如，将数据映射到特定设备的有用函数是

```
acc_get_device_num() 
```

```
omp_get_device_num() 
```

将 MPI ranks 分配到 GPU 设备的语法总结如下

示例：`设置设备`

```
 ! Initialise MPI communication 
  call MPI_Init(ierr)
  ! Identify the ID rank (process) 
  call MPI_COMM_RANK(  MPI_COMM_WORLD,  myid,  ierr  )
  ! Get number of active processes (from 0 to nproc-1) 
  call MPI_COMM_SIZE(  MPI_COMM_WORLD,  nproc,  ierr  )

  ! Split the world communicator into subgroups of commu, each of which
  ! contains processes that run on the same node, and which can create a
  ! shared memory region (via the type MPI_COMM_TYPE_SHARED).
  ! The call returns a new communicator "host_comm", which is created by
  ! each subgroup. 
  call MPI_COMM_SPLIT_TYPE(MPI_COMM_WORLD,  MPI_COMM_TYPE_SHARED,  0,&
  MPI_INFO_NULL,  host_comm,ierr)
  call MPI_COMM_RANK(host_comm,  host_rank,ierr)

  ! Gets the node name
  call MPI_GET_PROCESSOR_NAME(name,  resulten,  ierror)

  myDevice  =  host_rank

  ! Sets the device number and the device type to be used
  call acc_set_device_num(myDevice,  acc_get_device_type())

  ! Returns the number of devices available on the host
  numDevice  =  acc_get_num_devices(acc_get_device_type()) 
```

```
 ! Initialise MPI communication 
  call MPI_Init(ierr)
  ! Identify the ID rank (process) 
  call MPI_COMM_RANK(  MPI_COMM_WORLD,  myid,  ierr  )
  ! Get number of active processes (from 0 to nproc-1) 
  call MPI_COMM_SIZE(  MPI_COMM_WORLD,  nproc,  ierr  )

  ! Split the world communicator into subgroups of commu, each of which
  ! contains processes that run on the same node, and which can create a
  ! shared memory region (via the type MPI_COMM_TYPE_SHARED).
  ! The call returns a new communicator "host_comm", which is created by
  ! each subgroup. 
  call MPI_COMM_SPLIT_TYPE(MPI_COMM_WORLD,  MPI_COMM_TYPE_SHARED,  0,&
  MPI_INFO_NULL,  host_comm,ierr)
  call MPI_COMM_RANK(host_comm,  host_rank,ierr)

  ! Gets the node name
  call MPI_GET_PROCESSOR_NAME(name,  resulten,  ierror)

  myDevice  =  host_rank

  ! Sets the device number to use in device constructs by setting the initial value of the default-device-var 
  call omp_set_default_device(myDevice)

  ! Returns the number of devices available for offloading.
  numDevice  =  omp_get_num_devices() 
```

```
 // Initialize MPI communication.
  MPI_Init(&argc,  &argv);

  // Identify the ID rank (process).
  int  myid,  nproc;
  MPI_Comm_rank(MPI_COMM_WORLD,  &myid);
  // Get number of active processes.
  MPI_Comm_size(MPI_COMM_WORLD,  &nproc);

  // Split the world communicator into subgroups.
  MPI_Comm  host_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD,  MPI_COMM_TYPE_SHARED,  0,  MPI_INFO_NULL,
  &host_comm);
  int  host_rank;
  MPI_Comm_rank(host_comm,  &host_rank);

  // Get the node name.
  int  name_len;
  char  processor_name[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(processor_name,  &name_len);

  int  myDevice  =  host_rank;
  // Set the device number to use in device constructs.
  omp_set_default_device(myDevice);

  // Return the number of devices available for offloading.
  int  numDevice  =  omp_get_num_devices(); 
```

## 无 GPU 感知的混合 MPI-OpenACC/OpenMP 方法

在介绍了如何将每个 MPI-rank 分配到 GPU 设备之后，我们现在讨论将 MPI 与 OpenACC 或 OpenMP 卸载相结合的概念。在这种方法中，从 OpenACC 或 OpenMP API 调用 MPI 例程需要在 MPI 调用前后更新 CPU 主机中的数据。在这种情况下，数据在每次 MPI 调用前后在主机和设备之间来回复制。在混合 MPI-OpenACC 模型中，该过程通过指定在 MPI 调用之前将数据从设备复制到主机的指令 update host()来定义；通过在 MPI 调用之后指定指令 update device()来将数据复制回设备。在混合 MPI-OpenMP 中类似地，这里，可以通过指定 OpenMP 指令 update device() from()和 update device() to()来更新主机中的数据，分别用于将数据从设备复制到主机和从主机复制回设备。

为了说明混合 MPI-OpenACC/OpenMP 的概念，我们下面展示了一个涉及 MPI 函数 MPI_Send()和 MPI_Recv()的实现示例。

示例：`更新主机/设备指令`

```
 !offload f to GPUs
  !$acc enter data copyin(f)

  !update f: copy f from GPU to CPU
  !$acc update host(f)

  if(myid.lt.nproc-1)  then
 call MPI_Send(f(np:np),1,MPI_DOUBLE_PRECISION,myid+1,tag,MPI_COMM_WORLD,  ierr)
  endif

 if(myid.gt.0)  then
 call MPI_Recv(f(1),1,MPI_DOUBLE_PRECISION,myid-1,tag,MPI_COMM_WORLD,  status,ierr)
  endif

  !update f: copy f from CPU to GPU
  !$acc update device(f) 
```

```
 !offload f to GPUs
  !$omp target enter data device(myDevice) map(to:f)

  !update f: copy f from GPU to CPU
  !$omp target update device(myDevice) from(f)

  if(myid.lt.nproc-1)  then
 call MPI_Send(f(np:np),1,MPI_DOUBLE_PRECISION,myid+1,tag,MPI_COMM_WORLD,  ierr)
  endif

 if(myid.gt.0)  then
 call MPI_Recv(f(1),1,MPI_DOUBLE_PRECISION,myid-1,tag,MPI_COMM_WORLD,  status,ierr)
  endif

  !update f: copy f from CPU to GPU
  !$omp target update device(myDevice) to(f) 
```

```
// Offload f to GPUs
#pragma omp target enter data map(to : f[0 : np]) device(myDevice)

  // update f: copy f from GPU to CPU
#pragma omp target update device(myDevice) from(f)

  if  (myid  <  nproc  -  1)  {
  MPI_Send(&f[np  -  1],  1,  MPI_DOUBLE,  myid  +  1,  tag,  MPI_COMM_WORLD);
  }
  if  (myid  >  0)  {
  MPI_Recv(&f[0],  1,  MPI_DOUBLE,  myid  -  1,  tag,  MPI_COMM_WORLD,
  MPI_STATUS_IGNORE);
  }

// update f: copy f from CPU to GPU
#pragma omp target update device(myDevice) to(f) 
```

在这里，我们提供了一个代码示例，它结合了 MPI 与 OpenACC/OpenMP API。

示例：`更新主机/设备指令`

```
 call MPI_Scatter(f_send,np,MPI_DOUBLE_PRECISION,f,  np,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,  ierr)

  !offload f to GPUs
  !$acc enter data copyin(f)

  !update f: copy f from GPU to CPU
  !$acc update host(f)

  if(myid.lt.nproc-1)  then
 call MPI_Send(f(np:np),1,MPI_DOUBLE_PRECISION,myid+1,tag,MPI_COMM_WORLD,  ierr)
  endif

 if(myid.gt.0)  then
 call MPI_Recv(f(1),1,MPI_DOUBLE_PRECISION,myid-1,tag,MPI_COMM_WORLD,  status,ierr)
  endif

  !update f: copy f from CPU to GPU
  !$acc update device(f)

  !do something .e.g.
  !$acc kernels
  f  =  f/2.
  !$acc end kernels

  SumToT=0d0
  !$acc parallel loop reduction(+:SumToT)
  do k=1,np
  SumToT  =  SumToT  +  f(k)
  enddo
  !$acc end parallel loop

  !SumToT is by default copied back to the CPU
  call MPI_ALLREDUCE(MPI_IN_PLACE,SumToT,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr  )

  !$acc exit data delete(f) 
```

```
 call MPI_Scatter(f_send,np,MPI_DOUBLE_PRECISION,f,  np,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,  ierr)

  !offload f to GPUs
  !$omp target enter data device(myDevice) map(to:f)

  !update f: copy f from GPU to CPU
  !$omp target update device(myDevice) from(f)

  if(myid.lt.nproc-1)  then
 call MPI_Send(f(np:np),1,MPI_DOUBLE_PRECISION,myid+1,tag,MPI_COMM_WORLD,  ierr)
  endif

 if(myid.gt.0)  then
 call MPI_Recv(f(1),1,MPI_DOUBLE_PRECISION,myid-1,tag,MPI_COMM_WORLD,  status,ierr)
  endif

  !update f: copy f from CPU to GPU
  !$omp target update device(myDevice) to(f)

  !do something .e.g.
  !$omp target teams distribute parallel do
  do k=1,np
  f(k)  =  f(k)/2.
  enddo
  !$omp end target teams distribute parallel do

  SumToT=0d0
  !$omp target teams distribute parallel do reduction(+:SumToT)
  do k=1,np
  SumToT  =  SumToT  +  f(k)
  enddo
  !$omp end target teams distribute parallel do 

  !SumToT is by default copied back to the CPU
  call MPI_ALLREDUCE(MPI_IN_PLACE,SumToT,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr  )

  !$omp target exit data map(delete:f) 
```

```
 MPI_Scatter(f_send.data(),  np,  MPI_DOUBLE,  f,  np,  MPI_DOUBLE,  0,
  MPI_COMM_WORLD);

// Offload f to GPUs
#pragma omp target enter data map(to : f[0 : np]) device(myDevice)

  // update f: copy f from GPU to CPU
#pragma omp target update device(myDevice) from(f)

  if  (myid  <  nproc  -  1)  {
  MPI_Send(&f[np  -  1],  1,  MPI_DOUBLE,  myid  +  1,  tag,  MPI_COMM_WORLD);
  }
  if  (myid  >  0)  {
  MPI_Recv(&f[0],  1,  MPI_DOUBLE,  myid  -  1,  tag,  MPI_COMM_WORLD,
  MPI_STATUS_IGNORE);
  }

// update f: copy f from CPU to GPU
#pragma omp target update device(myDevice) to(f)

// Update, operate, and offload data back to GPUs
#pragma omp target teams distribute parallel for device(myDevice)
  for  (int  k  =  0;  k  <  np;  ++k)  {
  f[k]  /=  2.0;
  }

  double  SumToT  =  0.0;

#pragma omp target teams distribute parallel for reduction(+ : SumToT)         \
 device(myDevice)
  for  (int  k  =  0;  k  <  np;  ++k)  {
  SumToT  +=  f[k];
  }

  MPI_Allreduce(MPI_IN_PLACE,  &SumToT,  1,  MPI_DOUBLE,  MPI_SUM,  MPI_COMM_WORLD);

#pragma omp target exit data map(delete : f[0 : np]) 
```

尽管实现混合 MPI-OpenACC/OpenMP 卸载的过程很简单，但在调用 MPI 例程前后，主机和设备之间进行显式数据传输导致性能低下。这构成了 GPU 编程的瓶颈。为了提高数据传输过程中受主机阶段影响的性能，可以实施如以下章节所述的 GPU 感知 MPI 方法。

## 混合 MPI-OpenACC/OpenMP 采用 GPU 感知方法

GPU 感知 MPI 的概念允许 MPI 库直接访问 GPU 设备内存，而不必一定使用 CPU 主机内存作为中间缓冲区（例如，参见[OpenMPI 文档](https://docs.open-mpi.org/en/v5.0.1/tuning-apps/networking/cuda.html)）。这提供了从一台 GPU 传输数据到另一台 GPU 而不涉及 CPU 主机内存的好处。

具体来说，在 GPU 感知方法中，设备指针指向分配在 GPU 内存空间中的数据（数据应存在于 GPU 设备中）。在这里，指针作为参数传递给支持 GPU 内存的 MPI 例程。由于 MPI 例程可以直接访问 GPU 内存，它提供了在 GPU 之间进行通信的可能性，而不需要将数据传输回主机。

在混合 MPI-OpenACC 模型中，该概念通过结合指令 host_data 和 use_device(list_array)子句来定义。这种组合使得可以从主机访问 use_device(list_array)子句中列出的数组（参见[此处](https://www.openacc.org/sites/default/files/inline-images/Specification/OpenACC-3.2-final.pdf)）。已存在于 GPU 设备内存中的数组列表直接传递给 MPI 例程，无需为复制数据而设置主机内存阶段。请注意，对于最初将数据复制到 GPU，我们使用由 enter data 和 exit data 指令定义的无结构数据块。无结构数据具有允许在数据区域内分配和释放数组的优点。

为了说明 GPU 感知 MPI 的概念，我们下面展示了两个使用 OpenACC 和 OpenMP API 中的点对点和集体操作的示例。在第一个代码示例中，设备指针**f**被传递给 MPI 函数 MPI_Send()和 MPI_Recv()；在第二个示例中，指针**SumToT**被传递给 MPI 函数 MPI_Allreduce。在这里，MPI 操作 MPI_Send 和 MPI_Recv 以及 MPI_Allreduce 是在一对 GPU 之间执行的，而不需要通过 CPU 主机内存。

示例：`GPU 感知：MPI_Send & MPI_Recv`

```
 !Device pointer f will be passed to MPI_send & MPI_recv
  !$acc host_data use_device(f)
  if(myid.lt.nproc-1)  then
 call MPI_Send(f(np:np),1,MPI_DOUBLE_PRECISION,myid+1,tag,MPI_COMM_WORLD,  ierr)
  endif

 if(myid.gt.0)  then
 call MPI_Recv(f(1),1,MPI_DOUBLE_PRECISION,myid-1,tag,MPI_COMM_WORLD,  status,ierr)
  endif
  !$acc end host_data 
```

```
 !Device pointer f will be passed to MPI_send & MPI_recv
  !$omp target data use_device_ptr(f)
  if(myid.lt.nproc-1)  then
 call MPI_Send(f(np:np),1,MPI_DOUBLE_PRECISION,myid+1,tag,MPI_COMM_WORLD,  ierr)
  endif

 if(myid.gt.0)  then
 call MPI_Recv(f(1),1,MPI_DOUBLE_PRECISION,myid-1,tag,MPI_COMM_WORLD,  status,ierr)
  endif
  !$omp end target data 
```

```
#pragma omp target data use_device_ptr(f)
  {
  if  (myid  <  nproc  -  1)  {
  MPI_Send(&f[np  -  1],  1,  MPI_DOUBLE,  myid  +  1,  tag,  MPI_COMM_WORLD);
  }

  if  (myid  >  0)  {
  MPI_Recv(&f[0],  1,  MPI_DOUBLE,  myid  -  1,  tag,  MPI_COMM_WORLD,
  MPI_STATUS_IGNORE);
  }
  } 
```

示例：`GPU 感知：MPI_Allreduce`

```
 !$acc data copy(SumToT)
  !$acc host_data use_device(SumToT)
  call MPI_ALLREDUCE(MPI_IN_PLACE,SumToT,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr  )
  !$acc end host_data
  !$acc end data 
```

```
 !$omp target enter data device(myDevice) map(to:SumToT)
  !$omp target data use_device_ptr(SumToT)
  call MPI_ALLREDUCE(MPI_IN_PLACE,SumToT,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr  )
  !$omp end target data
  !$omp target exit data map(from:SumToT) 
```

```
#pragma omp target enter data device(myDevice) map(to : SumToT)
  double  *SumToTPtr  =  &SumToT;
#pragma omp target data use_device_ptr(SumToTPtr)
  {
  MPI_Allreduce(MPI_IN_PLACE,  SumToTPtr,  1,  MPI_DOUBLE,  MPI_SUM,
  MPI_COMM_WORLD);
  }
#pragma omp target exit data map(from : SumToT) 
```

我们下面提供了一个代码示例，说明了在 OpenACC/OpenMP API 中实现 MPI 函数 MPI_Send()、MPI_Recv()和 MPI_Allreduce()的方法。这种实现专门设计来支持 GPU 感知 MPI 操作。

示例：`GPU 感知方法`

```
 call MPI_Scatter(f_send,np,MPI_DOUBLE_PRECISION,f,  np,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,  ierr)

  !offload f to GPUs
  !$acc enter data copyin(f)

  !Device pointer f will be passed to MPI_send & MPI_recv
  !$acc host_data use_device(f)
  if(myid.lt.nproc-1)  then
 call MPI_Send(f(np:np),1,MPI_DOUBLE_PRECISION,myid+1,tag,MPI_COMM_WORLD,  ierr)
  endif

 if(myid.gt.0)  then
 call MPI_Recv(f(1),1,MPI_DOUBLE_PRECISION,myid-1,tag,MPI_COMM_WORLD,  status,ierr)
  endif
  !$acc end host_data

  !do something .e.g.
  !$acc kernels
  f  =  f/2.
  !$acc end kernels

  SumToT=0d0
  !$acc parallel loop reduction(+:SumToT)
  do k=1,np
  SumToT  =  SumToT  +  f(k)
  enddo
  !$acc end parallel loop

  !SumToT is by default copied back to the CPU

  !$acc data copy(SumToT)
  !$acc host_data use_device(SumToT)
  call MPI_ALLREDUCE(MPI_IN_PLACE,SumToT,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr  )
  !$acc end host_data
  !$acc end data

  !$acc exit data delete(f) 
```

```
 call MPI_Scatter(f_send,np,MPI_DOUBLE_PRECISION,f,  np,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,  ierr)

  !offload f to GPUs
  !$omp target enter data device(myDevice) map(to:f)

  !Device pointer f will be passed to MPI_send & MPI_recv
  !$omp target data use_device_ptr(f)
  if(myid.lt.nproc-1)  then
 call MPI_Send(f(np:np),1,MPI_DOUBLE_PRECISION,myid+1,tag,MPI_COMM_WORLD,  ierr)
  endif

 if(myid.gt.0)  then
 call MPI_Recv(f(1),1,MPI_DOUBLE_PRECISION,myid-1,tag,MPI_COMM_WORLD,  status,ierr)
  endif
  !$omp end target data

  !do something .e.g.
  !$omp target teams distribute parallel do
  do k=1,np
  f(k)  =  f(k)/2.
  enddo
  !$omp end target teams distribute parallel do

  SumToT=0d0
  !$omp target teams distribute parallel do reduction(+:SumToT)
  do k=1,np
  SumToT  =  SumToT  +  f(k)
  enddo
  !$omp end target teams distribute parallel do 

  !SumToT is by default copied back to the CPU

  !$omp target enter data device(myDevice) map(to:SumToT)
  !$omp target data use_device_ptr(SumToT)
  call MPI_ALLREDUCE(MPI_IN_PLACE,SumToT,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr  )
  !$omp end target data
  !$omp target exit data map(from:SumToT)

  !$omp target exit data map(delete:f) 
```

```
 call  MPI_Scatter(f_send,np,MPI_DOUBLE_PRECISION,f,  np,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,  ierr)

  !offload  f  to  GPUs
  !$omp  target  enter  data  device(myDevice)  map(to:f)

  !Device  pointer  f  will  be  passed  to  MPI_send  &  MPI_recv
  !$omp  target  data  use_device_ptr(f)
  if(myid.lt.nproc-1)  then
  call  MPI_Send(f(np:np),1,MPI_DOUBLE_PRECISION,myid+1,tag,MPI_COMM_WORLD,  ierr)
  endif

  if(myid.gt.0)  then
  call  MPI_Recv(f(1),1,MPI_DOUBLE_PRECISION,myid-1,tag,MPI_COMM_WORLD,  status,ierr)
  endif
  !$omp  end  target  data

  !do  something  .e.g.
  !$omp  target  teams  distribute  parallel  do
  do  k=1,np
  f(k)  =  f(k)/2.
  enddo
  !$omp  end  target  teams  distribute  parallel  do

  SumToT=0d0
  !$omp  target  teams  distribute  parallel  do  reduction(+:SumToT)
  do  k=1,np
  SumToT  =  SumToT  +  f(k)
  enddo
  !$omp  end  target  teams  distribute  parallel  do  

  !SumToT  is  by  default  copied  back  to  the  CPU

  !$omp  target  enter  data  device(myDevice)  map(to:SumToT)
  !$omp  target  data  use_device_ptr(SumToT)
  call  MPI_ALLREDUCE(MPI_IN_PLACE,SumToT,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr  )
  !$omp  end  target  data
  !$omp  target  exit  data  map(from:SumToT)

  !$omp  target  exit  data  map(delete:f) 
```

带有 OpenACC/OpenMP API 的 GPU 感知 MPI 具有在单个节点内直接在两个 GPU 之间通信的能力。然而，在多个节点之间执行 GPU 到 GPU 的通信需要 GPUDirect RDMA（远程直接内存访问）技术。这项技术可以通过减少延迟来进一步提高性能。

## 编译过程

下面描述了混合 MPI-OpenACC 和 MPI-OpenMP 卸载的编译过程。这个描述是为包装器 ftn 的 Cray 编译器提供的。在 LUMI-G 上，在编译之前可能需要以下模块（有关可用编程环境的更多详细信息，请参阅[LUMI 文档](https://docs.lumi-supercomputer.eu/development/compiling/prgenv/)）：

```
$ ml  LUMI/24.03
$ ml  PrgEnv-cray
$ ml  cray-mpich
$ ml  rocm
$ ml  craype-accel-amd-gfx90a 
```

示例：`编译过程`

```
$ ftn  -hacc  -o  mycode.mpiacc.exe  mycode_mpiacc.f90 
```

```
$ ftn  -homp  -o  mycode.mpiomp.exe  mycode_mpiomp.f90 
```

```
$ CC  -fopenmp  -fopenmp-targets=amdgcn-amd-amdhsa  -Xopenmp-target  -march=gfx90a  -o  mycode.mpiomp.exe  mycode_mpiomp.cpp 
```

在这里，标志 hacc 和 homp 分别启用混合 MPI-OpenACC 和 MPI-OpenMP 应用程序中的 OpenACC 和 OpenMP 指令。

**启用 GPU 感知支持**

要在 MPICH 库中启用 GPU 感知支持，在运行应用程序之前需要设置以下环境变量。

```
$ export MPICH_GPU_SUPPORT_ENABLED=1 
```

## 结论

总结来说，我们通过整合 GPU 指令模型，特别是 OpenACC 和 OpenMP API 与 MPI 库，展示了 GPU-混合编程的概述。这里采用的方法不仅允许我们在单个节点内利用多个 GPU 设备，而且还能扩展到分布式节点。特别是，我们解决了 GPU 感知的 MPI 方法，它具有允许 MPI 库与 GPU 设备内存直接交互的优势。换句话说，它允许在两个 GPU 之间执行 MPI 操作，从而减少由数据局部性引起的计算时间。

## 练习

我们考虑了一个解决二维拉普拉斯方程的 MPI Fortran 代码，并且部分加速。练习的重点是使用 OpenACC 或 OpenMP API，按照以下步骤完成加速。

访问练习材料

以下练习的代码示例可以在本存储库的 content/examples/exercise_multipleGPU 子目录中访问。要访问它们，您需要克隆存储库：

```
$ git  clone  https://github.com/ENCCS/gpu-programming.git
$ cd  gpu-programming/content/examples/exercise_multipleGPU
$ ls 
```

练习 I：设置 GPU 设备

1.  实现 OpenACC/OpenMP 函数，使每个 MPI 排名分配到 GPU 设备。

1.1 在多个 GPU 上编译和运行代码。

练习 II：应用传统的 MPI-OpenACC/OpenMP

2.1 在分别调用 MPI 函数前后，插入 OpenACC 指令`update host()`和`update device()`。

注意

OpenACC 指令`update host()`用于在数据区域内将数据从 GPU 传输到 CPU；而指令`update device()`用于将数据从 CPU 传输到 GPU。

2.2 在分别调用 MPI 函数前后，插入 OpenMP 指令`update device() from()`和`update device() to()`。

注意

OpenMP 指令 *update device() from()* 用于在数据区域内将数据从 GPU 转移到 CPU；而指令 *update device() to()* 用于将数据从 CPU 转移到 GPU。

2.3 在多个 GPU 上编译和运行代码。

练习 III：实现 GPU 感知支持

3.1 将 OpenACC 指令 *host_data use_device()* 集成到代码中，以将设备指针传递给 MPI 函数。

3.2 将 OpenMP 指令 *data use_device_ptr()* 集成到代码中，以将设备指针传递给 MPI 函数。

3.3 在多个 GPU 上编译和运行代码。

练习 IV：评估性能

1.  评估练习 II 和 III 中加速代码的执行时间，并将其与纯 MPI 实现进行比较。

## 参考内容

+   [GPU 感知 MPI](https://documentation.sigma2.no/code_development/guides/gpuaware_mpi.html).

+   [MPI 文档](https://www.mpi-forum.org/docs/mpi-4.0/mpi40-report.pdf).

+   [OpenACC 规范](https://www.openacc.org/sites/default/files/inline-images/Specification/OpenACC-3.2-final.pdf).

+   [OpenMP 规范](https://www.openmp.org/wp-content/uploads/OpenMP-API-Specification-5-2.pdf).

+   [LUMI 文档](https://docs.lumi-supercomputer.eu/development/compiling/prgenv/).

+   [OpenACC 与 OpenMP 转移](https://documentation.sigma2.no/code_development/guides/converting_acc2omp/openacc2openmp.html).

+   [OpenACC 课程](https://github.com/HichamAgueny/GPU-course). 上一节 下一节

* * *

© 版权所有 2023-2024，贡献者。

使用 [Sphinx](https://www.sphinx-doc.org/) 和由 [Read the Docs](https://readthedocs.org) 提供的 [主题](https://github.com/readthedocs/sphinx_rtd_theme) 构建。问题

+   应采用什么方法将同步的 OpenACC 和 OpenMP 转移模型扩展到多个节点上的多个 GPU 以利用其能力？

目标

+   了解如何将 MPI 与 OpenACC 或 OpenMP 转移模型相结合。

+   了解实现 GPU 感知 MPI 方法。

教师备注

+   30 分钟教学

+   30 分钟练习

## 简介

探索分布式节点上的多个 GPU（图形处理单元）提供了充分利用大规模现代高性能计算系统（HPC）能力的潜力。这里加速分布式系统计算的一种方法是将 MPI（消息传递接口）与 GPU 编程模型（如 OpenACC 和 OpenMP 应用程序编程接口）相结合。这种组合既受到这些 API 简单性的驱动，也受到 MPI 广泛使用的推动。

在本指南中，我们为熟悉 MPI 的读者提供了关于实现混合模型的见解，其中 MPI 通信框架与 OpenACC 或 OpenMP API 之一相结合。特别关注从 OpenACC 和 OpenMP API 执行点对点操作（例如 MPI_Send 和 MPI_Recv）和集体操作（例如 MPI_Allreduce）。我们讨论了两种场景：（i）在 CPU 主机执行 MPI 操作后，将数据卸载到 GPU 设备的场景；（ii）在涉及 CPU 主机内存的情况下，在两个 GPU 之间执行 MPI 操作的场景。后者被称为 GPU 感知 MPI，其优点是减少由于在异构通信期间通过主机内存传输数据而造成的计算时间，从而使 HPC 应用程序更高效。

本指南组织如下：我们首先介绍如何在同一节点内将每个 MPI 进程分配到 GPU 设备。我们考虑了一种情况，其中主机和设备具有不同的内存。随后，我们将介绍带有和没有 GPU 感知 MPI 的混合 MPI-OpenACC/OpenMP 转发。最后，提供了一些练习，以帮助理解这些概念。

## 将 MPI 进程分配到 GPU 设备

加速 MPI 应用以利用分布式节点上的多个 GPU，首先需要将每个 MPI 进程分配到特定的 GPU 设备上，确保两个 MPI 进程不会使用相同的 GPU 设备。这是为了防止应用程序发生潜在的崩溃。这是因为 GPU 设计用于处理多个线程任务，而不是多个 MPI 进程。

确保两个 MPI 进程不使用同一 GPU 的方法之一是确定哪些 MPI 进程运行在同一个节点上，这样每个进程都可以分配到同一节点内的 GPU 设备。例如，可以通过使用 MPI_COMM_SPLIT_TYPE() 例程将全局通信器拆分为子通信器组（或子通信器）来实现这一点。

```
 ! Split the world communicator into subgroups of commu, each of which
  ! contains processes that run on the same node, and which can create a
  ! shared memory region (via the type MPI_COMM_TYPE_SHARED).
  ! The call returns a new communicator "host_comm", which is created by
  ! each subgroup. 
  call MPI_COMM_SPLIT_TYPE(MPI_COMM_WORLD,  MPI_COMM_TYPE_SHARED,  0,&
  MPI_INFO_NULL,  host_comm,ierr)
  call MPI_COMM_RANK(host_comm,  host_rank,ierr) 
```

```
 // Split the world communicator into subgroups.
  MPI_Comm  host_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD,  MPI_COMM_TYPE_SHARED,  0,  MPI_INFO_NULL,
  &host_comm);
  int  host_rank;
  MPI_Comm_rank(host_comm,  &host_rank); 
```

在这里，每个子通信器的大小对应于每个节点上的 GPU 数量（这同时也是每个节点上的任务数量），每个子通信器包含一个由排名指示的进程列表。这些进程由 MPI_COMM_TYPE_SHARED 参数定义的共享内存区域共享（有关更多详细信息，请参阅 [MPI 报告](https://www.mpi-forum.org/docs/mpi-4.0/mpi40-report.pdf)）。调用 MPI_COMM_SPLIT_TYPE() 例程返回一个在代码中标记为 *“host_comm”* 的子通信器，其中 MPI 进程的排名从 0 到每个节点进程数减 1。这些 MPI 进程随后被分配到同一节点内的不同 GPU 设备。此过程根据实现的指令模型进行。检索到的 MPI 进程存储在变量 **myDevice** 中。该变量根据代码下面的指示传递给 OpenACC 或 OpenMP 例程。

示例：`分配设备`

```
 myDevice  =  host_rank

  ! Sets the device number and the device type to be used
  call acc_set_device_num(myDevice,  acc_get_device_type())

  ! Returns the number of devices available on the host
  numDevice  =  acc_get_num_devices(acc_get_device_type()) 
```

```
 myDevice  =  host_rank

  ! Sets the device number to use in device constructs by setting the initial value of the default-device-var 
  call omp_set_default_device(myDevice)

  ! Returns the number of devices available for offloading.
  numDevice  =  omp_get_num_devices() 
```

```
 int  myDevice  =  host_rank;
  // Set the device number to use in device constructs.
  omp_set_default_device(myDevice);

  // Return the number of devices available for offloading.
  int  numDevice  =  omp_get_num_devices(); 
```

另一个用于检索特定设备设备号的实用函数，这在例如将数据映射到特定设备时很有用

```
acc_get_device_num() 
```

```
omp_get_device_num() 
```

分配 MPI 排名到 GPU 设备的语法总结如下

示例：`设置设备`

```
 ! Initialise MPI communication 
  call MPI_Init(ierr)
  ! Identify the ID rank (process) 
  call MPI_COMM_RANK(  MPI_COMM_WORLD,  myid,  ierr  )
  ! Get number of active processes (from 0 to nproc-1) 
  call MPI_COMM_SIZE(  MPI_COMM_WORLD,  nproc,  ierr  )

  ! Split the world communicator into subgroups of commu, each of which
  ! contains processes that run on the same node, and which can create a
  ! shared memory region (via the type MPI_COMM_TYPE_SHARED).
  ! The call returns a new communicator "host_comm", which is created by
  ! each subgroup. 
  call MPI_COMM_SPLIT_TYPE(MPI_COMM_WORLD,  MPI_COMM_TYPE_SHARED,  0,&
  MPI_INFO_NULL,  host_comm,ierr)
  call MPI_COMM_RANK(host_comm,  host_rank,ierr)

  ! Gets the node name
  call MPI_GET_PROCESSOR_NAME(name,  resulten,  ierror)

  myDevice  =  host_rank

  ! Sets the device number and the device type to be used
  call acc_set_device_num(myDevice,  acc_get_device_type())

  ! Returns the number of devices available on the host
  numDevice  =  acc_get_num_devices(acc_get_device_type()) 
```

```
 ! Initialise MPI communication 
  call MPI_Init(ierr)
  ! Identify the ID rank (process) 
  call MPI_COMM_RANK(  MPI_COMM_WORLD,  myid,  ierr  )
  ! Get number of active processes (from 0 to nproc-1) 
  call MPI_COMM_SIZE(  MPI_COMM_WORLD,  nproc,  ierr  )

  ! Split the world communicator into subgroups of commu, each of which
  ! contains processes that run on the same node, and which can create a
  ! shared memory region (via the type MPI_COMM_TYPE_SHARED).
  ! The call returns a new communicator "host_comm", which is created by
  ! each subgroup. 
  call MPI_COMM_SPLIT_TYPE(MPI_COMM_WORLD,  MPI_COMM_TYPE_SHARED,  0,&
  MPI_INFO_NULL,  host_comm,ierr)
  call MPI_COMM_RANK(host_comm,  host_rank,ierr)

  ! Gets the node name
  call MPI_GET_PROCESSOR_NAME(name,  resulten,  ierror)

  myDevice  =  host_rank

  ! Sets the device number to use in device constructs by setting the initial value of the default-device-var 
  call omp_set_default_device(myDevice)

  ! Returns the number of devices available for offloading.
  numDevice  =  omp_get_num_devices() 
```

```
 // Initialize MPI communication.
  MPI_Init(&argc,  &argv);

  // Identify the ID rank (process).
  int  myid,  nproc;
  MPI_Comm_rank(MPI_COMM_WORLD,  &myid);
  // Get number of active processes.
  MPI_Comm_size(MPI_COMM_WORLD,  &nproc);

  // Split the world communicator into subgroups.
  MPI_Comm  host_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD,  MPI_COMM_TYPE_SHARED,  0,  MPI_INFO_NULL,
  &host_comm);
  int  host_rank;
  MPI_Comm_rank(host_comm,  &host_rank);

  // Get the node name.
  int  name_len;
  char  processor_name[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(processor_name,  &name_len);

  int  myDevice  =  host_rank;
  // Set the device number to use in device constructs.
  omp_set_default_device(myDevice);

  // Return the number of devices available for offloading.
  int  numDevice  =  omp_get_num_devices(); 
```

## 混合 MPI-OpenACC/OpenMP 无 GPU 感知方法

在介绍了如何将每个 MPI 排名分配给 GPU 设备之后，我们现在讨论将 MPI 与 OpenACC 或 OpenMP 转发相结合的概念。在这种方法中，从 OpenACC 或 OpenMP API 调用 MPI 例程需要在 MPI 调用前后更新 CPU 主机中的数据。在这种情况下，数据在每次 MPI 调用前后在主机和设备之间来回复制。在混合 MPI-OpenACC 模型中，过程是通过指定在 MPI 调用前复制数据到主机的指令 update host()；以及通过在 MPI 调用后指定的指令 update device() 来定义的，用于将数据复制回设备。在混合 MPI-OpenMP 中类似。在这里，可以通过指定 OpenMP 指令 update device() from() 和 update device() to() 分别来更新主机中的数据，用于将数据从设备复制到主机，然后再从主机复制回设备。

为了说明混合 MPI-OpenACC/OpenMP 的概念，我们下面展示了一个涉及 MPI 函数 MPI_Send() 和 MPI_Recv() 的实现示例。

示例：`更新主机/设备指令`

```
 !offload f to GPUs
  !$acc enter data copyin(f)

  !update f: copy f from GPU to CPU
  !$acc update host(f)

  if(myid.lt.nproc-1)  then
 call MPI_Send(f(np:np),1,MPI_DOUBLE_PRECISION,myid+1,tag,MPI_COMM_WORLD,  ierr)
  endif

 if(myid.gt.0)  then
 call MPI_Recv(f(1),1,MPI_DOUBLE_PRECISION,myid-1,tag,MPI_COMM_WORLD,  status,ierr)
  endif

  !update f: copy f from CPU to GPU
  !$acc update device(f) 
```

```
 !offload f to GPUs
  !$omp target enter data device(myDevice) map(to:f)

  !update f: copy f from GPU to CPU
  !$omp target update device(myDevice) from(f)

  if(myid.lt.nproc-1)  then
 call MPI_Send(f(np:np),1,MPI_DOUBLE_PRECISION,myid+1,tag,MPI_COMM_WORLD,  ierr)
  endif

 if(myid.gt.0)  then
 call MPI_Recv(f(1),1,MPI_DOUBLE_PRECISION,myid-1,tag,MPI_COMM_WORLD,  status,ierr)
  endif

  !update f: copy f from CPU to GPU
  !$omp target update device(myDevice) to(f) 
```

```
// Offload f to GPUs
#pragma omp target enter data map(to : f[0 : np]) device(myDevice)

  // update f: copy f from GPU to CPU
#pragma omp target update device(myDevice) from(f)

  if  (myid  <  nproc  -  1)  {
  MPI_Send(&f[np  -  1],  1,  MPI_DOUBLE,  myid  +  1,  tag,  MPI_COMM_WORLD);
  }
  if  (myid  >  0)  {
  MPI_Recv(&f[0],  1,  MPI_DOUBLE,  myid  -  1,  tag,  MPI_COMM_WORLD,
  MPI_STATUS_IGNORE);
  }

// update f: copy f from CPU to GPU
#pragma omp target update device(myDevice) to(f) 
```

在这里，我们提供了一个结合 MPI 与 OpenACC/OpenMP API 的代码示例。

示例：`更新主机/设备指令`

```
 call MPI_Scatter(f_send,np,MPI_DOUBLE_PRECISION,f,  np,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,  ierr)

  !offload f to GPUs
  !$acc enter data copyin(f)

  !update f: copy f from GPU to CPU
  !$acc update host(f)

  if(myid.lt.nproc-1)  then
 call MPI_Send(f(np:np),1,MPI_DOUBLE_PRECISION,myid+1,tag,MPI_COMM_WORLD,  ierr)
  endif

 if(myid.gt.0)  then
 call MPI_Recv(f(1),1,MPI_DOUBLE_PRECISION,myid-1,tag,MPI_COMM_WORLD,  status,ierr)
  endif

  !update f: copy f from CPU to GPU
  !$acc update device(f)

  !do something .e.g.
  !$acc kernels
  f  =  f/2.
  !$acc end kernels

  SumToT=0d0
  !$acc parallel loop reduction(+:SumToT)
  do k=1,np
  SumToT  =  SumToT  +  f(k)
  enddo
  !$acc end parallel loop

  !SumToT is by default copied back to the CPU
  call MPI_ALLREDUCE(MPI_IN_PLACE,SumToT,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr  )

  !$acc exit data delete(f) 
```

```
 call MPI_Scatter(f_send,np,MPI_DOUBLE_PRECISION,f,  np,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,  ierr)

  !offload f to GPUs
  !$omp target enter data device(myDevice) map(to:f)

  !update f: copy f from GPU to CPU
  !$omp target update device(myDevice) from(f)

  if(myid.lt.nproc-1)  then
 call MPI_Send(f(np:np),1,MPI_DOUBLE_PRECISION,myid+1,tag,MPI_COMM_WORLD,  ierr)
  endif

 if(myid.gt.0)  then
 call MPI_Recv(f(1),1,MPI_DOUBLE_PRECISION,myid-1,tag,MPI_COMM_WORLD,  status,ierr)
  endif

  !update f: copy f from CPU to GPU
  !$omp target update device(myDevice) to(f)

  !do something .e.g.
  !$omp target teams distribute parallel do
  do k=1,np
  f(k)  =  f(k)/2.
  enddo
  !$omp end target teams distribute parallel do

  SumToT=0d0
  !$omp target teams distribute parallel do reduction(+:SumToT)
  do k=1,np
  SumToT  =  SumToT  +  f(k)
  enddo
  !$omp end target teams distribute parallel do 

  !SumToT is by default copied back to the CPU
  call MPI_ALLREDUCE(MPI_IN_PLACE,SumToT,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr  )

  !$omp target exit data map(delete:f) 
```

```
 MPI_Scatter(f_send.data(),  np,  MPI_DOUBLE,  f,  np,  MPI_DOUBLE,  0,
  MPI_COMM_WORLD);

// Offload f to GPUs
#pragma omp target enter data map(to : f[0 : np]) device(myDevice)

  // update f: copy f from GPU to CPU
#pragma omp target update device(myDevice) from(f)

  if  (myid  <  nproc  -  1)  {
  MPI_Send(&f[np  -  1],  1,  MPI_DOUBLE,  myid  +  1,  tag,  MPI_COMM_WORLD);
  }
  if  (myid  >  0)  {
  MPI_Recv(&f[0],  1,  MPI_DOUBLE,  myid  -  1,  tag,  MPI_COMM_WORLD,
  MPI_STATUS_IGNORE);
  }

// update f: copy f from CPU to GPU
#pragma omp target update device(myDevice) to(f)

// Update, operate, and offload data back to GPUs
#pragma omp target teams distribute parallel for device(myDevice)
  for  (int  k  =  0;  k  <  np;  ++k)  {
  f[k]  /=  2.0;
  }

  double  SumToT  =  0.0;

#pragma omp target teams distribute parallel for reduction(+ : SumToT)         \
 device(myDevice)
  for  (int  k  =  0;  k  <  np;  ++k)  {
  SumToT  +=  f[k];
  }

  MPI_Allreduce(MPI_IN_PLACE,  &SumToT,  1,  MPI_DOUBLE,  MPI_SUM,  MPI_COMM_WORLD);

#pragma omp target exit data map(delete : f[0 : np]) 
```

尽管实现混合 MPI-OpenACC/OpenMP 转发的过程很简单，但它由于在调用 MPI 例程前后显式地在主机和设备之间传输数据而导致性能低下。这构成了 GPU 编程的瓶颈。为了提高数据传输过程中主机阶段影响到的性能，可以实施如以下章节所述的 GPU 感知 MPI 方法。

## 混合 MPI-OpenACC/OpenMP 和 GPU 感知方法

GPU 感知 MPI 的概念使得 MPI 库能够直接访问 GPU 设备内存，而不必使用 CPU 主机内存作为中间缓冲区（例如，参见 [OpenMPI 文档](https://docs.open-mpi.org/en/v5.0.1/tuning-apps/networking/cuda.html)）。这提供了从一台 GPU 转移数据到另一台 GPU 而不涉及 CPU 主机内存的好处。

具体来说，在 GPU 感知方法中，设备指针指向分配在 GPU 内存空间中的数据（数据应存在于 GPU 设备中）。在这里，指针作为参数传递给一个由 GPU 内存支持的 MPI 例程。由于 MPI 例程可以直接访问 GPU 内存，它提供了在不需要将数据传输回主机的情况下，在 GPU 对之间进行通信的可能性。

在混合 MPI-OpenACC 模型中，该概念是通过将指令 host_data 与 use_device(list_array)子句结合起来定义的。这种组合使得可以从主机访问 use_device(list_array)子句中列出的数组（有关如何从主机访问数组的详细信息，请参阅[这里](https://www.openacc.org/sites/default/files/inline-images/Specification/OpenACC-3.2-final.pdf)）。已经存在于 GPU 设备内存中的数组列表直接传递给一个 MPI 例程，无需在主机内存中进行数据复制的中转。请注意，对于最初将数据复制到 GPU，我们使用由 enter data 和 exit data 指令定义的无结构数据块。无结构数据具有允许在数据区域内分配和释放数组的优势。

为了说明 GPU 感知 MPI 的概念，我们下面展示了两个示例，它们使用了 OpenACC 和 OpenMP API 中的点对点和集体操作。在第一个代码示例中，设备指针**f**被传递给 MPI 函数 MPI_Send()和 MPI_Recv()；在第二个示例中，指针**SumToT**被传递给 MPI 函数 MPI_Allreduce。在这里，MPI 操作 MPI_Send 和 MPI_Recv 以及 MPI_Allreduce 是在一对 GPU 之间执行，而不需要通过 CPU 主机内存。

示例：`GPU 感知：MPI_Send & MPI_Recv`

```
 !Device pointer f will be passed to MPI_send & MPI_recv
  !$acc host_data use_device(f)
  if(myid.lt.nproc-1)  then
 call MPI_Send(f(np:np),1,MPI_DOUBLE_PRECISION,myid+1,tag,MPI_COMM_WORLD,  ierr)
  endif

 if(myid.gt.0)  then
 call MPI_Recv(f(1),1,MPI_DOUBLE_PRECISION,myid-1,tag,MPI_COMM_WORLD,  status,ierr)
  endif
  !$acc end host_data 
```

```
 !Device pointer f will be passed to MPI_send & MPI_recv
  !$omp target data use_device_ptr(f)
  if(myid.lt.nproc-1)  then
 call MPI_Send(f(np:np),1,MPI_DOUBLE_PRECISION,myid+1,tag,MPI_COMM_WORLD,  ierr)
  endif

 if(myid.gt.0)  then
 call MPI_Recv(f(1),1,MPI_DOUBLE_PRECISION,myid-1,tag,MPI_COMM_WORLD,  status,ierr)
  endif
  !$omp end target data 
```

```
#pragma omp target data use_device_ptr(f)
  {
  if  (myid  <  nproc  -  1)  {
  MPI_Send(&f[np  -  1],  1,  MPI_DOUBLE,  myid  +  1,  tag,  MPI_COMM_WORLD);
  }

  if  (myid  >  0)  {
  MPI_Recv(&f[0],  1,  MPI_DOUBLE,  myid  -  1,  tag,  MPI_COMM_WORLD,
  MPI_STATUS_IGNORE);
  }
  } 
```

示例：`GPU 感知：MPI_Allreduce`

```
 !$acc data copy(SumToT)
  !$acc host_data use_device(SumToT)
  call MPI_ALLREDUCE(MPI_IN_PLACE,SumToT,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr  )
  !$acc end host_data
  !$acc end data 
```

```
 !$omp target enter data device(myDevice) map(to:SumToT)
  !$omp target data use_device_ptr(SumToT)
  call MPI_ALLREDUCE(MPI_IN_PLACE,SumToT,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr  )
  !$omp end target data
  !$omp target exit data map(from:SumToT) 
```

```
#pragma omp target enter data device(myDevice) map(to : SumToT)
  double  *SumToTPtr  =  &SumToT;
#pragma omp target data use_device_ptr(SumToTPtr)
  {
  MPI_Allreduce(MPI_IN_PLACE,  SumToTPtr,  1,  MPI_DOUBLE,  MPI_SUM,
  MPI_COMM_WORLD);
  }
#pragma omp target exit data map(from : SumToT) 
```

我们提供了一个代码示例，说明了在 OpenACC/OpenMP API 内实现 MPI 函数 MPI_Send()、MPI_Recv()和 MPI_Allreduce()的实现。这个实现专门设计来支持 GPU 感知 MPI 操作。

示例：`GPU 感知方法`

```
 call MPI_Scatter(f_send,np,MPI_DOUBLE_PRECISION,f,  np,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,  ierr)

  !offload f to GPUs
  !$acc enter data copyin(f)

  !Device pointer f will be passed to MPI_send & MPI_recv
  !$acc host_data use_device(f)
  if(myid.lt.nproc-1)  then
 call MPI_Send(f(np:np),1,MPI_DOUBLE_PRECISION,myid+1,tag,MPI_COMM_WORLD,  ierr)
  endif

 if(myid.gt.0)  then
 call MPI_Recv(f(1),1,MPI_DOUBLE_PRECISION,myid-1,tag,MPI_COMM_WORLD,  status,ierr)
  endif
  !$acc end host_data

  !do something .e.g.
  !$acc kernels
  f  =  f/2.
  !$acc end kernels

  SumToT=0d0
  !$acc parallel loop reduction(+:SumToT)
  do k=1,np
  SumToT  =  SumToT  +  f(k)
  enddo
  !$acc end parallel loop

  !SumToT is by default copied back to the CPU

  !$acc data copy(SumToT)
  !$acc host_data use_device(SumToT)
  call MPI_ALLREDUCE(MPI_IN_PLACE,SumToT,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr  )
  !$acc end host_data
  !$acc end data

  !$acc exit data delete(f) 
```

```
 call MPI_Scatter(f_send,np,MPI_DOUBLE_PRECISION,f,  np,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,  ierr)

  !offload f to GPUs
  !$omp target enter data device(myDevice) map(to:f)

  !Device pointer f will be passed to MPI_send & MPI_recv
  !$omp target data use_device_ptr(f)
  if(myid.lt.nproc-1)  then
 call MPI_Send(f(np:np),1,MPI_DOUBLE_PRECISION,myid+1,tag,MPI_COMM_WORLD,  ierr)
  endif

 if(myid.gt.0)  then
 call MPI_Recv(f(1),1,MPI_DOUBLE_PRECISION,myid-1,tag,MPI_COMM_WORLD,  status,ierr)
  endif
  !$omp end target data

  !do something .e.g.
  !$omp target teams distribute parallel do
  do k=1,np
  f(k)  =  f(k)/2.
  enddo
  !$omp end target teams distribute parallel do

  SumToT=0d0
  !$omp target teams distribute parallel do reduction(+:SumToT)
  do k=1,np
  SumToT  =  SumToT  +  f(k)
  enddo
  !$omp end target teams distribute parallel do 

  !SumToT is by default copied back to the CPU

  !$omp target enter data device(myDevice) map(to:SumToT)
  !$omp target data use_device_ptr(SumToT)
  call MPI_ALLREDUCE(MPI_IN_PLACE,SumToT,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr  )
  !$omp end target data
  !$omp target exit data map(from:SumToT)

  !$omp target exit data map(delete:f) 
```

```
 call  MPI_Scatter(f_send,np,MPI_DOUBLE_PRECISION,f,  np,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,  ierr)

  !offload  f  to  GPUs
  !$omp  target  enter  data  device(myDevice)  map(to:f)

  !Device  pointer  f  will  be  passed  to  MPI_send  &  MPI_recv
  !$omp  target  data  use_device_ptr(f)
  if(myid.lt.nproc-1)  then
  call  MPI_Send(f(np:np),1,MPI_DOUBLE_PRECISION,myid+1,tag,MPI_COMM_WORLD,  ierr)
  endif

  if(myid.gt.0)  then
  call  MPI_Recv(f(1),1,MPI_DOUBLE_PRECISION,myid-1,tag,MPI_COMM_WORLD,  status,ierr)
  endif
  !$omp  end  target  data

  !do  something  .e.g.
  !$omp  target  teams  distribute  parallel  do
  do  k=1,np
  f(k)  =  f(k)/2.
  enddo
  !$omp  end  target  teams  distribute  parallel  do

  SumToT=0d0
  !$omp  target  teams  distribute  parallel  do  reduction(+:SumToT)
  do  k=1,np
  SumToT  =  SumToT  +  f(k)
  enddo
  !$omp  end  target  teams  distribute  parallel  do  

  !SumToT  is  by  default  copied  back  to  the  CPU

  !$omp  target  enter  data  device(myDevice)  map(to:SumToT)
  !$omp  target  data  use_device_ptr(SumToT)
  call  MPI_ALLREDUCE(MPI_IN_PLACE,SumToT,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr  )
  !$omp  end  target  data
  !$omp  target  exit  data  map(from:SumToT)

  !$omp  target  exit  data  map(delete:f) 
```

带有 OpenACC/OpenMP API 的 GPU 感知 MPI 具有在单个节点内直接在 GPU 对之间进行通信的能力。然而，在多个节点之间执行 GPU 到 GPU 的通信需要 GPUDirect RDMA（远程直接内存访问）技术。这项技术可以通过减少延迟来进一步提高性能。

## 编译过程

下面描述了混合 MPI-OpenACC 和 MPI-OpenMP 卸载的编译过程。这个描述是为包装器 ftn 的 Cray 编译器提供的。在 LUMI-G 上，在编译之前可能需要以下模块（有关可用编程环境的更多详细信息，请参阅[LUMI 文档](https://docs.lumi-supercomputer.eu/development/compiling/prgenv/)）：

```
$ ml  LUMI/24.03
$ ml  PrgEnv-cray
$ ml  cray-mpich
$ ml  rocm
$ ml  craype-accel-amd-gfx90a 
```

示例：`编译过程`

```
$ ftn  -hacc  -o  mycode.mpiacc.exe  mycode_mpiacc.f90 
```

```
$ ftn  -homp  -o  mycode.mpiomp.exe  mycode_mpiomp.f90 
```

```
$ CC  -fopenmp  -fopenmp-targets=amdgcn-amd-amdhsa  -Xopenmp-target  -march=gfx90a  -o  mycode.mpiomp.exe  mycode_mpiomp.cpp 
```

在这里，标志 hacc 和 homp 分别启用混合 MPI-OpenACC 和 MPI-OpenMP 应用程序中的 OpenACC 和 OpenMP 指令。

**启用 GPU 感知支持**

要在 MPICH 库中启用 GPU 感知支持，需要在运行应用程序之前设置以下环境变量。

```
$ export MPICH_GPU_SUPPORT_ENABLED=1 
```

## 结论

总之，我们通过整合 GPU 指令模型，特别是 OpenACC 和 OpenMP API 与 MPI 库，概述了 GPU-混合编程。这里采用的方法不仅允许我们在单个节点内利用多个 GPU 设备，而且扩展到分布式节点。特别是，我们讨论了 GPU-aware MPI 方法，它具有允许 MPI 库与 GPU 设备内存直接交互的优势。换句话说，它允许在成对 GPU 之间执行 MPI 操作，从而减少由数据局部性引起的计算时间。

## 练习

我们考虑一个解决二维拉普拉斯方程的 MPI fortran 代码，该代码部分加速。练习的重点是使用 OpenACC 或 OpenMP API 通过以下步骤完成加速。

访问练习材料

下面的练习代码示例可以在本存储库的 content/examples/exercise_multipleGPU 子目录中访问。要访问它们，您需要克隆存储库：

```
$ git  clone  https://github.com/ENCCS/gpu-programming.git
$ cd  gpu-programming/content/examples/exercise_multipleGPU
$ ls 
```

练习 I：设置 GPU 设备

1.  实现启用将每个 MPI 进程分配到 GPU 设备的 OpenACC/OpenMP 函数。

1.1 在多个 GPU 上编译和运行代码。

练习 II：应用传统的 MPI-OpenACC/OpenMP

2.1 在调用 MPI 函数前后分别包含 OpenACC 指令 *update host()* 和 *update device()*。

注意

OpenACC 指令 *update host()* 用于在数据区域内将数据从 GPU 传输到 CPU；而指令 *update device()* 用于将数据从 CPU 传输到 GPU。

2.2 在调用 MPI 函数前后分别包含 OpenMP 指令 *update device() from()* 和 *update device() to()*。

注意

OpenMP 指令 *update device() from()* 用于在数据区域内将数据从 GPU 传输到 CPU；而指令 *update device() to()* 用于将数据从 CPU 传输到 GPU。

2.3 在多个 GPU 上编译和运行代码。

练习 III：实现 GPU-aware 支持

3.1 在调用 MPI 函数时包含 OpenACC 指令 *host_data use_device()* 以传递设备指针。

3.2 在调用 MPI 函数时包含 OpenMP 指令 *data use_device_ptr()* 以传递设备指针。

3.3 在多个 GPU 上编译和运行代码。

练习 IV：评估性能

1.  评估练习 II 和 III 中加速代码的执行时间，并将其与纯 MPI 实现进行比较。

## 参见

+   [GPU-aware MPI](https://documentation.sigma2.no/code_development/guides/gpuaware_mpi.html).

+   [MPI 文档](https://www.mpi-forum.org/docs/mpi-4.0/mpi40-report.pdf).

+   [OpenACC 规范](https://www.openacc.org/sites/default/files/inline-images/Specification/OpenACC-3.2-final.pdf).

+   [OpenMP 规范](https://www.openmp.org/wp-content/uploads/OpenMP-API-Specification-5-2.pdf).

+   [LUMI 文档](https://docs.lumi-supercomputer.eu/development/compiling/prgenv/).

+   [OpenACC 与 OpenMP 卸载比较](https://documentation.sigma2.no/code_development/guides/converting_acc2omp/openacc2openmp.html)。

+   [OpenACC 课程](https://github.com/HichamAgueny/GPU-course)。

## 简介

在分布式节点上探索多个 GPU（图形处理单元）具有充分利用大规模现代高性能计算（HPC）系统能力的潜力。这里，加速分布式系统计算的一种方法是将 MPI（消息传递接口）与 OpenACC 和 OpenMP 应用程序编程接口（API）等 GPU 编程模型相结合。这种组合既受到这些 API 简单性的驱动，也受到 MPI 广泛使用的推动。

在本指南中，我们为熟悉 MPI 的读者提供了关于实现混合模型（其中 MPI 通信框架与 OpenACC 或 OpenMP API 相结合）的见解。特别关注从 OpenACC 和 OpenMP API 执行点对点操作（例如 MPI_Send 和 MPI_Recv）和集体操作（例如 MPI_Allreduce）。在此，我们讨论了两种场景：（i）在 CPU 主机上执行 MPI 操作后，将卸载到 GPU 设备；（ii）在两个 GPU 之间执行 MPI 操作，而不涉及 CPU 主机内存。后一种场景被称为 GPU 感知 MPI，其优点是减少由于通过主机内存进行异构通信而引起的数据传输时间，从而使得 HPC 应用程序更高效。

本指南的组织结构如下：我们首先介绍如何将每个 MPI 进程分配到同一节点内的 GPU 设备。我们考虑了一种情况，即主机和设备具有不同的内存。随后，我们将介绍带有和不带有 GPU 感知 MPI 的混合 MPI-OpenACC/OpenMP 卸载。最后，提供了一些练习，以帮助理解这些概念。

## 将 MPI-ranks 分配到 GPU-devices

将 MPI 应用程序加速以利用分布式节点上的多个 GPU，首先需要将每个 MPI 进程分配到 GPU 设备，以确保两个 MPI 进程不使用同一 GPU 设备。这是防止应用程序发生潜在崩溃的必要步骤。这是因为 GPU 被设计来处理多个线程任务，而不是多个 MPI 进程。

确保两个 MPI 进程不使用同一 GPU 的一种方法，是确定哪些 MPI 进程运行在同一个节点上，这样每个进程都可以被分配到同一节点内的一个 GPU 设备。这可以通过使用 MPI_COMM_SPLIT_TYPE()例程将世界通信器拆分为通信子组（或子通信器）来实现。

```
 ! Split the world communicator into subgroups of commu, each of which
  ! contains processes that run on the same node, and which can create a
  ! shared memory region (via the type MPI_COMM_TYPE_SHARED).
  ! The call returns a new communicator "host_comm", which is created by
  ! each subgroup. 
  call MPI_COMM_SPLIT_TYPE(MPI_COMM_WORLD,  MPI_COMM_TYPE_SHARED,  0,&
  MPI_INFO_NULL,  host_comm,ierr)
  call MPI_COMM_RANK(host_comm,  host_rank,ierr) 
```

```
 // Split the world communicator into subgroups.
  MPI_Comm  host_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD,  MPI_COMM_TYPE_SHARED,  0,  MPI_INFO_NULL,
  &host_comm);
  int  host_rank;
  MPI_Comm_rank(host_comm,  &host_rank); 
```

在这里，每个子通信器的尺寸对应于每个节点上的 GPU 数量（这也就是每个节点上的任务数量），每个子通信器包含一个由排名指示的进程列表。这些进程由 MPI_COMM_TYPE_SHARED 参数定义的共享内存区域共享（有关更多详细信息，请参阅 [MPI 报告](https://www.mpi-forum.org/docs/mpi-4.0/mpi40-report.pdf)）。调用 MPI_COMM_SPLIT_TYPE() 例程返回一个在代码中标记为 *“host_comm”* 的子通信器，其中 MPI-ranks 从 0 排序到每个节点上的进程数减 1。这些 MPI 排名随后被分配到同一节点内的不同 GPU 设备。该过程根据所实现的指令基于模型进行。检索到的 MPI 排名随后存储在变量 **myDevice** 中。该变量按照代码下面的指示传递给 OpenACC 或 OpenMP 例程。

示例：`分配设备`

```
 myDevice  =  host_rank

  ! Sets the device number and the device type to be used
  call acc_set_device_num(myDevice,  acc_get_device_type())

  ! Returns the number of devices available on the host
  numDevice  =  acc_get_num_devices(acc_get_device_type()) 
```

```
 myDevice  =  host_rank

  ! Sets the device number to use in device constructs by setting the initial value of the default-device-var 
  call omp_set_default_device(myDevice)

  ! Returns the number of devices available for offloading.
  numDevice  =  omp_get_num_devices() 
```

```
 int  myDevice  =  host_rank;
  // Set the device number to use in device constructs.
  omp_set_default_device(myDevice);

  // Return the number of devices available for offloading.
  int  numDevice  =  omp_get_num_devices(); 
```

另一个用于检索特定设备编号的有用函数，例如，将数据映射到特定设备的有用函数是

```
acc_get_device_num() 
```

```
omp_get_device_num() 
```

将 MPI 排名分配给 GPU 设备的语法总结如下

示例：`设置设备`

```
 ! Initialise MPI communication 
  call MPI_Init(ierr)
  ! Identify the ID rank (process) 
  call MPI_COMM_RANK(  MPI_COMM_WORLD,  myid,  ierr  )
  ! Get number of active processes (from 0 to nproc-1) 
  call MPI_COMM_SIZE(  MPI_COMM_WORLD,  nproc,  ierr  )

  ! Split the world communicator into subgroups of commu, each of which
  ! contains processes that run on the same node, and which can create a
  ! shared memory region (via the type MPI_COMM_TYPE_SHARED).
  ! The call returns a new communicator "host_comm", which is created by
  ! each subgroup. 
  call MPI_COMM_SPLIT_TYPE(MPI_COMM_WORLD,  MPI_COMM_TYPE_SHARED,  0,&
  MPI_INFO_NULL,  host_comm,ierr)
  call MPI_COMM_RANK(host_comm,  host_rank,ierr)

  ! Gets the node name
  call MPI_GET_PROCESSOR_NAME(name,  resulten,  ierror)

  myDevice  =  host_rank

  ! Sets the device number and the device type to be used
  call acc_set_device_num(myDevice,  acc_get_device_type())

  ! Returns the number of devices available on the host
  numDevice  =  acc_get_num_devices(acc_get_device_type()) 
```

```
 ! Initialise MPI communication 
  call MPI_Init(ierr)
  ! Identify the ID rank (process) 
  call MPI_COMM_RANK(  MPI_COMM_WORLD,  myid,  ierr  )
  ! Get number of active processes (from 0 to nproc-1) 
  call MPI_COMM_SIZE(  MPI_COMM_WORLD,  nproc,  ierr  )

  ! Split the world communicator into subgroups of commu, each of which
  ! contains processes that run on the same node, and which can create a
  ! shared memory region (via the type MPI_COMM_TYPE_SHARED).
  ! The call returns a new communicator "host_comm", which is created by
  ! each subgroup. 
  call MPI_COMM_SPLIT_TYPE(MPI_COMM_WORLD,  MPI_COMM_TYPE_SHARED,  0,&
  MPI_INFO_NULL,  host_comm,ierr)
  call MPI_COMM_RANK(host_comm,  host_rank,ierr)

  ! Gets the node name
  call MPI_GET_PROCESSOR_NAME(name,  resulten,  ierror)

  myDevice  =  host_rank

  ! Sets the device number to use in device constructs by setting the initial value of the default-device-var 
  call omp_set_default_device(myDevice)

  ! Returns the number of devices available for offloading.
  numDevice  =  omp_get_num_devices() 
```

```
 // Initialize MPI communication.
  MPI_Init(&argc,  &argv);

  // Identify the ID rank (process).
  int  myid,  nproc;
  MPI_Comm_rank(MPI_COMM_WORLD,  &myid);
  // Get number of active processes.
  MPI_Comm_size(MPI_COMM_WORLD,  &nproc);

  // Split the world communicator into subgroups.
  MPI_Comm  host_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD,  MPI_COMM_TYPE_SHARED,  0,  MPI_INFO_NULL,
  &host_comm);
  int  host_rank;
  MPI_Comm_rank(host_comm,  &host_rank);

  // Get the node name.
  int  name_len;
  char  processor_name[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(processor_name,  &name_len);

  int  myDevice  =  host_rank;
  // Set the device number to use in device constructs.
  omp_set_default_device(myDevice);

  // Return the number of devices available for offloading.
  int  numDevice  =  omp_get_num_devices(); 
```

## 混合 MPI-OpenACC/OpenMP 无 GPU 感知方法

在介绍如何将每个 MPI 排名分配给 GPU 设备之后，我们现在讨论将 MPI 与 OpenACC 或 OpenMP 脱载相结合的概念。在这种方法中，从 OpenACC 或 OpenMP API 调用 MPI 例程需要在 MPI 调用前后更新 CPU 主机中的数据。在这种情况下，数据在每次 MPI 调用前后在主机和设备之间来回复制。在混合 MPI-OpenACC 模型中，该过程通过指定在 MPI 调用之前将数据从设备复制到主机的指令 update host() 来定义；通过在 MPI 调用之后指定的指令 update device() 来将数据复制回设备。在混合 MPI-OpenMP 中类似地，这里可以通过指定 OpenMP 指令 update device() from() 和 update device() to() 来更新主机中的数据，分别用于将数据从设备复制到主机和从主机复制回设备。

为了说明混合 MPI-OpenACC/OpenMP 的概念，我们下面展示了一个涉及 MPI 函数 MPI_Send() 和 MPI_Recv() 的实现示例。

示例：`更新主机/设备指令`

```
 !offload f to GPUs
  !$acc enter data copyin(f)

  !update f: copy f from GPU to CPU
  !$acc update host(f)

  if(myid.lt.nproc-1)  then
 call MPI_Send(f(np:np),1,MPI_DOUBLE_PRECISION,myid+1,tag,MPI_COMM_WORLD,  ierr)
  endif

 if(myid.gt.0)  then
 call MPI_Recv(f(1),1,MPI_DOUBLE_PRECISION,myid-1,tag,MPI_COMM_WORLD,  status,ierr)
  endif

  !update f: copy f from CPU to GPU
  !$acc update device(f) 
```

```
 !offload f to GPUs
  !$omp target enter data device(myDevice) map(to:f)

  !update f: copy f from GPU to CPU
  !$omp target update device(myDevice) from(f)

  if(myid.lt.nproc-1)  then
 call MPI_Send(f(np:np),1,MPI_DOUBLE_PRECISION,myid+1,tag,MPI_COMM_WORLD,  ierr)
  endif

 if(myid.gt.0)  then
 call MPI_Recv(f(1),1,MPI_DOUBLE_PRECISION,myid-1,tag,MPI_COMM_WORLD,  status,ierr)
  endif

  !update f: copy f from CPU to GPU
  !$omp target update device(myDevice) to(f) 
```

```
// Offload f to GPUs
#pragma omp target enter data map(to : f[0 : np]) device(myDevice)

  // update f: copy f from GPU to CPU
#pragma omp target update device(myDevice) from(f)

  if  (myid  <  nproc  -  1)  {
  MPI_Send(&f[np  -  1],  1,  MPI_DOUBLE,  myid  +  1,  tag,  MPI_COMM_WORLD);
  }
  if  (myid  >  0)  {
  MPI_Recv(&f[0],  1,  MPI_DOUBLE,  myid  -  1,  tag,  MPI_COMM_WORLD,
  MPI_STATUS_IGNORE);
  }

// update f: copy f from CPU to GPU
#pragma omp target update device(myDevice) to(f) 
```

在这里，我们提供了一个结合 MPI 与 OpenACC/OpenMP API 的代码示例。

示例：`更新主机/设备指令`

```
 call MPI_Scatter(f_send,np,MPI_DOUBLE_PRECISION,f,  np,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,  ierr)

  !offload f to GPUs
  !$acc enter data copyin(f)

  !update f: copy f from GPU to CPU
  !$acc update host(f)

  if(myid.lt.nproc-1)  then
 call MPI_Send(f(np:np),1,MPI_DOUBLE_PRECISION,myid+1,tag,MPI_COMM_WORLD,  ierr)
  endif

 if(myid.gt.0)  then
 call MPI_Recv(f(1),1,MPI_DOUBLE_PRECISION,myid-1,tag,MPI_COMM_WORLD,  status,ierr)
  endif

  !update f: copy f from CPU to GPU
  !$acc update device(f)

  !do something .e.g.
  !$acc kernels
  f  =  f/2.
  !$acc end kernels

  SumToT=0d0
  !$acc parallel loop reduction(+:SumToT)
  do k=1,np
  SumToT  =  SumToT  +  f(k)
  enddo
  !$acc end parallel loop

  !SumToT is by default copied back to the CPU
  call MPI_ALLREDUCE(MPI_IN_PLACE,SumToT,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr  )

  !$acc exit data delete(f) 
```

```
 call MPI_Scatter(f_send,np,MPI_DOUBLE_PRECISION,f,  np,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,  ierr)

  !offload f to GPUs
  !$omp target enter data device(myDevice) map(to:f)

  !update f: copy f from GPU to CPU
  !$omp target update device(myDevice) from(f)

  if(myid.lt.nproc-1)  then
 call MPI_Send(f(np:np),1,MPI_DOUBLE_PRECISION,myid+1,tag,MPI_COMM_WORLD,  ierr)
  endif

 if(myid.gt.0)  then
 call MPI_Recv(f(1),1,MPI_DOUBLE_PRECISION,myid-1,tag,MPI_COMM_WORLD,  status,ierr)
  endif

  !update f: copy f from CPU to GPU
  !$omp target update device(myDevice) to(f)

  !do something .e.g.
  !$omp target teams distribute parallel do
  do k=1,np
  f(k)  =  f(k)/2.
  enddo
  !$omp end target teams distribute parallel do

  SumToT=0d0
  !$omp target teams distribute parallel do reduction(+:SumToT)
  do k=1,np
  SumToT  =  SumToT  +  f(k)
  enddo
  !$omp end target teams distribute parallel do 

  !SumToT is by default copied back to the CPU
  call MPI_ALLREDUCE(MPI_IN_PLACE,SumToT,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr  )

  !$omp target exit data map(delete:f) 
```

```
 MPI_Scatter(f_send.data(),  np,  MPI_DOUBLE,  f,  np,  MPI_DOUBLE,  0,
  MPI_COMM_WORLD);

// Offload f to GPUs
#pragma omp target enter data map(to : f[0 : np]) device(myDevice)

  // update f: copy f from GPU to CPU
#pragma omp target update device(myDevice) from(f)

  if  (myid  <  nproc  -  1)  {
  MPI_Send(&f[np  -  1],  1,  MPI_DOUBLE,  myid  +  1,  tag,  MPI_COMM_WORLD);
  }
  if  (myid  >  0)  {
  MPI_Recv(&f[0],  1,  MPI_DOUBLE,  myid  -  1,  tag,  MPI_COMM_WORLD,
  MPI_STATUS_IGNORE);
  }

// update f: copy f from CPU to GPU
#pragma omp target update device(myDevice) to(f)

// Update, operate, and offload data back to GPUs
#pragma omp target teams distribute parallel for device(myDevice)
  for  (int  k  =  0;  k  <  np;  ++k)  {
  f[k]  /=  2.0;
  }

  double  SumToT  =  0.0;

#pragma omp target teams distribute parallel for reduction(+ : SumToT)         \
 device(myDevice)
  for  (int  k  =  0;  k  <  np;  ++k)  {
  SumToT  +=  f[k];
  }

  MPI_Allreduce(MPI_IN_PLACE,  &SumToT,  1,  MPI_DOUBLE,  MPI_SUM,  MPI_COMM_WORLD);

#pragma omp target exit data map(delete : f[0 : np]) 
```

尽管混合 MPI-OpenACC/OpenMP 脱载的实现很简单，但它由于在调用 MPI 例程前后显式地在主机和设备之间传输数据而遭受低性能。这构成了 GPU 编程的瓶颈。为了提高数据传输期间主机阶段影响到的性能，可以实施如以下章节所述的 GPU 感知 MPI 方法。

## 混合 MPI-OpenACC/OpenMP 带 GPU 感知方法

GPU 意识 MPI 的概念使得 MPI 库能够直接访问 GPU 设备内存，而无需必要地使用 CPU-主机内存作为中间缓冲区（例如，参见[OpenMPI 文档](https://docs.open-mpi.org/en/v5.0.1/tuning-apps/networking/cuda.html)）。这提供了从一台 GPU 向另一台 GPU 转移数据而不涉及 CPU-主机内存的好处。

具体来说，在具有 GPU 意识的方法中，设备指针指向分配在 GPU 内存空间中的数据（数据应存在于 GPU 设备中）。在这里，指针作为参数传递给一个由 GPU 内存支持的 MPI 例程。由于 MPI 例程可以直接访问 GPU 内存，它提供了在成对 GPU 之间进行通信的可能性，而无需将数据传回主机。

在混合 MPI-OpenACC 模型中，该概念是通过结合指令 host_data 和子句 use_device(list_array) 来定义的。这种组合使得可以从主机访问子句 use_device(list_array) 中列出的数组（参见[这里](https://www.openacc.org/sites/default/files/inline-images/Specification/OpenACC-3.2-final.pdf)）。已经存在于 GPU 设备内存中的数组列表可以直接传递给一个 MPI 例程，而不需要为复制数据而设置一个中间的主机内存。请注意，对于最初将数据复制到 GPU，我们使用由指令 enter data 和 exit data 定义的未结构化数据块。未结构化数据具有允许在数据区域内分配和释放数组的优势。

为了说明 GPU 意识 MPI 的概念，我们下面展示了两个示例，它们利用了 OpenACC 和 OpenMP API 中的点对点和集体操作。在第一个代码示例中，设备指针 **f** 被传递给 MPI 函数 MPI_Send() 和 MPI_Recv()；在第二个示例中，指针 **SumToT** 被传递给 MPI 函数 MPI_Allreduce。在这里，MPI 操作 MPI_Send 和 MPI_Recv 以及 MPI_Allreduce 是在成对 GPU 之间执行的，而不需要通过 CPU-主机内存。

示例：`GPU 意识：MPI_Send & MPI_Recv`

```
 !Device pointer f will be passed to MPI_send & MPI_recv
  !$acc host_data use_device(f)
  if(myid.lt.nproc-1)  then
 call MPI_Send(f(np:np),1,MPI_DOUBLE_PRECISION,myid+1,tag,MPI_COMM_WORLD,  ierr)
  endif

 if(myid.gt.0)  then
 call MPI_Recv(f(1),1,MPI_DOUBLE_PRECISION,myid-1,tag,MPI_COMM_WORLD,  status,ierr)
  endif
  !$acc end host_data 
```

```
 !Device pointer f will be passed to MPI_send & MPI_recv
  !$omp target data use_device_ptr(f)
  if(myid.lt.nproc-1)  then
 call MPI_Send(f(np:np),1,MPI_DOUBLE_PRECISION,myid+1,tag,MPI_COMM_WORLD,  ierr)
  endif

 if(myid.gt.0)  then
 call MPI_Recv(f(1),1,MPI_DOUBLE_PRECISION,myid-1,tag,MPI_COMM_WORLD,  status,ierr)
  endif
  !$omp end target data 
```

```
#pragma omp target data use_device_ptr(f)
  {
  if  (myid  <  nproc  -  1)  {
  MPI_Send(&f[np  -  1],  1,  MPI_DOUBLE,  myid  +  1,  tag,  MPI_COMM_WORLD);
  }

  if  (myid  >  0)  {
  MPI_Recv(&f[0],  1,  MPI_DOUBLE,  myid  -  1,  tag,  MPI_COMM_WORLD,
  MPI_STATUS_IGNORE);
  }
  } 
```

示例：`GPU 意识：MPI_Allreduce`

```
 !$acc data copy(SumToT)
  !$acc host_data use_device(SumToT)
  call MPI_ALLREDUCE(MPI_IN_PLACE,SumToT,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr  )
  !$acc end host_data
  !$acc end data 
```

```
 !$omp target enter data device(myDevice) map(to:SumToT)
  !$omp target data use_device_ptr(SumToT)
  call MPI_ALLREDUCE(MPI_IN_PLACE,SumToT,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr  )
  !$omp end target data
  !$omp target exit data map(from:SumToT) 
```

```
#pragma omp target enter data device(myDevice) map(to : SumToT)
  double  *SumToTPtr  =  &SumToT;
#pragma omp target data use_device_ptr(SumToTPtr)
  {
  MPI_Allreduce(MPI_IN_PLACE,  SumToTPtr,  1,  MPI_DOUBLE,  MPI_SUM,
  MPI_COMM_WORLD);
  }
#pragma omp target exit data map(from : SumToT) 
```

我们下面提供了一个代码示例，展示了如何在 OpenACC/OpenMP API 中实现 MPI 函数 MPI_Send()、MPI_Recv() 和 MPI_Allreduce()。这个实现专门设计用来支持具有 GPU 意识的 MPI 操作。

示例：`GPU 意识方法`

```
 call MPI_Scatter(f_send,np,MPI_DOUBLE_PRECISION,f,  np,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,  ierr)

  !offload f to GPUs
  !$acc enter data copyin(f)

  !Device pointer f will be passed to MPI_send & MPI_recv
  !$acc host_data use_device(f)
  if(myid.lt.nproc-1)  then
 call MPI_Send(f(np:np),1,MPI_DOUBLE_PRECISION,myid+1,tag,MPI_COMM_WORLD,  ierr)
  endif

 if(myid.gt.0)  then
 call MPI_Recv(f(1),1,MPI_DOUBLE_PRECISION,myid-1,tag,MPI_COMM_WORLD,  status,ierr)
  endif
  !$acc end host_data

  !do something .e.g.
  !$acc kernels
  f  =  f/2.
  !$acc end kernels

  SumToT=0d0
  !$acc parallel loop reduction(+:SumToT)
  do k=1,np
  SumToT  =  SumToT  +  f(k)
  enddo
  !$acc end parallel loop

  !SumToT is by default copied back to the CPU

  !$acc data copy(SumToT)
  !$acc host_data use_device(SumToT)
  call MPI_ALLREDUCE(MPI_IN_PLACE,SumToT,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr  )
  !$acc end host_data
  !$acc end data

  !$acc exit data delete(f) 
```

```
 call MPI_Scatter(f_send,np,MPI_DOUBLE_PRECISION,f,  np,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,  ierr)

  !offload f to GPUs
  !$omp target enter data device(myDevice) map(to:f)

  !Device pointer f will be passed to MPI_send & MPI_recv
  !$omp target data use_device_ptr(f)
  if(myid.lt.nproc-1)  then
 call MPI_Send(f(np:np),1,MPI_DOUBLE_PRECISION,myid+1,tag,MPI_COMM_WORLD,  ierr)
  endif

 if(myid.gt.0)  then
 call MPI_Recv(f(1),1,MPI_DOUBLE_PRECISION,myid-1,tag,MPI_COMM_WORLD,  status,ierr)
  endif
  !$omp end target data

  !do something .e.g.
  !$omp target teams distribute parallel do
  do k=1,np
  f(k)  =  f(k)/2.
  enddo
  !$omp end target teams distribute parallel do

  SumToT=0d0
  !$omp target teams distribute parallel do reduction(+:SumToT)
  do k=1,np
  SumToT  =  SumToT  +  f(k)
  enddo
  !$omp end target teams distribute parallel do 

  !SumToT is by default copied back to the CPU

  !$omp target enter data device(myDevice) map(to:SumToT)
  !$omp target data use_device_ptr(SumToT)
  call MPI_ALLREDUCE(MPI_IN_PLACE,SumToT,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr  )
  !$omp end target data
  !$omp target exit data map(from:SumToT)

  !$omp target exit data map(delete:f) 
```

```
 call  MPI_Scatter(f_send,np,MPI_DOUBLE_PRECISION,f,  np,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,  ierr)

  !offload  f  to  GPUs
  !$omp  target  enter  data  device(myDevice)  map(to:f)

  !Device  pointer  f  will  be  passed  to  MPI_send  &  MPI_recv
  !$omp  target  data  use_device_ptr(f)
  if(myid.lt.nproc-1)  then
  call  MPI_Send(f(np:np),1,MPI_DOUBLE_PRECISION,myid+1,tag,MPI_COMM_WORLD,  ierr)
  endif

  if(myid.gt.0)  then
  call  MPI_Recv(f(1),1,MPI_DOUBLE_PRECISION,myid-1,tag,MPI_COMM_WORLD,  status,ierr)
  endif
  !$omp  end  target  data

  !do  something  .e.g.
  !$omp  target  teams  distribute  parallel  do
  do  k=1,np
  f(k)  =  f(k)/2.
  enddo
  !$omp  end  target  teams  distribute  parallel  do

  SumToT=0d0
  !$omp  target  teams  distribute  parallel  do  reduction(+:SumToT)
  do  k=1,np
  SumToT  =  SumToT  +  f(k)
  enddo
  !$omp  end  target  teams  distribute  parallel  do  

  !SumToT  is  by  default  copied  back  to  the  CPU

  !$omp  target  enter  data  device(myDevice)  map(to:SumToT)
  !$omp  target  data  use_device_ptr(SumToT)
  call  MPI_ALLREDUCE(MPI_IN_PLACE,SumToT,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr  )
  !$omp  end  target  data
  !$omp  target  exit  data  map(from:SumToT)

  !$omp  target  exit  data  map(delete:f) 
```

使用 OpenACC/OpenMP API 的 GPU 意识 MPI 具有在单个节点内成对 GPU 之间直接通信的能力。然而，在多个节点之间执行 GPU 到 GPU 的通信需要 GPUDirect RDMA（远程直接内存访问）技术。这项技术可以通过减少延迟来进一步提高性能。

## 编译过程

下面描述了混合 MPI-OpenACC 和 MPI-OpenMP 卸载的编译过程。此描述适用于包装器 ftn 的 Cray 编译器。在 LUMI-G 上，在编译之前可能需要以下模块（有关可用编程环境的详细信息，请参阅[LUMI 文档](https://docs.lumi-supercomputer.eu/development/compiling/prgenv/)）：

```
$ ml  LUMI/24.03
$ ml  PrgEnv-cray
$ ml  cray-mpich
$ ml  rocm
$ ml  craype-accel-amd-gfx90a 
```

示例：`编译过程`

```
$ ftn  -hacc  -o  mycode.mpiacc.exe  mycode_mpiacc.f90 
```

```
$ ftn  -homp  -o  mycode.mpiomp.exe  mycode_mpiomp.f90 
```

```
$ CC  -fopenmp  -fopenmp-targets=amdgcn-amd-amdhsa  -Xopenmp-target  -march=gfx90a  -o  mycode.mpiomp.exe  mycode_mpiomp.cpp 
```

在这里，标志 hacc 和 homp 分别启用混合 MPI-OpenACC 和 MPI-OpenMP 应用程序中的 OpenACC 和 OpenMP 指令。

**启用 GPU 感知支持**

要在 MPICH 库中启用 GPU 感知支持，需要在运行应用程序之前设置以下环境变量。

```
$ export MPICH_GPU_SUPPORT_ENABLED=1 
```

## 结论

总之，我们通过整合 GPU 指令模型，特别是 OpenACC 和 OpenMP API，与 MPI 库相结合，概述了 GPU 混合编程。这里采用的方法不仅允许我们在单个节点内利用多个 GPU 设备，而且扩展到分布式节点。特别是，我们解决了 GPU 感知 MPI 方法，该方法具有允许 MPI 库与 GPU 设备内存直接交互的优势。换句话说，它允许在 GPU 对之间执行 MPI 操作，从而减少由数据局部性引起的计算时间。

## 练习

我们考虑一个解决二维拉普拉斯方程的 MPI Fortran 代码，该代码部分加速。练习的重点是使用 OpenACC 或 OpenMP API 通过以下步骤完成加速。

访问练习材料

下面的练习代码示例可以在本存储库的 content/examples/exercise_multipleGPU 子目录中找到。要访问它们，您需要克隆存储库：

```
$ git  clone  https://github.com/ENCCS/gpu-programming.git
$ cd  gpu-programming/content/examples/exercise_multipleGPU
$ ls 
```

练习 I：设置 GPU 设备

1.  实现 OpenACC/OpenMP 函数，使每个 MPI 进程能够分配到一个 GPU 设备。

1.1 在多个 GPU 上编译和运行代码。

练习 II：应用传统的 MPI-OpenACC/OpenMP

2.1 在调用 MPI 函数前后分别包含 OpenACC 指令**update host()**和**update device()**。

注意

OpenACC 指令**update host()**用于在数据区域内将数据从 GPU 传输到 CPU；而指令**update device()**用于将数据从 CPU 传输到 GPU。

2.2 在调用 MPI 函数前后分别包含 OpenMP 指令**update device() from()**和**update device() to()**。

注意

OpenMP 指令**update device() from()**用于在数据区域内将数据从 GPU 传输到 CPU；而指令**update device() to()**用于将数据从 CPU 传输到 GPU。

2.3 在多个 GPU 上编译和运行代码。

练习 III：实现 GPU 感知支持

3.1 在调用 MPI 函数时包含 OpenACC 指令**host_data use_device()**，以传递设备指针到 MPI 函数。

3.2 在调用 MPI 函数时包含 OpenMP 指令**data use_device_ptr()**，以传递设备指针到 MPI 函数。

3.3 在多个 GPU 上编译和运行代码。

练习 IV：评估性能

1.  评估练习 II 和 III 中加速代码的执行时间，并将其与纯 MPI 实现进行比较。

## 参见

+   [GPU 感知 MPI](https://documentation.sigma2.no/code_development/guides/gpuaware_mpi.html).

+   [MPI 文档](https://www.mpi-forum.org/docs/mpi-4.0/mpi40-report.pdf).

+   [OpenACC 规范](https://www.openacc.org/sites/default/files/inline-images/Specification/OpenACC-3.2-final.pdf).

+   [OpenMP 规范](https://www.openmp.org/wp-content/uploads/OpenMP-API-Specification-5-2.pdf).

+   [LUMI 文档](https://docs.lumi-supercomputer.eu/development/compiling/prgenv/).

+   [OpenACC 与 OpenMP 转移](https://documentation.sigma2.no/code_development/guides/converting_acc2omp/openacc2openmp.html).

+   [OpenACC 课程](https://github.com/HichamAgueny/GPU-course)*

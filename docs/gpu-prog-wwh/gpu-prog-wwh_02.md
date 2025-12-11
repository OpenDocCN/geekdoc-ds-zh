# 设置

> 原文：[`enccs.github.io/gpu-programming/0-setup/`](https://enccs.github.io/gpu-programming/0-setup/)

*GPU 编程：为什么、何时以及如何？* **   设置

+   [在 GitHub 上编辑](https://github.com/ENCCS/gpu-programming/blob/main/content/0-setup.rst)

* * *

## 本地安装

由于本课程使用 HPC 集群进行教学，因此您自己的计算机无需安装任何软件。

## 在 LUMI 上运行

交互式作业，1 个节点，1 个 GPU，1 小时：

```
$ salloc  -A  project_465002387  -N  1  -t  1:00:00  -p  standard-g  --gpus-per-node=1
$ srun  <some-command> 
```

使用`exit`退出交互式分配。

计算节点上的交互式终端会话：

```
$ srun  --account=project_465002387  --partition=standard-g  --nodes=1  --cpus-per-task=1  --ntasks-per-node=1  --gpus-per-node=1  --time=1:00:00  --pty  bash
$ <some-command> 
```

对应的批处理脚本`submit.sh`：

```
#!/bin/bash -l
#SBATCH --account=project_465002387
#SBATCH --job-name=example-job
#SBATCH --output=examplejob.o%j
#SBATCH --error=examplejob.e%j
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00

srun  <some_command> 
```

+   提交作业：`sbatch submit.sh`

+   监控您的作业：`squeue --me`

+   终止作业：`scancel <JOB_ID>`

### 在 LUMI 上运行 Julia

为了在 LUMI 上使用`AMDGPU.jl`运行 Julia，我们使用以下目录结构，并假设它是我们的工作目录。

```
.
├── Project.toml  # Julia environment
├── script.jl     # Julia script
└── submit.sh     # Slurm batch script 
```

以下是一个`Project.toml`项目文件的示例。

```
[deps]
AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e" 
```

对于`submit.sh`批处理脚本，向上述批处理脚本中添加额外内容。

```
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1750

module  use  /appl/local/csc/modulefiles

module  load  julia
module  load  julia-amdgpu

julia  --project=.  -e  'using Pkg; Pkg.instantiate()'
julia  --project=.  script.jl 
```

下面提供了一个`script.jl`代码示例。

```
using  AMDGPU

A  =  rand(2⁹,  2⁹)
A_d  =  ROCArray(A)
B_d  =  A_d  *  A_d

println("----EOF----") 
```

## 运行 Python

### 在 LUMI 上

已创建包含所有必要依赖项的 singularity 容器。要启动容器及其内部的`IPython`解释器，请按照以下操作：

```
$ salloc  -p  small-g  -A  project_465002387  -t  1:00:00  -N  1  --gpus=1
$ srun  --pty  \
  singularity  exec  --no-home  \
  -B  $PWD:/work  \
  /scratch/project_465002387/containers/gpu-programming/python-from-docker/container.sif  \
  bash

Singularity> cd /work
Singularity> . /.venv/bin/activate
Singularity> python  # or ipython 
```

创建容器的配方

作为参考，以下文件被用来创建上述 Singularity 容器。首先是一个 singularity def 文件，

```
Bootstrap:  docker
From:  rocm/dev-ubuntu-24.04:6.4.4-complete

%environment
  CUPY_INSTALL_USE_HIP=1
  ROCM_HOME=/opt/rocm
  HCC_AMDGPU_TARGET=gfx90a
  LLVM_PATH=/opt/rocm/llvm

%post
  export  CUPY_INSTALL_USE_HIP=1
  export  ROCM_HOME=/opt/rocm
  export  HCC_AMDGPU_TARGET=gfx90a
  export  LLVM_PATH=/opt/rocm/llvm
  export  PATH="$HOME/.local/bin/:$PATH"

  apt-get  update  &&  apt-get  install  -y  --no-install-recommends  curl  ca-certificates  git
  curl  -L  https://astral.sh/uv/install.sh  -o  /uv-installer.sh
  sh  /uv-installer.sh  &&  rm  /uv-installer.sh

  .  $HOME/.local/bin/env

  uv  python  install  3.12
  uv  venv  -p  3.12  --seed
  uv  pip  install  --index-strategy  unsafe-best-match  -r  /tmp/envs/requirements.txt
  uv  pip  freeze  >>  /tmp/envs/requirements_pinned.txt

  touch  /usr/lib64/libjansson.so.4  /usr/lib64/libcxi.so.1  /usr/lib64/libjansson.so.4
  mkdir  /var/spool/slurmd  /opt/cray
  mkdir  /scratch  /projappl  /project  /flash  /appl 
```

以及一个用于构建容器的 bash 脚本，

```
#!/bin/sh
ml  purge
ml  LUMI/24.03  partition/G
ml  load  systools/24.03  # For proot

export  SINGULARITY_CACHEDIR="$PWD/singularity/cache"
export  SINGULARITY_TMPDIR="$FLASH/$USER/singularity/tmp"
singularity  build  -B  "$PWD":/tmp/envs  --fix-perms  container.sif  build_singularity.def 
```

小贴士

您还可以通过以下方式交互式构建：

```
singularity build --sandbox <other flags> container-sandbox build_singularity.def
singularity shell --writable container-sandbox 
```

最后是一个`requirements.txt`文件：

```
jupyterlab
jupyterlab-git
nbclassic
matplotlib
numpy
cupy
# jax[rocm]
jax-rocm7-pjrt @ https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jax_rocm7_pjrt-0.6.0-py3-none-manylinux_2_28_x86_64.whl
jax-rocm7-plugin @ https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jax_rocm7_plugin-0.6.0-cp312-cp312-manylinux_2_28_x86_64.whl
jaxlib @ https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jaxlib-0.6.2-cp312-cp312-manylinux2014_x86_64.whl
jax==0.6.2
--extra-index-url https://test.pypi.org/simple
numba-hip[rocm-6-4-4] @ git+https://github.com/ROCm/numba-hip.git 
```

LUMI 也有官方的 Jax singularity 镜像。这些可以在以下路径下找到：

```
/appl/local/containers/sif-images/ 
```

```
$ srun  --pty  \
  singularity  exec  -B  $PWD:/work  \
  /appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-0.4.35.sif  \
  bash
Singularity> cd /work
Singularity> $WITH_CONDA
Singularity> python
Python 3.12.9 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:56:27) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import jax
>>> jax.devices()
[RocmDevice(id=0)] 
```  ### 在 LUMI（仅 CPU）上

在 LUMI 上，您可以按照以下方式设置 Python 发行版：

```
$ module  load  cray-python/3.9.13.1
$ # install needed dependencies locally
$ pip3  install  --user  numpy  numba  matplotlib 
```  ### 在 Google Colab 上

Google Colaboratory，通常称为“Colab”，是一个基于云的 Jupyter 笔记本环境，在您的网络浏览器中运行。使用它需要使用 Google 账户登录。

这就是您如何在 Colab 上获取访问 NVIDIA GPU 的方式：

+   访问[`colab.research.google.com/`](https://colab.research.google.com/)并登录您的 Google 账户

+   在您面前的菜单中，点击右下角的“新建笔记本”

+   笔记本加载完成后，转到顶部的“运行时”菜单并选择“更改运行时类型”

+   在“硬件加速器”下选择“GPU”，并选择一个可用的 NVIDIA GPU 类型（例如 T4）

+   点击“保存”。运行时需要几秒钟来加载 - 您可以在右上角看到状态

+   运行时加载后，您可以输入`!nvidia-smi`来查看 GPU 信息。

+   您现在可以编写通过例如 numba 库在 GPU 上运行的 Python 代码。

## 访问代码示例

本课程的一些练习依赖于您应该在集群上的个人主目录中下载并修改的源代码。所有代码示例都可在与课程本身相同的 GitHub 仓库中找到。要下载，您应使用 Git：

```
$ git  clone  https://github.com/ENCCS/gpu-programming.git
$ cd  gpu-programming/content/examples/
$ ls 
``` 上一页 下一页

* * *

© 版权所有 2023-2024，贡献者。

使用[Sphinx](https://www.sphinx-doc.org/)构建，并使用[主题](https://github.com/readthedocs/sphinx_rtd_theme)提供的[Read the Docs](https://readthedocs.org)。## 本地安装

由于本课程使用 HPC 集群进行教学，因此您自己的计算机无需安装任何软件。

## 在 LUMI 上运行

交互式作业，1 个节点，1 个 GPU，1 小时：

```
$ salloc  -A  project_465002387  -N  1  -t  1:00:00  -p  standard-g  --gpus-per-node=1
$ srun  <some-command> 
```

使用`exit`退出交互式分配。

计算节点上的交互式终端会话：

```
$ srun  --account=project_465002387  --partition=standard-g  --nodes=1  --cpus-per-task=1  --ntasks-per-node=1  --gpus-per-node=1  --time=1:00:00  --pty  bash
$ <some-command> 
```

对应的批处理脚本`submit.sh`：

```
#!/bin/bash -l
#SBATCH --account=project_465002387
#SBATCH --job-name=example-job
#SBATCH --output=examplejob.o%j
#SBATCH --error=examplejob.e%j
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00

srun  <some_command> 
```

+   提交作业：`sbatch submit.sh`

+   监控您的作业：`squeue --me`

+   终止作业：`scancel <JOB_ID>`

### 在 LUMI 上运行 Julia

为了在 LUMI 上使用`AMDGPU.jl`运行 Julia，我们使用以下目录结构，并假设它是我们的工作目录。

```
.
├── Project.toml  # Julia environment
├── script.jl     # Julia script
└── submit.sh     # Slurm batch script 
```

一个`Project.toml`项目文件的示例。

```
[deps]
AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e" 
```

对于`submit.sh`批处理脚本，向上述批处理脚本中添加额外的内容。

```
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1750

module  use  /appl/local/csc/modulefiles

module  load  julia
module  load  julia-amdgpu

julia  --project=.  -e  'using Pkg; Pkg.instantiate()'
julia  --project=.  script.jl 
```

下面提供了一个`script.jl`代码示例。

```
using  AMDGPU

A  =  rand(2⁹,  2⁹)
A_d  =  ROCArray(A)
B_d  =  A_d  *  A_d

println("----EOF----") 
```

## 运行 Python

### 在 LUMI

已创建包含所有必要依赖项的 singularity 容器。要启动容器及其内部的`IPython`解释器，请按照以下操作：

```
$ salloc  -p  small-g  -A  project_465002387  -t  1:00:00  -N  1  --gpus=1
$ srun  --pty  \
  singularity  exec  --no-home  \
  -B  $PWD:/work  \
  /scratch/project_465002387/containers/gpu-programming/python-from-docker/container.sif  \
  bash

Singularity> cd /work
Singularity> . /.venv/bin/activate
Singularity> python  # or ipython 
```

创建容器的配方

为了参考，以下文件被用来创建上面的 singularity 容器。首先是一个 singularity def 文件，

```
Bootstrap:  docker
From:  rocm/dev-ubuntu-24.04:6.4.4-complete

%environment
  CUPY_INSTALL_USE_HIP=1
  ROCM_HOME=/opt/rocm
  HCC_AMDGPU_TARGET=gfx90a
  LLVM_PATH=/opt/rocm/llvm

%post
  export  CUPY_INSTALL_USE_HIP=1
  export  ROCM_HOME=/opt/rocm
  export  HCC_AMDGPU_TARGET=gfx90a
  export  LLVM_PATH=/opt/rocm/llvm
  export  PATH="$HOME/.local/bin/:$PATH"

  apt-get  update  &&  apt-get  install  -y  --no-install-recommends  curl  ca-certificates  git
  curl  -L  https://astral.sh/uv/install.sh  -o  /uv-installer.sh
  sh  /uv-installer.sh  &&  rm  /uv-installer.sh

  .  $HOME/.local/bin/env

  uv  python  install  3.12
  uv  venv  -p  3.12  --seed
  uv  pip  install  --index-strategy  unsafe-best-match  -r  /tmp/envs/requirements.txt
  uv  pip  freeze  >>  /tmp/envs/requirements_pinned.txt

  touch  /usr/lib64/libjansson.so.4  /usr/lib64/libcxi.so.1  /usr/lib64/libjansson.so.4
  mkdir  /var/spool/slurmd  /opt/cray
  mkdir  /scratch  /projappl  /project  /flash  /appl 
```

以及一个用于构建容器的 bash 脚本，

```
#!/bin/sh
ml  purge
ml  LUMI/24.03  partition/G
ml  load  systools/24.03  # For proot

export  SINGULARITY_CACHEDIR="$PWD/singularity/cache"
export  SINGULARITY_TMPDIR="$FLASH/$USER/singularity/tmp"
singularity  build  -B  "$PWD":/tmp/envs  --fix-perms  container.sif  build_singularity.def 
```

小贴士

您还可以使用以下方式交互式构建：

```
singularity build --sandbox <other flags> container-sandbox build_singularity.def
singularity shell --writable container-sandbox 
```

最后是一个`requirements.txt`文件：

```
jupyterlab
jupyterlab-git
nbclassic
matplotlib
numpy
cupy
# jax[rocm]
jax-rocm7-pjrt @ https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jax_rocm7_pjrt-0.6.0-py3-none-manylinux_2_28_x86_64.whl
jax-rocm7-plugin @ https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jax_rocm7_plugin-0.6.0-cp312-cp312-manylinux_2_28_x86_64.whl
jaxlib @ https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jaxlib-0.6.2-cp312-cp312-manylinux2014_x86_64.whl
jax==0.6.2
--extra-index-url https://test.pypi.org/simple
numba-hip[rocm-6-4-4] @ git+https://github.com/ROCm/numba-hip.git 
```

LUMI 还为 Jax 提供了官方的 singularity 镜像。这些镜像可以在以下路径下找到：

```
/appl/local/containers/sif-images/ 
```

```
$ srun  --pty  \
  singularity  exec  -B  $PWD:/work  \
  /appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-0.4.35.sif  \
  bash
Singularity> cd /work
Singularity> $WITH_CONDA
Singularity> python
Python 3.12.9 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:56:27) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import jax
>>> jax.devices()
[RocmDevice(id=0)] 
```  ### 在 LUMI（仅 CPU）

在 LUMI 上，您可以按照以下方式设置 Python 发行版：

```
$ module  load  cray-python/3.9.13.1
$ # install needed dependencies locally
$ pip3  install  --user  numpy  numba  matplotlib 
```  ### 在 Google Colab 上

Google Colaboratory，通常称为“Colab”，是一个基于云的 Jupyter 笔记本环境，它运行在您的网页浏览器中。使用它需要使用 Google 账户登录。

这就是您如何获取 Colab 上的 NVIDIA GPU 访问权限：

+   访问[`colab.research.google.com/`](https://colab.research.google.com/)并登录您的 Google 账户

+   在您面前的菜单中，点击右下角的“新建笔记本”

+   笔记本加载后，转到顶部的“运行时”菜单并选择“更改运行时类型”

+   在“硬件加速器”下选择“GPU”，并选择一种可用的 NVIDIA GPU 类型（例如 T4）

+   点击“保存”。运行时间需要几秒钟来加载 - 您可以在右上角查看状态

+   运行时间加载后，您可以使用`!nvidia-smi`来查看 GPU 信息。

+   您现在可以编写运行在 GPU 上的 Python 代码，例如通过 numba 库。

## 访问代码示例

本课程的一些练习依赖于您应该在集群上的个人主目录中下载并修改的源代码。所有代码示例都可在与课程本身相同的 GitHub 存储库中找到。要下载它，您应该使用 Git：

```
$ git  clone  https://github.com/ENCCS/gpu-programming.git
$ cd  gpu-programming/content/examples/
$ ls 
```

## 本地安装

由于本课程使用 HPC 集群进行教学，因此您自己的计算机无需安装任何软件。

## 在 LUMI 上运行

交互式作业，1 个节点，1 个 GPU，1 小时：

```
$ salloc  -A  project_465002387  -N  1  -t  1:00:00  -p  standard-g  --gpus-per-node=1
$ srun  <some-command> 
```

使用`exit`退出交互式分配。

计算节点上的交互式终端会话：

```
$ srun  --account=project_465002387  --partition=standard-g  --nodes=1  --cpus-per-task=1  --ntasks-per-node=1  --gpus-per-node=1  --time=1:00:00  --pty  bash
$ <some-command> 
```

对应的批处理脚本`submit.sh`：

```
#!/bin/bash -l
#SBATCH --account=project_465002387
#SBATCH --job-name=example-job
#SBATCH --output=examplejob.o%j
#SBATCH --error=examplejob.e%j
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00

srun  <some_command> 
```

+   提交作业：`sbatch submit.sh`

+   监控你的作业：`squeue --me`

+   终止作业：`scancel <JOB_ID>`

### 在 LUMI 上运行 Julia

为了在 LUMI 上使用`AMDGPU.jl`运行 Julia，我们使用了以下目录结构，并假设它是我们的工作目录。

```
.
├── Project.toml  # Julia environment
├── script.jl     # Julia script
└── submit.sh     # Slurm batch script 
```

下面提供了一个`Project.toml`项目文件示例。

```
[deps]
AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e" 
```

对于`submit.sh`批处理脚本，向上述提到的批处理脚本中添加额外的内容。

```
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1750

module  use  /appl/local/csc/modulefiles

module  load  julia
module  load  julia-amdgpu

julia  --project=.  -e  'using Pkg; Pkg.instantiate()'
julia  --project=.  script.jl 
```

下面提供了一个`script.jl`代码示例。

```
using  AMDGPU

A  =  rand(2⁹,  2⁹)
A_d  =  ROCArray(A)
B_d  =  A_d  *  A_d

println("----EOF----") 
```

### 在 LUMI 上运行 Julia

为了在 LUMI 上使用`AMDGPU.jl`运行 Julia，我们使用了以下目录结构，并假设它是我们的工作目录。

```
.
├── Project.toml  # Julia environment
├── script.jl     # Julia script
└── submit.sh     # Slurm batch script 
```

下面提供了一个`Project.toml`项目文件示例。

```
[deps]
AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e" 
```

对于`submit.sh`批处理脚本，向上述提到的批处理脚本中添加额外的内容。

```
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1750

module  use  /appl/local/csc/modulefiles

module  load  julia
module  load  julia-amdgpu

julia  --project=.  -e  'using Pkg; Pkg.instantiate()'
julia  --project=.  script.jl 
```

下面提供了一个`script.jl`代码示例。

```
using  AMDGPU

A  =  rand(2⁹,  2⁹)
A_d  =  ROCArray(A)
B_d  =  A_d  *  A_d

println("----EOF----") 
```

## 运行 Python

### 在 LUMI 上

已经创建了一个包含所有必要依赖项的 singularity 容器。要启动容器及其内部的`IPython`解释器，请按照以下步骤操作：

```
$ salloc  -p  small-g  -A  project_465002387  -t  1:00:00  -N  1  --gpus=1
$ srun  --pty  \
  singularity  exec  --no-home  \
  -B  $PWD:/work  \
  /scratch/project_465002387/containers/gpu-programming/python-from-docker/container.sif  \
  bash

Singularity> cd /work
Singularity> . /.venv/bin/activate
Singularity> python  # or ipython 
```

创建容器的步骤

作为参考，以下文件被用来创建上述 singularity 容器。首先是一个 singularity 定义文件，

```
Bootstrap:  docker
From:  rocm/dev-ubuntu-24.04:6.4.4-complete

%environment
  CUPY_INSTALL_USE_HIP=1
  ROCM_HOME=/opt/rocm
  HCC_AMDGPU_TARGET=gfx90a
  LLVM_PATH=/opt/rocm/llvm

%post
  export  CUPY_INSTALL_USE_HIP=1
  export  ROCM_HOME=/opt/rocm
  export  HCC_AMDGPU_TARGET=gfx90a
  export  LLVM_PATH=/opt/rocm/llvm
  export  PATH="$HOME/.local/bin/:$PATH"

  apt-get  update  &&  apt-get  install  -y  --no-install-recommends  curl  ca-certificates  git
  curl  -L  https://astral.sh/uv/install.sh  -o  /uv-installer.sh
  sh  /uv-installer.sh  &&  rm  /uv-installer.sh

  .  $HOME/.local/bin/env

  uv  python  install  3.12
  uv  venv  -p  3.12  --seed
  uv  pip  install  --index-strategy  unsafe-best-match  -r  /tmp/envs/requirements.txt
  uv  pip  freeze  >>  /tmp/envs/requirements_pinned.txt

  touch  /usr/lib64/libjansson.so.4  /usr/lib64/libcxi.so.1  /usr/lib64/libjansson.so.4
  mkdir  /var/spool/slurmd  /opt/cray
  mkdir  /scratch  /projappl  /project  /flash  /appl 
```

以及一个用于构建容器的 bash 脚本，

```
#!/bin/sh
ml  purge
ml  LUMI/24.03  partition/G
ml  load  systools/24.03  # For proot

export  SINGULARITY_CACHEDIR="$PWD/singularity/cache"
export  SINGULARITY_TMPDIR="$FLASH/$USER/singularity/tmp"
singularity  build  -B  "$PWD":/tmp/envs  --fix-perms  container.sif  build_singularity.def 
```

小贴士

你也可以通过以下方式交互式构建：

```
singularity build --sandbox <other flags> container-sandbox build_singularity.def
singularity shell --writable container-sandbox 
```

最后一个`requirements.txt`文件：

```
jupyterlab
jupyterlab-git
nbclassic
matplotlib
numpy
cupy
# jax[rocm]
jax-rocm7-pjrt @ https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jax_rocm7_pjrt-0.6.0-py3-none-manylinux_2_28_x86_64.whl
jax-rocm7-plugin @ https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jax_rocm7_plugin-0.6.0-cp312-cp312-manylinux_2_28_x86_64.whl
jaxlib @ https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jaxlib-0.6.2-cp312-cp312-manylinux2014_x86_64.whl
jax==0.6.2
--extra-index-url https://test.pypi.org/simple
numba-hip[rocm-6-4-4] @ git+https://github.com/ROCm/numba-hip.git 
```

LUMI 还提供了 Jax 的官方 singularity 镜像。这些镜像可以在以下路径下找到：

```
/appl/local/containers/sif-images/ 
```

```
$ srun  --pty  \
  singularity  exec  -B  $PWD:/work  \
  /appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-0.4.35.sif  \
  bash
Singularity> cd /work
Singularity> $WITH_CONDA
Singularity> python
Python 3.12.9 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:56:27) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import jax
>>> jax.devices()
[RocmDevice(id=0)] 
```  ### 在 LUMI（仅 CPU）

在 LUMI 上，你可以按照以下方式设置 Python 发行版：

```
$ module  load  cray-python/3.9.13.1
$ # install needed dependencies locally
$ pip3  install  --user  numpy  numba  matplotlib 
```  ### 在 Google Colab

Google Colaboratory，通常被称为“Colab”，是一个基于云的 Jupyter 笔记本环境，它在你的网络浏览器中运行。使用它需要使用 Google 账户登录。

这就是如何在 Colab 上获取访问 NVIDIA GPU 的方式：

+   访问[`colab.research.google.com/`](https://colab.research.google.com/)并登录你的 Google 账户

+   在你面前的菜单中，点击右下角的“新建笔记本”

+   笔记本加载完成后，转到顶部的“运行”菜单并选择“更改运行类型”

+   在“硬件加速器”下选择“GPU”并选择一个可用的 NVIDIA GPU 类型（例如 T4）

+   点击“保存”。运行时需要几秒钟来加载 - 你可以在右上角看到状态。

+   运行时加载完成后，你可以输入`!nvidia-smi`来查看 GPU 的信息。

+   你现在可以编写运行在 GPU 上的 Python 代码，例如通过 numba 库。### 在 LUMI 上

已经创建了一个包含所有必要依赖项的 singularity 容器。要启动容器及其内部的`IPython`解释器，请按照以下步骤操作：

```
$ salloc  -p  small-g  -A  project_465002387  -t  1:00:00  -N  1  --gpus=1
$ srun  --pty  \
  singularity  exec  --no-home  \
  -B  $PWD:/work  \
  /scratch/project_465002387/containers/gpu-programming/python-from-docker/container.sif  \
  bash

Singularity> cd /work
Singularity> . /.venv/bin/activate
Singularity> python  # or ipython 
```

创建容器的步骤

作为参考，以下文件被用来创建上述 singularity 容器。首先是一个 singularity 定义文件，

```
Bootstrap:  docker
From:  rocm/dev-ubuntu-24.04:6.4.4-complete

%environment
  CUPY_INSTALL_USE_HIP=1
  ROCM_HOME=/opt/rocm
  HCC_AMDGPU_TARGET=gfx90a
  LLVM_PATH=/opt/rocm/llvm

%post
  export  CUPY_INSTALL_USE_HIP=1
  export  ROCM_HOME=/opt/rocm
  export  HCC_AMDGPU_TARGET=gfx90a
  export  LLVM_PATH=/opt/rocm/llvm
  export  PATH="$HOME/.local/bin/:$PATH"

  apt-get  update  &&  apt-get  install  -y  --no-install-recommends  curl  ca-certificates  git
  curl  -L  https://astral.sh/uv/install.sh  -o  /uv-installer.sh
  sh  /uv-installer.sh  &&  rm  /uv-installer.sh

  .  $HOME/.local/bin/env

  uv  python  install  3.12
  uv  venv  -p  3.12  --seed
  uv  pip  install  --index-strategy  unsafe-best-match  -r  /tmp/envs/requirements.txt
  uv  pip  freeze  >>  /tmp/envs/requirements_pinned.txt

  touch  /usr/lib64/libjansson.so.4  /usr/lib64/libcxi.so.1  /usr/lib64/libjansson.so.4
  mkdir  /var/spool/slurmd  /opt/cray
  mkdir  /scratch  /projappl  /project  /flash  /appl 
```

以及一个用于构建容器的 bash 脚本，

```
#!/bin/sh
ml  purge
ml  LUMI/24.03  partition/G
ml  load  systools/24.03  # For proot

export  SINGULARITY_CACHEDIR="$PWD/singularity/cache"
export  SINGULARITY_TMPDIR="$FLASH/$USER/singularity/tmp"
singularity  build  -B  "$PWD":/tmp/envs  --fix-perms  container.sif  build_singularity.def 
```

小贴士

你也可以通过以下方式交互式构建：

```
singularity build --sandbox <other flags> container-sandbox build_singularity.def
singularity shell --writable container-sandbox 
```

最后一个`requirements.txt`文件：

```
jupyterlab
jupyterlab-git
nbclassic
matplotlib
numpy
cupy
# jax[rocm]
jax-rocm7-pjrt @ https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jax_rocm7_pjrt-0.6.0-py3-none-manylinux_2_28_x86_64.whl
jax-rocm7-plugin @ https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jax_rocm7_plugin-0.6.0-cp312-cp312-manylinux_2_28_x86_64.whl
jaxlib @ https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jaxlib-0.6.2-cp312-cp312-manylinux2014_x86_64.whl
jax==0.6.2
--extra-index-url https://test.pypi.org/simple
numba-hip[rocm-6-4-4] @ git+https://github.com/ROCm/numba-hip.git 
```

LUMI 还提供了 Jax 的官方 singularity 镜像。这些镜像可以在以下路径下找到：

```
/appl/local/containers/sif-images/ 
```

```
$ srun  --pty  \
  singularity  exec  -B  $PWD:/work  \
  /appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-0.4.35.sif  \
  bash
Singularity> cd /work
Singularity> $WITH_CONDA
Singularity> python
Python 3.12.9 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:56:27) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import jax
>>> jax.devices()
[RocmDevice(id=0)] 
```

### 在 LUMI（仅 CPU）

在 LUMI 上，你可以按照以下方式设置 Python 发行版：

```
$ module  load  cray-python/3.9.13.1
$ # install needed dependencies locally
$ pip3  install  --user  numpy  numba  matplotlib 
```

### 在 Google Colab

Google Colaboratory，通常被称为“Colab”，是一个基于云的 Jupyter 笔记本环境，它可以在你的网络浏览器中运行。使用它需要使用 Google 账户登录。

这就是如何在 Colab 上获取访问 NVIDIA GPU 的方法：

+   访问 [`colab.research.google.com/`](https://colab.research.google.com/) 并使用你的 Google 账户登录。

+   在您面前的菜单中，点击右下角的“新建笔记本”。

+   笔记本加载后，前往顶部的“运行”菜单并选择“更改运行类型”。

+   在“硬件加速器”下选择“GPU”并选择一个可用的 NVIDIA GPU 类型（例如 T4）。

+   点击“保存”。运行环境需要几秒钟来加载 - 你可以在右上角看到状态。

+   运行环境加载后，你可以输入 `!nvidia-smi` 来查看 GPU 的信息。

+   你现在可以通过例如 numba 库来编写在 GPU 上运行的 Python 代码。

## 访问代码示例

本课程的一些练习依赖于你应该下载并修改在集群上自己的主目录中的源代码。所有代码示例都可在与课程本身相同的 GitHub 仓库中找到。要下载它，你应该使用 Git：

```
$ git  clone  https://github.com/ENCCS/gpu-programming.git
$ cd  gpu-programming/content/examples/
$ ls 
```*

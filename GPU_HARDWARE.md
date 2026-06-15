# GPU 硬件信息

本文档记录当前 CUDA 开发环境及 GPU 的主要硬件参数。信息由
[`query.cu`](query.cu) 查询得到。

## CUDA 软件环境

| 属性 | 查询结果 |
| --- | --- |
| Driver-supported CUDA version | 13.3 |
| CUDA Runtime version | 13.3 |
| Visible CUDA devices | 1 |

## 设备信息

| 属性 | 查询结果 |
| --- | --- |
| Device | 0 |
| Name | NVIDIA GeForce RTX 4060 Laptop GPU |
| UUID | `GPU-acfd1c22-e29b-b24d-fe6a-1a7df1a4ee38` |
| Compute capability | 8.9 |
| PCI domain:bus:device | `0000:01:00` |
| Compute mode | Default |

## 内存层次

| 属性 | 查询结果 |
| --- | --- |
| Global memory | 8585216000 bytes（8187.50 MiB，8.00 GiB） |
| L2 cache | 33554432 bytes（32.00 MiB，0.03 GiB） |
| Shared memory / block | 49152 bytes（0.05 MiB，0.00 GiB） |
| Shared memory / SM | 102400 bytes（0.10 MiB，0.00 GiB） |
| Global memory bus width | 128 bits |
| Memory clock | 8001 MHz |
| Theoretical memory bandwidth | 256.03 GB/s（238.45 GiB/s） |
| Constant memory | 65536 bytes |
| Memory pitch limit | 2147483647 bytes |
| Texture alignment | 512 bytes |

理论显存带宽根据显存时钟和总线宽度计算：

```text
bandwidth = memory_clock × 2 × memory_bus_width / 8
```

该数值是理论峰值，不包含协议开销、访存模式、缓存命中以及功耗和温度限制，
因此不代表 CUDA kernel 实际能够达到的持续显存吞吐。

## 执行资源

| 属性 | 查询结果 |
| --- | --- |
| Streaming multiprocessors | 24 |
| Warp size | 32 |
| Max threads / block | 1024 |
| Max threads / SM | 1536 |
| Max block dimensions | `(1024, 1024, 64)` |
| Max grid dimensions | `(2147483647, 65535, 65535)` |
| Registers / block | 65536 |
| Registers / SM | 65536 |
| Core clock | 2370 MHz |

线程块各维度的上限不能直接组合为一个合法启动配置。线程块中的线程总数仍不能超过
1024。一个 SM 能够同时驻留的线程块数量还会受到线程数、寄存器和共享内存用量限制。

## 功能特性

| 属性 | 查询结果 |
| --- | --- |
| ECC enabled | no |
| Concurrent kernels | yes |
| Async copy engines | 1 |
| Unified virtual addressing | yes |
| Managed memory | yes |
| Concurrent managed access | no |
| Pageable memory access | no |
| Host memory mapping | yes |
| Cooperative launch | yes |
| Runtime active device | 0 |

# Reduce Sum 优化与性能分析

本文档记录 `reduce_*` CUDA kernel 的逐步优化过程及 Nsight Compute 分析结果。
每次只引入一种主要优化，并使用相同的输入规模和测量方法，以便直观比较各版本的效果。

GPU 硬件参数见 [`../../GPU_HARDWARE.md`](../../GPU_HARDWARE.md)。

## 固定实验条件

| 项目 | 配置 |
| --- | --- |
| 输入类型 | `float` |
| 输入规模 | `N = 2^27`，共 134217728 个元素 |
| 输入数据量 | 512 MiB |
| 每个线程块的线程数 | 256 |
| 编译优化 | `-O3` |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| 计算能力 | 8.9 |
| 理论显存带宽 | 256.03 GB/s |
| Nsight Compute | 2026.2.0.0 |

除非某一版本专门研究线程块大小，否则所有版本均保持 256 个线程。记录结果时还应注明
GPU 功耗模式、温度和频率是否稳定，避免将动态调频造成的差异误认为 kernel 优化效果。

## 编译与分析命令

编译时加入 `-lineinfo`，使 Nsight Compute 能够把指标关联到 CUDA 源代码行：

```bash
nvcc -O3 -lineinfo -std=c++17 reduce_sum.cu -o reduce_sum
```

收集某个 kernel 的完整报告：

```bash
ncu --set full \
  --kernel-name regex:reduce_v0 \
  --export reduce_v0 \
  --force-overwrite \
  ./reduce_sum
```

使用图形界面打开生成的报告：

```bash
ncu-ui reduce_v0.ncu-rep
```

快速收集多个版本都需要对比的核心指标：

```bash
ncu \
  --kernel-name regex:reduce_v[0-9]+ \
  --metrics \
gpu__time_duration.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__sass_average_branch_targets_threads_uniform.pct \
  ./reduce_sum
```

不同 Nsight Compute 版本可能调整指标名称。如果命令报告指标不存在，应使用以下命令
查询当前版本支持的名称，并在本文档中记录实际采用的指标：

```bash
ncu --query-metrics
```

## 对比总览

所有耗时都只记录 kernel 执行时间，不包含主机数据初始化、`cudaMalloc`、内存复制和
CPU 正确性检查。本轮 v0-v7 使用八个独立进程分别采集 Full Set，每个版本 45 passes，
每个进程只让 Nsight Compute 分析目标版本。v0 和 v1 首次采样时发生动态降频，因此
重新采集；下表最终采用的八份报告均处于约 2.37 GHz SM 和 7.99 GHz DRAM 频率。

下表是加入 v7 后重新进行的横向复测。v0-v6 的详细章节保留对应版本加入时的采样，
因此其中的时间可能与本表存在少量运行间波动；判断当前版本横向差异时以本表和 v7
章节的同轮数据为准。正式发布数据时仍应重复多轮，并使用中位数降低温度、功耗和
频率波动的影响。

最终采用的 v0、v1 报告分别为 `/tmp/reduce_v0_v7run_repeat.ncu-rep` 和
`/tmp/reduce_v1_v7run_repeat.ncu-rep`；v2-v7 使用对应的
`/tmp/reduce_vN_v7run.ncu-rep`。

| 版本 | 核心优化 | Kernel 时间 | 有效带宽 | DRAM 峰值占比 | 相对 v0 加速 |
| --- | --- | ---: | ---: | ---: | ---: |
| `reduce_v0` | 取模选择活跃线程 | 7.441696 ms | 72.43 GB/s | 28.49% | 1.00x |
| `reduce_v1` | 连续线程计算交错共享内存下标 | 4.652768 ms | 115.84 GB/s | 45.65% | 1.60x |
| `reduce_v2` | 逆序步长连续寻址规约 | 4.516288 ms | 119.34 GB/s | 46.86% | 1.65x |
| `reduce_v3` | 每线程预加两个输入 | 2.369344 ms | 227.03 GB/s | 89.47% | 3.14x |
| `reduce_v4` | 最后一个 warp 手动展开规约 | 2.165216 ms | 248.44 GB/s | 97.68% | 3.44x |
| `reduce_v5` | 模板参数编译期展开块级规约 | 2.174720 ms | 247.35 GB/s | 97.55% | 3.42x |
| `reduce_v6` | warp shuffle 两级规约 | 2.158944 ms | 249.16 GB/s | 97.68% | 3.45x |
| `reduce_v7` | float4 向量化加载与 grid 减半 | 2.160544 ms | 248.73 GB/s | 97.78% | 3.44x |

### 计算方法

相对 v0 加速比：

```text
speedup = time_v0 / time_current
```

所有版本均读取 `N` 个输入元素，每个线程块写出一个局部和。有效带宽按各版本实际
输出块数计算：

```text
bytes = N * sizeof(float) + grid_size * sizeof(float)
effective_bandwidth = bytes / kernel_time
```

v0-v2 的 `grid_size` 为 524288，逻辑输出为 2 MiB；v3-v6 的 `grid_size` 为
262144，逻辑输出为 1 MiB；v7 的 `grid_size` 为 131072，逻辑输出为 0.5 MiB。
如果后续版本执行多轮 kernel，必须继续按实际全局内存读写量计算。

## 重点观察指标

| 类别 | Nsight Compute 指标或面板 | 关注原因 |
| --- | --- | --- |
| 时间 | Duration | 判断优化是否真正缩短 kernel 时间 |
| 显存 | DRAM Throughput、Bytes Read/Write | 判断是否接近显存带宽上限 |
| SM | SM Throughput | 判断计算和控制流是否成为瓶颈 |
| 占用率 | Achieved Occupancy、Active Warps | 判断寄存器、共享内存和线程配置的限制 |
| 控制流 | Branch Efficiency、Warp State Statistics | 识别分支、谓词屏蔽和线程利用率变化 |
| 共享内存 | Shared Memory Throughput、Bank Conflicts | 检查共享内存访问模式 |
| 指令 | Source Counters、Instruction Statistics | 观察取模、分支和地址计算开销 |
| 停顿 | Warp Stall Reasons | 判断同步、依赖和内存等待的影响 |

不能只依据单个百分比判断优化是否有效。最终结论应以 kernel 时间为主，并结合带宽、
活跃 warp、指令数量和停顿原因解释时间变化。

## reduce_v0：交错寻址规约

### 实现方式

1. 每个线程从全局内存读取一个 `float` 到动态共享内存。
2. 规约步长从 1 开始，每轮翻倍。
3. 使用 `tid % (2 * stride) == 0` 选择执行加法的线程。
4. 每轮加法后调用 `__syncthreads()`。
5. 每个线程块输出一个局部和，共输出 524288 个 `float`。

启动配置：

```text
grid  = 524288 blocks
block = 256 threads
dynamic shared memory = 1024 bytes/block
```

### 基线问题

- 每轮只有部分线程参与加法，大量线程会被条件谓词屏蔽，线程束有效利用率下降。
- `%` 运算和随步长变化的下标计算增加了整数指令开销。
- 每轮均执行一次块级同步，共执行 `log2(256) = 8` 次同步。
- 后半程只有少量线程工作，但整个线程块仍需参与同步。
- kernel 只生成块级局部和，不会把输出继续规约为单个标量。

### Nsight Compute 记录

| 指标 | 结果 | 备注 |
| --- | ---: | --- |
| Kernel Duration | 7.443392 ms | 独立 Nsight Compute Full Set，45 passes |
| Effective Bandwidth | 72.41 GB/s | 按逻辑输入和输出字节数计算 |
| Measured DRAM Traffic Rate | 73.46 GB/s | 按实测 DRAM 读写字节数计算 |
| DRAM Bytes Read | 537.305472 MB | 包含实际 DRAM 内存事务 |
| DRAM Bytes Write | 9.498368 MB | 高于逻辑输出的 2 MiB |
| Memory Throughput | 53.99% | GPU Speed Of Light Throughput |
| DRAM Throughput | 28.72% | 显存带宽没有饱和 |
| Compute (SM) Throughput | 73.22% | 明显高于 DRAM Throughput |
| Theoretical Occupancy | 100% | 理论活跃 warp 为 48/SM |
| Achieved Occupancy | 93.94% | 平均活跃 warp 约为 45.09/SM |
| Registers / Thread | 16 | 线程块驻留首先受 warp 数限制 |
| Executed Instructions | 1239941120 | 整数和逻辑 ALU 是最繁忙管线 |
| Branch Efficiency | 100% | 编译器主要通过 predication 控制活跃线程 |
| Shared Bank Conflicts | 285291 | 规模较小，但不是完全为零 |
| Avg. Not Predicated Threads / Warp | 24.46 | 后续规约轮次中有效线程逐渐减少 |

有效带宽使用固定工作量计算：

```text
input bytes  = 134217728 * 4 = 536870912 bytes
output bytes = 524288 * 4    =   2097152 bytes
total bytes  =                      538968064 bytes

effective bandwidth = 538968064 / 0.007443392
                    = 72.41 GB/s
```

### 启动与运行记录

| 项目 | 结果 |
| --- | ---: |
| Grid Size | 524288 |
| Block Size | 256 |
| Threads | 134217728 |
| Streaming Multiprocessors | 24 |
| TPCs | 12 |
| Waves Per SM | 3640.89 |
| Function Cache Configuration | CachePreferNone |
| Shared Memory Configuration Size | 32.77 Kbyte |
| Dynamic Shared Memory / Block | 1.02 Kbyte |
| Driver Shared Memory / Block | 1.02 Kbyte |
| Static Shared Memory / Block | 0 bytes |
| Registers / Thread | 16 |
| Stack Size | 1024 bytes |
| Block Limit Registers | 16 blocks/SM |
| Block Limit Shared Memory | 16 blocks/SM |
| Block Limit Warps | 6 blocks/SM |
| DRAM Frequency | 7.99 GHz |
| SM Frequency | 2.37 GHz |
| Elapsed Cycles | 17640719 |

正确性验证通过：

```text
[PASS] reduce_v0 N=2^27: blocks=524288, failed_blocks=0,
expected_total=18.00000016,
actual_total=18.00000191
```

### 分析结论

- 当前主要瓶颈：计算与控制流开销。Compute Throughput 为 73.22%，而 DRAM
  Throughput 仅为 28.72%，说明 v0 没有充分利用显存带宽。
- 最显著的低效指标：有效带宽只有 72.41 GB/s，约为硬件理论显存带宽
  256.03 GB/s 的 28.28%。该比例与 Nsight Compute 报告的 28.72% DRAM
  Throughput 接近。
- 占用率不是首要问题：Achieved Occupancy 已达到 93.94%，每个 SM 平均有
  约 45.09 个活跃 warp。继续单纯提高 occupancy 的收益预计有限。
- 每线程仅使用 16 个寄存器，每块只使用约 1 KiB 动态共享内存。实际驻留上限是
  6 blocks/SM，由每个 SM 的 48 warp 上限决定，而不是寄存器或共享内存容量。
- 结果符合基线预期：交错寻址中的取模、谓词控制、逐轮同步以及后期低线程利用率，
  会让 SM 工作量高于显存工作量。
- 下一版本应只修改共享内存规约的寻址方式，移除 `%` 运算并让活跃线程连续，
  保持输入规模、线程块大小和输出语义不变。
- 预期下一版本会降低整数指令和谓词控制开销，使 Kernel Duration 下降、
  Effective Bandwidth 上升；Occupancy 不一定发生明显变化。

## reduce_v1：连续线程计算交错共享内存下标

### 优化方式

v1 保持全局内存读取、线程块大小、动态共享内存和输出语义不变，只替换规约阶段的
活跃线程选择方式：

```text
v0: tid % (2 * s) == 0
v1: idx = 2 * tid * s，idx < blockDim.x
```

这样每轮参与计算的线程集中在线程块前部，移除了 `%` 运算，也减少了 v0 中大量线程
反复执行条件判断的整数指令。共享内存访问仍然是交错的，访问步长会随 `s` 增长。

启动配置与 v0 相同：

```text
grid  = 524288 blocks
block = 256 threads
dynamic shared memory = 1024 bytes/block
```

### Nsight Compute 记录

| 指标 | v0 | v1 | 变化 |
| --- | ---: | ---: | ---: |
| Kernel Duration | 7.443392 ms | 4.652128 ms | -37.50% |
| Effective Bandwidth | 72.41 GB/s | 115.85 GB/s | +60.00% |
| Measured DRAM Traffic Rate | 73.46 GB/s | 116.57 GB/s | +58.68% |
| DRAM Bytes Read | 537.305472 MB | 536.986624 MB | -0.06% |
| DRAM Bytes Write | 9.498368 MB | 5.315072 MB | -44.04% |
| DRAM Throughput | 28.72% | 45.58% | +16.85 pp |
| Memory Throughput | 53.99% | 86.39% | +32.40 pp |
| Compute (SM) Throughput | 73.22% | 86.39% | +13.17 pp |
| Executed Instructions | 1239941120 | 463994880 | -62.58% |
| Achieved Occupancy | 93.94% | 89.37% | -4.58 pp |
| Active Warps / SM | 45.09 | 42.90 | -2.20 |
| Eligible Warps / Scheduler | 2.30 | 0.87 | -61.96% |
| No Eligible | 26.77% | 56.14% | +29.37 pp |
| Avg. Not Predicated Threads / Warp | 24.46 | 20.83 | -14.84% |
| Shared Bank Conflicts | 285291 | 55339147 | 约 193.97x |
| Shared Load Bank Conflicts | 2891 | 36704069 | 显著增加 |
| Shared Store Bank Conflicts | 282289 | 18635234 | 显著增加 |
| Branch Efficiency | 100% | 100% | 不变 |

v1 的有效带宽：

```text
effective bandwidth = 538968064 / 0.004652128
                    = 115.85 GB/s

speedup = 7.443392 / 4.652128
        = 1.60x
```

### 正确性验证

两个版本均使用相同的 `N = 2^27` 输入，并通过逐块 CPU 参考结果校验：

```text
[PASS] reduce_v0 N=2^27: blocks=524288, failed_blocks=0,
expected_total=18.00000016,
actual_total=18.00000191

[PASS] reduce_v1 N=2^27: blocks=524288, failed_blocks=0,
expected_total=18.00000016,
actual_total=18.00000191
```

### 优化结果分析

- v1 获得 `1.60x` 加速，Kernel Duration 减少 37.50%。最主要原因是执行指令数
  减少 62.58%，说明移除取模并让前部线程连续参与计算显著降低了控制和索引开销。
- 有效带宽从 72.41 GB/s 提升至 115.85 GB/s，DRAM Throughput 从 28.72%
  提升至 45.58%。输入输出工作量不变，因此这部分提升来自更快完成相同工作。
- v1 的 occupancy 从 93.94% 降至 89.37%，但性能仍大幅提升，进一步证明 v0
  的主要问题不是 occupancy，而是每个输出所需的指令和控制开销。
- v1 的主要新瓶颈是共享内存 bank conflict。总 conflict 从 285291 增加到
  55339147，Nsight Compute 报告共享加载平均约 3.9-way conflict、共享存储平均
  约 2.8-way conflict，并估计这部分具有较大的局部优化空间。
- 调度器无 eligible warp 的周期从约 26.77% 增加到约 56.14%，每个调度器平均 eligible
  warp 从 2.30 降到 0.87。交错共享内存访问的序列化会使 warp 更频繁等待数据。
- Branch Efficiency 均为 100%，但这不表示所有线程都有效工作。v1 的平均非
  predicated 线程数降至 20.83，说明编译器通过 predication 处理 `idx` 条件，
  后续轮次仍有越来越多线程不参与实际加法。
- v1 已同时达到 86.39% 的 Memory Throughput 和 Compute Throughput。继续优化时，
  应减少共享内存事务和 bank conflict，而不是增加 occupancy 或改变全局内存读取。

### 下一步优化方向

下一版本应改为顺序寻址规约：让每轮活跃线程读取相邻或按半区配对的共享内存元素，
避免 `idx = 2 * tid * s` 形成的高步长访问。实验时保持 `N`、block size、输入输出
语义和编译参数不变，以单独验证消除 bank conflict 的收益。

## reduce_v2：逆序步长连续寻址规约

### 优化方式

v2 将规约步长改为从线程块大小的一半开始逐轮减半：

```text
for s = blockDim.x / 2; s > 0; s /= 2:
    if tid < s:
        smem[tid] += smem[tid + s]
```

每轮活跃线程均为连续的 `tid = 0 ... s - 1`，共享内存源地址和目标地址也分别连续。
同一 warp 内线程访问的 bank 不再随着步长形成大范围重复映射，从而消除 v1 的高阶
bank conflict。全局内存读取、线程块大小、动态共享内存和输出语义保持不变。

启动配置与 v0、v1 相同：

```text
grid  = 524288 blocks
block = 256 threads
dynamic shared memory = 1024 bytes/block
```

### Nsight Compute 记录

| 指标 | v1 | v2 | 变化 |
| --- | ---: | ---: | ---: |
| Kernel Duration | 4.652128 ms | 4.517600 ms | -2.89% |
| Effective Bandwidth | 115.85 GB/s | 119.30 GB/s | +2.98% |
| Measured DRAM Traffic Rate | 116.57 GB/s | 120.61 GB/s | +3.47% |
| Elapsed Cycles | 11025425 | 10706593 | -2.89% |
| SM Frequency | 2.37 GHz | 2.37 GHz | 基本不变 |
| DRAM Frequency | 7.99 GHz | 7.99 GHz | 基本不变 |
| DRAM Bytes Read | 536.986624 MB | 537.122432 MB | +0.03% |
| DRAM Bytes Write | 5.315072 MB | 7.755008 MB | +45.91% |
| DRAM Throughput | 45.58% | 47.16% | +1.58 pp |
| Memory Throughput | 86.39% | 88.96% | +2.57 pp |
| Compute (SM) Throughput | 86.39% | 88.96% | +2.57 pp |
| Executed Instructions | 463994880 | 422051840 | -9.04% |
| Achieved Occupancy | 89.37% | 89.18% | -0.19 pp |
| Active Warps / SM | 42.90 | 42.81 | -0.09 |
| Eligible Warps / Scheduler | 0.87 | 0.81 | -7.25% |
| No Eligible | 56.14% | 58.94% | +2.80 pp |
| Avg. Not Predicated Threads / Warp | 20.83 | 20.03 | -3.84% |
| Shared Bank Conflicts | 55339147 | 169935 | -99.69% |
| Shared Load Bank Conflicts | 36704069 | 3142 | -99.99% |
| Shared Store Bank Conflicts | 18635234 | 165881 | -99.11% |
| Branch Efficiency | 100% | 100% | 不变 |

v2 的有效带宽和加速比：

```text
effective bandwidth = 538968064 / 0.004517600
                    = 119.30 GB/s

speedup vs v1 = 4.652128 / 4.517600
              = 1.03x

speedup vs v0 = 7.443392 / 4.517600
              = 1.65x
```

### 正确性验证

v2 对全部 524288 个线程块的局部和均通过 CPU 参考结果校验：

```text
[PASS] reduce_v2 N=2^27: blocks=524288, failed_blocks=0,
expected_total=18.00000016,
actual_total=18
```

`actual_total` 与前两版末位不同，是浮点加法顺序改变造成的正常舍入差异。逐块结果均在
既定误差容限内，因此不影响正确性结论。

### 优化结果分析

- v2 相比 v1 获得 `1.03x` 实测加速，Kernel Duration 减少 2.89%；相比 v0
  获得 `1.65x` 加速。
- 共享内存 bank conflict 从 55339147 降至 169935，减少 99.69%，也就是 v1
  的 conflict 数约为 v2 的 325.65 倍。Nsight Compute 不再对 v2 报告共享内存
  load/store bank conflict 优化建议，说明逆序步长连续寻址达到了核心目标。
- 执行指令数从 463994880 降至 422051840，减少 9.04%。v2 不再计算
  `idx = 2 * tid * s`，每轮只需比较 `tid < s` 并访问两个连续半区。
- v1 与 v2 的 SM 和 DRAM 频率基本一致，执行周期和实测时间都减少约 2.89%，因此
  本轮可以把这部分收益主要归因于寻址方式和指令数量变化。
- v2 的 Memory Throughput 与 Compute Throughput 均达到 88.96%，高于 v1 的
  86.39%；有效带宽达到 119.30 GB/s，说明相同输入输出工作量完成得更快。
- 本轮 v2 的实测 DRAM 写流量为 7.76 MB，高于逻辑输出的 2 MiB。该计数会受
  Full Set 多 pass 重放、缓存写回和测量时系统流量影响，不能解释为
  kernel 算法额外写出了这些数据；各版本的逻辑字节数仍按 kernel 语义计算。
- occupancy 基本不变，v1 为 89.37%，v2 为 89.18%。再次说明本次收益来自更高效的
  共享内存访问和更少指令，而不是增加驻留 warp。
- 调度器无 eligible warp 的比例仍约为 58.94%，平均每个调度器只有 0.81 个 eligible
  warp。bank conflict 已不再是主要原因，剩余等待更可能来自每轮块级同步、共享内存
  数据依赖，以及后半程大量线程被谓词屏蔽。
- Nsight Compute 仍提示全局输出存储的 sector 利用率较低。每个线程块只有线程 0
  写出一个 `float`，不同线程块的输出无法由同一 warp 合并成连续写事务，这是当前
  “每块输出一个局部和”语义带来的固定开销。

### 下一步优化方向

下一版本可以让每个线程先从全局内存加载并累加两个元素，使每个线程块处理
`2 * blockDim.x` 个输入。这样可以将线程块数量和局部和输出数量减半，并减少整体同步
与调度开销。该方向已由 v3 实现并验证。

## reduce_v3：每线程预加两个输入

### 优化方式

v3 保留 v2 的逆序步长连续共享内存规约，但每个线程在写入共享内存前先读取并累加
两个全局内存元素：

```text
gid = blockIdx.x * blockDim.x * 2 + tid
value = input[gid] + input[gid + blockDim.x]
smem[tid] = value
```

边界位置分别检查两个全局内存下标。每个线程块处理 `2 * blockDim.x = 512` 个输入，
因此 grid 使用：

```text
grid = (N + blockDim.x * 2 - 1) / (blockDim.x * 2)
     = 262144 blocks
```

与 v2 相比，v3 的线程块数量、线程总数、waves 和局部和输出数量均减半：

```text
block = 256 threads
threads = 67108864
waves per SM = 1820.44
dynamic shared memory = 1024 bytes/block
logical output = 262144 * 4 bytes = 1 MiB
```

### Nsight Compute 记录

| 指标 | v2 | v3 | 变化 |
| --- | ---: | ---: | ---: |
| Grid Size | 524288 | 262144 | -50.00% |
| Threads | 134217728 | 67108864 | -50.00% |
| Waves Per SM | 3640.89 | 1820.44 | -50.00% |
| Kernel Duration | 4.517600 ms | 2.366848 ms | -47.61% |
| Effective Bandwidth | 119.30 GB/s | 227.27 GB/s | +90.50% |
| Measured DRAM Traffic Rate | 120.61 GB/s | 228.82 GB/s | +89.71% |
| Elapsed Cycles | 10706593 | 5609358 | -47.61% |
| SM Frequency | 2.37 GHz | 2.37 GHz | 基本不变 |
| DRAM Frequency | 7.99 GHz | 7.99 GHz | 基本不变 |
| DRAM Bytes Read | 537.122432 MB | 537.166720 MB | +0.01% |
| DRAM Bytes Write | 7.755008 MB | 4.411520 MB | -43.11% |
| DRAM Throughput | 47.16% | 89.46% | +42.31 pp |
| Memory Throughput | 88.96% | 89.46% | +0.50 pp |
| Compute (SM) Throughput | 88.96% | 88.01% | -0.95 pp |
| Executed Instructions | 422051840 | 227803136 | -46.02% |
| Achieved Occupancy | 89.18% | 91.23% | +2.05 pp |
| Active Warps / SM | 42.81 | 43.79 | +0.98 |
| Eligible Warps / Scheduler | 0.81 | 0.84 | +3.00% |
| No Eligible | 58.94% | 57.67% | -1.27 pp |
| Avg. Not Predicated Threads / Warp | 20.03 | 20.91 | +4.39% |
| Shared Bank Conflicts | 169935 | 157938 | -7.06% |
| Shared Load Bank Conflicts | 3142 | 1789 | -43.06% |
| Shared Store Bank Conflicts | 165881 | 157676 | -4.95% |
| Branch Efficiency | 100% | 100% | 不变 |

v3 的逻辑字节数需要使用减半后的输出数量：

```text
input bytes  = 134217728 * 4 = 536870912 bytes
output bytes = 262144 * 4    =   1048576 bytes
total bytes  =                      537919488 bytes

effective bandwidth = 537919488 / 0.002366848
                    = 227.27 GB/s

speedup vs v2 = 4.517600 / 2.366848
              = 1.91x

speedup vs v0 = 7.443392 / 2.366848
              = 3.14x
```

### 正确性验证

v3 的 262144 个局部和均通过 CPU 参考结果校验：

```text
[PASS] reduce_v3 N=2^27: blocks=262144, failed_blocks=0,
expected_total=18.00000016,
actual_total=18.00000465
```

v3 改变了每个局部和内部的浮点加法顺序，因此总和末位与 v0-v2 不同。所有块均在既定
误差容限内，结果正确。

### 优化结果分析

- v3 相比 v2 获得 `1.91x` 加速，Kernel Duration 减少 47.61%；相比 v0 累计获得
  `3.14x` 加速。
- v2 和 v3 的 SM、DRAM 频率基本一致，执行周期与耗时均减少 47.61%，因此本次提升
  可以直接归因于工作组织变化，而不是动态频率差异。
- grid、线程数和 waves 减半后，每个输入元素仍只从全局内存读取一次，但只需要一半的
  线程块执行共享内存规约。每块包含 8 次规约循环同步，另有一次加载后的同步；局部和
  输出也从 2 MiB 减至 1 MiB。
- 执行指令从 422051840 降至 227803136，减少 46.02%，与线程块数量减半基本一致。
  每个线程增加一次全局读取和浮点加法，但省掉了一半线程块的索引、同步、共享内存和
  输出相关指令，总体指令数仍接近减半。
- DRAM Throughput 从 47.16% 提升至 89.46%，Measured DRAM Traffic Rate 达到
  228.82 GB/s。逻辑有效带宽为 227.27 GB/s，达到硬件理论带宽 256.03 GB/s 的
  88.77%，说明 v3 已从同步和规约开销受限转为接近 DRAM 带宽受限。
- Memory Throughput 只从 88.96% 增至 89.46%，但 v2 的峰值主要来自 L1/LSU
  共享内存管线；v3 的 DRAM Throughput 从 47.16% 跃升到 89.46%，表明消除的块级
  开销转化为了更高的全局内存供给效率。
- occupancy 提升至 91.23%，bank conflict 仍处于很低水平。它们都不是 v3 的首要瓶颈。
- Nsight Compute 仍提示全局输出 store 的 sector 利用率低，但实测 DRAM 写流量只有
  4.34 MB，相比约 537 MB 的读取很小。该建议的估算加速比不能直接视为整体 kernel
  可获得的收益，当前更应关注已经达到 89.46% 的 DRAM 吞吐上限。

### 下一步优化方向

v3 已接近当前 GPU 的显存带宽上限，后续收益预计明显收窄。v4 已实现“最后一个 warp
手动展开”这一方向，下面单独记录结果。其他可继续实验的方向包括：

- 每个线程加载 4 个或更多元素，进一步减少 grid 和同步，但要确认并行度仍足以饱和 GPU。
- 使用 warp shuffle 替代最后一个 warp 的共享内存通信，获得更明确的同步语义。
- 比较 128、256、512 个线程的 block size，观察 DRAM 吞吐和调度器 eligible warp。
- 对最终单标量规约增加后续 kernel；当前结果只衡量第一轮块级局部和，不代表完整归约。

## reduce_v4：最后一个 warp 手动展开规约

### 优化方式

v4 保留 v3 的“每线程预加两个输入”和相同的 grid 配置，但块级规约循环只执行
`s = 128` 和 `s = 64` 两轮：

```text
for s = blockDim.x / 2; s > 32; s /= 2:
    if tid < s:
        smem[tid] += smem[tid + s]
    __syncthreads()
```

剩余 64 个共享内存值由第一个 warp 通过 `warpReduce` 展开完成。v3 在输入写入共享
内存后执行 8 次循环内 `__syncthreads()`，加上加载后的同步，共有 9 次块级同步；
v4 只保留 2 次循环内同步，加上加载后的同步，共有 3 次块级同步，因此每个线程块
减少 6 次块级 barrier。

启动配置与 v3 完全相同：

```text
grid  = 262144 blocks
block = 256 threads
threads = 67108864
dynamic shared memory = 1024 bytes/block
logical output = 262144 * 4 bytes = 1 MiB
```

### Nsight Compute 记录

| 指标 | v3 | v4 | 变化 |
| --- | ---: | ---: | ---: |
| Kernel Duration | 2.366848 ms | 2.168224 ms | -8.39% |
| Effective Bandwidth | 227.27 GB/s | 248.09 GB/s | +9.16% |
| Measured DRAM Traffic Rate | 228.82 GB/s | 249.74 GB/s | +9.14% |
| Elapsed Cycles | 5609358 | 5138663 | -8.39% |
| SM Frequency | 2.37 GHz | 2.37 GHz | 基本不变 |
| DRAM Frequency | 7.99 GHz | 7.99 GHz | 基本不变 |
| DRAM Bytes Read | 537.166720 MB | 537.085312 MB | -0.02% |
| DRAM Bytes Write | 4.411520 MB | 4.399104 MB | -0.28% |
| DRAM Throughput | 89.46% | 97.64% | +8.18 pp |
| Memory Throughput | 89.46% | 97.64% | +8.18 pp |
| Compute (SM) Throughput | 88.01% | 42.51% | -45.50 pp |
| Executed Instructions | 227803136 | 118751232 | -47.87% |
| Achieved Occupancy | 91.23% | 82.16% | -9.06 pp |
| Active Warps / SM | 43.79 | 39.44 | -4.35 |
| Eligible Warps / Scheduler | 0.84 | 0.32 | -61.20% |
| No Eligible | 57.67% | 75.88% | +18.21 pp |
| Avg. Not Predicated Threads / Warp | 20.91 | 26.92 | +28.74% |
| Shared Bank Conflicts | 157938 | 224285 | +42.01% |
| Shared Load Bank Conflicts | 1789 | 2016 | +12.69% |
| Shared Store Bank Conflicts | 157676 | 209487 | +32.86% |
| Barrier Stall Sample Share | 24.67% | 10.54% | -14.13 pp |
| Long Scoreboard Sample Share | 26.59% | 70.11% | +43.53 pp |
| Branch Efficiency | 100% | 100% | 不变 |

PC Sampling 占比使用对应 stall 样本数除以该版本的总样本数。v3 共采集 135654 个
样本，其中 barrier 为 33467、long scoreboard 为 36065；v4 共采集 126800 个
样本，其中 barrier 为 13364、long scoreboard 为 88902。

v4 的有效带宽和加速比：

```text
input bytes  = 134217728 * 4 = 536870912 bytes
output bytes = 262144 * 4    =   1048576 bytes
total bytes  =                      537919488 bytes

effective bandwidth = 537919488 / 0.002168224
                    = 248.09 GB/s

speedup vs v3 = 2.366848 / 2.168224
              = 1.09x

speedup vs v0 = 7.443392 / 2.168224
              = 3.43x
```

### 正确性验证

v4 的 262144 个局部和均通过 CPU 参考结果校验：

```text
[PASS] reduce_v4 N=2^27: blocks=262144, failed_blocks=0,
expected_total=18.00000016,
actual_total=18.00000465
```

### 优化结果分析

- v4 相比 v3 获得 `1.09x` 加速，Kernel Duration 减少 8.39%；相比 v0 累计获得
  `3.43x` 加速。v3 和 v4 的 SM、DRAM 频率相同，因此这部分差异不是动态调频造成的。
- 块级同步次数从每块 9 次降到 3 次，Barrier Stall Sample Share 从 24.67% 降至
  10.54%。执行指令数同时减少 47.87%，说明展开最后一个 warp 不只是少执行 barrier，
  还避免了其余 224 个线程参与后六轮循环控制、谓词判断和地址计算。
- v4 的平均有效线程数从 20.91 提升到 26.92。最后六轮只调度一个 warp，不再让整个
  线程块的其他 warp 执行被谓词关闭的规约指令。
- v4 的逻辑有效带宽达到 248.09 GB/s，是理论显存带宽 256.03 GB/s 的 96.90%；
  Nsight Compute 的 DRAM Throughput 达到 97.64%。这说明 v4 已非常接近当前 GPU
  的显存带宽上限。
- v4 的 Compute (SM) Throughput 从 88.01% 降到 42.51% 不是性能退化。v3 的高值
  主要来自 LSU 上的大量共享内存和同步相关工作；v4 删除这些工作后，SM 管线压力下降，
  相同的全局内存数据反而在更短时间内完成。
- Eligible Warps 降至 0.32、No Eligible 升至 75.88%，Long Scoreboard Sample Share
  升至 70.11%。这与 DRAM 吞吐达到 97.64% 相互印证：主要等待已经从块级 barrier
  转移为等待全局内存返回。此时继续只减少规约指令，整体收益会受到显存上限约束。
- Shared Bank Conflicts 增加 42.01%，但绝对值只有 224285，远低于 v1 的 55339147，
  且 kernel 时间仍明显下降。该变化不是 v4 的主要瓶颈。
- Achieved Occupancy 降至 82.16%，但理论 occupancy 仍为 100%，每线程仍只使用
  16 个寄存器、每块仍只使用约 1 KiB 共享内存。这个动态平均值下降没有形成资源驻留
  限制，当前性能首先受 DRAM 长延迟和带宽上限约束。

### 正确性边界

当前 `warpReduce` 使用 `volatile float *`，会强制编译器实际执行共享内存读写，因而
本轮测试能够得到正确结果。但是 `volatile` 只约束编译器的内存访问优化，不提供线程
同步或内存栅栏。Volta 及之后架构支持 independent thread scheduling，不能把同一
warp 永远严格锁步作为 CUDA 内存模型保证。

因此，本节数据准确描述当前 v4 实现的实测表现，但该写法仍有可移植性风险。更稳健的
后续版本应使用带正确 active mask 的 `__syncwarp()` 明确分隔共享内存依赖阶段，或用
`__shfl_down_sync` 完成 warp 内规约；修改同步方式后需要重新进行正确性和性能采样。

### 下一步优化方向

- v5 已使用模板参数在编译期展开块级规约，下面记录其收益和正确性边界。
- 优先实现 warp shuffle 版本，并与当前 volatile 共享内存版本比较指令数和耗时。
- 若继续增加每线程加载元素数，应确认约 248 GB/s 的有效带宽是否还有稳定提升空间。
- 完整归约还需要继续处理 262144 个局部和，端到端优化不能只依据第一轮 kernel 时间。

## reduce_v5：模板参数编译期展开块级规约

### 优化方式

v5 保留 v4 的每线程双元素加载、相同的 grid 配置和最后一个 warp 共享内存规约，
但把线程块大小改为模板参数 `BLOCK_SIZE`。规约阶段使用编译期可判定的条件：

```text
if BLOCK_SIZE >= 512: 合并相距 256 的元素
if BLOCK_SIZE >= 256: 合并相距 128 的元素
if BLOCK_SIZE >= 128: 合并相距 64 的元素
最后一个 warp 完成剩余规约
```

本轮测试实例化 `reduce_v5<256>`。编译器可以删除 `BLOCK_SIZE >= 512` 分支，并将
其余固定阶段直接展开，避免 v4 中规约循环的步长更新、循环比较和跳转指令。

启动配置与 v4 完全相同：

```text
grid  = 262144 blocks
block = 256 threads
template BLOCK_SIZE = 256
threads = 67108864
dynamic shared memory = 1024 bytes/block
logical output = 262144 * 4 bytes = 1 MiB
```

### Nsight Compute 记录

| 指标 | v4 | v5 | 变化 |
| --- | ---: | ---: | ---: |
| Kernel Duration | 2.168224 ms | 2.154560 ms | -0.63% |
| Effective Bandwidth | 248.09 GB/s | 249.67 GB/s | +0.63% |
| Measured DRAM Traffic Rate | 249.74 GB/s | 249.81 GB/s | +0.03% |
| Elapsed Cycles | 5138663 | 5106192 | -0.63% |
| SM Frequency | 2.37 GHz | 2.37 GHz | 基本不变 |
| DRAM Frequency | 7.99 GHz | 7.99 GHz | 基本不变 |
| DRAM Bytes Read | 537.085312 MB | 536.945792 MB | -0.03% |
| DRAM Bytes Write | 4.399104 MB | 1.294848 MB | -70.57% |
| DRAM Throughput | 97.64% | 97.67% | +0.03 pp |
| Memory Throughput | 97.64% | 97.67% | +0.03 pp |
| Compute (SM) Throughput | 42.51% | 42.78% | +0.27 pp |
| Executed Instructions | 118751232 | 80740352 | -32.01% |
| Achieved Occupancy | 82.16% | 82.45% | +0.28 pp |
| Active Warps / SM | 39.44 | 39.57 | +0.14 |
| Eligible Warps / Scheduler | 0.32 | 0.20 | -37.97% |
| No Eligible | 75.88% | 83.51% | +7.62 pp |
| Avg. Not Predicated Threads / Warp | 26.92 | 27.33 | +1.52% |
| Shared Bank Conflicts | 224285 | 217167 | -3.17% |
| Shared Load Bank Conflicts | 2016 | 1694 | -15.97% |
| Shared Store Bank Conflicts | 209487 | 214298 | +2.30% |
| Barrier Stall Sample Share | 10.54% | 11.45% | +0.91 pp |
| Long Scoreboard Sample Share | 70.11% | 72.80% | +2.69 pp |
| Branch Efficiency | 100% | 100% | 不变 |

v5 共采集 119776 个 PC Sampling 样本，其中 barrier 为 13716，long scoreboard
为 87195。Nsight Compute 报告平均每条已发射指令对应的 long scoreboard 等待从
v4 的 28.93 cycles 增加到 v5 的 45.44 cycles。

v5 的有效带宽和加速比：

```text
input bytes  = 134217728 * 4 = 536870912 bytes
output bytes = 262144 * 4    =   1048576 bytes
total bytes  =                      537919488 bytes

effective bandwidth = 537919488 / 0.002154560
                    = 249.67 GB/s

speedup vs v4 = 2.168224 / 2.154560
              = 1.0063x

speedup vs v0 = 7.443392 / 2.154560
              = 3.45x
```

### 正确性验证

v5 的 262144 个局部和在当前 RTX 4060 Laptop GPU 和 CUDA 13.3 环境中均通过
CPU 参考结果校验：

```text
[PASS] reduce_v5 N=2^27: blocks=262144, failed_blocks=0,
expected_total=18.00000016,
actual_total=18.00000465
```

### 优化结果分析

- v5 将执行指令数从 118751232 降至 80740352，减少 32.01%，说明模板参数确实让
  编译器删除了无效分支并展开了固定规约阶段。
- Kernel Duration 减少 0.013664 ms，即 0.63%，获得 `1.0063x` 小幅加速。该收益远小于
  32.01% 的指令降幅，正式结论仍应通过多轮独立采样的中位数确认。
- v4 已达到 97.64% DRAM Throughput，v5 为 97.67%；v5 有效带宽为 249.67 GB/s，
  达到理论显存带宽 256.03 GB/s 的 97.51%。剩余规约指令大多已不在关键路径上。
- 指令数减少后，Eligible Warps 从 0.32 降至 0.20，No Eligible 升至 82.70%，
  long scoreboard 样本占比升至 72.80%。这不是模板展开导致性能变差，而是每个 warp
  更少执行计算指令、更多时间表现为等待全局内存数据。
- Achieved Occupancy 从 82.16% 小幅升至 82.45%，每线程寄存器数仍为 16、每块共享
  内存仍约为 1 KiB，理论 occupancy 仍为 100%。动态平均 occupancy 不是当前瓶颈。
- bank conflict 总数变化只有 -3.17%，绝对规模仍远低于 v1。v5 的性能上限由 DRAM
  决定，而不是共享内存访问。

### 正确性边界

当前 v5 已将各阶段的 `__syncthreads()` 放在 `tid` 条件之外，因此整个线程块都会
执行相同数量的块级 barrier，满足块级同步的一致性要求。

最后一个 warp 仍通过 volatile 共享内存完成展开规约，因此继承了 v4 所述的同步语义
风险：volatile 会约束编译器的内存访问优化，但不提供 warp 同步保证。后续仍建议使用
`__syncwarp()` 明确分隔共享内存依赖阶段，或改用 `__shfl_down_sync`。

### 下一步优化方向

- 通过多轮重复采样或 CUDA Event 基准确认约 0.63% 的 v4-v5 差异是否稳定。
- 实现 warp shuffle 版本，消除 volatile 共享内存规约的同步语义风险。
- 若目标是完整归约，应优化后续处理 262144 个局部和的阶段，而不是继续压缩第一轮指令。

## reduce_v6：warp shuffle 两级规约

### 优化方式

v6 保留 v5 的每线程双元素加载和相同 grid 配置，但不再把 256 个线程的中间值全部
写入共享内存。规约分为两个层次：

1. 每个 warp 使用 `__shfl_down_sync` 在寄存器之间规约 32 个线程的局部值。
2. 每个 warp 的 lane 0 把结果写入 `warp_results`，256 线程时共写入 8 个值。
3. 整个线程块执行一次 `__syncthreads()`，确保 8 个 warp 结果均已写入。
4. 第一个 warp 读取这些结果，其余 lane 使用 0，再执行一次 warp shuffle 规约。
5. 线程 0 将线程块局部和写入全局输出。

这样可以消除 v5 最后一个 warp 使用 volatile 共享内存通信的同步语义风险，并显著减少
共享内存访问与 bank conflict。v6 声明了 32 个 `float` 的静态共享内存，实际只使用
前 8 个元素。

启动配置：

```text
grid  = 262144 blocks
block = 256 threads
threads = 67108864
warps per block = 8
dynamic shared memory passed by test framework = 1024 bytes/block
static shared memory = 128 bytes/block
logical output = 262144 * 4 bytes = 1 MiB
```

v6 本身不需要动态共享内存，但当前统一启动函数仍传入 1024 字节。Nsight Compute
因此同时记录了 1024 字节动态共享内存和 128 字节静态共享内存；该配置的理论
occupancy 仍为 100%，没有形成实际资源限制。

### Nsight Compute 记录

本节的 v5 和 v6 数据来自加入 v6 后的同一轮 Full Set 采样，两份报告均为 45 passes，
SM 和 DRAM 频率均稳定在约 2.37 GHz 和 7.99 GHz。

| 指标 | v5 | v6 | 变化 |
| --- | ---: | ---: | ---: |
| Kernel Duration | 2.204832 ms | 2.206400 ms | +0.07% |
| Effective Bandwidth | 243.97 GB/s | 243.80 GB/s | -0.07% |
| Measured DRAM Traffic Rate | 249.50 GB/s | 249.42 GB/s | -0.03% |
| Elapsed Cycles | 5225381 | 5229075 | +0.07% |
| SM Frequency | 2.37 GHz | 2.37 GHz | 基本不变 |
| DRAM Frequency | 7.99 GHz | 7.99 GHz | 基本不变 |
| DRAM Bytes Read | 536.908416 MB | 537.123712 MB | +0.04% |
| DRAM Bytes Write | 13.191168 MB | 13.189120 MB | -0.02% |
| DRAM Throughput | 97.55% | 97.52% | -0.03 pp |
| Compute (SM) Throughput | 41.81% | 35.62% | -6.19 pp |
| Executed Instructions | 80740352 | 178782356 | +121.43% |
| Achieved Occupancy | 80.56% | 78.14% | -2.42 pp |
| Active Warps / SM | 38.67 | 37.51 | -3.00% |
| Eligible Warps / Scheduler | 0.19 | 0.53 | +175.21% |
| No Eligible | 84.18% | 63.47% | -20.71 pp |
| Avg. Not Predicated Threads / Warp | 27.34 | 28.33 | +3.62% |
| Shared Bank Conflicts | 224097 | 23096 | -89.69% |
| Shared Load Bank Conflicts | 1762 | 32 | -98.18% |
| Shared Store Bank Conflicts | 228046 | 19044 | -91.65% |
| Barrier Stall / Issued Instruction | 4.53 cycles | 1.63 cycles | -63.91% |
| Long Scoreboard / Issued Instruction | 45.75 cycles | 16.24 cycles | -64.50% |
| Barrier Stall Sample Share | 12.00% | 6.24% | -5.76 pp |
| Long Scoreboard Sample Share | 72.09% | 59.22% | -12.88 pp |
| Branch Efficiency | 100% | 100% | 不变 |

v5 共采集 122494 个 PC Sampling 样本，其中 barrier 为 14697，long scoreboard
为 88310；v6 共采集 122477 个样本，其中 barrier 为 7645，long scoreboard
为 72528。

Full Set 会在不同 replay pass 中采集共享内存总冲突和按 load/store 分类的冲突，
各 pass 之间存在轻微运行波动，因此分类值不要求与总值严格相加；本节主要比较同一
指标在 v5 和 v6 之间的数量级变化。

v6 的有效带宽和加速比：

```text
input bytes  = 134217728 * 4 = 536870912 bytes
output bytes = 262144 * 4    =   1048576 bytes
total bytes  =                      537919488 bytes

effective bandwidth = 537919488 / 0.002206400
                    = 243.80 GB/s

speedup vs v5 = 2.204832 / 2.206400
              = 0.9993x

speedup vs v0 = 7.445440 / 2.206400
              = 3.37x
```

### 配对复测

Full Set 中 v5 和 v6 只相差 0.07%，不足以证明存在稳定性能变化。因此额外进行了
四轮稳定频率下的配对采样，只收集 kernel 时间、SM 频率和 DRAM 频率：

| 稳定样本 | v5 | v6 | v6 相对 v5 |
| --- | ---: | ---: | ---: |
| 1 | 2.162240 ms | 2.187936 ms | +1.19% |
| 2 | 2.154144 ms | 2.154432 ms | +0.01% |
| 3 | 2.182816 ms | 2.155040 ms | -1.27% |
| 4 | 2.153696 ms | 2.155264 ms | +0.07% |
| 中位数 | 2.158192 ms | 2.155152 ms | -0.14% |

另有一轮 v6 降频到约 1.80 GHz SM 和 6.07 GHz DRAM，已从表中剔除。稳定样本的
逐轮差异在 `-1.27%` 到 `+1.19%` 之间，明显大于中位数的 0.14%，因此应把 v5 和
v6 判断为性能持平，而不是宣称 v6 获得了可重复加速。

### 正确性验证

v6 的 262144 个局部和通过 CPU 参考结果校验：

```text
[PASS] reduce_v6 N=2^27: blocks=262144, failed_blocks=0,
expected_total=18.00000016,
actual_total=18.00001478
```

v6 的浮点累加顺序与 v5 不同，因此最终总和存在正常的单精度舍入差异；逐块结果均在
测试容差内。

### 优化结果分析

- warp shuffle 达到了减少共享内存通信的目标。Shared Bank Conflicts 降低 89.69%，
  barrier 和 long scoreboard 的平均等待分别降低 63.91% 和 64.50%，Eligible Warps
  也从 0.19 增加到 0.53。
- Full Set 中 v6 比 v5 慢 0.07%，而稳定配对中位数快 0.14%；两者都远小于逐轮波动。
  因此 v6 的性能结论是与 v5 持平，但同步语义更明确、共享内存访问更干净。
- v6 的执行指令数反而增加 121.43%。一个合理解释是两段 warp shuffle 都使用运行时
  循环：第一次由全部 256 个线程执行，第二次再由第一个 warp 执行；每轮包含 shuffle、
  加法、步长更新、比较和跳转。相比之下，v5 的块大小是编译期常量，规约阶段已显式展开。
- 指令增加没有显著拉长时间，因为两个版本的 DRAM Throughput 都约为 97.5%，第一轮
  规约已经受显存带宽限制。v6 的逻辑有效带宽为 243.80 GB/s，达到理论显存带宽
  256.03 GB/s 的 95.22%。
- Achieved Occupancy 从 80.56% 降到 78.14%，但理论 occupancy 仍为 100%，寄存器数
  仍为 16/线程，驻留限制仍首先来自每个线程块的 warp 数。这个动态差异不是主要瓶颈。
- Compute (SM) Throughput 从 41.81% 降到 35.62% 不代表计算能力退化。v5 的共享内存
  LSU 活动较多；v6 移除大部分共享内存事务后，最繁忙计算管线的占比自然下降。

### 启动条件与下一步

当前实现的 shuffle 循环使用完整掩码 `0xffffffffu`，因此要求参与循环的 warp 中
32 个 lane 都执行对应指令。当前 `block = 256` 是 warp size 的整数倍，第二级规约也由
完整的第一个 warp 执行，所以该条件成立。若以后测试非 32 整数倍的 block 大小，需要
重新计算 active mask 和 warp 数，不能直接复用当前启动方式。

- 可尝试展开固定的五轮 shuffle，验证能否降低当前增加的指令数。
- 主机端可为 v6 传入 0 字节动态共享内存，使启动配置准确反映 kernel 的实际需求。
- 第一轮读取已经接近 DRAM 上限，后续更有价值的方向是完成多轮归约并测量端到端时间。

## reduce_v7：float4 向量化加载与 grid 减半

### 优化方式

v7 保留 v6 的两级 warp shuffle 规约，把全局内存读取改为 `float4`：

1. 将输入指针转换为 `float4 *`，每个线程通过一次向量 load 读取 4 个 `float`。
2. 使用 grid-stride loop，使线程在 grid 较小时仍可处理多个 `float4`。
3. 使用独立的标量循环处理不能被 4 整除的尾部元素。
4. 每个 warp 用 shuffle 规约线程局部和，再由第一个 warp 规约各 warp 的结果。

当前 `N = 2^27` 可以被 4 整除，因此标量尾部循环不执行。测试使用 131072 个 block，
使 `gridDim.x * blockDim.x = N / 4`，所以每个线程恰好读取一个 `float4`。`cudaMalloc`
返回的设备地址满足 `float4` 的 16 字节对齐要求。

启动配置：

```text
grid  = 131072 blocks
block = 256 threads
threads = 33554432
warps per block = 8
waves per SM = 910.22
dynamic shared memory passed by test framework = 1024 bytes/block
static shared memory = 128 bytes/block
logical output = 131072 * 4 bytes = 0.5 MiB
```

v7 自身与 v6 一样不需要动态共享内存，但统一测试框架仍传入 1024 字节。v7 使用
20 个寄存器/线程，理论 occupancy 仍为 100%，寄存器限制允许每个 SM 驻留 10 个
线程块，实际首先受每个线程块 8 个 warp、每个 SM 最多 48 个 warp 的限制。

### Nsight Compute 记录

本节的 v6 和 v7 来自加入 v7 后的同一轮 Full Set 采样，均为 45 passes，SM 和
DRAM 频率分别稳定在约 2.37 GHz 和 7.99 GHz。

| 指标 | v6 | v7 | 变化 |
| --- | ---: | ---: | ---: |
| Kernel Duration | 2.158944 ms | 2.160544 ms | +0.07% |
| Effective Bandwidth | 249.16 GB/s | 248.73 GB/s | -0.17% |
| Measured DRAM Traffic Rate | 249.83 GB/s | 250.08 GB/s | +0.10% |
| Elapsed Cycles | 5116645 | 5120368 | +0.07% |
| SM Frequency | 2.37 GHz | 2.37 GHz | 基本不变 |
| DRAM Frequency | 7.99 GHz | 7.99 GHz | 基本不变 |
| Grid Size | 262144 | 131072 | -50.00% |
| Threads | 67108864 | 33554432 | -50.00% |
| Waves / SM | 1820.44 | 910.22 | -50.00% |
| DRAM Bytes Read | 536.907008 MB | 536.941824 MB | +0.01% |
| DRAM Bytes Write | 2.472448 MB | 3.368704 MB | +36.25% |
| DRAM Throughput | 97.68% | 97.78% | +0.10 pp |
| Compute (SM) Throughput | 36.40% | 18.37% | -18.02 pp |
| Executed Instructions | 178782208 | 90309566 | -49.49% |
| Global Load Instructions | 4194304 | 1048576 | -75.00% |
| Global Load Requests | 4194304 | 1048576 | -75.00% |
| Global Load Sectors | 16777216 | 16777216 | 不变 |
| Data Bytes / Global Load Sector | 32 bytes | 32 bytes | 不变 |
| Branch Instructions | 19660800 | 15073280 | -23.33% |
| Registers / Thread | 16 | 20 | +4 |
| Achieved Occupancy | 79.25% | 87.70% | +8.45 pp |
| Active Warps / SM | 38.04 | 42.10 | +10.66% |
| Eligible Warps / Scheduler | 0.53 | 0.23 | -56.77% |
| No Eligible | 63.47% | 81.50% | +18.03 pp |
| Avg. Not Predicated Threads / Warp | 28.33 | 29.81 | +5.22% |
| Shared Bank Conflicts | 22966 | 12873 | -43.95% |
| Barrier Stall / Issued Instruction | 1.43 cycles | 7.89 cycles | +452.98% |
| Long Scoreboard / Issued Instruction | 15.97 cycles | 43.54 cycles | +172.66% |
| Barrier Stall Sample Share | 6.80% | 10.82% | +4.01 pp |
| Long Scoreboard Sample Share | 58.64% | 76.09% | +17.46 pp |
| Branch Efficiency | 100% | 100% | 不变 |

v6 共采集 120836 个 PC Sampling 样本，其中 barrier 为 8219、long scoreboard
为 70853；v7 共采集 126284 个样本，其中 barrier 为 13659、long scoreboard
为 96091。

逻辑输出字节数在 v7 中减半，但表中的 DRAM Bytes Write 来自不同 replay pass 下的
实际显存事务，会受到缓存写回和测量波动影响，不能直接当作 kernel 的逻辑 store 字节数。
有效带宽仍按确定的输入和逻辑输出工作量计算。

v7 的有效带宽和加速比：

```text
input bytes  = 134217728 * 4 = 536870912 bytes
output bytes = 131072 * 4    =    524288 bytes
total bytes  =                      537395200 bytes

effective bandwidth = 537395200 / 0.002160544
                    = 248.73 GB/s

speedup vs v6 = 2.158944 / 2.160544
              = 0.9993x

speedup vs v0 = 7.441696 / 2.160544
              = 3.44x
```

### 独立进程复测

最初尝试在同一进程中依次采集 v6 和 v7，但后执行的 v7 多次降频，无法公平比较。
因此改为每次只分析一个版本，并只保留约 2.37 GHz SM、7.99 GHz DRAM 的样本：

| 样本 | v6 | v7 | v7 相对 v6 |
| --- | ---: | ---: | ---: |
| 1 | 2.167712 ms | 2.155520 ms | -0.56% |
| 2 | 2.154912 ms | 2.166944 ms | +0.56% |
| 3 | 2.164704 ms | 2.167264 ms | +0.12% |
| 4 | 2.156768 ms | 2.154304 ms | -0.11% |
| 中位数 | 2.160736 ms | 2.161232 ms | +0.02% |

另有一份 v6 样本降频到约 1.99 GHz SM 和 6.73 GHz DRAM，已剔除。稳定样本中
v7 相对 v6 的差异在 `-0.56%` 到 `+0.56%` 之间，中位数只慢 0.02%，因此两者应
判断为性能持平。

### 正确性验证

v7 的 131072 个块级结果全部通过 CPU 参考校验：

```text
[PASS] reduce_v7 N=2^27: blocks=131072, failed_blocks=0,
expected_total=18.00000016,
actual_total=18.05136108
```

打印出的总和是把已经舍入为 `float` 的 131072 个块级结果再用 `double` 相加，仅作为
诊断信息。v7 改变了块大小和块内加法顺序，许多很小且方向一致的块级舍入误差会在总和
中累积；正确性判定仍以每个块级结果是否落在明确容差内为准，本轮失败块数为 0。

### 优化结果分析

- `float4` 确实生成了更少的全局 load 指令。Global Load Instructions 和 Requests
  都从 4194304 降到 1048576，减少 75%；总执行指令减少 49.49%，grid、线程数和
  waves 数均减半。
- Global Load Sectors 仍为 16777216，且每个 sector 的有效数据均为 32 字节。v6
  已经实现完全合并的标量读取，v7 的向量化没有减少必须从显存传输的 sector 数，
  只减少了发出这些访问所需的指令和请求。
- Full Set 中 v7 比 v6 慢 0.07%，独立稳定样本中位数慢 0.02%，均远小于运行波动。
  因此 v7 达到了降低指令开销的目标，但没有获得可重复的 kernel 时间加速。
- v7 的 DRAM Throughput 达到 97.78%，逻辑有效带宽为 248.73 GB/s，达到理论显存
  带宽 256.03 GB/s 的 97.15%。指令减少后仍需读取相同的 512 MiB 输入，显存带宽
  继续决定总时间。
- 总指令减少后，每条已发射指令分摊到的 barrier 和 long scoreboard 等待反而增加；
  Eligible Warps 从 0.53 降到 0.23，No Eligible 升至 81.50%。这不表示 `float4`
  读取不合并，而是说明更少的指令无法掩盖相同的 DRAM 延迟。
- Achieved Occupancy 升至 87.70%，但没有转化为更短时间。当前已有足够并行度填满
  DRAM 管线，继续提高 occupancy 不能突破显存带宽上限。
- v7 的分支指令只减少 23.33%，少于线程数的 50% 降幅，因为 grid-stride 主循环和
  尾部循环引入了额外的循环比较与跳转。当前尾部循环不执行，但控制指令仍存在。

### 启动条件与下一步

- `float4` 输入地址必须满足 16 字节对齐；当前 `cudaMalloc` 返回的地址满足要求。
- 当前实验的 `N` 可被 4 整除。若改变输入规模，必须继续验证标量尾部处理。
- 当前 grid 使每个线程只执行一次向量读取；若减少 grid，单个线程将处理多个 `float4`，
  需要重新测量寄存器压力、指令级并行和性能。
- 第一轮 kernel 已接近显存带宽极限。下一步更值得优化第二轮及后续局部和规约，或直接
  测量把 131072 个局部和归约为单个标量的端到端时间。

## 新版本记录模板

复制本节并将版本号替换为实际 kernel 名称。一次尽量只修改一个主要因素，使性能变化
能够归因到明确的优化。

### reduce_vX：优化名称

**优化目标**

说明上一版本的哪个 Nsight Compute 指标表明存在问题。

**代码变化**

- 具体修改：
- 保持不变的部分：
- 预期改善的指标：
- 可能引入的代价：

**启动配置**

```text
grid  =
block =
dynamic shared memory =
```

**Nsight Compute 结果**

| 指标 | 上一版本 | 当前版本 | 变化 |
| --- | ---: | ---: | ---: |
| Kernel Duration |  |  |  |
| Effective Bandwidth |  |  |  |
| DRAM Throughput |  |  |  |
| SM Throughput |  |  |  |
| Achieved Occupancy |  |  |  |
| Registers / Thread |  |  |  |
| Branch Efficiency |  |  |  |
| Shared Bank Conflicts |  |  |  |
| Barrier Stall |  |  |  |

**结论**

- 实测加速比：
- 优化是否达到预期：
- 性能变化原因：
- 当前主要瓶颈：
- 下一步优化：

## 实验纪律

- 每个版本必须先通过 CPU 参考结果校验，再记录性能。
- 所有版本必须处理相同的 `N = 2^27` 输入。
- 比较时必须保持编译参数、线程块大小和 GPU 环境一致，除非它们就是实验变量。
- 第一次运行可能包含上下文初始化开销，不应作为 kernel 性能数据。
- Nsight Compute 会重放 kernel，分析时长不能当作 kernel 的实际执行时间。
- 报告文件应使用版本名，例如 `reduce_v0.ncu-rep`，避免覆盖历史结果。
- 若版本改变了输出语义或全局内存访问量，必须在记录中明确说明。

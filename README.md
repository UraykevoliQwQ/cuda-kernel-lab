# cuda-kernel-lab

CUDA kernel 优化实验仓库。当前以 `float` 数组规约求和为例，从共享内存基线逐步优化到 warp shuffle 和 `float4` 向量化加载，结合 CPU 参考校验与 Nsight Compute 分析性能变化。

## 实验目标与条件

- 比较 v0–v7 的寻址、同步、规约和全局内存加载策略，理解性能瓶颈如何从指令与控制流转向显存带宽。
- 固定输入 `N = 2^27`（134217728 个 `float`，512 MiB）、每块 256 threads、编译优化 `-O3`。
- 已记录环境：RTX 4060 Laptop GPU（计算能力 8.9）、CUDA Runtime 13.3、Nsight Compute 2026.2.0.0；理论显存带宽 256.03 GB/s。详见 [GPU 硬件信息](GPU_HARDWARE.md)。

当前各 kernel 每个线程块输出一个局部和，尚未通过后续 GPU 规约生成整个数组的单个标量；性能数据仅衡量这一轮 kernel。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| [GPU_HARDWARE.md](GPU_HARDWARE.md) | CUDA 环境与 GPU 硬件参数 |
| [kernels/reduce_sum/reduce_sum.cu](kernels/reduce_sum/reduce_sum.cu) | v0–v7 实现及 CPU 参考校验 |
| [kernels/reduce_sum/REDUCE_SUM_OPTIMIZATION.md](kernels/reduce_sum/REDUCE_SUM_OPTIMIZATION.md) | 优化过程、性能指标与复测分析 |

## 编译与运行

需要支持 CUDA 的 NVIDIA GPU、提供 `nvcc` 的 CUDA 开发环境；性能分析另需 Nsight Compute。

在 `kernels/reduce_sum/` 目录中执行仓库已有命令：

```bash
nvcc -O3 -lineinfo -std=c++17 reduce_sum.cu -o reduce_sum
./reduce_sum
```

程序固定使用上述输入规模，依次运行 v0–v7，无需命令行参数。

## 正确性验证

运行同一程序即可完成校验：

- 所有版本使用相同输入；CPU 使用 `double` 累加对应输入分段，逐块比较 GPU 输出。
- 每块允许误差为 `1e-5 * max(1, abs(expected_block))`。
- 每个版本应输出 `[PASS]`、`failed_blocks=0`，最终输出 `All tests passed.`，并以成功状态退出。

仓库记录的 v0–v7 均通过上述校验。打印的 `expected_total` 和 `actual_total` 是诊断信息；浮点加法顺序和块级舍入会影响总和，不能把这些输出理解为总和已通过独立精度校验。

## 性能结论

以下摘自实验分析的最终 v0–v7 横向复测表，固定条件同上。耗时仅为 kernel 执行时间，不含初始化、内存分配、数据复制或 CPU 校验。

| 版本 | 核心优化 | Kernel 时间 | 有效带宽 | 相对 v0 加速 |
| --- | --- | ---: | ---: | ---: |
| v0 | 取模选择活跃线程 | 7.441696 ms | 72.43 GB/s | 1.00x |
| v1 | 连续线程计算交错共享内存下标 | 4.652768 ms | 115.84 GB/s | 1.60x |
| v2 | 逆序步长连续寻址规约 | 4.516288 ms | 119.34 GB/s | 1.65x |
| v3 | 每线程预加两个输入 | 2.369344 ms | 227.03 GB/s | 3.14x |
| v4 | 最后一个 warp 手动展开规约 | 2.165216 ms | 248.44 GB/s | 3.44x |
| v5 | 模板参数编译期展开块级规约 | 2.174720 ms | 247.35 GB/s | 3.42x |
| v6 | warp shuffle 两级规约 | 2.158944 ms | 249.16 GB/s | 3.45x |
| v7 | float4 向量化加载与 grid 减半 | 2.160544 ms | 248.73 GB/s | 3.44x |

v0 → v7 从 7.441696 ms 降至 2.160544 ms，约 **3.44x** 加速。v4 起已接近显存带宽上限；v5–v7 的小幅时间差不足以证明稳定加速。v6 明确了 warp 内通信的同步语义；v7 相对 v6 减少 75% 的全局 load 指令，但未减少输入传输量，独立复测仍表现为性能持平。

有效带宽按 `(N * sizeof(float) + grid_size * sizeof(float)) / kernel_time` 计算。v0–v2、v3–v6、v7 分别输出 524288、262144、131072 个局部和。详细数据、同步语义边界与复测方法见 [优化与性能分析](kernels/reduce_sum/REDUCE_SUM_OPTIMIZATION.md)。

## Nsight Compute 分析

在同一目录中采集 v0 完整报告并打开：

```bash
ncu --set full \
  --kernel-name regex:reduce_v0 \
  --export reduce_v0 \
  --force-overwrite \
  ./reduce_sum

ncu-ui reduce_v0.ncu-rep
```

其他版本的采样、核心指标命令及结果解释见 [实验文档](kernels/reduce_sum/REDUCE_SUM_OPTIMIZATION.md#编译与分析命令)。分析时应保持频率稳定；Nsight Compute 的重放分析时长不能当作 kernel 执行时间。

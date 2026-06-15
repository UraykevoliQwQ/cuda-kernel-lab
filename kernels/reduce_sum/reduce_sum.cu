#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * v0：使用交错寻址建立共享内存规约基线。
 *
 * 每个线程先把一个输入元素装入共享内存；最后一个线程块中的越界线程写入 0，
 * 因而不会访问 input 边界之外。规约步长从 1 开始逐轮翻倍，并通过取模选择
 * 每个分组中负责累加的线程。
 *
 * 启动时必须为每个线程提供一个 float 的动态共享内存，例如：
 * reduce_v0<<<blocks, threads, threads * sizeof(float)>>>(input, output, n);
 *
 * output 必须至少容纳 blocks 个 float。此 kernel 每个线程块只输出一个局部和，
 * 不会在一次启动中继续把所有线程块的局部和归并成单个标量。
 */
__global__ void reduce_v0(float *input, float *output, int n)
{
    extern __shared__ float smem[];

    int tid;
    int gid;

    tid = threadIdx.x;
    gid = blockIdx.x * blockDim.x + tid;

    // 所有线程都必须写入自己的共享内存槽位。越界线程写入加法单位元 0，
    // 使后续规约无需额外分支，并保证最后一个不完整线程块也能正确计算。
    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    // 步长每轮翻倍，只有每个分组的首线程执行加法。每轮都必须同步，
    // 确保上一轮写入已对整个线程块可见后，下一轮才读取共享内存。
    for (int s = 1; s < blockDim.x; s <<= 1)
    {
        if (tid % (s << 1) == 0)
        {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    // 线程 0 持有当前线程块的局部和。不同线程块写入不同下标，
    // 因而不需要 atomicAdd，但调用者需要进一步处理这些局部结果。
    if (tid == 0)
    {
        output[blockIdx.x] = smem[0];
    }
}

/**
 * v1 相对 v0：移除使用线程下标取模选择活跃线程的方式。
 *
 * 每轮让线程块前部的连续线程参与计算，并通过 idx = 2 * tid * s 定位需要合并的
 * 共享内存元素。这样可减少取模和控制流开销，但交错的共享内存地址可能产生
 * bank conflict。其余输入加载、同步次数和输出语义与 v0 相同。
 */
__global__ void reduce_v1(float *input, float *output, int n)
{
    extern __shared__ float smem[];

    int tid;
    int gid;

    tid = threadIdx.x;
    gid = blockIdx.x * blockDim.x + tid;

    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    // 连续线程替代 v0 中按取模结果分散的活跃线程；idx 仍形成交错共享内存访问。
    for (int s = 1; s < blockDim.x; s <<= 1)
    {
        int idx = 2 * tid * s;
        if (idx < blockDim.x)
            smem[idx] += smem[idx + s];
        __syncthreads();
    }

    if (tid == 0)
    {
        output[blockIdx.x] = smem[0];
    }
}

/**
 * v2 相对 v1：将共享内存规约改为逆序步长和连续寻址。
 *
 * 步长从 blockDim.x / 2 开始逐轮减半，连续线程 tid 读取 smem[tid] 和
 * smem[tid + s]。两个访问区间内部均连续，避免 v1 的交错地址造成高阶
 * bank conflict。其余输入加载、块级同步和输出语义与 v1 相同。
 */
__global__ void reduce_v2(float *input, float *output, int n)
{
    extern __shared__ float smem[];

    int tid;
    int gid;

    tid = threadIdx.x;
    gid = blockIdx.x * blockDim.x + tid;

    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    // 逆序步长使每轮活跃线程及其源、目标共享内存地址保持连续。
    for (int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
            smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0)
    {
        output[blockIdx.x] = smem[0];
    }
}

/**
 * v3 相对 v2：每个线程在进入共享内存规约前先累加两个全局内存元素。
 *
 * 每个线程块因此覆盖 2 * blockDim.x 个输入，grid 大小和块级局部和数量均可减半。
 * 两个输入下标分别执行边界检查，保证最后一个不完整线程块不会越界。共享内存中的
 * 逆序步长规约与 v2 相同。
 */
__global__ void reduce_v3(float *input, float *output, int n)
{
    extern __shared__ float smem[];

    int tid;
    int gid;

    tid = threadIdx.x;
    gid = blockIdx.x * blockDim.x * 2 + tid;

    // 先在寄存器中合并两个输入，使每个线程块处理的元素数量相对 v2 翻倍。
    float val = 0.0f;
    if (gid < n)
        val += input[gid];
    if (gid + blockDim.x < n)
        val += input[gid + blockDim.x];
    smem[tid] = val;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
            smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0)
    {
        output[blockIdx.x] = smem[0];
    }
}

/**
 * 展开 v4 最后一个 warp 内的六级共享内存规约。
 *
 * volatile 强制每条语句实际读取和写入共享内存，避免编译器把跨线程依赖的值长期
 * 保存在寄存器中。volatile 本身不提供线程同步；该实现用于记录当前优化实验，
 * 后续可使用 __syncwarp 或 warp shuffle 明确表达 warp 内通信。
 */
__device__ void warpReduce(volatile float *smem, int tid)
{
    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid + 8];
    smem[tid] += smem[tid + 4];
    smem[tid] += smem[tid + 2];
    smem[tid] += smem[tid + 1];
}

/**
 * v4 相对 v3：只用块级规约处理到 64 个中间值，最后六轮由一个 warp 手动展开。
 *
 * 循环条件 s > 32 使块级规约只执行 s = 128 和 s = 64 两轮，随后前 32 个线程
 * 合并剩余 64 个值。这样减少后六轮块级同步，也避免其他 warp 继续执行已被谓词
 * 屏蔽的循环指令。输入加载、grid 大小和输出语义与 v3 相同。
 */
__global__ void reduce_v4(float *input, float *output, int n)
{
    extern __shared__ float smem[];

    int tid;
    int gid;

    tid = threadIdx.x;
    gid = blockIdx.x * blockDim.x * 2 + tid;

    float val = 0.0f;
    if (gid < n)
        val += input[gid];
    if (gid + blockDim.x < n)
        val += input[gid + blockDim.x];
    smem[tid] = val;
    __syncthreads();

    // 块级规约在剩余 64 个值时停止，省去 v3 最后六轮的 __syncthreads()。
    for (int s = blockDim.x >> 1; s > 32; s >>= 1)
    {
        if (tid < s)
            smem[tid] += smem[tid + s];
        __syncthreads();
    }

    // 只让第一个 warp 完成剩余规约，其他 warp 不再参与后续循环控制。
    if (tid < 32)
        warpReduce(smem, tid);

    if (tid == 0)
    {
        output[blockIdx.x] = smem[0];
    }
}

template <int BLOCK_SIZE>
__global__ void reduce_v5(float *input, float *output, int n)
{
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = BLOCK_SIZE * blockIdx.x * 2 + tid;

    float val = 0.0f;
    if (gid < n)
        val += input[gid];
    if (gid + BLOCK_SIZE < n)
        val += input[gid + BLOCK_SIZE];
    smem[tid] = val;
    __syncthreads();

    if (BLOCK_SIZE >= 512)
    {
        if (tid < 256)
            smem[tid] += smem[tid + 256];
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256)
    {
        if (tid < 128)
            smem[tid] += smem[tid + 128];
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128)
    {
        if (tid < 64)
            smem[tid] += smem[tid + 64];
        __syncthreads();
    }

    if (tid < 32)
    {
        volatile float *vsmem = smem;
        if (BLOCK_SIZE >= 64)
            vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }
    if (tid == 0)
        output[blockIdx.x] = smem[0];
}

__global__ void reduce_v6(float *input, float *output, int n)
{
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x * 2 + tid;

    float val = 0.0f;
    if (gid < n)
        val += input[gid];
    if (gid + blockDim.x < n)
        val += input[gid + blockDim.x];

    for (int s = warpSize >> 1; s > 0; s >>= 1)
    {
        val += __shfl_down_sync(0xffffffffu, val, s);
    }

    __shared__ float warp_results[32];

    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;

    if (lane == 0)
    {
        warp_results[warpId] = val;
    }

    __syncthreads();

    int num_warps = blockDim.x / warpSize;
    if (warpId == 0)
    {
        val = lane < num_warps ? warp_results[lane] : 0;
        for (int s = warpSize >> 1; s > 0; s >>= 1)
        {
            val += __shfl_down_sync(0xffffffffu, val, s);
        }
    }
    if (tid == 0)
        output[blockIdx.x] = val;
}

__global__ void reduce_v7(float *input, float *output, int n)
{
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;

    float4 *input4 = reinterpret_cast<float4 *>(input);

    int n4 = n / 4;

    float val = 0.0f;
    for (int idx = blockIdx.x * blockDim.x + tid; idx < n4; idx += gridDim.x * blockDim.x)
    {
        float4 data = input4[idx];
        val += data.x + data.y + data.z + data.w;
    }

    int tail_start = n4 * 4;
    for (int idx = tail_start + blockIdx.x * blockDim.x + tid; idx < n; idx += gridDim.x * blockDim.x)
    {
        val += input[idx];
    }

    for (int s = warpSize >> 1; s > 0; s >>= 1)
        val += __shfl_down_sync(0xffffffff, val, s);

    __shared__ float warp_results[32];
    if (lane == 0)
        warp_results[wid] = val;

    __syncthreads();

    if (wid == 0)
    {
        val = (lane < blockDim.x / warpSize) ? warp_results[lane] : 0.0f;
        for (int s = warpSize >> 1; s > 0; s >>= 1)
            val += __shfl_down_sync(0xffffffff, val, s);
    }

    if (tid == 0)
        output[blockIdx.x] = val;
}

/**
 * 标识当前测试需要启动的 reduce_sum kernel 版本。
 *
 * 测试框架使用枚举而不是字符串判断版本，避免名称拼写错误导致启动错误的 kernel。
 * 新增优化版本时，应在此处添加枚举值，并同步扩展 launch_reduce_kernel 的分支。
 */
enum reduce_version
{
    REDUCE_VERSION_V0,
    REDUCE_VERSION_V1,
    REDUCE_VERSION_V2,
    REDUCE_VERSION_V3,
    REDUCE_VERSION_V4,
    REDUCE_VERSION_V5,
    REDUCE_VERSION_V6,
    REDUCE_VERSION_V7
};

/**
 * 启动指定版本的 reduce_sum kernel。
 *
 * 所有版本使用相同的输入、输出和动态共享内存约定，因此可以共享测试和资源管理代码。
 * 该函数只负责选择 kernel 并提交启动；CUDA 启动错误和异步执行错误仍由调用者通过
 * cudaGetLastError 和 cudaDeviceSynchronize 分别检查。
 */
static void launch_reduce_kernel(enum reduce_version version,
                                 int blocks,
                                 int threads_per_block,
                                 size_t shared_memory_size,
                                 float *d_input,
                                 float *d_output,
                                 int n)
{
    switch (version)
    {
    case REDUCE_VERSION_V0:
        reduce_v0<<<blocks, threads_per_block, shared_memory_size>>>(
            d_input,
            d_output,
            n);
        break;
    case REDUCE_VERSION_V1:
        reduce_v1<<<blocks, threads_per_block, shared_memory_size>>>(
            d_input,
            d_output,
            n);
        break;
    case REDUCE_VERSION_V2:
        reduce_v2<<<blocks, threads_per_block, shared_memory_size>>>(
            d_input,
            d_output,
            n);
        break;
    case REDUCE_VERSION_V3:
        reduce_v3<<<blocks, threads_per_block, shared_memory_size>>>(
            d_input,
            d_output,
            n);
        break;
    case REDUCE_VERSION_V4:
        reduce_v4<<<blocks, threads_per_block, shared_memory_size>>>(
            d_input,
            d_output,
            n);
        break;
    case REDUCE_VERSION_V5:
        // 当前测试统一使用 256 个线程，因此模板参数必须与实际 blockDim.x 一致。
        // 若二者不同，v5 的全局内存索引和分阶段规约条件将不再匹配启动配置。
        reduce_v5<256><<<blocks, threads_per_block, shared_memory_size>>>(
            d_input,
            d_output,
            n);
        break;
    case REDUCE_VERSION_V6:
        // v6 使用 warp shuffle 和静态共享内存，不依赖动态共享内存参数；
        // 为保持所有版本的统一启动接口，仍传入 shared_memory_size。
        reduce_v6<<<blocks, threads_per_block, shared_memory_size>>>(
            d_input,
            d_output,
            n);
        break;
    case REDUCE_VERSION_V7:
        // v7 使用 float4 向量化读取和静态共享内存。当前 N=2^27 且 block 为 256，
        // grid 按每线程四个元素计算，使每个线程恰好执行一次主循环向量读取。
        reduce_v7<<<blocks, threads_per_block, shared_memory_size>>>(
            d_input,
            d_output,
            n);
        break;
    default:
        fprintf(stderr, "unknown reduce kernel version: %d\n", version);
        exit(EXIT_FAILURE);
    }
}

/**
 * 返回一个线程块负责处理的输入元素数量。
 *
 * v0-v2 中每个线程只读取一个元素，因此每块覆盖 blockDim.x 个输入。v3-v6 中每个
 * 线程先累加相距 blockDim.x 的两个元素，因此每块覆盖 2 * blockDim.x 个输入。v7
 * 使用 float4 向量化读取；当前测试让每个线程读取一个 float4，因此每块覆盖
 * 4 * blockDim.x 个输入。该信息同时决定 grid 大小和 CPU 参考结果的分段范围。
 */
static int input_values_per_block(enum reduce_version version,
                                  int threads_per_block)
{
    if (version == REDUCE_VERSION_V3 ||
        version == REDUCE_VERSION_V4 ||
        version == REDUCE_VERSION_V5 ||
        version == REDUCE_VERSION_V6)
    {
        return threads_per_block * 2;
    }
    if (version == REDUCE_VERSION_V7)
    {
        return threads_per_block * 4;
    }

    return threads_per_block;
}

/**
 * 检查 CUDA Runtime API 的返回值。
 *
 * CUDA 的许多错误会延迟到同步操作时才报告，因此测试既检查每个 API 的返回值，
 * 也在 kernel 启动后检查启动状态并显式同步，避免把执行错误误判为数值误差。
 */
static void check_cuda(cudaError_t error, const char *operation)
{
    if (error != cudaSuccess)
    {
        fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

/**
 * 分配指定数量的主机端 float，并检查分配是否成功。
 *
 * 返回的内存归调用者所有，使用结束后必须调用 free。测试不会请求零字节内存，
 * 因为空输入不会产生任何线程块输出。
 */
static float *allocate_host_values(int count)
{
    float *values;

    values = (float *)malloc((size_t)count * sizeof(float));
    if (values == NULL)
    {
        fprintf(stderr, "malloc failed for %d float values\n", count);
        exit(EXIT_FAILURE);
    }

    return values;
}

/**
 * 计算一个线程块应产生的 CPU 参考结果。
 *
 * begin 是当前线程块对应的首元素下标，最多累加 values_per_block 个元素。
 * 使用 double 累加可降低参考结果自身的舍入误差；最后与 GPU 的 float 结果按容差比较。
 */
static double reduce_block_cpu(const float *input,
                               int n,
                               int begin,
                               int values_per_block)
{
    int end;
    int i;
    double result;

    end = begin + values_per_block;
    if (end > n)
    {
        end = n;
    }

    result = 0.0;
    for (i = begin; i < end; ++i)
    {
        result += (double)input[i];
    }

    return result;
}

/**
 * 执行一个完整的 reduce_sum 正确性测试。
 *
 * 当前 reduce kernel 不会直接生成整个数组的最终和，而是为每个线程块生成一个局部和。
 * 测试逐项验证这些局部和，同时再累加它们验证总和。n 为 0 时允许 input 为 NULL，
 * 此时不能启动零维网格，测试直接把结果定义为零。
 */
static int run_test(const char *name,
                    enum reduce_version version,
                    const float *input,
                    int n)
{
    const int threads_per_block = 256;
    int blocks;
    int values_per_block;
    size_t input_size;
    size_t output_size;
    size_t shared_memory_size;
    float *d_input;
    float *d_output;
    float *gpu_block_sums;
    double expected_total;
    double gpu_total;
    int passed;
    int block;
    int failed_blocks;

    // 负长度没有合法含义；非空输入必须提供有效的主机端数组。
    if (n < 0 || (n > 0 && input == NULL))
    {
        fprintf(stderr, "[FAIL] %s: invalid input\n", name);
        return 0;
    }

    // 空输入没有线程块输出。单独处理该情况，也避免 cudaMalloc 请求零字节以及
    // CUDA kernel 使用 blocks=0 的非法启动配置。
    if (n == 0)
    {
        printf("[PASS] %s: expected=0, actual=0\n", name);
        return 1;
    }

    // v0-v2 每块处理 blockDim.x 个输入，v3-v6 每块处理 2 * blockDim.x 个输入，
    // v7 每块处理 4 * blockDim.x 个输入。使用统一的每块输入数量向上取整；
    // 对当前 N=2^27，v7 的输入可被 1024 整除，不会进入标量尾部处理循环。
    values_per_block = input_values_per_block(version, threads_per_block);
    blocks = (n + values_per_block - 1) / values_per_block;
    input_size = (size_t)n * sizeof(float);
    output_size = (size_t)blocks * sizeof(float);
    shared_memory_size = (size_t)threads_per_block * sizeof(float);
    d_input = NULL;
    d_output = NULL;
    gpu_block_sums = allocate_host_values(blocks);

    check_cuda(cudaMalloc((void **)&d_input, input_size),
               "cudaMalloc(d_input)");
    check_cuda(cudaMalloc((void **)&d_output, output_size),
               "cudaMalloc(d_output)");
    check_cuda(cudaMemcpy(d_input,
                          input,
                          input_size,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(d_input)");

    // 当前 kernel 会覆盖每个有效的 output 元素，本身不依赖输出初值。仍将输出清零，
    // 可以使测试行为确定，并防止未来修改启动范围后把未写入的显存误当成有效结果。
    check_cuda(cudaMemset(d_output, 0, output_size),
               "cudaMemset(d_output)");

    // 动态共享内存必须至少容纳每个线程的一个 float。若遗漏第三个启动参数，
    // extern __shared__ 数组将没有可用空间，kernel 会发生非法内存访问。
    launch_reduce_kernel(version,
                         blocks,
                         threads_per_block,
                         shared_memory_size,
                         d_input,
                         d_output,
                         n);

    check_cuda(cudaGetLastError(), name);
    check_cuda(cudaDeviceSynchronize(), name);
    check_cuda(cudaMemcpy(gpu_block_sums,
                          d_output,
                          output_size,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(gpu_block_sums)");

    // 设备资源在结果复制完成后即可释放；主机端输出还要用于逐块比较。
    check_cuda(cudaFree(d_input), "cudaFree(d_input)");
    check_cuda(cudaFree(d_output), "cudaFree(d_output)");

    expected_total = 0.0;
    gpu_total = 0.0;
    passed = 1;
    failed_blocks = 0;

    for (block = 0; block < blocks; ++block)
    {
        double expected_block;
        double actual_block;
        double absolute_error;
        double tolerance;

        expected_block = reduce_block_cpu(input,
                                          n,
                                          block * values_per_block,
                                          values_per_block);
        actual_block = (double)gpu_block_sums[block];
        absolute_error = fabs(actual_block - expected_block);

        // 规约改变了浮点加法顺序，因此结果不应与 CPU 顺序累加做精确相等比较。
        // 相对容差随结果规模增长，1e-5 的绝对下限用于处理期望值接近零的情况。
        tolerance = 1e-5 * fmax(1.0, fabs(expected_block));
        if (absolute_error > tolerance)
        {
            // 大规模输入可能同时暴露大量错误块。只打印前 10 个详细样本，避免日志
            // 淹没最终统计；所有线程块仍会继续比较，failed_blocks 记录完整失败数量。
            if (failed_blocks < 10)
            {
                fprintf(stderr,
                        "[FAIL] %s block %d: expected=%.10g, actual=%.10g, "
                        "error=%.10g, tolerance=%.10g\n",
                        name,
                        block,
                        expected_block,
                        actual_block,
                        absolute_error,
                        tolerance);
            }

            ++failed_blocks;
            passed = 0;
        }

        expected_total += expected_block;
        gpu_total += actual_block;
    }

    printf("[%s] %s: blocks=%d, failed_blocks=%d, "
           "expected_total=%.10g, actual_total=%.10g\n",
           passed ? "PASS" : "FAIL",
           name,
           blocks,
           failed_blocks,
           expected_total,
           gpu_total);

    free(gpu_block_sums);
    return passed;
}

/**
 * 生成可重复的大规模测试数据。
 *
 * 使用确定性公式而不是随机数，失败时无需保存随机种子即可复现。生成值包含正数、
 * 负数和零，并限制在较小范围内，以免单精度累加误差掩盖索引问题。先对下标取模，
 * 可以避免 N=2^27 时 i * 17 超出 int 范围而产生未定义行为。
 */
static void fill_values(float *values, int n)
{
    int i;

    for (i = 0; i < n; ++i)
    {
        int reduced_index;
        int value;

        reduced_index = i % 101;
        value = ((reduced_index * 17 + 26) % 101) - 50;
        values[i] = (float)value / 10.0f;
    }
}

int main(void)
{
    // 固定使用 2^27 个 float，主机输入和设备输入分别占用 512 MiB。
    // v0-v2 启动 524288 个线程块；v3-v6 每个线程读取两个元素，启动 262144 个
    // 线程块；v7 每个线程读取一个 float4，启动 131072 个线程块。所有版本均使用
    // 256 个线程，用于比较不同规约和全局内存读取策略的收益。
    const int n = 1 << 27;
    float *values;
    int all_passed;

    values = allocate_host_values(n);
    fill_values(values, n);

    // 所有版本使用完全相同的输入及 CPU 参考校验，保证测试结果可以直接比较。
    // 使用按位与赋值可确保某个版本结果错误后仍会继续测试后续版本。run_test 只借用
    // values，不取得其所有权，因此全部版本测试结束后由 main 统一释放主机内存。
    all_passed = 1;
    all_passed &= run_test("reduce_v0 N=2^27",
                           REDUCE_VERSION_V0,
                           values,
                           n);
    all_passed &= run_test("reduce_v1 N=2^27",
                           REDUCE_VERSION_V1,
                           values,
                           n);
    all_passed &= run_test("reduce_v2 N=2^27",
                           REDUCE_VERSION_V2,
                           values,
                           n);
    all_passed &= run_test("reduce_v3 N=2^27",
                           REDUCE_VERSION_V3,
                           values,
                           n);
    all_passed &= run_test("reduce_v4 N=2^27",
                           REDUCE_VERSION_V4,
                           values,
                           n);
    all_passed &= run_test("reduce_v5 N=2^27",
                           REDUCE_VERSION_V5,
                           values,
                           n);
    all_passed &= run_test("reduce_v6 N=2^27",
                           REDUCE_VERSION_V6,
                           values,
                           n);
    all_passed &= run_test("reduce_v7 N=2^27",
                           REDUCE_VERSION_V7,
                           values,
                           n);

    free(values);

    printf("%s\n",
           all_passed ? "All tests passed." : "Some tests failed.");
    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

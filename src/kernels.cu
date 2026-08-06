#include <vector>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../tester/utils.h"

// *********************************************************************
// rmsNorm
// *********************************************************************
template <typename T>
__global__ void rmsNormKernel(
    const T* input,
    const T* weight,
    T* output,
    int rows,
    int hidden_dim,
    float eps
)
{
    // 一个block处理一行
    int row = blockIdx.x;

    if(row >= rows)
        return;

    // 每个线程保存局部平方和
    __shared__ float shared_sum[256];

    float sum = 0.0f;

    // 每个线程计算部分元素
    for(int col = threadIdx.x; col < hidden_dim; col += blockDim.x)
    {
        float value = static_cast<float>(input[row * hidden_dim + col]);
        sum += value * value;
    }

    shared_sum[threadIdx.x] = sum;
    __syncthreads();

    // reduction 求整行平方和
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(threadIdx.x < stride)
        {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // RMS缩放因子
    float scale = rsqrtf(shared_sum[0] / hidden_dim + eps);

    // 写输出
    for(int col = threadIdx.x; col < hidden_dim; col += blockDim.x)
    {
        float value = static_cast<float>(input[row * hidden_dim + col]);
        float w = static_cast<float>(weight[col]);
        output[row * hidden_dim + col] = static_cast<T>(value * scale * w);
    }
}

/**
 * @brief Computes RMSNorm over the last dimension of a 2D tensor.
 *
 * The input is a row-major matrix with shape [rows, hidden_dim]. For each row
 * i and column j:
 *
 *   output[i, j] = input[i, j] * rsqrt(mean(input[i, :]^2) + eps) * weight[j]
 *
 * The output vector is preallocated with rows * hidden_dim elements.
 *
 * @tparam T Data type of input, weight, and output tensors.
 * @param[in] h_input Flattened input matrix of shape [rows, hidden_dim].
 * @param[in] h_weight Per-column scale vector of shape [hidden_dim].
 * @param[out] h_output Flattened output matrix of shape [rows, hidden_dim].
 * @param[in] rows Number of rows/tokens.
 * @param[in] hidden_dim Size of the normalized dimension.
 * @param[in] eps Numerical stability epsilon.
 */
template <typename T>
void rmsNorm(
    const std::vector<T>& h_input,
    const std::vector<T>& h_weight,
    std::vector<T>& h_output,
    size_t rows,
    size_t hidden_dim,
    float eps
)
{
    T* d_input;
    T* d_weight;
    T* d_output;

    size_t input_bytes = rows * hidden_dim * sizeof(T);
    size_t weight_bytes = hidden_dim * sizeof(T);

    // GPU申请显存
    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_output, input_bytes);
    cudaMalloc(&d_weight, weight_bytes);

    // CPU -> GPU
    cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight.data(), weight_bytes, cudaMemcpyHostToDevice);

    // 一个row对应一个block
    dim3 grid(rows);
    dim3 block(256);

    rmsNormKernel<T><<<grid, block>>>(
        d_input, d_weight, d_output, rows, hidden_dim, eps
    );

    cudaDeviceSynchronize();

    // GPU -> CPU
    cudaMemcpy(h_output.data(), d_output, input_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}

// *********************************************************************
// flashAttention
// ********************************************************************

// 0: [Batch, SeqLen, Heads, HeadDim]
// 1: [Batch, Heads, SeqLen, HeadDim]

#define LAYOUT_BHSD 0

#if LAYOUT_BHSD

#define QO_OFFSET(b,s,h,d) ((((b)*query_heads+(h))*target_seq_len+(s))*head_dim+(d))
#define KV_OFFSET(b,s,h,d) ((((b)*kv_heads+(h))*src_seq_len+(s))*head_dim+(d))

#else

#define QO_OFFSET(b,s,h,d) ((((b)*target_seq_len+(s))*query_heads+(h))*head_dim+(d))
#define KV_OFFSET(b,s,h,d) ((((b)*src_seq_len+(s))*kv_heads+(h))*head_dim+(d))

#endif

template<typename T>
__global__ void flashAttentionKernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    T* __restrict__ o,

    int batch_size,
    int target_seq_len,
    int src_seq_len,

    int query_heads,
    int kv_heads,

    int head_dim,

    bool is_causal
)
{

    int total = batch_size * target_seq_len * query_heads;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gtid >= total)
        return;

    // gtid -> (batch, t_idx, q_head)，与参考的索引分解一致
    int q_head_idx = gtid % query_heads;
    int rem = gtid / query_heads;
    int t_idx = rem % target_seq_len;
    int b_idx = rem / target_seq_len;

    // GQA
    int kv_head_idx = q_head_idx / (query_heads / kv_heads);

    // scale = rcp.rn(sqrt.rn(head_dim))，与参考同
    float scale = __frcp_rn(sqrtf((float)head_dim));

    // per-thread 本地缓冲（参考用 local memory 存 Q/O；head_dim ≤ 256）
    float q_buf[256];
    float o_buf[256];

    int qo_base = QO_OFFSET(b_idx, t_idx, q_head_idx, 0);

    // load Q
    for(int d = 0; d < head_dim; d++)
        q_buf[d] = (float)q[qo_base + d];

    // zero O
    for(int d = 0; d < head_dim; d++)
        o_buf[d] = 0.0f;

    // =====================================================
    // Pass 1: global_max = max_s (Q·K[s]) * scale
    // =====================================================
    float global_max = -INFINITY;

    for(int s_idx = 0; s_idx < src_seq_len; s_idx++)
    {
        if(is_causal && s_idx > t_idx)
            continue;

        int kv_base = KV_OFFSET(b_idx, s_idx, kv_head_idx, 0);

        float score = 0.0f;
        for(int d = 0; d < head_dim; d++)
            score = fmaf(q_buf[d], (float)k[kv_base + d], score);

        global_max = fmaxf(global_max, score * scale);
    }

    // =====================================================
    // Pass 2: sum = Σ exp(Q·K[s]*scale - global_max)
    // =====================================================
    float sum = 0.0f;

    for(int s_idx = 0; s_idx < src_seq_len; s_idx++)
    {
        if(is_causal && s_idx > t_idx)
            continue;

        int kv_base = KV_OFFSET(b_idx, s_idx, kv_head_idx, 0);

        float score = 0.0f;
        for(int d = 0; d < head_dim; d++)
            score = fmaf(q_buf[d], (float)k[kv_base + d], score);

        sum += expf(score * scale - global_max);
    }

    // inv_sum = rcp.rn(sum)，sum==0 时保持 0（与参考一致）
    float inv_sum = (sum != 0.0f) ? __frcp_rn(sum) : 0.0f;

    // =====================================================
    // Pass 3: O[d] += (inv_sum * prob) * V[s,d]
    // =====================================================
    for(int s_idx = 0; s_idx < src_seq_len; s_idx++)
    {
        if(is_causal && s_idx > t_idx)
            continue;

        int kv_base = KV_OFFSET(b_idx, s_idx, kv_head_idx, 0);

        float score = 0.0f;
        for(int d = 0; d < head_dim; d++)
            score = fmaf(q_buf[d], (float)k[kv_base + d], score);

        float prob = expf(score * scale - global_max);
        float factor = inv_sum * prob;

        for(int d = 0; d < head_dim; d++)
            o_buf[d] = fmaf(factor, (float)v[kv_base + d], o_buf[d]);
    }

    // =====================================================
    // write O
    // =====================================================
    for(int d = 0; d < head_dim; d++)
        o[qo_base + d] = (T)o_buf[d];
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */

template<typename T>
void flashAttention(
    const std::vector<T>& h_q,
    const std::vector<T>& h_k,
    const std::vector<T>& h_v,
    std::vector<T>& h_o,

    int batch_size,
    int target_seq_len,
    int src_seq_len,

    int query_heads,
    int kv_heads,

    int head_dim,

    bool is_causal
)
{
    T *d_q;
    T *d_k;
    T *d_v;
    T *d_o;

    size_t q_size = h_q.size() * sizeof(T);
    size_t k_size = h_k.size() * sizeof(T);
    size_t v_size = h_v.size() * sizeof(T);

    cudaMalloc(&d_q, q_size);
    cudaMalloc(&d_k, k_size);
    cudaMalloc(&d_v, v_size);
    cudaMalloc(&d_o, q_size);

    cudaMemcpy(d_q, h_q.data(), q_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), k_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), v_size, cudaMemcpyHostToDevice);

    // 一个线程处理一个 query：grid 在 query 维度并行，无需 shared memory
    int total = batch_size * target_seq_len * query_heads;

    dim3 block(256);
    dim3 grid((total + block.x - 1) / block.x);

    flashAttentionKernel<T><<<grid, block>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads,
        head_dim,
        is_causal
    );

    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess)
    {
        printf("FlashAttention kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_o.data(), d_o, q_size, cudaMemcpyDeviceToHost);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float>&, const std::vector<float>&,
  std::vector<float>&, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&, const std::vector<half>&,
  std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);

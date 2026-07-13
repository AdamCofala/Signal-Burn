#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <unordered_map>
#include <mutex>

extern "C" {

struct IQSample { int16_t r; int16_t i; };

// Kernel to convert interleaved IQ samples to floats
__global__ void convert(const IQSample* in, cufftComplex* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx].x = (float)in[idx].r;
        out[idx].y = (float)in[idx].i;
    }
}

// Kernel to compute power spectrum from FFT output
__global__ void power(const cufftComplex* in, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float r = in[idx].x;
        float i = in[idx].y;
        out[idx] = r * r + i * i;
    }
}

// Kernel to downsample the power spectrum by averaging
__global__ void downsample_average(const float* in_pow, float* out_downsampled, int orig_size, int out_size) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx < out_size) {
        int factor = orig_size / out_size;
        int start_idx = out_idx * factor;
        int end_idx = start_idx + factor;

        if (end_idx > orig_size) end_idx = orig_size;

        double sum = 0.0;
        for (int i = start_idx; i < end_idx; i++) {
            sum += in_pow[i];
        }

        int count = end_idx - start_idx;
        out_downsampled[out_idx] = (count > 0) ? (float)(sum / count) : 0.0f;
    }
}

// Global state for CUDA resources
static std::unordered_map<int, cufftHandle> g_plans;
static std::mutex g_state_mutex;

static IQSample* g_d_in = nullptr;
static cufftComplex* g_d_fft = nullptr;
static float* g_d_pow = nullptr;
static float* g_d_downsampled = nullptr;
static int g_allocated_size = 0;
static int g_allocated_out_size = 0;

static cufftHandle get_plan(int n) {
    auto it = g_plans.find(n);
    if (it != g_plans.end()) return it->second;

    cufftHandle plan;
    cufftResult r = cufftPlan1d(&plan, n, CUFFT_C2C, 1);
    if (r != CUFFT_SUCCESS) return 0;
    g_plans[n] = plan;
    return plan;
}


int sb_process_iq(const int16_t* interleaved_iq, size_t num_samples, float* out_pow_downsampled, int out_size) {

    if (num_samples == 0 || num_samples > (size_t)INT32_MAX || out_size <= 0) return -1;
    int n = (int)num_samples;

    std::lock_guard<std::mutex> lock(g_state_mutex);

    if (n > g_allocated_size || out_size > g_allocated_out_size) {
        if (g_d_in)          cudaFree(g_d_in);
        if (g_d_fft)         cudaFree(g_d_fft);
        if (g_d_pow)         cudaFree(g_d_pow);
        if (g_d_downsampled) cudaFree(g_d_downsampled);

        cudaError_t err;
        err = cudaMalloc(&g_d_in, n * sizeof(IQSample));
        err = cudaMalloc(&g_d_fft, n * sizeof(cufftComplex));
        err = cudaMalloc(&g_d_pow, n * sizeof(float));
        err = cudaMalloc(&g_d_downsampled, out_size * sizeof(float));
        if (err != cudaSuccess) return -10;

        g_allocated_size = n;
        g_allocated_out_size = out_size;
    }

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    if (cudaMemcpy(g_d_in, interleaved_iq, n * sizeof(IQSample), cudaMemcpyHostToDevice) != cudaSuccess) return -2;

    convert<<<blocks, threads>>>(g_d_in, g_d_fft, n);
    cufftHandle plan = get_plan(n);
    if (plan == 0 || cufftExecC2C(plan, g_d_fft, g_d_fft, CUFFT_FORWARD) != CUFFT_SUCCESS) return -3;

    power<<<blocks, threads>>>(g_d_fft, g_d_pow, n);

    int ds_threads = 256;
    int ds_blocks = (out_size + ds_threads - 1) / ds_threads;
    downsample_average<<<ds_blocks, ds_threads>>>(g_d_pow, g_d_downsampled, n, out_size);

    if (cudaMemcpy(out_pow_downsampled, g_d_downsampled, out_size * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return -5;

    return 0;
}

void sb_shutdown() {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    for (auto& kv : g_plans) cufftDestroy(kv.second);
    g_plans.clear();

    if (g_d_in)  { cudaFree(g_d_in);  g_d_in = nullptr; }
    if (g_d_fft) { cudaFree(g_d_fft); g_d_fft = nullptr; }
    if (g_d_pow) { cudaFree(g_d_pow); g_d_pow = nullptr; }
    if (g_d_downsampled) { cudaFree(g_d_downsampled); g_d_downsampled = nullptr; }
    g_allocated_size = 0;
    g_allocated_out_size = 0;
}

} // extern "C"
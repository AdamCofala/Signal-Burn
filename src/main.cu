#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <unordered_map>
#include <mutex>

extern "C" {

struct IQSample { int16_t r; int16_t i; };

// --- Kernels ---

__global__ void convert(const IQSample* in, cufftComplex* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx].x = (float)in[idx].r;
        out[idx].y = (float)in[idx].i;
    }
}

__global__ void fftPower(const cufftComplex* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float r = in[idx].x;
        float i = in[idx].y;
        out[idx] = r * r + i * i;
    }
}

__global__ void accumulate(const float* current_pow, float* accum, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        accum[idx] += current_pow[idx];
    }
}

__global__ void normalize(float* accum, int num_windows, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        accum[idx] /= (float)num_windows;
    }
}

__global__ void shiftFFT(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int shift = n / 2;
        int src_idx = (idx + shift) % n;
        out[idx] = in[src_idx];
    }
}

// --- Global State ---

static std::unordered_map<int, cufftHandle> g_plans;
static std::mutex g_state_mutex;

static IQSample* g_d_in = nullptr;
static cufftComplex* g_d_fft = nullptr;
static float* g_d_pow = nullptr;
static float* g_d_accum = nullptr;
static float* g_d_result = nullptr;
static int g_allocated_n = 0;
static size_t g_allocated_in_size = 0;

static cufftHandle get_plan(int n) {
    auto it = g_plans.find(n);
    if (it != g_plans.end()) return it->second;
    cufftHandle plan;
    if (cufftPlan1d(&plan, n, CUFFT_C2C, 1) != CUFFT_SUCCESS) return 0;
    g_plans[n] = plan;
    return plan;
}

// --- API Functions ---

int sb_process_fft(const int16_t* interleaved_iq, size_t num_samples, float* out_spectrum, int fft_size) {
    if (num_samples < (size_t)fft_size) return -1;

    std::lock_guard<std::mutex> lock(g_state_mutex);

    if (fft_size > g_allocated_n) {
        if (g_d_fft) cudaFree(g_d_fft);
        if (g_d_pow) cudaFree(g_d_pow);
        if (g_d_accum) cudaFree(g_d_accum);
        if (g_d_result) cudaFree(g_d_result);

        cudaMalloc(&g_d_fft, fft_size * sizeof(cufftComplex));
        cudaMalloc(&g_d_pow, fft_size * sizeof(float));
        cudaMalloc(&g_d_accum, fft_size * sizeof(float));
        cudaMalloc(&g_d_result, fft_size * sizeof(float));
        g_allocated_n = fft_size;
    }

    if (num_samples > g_allocated_in_size) {
        if (g_d_in) cudaFree(g_d_in);
        cudaMalloc(&g_d_in, num_samples * sizeof(IQSample));
        g_allocated_in_size = num_samples;
    }

    cudaMemcpy(g_d_in, interleaved_iq, num_samples * sizeof(IQSample), cudaMemcpyHostToDevice);
    cudaMemset(g_d_accum, 0, fft_size * sizeof(float));

    int num_windows = 0;
    int stride = fft_size / 2;

    cufftHandle plan = get_plan(fft_size);
    if (plan == 0) return -2;

    for (size_t offset = 0; offset + fft_size <= num_samples; offset += stride) {

        convert<<< (fft_size + 255)/256, 256 >>>(g_d_in + offset, g_d_fft, fft_size);

        cufftExecC2C(plan, g_d_fft, g_d_fft, CUFFT_FORWARD);

        fftPower<<< (fft_size + 255)/256, 256 >>>(g_d_fft, g_d_pow, fft_size);

        accumulate<<< (fft_size + 255)/256, 256 >>>(g_d_pow, g_d_accum, fft_size);

        num_windows++;
    }

    if (num_windows > 0) {

        normalize<<< (fft_size + 255)/256, 256 >>>(g_d_accum, num_windows, fft_size);

        shiftFFT<<< (fft_size + 255)/256, 256 >>>(g_d_accum, g_d_result, fft_size);

        cudaMemcpy(out_spectrum, g_d_result, fft_size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    return 0;
}

void sb_shutdown() {
    std::lock_guard<std::mutex> lock(g_state_mutex);

    for (auto& kv : g_plans) {
        cufftDestroy(kv.second);
    }
    g_plans.clear();

    if (g_d_in)      { cudaFree(g_d_in);      g_d_in = nullptr; }
    if (g_d_fft)     { cudaFree(g_d_fft);     g_d_fft = nullptr; }
    if (g_d_pow)     { cudaFree(g_d_pow);     g_d_pow = nullptr; }
    if (g_d_accum)   { cudaFree(g_d_accum);   g_d_accum = nullptr; }
    if (g_d_result)  { cudaFree(g_d_result);  g_d_result = nullptr; }

    g_allocated_n = 0;
    g_allocated_in_size = 0;
}

} // extern "C"
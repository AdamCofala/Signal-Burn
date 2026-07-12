#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <float.h>
#include <unordered_map>
#include <mutex>

extern "C" {

struct IQSample { int16_t r; int16_t i; };

__global__ void convertInt16ToComplex(const IQSample* in, cufftComplex* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx].x = (float)in[idx].r;
        out[idx].y = (float)in[idx].i;
    }
}

__global__ void calcMagnitude(const cufftComplex* in, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float real = in[idx].x;
        float imag = in[idx].y;
        out[idx] = sqrtf(real * real + imag * imag);
    }
}

// Redukcja (max-pool) + fftshift w jednym przejściu.
__global__ void downsampleShiftedMax(const float* full_mag, float* out_bins, int N, int num_bins) {
    int bin = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin >= num_bins) return;

    long long start = ((long long)bin * N) / num_bins;
    long long end   = ((long long)(bin + 1) * N) / num_bins;
    if (end <= start) end = start + 1;
    if (end > N) end = N;

    float max_val = -FLT_MAX;
    for (long long shifted_idx = start; shifted_idx < end; shifted_idx++) {
        long long orig_idx = (shifted_idx + N / 2) % N;  // shifted -> naturalny indeks FFT
        float v = full_mag[orig_idx];
        if (v > max_val) max_val = v;
    }
    out_bins[bin] = max_val;
}

static std::unordered_map<int, cufftHandle> g_plans;
static std::mutex g_plan_mutex;

static cufftHandle get_plan(int n) {
    std::lock_guard<std::mutex> lock(g_plan_mutex);
    auto it = g_plans.find(n);
    if (it != g_plans.end()) return it->second;

    cufftHandle plan;
    cufftResult r = cufftPlan1d(&plan, n, CUFFT_C2C, 1);
    if (r != CUFFT_SUCCESS) {
        fprintf(stderr, "cufftPlan1d failed: %d\n", r);
        return 0;
    }
    g_plans[n] = plan;
    return plan;
}

// out_mag musi miec miejsce na `out_bins` floatow (NIE num_samples floatow!)
int sb_process_iq(const int16_t* interleaved_iq, size_t num_samples, int out_bins, float* out_mag) {
    if (num_samples == 0 || num_samples > (size_t)INT32_MAX) return -1;
    if (out_bins <= 0 || (size_t)out_bins > num_samples) return -7;

    IQSample* d_in = nullptr;
    cufftComplex* d_fft = nullptr;
    float* d_mag = nullptr;
    float* d_out = nullptr;
    int status = 0;

    do {
        cudaError_t err;
        err = cudaMalloc((void**)&d_in, num_samples * sizeof(IQSample));
        if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_in: %s\n", cudaGetErrorString(err)); status = -2; break; }

        err = cudaMalloc((void**)&d_fft, num_samples * sizeof(cufftComplex));
        if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_fft: %s\n", cudaGetErrorString(err)); status = -2; break; }

        err = cudaMalloc((void**)&d_mag, num_samples * sizeof(float));
        if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_mag: %s\n", cudaGetErrorString(err)); status = -2; break; }

        err = cudaMalloc((void**)&d_out, out_bins * sizeof(float));
        if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_out: %s\n", cudaGetErrorString(err)); status = -2; break; }

        err = cudaMemcpy(d_in, interleaved_iq, num_samples * sizeof(IQSample), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { fprintf(stderr, "H2D copy: %s\n", cudaGetErrorString(err)); status = -3; break; }

        int n = (int)num_samples;
        int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        convertInt16ToComplex<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_fft, n);
        if ((err = cudaGetLastError()) != cudaSuccess) { fprintf(stderr, "convert kernel: %s\n", cudaGetErrorString(err)); status = -4; break; }

        cufftHandle plan = get_plan(n);
        if (plan == 0) { status = -5; break; }

        cufftResult r = cufftExecC2C(plan, d_fft, d_fft, CUFFT_FORWARD);
        if (r != CUFFT_SUCCESS) { fprintf(stderr, "cufftExecC2C: %d\n", r); status = -5; break; }

        calcMagnitude<<<blocksPerGrid, threadsPerBlock>>>(d_fft, d_mag, n);
        if ((err = cudaGetLastError()) != cudaSuccess) { fprintf(stderr, "magnitude kernel: %s\n", cudaGetErrorString(err)); status = -4; break; }

        int binsBlocks = (out_bins + threadsPerBlock - 1) / threadsPerBlock;
        downsampleShiftedMax<<<binsBlocks, threadsPerBlock>>>(d_mag, d_out, n, out_bins);
        if ((err = cudaGetLastError()) != cudaSuccess) { fprintf(stderr, "downsample kernel: %s\n", cudaGetErrorString(err)); status = -4; break; }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { fprintf(stderr, "sync: %s\n", cudaGetErrorString(err)); status = -6; break; }

        err = cudaMemcpy(out_mag, d_out, out_bins * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { fprintf(stderr, "D2H copy: %s\n", cudaGetErrorString(err)); status = -3; break; }

    } while (false);

    if (d_in)  cudaFree(d_in);
    if (d_fft) cudaFree(d_fft);
    if (d_mag) cudaFree(d_mag);
    if (d_out) cudaFree(d_out);

    return status;
}

void sb_shutdown() {
    std::lock_guard<std::mutex> lock(g_plan_mutex);
    for (auto& kv : g_plans) cufftDestroy(kv.second);
    g_plans.clear();
}

} // extern "C"
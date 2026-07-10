#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <unordered_map>
#include <mutex>

extern "C" {

struct IQSample { int16_t r; int16_t i; };

// CUDA kernel: int16 to float
__global__ void convertInt16ToComplex(const IQSample* in, cufftComplex* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx].x = (float)in[idx].r;
        out[idx].y = (float)in[idx].i;
    }
}

// CUDA kernel: calculate magnitude of complex numbers
__global__ void calcMagnitude(const cufftComplex* in, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float real = in[idx].x;
        float imag = in[idx].y;
        out[idx] = sqrtf(real * real + imag * imag);
    }
}

// Global map to cache CUFFT plans for different sizes
static std::unordered_map<int, cufftHandle> g_plans;
// Mutex to protect access to the global plan map
static std::mutex g_plan_mutex;

static cufftHandle get_plan(int n) {

    std::lock_guard<std::mutex> lock(g_plan_mutex); // Ensure thread safety

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


int process_iq(const int16_t* interleaved_iq, size_t num_samples, float* out_mag) {
    if (num_samples == 0 || num_samples > (size_t)INT32_MAX) return -1;

    IQSample* d_in = nullptr;
    cufftComplex* d_out = nullptr;
    float* d_mag = nullptr;
    int status = 0;

    do {
        cudaError_t err;
        err = cudaMalloc((void**)&d_in, num_samples * sizeof(IQSample));
        if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_in: %s\n", cudaGetErrorString(err)); status = -2; break; }

        err = cudaMalloc((void**)&d_out, num_samples * sizeof(cufftComplex));
        if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_out: %s\n", cudaGetErrorString(err)); status = -2; break; }

        err = cudaMalloc((void**)&d_mag, num_samples * sizeof(float));
        if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_mag: %s\n", cudaGetErrorString(err)); status = -2; break; }

        // Copy from numpy RAM
        err = cudaMemcpy(d_in, interleaved_iq, num_samples * sizeof(IQSample), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { fprintf(stderr, "H2D copy: %s\n", cudaGetErrorString(err)); status = -3; break; }

        int n = (int)num_samples;
        int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        convertInt16ToComplex<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, n);
        if ((err = cudaGetLastError()) != cudaSuccess) { fprintf(stderr, "convert kernel: %s\n", cudaGetErrorString(err)); status = -4; break; }

        cufftHandle plan = get_plan(n);
        if (plan == 0) { status = -5; break; }
        cufftResult r = cufftExecC2C(plan, d_out, d_out, CUFFT_FORWARD);
        if (r != CUFFT_SUCCESS) { fprintf(stderr, "cufftExecC2C: %d\n", r); status = -5; break; }

        calcMagnitude<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_mag, n);
        if ((err = cudaGetLastError()) != cudaSuccess) { fprintf(stderr, "magnitude kernel: %s\n", cudaGetErrorString(err)); status = -4; break; }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { fprintf(stderr, "sync: %s\n", cudaGetErrorString(err)); status = -6; break; }

        err = cudaMemcpy(out_mag, d_mag, num_samples * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { fprintf(stderr, "D2H copy: %s\n", cudaGetErrorString(err)); status = -3; break; }

    } while (false);

    if (d_in)  cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    if (d_mag) cudaFree(d_mag);

    return status;
}

void dgp_shutdown() {
    std::lock_guard<std::mutex> lock(g_plan_mutex);
    for (auto& kv : g_plans) cufftDestroy(kv.second);
    g_plans.clear();
}

} // extern "C"
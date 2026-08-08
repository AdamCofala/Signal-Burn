#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <unordered_map>
#include <mutex>
#include <chrono>

extern "C" {

struct IQSample { int16_t r; int16_t i; };

// ---------------------------------------------------------------------
//  Kernels
// ---------------------------------------------------------------------
__global__ void convert(const IQSample* in, cufftComplex* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx].x = (float)in[idx].r;
        out[idx].y = (float)in[idx].i;
    }
}

__global__ void applyWindowHann(cufftComplex* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float w = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * idx / (n - 1)));
        data[idx].x *= w;
        data[idx].y *= w;
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

__global__ void magnitude(const cufftComplex* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float r = in[idx].x;
        float i = in[idx].y;
        out[idx] = sqrtf(r * r + i * i);
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

__global__ void correctGain(float* data, float gain_factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= gain_factor;
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

__global__ void crossAccumulate(const cufftComplex* fft1, const cufftComplex* fft2,
                                cufftComplex* accum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float real = fft1[i].x * fft2[i].x + fft1[i].y * fft2[i].y;
        float imag = fft1[i].y * fft2[i].x - fft1[i].x * fft2[i].y;
        accum[i].x += real;
        accum[i].y += imag;
    }
}

__global__ void normalizeComplex(cufftComplex* accum, int num_windows, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        accum[idx].x /= (float)num_windows;
        accum[idx].y /= (float)num_windows;
    }
}

__global__ void calculateCoherence(const cufftComplex* cross_acc, const float* pow1,
                                   const float* pow2, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float cross_mag_sq = cross_acc[i].x * cross_acc[i].x + cross_acc[i].y * cross_acc[i].y;
        float den = pow1[i] * pow2[i];
        out[i] = (den > 1e-12f) ? (cross_mag_sq / den) : 0.0f;
    }
}

__global__ void applyBasebandShift(cufftComplex* data, float phase_inc, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float angle = phase_inc * (float)idx;
        float s, c;
        sincosf(angle, &s, &c);
        float r = data[idx].x;
        float i = data[idx].y;
        data[idx].x = r * c - i * s;
        data[idx].y = r * s + i * c;
    }
}

// Streaming reduction – used by sb_process_baseband_iq_full
__global__ void complexAvgStream(const IQSample* in, float* real_sum, float* imag_sum,
                                 float phase_inc, int n) {
    extern __shared__ float sdata[];
    float* s_real = sdata;
    float* s_imag = sdata + blockDim.x;
    int tid = threadIdx.x;
    float sum_r = 0.0f;
    float sum_i = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x) {
        float r = (float)in[i].r;
        float im = (float)in[i].i;
        if (phase_inc != 0.0f) {
            float angle = phase_inc * (float)i;
            float s, c;
            sincosf(angle, &s, &c);
            float new_r = r * c - im * s;
            float new_i = r * s + im * c;
            r = new_r;
            im = new_i;
        }
        float w = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (n - 1)));
        r *= w;
        im *= w;
        sum_r += r;
        sum_i += im;
    }
    s_real[tid] = sum_r;
    s_imag[tid] = sum_i;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_real[tid] += s_real[tid + s];
            s_imag[tid] += s_imag[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(real_sum, s_real[0]);
        atomicAdd(imag_sum, s_imag[0]);
    }
}


// ---------------------------------------------------------------------
//  Global state & plans
// ---------------------------------------------------------------------
static std::unordered_map<int, cufftHandle> g_plans;
static std::mutex g_state_mutex;

static IQSample*       g_d_in          = nullptr;
static cufftComplex*   g_d_fft         = nullptr;
static float*          g_d_pow         = nullptr;
static float*          g_d_accum       = nullptr;
static float*          g_d_result      = nullptr;
static int             g_allocated_n   = 0;
static size_t          g_allocated_in_size = 0;

static IQSample*       g_d_in2         = nullptr;
static cufftComplex*   g_d_fft_saved   = nullptr;
static cufftComplex*   g_d_cross_accum = nullptr;
static float*          g_d_cross_result= nullptr;
static int             g_allocated_n_cross = 0;
static size_t          g_allocated_in2_size = 0;

static float*          g_d_pow1 = nullptr;
static float*          g_d_pow2 = nullptr;
static int             g_allocated_coh_n = 0;

static IQSample*       g_d_in3         = nullptr;
static cufftComplex*   g_d_fft_saved1  = nullptr;
static cufftComplex*   g_d_fft_saved2  = nullptr;
static cufftComplex*   g_d_cross13_accum = nullptr;
static cufftComplex*   g_d_cross23_accum = nullptr;
static float*          g_d_pow3        = nullptr;
static int             g_allocated_n_triple = 0;
static size_t          g_allocated_in3_size = 0;

static float*          g_real_sum = nullptr;
static float*          g_imag_sum = nullptr;

static cufftHandle get_plan(int n) {
    auto it = g_plans.find(n);
    if (it != g_plans.end()) return it->second;
    cufftHandle plan;
    if (cufftPlan1d(&plan, n, CUFFT_C2C, 1) != CUFFT_SUCCESS) return 0;
    g_plans[n] = plan;
    return plan;
}


// ---------------------------------------------------------------------
//  Reallocation helpers
// ---------------------------------------------------------------------
void reallocate_FFT_buffers(int fft_size) {
    if (g_d_fft)    cudaFree(g_d_fft);
    if (g_d_pow)    cudaFree(g_d_pow);
    if (g_d_accum)  cudaFree(g_d_accum);
    if (g_d_result) cudaFree(g_d_result);
    cudaMalloc(&g_d_fft,    fft_size * sizeof(cufftComplex));
    cudaMalloc(&g_d_pow,    fft_size * sizeof(float));
    cudaMalloc(&g_d_accum,  fft_size * sizeof(float));
    cudaMalloc(&g_d_result, fft_size * sizeof(float));
    g_allocated_n = fft_size;
}

void reallocate_cross_corelation_buffers(int fft_size) {
    if (g_d_fft_saved)    cudaFree(g_d_fft_saved);
    if (g_d_cross_accum)  cudaFree(g_d_cross_accum);
    if (g_d_cross_result) cudaFree(g_d_cross_result);
    cudaMalloc(&g_d_fft_saved,    fft_size * sizeof(cufftComplex));
    cudaMalloc(&g_d_cross_accum,  fft_size * sizeof(cufftComplex));
    cudaMalloc(&g_d_cross_result, 2 * fft_size * sizeof(float));
    g_allocated_n_cross = fft_size;
}

void reallocate_coherence_buffers(int fft_size) {
    if (fft_size <= g_allocated_coh_n) return;
    if (g_d_pow1) cudaFree(g_d_pow1);
    if (g_d_pow2) cudaFree(g_d_pow2);
    cudaMalloc(&g_d_pow1, fft_size * sizeof(float));
    cudaMalloc(&g_d_pow2, fft_size * sizeof(float));
    g_allocated_coh_n = fft_size;
}

void reallocate_triple_buffers(int fft_size) {
    if (fft_size <= g_allocated_n_triple) return;
    if (g_d_in3)           cudaFree(g_d_in3);
    if (g_d_fft_saved1)    cudaFree(g_d_fft_saved1);
    if (g_d_fft_saved2)    cudaFree(g_d_fft_saved2);
    if (g_d_cross13_accum) cudaFree(g_d_cross13_accum);
    if (g_d_cross23_accum) cudaFree(g_d_cross23_accum);
    if (g_d_pow3)          cudaFree(g_d_pow3);
    cudaMalloc(&g_d_in3,           fft_size * sizeof(IQSample));
    cudaMalloc(&g_d_fft_saved1,    fft_size * sizeof(cufftComplex));
    cudaMalloc(&g_d_fft_saved2,    fft_size * sizeof(cufftComplex));
    cudaMalloc(&g_d_cross13_accum, fft_size * sizeof(cufftComplex));
    cudaMalloc(&g_d_cross23_accum, fft_size * sizeof(cufftComplex));
    cudaMalloc(&g_d_pow3,          fft_size * sizeof(float));
    g_allocated_n_triple = fft_size;
}

void reallocate_input(size_t num_samples) {
    if (g_d_in) cudaFree(g_d_in);
    cudaMalloc(&g_d_in, num_samples * sizeof(IQSample));
    g_allocated_in_size = num_samples;
}

void reallocate_input2(size_t num_samples) {
    if (g_d_in2) cudaFree(g_d_in2);
    cudaMalloc(&g_d_in2, num_samples * sizeof(IQSample));
    g_allocated_in2_size = num_samples;
}

void reallocate_input3(size_t num_samples) {
    if (g_d_in3) cudaFree(g_d_in3);
    cudaMalloc(&g_d_in3, num_samples * sizeof(IQSample));
    g_allocated_in3_size = num_samples;
}


// ---------------------------------------------------------------------
//  Shared window processing helper
// ---------------------------------------------------------------------
static void process_single_window(const IQSample* d_input, int offset,
                                  cufftComplex* d_fft, float* d_pow_out,
                                  cufftComplex* d_fft_save,
                                  int fft_size, cufftHandle plan) {
    dim3 block(256);
    dim3 grid((fft_size + 255) / 256);
    convert<<<grid, block>>>(d_input + offset, d_fft, fft_size);
    applyWindowHann<<<grid, block>>>(d_fft, fft_size);
    cufftExecC2C(plan, d_fft, d_fft, CUFFT_FORWARD);

    if (d_pow_out) {
        fftPower<<<grid, block>>>(d_fft, d_pow_out, fft_size);
    }

    if (d_fft_save) {
        cudaMemcpy(d_fft_save, d_fft, fft_size * sizeof(cufftComplex), cudaMemcpyDeviceToDevice);
    }
}


// ---------------------------------------------------------------------
//  API functions
// ---------------------------------------------------------------------
int sb_process_fft(const int16_t* interleaved_iq, size_t num_samples,
                   float* out_spectrum, int fft_size) {
    if (num_samples < (size_t)fft_size) return -1;
    std::lock_guard<std::mutex> lock(g_state_mutex);

    if (fft_size > g_allocated_n) reallocate_FFT_buffers(fft_size);
    if (num_samples > g_allocated_in_size) reallocate_input(num_samples);

    cudaMemcpy(g_d_in, interleaved_iq, num_samples * sizeof(IQSample), cudaMemcpyHostToDevice);
    cudaMemset(g_d_accum, 0, fft_size * sizeof(float));

    int num_windows = 0;
    int stride = fft_size / 2;
    cufftHandle plan = get_plan(fft_size);
    if (plan == 0) return -2;

    for (size_t offset = 0; offset + fft_size <= num_samples; offset += stride) {
        process_single_window(g_d_in, (int)offset, g_d_fft, g_d_pow, nullptr, fft_size, plan);
        dim3 block(256);
        dim3 grid((fft_size + 255) / 256);
        accumulate<<<grid, block>>>(g_d_pow, g_d_accum, fft_size);
        num_windows++;
    }

    if (num_windows > 0) {
        dim3 block(256);
        dim3 grid((fft_size + 255) / 256);
        normalize<<<grid, block>>>(g_d_accum, num_windows, fft_size);
        correctGain<<<grid, block>>>(g_d_accum, 4.0f, fft_size);
        shiftFFT<<<grid, block>>>(g_d_accum, g_d_result, fft_size);
        cudaMemcpy(out_spectrum, g_d_result, fft_size * sizeof(float), cudaMemcpyDeviceToHost);
    }
    return 0;
}

int sb_process_cross_fft(const int16_t* in1, const int16_t* in2, size_t num_samples,
                         float* out_cross, int fft_size) {
    if (num_samples < (size_t)fft_size) return -1;
    std::lock_guard<std::mutex> lock(g_state_mutex);

    if (fft_size > g_allocated_n_cross) reallocate_cross_corelation_buffers(fft_size);
    if (fft_size > g_allocated_n)       reallocate_FFT_buffers(fft_size);
    if (num_samples > g_allocated_in_size)  reallocate_input(num_samples);
    if (num_samples > g_allocated_in2_size) reallocate_input2(num_samples);

    cudaMemcpy(g_d_in,  in1, num_samples * sizeof(IQSample), cudaMemcpyHostToDevice);
    cudaMemcpy(g_d_in2, in2, num_samples * sizeof(IQSample), cudaMemcpyHostToDevice);
    cudaMemset(g_d_cross_accum, 0, fft_size * sizeof(cufftComplex));

    int num_windows = 0;
    int stride = fft_size / 2;
    cufftHandle plan = get_plan(fft_size);
    if (plan == 0) return -2;

    dim3 block(256);
    dim3 grid((fft_size + 255) / 256);
    for (size_t offset = 0; offset + fft_size <= num_samples; offset += stride) {
        process_single_window(g_d_in,  (int)offset, g_d_fft, nullptr, g_d_fft_saved, fft_size, plan);
        process_single_window(g_d_in2, (int)offset, g_d_fft, nullptr, nullptr,       fft_size, plan);
        crossAccumulate<<<grid, block>>>(g_d_fft_saved, g_d_fft, g_d_cross_accum, fft_size);
        num_windows++;
    }

    if (num_windows > 0) {
        normalizeComplex<<<grid, block>>>(g_d_cross_accum, num_windows, fft_size);
        magnitude<<<grid, block>>>(g_d_cross_accum, g_d_accum, fft_size);
        correctGain<<<grid, block>>>(g_d_accum, 2.0f, fft_size);
        shiftFFT<<<grid, block>>>(g_d_accum, g_d_result, fft_size);
        cudaMemcpy(out_cross, g_d_result, fft_size * sizeof(float), cudaMemcpyDeviceToHost);
    }
    return 0;
}

int sb_process_coherence(const int16_t* in1, const int16_t* in2, size_t num_samples,
                         float* out_coherence, int fft_size) {
    if (num_samples < (size_t)fft_size) return -1;
    std::lock_guard<std::mutex> lock(g_state_mutex);

    if (fft_size > g_allocated_n_cross) reallocate_cross_corelation_buffers(fft_size);
    if (fft_size > g_allocated_n)       reallocate_FFT_buffers(fft_size);
    reallocate_coherence_buffers(fft_size);
    if (num_samples > g_allocated_in_size)  reallocate_input(num_samples);
    if (num_samples > g_allocated_in2_size) reallocate_input2(num_samples);

    cudaMemcpy(g_d_in,  in1, num_samples * sizeof(IQSample), cudaMemcpyHostToDevice);
    cudaMemcpy(g_d_in2, in2, num_samples * sizeof(IQSample), cudaMemcpyHostToDevice);

    cudaMemset(g_d_cross_accum, 0, fft_size * sizeof(cufftComplex));
    cudaMemset(g_d_pow1, 0, fft_size * sizeof(float));
    cudaMemset(g_d_pow2, 0, fft_size * sizeof(float));

    int num_windows = 0;
    int stride = fft_size / 2;
    cufftHandle plan = get_plan(fft_size);
    if (plan == 0) return -2;

    dim3 block(256);
    dim3 grid((fft_size + 255) / 256);
    for (size_t offset = 0; offset + fft_size <= num_samples; offset += stride) {
        process_single_window(g_d_in, (int)offset, g_d_fft, g_d_pow, g_d_fft_saved, fft_size, plan);
        accumulate<<<grid, block>>>(g_d_pow, g_d_pow1, fft_size);

        process_single_window(g_d_in2, (int)offset, g_d_fft, g_d_pow, nullptr, fft_size, plan);
        accumulate<<<grid, block>>>(g_d_pow, g_d_pow2, fft_size);

        crossAccumulate<<<grid, block>>>(g_d_fft_saved, g_d_fft, g_d_cross_accum, fft_size);
        num_windows++;
    }

    if (num_windows > 0) {
        normalize<<<grid, block>>>(g_d_pow1, num_windows, fft_size);
        normalize<<<grid, block>>>(g_d_pow2, num_windows, fft_size);
        normalizeComplex<<<grid, block>>>(g_d_cross_accum, num_windows, fft_size);

        calculateCoherence<<<grid, block>>>(g_d_cross_accum, g_d_pow1, g_d_pow2,
                                            g_d_accum, fft_size);
        shiftFFT<<<grid, block>>>(g_d_accum, g_d_result, fft_size);
        cudaMemcpy(out_coherence, g_d_result, fft_size * sizeof(float), cudaMemcpyDeviceToHost);
    }
    return 0;
}

int sb_process_pair_full(const int16_t* in1, const int16_t* in2, size_t num_samples,
                         float* out_pow1, float* out_pow2,
                         float* out_cross_mag, float* out_coherence,
                         int fft_size) {
    if (num_samples < (size_t)fft_size) return -1;
    std::lock_guard<std::mutex> lock(g_state_mutex);

    if (fft_size > g_allocated_n)        reallocate_FFT_buffers(fft_size);
    if (fft_size > g_allocated_n_cross)  reallocate_cross_corelation_buffers(fft_size);
    reallocate_coherence_buffers(fft_size);
    if (num_samples > g_allocated_in_size)  reallocate_input(num_samples);
    if (num_samples > g_allocated_in2_size) reallocate_input2(num_samples);

    cudaMemcpy(g_d_in,  in1, num_samples * sizeof(IQSample), cudaMemcpyHostToDevice);
    cudaMemcpy(g_d_in2, in2, num_samples * sizeof(IQSample), cudaMemcpyHostToDevice);

    cudaMemset(g_d_pow1,        0, fft_size * sizeof(float));
    cudaMemset(g_d_pow2,        0, fft_size * sizeof(float));
    cudaMemset(g_d_cross_accum, 0, fft_size * sizeof(cufftComplex));

    int num_windows = 0;
    int stride = fft_size / 2;
    cufftHandle plan = get_plan(fft_size);
    if (plan == 0) return -2;

    dim3 block(256);
    dim3 grid((fft_size + 255) / 256);
    for (size_t offset = 0; offset + fft_size <= num_samples; offset += stride) {
        process_single_window(g_d_in, (int)offset, g_d_fft, g_d_pow, g_d_fft_saved, fft_size, plan);
        accumulate<<<grid, block>>>(g_d_pow, g_d_pow1, fft_size);

        process_single_window(g_d_in2, (int)offset, g_d_fft, g_d_pow, nullptr, fft_size, plan);
        accumulate<<<grid, block>>>(g_d_pow, g_d_pow2, fft_size);

        crossAccumulate<<<grid, block>>>(g_d_fft_saved, g_d_fft, g_d_cross_accum, fft_size);
        num_windows++;
    }

    if (num_windows > 0) {
        normalize<<<grid, block>>>(g_d_pow1, num_windows, fft_size);
        normalize<<<grid, block>>>(g_d_pow2, num_windows, fft_size);
        normalizeComplex<<<grid, block>>>(g_d_cross_accum, num_windows, fft_size);

        calculateCoherence<<<grid, block>>>(g_d_cross_accum, g_d_pow1, g_d_pow2,
                                            g_d_accum, fft_size);
        shiftFFT<<<grid, block>>>(g_d_accum, g_d_result, fft_size);
        cudaMemcpy(out_coherence, g_d_result, fft_size * sizeof(float), cudaMemcpyDeviceToHost);

        correctGain<<<grid, block>>>(g_d_pow1, 4.0f, fft_size);
        shiftFFT<<<grid, block>>>(g_d_pow1, g_d_result, fft_size);
        cudaMemcpy(out_pow1, g_d_result, fft_size * sizeof(float), cudaMemcpyDeviceToHost);

        correctGain<<<grid, block>>>(g_d_pow2, 4.0f, fft_size);
        shiftFFT<<<grid, block>>>(g_d_pow2, g_d_result, fft_size);
        cudaMemcpy(out_pow2, g_d_result, fft_size * sizeof(float), cudaMemcpyDeviceToHost);

        magnitude<<<grid, block>>>(g_d_cross_accum, g_d_accum, fft_size);
        correctGain<<<grid, block>>>(g_d_accum, 2.0f, fft_size);
        shiftFFT<<<grid, block>>>(g_d_accum, g_d_result, fft_size);
        cudaMemcpy(out_cross_mag, g_d_result, fft_size * sizeof(float), cudaMemcpyDeviceToHost);
    }
    return 0;
}

int sb_process_triple_all(
    const int16_t* in1, const int16_t* in2, const int16_t* in3, size_t num_samples,
    float* out_pow1, float* out_pow2, float* out_pow3,
    float* out_cross12, float* out_cross13, float* out_cross23,
    float* out_coh12, float* out_coh13, float* out_coh23,
    int fft_size) {

    if (num_samples < (size_t)fft_size) return -1;
    std::lock_guard<std::mutex> lock(g_state_mutex);

    if (fft_size > g_allocated_n)        reallocate_FFT_buffers(fft_size);
    if (fft_size > g_allocated_n_cross)  reallocate_cross_corelation_buffers(fft_size);
    reallocate_coherence_buffers(fft_size);
    reallocate_triple_buffers(fft_size);

    if (num_samples > g_allocated_in_size)  reallocate_input(num_samples);
    if (num_samples > g_allocated_in2_size) reallocate_input2(num_samples);
    if (num_samples > g_allocated_in3_size) reallocate_input3(num_samples);

    cudaMemcpy(g_d_in,  in1, num_samples * sizeof(IQSample), cudaMemcpyHostToDevice);
    cudaMemcpy(g_d_in2, in2, num_samples * sizeof(IQSample), cudaMemcpyHostToDevice);
    cudaMemcpy(g_d_in3, in3, num_samples * sizeof(IQSample), cudaMemcpyHostToDevice);

    cudaMemset(g_d_pow1,          0, fft_size * sizeof(float));
    cudaMemset(g_d_pow2,          0, fft_size * sizeof(float));
    cudaMemset(g_d_pow3,          0, fft_size * sizeof(float));
    cudaMemset(g_d_cross_accum,   0, fft_size * sizeof(cufftComplex));
    cudaMemset(g_d_cross13_accum, 0, fft_size * sizeof(cufftComplex));
    cudaMemset(g_d_cross23_accum, 0, fft_size * sizeof(cufftComplex));

    int num_windows = 0;
    int stride = fft_size / 2;
    cufftHandle plan = get_plan(fft_size);
    if (plan == 0) return -2;

    dim3 block(256);
    dim3 grid((fft_size + 255) / 256);

    for (size_t offset = 0; offset + fft_size <= num_samples; offset += stride) {
        process_single_window(g_d_in, (int)offset, g_d_fft, g_d_pow, g_d_fft_saved1, fft_size, plan);
        accumulate<<<grid, block>>>(g_d_pow, g_d_pow1, fft_size);

        process_single_window(g_d_in2, (int)offset, g_d_fft, g_d_pow, g_d_fft_saved2, fft_size, plan);
        accumulate<<<grid, block>>>(g_d_pow, g_d_pow2, fft_size);
        crossAccumulate<<<grid, block>>>(g_d_fft_saved1, g_d_fft, g_d_cross_accum, fft_size);

        process_single_window(g_d_in3, (int)offset, g_d_fft, g_d_pow, nullptr, fft_size, plan);
        accumulate<<<grid, block>>>(g_d_pow, g_d_pow3, fft_size);
        crossAccumulate<<<grid, block>>>(g_d_fft_saved1, g_d_fft, g_d_cross13_accum, fft_size);
        crossAccumulate<<<grid, block>>>(g_d_fft_saved2, g_d_fft, g_d_cross23_accum, fft_size);

        num_windows++;
    }

    if (num_windows > 0) {
        normalize<<<grid, block>>>(g_d_pow1, num_windows, fft_size);
        normalize<<<grid, block>>>(g_d_pow2, num_windows, fft_size);
        normalize<<<grid, block>>>(g_d_pow3, num_windows, fft_size);
        normalizeComplex<<<grid, block>>>(g_d_cross_accum,   num_windows, fft_size);
        normalizeComplex<<<grid, block>>>(g_d_cross13_accum, num_windows, fft_size);
        normalizeComplex<<<grid, block>>>(g_d_cross23_accum, num_windows, fft_size);

        calculateCoherence<<<grid, block>>>(g_d_cross_accum,   g_d_pow1, g_d_pow2, g_d_accum, fft_size);
        shiftFFT<<<grid, block>>>(g_d_accum, g_d_result, fft_size);
        cudaMemcpy(out_coh12, g_d_result, fft_size * sizeof(float), cudaMemcpyDeviceToHost);

        calculateCoherence<<<grid, block>>>(g_d_cross13_accum, g_d_pow1, g_d_pow3, g_d_accum, fft_size);
        shiftFFT<<<grid, block>>>(g_d_accum, g_d_result, fft_size);
        cudaMemcpy(out_coh13, g_d_result, fft_size * sizeof(float), cudaMemcpyDeviceToHost);

        calculateCoherence<<<grid, block>>>(g_d_cross23_accum, g_d_pow2, g_d_pow3, g_d_accum, fft_size);
        shiftFFT<<<grid, block>>>(g_d_accum, g_d_result, fft_size);
        cudaMemcpy(out_coh23, g_d_result, fft_size * sizeof(float), cudaMemcpyDeviceToHost);

        correctGain<<<grid, block>>>(g_d_pow1, 4.0f, fft_size);
        shiftFFT<<<grid, block>>>(g_d_pow1, g_d_result, fft_size);
        cudaMemcpy(out_pow1, g_d_result, fft_size * sizeof(float), cudaMemcpyDeviceToHost);

        correctGain<<<grid, block>>>(g_d_pow2, 4.0f, fft_size);
        shiftFFT<<<grid, block>>>(g_d_pow2, g_d_result, fft_size);
        cudaMemcpy(out_pow2, g_d_result, fft_size * sizeof(float), cudaMemcpyDeviceToHost);

        correctGain<<<grid, block>>>(g_d_pow3, 4.0f, fft_size);
        shiftFFT<<<grid, block>>>(g_d_pow3, g_d_result, fft_size);
        cudaMemcpy(out_pow3, g_d_result, fft_size * sizeof(float), cudaMemcpyDeviceToHost);

        magnitude<<<grid, block>>>(g_d_cross_accum,   g_d_accum, fft_size);
        correctGain<<<grid, block>>>(g_d_accum, 2.0f, fft_size);
        shiftFFT<<<grid, block>>>(g_d_accum, g_d_result, fft_size);
        cudaMemcpy(out_cross12, g_d_result, fft_size * sizeof(float), cudaMemcpyDeviceToHost);

        magnitude<<<grid, block>>>(g_d_cross13_accum, g_d_accum, fft_size);
        correctGain<<<grid, block>>>(g_d_accum, 2.0f, fft_size);
        shiftFFT<<<grid, block>>>(g_d_accum, g_d_result, fft_size);
        cudaMemcpy(out_cross13, g_d_result, fft_size * sizeof(float), cudaMemcpyDeviceToHost);

        magnitude<<<grid, block>>>(g_d_cross23_accum, g_d_accum, fft_size);
        correctGain<<<grid, block>>>(g_d_accum, 2.0f, fft_size);
        shiftFFT<<<grid, block>>>(g_d_accum, g_d_result, fft_size);
        cudaMemcpy(out_cross23, g_d_result, fft_size * sizeof(float), cudaMemcpyDeviceToHost);
    }
    return 0;
}

int sb_process_baseband_iq_full(const int16_t* interleaved_iq, size_t num_samples,
                                float phase_inc, float* out_real, float* out_imag) {
    std::lock_guard<std::mutex> lock(g_state_mutex);

    if (num_samples > g_allocated_in_size) reallocate_input(num_samples);
    cudaMemcpy(g_d_in, interleaved_iq, num_samples * sizeof(IQSample), cudaMemcpyHostToDevice);

    if (!g_real_sum) cudaMalloc(&g_real_sum, sizeof(float));
    if (!g_imag_sum) cudaMalloc(&g_imag_sum, sizeof(float));
    cudaMemset(g_real_sum, 0, sizeof(float));
    cudaMemset(g_imag_sum, 0, sizeof(float));

    int blockSize = 256;
    int gridSize = min(65535, (int)((num_samples + blockSize - 1) / blockSize));
    size_t sharedMemSize = 2 * blockSize * sizeof(float);

    complexAvgStream<<<gridSize, blockSize, sharedMemSize>>>(
        g_d_in, g_real_sum, g_imag_sum, phase_inc, (int)num_samples);
    cudaDeviceSynchronize();

    float real_sum, imag_sum;
    cudaMemcpy(&real_sum, g_real_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&imag_sum, g_imag_sum, sizeof(float), cudaMemcpyDeviceToHost);

    *out_real = real_sum / (float)num_samples;
    *out_imag = imag_sum / (float)num_samples;

    return 0;
}

void sb_shutdown() {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    for (auto& kv : g_plans) cufftDestroy(kv.second);
    g_plans.clear();

    if (g_d_in)          { cudaFree(g_d_in);           g_d_in = nullptr; }
    if (g_d_fft)         { cudaFree(g_d_fft);          g_d_fft = nullptr; }
    if (g_d_pow)         { cudaFree(g_d_pow);          g_d_pow = nullptr; }
    if (g_d_accum)       { cudaFree(g_d_accum);        g_d_accum = nullptr; }
    if (g_d_result)      { cudaFree(g_d_result);       g_d_result = nullptr; }
    if (g_d_in2)         { cudaFree(g_d_in2);          g_d_in2 = nullptr; }
    if (g_d_fft_saved)   { cudaFree(g_d_fft_saved);    g_d_fft_saved = nullptr; }
    if (g_d_cross_accum) { cudaFree(g_d_cross_accum);  g_d_cross_accum = nullptr; }
    if (g_d_cross_result){ cudaFree(g_d_cross_result); g_d_cross_result = nullptr; }
    if (g_d_pow1)        { cudaFree(g_d_pow1);         g_d_pow1 = nullptr; }
    if (g_d_pow2)        { cudaFree(g_d_pow2);         g_d_pow2 = nullptr; }
    if (g_d_in3)           { cudaFree(g_d_in3);           g_d_in3 = nullptr; }
    if (g_d_fft_saved1)    { cudaFree(g_d_fft_saved1);    g_d_fft_saved1 = nullptr; }
    if (g_d_fft_saved2)    { cudaFree(g_d_fft_saved2);    g_d_fft_saved2 = nullptr; }
    if (g_d_cross13_accum) { cudaFree(g_d_cross13_accum); g_d_cross13_accum = nullptr; }
    if (g_d_cross23_accum) { cudaFree(g_d_cross23_accum); g_d_cross23_accum = nullptr; }
    if (g_d_pow3)          { cudaFree(g_d_pow3);          g_d_pow3 = nullptr; }
    if (g_real_sum)        { cudaFree(g_real_sum);        g_real_sum = nullptr; }
    if (g_imag_sum)        { cudaFree(g_imag_sum);        g_imag_sum = nullptr; }

    g_allocated_n = 0;
    g_allocated_in_size = 0;
    g_allocated_n_cross = 0;
    g_allocated_in2_size = 0;
    g_allocated_coh_n = 0;
    g_allocated_n_triple = 0;
    g_allocated_in3_size = 0;
}

} // extern "C"